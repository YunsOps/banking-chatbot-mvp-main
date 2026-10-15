"""배치 평가: 미리 만든 질문을 서비스 POST /chat에 순차 호출하고 응답·측정값·통과 여부를 기록.

실제 분류기 예측을 거친 서비스 처리 흐름을 local·cloud 모델별로 같은 조건에서 실행한다.
생성 답변은 HTTP(GET /health, POST /chat)로만 받는다.

측정
- 생성 시간: 응답의 latency_s.generate (서버에서 생성 LLM 호출 시작~전체 답변 수신)
- 서비스 전체 시간: 응답의 latency_s.total (질문 접수~응답), 클라이언트 측 client_s
- 워밍업 요청은 집계에서 제외하고 run_config.json에 따로 기록
- 실패(호출 실패·시간 초과·빈 응답)도 제외하지 않고 기록

호출 주소: .env의 EVAL_API_BASE_URL (예: http://10.2.10.71:8000), 없으면 http://localhost:8000.
--base-url을 주면 그 실행에서만 바꾼다.
서버의 모델 이름은 API 응답에 없으므로 .env의 EVAL_API_LOCAL_MODEL·EVAL_API_CLOUD_MODEL에 적어 두면 결과에 기록된다.

결과: evaluation/results/<run_id>/
- run_config.json             실행 조건 (질문 파일, 모델, 반복·워밍업, 시간 제한·재시도, 생성 설정, 서버 상태,
                              로컬 모델 메모리, 서비스 파일 해시, 실행 환경)
- responses.jsonl, .csv       요청별 원본 응답과 측정값, 기대 결과 통과 여부
- scenarios.csv               예외 처리 시나리오(입력 오류 422, LLM 미설정) 통과 여부
- summary.csv, summary.md     모델별 집계와 비교표, 실패 사례 (채점 전 빠른 확인용)

다음 단계: judge_answers.py로 채점 → build_report.py로 보고서 생성
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

from common import (
    API_BASE_URL, DEFAULT_POLICIES, DEFAULT_QUESTIONS, EXPECTED_ROUTES, RESULTS_DIR, environment, load_questions,
    question_fields, recorded_env, service_files, service_generation_settings, sha256, write_csv, write_json,
    write_jsonl,
)


EVAL_TYPE = "api_integrated"
RESPONSE_FIELDS = [
    "run_id", "eval_type", "model", "repeat", "question_id", "topic", "source_type", "policy_id", "policy_name",
    "expected_status", "expected_behavior", "question", "http_status", "route", "route_pass",
    "success", "label", "label_match", "confidence", "response_model", "latency_classify",
    "latency_generate", "latency_total", "client_s", "error", "request_error", "answer", "requested_at",
]
# 예외 처리 시나리오: (이름, 요청 본문, 기대 HTTP 상태)
INPUT_SCENARIOS = [
    ("question 없음", {}, 422),
    ("빈 질문", {"question": ""}, 422),
    ("공백만 있는 질문", {"question": "   "}, 422),
    ("500자 초과", {"question": "가" * 501}, 422),
    ("허용되지 않은 model", {"question": "잔액 조회", "model": "gpt"}, 422),
]


def open_client(timeout):
    return httpx.Client(timeout=timeout)


def post_chat(client, base_url, question, model):
    """POST /chat 한 건. 클라이언트 측 소요 시간과 요청 오류도 함께 반환."""
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url}/chat", json={"question": question, "model": model})
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return response.status_code, body, time.perf_counter() - started, None
    except httpx.HTTPError as error:
        return None, {}, time.perf_counter() - started, f"{type(error).__name__}: {error}"


def to_row(run_id, model, repeat, question, http_status, body, client_s, request_error):
    latency = body.get("latency_s") or {}
    route = body.get("route")
    answer = body.get("answer") or ""
    return {
        "run_id": run_id,
        "eval_type": EVAL_TYPE,
        "model": model,
        "repeat": repeat,
        **question_fields(question),
        "http_status": http_status,
        "route": route,
        "route_pass": route in EXPECTED_ROUTES.get(question.get("expected_status"), set()),
        # 평가 기준의 실패: 호출 실패·시간 초과·빈 응답
        "success": http_status == 200 and route not in (None, "error") and bool(answer.strip()),
        "label": body.get("label"),
        "label_match": body.get("label") == question.get("topic"),
        "confidence": body.get("confidence"),
        "response_model": body.get("model"),
        "latency_classify": latency.get("classify"),
        "latency_generate": latency.get("generate"),
        "latency_total": latency.get("total"),
        "client_s": round(client_s, 3),
        "error": body.get("error"),
        "request_error": request_error,
        "answer": answer,
        "requested_at": datetime.now().isoformat(timespec="seconds"),
    }


def ollama_memory():
    """로컬 모델 메모리: .env의 OLLAMA_BASE_URL /api/ps (서버 Ollama에 접속 가능할 때만). 실패 시 빈 목록."""
    base_url = (os.getenv("OLLAMA_BASE_URL") or "").rstrip("/")
    if not base_url:
        return []
    try:
        models = httpx.get(f"{base_url}/api/ps", timeout=3).json().get("models", [])
    except Exception:
        return []
    return [{"name": m.get("name"), "size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram")} for m in models]


def model_ready(health, model):
    return bool((health.get("llm") or {}).get(model))


def generation_settings():
    try:
        return {"local": service_generation_settings(), "cloud": "서비스에 미구현 (상용 LLM 연결 없음)"}
    except Exception as error:  # 서비스 의존성이 없는 환경
        return {"error": f"{type(error).__name__}: {error}"[:200]}


def run_scenarios(client, base_url, health):
    rows = []
    for name, payload, expected in INPUT_SCENARIOS:
        response = client.post(f"{base_url}/chat", json=payload)
        rows.append({"scenario": name, "request": json.dumps(payload, ensure_ascii=False)[:80],
                     "expected": expected, "actual": response.status_code,
                     "passed": response.status_code == expected})
    # 설정되지 않은 LLM을 요청하면 200 + route=error + error=llm_not_configured
    for model in ("local", "cloud"):
        if not model_ready(health, model):
            status, body, _, _ = post_chat(client, base_url, "적금 해지 방법 알려 주세요", model)
            actual = f"{status} route={body.get('route')} error={body.get('error')}"
            passed = status == 200 and body.get("route") == "error" and body.get("error") in (
                "llm_not_configured", "llm_error", "llm_timeout")
            rows.append({"scenario": f"{model} LLM 미준비", "request": f"model={model}",
                         "expected": "200 route=error error=llm_*", "actual": actual, "passed": passed})
    return rows


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def summarize(rows, model):
    rows = [row for row in rows if row["model"] == model]
    total = len(rows)
    successes = [row for row in rows if row["success"]]
    errors = {}
    for row in rows:
        if not row["success"]:
            key = row["error"] or row["request_error"] or "empty_answer"
            errors[key] = errors.get(key, 0) + 1
    routes = {}
    for row in rows:
        routes[row["route"]] = routes.get(row["route"], 0) + 1
    by_case = {}
    for row in rows:
        case = f"{row['policy_id']} {row['policy_name']}"
        passed, count = by_case.get(case, (0, 0))
        by_case[case] = (passed + row["route_pass"], count + 1)
    return {
        "model": model,
        "requests": total,
        "success": len(successes),
        "failure": total - len(successes),
        "failure_rate": round((total - len(successes)) / total, 3) if total else None,
        "failure_by_error": errors,
        "route_pass": sum(row["route_pass"] for row in rows),
        "route_pass_rate": round(sum(row["route_pass"] for row in rows) / total, 3) if total else None,
        "route_pass_by_case": {case: f"{passed}/{count}" for case, (passed, count) in by_case.items()},
        "routes": routes,
        "label_match_rate": round(sum(row["label_match"] for row in rows) / total, 3) if total else None,
        # 응답 시간은 성공 요청만 집계. 실패는 위 건수로 따로 보고
        "latency_total": describe([row["latency_total"] for row in successes]),
        "latency_generate": describe([row["latency_generate"] for row in successes]),
        "client_s": describe([row["client_s"] for row in successes]),
    }


def write_summary_md(path, config, summaries, rows, scenarios):
    def cell(summary, key, sub=None):
        value = summary[key] if sub is None else summary[key][sub]
        return "-" if value is None else value

    lines = [
        f"# API 배치 검증 결과 ({config['run_id']})",
        "",
        f"- 질문 파일: `{config['questions']['file']}` ({config['questions']['count']}건), 반복 {config['repeats']}회, "
        f"워밍업 {config['warmup']}회",
        f"- 서버: {config['base_url']}, 서버 상태: {config['health'].get('status')}",
        f"- 실행 모델: {', '.join(config['models_run']) or '없음'}"
        + (f" / 제외: {config['models_skipped']}" if config["models_skipped"] else ""),
        f"- 실행 시간: {config['started_at']} ~ {config['finished_at']}",
        "",
        "## 모델별 비교",
        "",
        "| 항목 | " + " | ".join(summary["model"] for summary in summaries) + " |",
        "| --- |" + " --- |" * len(summaries),
    ]
    items = [
        ("요청 수", "requests", None),
        ("성공", "success", None),
        ("실패 (호출 실패·시간 초과·빈 응답)", "failure", None),
        ("실패율", "failure_rate", None),
        ("기대 route 통과", "route_pass", None),
        ("기대 route 통과율", "route_pass_rate", None),
        ("분류 라벨 일치율", "label_match_rate", None),
        ("전체 응답 시간 평균(초, 성공)", "latency_total", "mean"),
        ("전체 응답 시간 중앙값(초, 성공)", "latency_total", "median"),
        ("생성 시간 평균(초, 성공)", "latency_generate", "mean"),
        ("생성 시간 중앙값(초, 성공)", "latency_generate", "median"),
    ]
    for title, key, sub in items:
        lines.append(f"| {title} | " + " | ".join(str(cell(summary, key, sub)) for summary in summaries) + " |")
    lines.append("| 실패 원인 | " + " | ".join(
        json.dumps(summary["failure_by_error"], ensure_ascii=False) or "-" for summary in summaries) + " |")
    lines.append("| route 분포 | " + " | ".join(
        json.dumps(summary["routes"], ensure_ascii=False) for summary in summaries) + " |")

    cases = sorted({case for summary in summaries for case in summary["route_pass_by_case"]})
    lines += ["", "## 정책별 기대 route 통과", "",
              "| 정책 | " + " | ".join(summary["model"] for summary in summaries) + " |",
              "| --- |" + " --- |" * len(summaries)]
    for case in cases:
        lines.append(f"| {case} | " + " | ".join(
            summary["route_pass_by_case"].get(case, "-") for summary in summaries) + " |")

    lines += ["", "## 예외 처리 시나리오", "", "| 시나리오 | 기대 | 실제 | 통과 |", "| --- | --- | --- | --- |"]
    for scenario in scenarios:
        lines.append(f"| {scenario['scenario']} | {scenario['expected']} | {scenario['actual']} | "
                     f"{'O' if scenario['passed'] else 'X'} |")

    failures = [row for row in rows if not row["route_pass"] or not row["success"]]
    lines += ["", f"## 기대와 다른 결과·실패 ({len(failures)}건)", ""]
    if failures:
        lines += ["| 모델 | 반복 | 질문 ID | 정책 | 기대 | route | error | 답변 앞부분 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in failures:
            answer = row["answer"].replace("\n", " ").replace("|", "/")[:60]
            lines.append(f"| {row['model']} | {row['repeat']} | {row['question_id']} | {row['policy_id']} | "
                         f"{row['expected_status']} | {row['route']} | {row['error'] or row['request_error'] or ''} | {answer} |")
    else:
        lines.append("없음")

    lines += [
        "",
        "## 참고",
        "",
        "- 기대 route 통과는 처리 결과(route)만 본 것으로, 답변 내용의 정확성·안전성 채점은 포함하지 않는다.",
        "- 응답 시간 통계는 성공 요청만 집계하고, 실패 건수와 원인은 별도로 표시한다.",
        "- 원본 응답 전체는 responses.jsonl, 실행 조건은 run_config.json에 있다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=API_BASE_URL,
                        help=f"서비스 API 주소. 기본값은 .env의 EVAL_API_BASE_URL (현재 {API_BASE_URL})")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="질문 파일 (.jsonl·.csv)")
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES, help="정책표 (.jsonl·.csv)")
    parser.add_argument("--models", nargs="+", choices=["local", "cloud"], default=["local", "cloud"])
    parser.add_argument("--repeats", type=int, default=int(os.getenv("EVAL_REPEATS", "1")),
                        help="질문 목록 반복 횟수. 기본값은 .env의 EVAL_REPEATS")
    parser.add_argument("--warmup", type=int, default=1, help="모델별 워밍업 요청 수 (결과 집계 제외)")
    parser.add_argument("--timeout", type=float, default=120, help="클라이언트 요청 시간 제한(초). 서버 LLM_TIMEOUT_S보다 길게")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    questions = load_questions(args.questions, args.policies)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    with open_client(args.timeout) as client:
        health = client.get(f"{base_url}/health").json()
        print(f"서버 상태: {json.dumps(health, ensure_ascii=False)}")

        models_run = [model for model in args.models if model_ready(health, model)]
        models_skipped = {model: "서버 /health에서 준비되지 않음" for model in args.models if model not in models_run}
        for model, reason in models_skipped.items():
            print(f"[제외] {model}: {reason}")

        warmups = {}
        rows = []
        memory = {}
        for model in models_run:
            # 모델 로딩·최초 요청 시간을 본 측정과 분리
            warmups[model] = []
            for _ in range(args.warmup):
                status, body, client_s, request_error = post_chat(client, base_url, questions[0]["question"], model)
                warmups[model].append({"http_status": status, "route": body.get("route"),
                                       "client_s": round(client_s, 3), "request_error": request_error})
                print(f"[워밍업] {model} {client_s:.2f}s route={body.get('route')}")
            if model == "local":
                memory["after_warmup"] = ollama_memory()

            for repeat in range(1, args.repeats + 1):
                for question in questions:
                    status, body, client_s, request_error = post_chat(client, base_url, question["question"], model)
                    row = to_row(args.run_id, model, repeat, question, status, body, client_s, request_error)
                    rows.append(row)
                    print(f"{model} #{repeat} {row['question_id']} route={row['route']} "
                          f"{'통과' if row['route_pass'] else '불일치'} {client_s:.2f}s"
                          + (f" error={row['error'] or row['request_error']}" if not row["success"] else ""))
            if model == "local":
                memory["after_run"] = ollama_memory()

        scenarios = run_scenarios(client, base_url, health)

    config = {
        "run_id": args.run_id,
        "eval_type": EVAL_TYPE,
        "description": "서비스 POST /chat 통합 평가. 실제 분류기 예측과 정책 선택을 거친 전체 처리 흐름",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "questions": {"file": args.questions.name, "sha256": sha256(args.questions), "count": len(questions),
                      "policies_file": args.policies.name, "policies_sha256": sha256(args.policies)},
        "models_requested": args.models,
        "models_run": models_run,
        "models_skipped": models_skipped,
        # 서버 모델 이름은 API로 알 수 없어 .env에 적어 둔 값을 기록
        "model_ids": {"local": os.getenv("EVAL_API_LOCAL_MODEL"), "cloud": os.getenv("EVAL_API_CLOUD_MODEL")},
        "repeats": args.repeats,
        "warmup": args.warmup,
        "warmup_results": warmups,
        "client_timeout_s": args.timeout,
        "retries": "없음 (클라이언트 요청 1회, 서버 생성 호출 1회)",
        "call_order": "모델별로 질문 파일 순서대로 순차 호출 (동시 요청 없음)",
        # 서버 생성 설정은 API로 알 수 없어 평가 저장소의 서비스 코드에서 읽음 (service_files_sha256로 서버 코드와 대조)
        "generation_settings": generation_settings(),
        # 로컬 모델 메모리 (Ollama /api/ps). 비어 있으면 미측정
        "local_memory": memory,
        "health": health,
        # 평가를 실행한 쪽의 설정. 서버 설정과 다를 수 있음
        "settings": recorded_env(),
        # 평가를 실행한 저장소의 서비스 파일. 서버에 배포된 코드와 같은지 확인용
        "service_files_sha256": service_files(),
        "environment": environment(),
    }

    write_json(run_dir / "run_config.json", config)
    write_jsonl(run_dir / "responses.jsonl", rows)
    write_csv(run_dir / "responses.csv", rows, RESPONSE_FIELDS)
    write_csv(run_dir / "scenarios.csv", scenarios, ["scenario", "request", "expected", "actual", "passed"])

    summaries = [summarize(rows, model) for model in models_run]
    latency_keys = ("latency_total", "latency_generate", "client_s")
    write_csv(run_dir / "summary.csv", [
        {
            **{key: value for key, value in summary.items() if key not in latency_keys},
            **{f"{key}_{stat}": summary[key][stat] for key in latency_keys for stat in ("mean", "median", "max")},
        }
        for summary in summaries
    ], [
        "model", "requests", "success", "failure", "failure_rate", "route_pass", "route_pass_rate",
        "label_match_rate", "latency_total_mean", "latency_total_median", "latency_total_max",
        "latency_generate_mean", "latency_generate_median", "latency_generate_max",
        "client_s_mean", "client_s_median", "client_s_max", "failure_by_error", "routes", "route_pass_by_case",
    ] if summaries else ["model"])
    write_summary_md(run_dir / "summary.md", config, summaries, rows, scenarios)

    print(f"\n결과 폴더: {run_dir}")
    for summary in summaries:
        print(f"{summary['model']}: 성공 {summary['success']}/{summary['requests']}, "
              f"기대 route 통과 {summary['route_pass']}/{summary['requests']}, "
              f"응답 시간 중앙값 {summary['latency_total']['median']}s")
    print(f"예외 처리 시나리오: {sum(s['passed'] for s in scenarios)}/{len(scenarios)} 통과")


if __name__ == "__main__":
    main()
