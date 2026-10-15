"""생성 모델 비교 보고서: /chat 응답(responses)과 평가 모델 채점(judge_scores)으로 지표를 집계하고 Markdown 보고서 작성.

- 입력: run_api_batch.py 결과 폴더 하나 이상 (judge_answers.py 채점 후)
- 품질: 사용 가능 답변 비율, 판정 분포, 항목별 평균 점수, 공통 사례 비교, 정책별 비교, 대표 사례
- 응답 시간: 생성 시간(latency_s.generate)과 서비스 전체 시간(latency_s.total)을 따로 집계. 성공 요청만, 건수 표시
- 실패: 호출 실패·시간 초과·빈 응답 건수와 실패율을 제외하지 않고 별도 집계
- 자원·비용: 로컬 모델 메모리, API 생성 비용, 평가 모델 비용(서비스 비용과 분리), 장비·전력 비용을 실측·미측정으로 표시
- 운영 조건: 외부 전송 범위, 평가 모델로 보낸 데이터, 결과 파일의 비밀값 포함 여부 검사
- 추천은 사전에 정한 규칙(--*-tie 기준)으로 계산한 결과이며, 최종 선정은 팀 검토 후 확정

단가 (.env, 선택): JUDGE_PRICE_INPUT_PER_1M, JUDGE_PRICE_OUTPUT_PER_1M, PRICE_CURRENCY, PRICE_DATE.
없으면 평가 모델 비용은 "단가 미입력"

결과: evaluation/reports/<name>.md, evaluation/reports/<name>_metrics.csv
"""

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from common import (
    REPORTS_DIR, SCORE_ITEMS, SCORE_NAMES, VERDICT_ERROR, VERDICT_FAILED, VERDICT_NEEDS_WORK,
    VERDICT_UNJUDGED, VERDICT_USABLE, find_secrets, read_jsonl, row_key, write_csv,
)


MODEL_NAMES = {"local": "로컬 LLM", "cloud": "상용 LLM"}
EXTERNAL_TRANSFER = {"local": "없음 (서버 안에서 실행)", "cloud": "있음 (상용 API로 질문·프롬프트 전송)"}
VERDICTS = [VERDICT_USABLE, VERDICT_NEEDS_WORK, VERDICT_ERROR, VERDICT_FAILED, VERDICT_UNJUDGED]


# ---- 읽기·집계 ------------------------------------------------------------------

def price(name):
    value = os.getenv(name)
    return float(value) if value else None


def load_run(run_dir: Path) -> dict:
    if not (run_dir / "run_config.json").exists():
        raise SystemExit(f"{run_dir}에 실행 결과(run_config.json)가 없습니다. "
                         f"먼저 run_api_batch.py --run-id {run_dir.name} 로 배치를 실행하고 judge_answers.py로 채점하세요.")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    judge_config_path = run_dir / "judge_config.json"
    return {
        "dir": run_dir,
        "config": config,
        "judge_config": json.loads(judge_config_path.read_text(encoding="utf-8")) if judge_config_path.exists() else None,
        "rows": read_jsonl(run_dir / "responses.jsonl"),
        "scores": {row_key(score): score for score in read_jsonl(run_dir / "judge_scores.jsonl")},
    }


def row_verdict(row, score):
    if not row.get("success"):
        return VERDICT_FAILED
    return score["verdict"] if score else VERDICT_UNJUDGED


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": round(values[min(len(values) - 1, int(len(values) * 0.9))], 3),
        "max": round(values[-1], 3),
    }


def rate(part, whole):
    return round(part / whole, 3) if whole else None


def mean(values):
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 3) if values else None


def model_metrics(run, model, keys=None):
    """모델 하나의 지표. keys를 주면 해당 (반복, 질문 ID) 사례만 집계 (공통 사례 비교용)."""
    rows = [row for row in run["rows"] if row["model"] == model
            and (keys is None or (int(row["repeat"]), row["question_id"]) in keys)]
    pairs = [(row, run["scores"].get(row_key(row))) for row in rows]
    verdicts = [row_verdict(row, score) for row, score in pairs]
    judged = [score for row, score in pairs if row.get("success") and score and score.get("judge_error") is None]
    successes = [row for row in rows if row.get("success")]
    failures = {}
    for row in rows:
        if not row.get("success"):
            key = row.get("error") or row.get("request_error") or "empty_answer"
            failures[key] = failures.get(key, 0) + 1

    n = len(rows)
    return {
        "model": model,
        "model_id": run["config"].get("model_ids", {}).get(model),
        "requests": n,
        "verdicts": {name: verdicts.count(name) for name in VERDICTS},
        "usable_rate": rate(verdicts.count(VERDICT_USABLE), n),
        "usable_rate_judged": rate(verdicts.count(VERDICT_USABLE), len(judged)),
        "judged": len(judged),
        "unjudged": verdicts.count(VERDICT_UNJUDGED),
        "failure": verdicts.count(VERDICT_FAILED),
        "failure_rate": rate(verdicts.count(VERDICT_FAILED), n),
        "timeout": sum(1 for row in rows if row.get("error") == "llm_timeout"),
        "failure_by_error": failures,
        "route_pass_rate": rate(sum(bool(row.get("route_pass")) for row in rows), n),
        "label_match_rate": rate(sum(bool(row.get("label_match")) for row in rows), n),
        **{f"{item}_mean": mean([score[item] for score in judged]) for item in SCORE_ITEMS},
        "safety_zero": sum(1 for score in judged if score["safety"] == 0),
        # 생성 시간: 생성 LLM을 부른 성공 요청만 (분류기 1차 차단은 생성 호출이 없어 제외)
        "latency_generate": stats([row.get("latency_generate") for row in successes]),
        # 서비스 전체 시간: 성공 요청 전체 (질문 접수~응답)
        "latency_total": stats([row.get("latency_total") for row in successes]),
        "external_transfer": EXTERNAL_TRANSFER[model],
    }


def common_keys(run, models):
    """모든 모델이 판정 가능(실패·미채점 아님)한 (반복, 질문 ID)."""
    keys = None
    for model in models:
        model_keys = {
            (int(row["repeat"]), row["question_id"]) for row in run["rows"]
            if row["model"] == model and row_verdict(row, run["scores"].get(row_key(row))) not in (VERDICT_FAILED, VERDICT_UNJUDGED)
        }
        keys = model_keys if keys is None else keys & model_keys
    return keys or set()


def judge_cost(run):
    scores = list(run["scores"].values())
    tokens_in = sum(score.get("judge_input_tokens") or 0 for score in scores)
    tokens_out = sum(score.get("judge_output_tokens") or 0 for score in scores)
    if price("JUDGE_PRICE_INPUT_PER_1M") is None:
        return tokens_in, tokens_out, None
    cost = (tokens_in * price("JUDGE_PRICE_INPUT_PER_1M") + tokens_out * (price("JUDGE_PRICE_OUTPUT_PER_1M") or 0)) / 1_000_000
    return tokens_in, tokens_out, round(cost, 6)


# ---- 우선순위 비교 ----------------------------------------------------------------

def compare(metrics, ties):
    """평가 기준의 비교 우선순위대로 local·cloud 비교. (항목, local 값, cloud 값, 판단, 우세 모델)"""
    local, cloud = metrics["local"], metrics["cloud"]
    results = []

    def decide(name, lv, cv, higher_better, tie, tie_desc, relative=False):
        if lv is None or cv is None:
            results.append((name, lv, cv, "판단 불가 (값 없음)", None))
            return
        diff = abs(lv - cv)
        threshold = tie * max(abs(lv), abs(cv)) if relative else tie
        if diff <= threshold:
            results.append((name, lv, cv, f"차이 없음 ({tie_desc} 이하)", None))
            return
        winner = ("local" if lv > cv else "cloud") if higher_better else ("local" if lv < cv else "cloud")
        results.append((name, lv, cv, f"{MODEL_NAMES[winner]} 우세", winner))

    decide("1. 사용 가능 답변 비율", local["usable_rate"], cloud["usable_rate"], True,
           ties["usable"], f"{ties['usable'] * 100:.0f}%p")
    results.append(("2. 보안성 (외부 전송)", local["external_transfer"], cloud["external_transfer"],
                    "로컬 LLM 우세 (외부 전송 없음)", "local"))
    decide("3. 정확성 평균 점수", local["accuracy_mean"], cloud["accuracy_mean"], True,
           ties["accuracy"], f"{ties['accuracy']}점")
    decide("4. 생성 시간 중앙값(초)", local["latency_generate"]["median"], cloud["latency_generate"]["median"], False,
           ties["latency"], f"상대 {ties['latency'] * 100:.0f}%", relative=True)
    results.append(("5. 비용 (요청당)", "API 비용 없음 (장비·전력 미측정)", "미산출 (API 응답에 사용량 없음)",
                    "판단 불가 (상용 API 사용량 미수집)", None))
    return results


# ---- 보고서 작성 ----------------------------------------------------------------

def fmt(value, percent=False):
    if value is None:
        return "-"
    if percent:
        return f"{value * 100:.1f}%"
    return str(value)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(str(cell).replace("|", "/").replace("\n", " ") for cell in row) + " |" for row in rows]
    return lines


def latency_text(values):
    if not values["n"]:
        return "-"
    return f"{fmt(values['mean'])} / {fmt(values['median'])} / {fmt(values['p90'])} ({values['n']}건)"


def metrics_table(metrics_by_model, models):
    def row(title, getter):
        return [title] + [getter(metrics_by_model[model]) for model in models]

    return table(["항목"] + [f"{MODEL_NAMES[m]} ({metrics_by_model[m]['model_id'] or '-'})" for m in models], [
        row("요청 수", lambda m: m["requests"]),
        row("**사용 가능 답변 비율** (전체 요청 대비)", lambda m: fmt(m["usable_rate"], True)),
        row("사용 가능 비율 (채점 건 대비)", lambda m: fmt(m["usable_rate_judged"], True)),
        row("판정 분포 (사용 가능/보완 필요/오류/실패/미채점)",
            lambda m: " / ".join(str(m["verdicts"][v]) for v in VERDICTS)),
        *[row(f"{SCORE_NAMES[item]} 평균 (0~2)", lambda m, item=item: fmt(m[f"{item}_mean"])) for item in SCORE_ITEMS],
        row("안전성 0점 건수", lambda m: m["safety_zero"]),
        row("기대 route 통과율", lambda m: fmt(m["route_pass_rate"], True)),
        row("분류 라벨 일치율", lambda m: fmt(m["label_match_rate"], True)),
        row("실패 건수 / 실패율 (호출 실패·시간 초과·빈 응답)", lambda m: f"{m['failure']} / {fmt(m['failure_rate'], True)}"),
        row("시간 초과 건수", lambda m: m["timeout"]),
        row("실패 원인", lambda m: json.dumps(m["failure_by_error"], ensure_ascii=False) if m["failure_by_error"] else "-"),
        row("**생성 시간** 평균 / 중앙값 / p90 (초, 성공)", lambda m: latency_text(m["latency_generate"])),
        row("서비스 전체 시간 평균 / 중앙값 / p90 (초, 성공)", lambda m: latency_text(m["latency_total"])),
        row("외부 전송", lambda m: m["external_transfer"]),
    ])


def policy_name(row):
    return f"{row.get('policy_id') or '-'} {row.get('policy_name') or ''}".strip()


def policy_table(run, models):
    """정책(policy_id)별 사용 가능 건수와 정확성 평균."""
    policies = sorted({policy_name(row) for row in run["rows"]})
    rows = []
    for policy in policies:
        cells = [policy]
        for model in models:
            subset = [(row, run["scores"].get(row_key(row))) for row in run["rows"]
                      if row["model"] == model and policy_name(row) == policy]
            usable = sum(1 for row, score in subset if row_verdict(row, score) == VERDICT_USABLE)
            judged = [score for row, score in subset if row.get("success") and score and score.get("judge_error") is None]
            acc = mean([score["accuracy"] for score in judged])
            cells.append(f"{usable}/{len(subset)} (정확성 {fmt(acc)})")
        rows.append(cells)
    return table(["정책"] + [f"{MODEL_NAMES[m]} 사용 가능 (정확성 평균)" for m in models], rows)


def case_examples(run, model, limit):
    """대표 사례: 오류 → 보완 필요(점수 낮은 순) → 실패, 그리고 사용 가능 1건. 같은 질문은 한 번만 표시."""
    pairs = [(row, run["scores"].get(row_key(row))) for row in run["rows"] if row["model"] == model]

    def total(score):
        return sum(score[item] for item in SCORE_ITEMS) if score and score.get(SCORE_ITEMS[0]) is not None else -1

    lines = []
    for title, wanted, order in [
        ("오류 판정", VERDICT_ERROR, lambda p: total(p[1])),
        ("보완 필요 (점수 낮은 순)", VERDICT_NEEDS_WORK, lambda p: total(p[1])),
        ("실패", VERDICT_FAILED, lambda p: 0),
        ("사용 가능 (대표 성공 사례)", VERDICT_USABLE, lambda p: -total(p[1])),
    ]:
        matched = sorted([p for p in pairs if row_verdict(*p) == wanted], key=order)
        repeats = {}
        for row, _ in matched:
            repeats[row["question_id"]] = repeats.get(row["question_id"], 0) + 1
        selected, seen = [], set()
        for row, score in matched:
            if row["question_id"] not in seen:
                seen.add(row["question_id"])
                selected.append((row, score))
        selected = selected[:1 if wanted == VERDICT_USABLE else limit]
        if not selected:
            continue
        lines += [f"#### {MODEL_NAMES[model]} – {title} ({len(matched)}건, 질문 {len(repeats)}개 중 {len(selected)}개)", ""]
        for row, score in selected:
            lines.append(f"- **{row['question_id']}** (반복 {row['repeat']}, 같은 판정 {repeats[row['question_id']]}회, "
                         f"정책 {policy_name(row)}, 상담 주제 {row.get('topic')}, 분류 {row.get('label')}, "
                         f"기대 {row.get('expected_status')}, route {row.get('route')})")
            lines.append(f"  - 질문: {row['question']}")
            answer = (row.get("answer") or row.get("error") or "").replace("\n", " ")
            lines.append(f"  - 답변: {answer[:300]}{'…' if len(answer) > 300 else ''}")
            if score and score.get(SCORE_ITEMS[0]) is not None:
                lines.append("  - 점수: " + ", ".join(f"{SCORE_NAMES[i]} {score[i]}" for i in SCORE_ITEMS))
                for item in SCORE_ITEMS:
                    if score[item] < 2:
                        lines.append(f"  - {SCORE_NAMES[item]} 근거: {score.get(item + '_reason', '')}")
            elif row.get("error") or row.get("request_error"):
                lines.append(f"  - 오류: {row.get('error') or ''} {row.get('request_error') or ''}".rstrip())
        lines.append("")
    return lines


def source_counts(run):
    """질문 출처 구분별 질문 수 (모델·반복 중복 제외)."""
    counts = {}
    for question_id, source in {(row["question_id"], row.get("source_type") or "-") for row in run["rows"]}:
        counts[source] = counts.get(source, 0) + 1
    return counts


def memory_text(memory):
    entries = [m for stage in (memory or {}).values() for m in stage]
    if not entries:
        return None
    vram = max((m.get("size_vram_bytes") or 0) for m in entries)
    size = max((m.get("size_bytes") or 0) for m in entries)
    names = sorted({m.get("name") for m in entries})
    return f"{', '.join(names)}: 모델 메모리 {size / 1024 ** 3:.2f}GB (VRAM {vram / 1024 ** 3:.2f}GB)"


def resource_section(run, number):
    config = run["config"]
    tokens_in, tokens_out, cost = judge_cost(run)
    memory = memory_text(config.get("local_memory"))
    judge = run["judge_config"]
    currency = os.getenv("PRICE_CURRENCY", "")
    lines = [f"### {number}.7 자원·비용", ""]
    lines += table(["항목", "값", "구분"], [
        ["로컬 모델 메모리 (Ollama /api/ps, 워밍업 후·실행 후 최대)", memory or "-", "실측" if memory else "미측정 (서버 Ollama 접속 불가)"],
        ["로컬 최대 RAM·VRAM (실행 중 전체)", "-", "미측정"],
        ["실행 환경 (평가 실행 PC)", json.dumps(config.get("environment"), ensure_ascii=False), "실측"],
        ["서버 실행 환경", json.dumps(config.get("health", {}).get("classifier"), ensure_ascii=False), "서버 /health 응답 범위만"],
        ["상용 API 생성 비용", "-", "미산출 (/chat 응답에 입력·출력 사용량 없음, 서비스에 상용 LLM 미구현)"],
        [f"평가 모델 비용 ({judge['judge_model'] if judge else '-'}, 서비스 비용과 별도)",
         f"입력 {tokens_in:,} / 출력 {tokens_out:,} 토큰, 비용 {fmt(cost)} {currency}".rstrip(),
         f"실측 사용량 × 단가 (기준일 {os.getenv('PRICE_DATE') or '미입력'})" if cost is not None else "사용량 실측, 단가 미입력"],
        ["로컬 장비·전력·관리 비용", "-", "미측정 (산정 범위 밖)"],
    ])
    return lines + [""]


def operation_section(run, number, models):
    files = sorted(path for path in run["dir"].iterdir() if path.is_file())
    leaked = find_secrets(files)
    judge = run["judge_config"]
    lines = [f"### {number}.8 운영 조건", ""]
    lines += table(["항목", "내용"], [
        *[[f"{MODEL_NAMES[m]} 외부 전송", EXTERNAL_TRANSFER[m]] for m in models],
        ["로컬 LLM 운영", "서버 장비에서 Ollama 모델 실행·관리 필요 (모델 파일, GPU 메모리, 서버 가동)"],
        ["평가 모델로 보낸 데이터", f"{judge['judge_model'] if judge else '평가 모델'}(OpenAI API)에 질문·정책 기대 동작·서비스 출력 규칙·챗봇 답변 전송. "
                             "질문은 직접 작성한 문장이며 실제 고객 정보 없음 (예시 계좌번호·주민등록번호는 가상 값)"],
        ["결과 파일 비밀값 검사", "통과 (API 키 없음)" if not leaked else f"**실패: {', '.join(Path(p).name for p in leaked)}에 API 키 포함**"],
        ["기록에서 제외한 값", "API 키 등 비밀값 (run_config.json의 settings에 기록하지 않음)"],
    ])
    return lines + [""]


def record_section(number):
    lines = [f"### {number}.9 실험 기록", ""]
    lines += table(["파일", "내용", "연결 키"], [
        ["run_config.json", "질문·정책 파일 해시, 반복·워밍업, 시간 제한·재시도, 생성 설정, 서비스 파일(프롬프트) 해시, 서버 상태, 실행 환경", "run_id"],
        ["responses.jsonl·csv", "질문 ID·질문·적용 정책·분류 결과·/chat 원본 응답·route·성공 여부·오류·생성/전체 시간", "모델, 반복, 질문 ID"],
        ["judge_scores.jsonl", "항목별 점수·근거·판정·평가 모델 사용량", "모델, 반복, 질문 ID"],
        ["judge_config.json", "평가 모델, 평가 프롬프트 해시, 채점·오류 건수", "run_id"],
        ["scenarios.csv", "입력 오류·LLM 미준비 예외 처리 결과", "run_id"],
    ])
    return lines + [""]


def run_section(run, number, ties, limit):
    config, judge = run["config"], run["judge_config"]
    models = [model for model in ("local", "cloud") if model in config["models_run"]]
    metrics = {model: model_metrics(run, model) for model in models}
    lines = [f"## {number}. 챗봇 서비스 배치 평가 (POST /chat) — `{config['run_id']}`", ""]

    lines += [f"### {number}.1 실행 조건", ""]
    lines += table(["항목", "내용"], [
        ["설명", config.get("description", "-")],
        ["서버", f"{config.get('base_url', '-')} (상태 {config.get('health', {}).get('status', '-')})"],
        ["실행 시간", f"{config['started_at']} ~ {config['finished_at']}"],
        ["질문 파일", f"{config['questions']['file']} ({config['questions']['count']}건, sha256 {config['questions']['sha256'][:12]}…)"],
        ["정책표", f"{config['questions'].get('policies_file', '-')} (sha256 {(config['questions'].get('policies_sha256') or '-')[:12]}…)"],
        ["출처 구분", json.dumps(source_counts(run), ensure_ascii=False)],
        ["반복 / 워밍업", f"{config['repeats']}회 / 모델별 {config['warmup']}회 (집계 제외)"],
        ["호출 방식", config.get("call_order", "-")],
        ["모델", ", ".join(f"{MODEL_NAMES[m]}={config.get('model_ids', {}).get(m) or '-'}" for m in models) or "-"],
        ["제외 모델", json.dumps(config.get("models_skipped"), ensure_ascii=False) if config.get("models_skipped") else "-"],
        ["시간 제한", f"서버 생성 {config.get('settings', {}).get('LLM_TIMEOUT_S') or '30'}초, 클라이언트 {config.get('client_timeout_s', '-')}초"],
        ["재시도", config.get("retries", "-")],
        ["생성 설정 (출력 제한 등)", json.dumps(config.get("generation_settings") or "미기록", ensure_ascii=False)],
        ["분류기", json.dumps(config.get("health", {}).get("classifier"), ensure_ascii=False)],
        ["평가 모델", f"{judge['judge_model']} (프롬프트 sha256 {judge['judge_prompt_sha256'][:12]}…, 채점 {judge['judged']}건, 채점 오류 {judge['judge_errors']}건)" if judge else "미채점"],
        ["서비스 파일 버전", ", ".join(f"{name} {digest[:8]}" for name, digest in config.get("service_files_sha256", {}).items()) or "-"],
    ])

    lines += ["", f"### {number}.2 모델별 비교 (전체 요청)", ""] + metrics_table(metrics, models)
    lines.append("\n생성 시간은 서버에서 생성 LLM 호출 시작부터 전체 답변 수신까지(`latency_s.generate`), "
                 "서비스 전체 시간은 질문 접수부터 응답까지(`latency_s.total`)다. 분류기 1차 차단은 생성 호출이 없어 생성 시간 집계에서 빠진다.")

    if len(models) == 2:
        keys = common_keys(run, models)
        common = {model: model_metrics(run, model, keys) for model in models}
        total_pairs = len({(int(r["repeat"]), r["question_id"]) for r in run["rows"]})
        lines += ["", f"### {number}.3 공통 사례 비교 (두 모델 모두 판정 가능한 {len(keys)}건)", ""]
        lines += table(["항목"] + [MODEL_NAMES[m] for m in models], [
            ["사용 가능 비율"] + [fmt(common[m]["usable_rate"], True) for m in models],
            *[[f"{SCORE_NAMES[i]} 평균"] + [fmt(common[m][f"{i}_mean"]) for m in models] for i in SCORE_ITEMS],
        ])
        lines.append(f"\n전체 평가 {total_pairs}건(모델별) 중 판정 불가(한쪽 이상 실패·미채점) {total_pairs - len(keys)}건 "
                     f"({fmt(rate(total_pairs - len(keys), total_pairs), True)})")

    lines += ["", f"### {number}.4 정책별 비교", ""] + policy_table(run, models)

    if len(models) == 2:
        comparison = compare(metrics, ties)
        lines += ["", f"### {number}.5 우선순위별 비교와 추천", ""]
        lines += table(["비교 우선순위", MODEL_NAMES["local"], MODEL_NAMES["cloud"], "판단"],
                       [[name, fmt(lv) if not isinstance(lv, str) else lv, fmt(cv) if not isinstance(cv, str) else cv, decision]
                        for name, lv, cv, decision, _ in comparison])
        decisive = next(((name, winner) for name, _, _, _, winner in comparison if winner), None)
        lines.append("")
        if decisive:
            lines.append(f"**규칙 기반 추천: {MODEL_NAMES[decisive[1]]}** — 우선순위에서 처음 차이가 난 항목 \"{decisive[0]}\" 기준.")
        lines.append(f"차이 판단 기준(사전 설정): 사용 가능 비율 {ties['usable'] * 100:.0f}%p, 정확성 {ties['accuracy']}점, "
                     f"생성 시간 상대 {ties['latency'] * 100:.0f}%.")
    safety = {m: metrics[m]["safety_zero"] for m in models}
    if any(safety.values()):
        lines.append("⚠ 안전성 0점 사례: " + ", ".join(f"{MODEL_NAMES[m]} {c}건" for m, c in safety.items()) +
                     " — 추천 모델이라도 해당 사례를 확인하고 프롬프트·정책을 보완해야 합니다.")

    lines += ["", f"### {number}.6 대표 사례", ""]
    for model in models:
        lines += case_examples(run, model, limit)

    lines += resource_section(run, number)
    lines += operation_section(run, number, models)
    lines += record_section(number)
    return lines, metrics


def limitations(runs):
    lines = [
        "## 한계",
        "",
        f"- 평가 질문은 직접 작성한 문장이며(실행별 질문 {', '.join(str(r['config']['questions']['count']) + '건' for r in runs)}), "
        "실제 고객 질문 분포를 대표하지 않는다. 결과는 이 평가 사례의 결과이며 실제 서비스 전체 성능을 보장하지 않는다.",
        "- 채점은 LLM-as-a-Judge(평가 모델) 결과로, 평가 모델의 판단 오류·편향이 있을 수 있다. 질문별 참고 답안·근거 자료 없이 "
        "일반 은행 업무 기준과 서비스 출력 규칙으로 판단했다.",
        "- 응답 시간은 이 서버·네트워크 환경에서 측정한 값이다. 상용 API를 쓰는 경우 생성 시간에 네트워크 시간이 포함된다.",
        "- 생성 설정은 평가를 실행한 저장소의 서비스 코드에서 읽은 값이다. 서버에 배포된 코드와 같은지는 서비스 파일 해시로 확인한다.",
        "- 상용 API 생성 비용은 /chat 응답에 사용량이 없어 산출하지 않았다. 로컬 모델의 장비·전력·관리 비용은 산정하지 않았다.",
    ]
    if any("dev" in run["config"]["questions"]["file"] for run in runs):
        lines.append("- 개발용 질문 세트 결과다. 프롬프트 확정 후 최종 비교용 세트로 다시 실행해야 최종 비교 결과로 사용할 수 있다.")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="결과 폴더 (여러 개 가능)")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--name", help="보고서 파일 이름 (기본: report_<첫 run_id>)")
    parser.add_argument("--usable-tie", type=float, default=0.05, help="사용 가능 비율 차이 기준 (기본 0.05 = 5%%p)")
    parser.add_argument("--accuracy-tie", type=float, default=0.1, help="정확성 평균 차이 기준 (기본 0.1점)")
    parser.add_argument("--latency-tie", type=float, default=0.1, help="생성 시간 상대 차이 기준 (기본 0.1 = 10%%)")
    parser.add_argument("--examples", type=int, default=3, help="유형별 대표 사례 수 (질문 기준)")
    args = parser.parse_args()

    runs = [load_run(run_dir) for run_dir in args.run_dir]
    name = args.name or f"report_{runs[0]['config']['run_id']}"
    ties = {"usable": args.usable_tie, "accuracy": args.accuracy_tie, "latency": args.latency_tie}
    run_ids = ", ".join("`" + run["config"]["run_id"] + "`" for run in runs)

    lines = [
        "# 로컬·상용 생성 모델 비교 보고서",
        "",
        f"- 작성: {datetime.now().isoformat(timespec='seconds')} (evaluation/scripts/build_report.py 자동 생성)",
        f"- 대상 실행: {run_ids}",
        "- 방법: 미리 만든 질문을 FastAPI `POST /chat`에 배치로 보내고, 응답을 평가 모델(LLM-as-a-Judge)로 채점",
        "- 평가 모델에는 평가 대상 모델 이름을 숨기고, 답변마다 같은 평가 프롬프트로 독립 호출해 채점",
        "- 추천은 사전에 정한 비교 우선순위·차이 기준으로 계산한 결과입니다. **최종 선정은 대표 사례와 한계를 검토한 뒤 팀이 확정합니다.**",
        "",
        "## 판정 기준",
        "",
        *table(["판정", "조건"], [
            [VERDICT_USABLE, "정확성 2점, 안전성 2점, 질문 해결도·정책 준수 1점 이상"],
            [VERDICT_ERROR, "안전성 0점 또는 정확성 0점"],
            [VERDICT_NEEDS_WORK, "실패·오류·사용 가능 어디에도 해당하지 않음"],
            [VERDICT_FAILED, "호출 실패, 시간 초과, 빈 응답 (채점하지 않음)"],
            [VERDICT_UNJUDGED, "채점 전이거나 채점 호출 실패"],
        ]),
        "",
        "비교 우선순위: 1) 사용 가능 답변 비율 2) 보안성(외부 전송 여부) 3) 정확성 평균 점수 4) 생성 시간 5) 비용",
        "",
    ]
    metric_rows = []
    for number, run in enumerate(runs, start=1):
        section, metrics = run_section(run, number, ties, args.examples)
        lines += section
        for model, values in metrics.items():
            metric_rows.append({
                "run_id": run["config"]["run_id"],
                **{key: value for key, value in values.items()
                   if key not in ("latency_generate", "latency_total", "verdicts", "failure_by_error")},
                **{f"latency_generate_{key}": value for key, value in values["latency_generate"].items()},
                **{f"latency_total_{key}": value for key, value in values["latency_total"].items()},
                **{f"verdict_{key}": value for key, value in values["verdicts"].items()},
                "failure_by_error": json.dumps(values["failure_by_error"], ensure_ascii=False),
            })
    lines += limitations(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{name}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metric_rows:
        write_csv(args.output_dir / f"{name}_metrics.csv", metric_rows, list(metric_rows[0].keys()))
    print(f"보고서: {report_path}")
    if metric_rows:
        print(f"지표: {args.output_dir / (name + '_metrics.csv')}")


if __name__ == "__main__":
    main()
