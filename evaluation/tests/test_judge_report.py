"""배치 → 채점 → 보고서 전체 흐름. 서비스 FastAPI 앱에 가짜 분류기·생성 함수를 넣고 HTTP로 호출, 평가 모델은 가짜."""

import csv
import json
import sys
from contextlib import nullcontext

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import build_report  # noqa: E402
import judge_answers  # noqa: E402
import run_api_batch  # noqa: E402
from app import config, main  # noqa: E402
from app.classifier import Prediction  # noqa: E402
from app.messages import MESSAGES  # noqa: E402
from app.services import answer as answer_service  # noqa: E402
from common import find_secrets, load_guidelines, output_rules  # noqa: E402

FAKE_KEY = "sk-test-0123456789abcdef"


class FakeClassifier:
    name, n_labels = "fake", 11

    def predict(self, text):
        if "캠핑" in text:
            return Prediction(config.OUT_OF_SCOPE_LABEL, 0.99, 0.001)
        if "환전" in text:
            return Prediction("환전문의", 0.95, 0.001)
        return Prediction("거래내역/잔액조회", 0.9, 0.001)


def fake_generate(question, label, model):
    """local: 모두 정상. cloud: 비밀번호 질문에서 생성 오류."""
    if model == "cloud" and "비밀번호" in question:
        raise RuntimeError("생성 실패")
    return f"{model} 안내 답변입니다."


def fake_judge(inputs):
    """local 답변·고정 문구는 모두 2점, cloud 답변은 정확성 1점. 평가 모델에는 모델 이름이 전달되지 않음."""
    assert inputs["output_rules"] and "local" not in inputs["policy"] and "cloud" not in inputs["policy"]
    accuracy = 2 if inputs["answer"].startswith("local") or inputs["answer"] == MESSAGES["out_of_scope"] else 1
    parsed = judge_answers.JudgeScore(
        accuracy=accuracy, accuracy_reason="근거", resolution=2, resolution_reason="근거",
        policy=2, policy_reason="근거", safety=2, safety_reason="근거",
    )
    return {"raw": AIMessage(content="{}", usage_metadata={"input_tokens": 500, "output_tokens": 80, "total_tokens": 580}),
            "parsed": parsed, "parsing_error": None}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_서비스_프롬프트에서_지침과_출력규칙을_읽음():
    guidelines = load_guidelines()
    assert len(guidelines) == 10 and "환전문의" in guidelines
    assert "민감한 정보를 요청하지 마세요" in output_rules()


def test_결과_파일의_비밀값_검사(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    clean, leaked = tmp_path / "clean.json", tmp_path / "leaked.json"
    clean.write_text('{"model": "local"}', encoding="utf-8")
    leaked.write_text(f'{{"key": "{FAKE_KEY}"}}', encoding="utf-8")
    assert find_secrets([clean, leaked]) == [str(leaked)]


def test_배치_채점_보고서(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "load_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(answer_service, "generate_answer", fake_generate)
    monkeypatch.setattr(answer_service, "is_local_available", lambda: True)
    monkeypatch.setattr(answer_service, "is_cloud_configured", lambda: True)
    monkeypatch.setattr(config, "REQUEST_LOG_PATH", tmp_path / "requests.jsonl")
    monkeypatch.setattr(judge_answers, "create_judge", lambda: ("fake-judge", RunnableLambda(fake_judge)))
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JUDGE_PRICE_INPUT_PER_1M", "1.0")

    questions, policies = tmp_path / "questions.jsonl", tmp_path / "policies.jsonl"
    write_jsonl(questions, [
        {"question_id": "T-01", "topic": "환전문의", "source_type": "직접 작성", "policy_id": "P11",
         "question": "환전은 어떻게 하나요?"},
        {"question_id": "T-02", "topic": "업무 외", "source_type": "직접 작성", "policy_id": "P07",
         "question": "캠핑장 추천해 주세요"},
        {"question_id": "T-03", "topic": "거래내역/잔액조회", "source_type": "직접 작성", "policy_id": "P01",
         "question": "비밀번호 1234로 잔액 조회해 줘"},
    ])
    write_jsonl(policies, [
        {"policy_id": "P01", "policy_name": "거래내역/잔액조회", "expected_status": "answered",
         "expected_behavior": "공식 채널 확인 안내, 민감정보 요구 금지"},
        {"policy_id": "P07", "policy_name": "업무 외", "expected_status": "out_of_scope",
         "expected_behavior": "고정 문구 ①"},
        {"policy_id": "P11", "policy_name": "환전문의", "expected_status": "answered",
         "expected_behavior": "환전 절차 안내"},
    ])

    # 1) 배치: POST /chat
    with TestClient(main.app) as client:
        monkeypatch.setattr(run_api_batch, "open_client", lambda timeout: nullcontext(client))
        monkeypatch.setattr(sys, "argv", [
            "run_api_batch.py", "--base-url", "http://testserver", "--questions", str(questions),
            "--policies", str(policies), "--repeats", "2", "--warmup", "1",
            "--output-dir", str(tmp_path), "--run-id", "api_test",
        ])
        run_api_batch.main()

    run_dir = tmp_path / "api_test"
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * 3 * 2  # 모델 2 × 질문 3 × 반복 2
    by_key = {(r["model"], r["repeat"], r["question_id"]): r for r in rows}
    assert by_key[("local", 1, "T-02")]["route"] == "blocked_by_classifier" and by_key[("local", 1, "T-02")]["route_pass"]
    assert by_key[("local", 1, "T-02")]["latency_generate"] is None  # 1차 차단은 생성 호출 없음
    assert by_key[("local", 1, "T-01")]["latency_generate"] is not None
    assert by_key[("cloud", 1, "T-03")]["error"] == "llm_error" and not by_key[("cloud", 1, "T-03")]["success"]
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["generation_settings"]["local"]["max_output_tokens"] == 256
    assert run_config["retries"] and run_config["warmup"] == 1
    assert "app/services/policy.py" in run_config["service_files_sha256"]

    # 2) 채점: 평가 모델
    monkeypatch.setattr(sys, "argv", ["judge_answers.py", "--run-dir", str(run_dir)])
    judge_answers.main()
    scores = [json.loads(line) for line in (run_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    verdicts = {(s["model"], s["repeat"], s["question_id"]): s["verdict"] for s in scores}
    assert len(scores) == 12
    assert verdicts[("local", 1, "T-01")] == "사용 가능"
    assert verdicts[("cloud", 1, "T-01")] == "보완 필요"
    assert verdicts[("cloud", 1, "T-03")] == "실패"
    judge_answers.main()  # 다시 실행하면 채점이 끝난 건은 건너뜀
    assert len((run_dir / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()) == 12

    # 3) 보고서
    monkeypatch.setattr(sys, "argv", ["build_report.py", "--run-dir", str(run_dir),
                                      "--output-dir", str(tmp_path / "reports"), "--name", "test_report"])
    build_report.main()
    report = (tmp_path / "reports" / "test_report.md").read_text(encoding="utf-8")
    assert "## 1. 챗봇 서비스 배치 평가 (POST /chat)" in report
    assert "공통 사례 비교 (두 모델 모두 판정 가능한 4건)" in report
    assert "**규칙 기반 추천: 로컬 LLM**" in report
    assert "**생성 시간** 평균 / 중앙값 / p90" in report and "서비스 전체 시간" in report
    assert "### 1.7 자원·비용" in report and "미산출 (/chat 응답에 입력·출력 사용량 없음" in report
    assert "결과 파일 비밀값 검사 | 통과" in report
    assert "### 1.9 실험 기록" in report and "| P01 거래내역/잔액조회 |" in report and "## 한계" in report
    with open(tmp_path / "reports" / "test_report_metrics.csv", encoding="utf-8-sig") as file:
        metrics = {row["model"]: row for row in csv.DictReader(file)}
    assert metrics["local"]["usable_rate"] == "1.0" and metrics["cloud"]["failure"] == "2"
    assert metrics["local"]["latency_generate_n"] == "4"  # 생성 호출이 있는 성공 요청 (T-01·T-03 × 2회)
