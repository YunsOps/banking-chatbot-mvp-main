"""배치 평가(run_api_batch.py) 전체 실행 확인. 서비스 FastAPI 앱에 가짜 분류기·생성 함수를 넣고 HTTP로 호출한다."""

import csv
import importlib
import json
import sys
from contextlib import nullcontext

from fastapi.testclient import TestClient

import common  # noqa: E402
import run_api_batch  # noqa: E402
from app import config, main  # noqa: E402
from app.classifier import Prediction  # noqa: E402
from app.messages import MESSAGES  # noqa: E402
from app.services import answer as answer_service  # noqa: E402


class FakeClassifier:
    name, n_labels = "fake", 11

    def predict(self, text):
        if "캠핑" in text:
            return Prediction(config.OUT_OF_SCOPE_LABEL, 0.99, 0.001)
        return Prediction("기타(은행)", 0.8, 0.001)


def fake_generate(question, label, model):
    # 서비스 generate_answer처럼 상용 LLM은 미설정 예외
    if model == "cloud":
        raise answer_service.LLMNotConfigured("상용 LLM 미설정")
    if "그거" in question:
        return MESSAGES["unclear"]
    return f"{label} 안내 답변입니다."


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_호출_주소는_env의_EVAL_API_BASE_URL(monkeypatch):
    monkeypatch.setenv("EVAL_API_BASE_URL", "http://10.2.10.71:8000/")
    try:
        assert importlib.reload(common).API_BASE_URL == "http://10.2.10.71:8000"
    finally:
        monkeypatch.delenv("EVAL_API_BASE_URL")
        importlib.reload(common)


def test_배치_실행_결과(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "load_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(answer_service, "generate_answer", fake_generate)
    monkeypatch.setattr(answer_service, "is_local_available", lambda: True)
    monkeypatch.setattr(answer_service, "is_cloud_configured", lambda: False)
    monkeypatch.setattr(config, "REQUEST_LOG_PATH", tmp_path / "requests.jsonl")
    monkeypatch.setenv("EVAL_API_LOCAL_MODEL", "qwen-on-server")

    questions, policies = tmp_path / "questions.jsonl", tmp_path / "policies.jsonl"
    write_jsonl(questions, [
        {"question_id": "T-01", "topic": "기타(은행)", "source_type": "직접 작성", "policy_id": "P03",
         "question": "통장 재발급 방법"},
        {"question_id": "T-02", "topic": "업무 외", "source_type": "직접 작성", "policy_id": "P07",
         "question": "캠핑장 추천해 주세요"},
        {"question_id": "T-03", "topic": "기타(은행)", "source_type": "직접 작성", "policy_id": "P03",
         "question": "그거 어떻게 됐어요?"},
    ])
    write_jsonl(policies, [
        {"policy_id": "P03", "policy_name": "기타(은행)", "expected_status": "answered",
         "expected_behavior": "일반 안내"},
        {"policy_id": "P07", "policy_name": "업무 외", "expected_status": "out_of_scope", "expected_behavior": "고정 문구 ①"},
    ])

    with TestClient(main.app) as client:
        # 스크립트의 HTTP 클라이언트를 이미 시작한 TestClient로 교체 (with로 다시 열지 않도록 nullcontext)
        monkeypatch.setattr(run_api_batch, "open_client", lambda timeout: nullcontext(client))
        monkeypatch.setattr(sys, "argv", [
            "run_api_batch.py", "--base-url", "http://testserver", "--questions", str(questions),
            "--policies", str(policies), "--repeats", "2", "--warmup", "1",
            "--output-dir", str(tmp_path), "--run-id", "test",
        ])
        run_api_batch.main()

    run_dir = tmp_path / "test"
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    # local만 준비됨 → 질문 3개 × 반복 2회. 워밍업은 집계 제외
    assert len(rows) == 6 and {row["model"] for row in rows} == {"local"}
    assert {row["question_id"]: row["route"] for row in rows} == {
        "T-01": "answered", "T-02": "blocked_by_classifier", "T-03": "unclear"}
    assert all(row["success"] and row["label_match"] for row in rows)
    # 답변을 기대한 질문에 의도 불명 문구가 오면 기대 route 불일치
    assert {row["question_id"]: row["route_pass"] for row in rows} == {"T-01": True, "T-02": True, "T-03": False}

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["eval_type"] == "api_integrated" and run_config["base_url"] == "http://testserver"
    assert run_config["models_run"] == ["local"] and "cloud" in run_config["models_skipped"]
    assert run_config["model_ids"]["local"] == "qwen-on-server"

    with open(run_dir / "summary.csv", encoding="utf-8-sig") as file:
        summary = next(csv.DictReader(file))
    assert summary["route_pass"] == "4" and summary["failure"] == "0"
    with open(run_dir / "scenarios.csv", encoding="utf-8-sig") as file:
        scenarios = list(csv.DictReader(file))
    assert all(scenario["passed"] == "True" for scenario in scenarios)
    assert any(scenario["scenario"] == "cloud LLM 미준비" for scenario in scenarios)
