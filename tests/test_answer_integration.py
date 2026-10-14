"""실제 생성 어댑터와 API 연결을 검증한다. 외부 LLM 호출만 대체한다."""

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.classifier import Prediction
from app.messages import MESSAGES
from app.services import answer as answer_service


class BankClassifier:
    name, n_labels = "integration fake", 11

    def predict(self, text):
        return Prediction("환전문의", 0.9, 0.001)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_classifier", lambda: BankClassifier())
    monkeypatch.setattr(config, "REQUEST_LOG_PATH", tmp_path / "requests.jsonl")
    with TestClient(main.app) as client:
        yield client


def set_chain(monkeypatch, output=None, failure=None):
    class Chain:
        def invoke(self, inputs):
            assert inputs == {"question": "환전 문의", "label": "환전문의"}
            if failure is not None:
                raise failure
            return output

    monkeypatch.setattr(answer_service, "get_local_chain", lambda: Chain())


@pytest.mark.parametrize(
    "key, route",
    [("out_of_scope", "blocked_by_llm"), ("unclear", "unclear")],
)
def test_fixed_response_discards_followup(client, monkeypatch, key, route):
    set_chain(monkeypatch, f'“{MESSAGES[key]}”\n어떤 문의인지 자세히 알려주시면 다음 질문에서 안내하겠습니다.')
    response = client.post("/chat", json={"question": "환전 문의", "model": "local"})
    body = response.json()
    assert response.status_code == 200
    assert body["route"] == route
    assert body["answer"] == MESSAGES[key]
    assert body["error"] is None


def test_transport_timeout_uses_timeout_route(client, monkeypatch):
    set_chain(monkeypatch, failure=httpx.ReadTimeout("test timeout"))
    body = client.post("/chat", json={"question": "환전 문의", "model": "local"}).json()
    assert body["route"] == "error"
    assert body["error"] == "llm_timeout"
    assert body["answer"] == MESSAGES["error"]


def test_empty_generation_uses_error_route(client, monkeypatch):
    set_chain(monkeypatch, "   ")
    body = client.post("/chat", json={"question": "환전 문의", "model": "local"}).json()
    assert body["route"] == "error"
    assert body["error"] == "llm_error"
    assert body["answer"] == MESSAGES["error"]


def test_unconfigured_cloud_does_not_call_local(client, monkeypatch):
    def unexpected_call():
        pytest.fail("cloud 요청에서 local 체인을 호출했습니다")

    monkeypatch.setattr(answer_service, "get_local_chain", unexpected_call)
    body = client.post("/chat", json={"question": "환전 문의", "model": "cloud"}).json()
    assert body["route"] == "error"
    assert body["error"] == "llm_not_configured"
    assert body["answer"] == MESSAGES["error"]
