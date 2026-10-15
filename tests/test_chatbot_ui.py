"""UI와 API의 연결 동작을 실제 모델 없이 확인한다."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "chatbot.py"


@pytest.mark.parametrize("route", ["answered", "blocked_by_classifier", "blocked_by_llm", "unclear", "error"])
def test_response_and_reset(route):
    app = AppTest.from_file(str(UI_PATH)).run()
    response = httpx.Response(
        200, request=httpx.Request("POST", "http://test/chat"),
        json={"answer": "테스트 안내 문구", "label": "환전문의", "route": route},
    )
    with patch("httpx.post", return_value=response) as post:
        app.chat_input[0].set_value("환전 신청 방법").run()
        assert not app.exception
        assert post.call_count == 1
        assert post.call_args.kwargs["json"] == {"question": "환전 신청 방법", "model": "local"}
        assert len(app.chat_message) == 2
        assert len(app.error) == (1 if route == "error" else 0)
        assert not app.chat_input[0].disabled
        # 일반적인 화면 재실행이나 초기화로 같은 요청을 다시 전송하지 않는다.
        app.run()
        app.button[0].click().run()
        assert len(app.chat_message) == 0
        assert post.call_count == 1


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("timeout")])
def test_connection_failure_recovers(failure):
    app = AppTest.from_file(str(UI_PATH)).run()
    with patch("httpx.post", side_effect=failure):
        app.chat_input[0].set_value("환전 신청 방법").run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.chat_input[0].disabled


def test_blank_question_is_not_sent():
    app = AppTest.from_file(str(UI_PATH)).run()
    with patch("httpx.post") as post:
        app.chat_input[0].set_value("   ").run()
    assert not app.exception
    post.assert_not_called()
    assert len(app.warning) == 1
