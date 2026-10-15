"""실행: python -m streamlit run ui/chatbot.py"""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
# API의 생성 제한 시간보다 조금 더 기다린다.
TIMEOUT = float(os.getenv("LLM_TIMEOUT_S", "30")) + 15


def request_answer(question):
    """기존 API를 호출한다. HTTP 200이어도 route=error이면 실패다."""
    try:
        response = httpx.post(
            f"{API_URL}/chat",
            json={"question": question, "model": "local"},
            timeout=httpx.Timeout(TIMEOUT, connect=5),
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ValueError("잘못된 응답 형식")
        return {
            "role": "assistant",
            "content": result["answer"],
            "label": result.get("label"),
            "failed": result.get("route") == "error",
        }
    except httpx.TimeoutException:
        message = "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    except httpx.HTTPStatusError as exc:
        message = (
            f"질문을 확인해 주세요. 공백을 제외한 내용이 있어야 하며 {MAX_CHARS}자 이하여야 합니다."
            if exc.response.status_code == 422
            else "현재 답변을 제공하기 어렵습니다. 잠시 후 다시 시도해 주세요."
        )
    except httpx.RequestError:
        message = "서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
    except ValueError:
        message = "서버 응답을 읽을 수 없습니다. 잠시 후 다시 시도해 주세요."
    return {"role": "assistant", "content": message, "failed": True}


st.set_page_config(page_title="은행 상담 챗봇", page_icon="🏦")
st.title("은행 상담 챗봇")
st.caption("은행 업무의 일반적인 절차를 안내합니다. 각 질문은 이전 대화와 관계없이 처리됩니다.")

# 접속한 사용자별로 화면 기록만 보관한다. API에는 새 질문 하나만 보낸다.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
busy = st.session_state.pending is not None

if st.button("화면 초기화", disabled=busy):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("label"):
            st.caption(f"분류된 상담 유형: {message['label']}")
        if message.get("failed"):
            st.error(message["content"])
        else:
            st.write(message["content"])

question = st.chat_input(
    "은행 업무를 질문해 주세요.",
    max_chars=MAX_CHARS,
    disabled=busy,
)
if question:
    question = question.strip()
    if not question:
        st.warning("질문을 입력해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending = question
        # 입력창을 비활성화한 화면을 먼저 그린 후 API를 호출한다.
        st.rerun()

if busy:
    with st.spinner("답변을 생성하고 있습니다…"):
        result = request_answer(st.session_state.pending)
    st.session_state.messages.append(result)
    st.session_state.pending = None
    st.rerun()
