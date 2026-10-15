import os
import urllib.request
from functools import lru_cache

import httpx
from langchain_core.output_parsers import StrOutputParser
from openai import APITimeoutError
from langchain_ollama import ChatOllama

from app import config
from app.messages import MESSAGES, match_fixed_message, normalize
from app.services.policy import PROMPT


class LLMNotConfigured(Exception):
    """요청한 생성 모델을 사용할 설정이나 구현이 없음."""


@lru_cache(maxsize=1)
def get_local_chain():
    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL,
        temperature=0,
        reasoning=False,
        num_ctx=4096,
        num_predict=256,
        client_kwargs={"timeout": config.LLM_TIMEOUT_S},
    )

    return PROMPT | llm | StrOutputParser()


def cloud_api_key() -> str | None:
    return os.getenv(config.CLOUD_API_KEY_ENV) or None


@lru_cache(maxsize=1)
def get_cloud_chain():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=config.CLOUD_LLM_MODEL,
        api_key=cloud_api_key(),
        temperature=0,
        max_tokens=256,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=0,
    )

    return PROMPT | llm | StrOutputParser()


def generate_answer(question: str, label: str, model: str) -> str:
    if model not in ("local", "cloud"):
        raise ValueError(f"지원하지 않는 생성 모델 선택: {model}")

    if model == "cloud" and not is_cloud_configured():
        raise LLMNotConfigured(f"상용 LLM 키({config.CLOUD_API_KEY_ENV})가 설정되지 않았습니다.")

    chain = get_cloud_chain() if model == "cloud" else get_local_chain()
    try:
        answer = chain.invoke({
            "question": question,
            "label": label,
        }).strip()
    except (httpx.TimeoutException, APITimeoutError) as exc:
        raise TimeoutError("생성 모델 응답 제한시간 초과") from exc

    if not answer:
        raise RuntimeError("생성 모델이 빈 답변을 반환했습니다.")

    # 고정 문구로 시작하면 뒤에 붙은 추가 설명을 제거
    normalized_answer = normalize(answer)

    for key in ("out_of_scope", "unclear"):
        if normalized_answer.startswith(normalize(MESSAGES[key])):
            return MESSAGES[key]

    # 기존 고정 문구 판별도 유지
    matched = match_fixed_message(answer)
    return MESSAGES[matched] if matched else answer


def is_local_available() -> bool:
    """Ollama 서버 응답 여부 확인."""
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=2,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def is_cloud_configured() -> bool:
    # 키가 있을 때만 상용 모델을 준비된 것으로 본다 (호출 자체는 실제 요청으로 확인)
    return cloud_api_key() is not None
