"""생성 LLM 연결 지점.

LangChain 담당이 처리 흐름을 완성하면 **이 파일의 generate_answer만** 실제 코드로 바꾼다.
FastAPI의 나머지 코드는 고칠 필요가 없다.

약속 (docs/api_spec.md)
- 입력: 사용자 원문, 분류기 예측 주제, 사용할 LLM("local" / "cloud")
- 출력: LLM 답변 문자열. 업무 외·의도 불명으로 판단하면 고정 문구 ①·②를 가공하지 않고 그대로 반환
- 실패: 예외를 발생시킨다. (FastAPI가 잡아서 고정 문구 ③으로 응답)
"""

import os
import urllib.request

from app import config


class LLMNotConfigured(Exception):
    """요청한 LLM을 사용할 설정이 없음 (예: 상용 API 키 없음)."""


def generate_answer(question: str, label: str, model: str) -> str:
    # ---- 임시 구현: LangChain 처리 흐름이 연결되면 아래를 교체 ----
    if model == "cloud" and not is_cloud_configured():
        raise LLMNotConfigured("상용 LLM API 키가 설정되지 않았습니다.")
    return f"[임시 답변] '{label}' 유형의 답변 지침으로 안내할 예정입니다. (질문: {question[:30]})"
    # ----------------------------------------------------------------


def is_local_available() -> bool:
    """Ollama 서버가 응답하는지 확인."""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_cloud_configured() -> bool:
    """상용 LLM API 키가 환경 변수에 있는지 확인. (키 값은 읽기만 하고 기록하지 않는다)"""
    return bool(os.getenv(config.CLOUD_API_KEY_ENV))
