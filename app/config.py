"""서비스 설정. 값은 환경 변수(.env)로 바꿀 수 있고, 없으면 아래 기본값을 사용한다.

기본값은 학습 담당 스크립트의 저장 형식에 맞춘다.
- BERT: training/evaluate_klue_bert_base.py → models/klue_bert_base (save_pretrained, 토크나이저 포함), max_length 512
- 베이스라인: training/train_tfidf_logistic_regression.py → model/tfidf_logistic_regression.joblib
  ({"vectorizer": ..., "model": ...} 형태로 저장)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_DIR / value


# 분류기
CLASSIFIER_TYPE = os.getenv("CLASSIFIER_TYPE", "bert")              # "bert" 또는 "baseline"
BERT_MODEL_DIR = _path("BERT_MODEL_DIR", "models/klue_bert_base")
BERT_TOKENIZER_FALLBACK = os.getenv("BERT_TOKENIZER_FALLBACK", "klue/bert-base")
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "512"))         # 학습 스크립트와 같은 값
BASELINE_MODEL_PATH = _path("BASELINE_MODEL_PATH", "model/tfidf_logistic_regression.joblib")
# 베이스라인이 직접 만든 토크나이저(training/korean_tokenizer.py)를 쓰면, 불러올 때 이 폴더에서 찾는다
TRAINING_DIR = _path("TRAINING_DIR", "training")
OUT_OF_SCOPE_LABEL = os.getenv("OUT_OF_SCOPE_LABEL", "업무 외")

# 요청 제한 [합의 필요]
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "local")                    # "local" 또는 "cloud"

# 고정 문구 파일 (없으면 기본 문구 사용)
FIXED_MESSAGES_PATH = _path("FIXED_MESSAGES_PATH", "prompts/fixed_messages.json")

# LLM 연결 확인용
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CLOUD_API_KEY_ENV = os.getenv("CLOUD_API_KEY_ENV", "OPENAI_API_KEY")  # 상용 API 키가 들어 있는 환경 변수 이름

# 실행 기록 (Git 제외: /logs/)
REQUEST_LOG_PATH = _path("REQUEST_LOG_PATH", "logs/requests.jsonl")

# 챗봇 화면에서 호출할 수 있도록 허용할 주소 (쉼표로 구분)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
