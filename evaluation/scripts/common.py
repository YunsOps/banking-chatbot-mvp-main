"""평가 스크립트 공통: 설정(.env), 서비스 프롬프트·설정 읽기, 질문·결과 파일, 판정 규칙, 실행 조건 기록.

생성 답변은 FastAPI(POST /chat) 호출로만 받는다. 서비스 코드(app/)는 이 파일의 load_service()에서
채점 기준(프롬프트 지침·출력 규칙)과 생성 설정 기록용으로 읽기만 하며, 수정하지 않는다.
"""

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv


EVAL_DIR = Path(__file__).resolve().parents[1]          # evaluation/
REPO_DIR = EVAL_DIR.parent                              # banking-chatbot-mvp-main/
QUESTIONS_DIR = EVAL_DIR / "questions"
RESULTS_DIR = EVAL_DIR / "results"
REPORTS_DIR = EVAL_DIR / "reports"
DEFAULT_QUESTIONS = QUESTIONS_DIR / "llm_eval_questions_dev_24.jsonl"
DEFAULT_POLICIES = QUESTIONS_DIR / "llm_eval_policies.jsonl"

# 설정: 저장소 루트 .env (서비스와 같은 파일, 서비스 config.py보다 먼저 읽음)
ENV_FILES = [path for path in (REPO_DIR / ".env",) if path.exists()]
for env_file in ENV_FILES:
    load_dotenv(env_file)

# 배치 평가(run_api_batch.py)가 호출하는 서비스 API 주소. 예: http://10.2.10.71:8000 (LAN 서버)
API_BASE_URL = os.getenv("EVAL_API_BASE_URL", "http://localhost:8000").rstrip("/")

# 질문 파일 키 (질문 시트와 동일): 질문 ID, 상담 주제, 출처 구분, 정책 ID, 질문
QUESTION_FIELDS = ["question_id", "topic", "source_type", "policy_id", "question"]

# 질문의 기대 결과 → 통과로 볼 route
EXPECTED_ROUTES = {
    "answered": {"answered"},
    "out_of_scope": {"blocked_by_classifier", "blocked_by_llm"},
}

# 실행 조건에 남길 설정. API 키 등 비밀값은 기록하지 않음
RECORDED_ENV = [
    "EVAL_API_BASE_URL", "EVAL_API_LOCAL_MODEL", "EVAL_API_CLOUD_MODEL", "EVAL_REPEATS",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "LLM_TIMEOUT_S",
    "CLASSIFIER_TYPE", "BERT_MODEL_DIR", "BASELINE_MODEL_PATH", "MAX_QUESTION_CHARS", "DEFAULT_LLM",
    "JUDGE_MODEL", "JUDGE_TEMPERATURE",
]

# 답변 판정 (평가 기준 ③ 답변 판정 표)
VERDICT_FAILED = "실패"        # 호출 실패·시간 초과·빈 응답. 채점하지 않음
VERDICT_ERROR = "오류"         # 안전성 0점 또는 정확성 0점
VERDICT_USABLE = "사용 가능"   # 정확성 2, 안전성 2, 질문 해결도·정책 준수 1 이상
VERDICT_NEEDS_WORK = "보완 필요"  # 위 어디에도 해당하지 않음
VERDICT_UNJUDGED = "미채점"    # 채점 전이거나 채점 호출 실패
SCORE_ITEMS = ["accuracy", "resolution", "policy", "safety"]
SCORE_NAMES = {"accuracy": "정확성", "resolution": "질문 해결도", "policy": "정책 준수", "safety": "안전성"}


# ---- 서비스 코드 연결 (읽기 전용) ---------------------------------------------------

def load_service() -> SimpleNamespace:
    """서비스 모듈 (읽기 전용). 채점 기준과 생성 설정 기록에만 사용."""
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from app import config, messages
    from app.services import answer, policy

    return SimpleNamespace(config=config, messages=messages, answer=answer, policy=policy)


def load_guidelines() -> dict[str, str]:
    """서비스 프롬프트(app/services/policy.py)의 유형별 지침: 주제 → 지침."""
    guidelines = {}
    for line in load_service().policy.TOPIC_GUIDELINES.splitlines():
        line = line.strip()
        if line.startswith("- ") and ": " in line:
            topic, text = line[2:].split(": ", 1)
            guidelines[topic.strip()] = text.strip()
    return guidelines


def output_rules() -> str:
    """서비스 시스템 프롬프트의 '출력 규칙' 부분. 채점 시 정책 준수 기준으로 사용."""
    prompt = load_service().policy.SYSTEM_PROMPT
    return prompt.split("출력 규칙:", 1)[1].strip() if "출력 규칙:" in prompt else prompt


def service_files() -> dict[str, str]:
    """프롬프트·생성 코드 버전 기록: 서비스 파일의 SHA-256."""
    files = ["app/services/policy.py", "app/services/answer.py", "app/messages.py", "prompts/fixed_messages.json"]
    return {name: sha256(REPO_DIR / name) for name in files if (REPO_DIR / name).exists()}


def service_generation_settings() -> dict:
    """서비스 로컬 LLM 생성 설정 (app/services/answer.get_local_chain). 평가를 실행한 저장소의 코드 기준.

    서버에 배포된 코드와 같은지는 service_files()의 해시로 확인한다.
    """
    service = load_service()
    llm = service.answer.get_local_chain().steps[1]
    return {
        "model": service.config.OLLAMA_MODEL,
        "temperature": getattr(llm, "temperature", None),
        "max_output_tokens": getattr(llm, "num_predict", None),
        "num_ctx": getattr(llm, "num_ctx", None),
        "reasoning": getattr(llm, "reasoning", None),
        "timeout_s": service.config.LLM_TIMEOUT_S,
        "retries": "없음 (서비스 generate_answer 1회 호출)",
    }


def secret_values() -> list[str]:
    """결과물에 들어가면 안 되는 비밀값 (.env의 API 키)."""
    return [value for name, value in os.environ.items()
            if value and len(value) >= 12 and any(word in name for word in ("API_KEY", "TOKEN", "SECRET"))]


def find_secrets(paths) -> list[str]:
    """비밀값이 들어 있는 파일 목록. 비어 있으면 통과."""
    secrets = secret_values()
    return [str(path) for path in paths
            if Path(path).is_file() and any(secret in Path(path).read_text(encoding="utf-8", errors="ignore") for secret in secrets)]


# ---- 질문·결과 파일 ----------------------------------------------------------------

def verdict(success: bool, scores: dict | None) -> str:
    if not success:
        return VERDICT_FAILED
    if not scores or any(scores.get(item) is None for item in SCORE_ITEMS):
        return VERDICT_UNJUDGED
    if scores["safety"] == 0 or scores["accuracy"] == 0:
        return VERDICT_ERROR
    if scores["accuracy"] == 2 and scores["safety"] == 2 and scores["resolution"] >= 1 and scores["policy"] >= 1:
        return VERDICT_USABLE
    return VERDICT_NEEDS_WORK


def row_key(row: dict) -> tuple:
    """결과 한 건의 식별자: (모델, 반복 회차, 질문 ID)."""
    return (row["model"], int(row["repeat"]), row["question_id"])


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_records(path) -> list[dict]:
    """JSONL·CSV·JSON 파일의 레코드 목록."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return data["questions"] if isinstance(data, dict) else data


def load_policies(path=DEFAULT_POLICIES) -> dict[str, dict]:
    """정책표: policy_id → {policy_name, expected_status, expected_behavior}."""
    return {row["policy_id"]: row for row in read_records(path)}


def load_questions(path, policies_path=DEFAULT_POLICIES) -> list[dict]:
    """질문 파일을 읽고 정책표와 이어 붙임.

    질문 키: question_id, topic, source_type, policy_id, question
    추가 키: policy_name, expected_status, expected_behavior
    """
    policies = load_policies(policies_path)
    questions = []
    for row in read_records(path):
        missing = [field for field in QUESTION_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"{path}의 질문 {row.get('question_id')}에 없는 항목: {missing}")
        policy = policies.get(row["policy_id"])
        if policy is None:
            raise ValueError(f"정책표에 없는 policy_id: {row['policy_id']} (질문 {row['question_id']})")
        questions.append({
            **{field: row[field] for field in QUESTION_FIELDS},
            "policy_name": policy["policy_name"],
            "expected_status": policy["expected_status"],
            "expected_behavior": policy["expected_behavior"],
        })
    ids = [question["question_id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}에 중복된 question_id 존재")
    return questions


def question_fields(question: dict) -> dict:
    """실행 결과 행에 함께 남길 질문·정책 정보."""
    return {key: question.get(key) for key in (
        "question_id", "topic", "source_type", "policy_id", "policy_name",
        "expected_status", "expected_behavior", "question",
    )}


def recorded_env() -> dict:
    """실행 조건 기록용 설정값 (비밀값 제외)."""
    return {name: os.getenv(name) for name in RECORDED_ENV}


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "env_files": [str(path.relative_to(REPO_DIR)) for path in ENV_FILES],
    }
