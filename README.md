# KANT BANK AI 상담 MVP

![Project](https://img.shields.io/badge/status-MVP-0D766E) ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white) ![License](https://img.shields.io/badge/license-%EB%AF%B8%EC%A0%95-lightgrey)

**고객 질문의 상담 유형을 분류하고, 유형별 응답 지침에 따라 일반적인 은행 이용 절차를 안내하는 챗봇 MVP입니다.**

KANT BANK는 시연용 **가상 은행**이며 실제 금융기관과 관계가 없습니다. 계좌 조회, 이체, 대출 심사, 예약 및 민원 접수 같은 실제 금융 업무를 수행하지 않습니다.

## 주요 기능

- 은행 상담 10개 유형과 업무 외 질문을 KLUE-BERT로 분류합니다.
- 업무 외 질문은 생성 모델을 호출하지 않고 고정 안내를 반환합니다.
- 은행 업무 질문은 분류 결과와 유형별 지침을 LangChain 프롬프트에 전달합니다. LLM은 질문 원문을 다시 확인하고 일반적인 절차와 확인 방법을 안내합니다.
- Streamlit 화면에서 대표 질문 4개 또는 직접 입력한 질문을 같은 FastAPI `/chat` 엔드포인트로 전송합니다.
- 질문과 답변, 분류 유형을 표시하고 처리 중 입력 잠금, 연결 실패 안내, 화면 초기화를 지원합니다.
- 화면의 이전 대화는 표시용이며 API나 생성 모델에는 전달하지 않습니다. **질문 하나에 응답 하나**를 반환합니다.

```text
Streamlit 화면 → FastAPI → KLUE-BERT 분류
                              ├─ 업무 외 → 고정 안내 (LLM 호출 없음)
                              └─ 은행 업무 → LangChain + 생성 LLM → 답변
```

> **현재 구현 상태:** Streamlit 화면은 `model: "local"`로 고정되어 Ollama의 `qwen3.5:9b`를 호출합니다. FastAPI에는 `cloud` 생성 경로도 구현되어 있지만, 화면에서 모델을 전환하는 기능은 없습니다. 상용 모델의 평가상 선정과 현재 화면의 적용 상태를 구분해 주세요.

## 기술 스택

| 영역 | 기술 | 역할 |
| --- | --- | --- |
| 화면 | Streamlit | 질문 입력, 대화 표시, 오류 안내 |
| API | FastAPI, Pydantic | 요청 검증, 분류·생성 흐름, 응답 스키마 |
| 분류 | `klue/bert-base`, Transformers, PyTorch | 11개 상담 유형 분류 |
| 분류 베이스라인 | TF-IDF, Kiwi, 로지스틱 회귀 | 분류 성능 비교 |
| 생성 | LangChain, Ollama `qwen3.5:9b` | 로컬 답변 생성 |
| 선택적 상용 생성 | LangChain `ChatOpenAI` | API의 `cloud` 경로 |
| 평가 | Python 평가 스크립트, pytest | 배치 수집, 자동 채점, 보고서 및 평가 도구 테스트 |

## 데이터와 평가 결과

### 분류 모델

AI Hub **금융분야 고객상담 데이터**의 은행 분야를 사용했습니다. 정제한 은행 상담 44,815건에 업무 외 질문 5,000건을 더해 총 **49,815건, 11개 클래스**로 구성했습니다. 분류기에는 질문만 입력하며 답변이나 지시문을 입력하지 않습니다.

| 데이터 | 학습 | 검증 | 테스트 | 합계 |
| --- | ---: | ---: | ---: | ---: |
| 은행 상담 10개 주제 | 35,852 | 4,481 | 4,482 | 44,815 |
| 업무 외 | 4,000 | 500 | 500 | 5,000 |
| **전체** | **39,852** | **4,981** | **4,982** | **49,815** |

동일한 11개 클래스의 테스트 분할에서 TF-IDF·로지스틱 회귀와 KLUE-BERT를 비교했습니다. 검증 Macro F1을 우선 지표로 삼아 KLUE-BERT를 선정했습니다.

| 지표 | TF-IDF·로지스틱 회귀 | KLUE-BERT |
| --- | ---: | ---: |
| 검증 Macro F1 | 0.7110 | **0.7374** |
| 테스트 Macro F1 | 0.7208 | **0.7354** |
| 테스트 Accuracy | 80.27% | **81.75%** |
| 테스트 오분류 (4,982건 중) | 983건 | **909건** |
| 업무 외 테스트 F1 | 0.9950 | **1.0000** |

검증 Macro F1은 0.0264 높았고 테스트 오분류는 74건 적었습니다. 다만 테스트의 `기타(은행)`은 18건이며 두 모델 모두 F1이 0입니다. 원천 상담 단위의 데이터 분리 여부도 추가 검증이 필요합니다.

### 생성 모델 서비스 통합 평가

최종 질문 30개를 모델마다 3회씩 FastAPI `/chat`으로 요청했습니다. 정확성, 질문 해결도, 정책 준수, 안전성을 각각 0~2점으로 채점했습니다. **정확성 2점 + 안전성 2점 + 나머지 두 항목 각각 1점 이상**을 사용 가능 답변으로 판정했습니다.

| 지표 | 로컬 `qwen3.5:9b` | 상용 모델 |
| --- | ---: | ---: |
| 사용 가능 / 전체 요청 | 42 / 90 (46.7%) | **78 / 90 (86.7%)** |
| 정확성 평균 | 1.289 / 2 | **1.953 / 2** |
| 질문 해결도 평균 | 1.200 / 2 | **1.977 / 2** |
| 정책 준수 평균 | 1.367 / 2 | **1.860 / 2** |
| 안전성 평균 | 1.878 / 2 | **2.000 / 2** |
| 채점된 응답 / 실패 | 90 / 0 | 86 / 4 |

사용 가능 비율은 실패까지 포함한 **모델별 전체 90건**이 분모이고, 평균 점수는 채점된 응답만의 평균입니다. 상용 모델은 평가상 선정되었지만 빈 응답·`llm_error` 4건이 기록됐습니다. 발표 자료에는 출력 예산 256토큰을 추론이 소진한 것이 원인으로 보고되어 있습니다. 이 비교는 분류기 차단과 고정 응답을 포함한 **서비스 통합 평가**이므로 생성 모델 자체만의 성능 차이로 해석하면 안 됩니다. 상용 생성 경로는 질문과 프롬프트를 외부 API로 전송합니다.

> 수치는 **2026-10-16 발표 자료의 기존 실험 기록**입니다. 이 README를 작성하며 학습·평가를 새로 실행한 결과가 아닙니다. 평가 기준과 산출물 형식은 [평가 문서](docs/evaluation.md)를 참고하세요. 해당 문서의 과거 구현 상태 설명은 현재 코드와 다를 수 있습니다.

## 시작하기

### 사전 요구사항

- Python 3.12, [`uv`](https://docs.astral.sh/uv/), 로컬 생성 경로를 위한 [Ollama](https://ollama.com/)
- 학습 완료된 BERT 모델과 토크나이저: `models/klue_bert_base/`
- 로컬 생성 모델: `qwen3.5:9b`
- 선택 사항: NVIDIA GPU. 아래 CUDA 12.8 설치 예시는 Windows PowerShell 기준입니다.

`data/`, `models/`, `logs/`, `.venv/` 및 `.env`는 Git으로 배포되지 않습니다. 모델과 데이터는 별도로 준비해야 합니다. 다른 OS의 설치·실행 절차는 이 저장소에서 검증되지 않았습니다.

### 설치 (Windows PowerShell)

```powershell
git clone https://github.com/YunsOps/banking-chatbot-mvp-main.git
cd banking-chatbot-mvp-main
uv venv --python 3.12

# NVIDIA CUDA 12.8 환경에서 GPU를 사용할 때만 먼저 실행
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv\Scripts\python.exe -r requirements.txt
ollama pull qwen3.5:9b
```

CPU 환경에서는 위 PyTorch CUDA 설치 명령을 생략합니다. BERT 추론은 CPU에서도 가능하지만, 기존 학습·평가 스크립트의 `bf16=True` 설정은 실행 장치에 맞춰 확인해야 합니다. 이미 서버 환경이 준비됐다면 가상환경을 새로 만들지 말고 의존성 설치부터 진행하세요.

### 환경변수 설정

```powershell
# 기존 설정은 덮어쓰지 않음
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`.env.example`을 참고해 필요한 값만 설정하세요. 환경 변수가 `.env`보다 우선하며, 모델 설정 변경 후에는 API 서버를 재시작해야 합니다.

| 변수 | 용도 |
| --- | --- |
| `BERT_MODEL_DIR` | 저장한 BERT 모델·토크나이저 경로 (기본 `models/klue_bert_base`) |
| `CLASSIFIER_TYPE` | `bert` (기본값) 또는 `baseline` |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | 로컬 Ollama 주소와 모델명 |
| `CHAT_API_URL` | Streamlit 화면이 호출할 API 주소 |
| `MAX_QUESTION_CHARS`, `LLM_TIMEOUT_S` | 입력 최대 길이와 생성 제한 시간 (기본 500자, 30초) |
| `DEFAULT_LLM` | API 요청에서 `model` 생략 시 기본값. 화면은 명시적으로 `local` 전송 |
| `CLOUD_LLM_MODEL`, `CLOUD_API_KEY_ENV` | 선택적 상용 생성 모델명 및 API 키 환경 변수 이름 (기본 `OPENAI_API_KEY`) |
| `OPENAI_API_KEY`, `JUDGE_MODEL` | 상용 생성 또는 외부 채점에 필요한 키와 평가 모델 설정 |

API 키 발급 방법은 제공된 자료에 없어 [여기에 작성]. 키는 `.env` 등 로컬 환경에서 관리하고 Git에 올리지 마세요. 상용 모델 호출과 외부 채점은 별도 비용과 데이터 외부 전송이 발생할 수 있습니다.

### 로컬 실행

Ollama를 실행하고 `models/klue_bert_base/`에 학습한 모델과 토크나이저를 준비한 다음, 프로젝트 루트에서 터미널 두 개를 사용합니다.

**터미널 1: FastAPI**

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**터미널 2: Streamlit**

```powershell
$env:CHAT_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run ui/chatbot.py --server.address 127.0.0.1 --server.port 8501
```

- 화면: <http://127.0.0.1:8501>
- API 상태: <http://127.0.0.1:8000/health>
- API 문서: <http://127.0.0.1:8000/docs>

`/health`의 `llm.local`은 Ollama 서버 응답 여부이고 `llm.cloud`는 상용 API 키 설정 여부입니다. 실제 답변 생성 성공 여부는 `/chat` 요청으로 확인하세요. 같은 LAN에서 시연할 때에는 서버 주소와 방화벽 허용 범위를 해당 네트워크에 맞춰 별도로 설정해야 합니다. 인터넷 공개나 터널을 전제로 하지 않습니다.

## 사용법

화면에서 대표 질문을 선택하거나 직접 질문을 입력할 수 있습니다. API를 직접 호출하려면 다음 요청을 사용하세요.

```powershell
$body = @{ question = "환전 신청은 일반적으로 어떻게 하나요?"; model = "local" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

`POST /chat` 응답에는 `answer`, `route`, `label`, `confidence`, `model`, `latency_s`, `error`가 포함됩니다. `route`는 `answered`, `blocked_by_classifier`, `blocked_by_llm`, `unclear`, `error` 중 하나입니다. 처리 실패가 HTTP 200과 `route=error`로 반환될 수도 있으므로 **HTTP 상태만이 아니라 `route`도 확인**하세요. `confidence`는 분류 확률이지 답변의 정확도가 아닙니다.

실제 계좌번호, 비밀번호, 주민등록번호, OTP 번호 등 민감정보를 입력하지 마세요.

## 프로젝트 구조

```text
app/                 FastAPI, 분류기, 생성 모델 연결, 응답 정책
ui/chatbot.py        Streamlit 채팅 화면
training/            TF-IDF·KLUE-BERT 학습과 분류 평가
evaluation/          질문 세트, 배치 수집·채점·보고서, 평가 도구 테스트
docs/evaluation.md   평가 기준과 기록
prompts/             업무 외·의도 불명·오류 고정 문구
.streamlit/          화면 테마 설정
.env.example         환경변수 예시
requirements.txt     Python 의존성
```

## 테스트 방법

평가 도구 테스트는 외부 모델 API를 호출하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests -q -p no:cacheprovider
```

실제 서비스의 연결은 `/health`, `/docs`, `/chat` 및 Streamlit 화면에서 별도로 확인하세요. 재학습이나 최종 질문 세트의 재평가에는 데이터·모델 준비, 실행 환경, 외부 채점 비용이 필요할 수 있습니다. 상세 실행 옵션은 [평가 문서](docs/evaluation.md)를 참고하세요.

## 한계와 후속 과제

- 화면은 현재 로컬 모델만 호출합니다. 평가에서 선정된 상용 모델을 화면에 반영하려면 UI 전환과 실제 서비스 검증이 필요합니다.
- `기타(은행)`처럼 표본이 적고 F1이 낮은 상담 유형이 있습니다. 원천 상담 단위 데이터 분리 여부도 추가 확인이 필요합니다.
- 일부 생성 답변에서 지침상 금지한 앱 메뉴 안내가 관찰됐습니다. 정책 지침이 있다고 해서 모든 답변의 준수가 보장되지는 않습니다.
- 금융 특화 분류 모델, 다른 로컬 LLM, 지침 조정, 실제 정책 문서를 활용한 RAG 등은 후속 실험 대상이며 성능 향상은 아직 확인되지 않았습니다.

## 기여 방법

이슈에서 변경 목적과 재현 방법을 먼저 공유하고, 변경 사항은 별도 브랜치의 Pull Request로 제안해 주세요. 관련 테스트 결과와, 분류·생성 정책 또는 평가 조건을 바꿨다면 그 영향도 함께 적어 주세요. 구체적인 기여 정책과 리뷰 절차는 [여기에 작성].

## 라이선스

라이선스는 아직 저장소에 명시되어 있지 않습니다. 사용·재배포 조건은 [여기에 작성]. 라이선스가 정해지기 전에는 사용 권한을 추정하지 마세요.

## 작성자 및 문의

- 프로젝트 참여자: **임윤성, 김찬영, 오주헌, 노영우**
- 문의처: [여기에 작성]
