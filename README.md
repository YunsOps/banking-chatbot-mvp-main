# 은행 상담 챗봇 MVP

은행 업무 질문을 **BERT로 분류하고, 유형별 지침을 적용한 Qwen으로 일반적인 이용 절차를 안내**하는 서비스입니다.
Streamlit 화면 → FastAPI → BERT → LangChain·Ollama 순서로 처리합니다.

- 은행 상담 10개 유형과 업무 외 질문을 분류합니다.
- 업무 외 질문은 고정 안내를 반환하며, 분류기를 통과한 질문도 LLM이 원문을 다시 확인합니다.
- 질문 하나당 응답 하나를 생성합니다. 화면의 이전 대화는 모델에 전달하지 않습니다.
- 실제 계좌 조회·거래·예약·민원 접수는 수행하지 않습니다.
- 현재 서비스 생성 모델은 Ollama의 `qwen3.5:9b`입니다. 상용 생성 모델 연결은 미구현입니다.
- 같은 LAN 안에서 시연합니다. 인터넷 공개나 터널은 사용하지 않습니다.

## 프로젝트 구성

| 경로 | 역할 |
| --- | --- |
| `app/` | FastAPI, 분류기 로딩, 답변 생성, 정책, 요청 기록 |
| `ui/chatbot.py` | 질문 입력, 상담 유형·답변 표시, 처리 중·실패 안내 |
| `prompts/fixed_messages.json` | 업무 외·의도 불명·처리 실패 고정 문구 |
| `training/` | TF-IDF·BERT 학습 및 분류 성능 평가 |
| `evaluation/` | 생성 답변 배치 수집·채점·보고서와 평가 도구 테스트 |
| [docs/evaluation.md](docs/evaluation.md) | 상세 평가 기준, 데이터 형식, 기존 실험 기록·한계 |
| `data/`, `models/`, `logs/` | 로컬 데이터·모델·기록. Git 제외 |

설치 목록은 `requirements.txt`, 설정 예시는 `.env.example` 하나로 관리합니다.
학습·평가용 패키지도 함께 설치하되, 실행 명령은 용도별로 구분합니다.

## 1. 설치와 모델 준비

아래 명령은 **프로젝트 루트의 PowerShell**에서 실행합니다. Python 3.12, uv, Ollama가 필요합니다.
서버를 이미 준비했다면 가상환경을 다시 만들지 말고 의존성 설치부터 진행하세요.

```powershell
# 새 환경에서만 실행
uv venv --python 3.12

# NVIDIA GPU 서버: CUDA 12.8용 PyTorch를 먼저 설치
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

GPU를 사용하지 않으면 첫 번째 `uv pip install` 명령을 생략합니다. BERT 추론은 CPU에서도 가능하지만,
기존 BERT 학습·평가 스크립트는 `bf16=True`이므로 실행 장치에 맞는 확인이 필요합니다.
GPU 설치 기준은 [PyTorch 2.11 공식 안내](https://pytorch.org/get-started/previous-versions/#v2110)를 따릅니다.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
ollama pull qwen3.5:9b
```

Ollama를 실행해 두고, 학습한 BERT 모델과 토크나이저를 `models/klue_bert_base/`에 준비합니다.
모델·데이터·`.venv`는 Git으로 전달되지 않으므로 PC마다 별도로 준비합니다.
가상환경은 원본 Python 설치를 참조합니다. `No Python at ...` 오류가 나면 참조한 Python 경로부터 확인해야 합니다.

## 2. 설정

```powershell
# 기존 .env가 있으면 보존
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

| 변수 | 용도 |
| --- | --- |
| `BERT_MODEL_DIR` | 저장한 BERT·토크나이저 폴더 |
| `CLASSIFIER_TYPE` | 기본 `bert`; 베이스라인 실행 시 `baseline` |
| `BASELINE_MODEL_PATH` | 기본 `models/tfidf_logistic_regression.joblib` |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama 주소·다운로드한 생성 모델 이름 |
| `CHAT_API_URL` | Streamlit이 호출할 API 주소 |
| `MAX_QUESTION_CHARS`, `LLM_TIMEOUT_S` | 질문 길이·생성 시간 제한 (기본 500자·30초) |
| `EVAL_API_BASE_URL` | 평가 스크립트가 호출할 API 주소 |
| `OPENAI_API_KEY`, `JUDGE_MODEL` | 외부 평가 모델로 채점할 때만 필요한 설정 |

환경 변수가 `.env`보다 우선합니다. 모델 설정 변경 후 API 서버를 재시작합니다.
기존 `.env.server.example`의 서비스 설정은 그대로 유효합니다. 파일 이름만 `.env.example`로 통합했습니다.
기존 베이스라인이 `model/`에 있으면 `BASELINE_MODEL_PATH=model/tfidf_logistic_regression.joblib`로 지정할 수 있습니다.
이 변수를 아예 설정하지 않은 기존 환경은 새 위치에 모델이 없을 때 기존 `model/`을 자동 확인합니다.

## 3. 서비스 실행

터미널 1 — API (Ollama가 실행 중이어야 함):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

터미널 2 — 화면 (같은 PC의 API 사용):

```powershell
$env:CHAT_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run ui/chatbot.py --server.address 127.0.0.1 --server.port 8501
```

- 화면: <http://127.0.0.1:8501>
- API 상태: <http://127.0.0.1:8000/health>
- 요청·응답 필드 및 호출 예시: <http://127.0.0.1:8000/docs> (`app/schemas.py` 기준)
- `/health`의 `llm.local`은 Ollama 응답 여부입니다. 모델이 실제로 답변을 생성하는지도 별도로 확인합니다.

### 같은 LAN에서 시연

서버의 실제 IPv4가 `10.2.10.71`인 경우, 화면 실행 명령의 주소를 다음처럼 바꿉니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/chatbot.py --server.address 10.2.10.71 --server.port 8501
```

팀원 기기에서는 <http://10.2.10.71:8501>에 접속합니다. 서버의 API와 화면 프로세스는 계속 실행되어 있어야 합니다.
화면이 API를 대신 호출하므로 팀원 기기에는 Python이나 모델 설치가 필요 없습니다.
8501 방화벽 규칙이 없다면 서버의 관리자 PowerShell에서 같은 LAN만 허용합니다.

```powershell
New-NetFirewallRule -DisplayName "BankingChatbot UI 8501 LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8501 -RemoteAddress 10.2.10.0/24 -Profile Any
```

다른 PC에서 API를 직접 테스트하려면 기존 TCP 8000 LAN 허용도 필요합니다. IP·서브넷이 바뀌면 주소와 규칙을 함께 수정합니다.

### API와 시연 확인

`POST /chat` 요청 예:

```json
{"question": "환전 신청은 일반적으로 어떻게 하나요?", "model": "local"}
```

응답은 `answer`, `route`, `label`, `confidence`, `model`, `latency_s`, `error`를 포함합니다.
`route`는 `answered`, `blocked_by_classifier`, `blocked_by_llm`, `unclear`, `error` 중 하나입니다.
처리 실패도 HTTP 200과 `route=error`로 반환될 수 있습니다. 입력 검증 오류는 HTTP 422입니다.
`confidence`는 분류 확률이며 답변 정확도가 아닙니다.

시연에서는 은행 질문, 업무 외 질문, 모호한 질문, 처리 중 입력 잠금, 연결 실패 안내, 화면 초기화를 확인합니다.
알려진 한계: 일부 생성 답변이 정책에서 금지한 앱 메뉴 이름을 안내합니다. 기능 동작 성공과 답변 품질을 구분합니다.

## 4. 분류 모델 학습·평가

`data/dataset_v2/`에 `train.csv`, `validation.csv`, `test.csv`를 준비합니다.
질문 컬럼은 `question`, 정답 컬럼은 `qa_topic`입니다. 학습은 서비스 실행과 별도로 수행합니다.

```powershell
.\.venv\Scripts\python.exe -m training.train_tfidf_logistic_regression
.\.venv\Scripts\python.exe -m training.train_klue_bert_base
.\.venv\Scripts\python.exe -m training.evaluate_klue_bert_base
```

- TF-IDF 결과: `models/tfidf_logistic_regression.joblib`
- BERT 체크포인트: `models/bert_run01/`. 재학습할 때는 기존 결과를 보존하도록 학습 코드의 `output_dir`을 새 경로로 지정합니다.
- BERT 평가 코드의 `checkpoint_dir`은 현재 `checkpoint-7473`으로 지정되어 있습니다. 검증 성능으로 선정한 체크포인트에 맞춰 변경한 뒤 실행합니다.
- BERT 평가 스크립트는 테스트 평가 후 `models/klue_bert_base/`에 모델과 토크나이저를 저장합니다.

## 5. 생성 답변 평가

설정·설치 방법은 위 공통 절차를 사용합니다. `.env`의 평가 구역을 채우고 API 서버를 실행해 둡니다.
`EVAL_API_LOCAL_MODEL` 등은 결과 기록용이며 서버 모델을 변경하지 않습니다.
`EVAL_REPEATS`는 예시 설정에서 3회, 변수를 생략한 코드 기본값은 1회입니다.
평가 모델 호출에는 비용이 발생하며 질문·답변·정책이 외부 평가 API로 전달됩니다.
`OPENAI_API_KEY`는 채점에만 필요하고 파일이나 Git에 공유하지 않습니다.

```powershell
$env:PYTHONIOENCODING = "utf-8"
# 개발용 질문 수집 → 채점 → 보고서 (실행마다 새 run-id 사용)
.\.venv\Scripts\python.exe evaluation/scripts/run_api_batch.py --run-id api_dev_01
.\.venv\Scripts\python.exe evaluation/scripts/judge_answers.py --run-dir evaluation/results/api_dev_01
.\.venv\Scripts\python.exe evaluation/scripts/build_report.py --run-dir evaluation/results/api_dev_01 --name report_dev_01

# 최종 비교용 질문은 프롬프트·조건 확정 후 사용
.\.venv\Scripts\python.exe evaluation/scripts/run_api_batch.py --questions evaluation/questions/llm_eval_questions_final_30.jsonl --run-id api_final_01

# 평가 도구 자체 검증: 실제 모델 호출 없음
.\.venv\Scripts\python.exe -m pytest evaluation/tests -q -p no:cacheprovider
```

최종 세트도 수집 후 해당 결과 폴더에 채점·보고서 명령을 적용합니다.
같은 `run-id`로 배치를 다시 실행하면 응답이 덮어써집니다. 채점은 완료된 건을 건너뛰며 `--rejudge`를 주면 전체 재채점합니다.
보고서는 기존 파일만 읽으므로 다시 생성해도 모델 호출 비용이 없습니다.
준비되지 않은 `cloud` 생성 모델은 평가 대상에서 제외됩니다.

질문·정책을 수정한 경우에만 아래 명령으로 CSV·JSONL을 함께 재생성합니다.

```powershell
.\.venv\Scripts\python.exe evaluation/scripts/build_questions.py
```

원본 응답은 `evaluation/results/`에 로컬 보관합니다. 공유할 보고서는 `evaluation/reports/`에서 검토 후 Git에 추가합니다.
평가 항목·판정 규칙·옵션·기록 해석은 [상세 평가 문서](docs/evaluation.md)를 참고하세요.
