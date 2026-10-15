# 생성 모델 비교 평가 기능 정의서

- 대상: `banking-chatbot-mvp-main/evaluation/`
- 목적: 미리 만든 질문을 FastAPI 서비스(`.env`의 `EVAL_API_BASE_URL`)의 `POST /chat`에 배치로 보내고, 받은 응답을 평가 모델(LLM-as-a-Judge, `.env`의 `JUDGE_MODEL` = `gpt-6-luna`)로 채점해 로컬 LLM·상용 LLM 비교 보고서를 만든다.
- 작성: 2026-10-14
- 관련 문서: [../README.md](../README.md)(실행 방법 요약), [../questions/README.md](../questions/README.md)(정책표), [policy_py.md](policy_py.md)(서비스 프롬프트), [evaluation_source.md](evaluation_source.md)(인수인계·개선 사항)

## 1. 개요

### 1.1 전체 흐름

```text
questions/ (질문 세트·정책표)  ← build_questions.py
        │
run_api_batch.py ── GET /health, POST /chat {"question", "model"} ──▶ FastAPI 서버 (분류기 → 생성 LLM)
        │  응답(answer, route, label, latency_s, error) + 측정값
        ▼
results/<run_id>/responses.jsonl
        │
judge_answers.py ── 평가 프롬프트 ──▶ 평가 모델 (OPENAI_API_KEY)
        │  항목별 점수·근거
        ▼
results/<run_id>/judge_scores.jsonl
        │
build_report.py ──▶ reports/<name>.md, <name>_metrics.csv
```

### 1.2 요구사항과 구현

| 요구사항 | 구현 |
| --- | --- |
| 평가 모델: 모델명을 숨기고 같은 평가 프롬프트·독립 세션으로 채점 | `judge_answers.py`: local·cloud 이름을 전달하지 않음, 답변마다 독립 호출, 같은 프롬프트(해시 기록), 채점 순서 섞기 |
| 응답 시간: 생성 모델 호출 시작~전체 답변 수신. 워밍업 분리. 서비스 전체 시간은 별도 | 응답의 `latency_s.generate`(생성)과 `latency_s.total`(서비스 전체)을 따로 기록·집계. 워밍업은 집계 제외하고 따로 기록 |
| 측정 조건: 같은 질문·반복 횟수로 순차 호출, 워밍업·출력 제한·시간 초과·재시도 사전 고정 | 모델별 순차 호출. `run_config.json`에 반복·워밍업·시간 제한·재시도(없음)·생성 설정(출력 제한 256토큰 등) 기록 |
| 성공 요청 응답 시간 평균·중앙값·건수, 실패·시간 초과 건수와 실패율 별도 | 보고서 모델별 비교표에 생성·전체 시간 평균/중앙값/p90(건수), 실패 건수·실패율·시간 초과 건수·실패 원인 |
| 자원·비용: 로컬 최대 RAM·VRAM, 실행 환경, API 비용, 평가 모델 비용 분리, 실측·추정·미측정 표시 | 보고서 "자원·비용" 표. 로컬 모델 메모리(Ollama `/api/ps`, 접속 가능 시 실측), 평가 모델 비용(사용량 실측 × 단가), 나머지는 미측정·미산출 표시 |
| 운영 조건: 외부 전송 범위, 공유 가능한 데이터만 전달, API 키·개인정보 미포함 확인 | 보고서 "운영 조건" 표. 평가 모델로 보낸 데이터 명시, 결과 파일의 API 키 포함 여부 자동 검사 |
| 실험 기록: 질문 ID·입력·정책·프롬프트·응답·설정·시간·성공 여부·오류·채점 결과 연결 | (모델, 반복, 질문 ID)로 응답·채점 연결, 프롬프트는 서비스 파일 해시로 기록. 보고서 "실험 기록" 표 |
| 실패 사례·대표 성공 사례, 공통 사례 비교, 전체 건수·판정 불가 비율 | 보고서 "대표 사례"(질문 기준 중복 제거), "공통 사례 비교"와 판정 불가 건수·비율 |

### 1.3 설계 원칙

| 원칙 | 구현 |
| --- | --- |
| 팀원 서비스 코드는 수정하지 않음 | 생성 답변은 HTTP로만 받음. 서비스 코드는 `common.load_service()`에서 채점 기준·설정 기록용으로 읽기만 함 |
| 실패도 결과에 포함 | 호출 실패·시간 초과·빈 응답을 "실패"로 기록하고 비율로 보고 |
| 재현성 | 질문·정책 파일 해시, 서비스 파일 해시, 평가 프롬프트 해시, 설정, 실행 환경을 결과에 기록 |
| 결과를 본 뒤 기준을 바꾸지 않음 | 비교 우선순위와 차이 기준을 사전에 정하고 규칙으로 추천 계산 |

## 2. 기능 정의

### 2.1 기능 목록

| 기능 ID | 기능 | 파일 | 주요 함수 |
| --- | --- | --- | --- |
| F-01 | 설정·공통 처리 | `scripts/common.py` | `load_questions`, `verdict`, `service_generation_settings`, `find_secrets` |
| F-02 | 질문 세트·정책표 생성 | `scripts/build_questions.py` | `build_policies`, `build_questions`, `check_similarity` |
| F-03 | 배치 실행 (POST /chat) | `scripts/run_api_batch.py` | `post_chat`, `to_row`, `run_scenarios`, `ollama_memory`, `summarize` |
| F-04 | 평가 모델 채점 | `scripts/judge_answers.py` | `create_judge`, `judge_inputs` |
| F-05 | 지표 집계·보고서 | `scripts/build_report.py` | `model_metrics`, `compare`, `resource_section`, `operation_section` |
| F-06 | 자동 테스트 | `tests/` | `test_배치_채점_보고서`, `test_배치_실행_결과` |

### F-01 설정·공통 처리 (`common.py`)

| 항목 | 내용 |
| --- | --- |
| 설정 로딩 | 저장소 루트 `.env` 하나만 읽음 (서비스와 같은 파일) |
| 서비스 코드 읽기 | `load_service()`: 채점 기준과 설정 기록에만 사용. 생성 답변을 만들지 않음 |
| 채점 기준 | `load_guidelines()`: `policy.TOPIC_GUIDELINES`의 주제별 지침. `output_rules()`: `policy.SYSTEM_PROMPT`의 "출력 규칙:" 이후 |
| 생성 설정 기록 | `service_generation_settings()`: `answer.get_local_chain()`의 temperature·최대 출력 토큰·컨텍스트·시간 제한, 재시도 없음 |
| 질문 로딩 | `load_questions()`: 질문 파일과 정책표를 `policy_id`로 이어 붙임. 필수 키 누락, 없는 `policy_id`, 중복 `question_id`면 오류 |
| 기대 route | `answered` → `answered`, `out_of_scope` → `blocked_by_classifier`·`blocked_by_llm` |
| 판정 | `verdict(success, scores)`: 아래 4.1 판정 규칙 |
| 비밀값 검사 | `find_secrets(paths)`: `.env`의 API 키 값이 결과 파일에 들어 있는지 검사 |
| 기록 | `service_files()`: 서비스 파일 4개 SHA-256. `recorded_env()`: 설정값 (API 키 제외). `environment()`: Python·OS·CPU |

### F-02 질문 세트·정책표 생성 (`build_questions.py`)

| 항목 | 내용 |
| --- | --- |
| 입력 | 코드 안 `QUESTIONS`(직접 작성한 `(상담 주제, 질문)`), `TEAM_POLICIES`(팀 정책 번호 순서), 서비스 `TOPIC_GUIDELINES` |
| 처리 | 팀 정책 P01~P11(상담 주제 11종). 은행 업무 10종의 기대 동작 = 서비스 유형별 지침 + 공통 문장, P07 업무 외 = 고정 문구 ①. 팀 정책과 서비스 프롬프트 주제가 다르면 오류. 질문에 `question_id`·`policy_id` 부여, 개발용·최종 비교용 문장 중복 검사 |
| 선택 기능 | `--compare-csv`: 학습 데이터와 문자 3-gram 자카드 유사도 비교. 0.6 이상이면 경고 |
| 출력 | `questions/llm_eval_policies.*`, `llm_eval_questions_dev_24.*`(Q01~Q24), `llm_eval_questions_final_30.*`(Q25~Q54). jsonl + csv(BOM 포함 UTF-8) |

### F-03 배치 실행 (`run_api_batch.py`)

| 항목 | 내용 |
| --- | --- |
| 입력 | 질문·정책 파일, `--base-url`(기본 `.env`의 `EVAL_API_BASE_URL`), `--models`, `--repeats`(기본 `EVAL_REPEATS`), `--warmup`(기본 1), `--timeout`(클라이언트, 기본 120초) |
| 서버 상태 확인 | `GET /health`의 `llm.local`·`llm.cloud`로 준비된 모델만 실행. 준비 안 된 모델은 제외 사유 기록 |
| 호출 | 모델별로 워밍업 후 질문 순서대로 `POST /chat {"question", "model"}` 순차 호출 (동시 요청 없음, 재시도 없음) |
| 기록 | HTTP 상태, 답변, route, 분류 라벨·신뢰도, 라벨 일치, 생성 시간(`latency_s.generate`), 분류 시간, 서비스 전체 시간(`latency_s.total`), 클라이언트 측 시간, 오류 |
| 성공 판정 | HTTP 200, route가 error가 아님, 답변이 비어 있지 않음. 나머지는 실패(호출 실패·시간 초과·빈 응답) |
| 측정 조건 기록 | 워밍업 결과(집계 제외), 시간 제한, 재시도, 서비스 생성 설정, 로컬 모델 메모리(`.env`의 `OLLAMA_BASE_URL` `/api/ps`, 워밍업 후·실행 후) |
| 예외 시나리오 | 입력 오류 5종(질문 없음, 빈 질문, 공백 질문, 500자 초과, 허용되지 않은 model) → 422 기대. 준비 안 된 LLM 요청 → `200 route=error error=llm_*` 기대 |
| 출력 | `run_config.json`, `responses.jsonl·csv`, `scenarios.csv`, `summary.csv·md`(채점 전 빠른 확인용) |

### F-04 평가 모델 채점 (`judge_answers.py`)

| 항목 | 내용 |
| --- | --- |
| 평가 모델 | `.env`의 `JUDGE_MODEL`(현재 `gpt-6-luna`)을 `OPENAI_API_KEY`로 호출. `JUDGE_TEMPERATURE` 선택 |
| 평가 프롬프트 입력 | 고객 질문, 적용 정책(이름·기대 결과·기대 동작), 분류 결과(`label`)와 그 정책 지침, 서비스 출력 규칙 원문, 챗봇 답변 |
| 채점 항목 | 정확성·질문 해결도·정책 준수·안전성, 각 0~2점 + 항목별 근거 |
| 구조화 출력 | `with_structured_output(JudgeScore, include_raw=True)`. 점수 형식은 `Literal[0, 1, 2]` |
| 공정성 | 모델 이름 미전달, 답변마다 독립 호출, 채점 순서 섞기(`--seed`, 기본 42) |
| 실패 건 | 생성 실패(`success=False`)는 호출하지 않고 "실패" 판정 |
| 이어서 채점 | 다시 실행하면 완료된 건은 건너뛰고 채점 오류 건만 다시 채점. `--rejudge`면 전체 재채점 |
| 출력 | `judge_scores.jsonl`(점수·근거·판정·평가 모델 토큰), `judge_config.json`(평가 모델, 프롬프트 해시, 채점·오류 건수) |

### F-05 지표 집계·보고서 (`build_report.py`)

| 보고서 절 | 내용 |
| --- | --- |
| 실행 조건 | 서버, 질문·정책 해시, 반복·워밍업, 시간 제한, 재시도, 생성 설정, 분류기, 평가 모델, 서비스 파일 버전 |
| 모델별 비교 | 사용 가능 비율, 판정 분포, 항목별 평균, 안전성 0점, 기대 route 통과율, 라벨 일치율, 실패 건수·실패율·시간 초과·원인, **생성 시간**·서비스 전체 시간(평균/중앙값/p90, 건수), 외부 전송 |
| 공통 사례 비교 | 두 모델 모두 판정 가능한 (반복, 질문)만 비교. 전체 건수 중 판정 불가 건수·비율 |
| 정책별 비교 | 팀 정책별 사용 가능 건수와 정확성 평균 |
| 우선순위 비교·추천 | 아래 4.2 규칙. 안전성 0점 경고 |
| 대표 사례 | 모델별 오류 → 보완 필요 → 실패 → 사용 가능 1건. 같은 질문은 한 번만 표시하고 같은 판정 횟수 표시 |
| 자원·비용 | 로컬 모델 메모리, 최대 RAM·VRAM, 실행 환경, 상용 API 생성 비용, 평가 모델 비용(서비스 비용과 분리), 장비·전력 비용. 각 항목에 실측·미측정·미산출 표시 |
| 운영 조건 | 모델별 외부 전송, 로컬 LLM 운영 조건, 평가 모델로 보낸 데이터, 결과 파일 비밀값 검사 결과 |
| 실험 기록 | 결과 파일별 내용과 연결 키 |
| 한계 | 질문 대표성, 평가 모델 편향, 측정 환경, 생성 설정 출처, 비용 산출 범위 |

### F-06 자동 테스트 (`tests/`)

| 테스트 | 확인 내용 |
| --- | --- |
| `test_배치_채점_보고서` | 서비스 FastAPI 앱 + 가짜 분류기·생성 함수 → 배치(local·cloud) → 가짜 평가 모델 채점 → 이어서 채점 → 보고서 절·지표 |
| `test_배치_실행_결과` | 배치 실행, 예외 시나리오, 의도 불명 응답의 기대 route 불일치, 요약 |
| `test_서비스_프롬프트에서_지침과_출력규칙을_읽음` | 서비스 프롬프트에서 주제 10종 지침과 출력 규칙 파싱 |
| `test_결과_파일의_비밀값_검사` | API 키가 들어간 파일 검출 |
| `test_호출_주소는_env의_EVAL_API_BASE_URL` | API 주소 설정 |

실제 서버·Ollama·평가 모델은 호출하지 않는다.

## 3. 데이터 정의

| 파일 | 주요 키 |
| --- | --- |
| 질문 `questions/llm_eval_questions_*.jsonl` | `question_id`, `topic`(정답 라벨), `source_type`, `policy_id`, `question` |
| 정책표 `questions/llm_eval_policies.jsonl` | `policy_id`, `policy_name`, `expected_status`, `expected_behavior` (설명: [../questions/README.md](../questions/README.md)) |
| 응답 `results/<run_id>/responses.jsonl` | `run_id, eval_type, model, repeat`, 질문·정책 키, `http_status, route, route_pass, success, label, label_match, confidence, latency_classify, latency_generate, latency_total, client_s, error, request_error, answer` |
| 채점 `results/<run_id>/judge_scores.jsonl` | `model, repeat, question_id`, 항목별 점수·근거, `verdict`, `judge_error`, `judge_input_tokens`, `judge_output_tokens` |

결과 한 건의 식별자는 (모델, 반복 회차, 질문 ID)다.

## 4. 판정·비교 규칙

### 4.1 판정

| 판정 | 조건 |
| --- | --- |
| 사용 가능 | 정확성 2, 안전성 2, 질문 해결도·정책 준수 1 이상 |
| 오류 | 안전성 0 또는 정확성 0 |
| 보완 필요 | 위 어디에도 해당하지 않음 |
| 실패 | 호출 실패·시간 초과·빈 응답 (채점하지 않음) |
| 미채점 | 채점 전이거나 채점 호출 실패 |

### 4.2 비교 우선순위와 차이 기준

| 순서 | 항목 | 차이 기준 (`build_report.py` 인자) |
| --- | --- | --- |
| 1 | 사용 가능 답변 비율 | 5%p 이하면 차이 없음 (`--usable-tie 0.05`) |
| 2 | 보안성 (외부 전송) | local은 외부 전송 없음 → 항상 local 우세 |
| 3 | 정확성 평균 | 0.1점 이하면 차이 없음 (`--accuracy-tie 0.1`) |
| 4 | 생성 시간 중앙값 | 상대 10% 이하면 차이 없음 (`--latency-tie 0.1`) |
| 5 | 비용 | 상용 API 사용량이 /chat 응답에 없어 현재 판단 불가 |

추천은 규칙으로 계산한 결과이며, 최종 선정은 대표 사례와 한계를 검토한 뒤 팀이 확정한다.

## 5. 실행 workflow

모든 명령은 저장소 루트(`banking-chatbot-mvp-main`)의 PowerShell에서 실행한다.

### 5.1 준비 (처음 한 번)

| 순서 | 단계 | 명령 / 작업 | 확인 |
| --- | --- | --- | --- |
| 1 | 가상환경 생성 | `uv venv --python 3.12` | 저장소 루트에 `.venv` 생성 |
| 2 | 의존성 설치 | `uv pip install --python .venv\Scripts\python.exe -r evaluation\requirements.txt` | 서비스 의존성 + `langchain-openai` |
| 3 | 설정 입력 | 저장소 루트 `.env`의 평가 구역에 값 입력 (아래 5.4) | `EVAL_API_BASE_URL`, `OPENAI_API_KEY`, `JUDGE_MODEL` |
| 4 | 콘솔 한글 설정 | `$env:PYTHONIOENCODING = "utf-8"` (터미널마다) | 한글 출력이 깨지지 않음 |
| 5 | 코드 동작 확인 | `uv run python -m pytest evaluation\tests tests -q -p no:cacheprovider` | 모두 통과 |

### 5.2 평가 실행 (개발용 세트)

| 순서 | 단계 | 명령 | 입력 | 출력 | 확인 사항 |
| --- | --- | --- | --- | --- | --- |
| 1 | 질문·정책 생성 (질문을 바꿨을 때만) | `uv run python evaluation\scripts\build_questions.py` | 코드 안 질문, 서비스 프롬프트 | `evaluation\questions\*.jsonl·csv` | 정책별 질문 수 출력. 바꾸지 않았다면 생략 |
| 2 | 서버 확인 | 브라우저나 `curl`로 `EVAL_API_BASE_URL/health` | — | — | `status`, `classifier.loaded`, `llm.local` |
| 3 | 배치 실행 | `uv run python evaluation\scripts\run_api_batch.py --run-id api_dev_01` | 개발용 질문 24개 × 반복 3회 × 준비된 모델 | `evaluation\results\api_dev_01\` | 콘솔의 성공·기대 route 통과 건수, `summary.md`의 실패·불일치, 예외 시나리오 통과 수. `[제외]`가 나오면 서버 모델 상태 확인 |
| 4 | 채점 | `uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_dev_01` | `responses.jsonl` | `judge_scores.jsonl`, `judge_config.json` | 채점 오류가 있으면 같은 명령을 다시 실행 (오류 건만 재채점) |
| 5 | 보고서 생성 | `uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_dev_01 --name report_dev_01` | 결과 폴더 | `evaluation\reports\report_dev_01.md`, `_metrics.csv` | 모델별 비교, 추천, 안전성 경고, 대표 사례, 자원·비용, 비밀값 검사 "통과" |
| 6 | 프롬프트 보완 | 대표 사례와 감점 근거를 보고 서비스 프롬프트 수정 (팀원과 협의) | 보고서 | — | 수정했으면 서버 재배포 후 3부터 다시 실행 (`run_id`를 바꿔 이전 결과 보존) |

### 5.3 최종 비교 (프롬프트 확정 후 한 번)

| 순서 | 단계 | 명령 | 확인 사항 |
| --- | --- | --- | --- |
| 1 | 비교 기준 확정 | 차이 기준(5%p·0.1점·10%)과 반복 횟수를 팀에서 확정 | 결과를 본 뒤 바꾸지 않음 |
| 2 | 배치 실행 | `uv run python evaluation\scripts\run_api_batch.py --questions evaluation\questions\llm_eval_questions_final_30.jsonl --run-id api_final_01` | 최종 비교용 질문 30개 |
| 3 | 채점 | `uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_final_01` | 채점 오류 0건까지 반복 |
| 4 | 최종 보고서 | `uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_final_01 --name report_final` | 추천 결과 + 대표 사례·한계 검토 후 팀이 최종 선정 |

### 5.4 설정 (`.env`, 저장소 루트)

| 변수 | 용도 | 사용 단계 |
| --- | --- | --- |
| `EVAL_API_BASE_URL` | 호출할 서비스 주소 | 배치 |
| `EVAL_API_LOCAL_MODEL`, `EVAL_API_CLOUD_MODEL` | 서버 모델 이름 (결과 기록용) | 배치 |
| `EVAL_REPEATS` | 반복 횟수 (3) | 배치 |
| `OLLAMA_BASE_URL` | 로컬 모델 메모리 측정(`/api/ps`). 접속 불가면 미측정 | 배치 |
| `OPENAI_API_KEY` | 평가 모델 호출 키 | 채점 |
| `JUDGE_MODEL`, `JUDGE_TEMPERATURE` | 평가 모델 | 채점 |
| `JUDGE_PRICE_*`, `PRICE_CURRENCY`, `PRICE_DATE` | 평가 모델 100만 토큰당 단가, 통화, 단가 기준일 | 보고서 |

### 5.5 자주 쓰는 옵션

| 스크립트 | 옵션 | 용도 |
| --- | --- | --- |
| `run_api_batch.py` | `--models local` | 한 모델만 실행 |
| `run_api_batch.py` | `--repeats 1` | 빠른 확인용 1회 실행 |
| `run_api_batch.py` | `--base-url http://10.2.10.71:8000` | 이번 실행만 다른 서버 사용 |
| `judge_answers.py` | `--rejudge` | 기존 채점을 지우고 전체 재채점 |
| `build_report.py` | `--usable-tie`, `--accuracy-tie`, `--latency-tie`, `--examples` | 차이 기준, 대표 사례 수 |

