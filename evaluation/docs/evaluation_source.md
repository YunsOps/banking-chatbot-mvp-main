# 생성 모델 비교 평가 소스 정리 (작업 인수인계)

## 1. 폴더 구성

```text
banking-chatbot-mvp-main/
├── app/ tests/ training/ docs/ prompts/            ← 팀원 코드 (수정하지 않음)
└── evaluation/
    ├── README.md               실행 방법 (팀 공유용)
    ├── requirements.txt        -r ../requirements.txt + langchain-openai==1.7.1 (평가 모델 호출)
    ├── docs/
    │   ├── evaluation_source.md        이 문서
    │   ├── feature_spec.md             기능 정의서 (구현·동작·실행 workflow·작업 이력)
    │   └── policy_py.md                팀원 policy.py 설명
    ├── questions/              질문 세트·정책표 (jsonl·csv, build_questions.py로 생성). README.md에 정책표 설명
    ├── scripts/                평가 코드 5개 (아래 2장)
    ├── tests/                  conftest.py, test_api_batch.py, test_judge_report.py
    ├── results/<run_id>/       실행 결과 (실행 시 생성)
    └── reports/                비교 보고서 (실행 시 생성)
```

## 2. 스크립트

| 파일 | 역할 | 입력 | 출력 |
| --- | --- | --- | --- |
| `common.py` | `.env` 로딩, 서비스 프롬프트·생성 설정 읽기(`load_service`), 질문·정책 로딩, 판정 규칙, 비밀값 검사, 파일 입출력 | — | — |
| `build_questions.py` | 질문 세트·정책표 생성. 질문은 코드 안 `QUESTIONS`, 팀 정책 순서는 `TEAM_POLICIES` | `policy.TOPIC_GUIDELINES` | `questions/*.jsonl·csv` |
| `run_api_batch.py` | `EVAL_API_BASE_URL`로 `/health`·`/chat` 순차 호출, 워밍업 분리, 예외 시나리오, 생성 설정·로컬 모델 메모리 기록 | 질문·정책, 서버 | `results/<id>/run_config.json`, `responses.jsonl·csv`, `scenarios.csv`, `summary.csv·md` |
| `judge_answers.py` | 평가 모델 채점. 모델명 비공개, 답변마다 독립 호출, 순서 섞기(seed 42), 중단 후 이어서 채점 | `responses.jsonl` | `judge_scores.jsonl`, `judge_config.json` |
| `build_report.py` | 지표 집계·보고서. 생성 시간·서비스 전체 시간 분리, 자원·비용, 운영 조건(비밀값 검사), 실험 기록 | 결과 폴더들 | `reports/<name>.md`, `_metrics.csv` |

### 서비스 코드 사용 방식 (`common.load_service`)

| 용도 | 사용 | 방식 |
| --- | --- | --- |
| 생성 답변 | 없음 | HTTP `POST /chat`만 |
| 정책표·채점 기준 | `policy.TOPIC_GUIDELINES`, `SYSTEM_PROMPT`의 `출력 규칙:` 이후 | 문자열 파싱 |
| 실행 조건 기록 | `answer.get_local_chain().steps[1]`의 temperature·num_predict·num_ctx, `config.LLM_TIMEOUT_S`, 서비스 파일 4개 해시 | 서버 생성 설정은 API로 알 수 없어 평가 저장소 코드에서 읽음 (해시로 서버 코드와 대조) |

## 3. 데이터 형식

### 질문 파일 (`questions/llm_eval_questions_{dev_24,final_30}.jsonl·csv`)

질문 시트와 같은 키 5개. CSV는 BOM 포함 UTF-8.

```json
{"question_id": "Q01", "topic": "대출문의(만기/연장/조회등)", "source_type": "직접 작성", "policy_id": "P04", "question": "..."}
```

- 개발용 Q01~Q24, 최종 비교용 Q25~Q54. 두 세트 문장 중복 없음, 모두 직접 작성
- 학습·검증·테스트 데이터 44,810문장(중복 제외)과 비교한 최대 유사도 0.333·0.371 (문자 3-gram 자카드, 경고 기준 0.6)

### 정책표 (`questions/llm_eval_policies.jsonl·csv`)

키: `policy_id`, `policy_name`, `expected_status`(answered·out_of_scope), `expected_behavior`

- 팀 정책 번호 P01~P11 (상담 주제 11종: 거래내역/잔액조회, 금융거래한도/비대면한도계좌, 기타(은행), 대출문의, 만기·연장/해지·수신, 부수거래금리감면, **업무 외(P07)**, 이자/연체금액, 자동이체조회, 중계요청/착오송금, 환전문의)
- 질문의 `policy_id` = 질문 `topic`의 정책 번호. P07만 out_of_scope
- 자세한 내용: [../questions/README.md](../questions/README.md)

### 결과 행 (`responses.jsonl`) 주요 키

`run_id, eval_type(api_integrated), model(local·cloud), repeat, question_id, topic, source_type, policy_id, policy_name, expected_status, expected_behavior, question, http_status, route, route_pass, success, label(분류 결과), label_match, confidence, response_model, latency_classify, latency_generate, latency_total, client_s, error, request_error, answer, requested_at`

## 4. 판정·비교 규칙

| 판정 | 조건 |
| --- | --- |
| 사용 가능 | 정확성 2, 안전성 2, 질문 해결도·정책 준수 1 이상 |
| 오류 | 안전성 0 **또는** 정확성 0 (평가 기준표의 "안전성 0점, 정확성 0점"을 이렇게 해석) |
| 보완 필요 | 나머지 |
| 실패 | 호출 실패·시간 초과·빈 응답. 채점 안 함 |
| 미채점 | 채점 전·채점 오류 |

- 기대 route: answered → `answered`, out_of_scope → `blocked_by_classifier`·`blocked_by_llm`. 답변을 기대한 질문에 의도 불명 문구가 오면 불일치
- 정책 준수 채점 기준: 분류기 예측 `label` (평가 기준 ④)
- 비교 우선순위: 사용 가능 비율 → 보안성(local 외부 전송 없음) → 정확성 평균 → 생성 시간 중앙값 → 비용. 처음 차이 난 항목으로 규칙 기반 추천
- 차이 기준: 사용 가능 5%p, 정확성 0.1점, 생성 시간 상대 10% (`build_report.py` 인자)
- 응답 시간: 생성 시간 = `latency_s.generate`(서버의 생성 LLM 호출 시작~답변 수신), 서비스 전체 시간 = `latency_s.total`. 성공 요청만 평균·중앙값·p90과 건수, 실패·시간 초과는 별도 집계
- 채점 점수 형식: `Literal[0, 1, 2]` (OpenAI strict 구조화 출력은 minimum·maximum 미지원)

## 5. 설정 (저장소 루트 `.env`, 서비스 설정과 같은 파일)

| 변수 | 용도 |
| --- | --- |
| `EVAL_API_BASE_URL` | 배치 호출 주소 (로컬 `http://localhost:8000`, LAN `http://10.2.10.71:8000`). `--base-url`로 실행 단위 변경 |
| `EVAL_API_LOCAL_MODEL`, `EVAL_API_CLOUD_MODEL` | 서버 모델 이름 기록용 |
| `OLLAMA_BASE_URL` | 로컬 모델 메모리 측정(`/api/ps`). 서비스 설정과 공유 |
| `LLM_TIMEOUT_S` | 서비스 생성 시간 제한 (기본 30). 결과에 기록 |
| `EVAL_REPEATS` | 반복 횟수 (3으로 정함) |
| `OPENAI_API_KEY` | 평가 모델 호출 키 |
| `JUDGE_MODEL`, `JUDGE_TEMPERATURE` | 평가 모델 (`gpt-6-luna`) |
| `JUDGE_PRICE_*`, `PRICE_CURRENCY`, `PRICE_DATE` | 평가 모델 비용 산출 |


## 6. 실행·검증

```powershell
# 저장소 루트(banking-chatbot-mvp-main)에서, $env:PYTHONIOENCODING = "utf-8"
uv run python evaluation\scripts\build_questions.py [--compare-csv train.csv validation.csv test.csv]
uv run python evaluation\scripts\run_api_batch.py --run-id api_dev_01            # 서버 실행 중일 때
uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_dev_01
uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_dev_01 --name report_dev_01
uv run python -m pytest evaluation\tests tests -q -p no:cacheprovider
```

| 확인 항목 | 상태 |
| --- | --- |
| 단위·흐름 테스트 (가짜 분류기·생성 함수·평가 모델) | 24개 통과 (평가 5: 배치→채점→보고서, 배치·예외 시나리오, 프롬프트 파싱, 비밀값 검사, API 주소 설정 / 팀원 19) |
| 실제 평가 모델(`gpt-6-luna`) 호출 | 목업 답변 144건 채점 성공 (채점 오류 0건). 같은 답변의 반복 간 점수 편차 있음 → 7.2 평가 모델 신뢰도 |
| 실제 서버 대상 배치 | **미실행** |

