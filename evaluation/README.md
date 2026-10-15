# 생성 모델 비교 평가

미리 만든 질문을 FastAPI 서비스(`.env`의 `EVAL_API_BASE_URL`)의 `POST /chat`에 배치로 보내고, 받은 응답을 평가 모델(`.env`의 `JUDGE_MODEL`, 현재 `gpt-6-luna`)로 채점해 로컬 LLM·상용 LLM 비교 보고서를 만든다.
서비스 코드(`app/`)는 수정하지 않는다. 명령은 저장소 루트에서 실행한다.

```text
questions/ ──▶ run_api_batch.py ──POST /chat──▶ FastAPI 서버
                     │ (응답·측정값)
                     ▼
              judge_answers.py ──▶ 평가 모델 (LLM-as-a-Judge)
                     │ (점수·근거)
                     ▼
              build_report.py ──▶ reports/<name>.md
```

## 구성

```text
evaluation/
├── run.md                  실행 방법 요약 (실제 서버 대상)
├── env_setup.md            .env 설정 방법 (추가할 변수, 확인 방법)
├── requirements.txt        서비스 의존성 + langchain-openai (평가 모델 호출)
├── docs/
│   ├── evaluation_source.md          소스 구성·결정 사항·개선 필요 사항 (작업 인수인계)
│   ├── feature_spec.md               기능 정의서 (구현·동작·실행 workflow·작업 이력)
│   └── policy_py.md                  서비스 프롬프트(app/services/policy.py) 설명
├── questions/              질문 세트·정책표 (jsonl·csv)
│   ├── README.md                     정책표 설명 (열, 정책 목록, 사용처, 수정 방법)
│   ├── llm_eval_questions_dev_24     개발용 Q01~Q24 (프롬프트 수정 단계)
│   ├── llm_eval_questions_final_30   최종 비교용 Q25~Q54 (프롬프트 확정 후 한 번)
│   └── llm_eval_policies             팀 정책 P01~P11의 기대 결과·기대 동작
├── scripts/
│   ├── common.py                     설정·공통 함수·판정 규칙
│   ├── build_questions.py            질문 세트·정책표 생성
│   ├── run_api_batch.py              질문 배치 → POST /chat → 응답·측정값 기록
│   ├── judge_answers.py              평가 모델 채점
│   └── build_report.py               지표 집계·보고서
├── tests/                  가짜 분류기·생성 함수·평가 모델로 전체 흐름 확인
├── results/<run_id>/       실행 결과 (실행 시 생성)
└── reports/                비교 보고서 (실행 시 생성)
```

질문 파일 키는 질문 시트와 같다: `question_id`(질문 ID), `topic`(상담 주제), `source_type`(출처 구분), `policy_id`(정책 ID), `question`(질문).
정책 ID: 팀 정책 번호 P01~P11 (상담 주제 11종). 질문의 `policy_id`는 그 질문 `topic`의 정책 번호. 자세한 내용은 [questions/README.md](questions/README.md).

## 서비스 코드와의 관계

| 용도 | 서비스 코드 사용 | 방식 |
| --- | --- | --- |
| 생성 답변 | 없음 | `.env`의 `EVAL_API_BASE_URL`로 `GET /health`, `POST /chat` HTTP 호출 |
| 채점 기준 | `policy.TOPIC_GUIDELINES`, `policy.SYSTEM_PROMPT`의 출력 규칙 | 지침 원문을 읽어 평가 모델에만 전달 |
| 실행 조건 기록 | `answer.get_local_chain()`의 생성 설정, 서비스 파일 4개 | 출력 제한·temperature 등과 SHA-256을 결과에 기록 (서버 코드와 대조용) |

서비스 코드는 `scripts/common.py`의 `load_service()` 한 곳에서만 읽는다.

## 준비

```powershell
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r evaluation\requirements.txt
$env:PYTHONIOENCODING = "utf-8"
```

설정은 저장소 루트 `.env` 하나만 사용한다 (서비스 설정 아래 "평가" 구역). 추가할 변수와 확인 방법은 [env_setup.md](env_setup.md).

| 설정 | 용도 |
| --- | --- |
| `EVAL_API_BASE_URL` | 호출할 서비스 주소. 로컬 `http://localhost:8000`, LAN 서버 `http://10.2.10.71:8000` |
| `EVAL_API_LOCAL_MODEL`, `EVAL_API_CLOUD_MODEL` | 서버에 설정된 모델 이름 (결과 기록용) |
| `OLLAMA_BASE_URL` | 로컬 모델 메모리 측정(`/api/ps`)용. 서버 Ollama에 접속할 수 없으면 "미측정" |
| `OPENAI_API_KEY` | 평가 모델 호출 키 |
| `JUDGE_MODEL`, `JUDGE_TEMPERATURE` | 평가 모델 ID (현재 `gpt-6-luna`), temperature(선택) |
| `EVAL_REPEATS` | 질문 목록 반복 횟수 (기본 3) |
| `JUDGE_PRICE_*`, `PRICE_CURRENCY`, `PRICE_DATE` | 평가 모델 비용 단가·기준일 (선택) |

## 실행 (저장소 루트에서)

```powershell
# 서버(EVAL_API_BASE_URL)가 떠 있어야 함
uv run python evaluation\scripts\run_api_batch.py --run-id api_dev_01
uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_dev_01
uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_dev_01 --name report_dev_01

# 최종 비교: 프롬프트 확정 후 최종 세트로 한 번
uv run python evaluation\scripts\run_api_batch.py --questions evaluation\questions\llm_eval_questions_final_30.jsonl --run-id api_final_01
```

- `run_api_batch.py --base-url <주소>`로 그 실행에서만 주소를 바꿀 수 있다.
- 서비스에 상용 LLM이 연결되기 전까지 cloud는 `/health`에서 준비되지 않음으로 나와 제외된다.
- 결과를 본 뒤 기준(질문, 반복 횟수, 차이 기준)을 바꾸지 않는다.
- 단계별 상세 workflow는 [docs/feature_spec.md](docs/feature_spec.md) 5장.

## 측정과 보고 내용

| 항목 | 내용 |
| --- | --- |
| 채점 | 평가 모델이 정확성·질문 해결도·정책 준수·안전성을 0~2점 채점. 모델 이름 비공개, 답변마다 독립 호출, 같은 평가 프롬프트 |
| 응답 시간 | 생성 시간(`latency_s.generate`, 생성 LLM 호출 시작~답변 수신)과 서비스 전체 시간(`latency_s.total`)을 따로 집계. 성공 요청의 평균·중앙값·p90과 건수 |
| 실패 | 호출 실패·시간 초과·빈 응답 건수와 실패율을 제외하지 않고 별도 집계 |
| 측정 조건 | 같은 질문·반복 횟수로 순차 호출. 워밍업(집계 제외)·출력 제한·시간 제한·재시도 조건을 결과에 기록 |
| 자원·비용 | 로컬 모델 메모리, 평가 모델 비용(서비스 비용과 분리), 실측·미측정 구분 |
| 운영 조건 | 외부 전송 범위, 평가 모델로 보낸 데이터, 결과 파일의 API 키 포함 여부 검사 |
| 실험 기록 | 질문 ID·질문·정책·응답·설정·시간·판정·채점을 (모델, 반복, 질문 ID)로 연결 |

## 판정과 비교 기준

- 판정: 사용 가능(정확성 2·안전성 2·질문 해결도/정책 준수 1 이상), 오류(안전성 0 또는 정확성 0), 보완 필요, 실패(호출 실패·시간 초과·빈 응답), 미채점
- 정책 준수 채점 기준: 분류기가 실제로 고른 정책(응답의 `label`)
- 비교 우선순위: 사용 가능 비율 → 보안성(외부 전송) → 정확성 평균 → 생성 시간 → 비용
- 차이 기준(사전 설정, `build_report.py` 인자): 사용 가능 비율 5%p, 정확성 0.1점, 생성 시간 상대 10%
- 추천은 규칙 계산 결과이며, 최종 선정은 대표 사례와 한계를 검토한 뒤 팀이 확정한다.

## 테스트

```powershell
uv run python -m pytest evaluation\tests -q -p no:cacheprovider
```

## 주의

- `results/`의 응답 파일에는 질문 원문이 들어 있다. API 키는 어떤 파일에도 기록하지 않으며, 보고서 생성 시 결과 파일에 API 키가 있는지 검사한다.
- 채점이 중간에 끊기면 같은 명령을 다시 실행한다. 채점이 끝난 건은 건너뛴다.
