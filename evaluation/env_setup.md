# 평가용 .env 설정 방법

평가 코드는 **저장소 루트의 `.env` 하나**만 읽는다 (서비스와 같은 파일). 서비스 설정(`.env.server.example` 내용)이 이미 있다면, 아래 "평가" 구역만 파일 끝에 추가한다.

## 1. 추가할 내용

```env
# ==== 평가 (evaluation/) ====
# 필수
EVAL_API_BASE_URL=http://localhost:8000
OPENAI_API_KEY=
JUDGE_MODEL=gpt-6-luna

# 선택
EVAL_REPEATS=3
EVAL_API_LOCAL_MODEL=qwen3.5:9b
EVAL_API_CLOUD_MODEL=
JUDGE_TEMPERATURE=
JUDGE_PRICE_INPUT_PER_1M=
JUDGE_PRICE_OUTPUT_PER_1M=
PRICE_CURRENCY=USD
PRICE_DATE=
```

## 2. 변수 설명

| 변수 | 필수 | 용도 | 비워 두면 |
| --- | --- | --- | --- |
| `EVAL_API_BASE_URL` | 필수 | 배치가 호출할 FastAPI 서버 주소. LAN 서버는 `http://10.2.10.71:8000` | `http://localhost:8000` |
| `OPENAI_API_KEY` | 필수 | 평가 모델 호출 키 (본인 키) | 채점 중단 |
| `JUDGE_MODEL` | 필수 | 평가 모델 ID | 채점 중단 |
| `EVAL_REPEATS` | 선택 | 질문 목록 반복 횟수 | 1회 |
| `EVAL_API_LOCAL_MODEL`, `EVAL_API_CLOUD_MODEL` | 선택 | 서버에 설정된 모델 이름. API 응답에 없어 결과·보고서 기록용으로 적음 | 보고서에 "local"·"-" 표시 |
| `JUDGE_TEMPERATURE` | 선택 | 평가 모델 temperature | 모델 기본값 |
| `JUDGE_PRICE_INPUT_PER_1M`, `JUDGE_PRICE_OUTPUT_PER_1M` | 선택 | 평가 모델 100만 토큰당 단가 | 비용 "단가 미입력" |
| `PRICE_CURRENCY`, `PRICE_DATE` | 선택 | 통화, 단가 확인 기준일 (예: `2026-10-14`) | 통화 빈칸, 기준일 "미입력" |

서비스 설정의 `OLLAMA_BASE_URL`도 함께 읽는다 (로컬 모델 메모리 측정, 서버 Ollama에 접속할 수 없으면 "미측정"). 이미 서비스 설정에 있으므로 추가하지 않는다.

## 3. 설정 확인

저장소 루트에서 실행한다. API 키 값은 출력하지 않고 설정 여부만 보여 준다.

```powershell
uv run python -c "import sys; sys.path.insert(0, 'evaluation/scripts'); import common, os; [print(k, '=', ('설정됨' if os.getenv(k) else '비어 있음') if 'KEY' in k else os.getenv(k)) for k in ['EVAL_API_BASE_URL', 'OPENAI_API_KEY', 'JUDGE_MODEL', 'EVAL_REPEATS']]"
```

기대 출력 예:

```text
EVAL_API_BASE_URL = http://localhost:8000
OPENAI_API_KEY = 설정됨
JUDGE_MODEL = gpt-6-luna
EVAL_REPEATS = 3
```

## 4. 이 터미널에서만 값 바꾸기

터미널에서 설정한 환경 변수가 `.env`보다 우선한다. 파일을 고치지 않고 다른 서버를 호출할 때 쓴다.

```powershell
$env:EVAL_API_BASE_URL = "http://10.2.10.71:8000"   # 이 터미널에서만 적용
$env:EVAL_API_BASE_URL = $null                      # .env 값으로 되돌림
```

## 5. 주의

- `.env`는 Git에서 제외된다. **`OPENAI_API_KEY`는 공유하지 않는다** (예시 파일, 메신저, 문서에도 쓰지 않음). 각자 본인 키를 넣는다.
- 키의 OpenAI 프로젝트가 `gpt-6-luna`에 접근할 수 있어야 한다. 권한이 없으면 채점에서 403 `model_not_found` 오류가 난다.
- 값 뒤 주석은 공백 뒤에 `#`로 쓴다 (예: `EVAL_API_BASE_URL=http://localhost:8000  # 로컬`).
- 같은 변수를 두 번 쓰면 아래쪽 값이 적용된다. 서비스 구역과 평가 구역에 같은 변수를 중복으로 넣지 않는다.
- 평가 코드는 `OPENAI_MODEL`, `EVAL_CLOUD_TEMPERATURE`, `CLOUD_PRICE_*`를 쓰지 않는다. 넣을 필요 없다.
