# 평가 실행 방법

실행하고자 하는 pc나 서버에서 FastAPI 서버(`POST /chat`)에 질문을 배치로 보내고 → 평가 모델(`gpt-6-luna`)로 채점 → 보고서를 만든다.
모든 명령은 저장소 루트(`banking-chatbot-mvp-main`)의 PowerShell에서 실행한다.

## 1. 준비 (처음 한 번)

저장소 루트 `.env`에 아래 값이 있어야 한다. 전체 변수와 확인 방법은 [env_setup.md](env_setup.md).

| 변수 | 예 |
| --- | --- |
| `EVAL_API_BASE_URL` | `http://localhost:8000` (LAN 서버면 `http://10.2.10.71:8000`) |
| `OPENAI_API_KEY` | 평가 모델 호출 키 |
| `JUDGE_MODEL` | `gpt-6-luna` |
| `EVAL_REPEATS` | `3` |

## 2. 실행 (순서대로)

```powershell
$env:PYTHONIOENCODING = "utf-8"

# 0) 서버 확인: status가 ok, llm.local이 true인지
curl.exe http://localhost:8000/health

# 1) 배치: 질문을 POST /chat으로 전송 → evaluation\results\api_dev_01\
uv run python evaluation\scripts\run_api_batch.py --run-id api_dev_01

# 2) 채점: 평가 모델 호출 (비용 발생)
uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_dev_01

# 3) 보고서 → evaluation\reports\report_dev_01.md
uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_dev_01 --name report_dev_01
```

| 단계 | 결과 파일 | 확인 |
| --- | --- | --- |
| 1) 배치 | `results\<run_id>\responses.jsonl·csv`, `run_config.json`, `scenarios.csv`, `summary.md` | `summary.md`의 실패·route 불일치, 예외 시나리오 통과 수 |
| 2) 채점 | 같은 폴더의 `judge_scores.jsonl`, `judge_config.json` | 채점 오류 0건 (오류가 있으면 같은 명령 다시 실행) |
| 3) 보고서 | `reports\<name>.md`, `<name>_metrics.csv` | 모델별 비교, 추천, 대표 사례, 자원·비용, 비밀값 검사 "통과" |

## 3. 최종 비교 (프롬프트 확정 후 한 번)

```powershell
uv run python evaluation\scripts\run_api_batch.py --questions evaluation\questions\llm_eval_questions_final_30.jsonl --run-id api_final_01
uv run python evaluation\scripts\judge_answers.py --run-dir evaluation\results\api_final_01
uv run python evaluation\scripts\build_report.py --run-dir evaluation\results\api_final_01 --name report_final
```

## 4. 자주 쓰는 옵션

| 옵션 | 용도 |
| --- | --- |
| `run_api_batch.py --repeats 1` | 빠른 확인용 1회 실행 |
| `run_api_batch.py --models local` | 한 모델만 실행 |
| `$env:EVAL_API_BASE_URL = "http://10.2.10.71:8000"` | 이 터미널에서만 다른 서버 호출 (`.env`보다 우선). `$env:EVAL_API_BASE_URL = $null`로 되돌림 |
| `judge_answers.py --rejudge` | 기존 채점을 지우고 전체 재채점 |
| `build_report.py --run-dir A B` | 여러 실행을 한 보고서로 |

## 5. 주의

- 실행마다 새 `--run-id`를 쓴다. 같은 `run_id`로 배치를 다시 실행하면 그 폴더의 응답을 덮어쓴다.
- 채점이 중간에 끊기면 같은 명령을 다시 실행한다. 채점이 끝난 건은 건너뛴다.
- 보고서는 파일만 읽으므로 비용 없이 몇 번이든 다시 만들 수 있다.
- `run_config.json이 없습니다` 오류: 1) 배치를 먼저 실행해야 한다.
- 서비스에 상용 LLM이 연결되기 전까지 cloud는 `/health`에서 준비되지 않음으로 나와 제외된다.
