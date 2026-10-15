# 정책표 (llm_eval_policies)

`POST /chat` 배치 평가에서 **질문마다 어떤 결과와 답변 동작을 기대하는지** 정의한 표다.
팀 정책 번호(상담 주제 11종, P01~P11)를 그대로 쓰며, 질문 파일(`llm_eval_questions_*`)의 `policy_id`는 그 질문 `topic`의 정책 번호다.

| 파일 | 형식 |
| --- | --- |
| `llm_eval_policies.jsonl` | 스크립트 입력용 (한 줄에 정책 하나) |
| `llm_eval_policies.csv` | 사람이 보는 용도. 엑셀에서 한글이 깨지지 않도록 BOM 포함 UTF-8 |

두 파일은 내용이 같다. **직접 수정하지 말고** `evaluation/scripts/build_questions.py`로 다시 만든다 (아래 "수정 방법").

## 열

| 열 | 의미 | 예 |
| --- | --- | --- |
| `policy_id` | 팀 정책 번호 | `P04` |
| `policy_name` | 상담 주제 (분류기 라벨과 같음) | `대출문의(만기/연장/조회등)` |
| `expected_status` | 기대 결과: `answered`(답변), `out_of_scope`(업무 외 차단) | `answered` |
| `expected_behavior` | 기대 동작. 평가 모델(judge)이 채점할 때 기준으로 받는 문장 | "승인이나 연장 가능 여부는 단정하지 않음 …" |

## 정책 목록 (팀 정책 번호)

| 정책 | 상담 주제 | 기대 결과 | 기대 동작 요약 |
| --- | --- | --- | --- |
| P01 | 거래내역/잔액조회 | answered | 공식 채널에서 확인하도록 안내, 실제 내역·잔액 제시 안 함 |
| P02 | 금융거래한도/비대면한도계좌 | answered | 한도 확인·해제 절차 안내, 해제 가능 여부 단정 안 함 |
| P03 | 기타(은행) | answered | 분실·비밀번호 오류·민원 등 기타 업무 안내 |
| P04 | 대출문의(만기/연장/조회등) | answered | 조회·만기·연장 절차 안내, 승인·연장 가능 여부 단정 안 함 |
| P05 | 만기,연장/해지,수신 | answered | 예적금 만기·해지 안내, 금액·이율은 상품 조건 확인 필요 |
| P06 | 부수거래금리감면 | answered | 우대·감면 조건 확인 방법 안내, 적용 단정 안 함 |
| P07 | 업무 외 | out_of_scope | 업무 외 고정 문구(①)만 반환 |
| P08 | 이자/연체금액 | answered | 확인 방법 안내, 금액 임의 계산 안 함 |
| P09 | 자동이체조회 | answered | 등록·변경·해지 절차 안내, 처리 완료 주장 안 함 |
| P10 | 중계요청/착오송금 | answered | 착오송금 반환 절차 안내, 반환 완료 주장 안 함 |
| P11 | 환전문의 | answered | 환전 신청·수령 안내, 환율·우대율·수수료 임의 제시 안 함 |

- **은행 업무 10종:** 기대 동작은 서비스 프롬프트(`app/services/policy.py`의 `TOPIC_GUIDELINES`) 유형별 지침에 공통 문장("일반 절차·확인 방법 안내. 개인별 승인·금리·한도·처리 결과 단정 금지, 민감정보 요구 금지")을 붙인 것이다. 서비스 프롬프트가 바뀌면 다시 생성해야 한다.
- **P07 업무 외:** 분류기가 1차로 차단하거나(`blocked_by_classifier`), 분류기를 통과하면 생성 LLM이 2차로 차단한다(`blocked_by_llm`). 둘 다 기대 결과 통과다.
- 생성 시 팀 정책 11종과 서비스 프롬프트의 주제 10종 + 업무 외가 일치하는지 검사한다. 다르면 오류로 멈춘다.

## 평가에서 쓰이는 곳

| 단계 | 사용하는 열 | 내용 |
| --- | --- | --- |
| 질문 로딩 (`common.load_questions`) | 전체 | 질문 파일과 `policy_id`로 이어 붙임. 정책표에 없는 `policy_id`면 오류 |
| 배치 실행 (`run_api_batch.py`) | `expected_status` | `POST /chat` 응답의 route가 기대 결과와 맞는지 기록 |
| 기대 route 통과 판정 | `expected_status` | `answered` → `answered`, `out_of_scope` → `blocked_by_classifier`·`blocked_by_llm` |
| 채점 (`judge_answers.py`) | `policy_name`, `expected_status`, `expected_behavior` | 평가 모델 프롬프트의 "적용 정책과 기대 동작"에 들어감. 서비스(생성 모델)에는 전달하지 않음 |
| 보고서 (`build_report.py`) | `policy_id`, `policy_name` | 정책별 사용 가능 건수·정확성 평균 표 |

## 세트별 질문 수

| 세트 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | 합계 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 개발용 `dev_24` (Q01~Q24) | 3 | 2 | 2 | 3 | 3 | 1 | 3 | 2 | 2 | 1 | 2 | 24 |
| 최종 비교용 `final_30` (Q25~Q54) | 3 | 3 | 2 | 4 | 2 | 2 | 3 | 3 | 2 | 3 | 3 | 30 |

질문은 모두 직접 작성했다. 거래 처리 요청(예: "지금 바로 이체해 주세요"), 민감정보가 포함된 질문, 지시 변경 시도 같은 질문도 별도 정책 없이 해당 상담 주제의 정책으로 분류한다. 이런 질문의 처리 기준은 서비스 출력 규칙("처리 완료를 주장하지 마세요", "민감한 정보를 요청하지 마세요" 등)이며, 평가 모델은 이 규칙을 직접 받아 채점한다.

## 수정 방법

1. 은행 업무 10종의 기대 동작을 바꾸려면 서비스 프롬프트 `app/services/policy.py`의 유형별 지침을 수정한다. 평가 쪽에서 따로 고치지 않는다.
2. 정책 번호·순서는 `evaluation/scripts/build_questions.py`의 `TEAM_POLICIES`(팀 정책 번호 순서)에 있다. 팀 정책이 바뀔 때만 수정한다.
3. 질문은 같은 파일의 `QUESTIONS`에 `(상담 주제, 질문)` 형식으로 적는다.
4. 저장소 루트에서 다시 생성한다. 정책표와 질문 파일(jsonl·csv)이 함께 다시 만들어진다. 학습 데이터 CSV를 주면 비슷한 문장이 있는지도 검사한다.

   ```bash
   uv run python evaluation/scripts/build_questions.py [--compare-csv train.csv validation.csv test.csv]
   ```

5. 결과 폴더의 `run_config.json`에 정책표 해시(`policies_sha256`)가 기록된다. 정책표가 바뀐 뒤의 실행 결과는 이전 결과와 직접 비교하지 않는다.
