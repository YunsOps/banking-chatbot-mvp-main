# 은행 상담 챗봇 API 명세

LAN 테스트 서버의 Base URL은 `http://10.2.10.71:8000`이다.
요청 본문은 JSON이며 모든 시간 값은 초 단위다. 질문 1개에 응답 1개를 반환한다.

## GET /health

파라미터 없이 HTTP 200으로 상태를 반환한다.

| 필드 | 타입 | 의미 |
| --- | --- | --- |
| status | string | `ok`: 분류기 로드 완료 및 생성 모델 하나 이상 준비, 그 외 `degraded` |
| classifier.loaded | boolean | 분류기 로드 여부 |
| classifier.model | string 또는 null | 분류기 이름과 실행 장치 |
| classifier.labels | integer 또는 null | 로드된 클래스 수, 정상 BERT는 11 |
| llm.local | boolean | Ollama 서버가 응답하는지 여부. 특정 모델의 다운로드·생성 성공까지 보장하지 않음 |
| llm.cloud | boolean | 상용 모델 연결 준비 여부. 현재 구현은 false |

## POST /classify

```json
{"question": "환전 신청은 어떻게 하나요?"}
```

`question`은 필수 문자열이다. 앞뒤 공백을 제거한 뒤 1~500자를 허용한다.
최대 길이는 서버의 `MAX_QUESTION_CHARS` 설정에 따른다.

응답 예시의 숫자는 형식 설명용이다.

```json
{
  "label": "환전문의",
  "confidence": 0.99,
  "latency_s": {"classify": 0.04}
}
```

- `label`: 분류기가 예측한 상담 유형.
- `confidence`: 예측 클래스의 최대 확률 점수. BERT는 softmax 점수이며 전체 모델 정확도를 뜻하지 않는다.
- `latency_s.classify`: 해당 요청의 분류 시간.
- 입력 검증 실패는 HTTP 422, 분류기를 로드하지 못한 경우 HTTP 503.

## POST /chat

```json
{"question": "환전 신청은 어떻게 하나요?", "model": "local"}
```

| 요청 필드 | 타입 | 필수 | 허용값·기본값 |
| --- | --- | --- | --- |
| question | string | 필수 | 공백 제거 후 1~500자 |
| model | string | 선택 | `local` 또는 `cloud`. 기본값 `local` |

`model`은 생성 모델의 실행 방식을 선택한다. 실제 로컬 모델 이름은 서버의 `OLLAMA_MODEL` 설정을 사용한다.
현재 `cloud` 호출은 아직 연결하지 않았으므로 `llm_not_configured` 오류로 처리한다.

```json
{
  "answer": "해당 은행의 공식 채널에서 환전 신청 방법을 확인하세요.",
  "route": "answered",
  "label": "환전문의",
  "confidence": 0.99,
  "model": "local",
  "latency_s": {"classify": 0.04, "generate": 3.0, "total": 3.04},
  "error": null
}
```

| 응답 필드 | 타입 | 의미 |
| --- | --- | --- |
| answer | string | 생성 답변 또는 고정 안내 문구 |
| route | string | 아래 처리 결과 구분 |
| label | string 또는 null | BERT의 예측 라벨. LLM이 원문에 다른 지침을 적용해도 라벨은 BERT 결과 |
| confidence | number 또는 null | 분류 점수 |
| model | string 또는 null | 생성 단계에서 선택된 `local`/`cloud`. 1차 차단이면 null |
| latency_s.classify | number 또는 null | 분류 시간. 미실행이면 null |
| latency_s.generate | number 또는 null | 생성 단계 시간. 미실행이면 null |
| latency_s.total | number | 전체 처리 시간 |
| error | string 또는 null | 오류 원인 코드. 오류가 없으면 null |

| route | 의미 |
| --- | --- |
| answered | 일반 은행 안내 답변 생성 |
| blocked_by_classifier | BERT가 업무 외로 판단하여 LLM 호출 없이 차단 |
| blocked_by_llm | BERT 통과 후 LLM이 업무 외로 판단 |
| unclear | LLM이 원문 의도를 식별하지 못함 |
| error | 분류·생성 단계 처리 실패 |

| error | 의미 |
| --- | --- |
| classifier_unavailable | 분류기 미로드 또는 분류 실패 |
| llm_timeout | 생성 단계 시간 초과. 기본 제한 30초 |
| llm_not_configured | 선택한 생성 모델 미구성 |
| llm_error | 생성 호출 실패 또는 빈 답변 |

입력 검증 실패는 HTTP 422다. 현재 `/chat`의 차단·의도 불명·처리 실패는 HTTP 200의 JSON으로 반환한다.
클라이언트는 HTTP 상태만 보지 말고 `route`와 `error`를 함께 확인한다.
요청 기록은 서버의 `logs/requests.jsonl`에 저장되며 Git에서는 제외한다.
