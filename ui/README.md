# 은행 상담 챗봇 화면

기존 FastAPI의 `/chat`을 호출하는 Streamlit MVP입니다. 모델은 API 서버에서만 실행합니다.
질문·답변 기록은 브라우저 세션 동안 화면에 표시하며, API에는 새 질문 하나만 전달합니다.

## 로컬 실행 (프로젝트 루트, PowerShell)

```powershell
uv pip install -r requirements-ui.txt
$env:CHAT_API_URL = "http://10.2.10.71:8000"
python -m streamlit run ui/chatbot.py --server.address 127.0.0.1
```

API 서버가 실행 중이어야 합니다. 브라우저에서 `http://localhost:8501`을 엽니다.
API가 같은 PC에 있으면 `CHAT_API_URL`을 `http://127.0.0.1:8000`으로 설정합니다.
이 값은 프로젝트 루트의 `.env`에도 지정할 수 있습니다. 환경 변수가 우선합니다.

## 서버에서 실행하고 같은 LAN의 팀원에게 공유

서버에서 최신 코드와 UI 의존성을 준비하고, FastAPI와 Ollama를 실행한 상태에서:

```powershell
$env:CHAT_API_URL = "http://127.0.0.1:8000"
python -m streamlit run ui/chatbot.py --server.address 10.2.10.71 --server.port 8501
```

팀원은 `http://10.2.10.71:8501`에 접속합니다. 팀원 PC에는 Python이나 모델 설치가 필요 없습니다.
서버의 Windows 방화벽에서 TCP 8501 인바운드를 동일 LAN(`10.2.10.0/24`)에 허용해야 합니다.
공유기 포트 포워딩, 터널, 인터넷 공개는 사용하지 않습니다. 서버 IP가 바뀌면 주소도 수정합니다.

## 시연 확인

- 은행 질문: 분류된 상담 유형과 답변 표시
- 업무 외·의도 불명 질문: API가 반환한 고정 안내 표시
- 요청 중: 처리 상태 표시, 입력과 초기화 비활성화
- API 연결 실패·시간 초과: 실패 안내 후 다시 입력 가능
- 화면 초기화: 현재 세션의 질문·답변만 제거

분류 결과는 BERT의 예측이며 답변의 정확성을 보장하지 않습니다.
