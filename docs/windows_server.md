# Windows LAN 서버 실행

서버 주소는 `10.2.10.71`, 팀원 접속 대역은 `10.2.10.0/24`이다.
Windows에서 BERT와 Ollama를 실행하며 WSL은 사용하지 않는다.
아래 명령은 별도 표시가 없으면 서버의 프로젝트 루트 PowerShell에서 실행한다.

## 1. 코드 가져오기

Git, uv, Ollama for Windows와 NVIDIA 드라이버를 설치한다.
처음 준비할 때는 다음 명령을 사용한다.

```powershell
git clone https://github.com/YunsOps/banking-chatbot-mvp-main.git
cd .\banking-chatbot-mvp-main
```

이미 프로젝트가 있으면 해당 폴더에서 다음 명령을 사용한다.

```powershell
git pull --ff-only origin main
```

서버에서는 코드를 직접 변경하기보다 이 PC에서 수정·push한 코드를 pull한다.
업데이트할 때는 API를 Ctrl+C로 종료하고 pull한 뒤 다시 실행한다.

## 2. Python과 의존성 설치

```powershell
uv python install 3.12
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .\.venv\Scripts\python.exe -r requirements-server.txt
```

GPU 사용 여부를 확인한다.

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA: True`가 기대값이다. RAM 32GB와 GPU VRAM은 별개의 용량이며 실제 VRAM은 `nvidia-smi`에서 확인한다.

## 3. 최종 BERT 모델 복사

`models/`는 Git에서 제외되므로 git pull로 모델 가중치가 내려오지 않는다.
이 PC의 `models/klue_bert_base` 폴더를 서버의 같은 경로에 한 번 복사한다.
필요 파일은 `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`이다.

제공된 `klue_bert_base-server.zip`을 서버 프로젝트 루트에 복사했다면 다음과 같이 풀 수 있다.
아직 해당 모델 폴더가 없는 초기 설치에서 실행한다.

```powershell
Expand-Archive -LiteralPath .\klue_bert_base-server.zip -DestinationPath .
```

## 4. Ollama와 환경 설정

Ollama for Windows 앱을 실행하고 모델을 다운로드한다.

```powershell
ollama pull qwen3.5:9b
ollama list
Copy-Item -LiteralPath .\.env.server.example -Destination .\.env
```

`.env`가 이미 있으면 복사 대신 기존 파일의 설정을 확인한다.
기본 생성 모델은 이 PC에서 연결을 검증한 `qwen3.5:9b`이다.
VRAM 부족이나 CPU 오프로딩으로 응답이 느리면 작은 후보를 다운로드하고 `.env`의 모델을 변경할 수 있다.

```powershell
ollama pull qwen3.5:4b
```

그 경우 `.env`를 `OLLAMA_MODEL=qwen3.5:4b`로 바꾼 뒤 서버를 재시작하고 응답 정책을 다시 확인한다.
모델 변경은 생성 결과를 바꾸므로 9b에서 확인한 정책 준수를 4b의 평가 결과로 간주하지 않는다.
Ollama는 기본 로컬 주소 `127.0.0.1:11434`를 유지한다. 팀원은 FastAPI의 8000번 포트로 접속한다.

## 5. 방화벽 허용

서버에서 관리자 PowerShell을 열고 실행한다.
이 규칙은 TCP 8000번 포트에 대해 같은 LAN 대역에서의 접근을 허용한다.

```powershell
New-NetFirewallRule -DisplayName 'BankingChatbot API 8000 LAN' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress 10.2.10.0/24 -Profile Any
```

이미 같은 이름의 규칙이 있으면 해당 규칙을 확인하여 중복 생성하지 않는다.
공유기 포트포워딩은 같은 LAN에서의 테스트에 필요하지 않다.

## 6. 테스트와 서버 시작

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
powershell -File .\scripts\start_server.ps1
```

실행 스크립트는 `0.0.0.0:8000`에 바인딩하고 worker 1개를 사용한다.
개발용 `--reload`는 사용하지 않는다. 종료하려면 Ctrl+C를 누른다.
명령을 직접 실행하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

서버 노트북에서 `http://127.0.0.1:8000/health`를 먼저 확인한다.
다른 팀원은 다음 주소로 접속한다.

- 상태 확인: http://10.2.10.71:8000/health
- API 문서·테스트: http://10.2.10.71:8000/docs
- 질문·응답 API: POST http://10.2.10.71:8000/chat

PowerShell 호출 예시:

```powershell
$body = @{ question = '환전 신청은 일반적으로 어떻게 하나요?'; model = 'local' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://10.2.10.71:8000/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

브라우저에서 `/chat` 주소를 직접 여는 것은 GET 요청이다. 질문은 `/docs` 또는 POST 요청으로 보낸다.
`0.0.0.0`는 수신 주소이며 팀원이 입력하는 접속 주소는 서버 IPv4인 `10.2.10.71`이다.

## 접속 문제 확인

- 서버 자신도 `/health`에 접근하지 못하면 서버 프로세스와 포트 번호를 확인한다.
- `/health`가 `degraded`이면 분류기 로드 결과, 모델 파일, Ollama 실행 여부를 확인한다.
- 서버에서는 접속되지만 팀원에게서 실패하면 방화벽 규칙과 LAN 대역을 확인한다.
- 같은 Wi-Fi여도 AP의 클라이언트 격리 기능이 있으면 단말 간 접속이 차단될 수 있다.
- IP가 DHCP로 변경되면 접속 URL도 바뀐다. 현재 서버 IP를 `ipconfig`로 확인한다.
- 노트북은 전원 연결 상태를 유지하고 서버 사용 중 절전·최대 절전 모드로 들어가지 않도록 설정한다.

참고: [FastAPI 서버 실행](https://fastapi.tiangolo.com/deployment/manually/), [PyTorch CUDA 설치](https://pytorch.org/get-started/previous-versions/), [Ollama Windows](https://docs.ollama.com/windows).
