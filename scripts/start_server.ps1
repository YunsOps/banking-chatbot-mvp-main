param(
    [string]$BindAddress = '0.0.0.0',
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '가상환경이 없습니다. docs/windows_server.md의 설치 절차를 먼저 실행하세요.'
}

Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m uvicorn app.main:app --host $BindAddress --port $Port --workers 1
    if ($LASTEXITCODE -ne 0) {
        throw "API 서버가 종료되었습니다. exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
