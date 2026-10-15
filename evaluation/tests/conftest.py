"""평가 테스트 공통: 경로 설정.

실행 (저장소 루트에서): uv run python -m pytest evaluation/tests -v
실제 서버·Ollama·평가 모델을 호출하지 않는다 (가짜 분류기·생성 함수·평가 모델 사용).
"""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
for path in (EVAL_DIR / "scripts", EVAL_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
