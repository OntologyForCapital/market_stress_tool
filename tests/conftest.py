"""pytest 설정.

- 프로젝트 루트를 sys.path에 추가 (src.data_collection import 가능하게)
- 실제 API 호출 테스트는 환경 변수로 토글: RUN_LIVE_TESTS=1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """커스텀 마커 등록."""
    config.addinivalue_line(
        "markers",
        "live: 실제 외부 API 호출이 필요한 테스트 (RUN_LIVE_TESTS=1 환경변수로 활성화)",
    )


def pytest_collection_modifyitems(config, items):
    """RUN_LIVE_TESTS=1이 아니면 live 마커 테스트를 스킵."""
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="실 API 호출 테스트는 RUN_LIVE_TESTS=1 환경변수로 활성화"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
