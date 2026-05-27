"""Streamlit UI에서 사용하는 기본 날짜 계산 헬퍼."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


APP_TIMEZONE = ZoneInfo("Asia/Seoul")


def today_kst() -> date:
    """현재 한국 날짜를 반환."""
    return datetime.now(APP_TIMEZONE).date()


def previous_business_day(today: date | None = None) -> date:
    """주말을 건너뛴 직전 영업일을 반환.

    한국 휴장일처럼 데이터가 없는 날짜는 진단 파이프라인에서 실제 관측 가능한
    직전 날짜로 한 번 더 보정한다.
    """
    current = today or today_kst()
    candidate = current - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate
