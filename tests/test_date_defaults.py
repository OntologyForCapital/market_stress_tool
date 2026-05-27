"""UI 기본 날짜 계산 테스트."""

from __future__ import annotations

from datetime import date

from src.ui.date_defaults import previous_business_day


def test_previous_business_day_from_tuesday_is_monday():
    assert previous_business_day(date(2026, 5, 26)) == date(2026, 5, 25)


def test_previous_business_day_from_monday_is_previous_friday():
    assert previous_business_day(date(2026, 5, 25)) == date(2026, 5, 22)


def test_previous_business_day_from_sunday_is_previous_friday():
    assert previous_business_day(date(2026, 5, 24)) == date(2026, 5, 22)
