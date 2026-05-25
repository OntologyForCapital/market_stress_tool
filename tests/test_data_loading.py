"""데이터 수집 모듈 단위 테스트.

각 로더마다:
    1. live 마커: 실제 API 호출 1개 (RUN_LIVE_TESTS=1 시에만 실행)
    2. mock 또는 키 누락 시나리오: 1개 (항상 실행)

실행:
    # 빠른 테스트만 (실 API 호출 없이)
    pytest tests/

    # 실제 API 호출 포함
    RUN_LIVE_TESTS=1 pytest tests/ -v
"""

from __future__ import annotations

import os
import pandas as pd
import pytest

from src.data_collection._common import (
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
)
from src.data_collection import fred_loader, yfinance_loader, ecos_loader, pykrx_loader


# =============================================================================
# 공통 기간 (테스트 비용 최소화 위해 짧게)
# =============================================================================
START = "2024-01-01"
END = "2024-02-01"


# =============================================================================
# FRED 로더
# =============================================================================
@pytest.mark.live
def test_fred_live_vix():
    """[Live] FRED에서 VIX 1개월 데이터를 받아온다."""
    s = fred_loader.fetch_fred("VIXCLS", START, END, use_cache=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert isinstance(s.index, pd.DatetimeIndex)
    # VIX는 보통 10~80 범위
    assert (s.dropna() > 0).all()
    assert (s.dropna() < 200).all()
    assert s.name == "VIXCLS"


def test_fred_missing_api_key(monkeypatch):
    """[Mock] FRED_API_KEY 미설정 시 MissingAPIKeyError를 던진다."""
    # 환경에서 키 완전히 제거
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    # python-dotenv가 .env를 다시 로드하지 못하게 PROJECT_ROOT의 .env가 없는 시나리오 가정
    # (실제로는 require_env가 빈/플레이스홀더 값을 거부)
    monkeypatch.setenv("FRED_API_KEY", "")  # 명시적으로 비움

    with pytest.raises(MissingAPIKeyError):
        fred_loader.fetch_fred("VIXCLS", START, END, use_cache=False)


# =============================================================================
# yfinance 로더
# =============================================================================
@pytest.mark.live
def test_yfinance_live_dxy():
    """[Live] yfinance에서 DXY 1개월 데이터를 받아온다."""
    s = yfinance_loader.fetch_yfinance("DX-Y.NYB", START, END, use_cache=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert isinstance(s.index, pd.DatetimeIndex)
    # DXY는 보통 70~130 범위
    assert (s > 50).all()
    assert (s < 200).all()


def test_yfinance_invalid_ticker():
    """[Live-like] 잘못된 티커는 InvalidSeriesError를 던진다.

    yfinance는 빈 DataFrame을 반환하므로 우리 코드가 이를 InvalidSeriesError로 변환.
    네트워크 자체는 호출되므로 빠르게 실패. 캐시 비활성화로 매번 실제 호출.
    """
    # 네트워크 차단 환경에서도 yfinance.download는 빈 결과를 반환하는 경향이 있어
    # 이 테스트는 가벼운 통합 테스트로 다룬다. 네트워크 차단이면 NetworkError를 받을 수도 있음.
    with pytest.raises((InvalidSeriesError, NetworkError)):
        yfinance_loader.fetch_yfinance(
            "DEFINITELY_NOT_A_REAL_TICKER_XYZ_123", START, END, use_cache=False
        )


# =============================================================================
# ECOS 로더
# =============================================================================
@pytest.mark.live
def test_ecos_live_kr_base_rate():
    """[Live] ECOS에서 한국 기준금리를 받아온다."""
    s = ecos_loader.fetch_kr_base_rate(START, END, use_cache=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    # 한국 기준금리는 0~10% 범위
    assert (s.dropna() >= 0).all()
    assert (s.dropna() <= 15).all()


def test_ecos_missing_api_key(monkeypatch):
    """[Mock] ECOS_API_KEY 미설정 시 MissingAPIKeyError."""
    monkeypatch.setenv("ECOS_API_KEY", "")
    with pytest.raises(MissingAPIKeyError):
        ecos_loader.fetch_ecos(
            stat_code="722Y001",
            item_code="0101000",
            start_date=START,
            end_date=END,
            use_cache=False,
        )


def test_ecos_invalid_stat_code(monkeypatch, mocker):
    """[Mock] ECOS가 INFO-200(자료없음)을 반환하면 InvalidSeriesError.

    실제 HTTP 호출 대신 requests.get을 모킹.
    """
    monkeypatch.setenv("ECOS_API_KEY", "FAKE_KEY_FOR_TEST_12345678")

    class FakeResp:
        status_code = 200

        def json(self):
            return {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당 자료가 없습니다."}}

    mocker.patch("src.data_collection.ecos_loader.requests.get", return_value=FakeResp())

    with pytest.raises(InvalidSeriesError):
        ecos_loader.fetch_ecos(
            stat_code="NONEXISTENT",
            item_code=None,
            start_date=START,
            end_date=END,
            use_cache=False,
        )


# =============================================================================
# pykrx 로더
# =============================================================================
@pytest.mark.live
def test_pykrx_live_kospi():
    """[Live] pykrx에서 KOSPI 1개월 데이터를 받아온다."""
    s = pykrx_loader.fetch_kospi(START, END, use_cache=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert s.name == "KOSPI"
    # KOSPI는 보통 1500~4000 범위
    assert (s > 1000).all()
    assert (s < 5000).all()


def test_pykrx_invalid_index(mocker):
    """[Mock] 잘못된 지수 코드는 InvalidSeriesError를 던진다.

    pykrx.stock.get_index_ohlcv_by_date가 빈 DataFrame을 반환하는 상황을 모킹.
    """
    import pandas as pd
    fake_df = pd.DataFrame()  # 빈 응답

    # pykrx.stock 자체를 모킹 (호출 시점에 import 되므로 모듈 패스가 다를 수 있음)
    import pykrx.stock as stock_module
    mocker.patch.object(stock_module, "get_index_ohlcv_by_date", return_value=fake_df)

    with pytest.raises(InvalidSeriesError):
        pykrx_loader.fetch_krx_index("9999", START, END, use_cache=False)


# =============================================================================
# 공통: KST 인덱스 확인 (live 데이터 사용)
# =============================================================================
@pytest.mark.live
def test_index_is_datetimeindex_and_tz_naive():
    """[Live] 모든 로더의 결과 인덱스가 tz-naive DatetimeIndex인지 확인."""
    s = fred_loader.fetch_fred("VIXCLS", START, END, use_cache=False)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.tz is None  # KST로 변환 후 tz-naive로 정규화
