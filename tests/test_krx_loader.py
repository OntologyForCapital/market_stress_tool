"""KRX OPEN API 로더 통합 테스트.

테스트 범위:
    1. _coerce_numeric (엣지케이스 7종)
    2. _parse_krx_response (정상/휴일/결측/잘못된 응답/중복)
    3. fetch_krx_index_one_day (mock 응답으로 캐시 hit/miss)
    4. fetch_krx_index_range (mock으로 영업일 루프/휴일 스킵/지수명 누락)
    5. fetch_kospi, fetch_kosdaq (편의 함수 위임 확인)
    6. clear_krx_cache, get_krx_cache_status (캐시 유틸)
    7. 약관 출처 표기 상수
    8. [Live] 실제 KRX OPEN API 호출 (RUN_LIVE_TESTS=1 + KRX_API_KEY 설정 시)

실행:
    pytest tests/test_krx_loader.py -v
    RUN_LIVE_TESTS=1 pytest tests/test_krx_loader.py -v  # 라이브 포함
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_collection import krx_loader as krx
from src.data_collection._common import (
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
)


# =============================================================================
# 픽스처: 격리된 임시 캐시 디렉터리
# =============================================================================
@pytest.fixture
def isolated_cache(monkeypatch):
    """krx_loader.RAW_CACHE_DIR을 임시 디렉터리로 교체.

    각 테스트 후 자동 정리. 다른 로더의 실제 캐시는 건드리지 않음.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="krx_test_cache_"))
    monkeypatch.setattr(krx, "RAW_CACHE_DIR", tmp_dir)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# 헬퍼: mock 응답 생성
# =============================================================================
def _make_krx_payload(rows: list[tuple[str, float]] | None) -> dict[str, Any]:
    """KRX OPEN API JSON 응답 형식의 mock payload 생성.

    Args:
        rows: [(idx_nm, close), ...] 또는 None/빈 리스트.

    Returns:
        {"OutBlock_1": [...]} 형태의 dict.
    """
    if not rows:
        return {"OutBlock_1": []}
    block = []
    for nm, close in rows:
        block.append({
            "BAS_DD": "20240102",
            "IDX_CLSS": "KRX",
            "IDX_NM": nm,
            "CLSPRC_IDX": str(close),
            "CMPPREVDD_IDX": "0",
            "FLUC_RT": "0",
            "OPNPRC_IDX": "0",
            "HGPRC_IDX": "0",
            "LWPRC_IDX": "0",
            "ACC_TRDVOL": "0",
            "ACC_TRDVAL": "0",
            "MKTCAP": "0",
        })
    return {"OutBlock_1": block}


# =============================================================================
# 1. _coerce_numeric
# =============================================================================
class TestCoerceNumeric:
    @pytest.mark.parametrize("inp,expected", [
        ("1,234.56", 1234.56),
        ("1000", 1000.0),
        ("0", 0.0),
        (1234, 1234.0),
        (3.14, 3.14),
        ("  2,500.5 ", 2500.5),
        ("-1,234.5", -1234.5),
    ])
    def test_valid_values(self, inp, expected):
        assert math.isclose(krx._coerce_numeric(inp), expected)

    @pytest.mark.parametrize("inp", ["-", "", None, "N/A", "null", "None", "abc"])
    def test_invalid_to_nan(self, inp):
        assert math.isnan(krx._coerce_numeric(inp))


# =============================================================================
# 2. _parse_krx_response
# =============================================================================
class TestParseKrxResponse:
    def test_normal_response(self):
        payload = _make_krx_payload([("코스피", 2669.81), ("코스닥", 878.93)])
        df = krx._parse_krx_response(payload, base_date="2024-01-02")
        assert df.shape == (2, 11)
        assert df.index.name == "idx_nm"
        assert "코스피" in df.index
        assert "코스닥" in df.index
        assert math.isclose(df.loc["코스피", "close"], 2669.81)
        # 컬럼 순서 고정
        expected_cols = ["bas_dd", "idx_clss", "close", "change", "fluc_rt",
                         "open", "high", "low", "trd_vol", "trd_val", "mkt_cap"]
        assert list(df.columns) == expected_cols

    def test_holiday_empty_response(self):
        """빈 OutBlock_1은 동일 스키마의 빈 DataFrame을 반환."""
        df = krx._parse_krx_response({"OutBlock_1": []}, base_date="2024-01-01")
        assert df.empty
        assert df.index.name == "idx_nm"
        expected_cols = ["bas_dd", "idx_clss", "close", "change", "fluc_rt",
                         "open", "high", "low", "trd_vol", "trd_val", "mkt_cap"]
        assert list(df.columns) == expected_cols

    def test_missing_values(self):
        """'-', '', None 등은 NaN으로 변환된다."""
        payload = {
            "OutBlock_1": [{
                "BAS_DD": "20240105", "IDX_CLSS": "KRX", "IDX_NM": "테스트",
                "CLSPRC_IDX": "100.0", "CMPPREVDD_IDX": "-", "FLUC_RT": "",
                "OPNPRC_IDX": "99", "HGPRC_IDX": "101", "LWPRC_IDX": "98",
                "ACC_TRDVOL": "1,000", "ACC_TRDVAL": "100,000", "MKTCAP": None,
            }]
        }
        df = krx._parse_krx_response(payload, base_date="2024-01-05")
        assert math.isclose(df.loc["테스트", "close"], 100.0)
        assert math.isnan(df.loc["테스트", "change"])
        assert math.isnan(df.loc["테스트", "fluc_rt"])
        assert math.isnan(df.loc["테스트", "mkt_cap"])
        assert math.isclose(df.loc["테스트", "trd_vol"], 1000.0)

    def test_missing_outblock_key(self):
        with pytest.raises(InvalidSeriesError, match="OutBlock_1"):
            krx._parse_krx_response({"error": "invalid auth"}, base_date="2024-01-02")

    def test_outblock_not_list(self):
        with pytest.raises(InvalidSeriesError, match="배열이 아닙니다"):
            krx._parse_krx_response({"OutBlock_1": "string"}, base_date="2024-01-02")

    def test_payload_not_dict(self):
        with pytest.raises(InvalidSeriesError, match="dict가 아닙니다"):
            krx._parse_krx_response([], base_date="2024-01-02")  # type: ignore[arg-type]

    def test_duplicate_idx_nm_keeps_first(self):
        payload = _make_krx_payload([("X", 1.0), ("X", 999.0)])
        df = krx._parse_krx_response(payload, base_date="2024-01-02")
        assert df.shape == (1, 11)
        assert math.isclose(df.loc["X", "close"], 1.0)


# =============================================================================
# 3. fetch_krx_index_one_day (mock 응답)
# =============================================================================
class TestFetchOneDay:
    def test_cache_miss_then_hit(self, isolated_cache, monkeypatch):
        """첫 호출은 API, 두 번째는 캐시 hit."""
        call_log = []

        def mock_request(base_date: str):
            call_log.append(base_date)
            return _make_krx_payload([("코스피", 2669.81)])

        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)

        df1 = krx.fetch_krx_index_one_day("2024-01-02")
        df2 = krx.fetch_krx_index_one_day("2024-01-02")
        assert len(call_log) == 1
        assert df1.equals(df2)
        assert math.isclose(df1.loc["코스피", "close"], 2669.81)

    def test_holiday_cached(self, isolated_cache, monkeypatch):
        """휴일(빈 배열)도 캐시되어 재호출 시 API 호출 안 함."""
        call_log = []

        def mock_request(base_date: str):
            call_log.append(base_date)
            return {"OutBlock_1": []}

        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)

        df1 = krx.fetch_krx_index_one_day("2024-01-01")
        df2 = krx.fetch_krx_index_one_day("2024-01-01")
        assert len(call_log) == 1
        assert df1.empty and df2.empty
        # 메타에 is_holiday=True
        meta_path = krx._krx_day_meta_path("2024-01-01")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_holiday"] is True
        assert meta["rows"] == 0

    def test_use_cache_false_bypasses_cache(self, isolated_cache, monkeypatch):
        """use_cache=False는 캐시를 무시하고 항상 API 호출."""
        call_log = []

        def mock_request(base_date: str):
            call_log.append(base_date)
            return _make_krx_payload([("코스피", 100.0)])

        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)

        krx.fetch_krx_index_one_day("2024-01-02", use_cache=False)
        krx.fetch_krx_index_one_day("2024-01-02", use_cache=False)
        assert len(call_log) == 2


# =============================================================================
# 4. fetch_krx_index_range (mock)
# =============================================================================
class TestFetchRange:
    def _setup_mock(self, monkeypatch, mock_data):
        """mock_data: {'YYYYMMDD': [(idx_nm, close), ...]} 형태."""
        call_log = []

        def mock_request(base_date: str):
            call_log.append(base_date)
            key = base_date.replace("-", "")[:8]
            return _make_krx_payload(mock_data.get(key))

        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)
        return call_log

    def test_normal_range_with_holiday(self, isolated_cache, monkeypatch):
        """4 영업일 중 1일 휴일 → 3건 시리즈."""
        mock_data = {
            "20240102": [("코스피", 2669.81), ("코스닥", 878.93)],
            "20240103": [("코스피", 2607.31), ("코스닥", 866.25)],
            "20240104": [],  # 휴일
            "20240105": [("코스피", 2578.08), ("코스닥", 860.32)],
        }
        self._setup_mock(monkeypatch, mock_data)

        s = krx.fetch_krx_index_range(
            "코스피", "2024-01-02", "2024-01-05",
            request_delay_sec=0.0,
        )
        assert isinstance(s, pd.Series)
        assert s.name == "코스피"
        assert str(s.dtype) == "float64"
        assert s.index.name == "date"
        assert s.index.tz is None
        assert len(s) == 3  # 1/4 휴일 제외
        assert math.isclose(s.iloc[0], 2669.81)
        assert math.isclose(s.iloc[-1], 2578.08)

    def test_missing_index_name(self, isolated_cache, monkeypatch):
        """응답에 해당 지수명이 없으면 빈 시리즈."""
        mock_data = {
            "20240102": [("코스피", 2669.81)],
            "20240103": [("코스피", 2607.31)],
        }
        self._setup_mock(monkeypatch, mock_data)

        s = krx.fetch_krx_index_range(
            "존재하지않는지수", "2024-01-02", "2024-01-03",
            request_delay_sec=0.0,
        )
        assert len(s) == 0
        assert s.name == "존재하지않는지수"

    def test_invalid_start_date_before_krx_start(self, isolated_cache):
        with pytest.raises(ValueError, match="2010-01-04"):
            krx.fetch_krx_index_range("코스피", "2009-01-01", "2024-01-05")

    def test_start_after_end(self, isolated_cache):
        with pytest.raises(ValueError, match="늦습니다"):
            krx.fetch_krx_index_range("코스피", "2024-01-05", "2024-01-02")

    def test_invalid_field(self, isolated_cache):
        with pytest.raises(ValueError, match="지원되지 않습니다"):
            krx.fetch_krx_index_range(
                "코스피", "2024-01-02", "2024-01-05",
                field="invalid_col",
            )

    def test_weekend_only_no_business_days(self, isolated_cache, monkeypatch):
        """주말만 → 호출 0회, 빈 시리즈."""
        call_log = self._setup_mock(monkeypatch, {})
        s = krx.fetch_krx_index_range(
            "코스피", "2024-01-06", "2024-01-07",  # 토, 일
            request_delay_sec=0.0,
        )
        assert len(s) == 0
        assert len(call_log) == 0

    def test_yyyymmdd_format(self, isolated_cache, monkeypatch):
        """'YYYYMMDD' 형식 입력도 정상 동작."""
        mock_data = {
            "20240102": [("코스피", 2669.81)],
        }
        self._setup_mock(monkeypatch, mock_data)
        s = krx.fetch_krx_index_range(
            "코스피", "20240102", "20240102",
            request_delay_sec=0.0,
        )
        assert len(s) == 1


# =============================================================================
# 5. fetch_kospi, fetch_kosdaq 편의 함수
# =============================================================================
class TestConvenienceFunctions:
    def test_fetch_kospi(self, isolated_cache, monkeypatch):
        def mock_request(base_date: str):
            return _make_krx_payload([
                ("코스피", 2669.81),
                ("코스닥", 878.93),
                ("KOSPI 200", 360.0),  # fetch_kospi에 잡혀선 안 됨
            ])
        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)

        s = krx.fetch_kospi("2024-01-02", "2024-01-02", request_delay_sec=0.0)
        assert s.name == "코스피"
        assert len(s) == 1
        assert math.isclose(s.iloc[0], 2669.81)  # NOT 360.0

    def test_fetch_kosdaq(self, isolated_cache, monkeypatch):
        def mock_request(base_date: str):
            return _make_krx_payload([("코스피", 2669.81), ("코스닥", 878.93)])
        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)

        s = krx.fetch_kosdaq("2024-01-02", "2024-01-02", request_delay_sec=0.0)
        assert s.name == "코스닥"
        assert len(s) == 1
        assert math.isclose(s.iloc[0], 878.93)

    def test_constants(self):
        assert krx.KOSPI_INDEX_NAME == "코스피"
        assert krx.KOSDAQ_INDEX_NAME == "코스닥"


# =============================================================================
# 6. 캐시 유틸리티
# =============================================================================
class TestCacheUtilities:
    def _seed_cache(self, monkeypatch, dates_data):
        """캐시 시드 헬퍼."""
        def mock_request(base_date: str):
            key = base_date.replace("-", "")[:8]
            return _make_krx_payload(dates_data.get(key))
        monkeypatch.setattr(krx, "_request_krx_one_day", mock_request)
        for date_key in dates_data:
            date_str = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}"
            krx.fetch_krx_index_one_day(date_str)

    def test_get_cache_status_all_cached(self, isolated_cache, monkeypatch):
        self._seed_cache(monkeypatch, {
            "20240102": [("코스피", 2669.81)],
            "20240103": [("코스피", 2607.31)],
            "20240104": [],  # 휴일
            "20240105": [("코스피", 2578.08)],
        })
        status = krx.get_krx_cache_status("2024-01-02", "2024-01-05")
        assert status["biz_days"] == 4
        assert status["cached"] == 4
        assert status["cached_data"] == 3
        assert status["cached_holiday"] == 1
        assert status["missing"] == 0

    def test_get_cache_status_nothing_cached(self, isolated_cache):
        status = krx.get_krx_cache_status("2024-02-01", "2024-02-05")
        assert status["cached"] == 0
        assert status["missing"] == status["biz_days"] > 0

    def test_get_cache_status_invalid_range(self, isolated_cache):
        with pytest.raises(ValueError, match="늦습니다"):
            krx.get_krx_cache_status("2024-01-05", "2024-01-02")

    def test_clear_cache_range(self, isolated_cache, monkeypatch):
        """구간 삭제: 1/3, 1/4만 삭제, 1/2, 1/5 보존."""
        self._seed_cache(monkeypatch, {
            "20240102": [("코스피", 2669.81)],
            "20240103": [("코스피", 2607.31)],
            "20240104": [],
            "20240105": [("코스피", 2578.08)],
        })
        deleted = krx.clear_krx_cache("2024-01-03", "2024-01-04")
        # parquet + meta가 각각 2일치 → 4
        assert deleted == 4
        assert krx._krx_day_cache_path("2024-01-02").exists()
        assert krx._krx_day_cache_path("2024-01-05").exists()
        assert not krx._krx_day_cache_path("2024-01-03").exists()
        assert not krx._krx_day_cache_path("2024-01-04").exists()

    def test_clear_cache_all(self, isolated_cache, monkeypatch):
        self._seed_cache(monkeypatch, {
            "20240102": [("코스피", 2669.81)],
            "20240103": [("코스피", 2607.31)],
        })
        n_before = len(list(isolated_cache.glob("krx_dd_trd_*")))
        deleted = krx.clear_krx_cache()
        n_after = len(list(isolated_cache.glob("krx_dd_trd_*")))
        assert n_after == 0
        assert deleted == n_before

    def test_clear_cache_no_directory(self, isolated_cache):
        """캐시 디렉터리 없을 때 0 반환."""
        shutil.rmtree(isolated_cache)
        assert krx.clear_krx_cache() == 0

    def test_ensure_cache_dir(self, isolated_cache):
        """디렉터리 삭제 후 자동 재생성."""
        shutil.rmtree(isolated_cache)
        assert not isolated_cache.exists()
        krx._ensure_cache_dir()
        assert isolated_cache.exists()


# =============================================================================
# 7. 약관 출처 표기 상수
# =============================================================================
def test_data_source_notice_constant():
    """약관 제10조 3항 의무 출처 표기."""
    assert krx.DATA_SOURCE_NOTICE_KR == "데이터 출처: 한국거래소 통계정보"


# =============================================================================
# 8. 인증 헤더
# =============================================================================
class TestAuthHeaders:
    def test_missing_api_key(self, monkeypatch):
        """KRX_API_KEY 미설정 시 MissingAPIKeyError."""
        monkeypatch.setenv("KRX_API_KEY", "")
        with pytest.raises(MissingAPIKeyError):
            krx._auth_headers()

    def test_auth_header_format(self, monkeypatch):
        monkeypatch.setenv("KRX_API_KEY", "TEST_KEY_1234567890ABCDEF")
        headers = krx._auth_headers()
        assert headers == {"AUTH_KEY": "TEST_KEY_1234567890ABCDEF"}


# =============================================================================
# 9. _request_krx_one_day: HTTP 오류 처리
# =============================================================================
class TestRequestErrorHandling:
    def test_http_error_status(self, isolated_cache, monkeypatch):
        """HTTP 4xx/5xx 응답 시 NetworkError."""
        monkeypatch.setenv("KRX_API_KEY", "TEST_KEY")

        class FakeResp:
            status_code = 401
            text = '{"error":"invalid auth"}'

        def fake_get(*args, **kwargs):
            return FakeResp()

        monkeypatch.setattr(krx.requests, "get", fake_get)
        with pytest.raises(NetworkError, match="HTTP 401"):
            krx._request_krx_one_day("2024-01-02")

    def test_invalid_json_response(self, isolated_cache, monkeypatch):
        """JSON 파싱 실패 시 NetworkError."""
        monkeypatch.setenv("KRX_API_KEY", "TEST_KEY")

        class FakeResp:
            status_code = 200
            text = "<html>error page</html>"

            def json(self):
                raise ValueError("invalid json")

        def fake_get(*args, **kwargs):
            return FakeResp()

        monkeypatch.setattr(krx.requests, "get", fake_get)
        with pytest.raises(NetworkError, match="JSON이 아닙니다"):
            krx._request_krx_one_day("2024-01-02")

    def test_network_exception(self, isolated_cache, monkeypatch):
        """requests 예외 → NetworkError로 래핑."""
        monkeypatch.setenv("KRX_API_KEY", "TEST_KEY")

        def fake_get(*args, **kwargs):
            raise krx.requests.ConnectionError("DNS lookup failed")

        monkeypatch.setattr(krx.requests, "get", fake_get)
        with pytest.raises(NetworkError, match="호출 실패"):
            krx._request_krx_one_day("2024-01-02")


# =============================================================================
# 10. [Live] 실제 KRX OPEN API 호출
# =============================================================================
@pytest.mark.live
def test_krx_live_kospi_one_day():
    """[Live] KRX OPEN API에서 코스피 1일치 데이터를 받는다.

    조건:
        - RUN_LIVE_TESTS=1
        - .env에 KRX_API_KEY 설정
    """
    if not os.environ.get("KRX_API_KEY"):
        pytest.skip("KRX_API_KEY 환경변수가 없습니다.")

    # 2024-01-02 (월요일, 거래일)
    df = krx.fetch_krx_index_one_day("2024-01-02", use_cache=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "코스피" in df.index
    assert "코스닥" in df.index
    # 종가는 양수
    assert df.loc["코스피", "close"] > 0
    assert df.loc["코스닥", "close"] > 0


@pytest.mark.live
def test_krx_live_kospi_range():
    """[Live] KRX에서 코스피 1주일치 종가 시리즈를 받는다."""
    if not os.environ.get("KRX_API_KEY"):
        pytest.skip("KRX_API_KEY 환경변수가 없습니다.")

    s = krx.fetch_kospi("2024-01-02", "2024-01-08", use_cache=False)
    assert isinstance(s, pd.Series)
    assert len(s) > 0
    assert s.name == "코스피"
    # 코스피는 1000~5000 범위
    assert (s > 1000).all()
    assert (s < 5000).all()
    # 인덱스는 tz-naive DatetimeIndex
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.tz is None
