"""dispatcher.py 통합 테스트.

테스트 전략:
    - 각 로더 함수를 monkeypatch로 mock하여 네트워크 없이 라우팅 검증
    - source별 분기, computed 산식, 실패 격리, 빈 시리즈 처리 등 시나리오 커버
    - 실제 variables.yaml을 읽어서 KRW_USD가 결과에 포함되는지 확인 (작업 3 요구사항)

실행:
    pytest tests/test_dispatcher.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config import (
    ComponentRef,
    Variable,
    load_variables,
    load_target_variables,
    get_enabled_variables,
)
from src.data_collection import dispatcher
from src.data_collection._common import (
    InvalidSeriesError,
    NetworkError,
)


# =============================================================================
# 헬퍼: mock 시리즈 생성
# =============================================================================
def _make_series(name: str, n: int = 5, base: float = 100.0) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.Series([base + i for i in range(n)], index=idx, name=name)


def _make_variable(
    code: str, source: str, **kwargs: Any
) -> Variable:
    """테스트용 Variable 생성."""
    defaults = {
        "name_kr": code,
        "name_en": code,
        "source": source,
        "channel": 1,
        "risk_direction": "positive",
        "frequency": "daily",
        "enabled": True,
    }
    defaults.update(kwargs)
    return Variable(code=code, **defaults)  # type: ignore[arg-type]


# =============================================================================
# 1. source별 라우팅
# =============================================================================
class TestSourceRouting:
    def test_fred_routing(self, monkeypatch):
        called = {}

        def mock_fred(series_id, start, end, use_cache=True):
            called["series_id"] = series_id
            return _make_series(series_id)

        monkeypatch.setattr(dispatcher.fred_loader, "fetch_fred", mock_fred)
        v = _make_variable("VIX", "fred", series_id="VIXCLS")
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert called["series_id"] == "VIXCLS"
        assert s.name == "VIX"  # variable.code로 정규화됨

    def test_yfinance_routing_with_fallback(self, monkeypatch):
        called = {}

        def mock_yfin(primary_ticker, fallback_ticker, start_date, end_date, use_cache=True):
            called["primary"] = primary_ticker
            called["fallback"] = fallback_ticker
            return _make_series(primary_ticker)

        monkeypatch.setattr(
            dispatcher.yfinance_loader, "fetch_yfinance_with_fallback", mock_yfin,
        )
        v = _make_variable(
            "BDI", "yfinance",
            series_id="^BDIY", fallback_series_id="BDRY",
        )
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert called["primary"] == "^BDIY"
        assert called["fallback"] == "BDRY"
        assert s.name == "BDI"

    def test_ecos_routing_with_item_code(self, monkeypatch):
        called = {}

        def mock_ecos(stat_code, item_code, start_date, end_date, use_cache=True):
            called["stat_code"] = stat_code
            called["item_code"] = item_code
            return _make_series(stat_code)

        monkeypatch.setattr(dispatcher.ecos_loader, "fetch_ecos", mock_ecos)
        v = _make_variable(
            "KR_EXPORT", "ecos",
            series_id="403Y001",
            item_code="*AA",
        )
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert called["stat_code"] == "403Y001"
        assert called["item_code"] == "*AA"
        assert s.name == "KR_EXPORT"

    def test_krx_kospi_routing(self, monkeypatch):
        called = {}

        def mock_kospi(start, end, use_cache=True):
            called["called"] = "kospi"
            return _make_series("코스피")

        monkeypatch.setattr(dispatcher.krx_loader, "fetch_kospi", mock_kospi)
        v = _make_variable(
            "KOSPI", "krx", series_id="코스피",
            channel=None, risk_direction=None,
        )
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert called["called"] == "kospi"
        assert s.name == "KOSPI"  # 정규화

    def test_krx_kosdaq_routing(self, monkeypatch):
        called = {}

        def mock_kosdaq(start, end, use_cache=True):
            called["called"] = "kosdaq"
            return _make_series("코스닥")

        monkeypatch.setattr(dispatcher.krx_loader, "fetch_kosdaq", mock_kosdaq)
        v = _make_variable(
            "KOSDAQ", "krx", series_id="코스닥",
            channel=None, risk_direction=None,
        )
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert called["called"] == "kosdaq"

    def test_krx_unknown_code_rejected(self, monkeypatch):
        v = _make_variable("UNKNOWN_KRX", "krx", series_id="기타지수")
        with pytest.raises(InvalidSeriesError, match="KOSPI/KOSDAQ만 지원"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")

    def test_pykrx_raises_not_implemented(self, monkeypatch):
        v = _make_variable("FOREIGN_NET_BUY", "pykrx")
        with pytest.raises(NotImplementedError, match="비활성"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")

    def test_unknown_source_rejected(self):
        v = _make_variable("X", "bloomberg")
        with pytest.raises(InvalidSeriesError, match="알 수 없는 source"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")

    def test_fred_missing_series_id(self):
        v = _make_variable("X", "fred", series_id=None)
        with pytest.raises(InvalidSeriesError, match="series_id 누락"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")


# =============================================================================
# 2. computed 변수 (KR_US_RATE_DIFF)
# =============================================================================
class TestComputedVariable:
    def test_kr_us_rate_diff(self, monkeypatch):
        """kr_rate(ECOS) - us_rate(FRED) 산식 검증."""

        # mock: 일별로 채워진 두 시리즈
        idx = pd.date_range("2024-01-02", periods=4, freq="B")
        kr_series = pd.Series([3.50, 3.50, 3.50, 3.50], index=idx, name="kr_base")
        us_series = pd.Series([5.25, 5.25, 5.33, 5.33], index=idx, name="us_ffr")

        def mock_kr_rate(start, end, use_cache=True):
            return kr_series

        def mock_fred(series_id, start, end, use_cache=True):
            assert series_id == "DFF"
            return us_series

        monkeypatch.setattr(
            dispatcher.ecos_loader, "fetch_kr_base_rate", mock_kr_rate,
        )
        monkeypatch.setattr(dispatcher.fred_loader, "fetch_fred", mock_fred)

        v = Variable(
            code="KR_US_RATE_DIFF",
            source="computed",
            channel=2,
            risk_direction="negative",
            components={
                "kr_rate": ComponentRef(
                    name="kr_rate", source="ecos", stat_code="722Y001",
                ),
                "us_rate": ComponentRef(
                    name="us_rate", source="fred", series_id="DFF",
                ),
            },
        )

        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        assert s.name == "KR_US_RATE_DIFF"
        assert len(s) == 4
        # 첫 값: 3.50 - 5.25 = -1.75
        assert s.iloc[0] == pytest.approx(-1.75)
        # 셋째 값: 3.50 - 5.33 = -1.83
        assert s.iloc[2] == pytest.approx(-1.83)

    def test_kr_us_rate_diff_with_misaligned_dates(self, monkeypatch):
        """저빈도 시리즈가 forward fill되어 매칭되는지."""
        # kr_rate는 1/2만, us_rate는 1/2, 1/3, 1/4
        kr = pd.Series(
            [3.50], index=pd.DatetimeIndex(["2024-01-02"]), name="kr",
        )
        us = pd.Series(
            [5.25, 5.25, 5.33],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
            name="us",
        )

        monkeypatch.setattr(
            dispatcher.ecos_loader, "fetch_kr_base_rate",
            lambda s, e, use_cache=True: kr,
        )
        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: us,
        )

        v = Variable(
            code="KR_US_RATE_DIFF",
            source="computed",
            components={
                "kr_rate": ComponentRef(name="kr_rate", source="ecos", stat_code="722Y001"),
                "us_rate": ComponentRef(name="us_rate", source="fred", series_id="DFF"),
            },
        )
        s = dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")
        # kr가 1/2 값 그대로 1/3, 1/4까지 ffill되므로 3일치 모두 산출
        assert len(s) == 3
        assert s.iloc[0] == pytest.approx(-1.75)
        assert s.iloc[2] == pytest.approx(-1.83)

    def test_computed_without_components_raises(self):
        v = Variable(code="BROKEN_COMPUTED", source="computed", components={})
        with pytest.raises(InvalidSeriesError, match="components가 정의되지 않음"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")

    def test_computed_unknown_code_raises(self, monkeypatch):
        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: _make_series(sid),
        )
        v = Variable(
            code="UNKNOWN_COMPUTED",
            source="computed",
            components={
                "a": ComponentRef(name="a", source="fred", series_id="X"),
            },
        )
        with pytest.raises(NotImplementedError, match="산식이 dispatcher에 구현되지 않음"):
            dispatcher.fetch_variable(v, "2024-01-01", "2024-01-10")


# =============================================================================
# 3. fetch_all_variables: 일괄 수집 + 실패 격리
# =============================================================================
class TestFetchAllVariables:
    def test_success_all(self, monkeypatch):
        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: _make_series(sid),
        )
        vars_ = [
            _make_variable("A", "fred", series_id="A1"),
            _make_variable("B", "fred", series_id="B1"),
            _make_variable("C", "fred", series_id="C1"),
        ]
        result = dispatcher.fetch_all_variables(vars_, "2024-01-01", "2024-01-10")
        assert set(result.keys()) == {"A", "B", "C"}
        assert all(s.name in {"A", "B", "C"} for s in result.values())

    def test_partial_failure_isolated(self, monkeypatch):
        """한 변수 실패가 다른 변수 수집을 막지 않음."""
        def mock_fred(sid, s, e, use_cache=True):
            if sid == "BAD":
                raise NetworkError("fake network failure")
            return _make_series(sid)

        monkeypatch.setattr(dispatcher.fred_loader, "fetch_fred", mock_fred)
        vars_ = [
            _make_variable("GOOD", "fred", series_id="GOOD1"),
            _make_variable("BAD_VAR", "fred", series_id="BAD"),
            _make_variable("ALSO_GOOD", "fred", series_id="ALSO1"),
        ]
        result = dispatcher.fetch_all_variables(vars_, "2024-01-01", "2024-01-10")
        assert set(result.keys()) == {"GOOD", "ALSO_GOOD"}
        assert "BAD_VAR" not in result

    def test_disabled_variables_skipped(self, monkeypatch):
        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: _make_series(sid),
        )
        vars_ = [
            _make_variable("ENABLED", "fred", series_id="E1", enabled=True),
            _make_variable("DISABLED", "fred", series_id="D1", enabled=False),
        ]
        result = dispatcher.fetch_all_variables(vars_, "2024-01-01", "2024-01-10")
        assert set(result.keys()) == {"ENABLED"}

    def test_include_disabled_overrides(self, monkeypatch):
        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: _make_series(sid),
        )
        vars_ = [
            _make_variable("ENABLED", "fred", series_id="E1", enabled=True),
            _make_variable("DISABLED", "fred", series_id="D1", enabled=False),
        ]
        result = dispatcher.fetch_all_variables(
            vars_, "2024-01-01", "2024-01-10", include_disabled=True,
        )
        assert set(result.keys()) == {"ENABLED", "DISABLED"}

    def test_empty_series_included_with_warning(self, monkeypatch, caplog):
        """빈 시리즈도 결과에 포함되며 경고 로그가 남음."""
        empty = pd.Series(dtype="float64", name="EMPTY")
        empty.index = pd.DatetimeIndex([], name="date")

        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: empty,
        )
        vars_ = [_make_variable("EMPTY", "fred", series_id="X")]
        with caplog.at_level("WARNING"):
            result = dispatcher.fetch_all_variables(vars_, "2024-01-01", "2024-01-10")
        assert "EMPTY" in result
        assert any("빈 시리즈" in record.message for record in caplog.records)

    def test_unexpected_exception_isolated(self, monkeypatch):
        """LOADER_EXCEPTIONS 밖의 예외도 격리되어야 함."""
        def mock_fred(sid, s, e, use_cache=True):
            if sid == "WEIRD":
                raise RuntimeError("unexpected internal error")
            return _make_series(sid)

        monkeypatch.setattr(dispatcher.fred_loader, "fetch_fred", mock_fred)
        vars_ = [
            _make_variable("GOOD", "fred", series_id="GOOD1"),
            _make_variable("WEIRD_VAR", "fred", series_id="WEIRD"),
        ]
        result = dispatcher.fetch_all_variables(vars_, "2024-01-01", "2024-01-10")
        assert "GOOD" in result
        assert "WEIRD_VAR" not in result


# =============================================================================
# 4. 실제 variables.yaml 통합 (작업 3 요구사항)
# =============================================================================
class TestRealYamlIntegration:
    """실제 config/variables.yaml을 읽고 mock 로더로 dispatcher 동작 검증."""

    def _mock_all_loaders(self, monkeypatch):
        """모든 로더를 mock으로 교체 (네트워크 없이)."""
        def mock_simple(sid_or_args, *args, **kwargs):
            # 첫 인자가 series_id (str)이라고 가정
            name = sid_or_args if isinstance(sid_or_args, str) else "X"
            return _make_series(name)

        monkeypatch.setattr(
            dispatcher.fred_loader, "fetch_fred",
            lambda sid, s, e, use_cache=True: _make_series(sid),
        )
        monkeypatch.setattr(
            dispatcher.yfinance_loader, "fetch_yfinance_with_fallback",
            lambda primary_ticker, fallback_ticker, start_date, end_date, use_cache=True:
                _make_series(primary_ticker),
        )
        monkeypatch.setattr(
            dispatcher.ecos_loader, "fetch_ecos",
            lambda stat_code, item_code, start_date, end_date, use_cache=True:
                _make_series(stat_code),
        )
        monkeypatch.setattr(
            dispatcher.ecos_loader, "fetch_kr_base_rate",
            lambda s, e, use_cache=True: _make_series("kr_base"),
        )
        monkeypatch.setattr(
            dispatcher.krx_loader, "fetch_kospi",
            lambda s, e, use_cache=True: _make_series("코스피"),
        )
        monkeypatch.setattr(
            dispatcher.krx_loader, "fetch_kosdaq",
            lambda s, e, use_cache=True: _make_series("코스닥"),
        )

    def test_krw_usd_included(self, monkeypatch):
        """[작업 3 요구사항] enabled=True 변수 전체 수집 시 KRW_USD 포함."""
        self._mock_all_loaders(monkeypatch)
        enabled_vars = get_enabled_variables()
        result = dispatcher.fetch_all_variables(
            enabled_vars, "2024-01-01", "2024-01-10",
        )
        assert "KRW_USD" in result, (
            "KRW_USD가 fetch_all_variables 결과에 포함되어야 합니다 "
            "(variables.yaml의 enabled=true 변수)"
        )
        assert result["KRW_USD"].name == "KRW_USD"

    def test_all_enabled_variables_collected(self, monkeypatch):
        """variables.yaml의 모든 enabled=True 변수가 수집되는지 확인."""
        self._mock_all_loaders(monkeypatch)
        enabled_vars = get_enabled_variables()
        result = dispatcher.fetch_all_variables(
            enabled_vars, "2024-01-01", "2024-01-10",
        )
        expected_codes = {v.code for v in enabled_vars}
        assert set(result.keys()) == expected_codes, (
            f"실패한 변수: {expected_codes - set(result.keys())}"
        )

    def test_targets_collected(self, monkeypatch):
        """target_variables (KOSPI, KOSDAQ)이 함께 수집됨."""
        self._mock_all_loaders(monkeypatch)
        all_vars = get_enabled_variables() + load_target_variables()
        result = dispatcher.fetch_all_variables(
            all_vars, "2024-01-01", "2024-01-10",
        )
        assert "KOSPI" in result
        assert "KOSDAQ" in result

    def test_disabled_variables_excluded(self, monkeypatch):
        """SP500_EPS, MOVE 같은 disabled 변수는 결과에 없음."""
        self._mock_all_loaders(monkeypatch)
        all_vars = load_variables()  # disabled 포함
        result = dispatcher.fetch_all_variables(
            all_vars, "2024-01-01", "2024-01-10",
        )
        # variables.yaml에서 disabled로 설정된 SP500_EPS, MOVE
        assert "SP500_EPS" not in result
        assert "MOVE" not in result
