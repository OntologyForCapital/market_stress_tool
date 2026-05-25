"""변수 사전 변환(transform) 기능 단위 테스트 — v13 추가.

검증 대상:
    1. apply_transform: 4가지 변환 케이스 + 잘못된 값 ValueError
    2. apply_transforms_panel: 매핑 누락 컬럼은 'level'로 처리
    3. Variable dataclass: transform 필드 기본값 + 검증
    4. load_transform_map: enabled=True 변수만 포함
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    VALID_TRANSFORMS,
    Variable,
    _parse_variable_entry,
    load_transform_map,
)
from src.preprocessing.standardize import (
    HALFYEAR_TRADING_DAYS,
    TRADING_DAYS_PER_YEAR,
    apply_transform,
    apply_transforms_panel,
)


# =============================================================================
# 헬퍼: 영업일 인덱스 시계열 생성
# =============================================================================
def _make_business_series(n: int = 600, start: float = 100.0, drift: float = 0.5) -> pd.Series:
    """선형 증가하는 영업일 시계열을 생성. n=600 ≈ 약 2.4년 길이."""
    idx = pd.bdate_range("2020-01-02", periods=n)
    values = start + drift * np.arange(n, dtype=float)
    return pd.Series(values, index=idx, name="X")


# =============================================================================
# 1) apply_transform — 4가지 케이스
# =============================================================================
class TestApplyTransform:
    def test_level_returns_input_unchanged(self):
        s = _make_business_series(n=300)
        out = apply_transform(s, "level")
        # level은 입력 그대로 (동일 객체 또는 동일 값)
        pd.testing.assert_series_equal(out, s)

    def test_yoy_pct_first_252_are_nan(self):
        # yoy_pct는 252영업일 워밍아웃 → 앞 252개는 NaN
        s = _make_business_series(n=400, start=100.0, drift=1.0)
        out = apply_transform(s, "yoy_pct")
        # 워밍아웃 구간 NaN
        assert out.iloc[:TRADING_DAYS_PER_YEAR].isna().all()
        # 그 이후는 유효값 (선형 증가 시리즈라 양수 %)
        assert out.iloc[TRADING_DAYS_PER_YEAR:].notna().all()
        # 252번째 값: (100+252)/100 - 1 = 2.52 → 252.0%
        expected_252 = (s.iloc[252] / s.iloc[0] - 1.0) * 100.0
        assert math.isclose(out.iloc[252], expected_252, rel_tol=1e-9)

    def test_pct_change_6m_first_126_are_nan(self):
        s = _make_business_series(n=300, start=100.0, drift=1.0)
        out = apply_transform(s, "pct_change_6m")
        assert out.iloc[:HALFYEAR_TRADING_DAYS].isna().all()
        assert out.iloc[HALFYEAR_TRADING_DAYS:].notna().all()
        # 126번째: (100+126)/100 - 1 = 1.26 → 126.0%
        expected_126 = (s.iloc[126] / s.iloc[0] - 1.0) * 100.0
        assert math.isclose(out.iloc[126], expected_126, rel_tol=1e-9)

    def test_diff_6m_first_126_are_nan(self):
        # diff는 단순 차분 (% 아님). 선형 시리즈에서는 일정한 차분.
        s = _make_business_series(n=300, start=100.0, drift=0.5)
        out = apply_transform(s, "diff_6m")
        assert out.iloc[:HALFYEAR_TRADING_DAYS].isna().all()
        assert out.iloc[HALFYEAR_TRADING_DAYS:].notna().all()
        # 126번째: (100 + 126*0.5) - 100 = 63.0
        assert math.isclose(out.iloc[126], 63.0, rel_tol=1e-9)

    def test_unknown_transform_raises(self):
        s = _make_business_series(n=10)
        with pytest.raises(ValueError, match="지원하지 않는 transform"):
            apply_transform(s, "invalid_xyz")

    def test_index_preserved(self):
        s = _make_business_series(n=400)
        for t in ("level", "yoy_pct", "pct_change_6m", "diff_6m"):
            out = apply_transform(s, t)
            assert (out.index == s.index).all(), f"transform={t} 인덱스 불일치"


# =============================================================================
# 2) apply_transforms_panel
# =============================================================================
class TestApplyTransformsPanel:
    def test_per_column_transform_routing(self):
        # 두 컬럼에 서로 다른 변환을 매핑
        idx = pd.bdate_range("2020-01-02", periods=300)
        df = pd.DataFrame({
            "A": np.linspace(100, 200, 300),  # diff_6m
            "B": np.linspace(50, 150, 300),   # level
        }, index=idx)

        out = apply_transforms_panel(df, transforms={"A": "diff_6m", "B": "level"})
        assert list(out.columns) == ["A", "B"]
        # A: 워밍아웃 NaN
        assert out["A"].iloc[:HALFYEAR_TRADING_DAYS].isna().all()
        # B: level → 변환 없음, NaN 없음
        assert out["B"].notna().all()
        pd.testing.assert_series_equal(out["B"], df["B"])

    def test_missing_mapping_defaults_to_level(self):
        # transforms에 없는 컬럼은 'level'로 처리되어야 함
        idx = pd.bdate_range("2020-01-02", periods=100)
        df = pd.DataFrame({"X": np.arange(100, dtype=float)}, index=idx)
        out = apply_transforms_panel(df, transforms={})  # 매핑 비어있음
        pd.testing.assert_series_equal(out["X"], df["X"])

    def test_empty_dataframe(self):
        out = apply_transforms_panel(pd.DataFrame(), transforms={"X": "yoy_pct"})
        assert out.empty


# =============================================================================
# 3) Variable dataclass + yaml 파싱
# =============================================================================
class TestVariableTransformField:
    def test_default_is_level_when_yaml_missing(self):
        """transform 키가 없는 yaml entry는 'level' 기본값."""
        entry = {
            "code": "TEST_VAR",
            "source": "fred",
            "series_id": "X",
            "risk_direction": "positive",
            "enabled": True,
        }
        v = _parse_variable_entry(entry, section="variables")
        assert v.transform == "level"

    def test_explicit_value_parsed(self):
        entry = {
            "code": "TEST_VAR",
            "source": "fred",
            "risk_direction": "positive",
            "transform": "yoy_pct",
        }
        v = _parse_variable_entry(entry, section="variables")
        assert v.transform == "yoy_pct"

    def test_invalid_value_raises(self):
        entry = {
            "code": "TEST_VAR",
            "source": "fred",
            "risk_direction": "positive",
            "transform": "garbage",
        }
        with pytest.raises(ValueError, match="transform"):
            _parse_variable_entry(entry, section="variables")

    def test_dataclass_default(self):
        """dataclass 직접 생성 시 transform 기본값이 'level'."""
        v = Variable(code="ABC")
        assert v.transform == "level"

    def test_valid_transforms_set(self):
        assert VALID_TRANSFORMS == frozenset(
            {"level", "yoy_pct", "pct_change_6m", "diff_6m"}
        )


# =============================================================================
# 4) load_transform_map (실제 variables.yaml 기준)
# =============================================================================
class TestLoadTransformMap:
    def test_real_yaml_loads(self):
        """실제 variables.yaml에서 transform_map이 정상 로드되는지."""
        tmap = load_transform_map()
        # 최소한 enabled 변수가 1개 이상은 있어야 함
        assert len(tmap) > 0
        # 모든 값이 VALID_TRANSFORMS 안에 있어야 함
        for code, t in tmap.items():
            assert t in VALID_TRANSFORMS, f"{code}: 잘못된 transform {t!r}"

    def test_expected_assignments(self):
        """v13 의뢰서의 변수별 transform 매핑이 yaml에 반영되었는지 확인."""
        tmap = load_transform_map()
        expected = {
            "INDPRO": "yoy_pct",
            "KR_EXPORT": "level",
            "BDI": "pct_change_6m",
            "US_TIPS_10Y": "diff_6m",
            "US_BEI_10Y": "diff_6m",
            "US_POLICY_RATE_EXPECT": "diff_6m",
            "KR_US_RATE_DIFF": "diff_6m",
            "VIX": "level",
            "US_HY_SPREAD": "level",
            "DXY": "pct_change_6m",
            "BRENT": "pct_change_6m",
            "KRW_USD": "pct_change_6m",
            "JPY_USD": "pct_change_6m",
        }
        for code, t in expected.items():
            assert tmap.get(code) == t, (
                f"{code}: 기대 {t!r}, 실제 {tmap.get(code)!r}"
            )
