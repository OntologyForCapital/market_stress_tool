"""(v14) risk_direction 처리 및 양방향(bidirectional) 임계값 로직 테스트.

테스트 대상:
    - src.preprocessing.standardize.apply_risk_direction
    - src.preprocessing.standardize.standardize_panel (양방향 통합)
    - src.config.VALID_RISK_DIRECTIONS
    - src.config.load_bidirectional_thresholds / load_bidirectional_threshold_default
    - variables.yaml: BRENT는 positive_tail, US_BEI_10Y는 bidirectional + threshold=1.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_BIDIRECTIONAL_THRESHOLD,
    VALID_RISK_DIRECTIONS,
    load_bidirectional_threshold_default,
    load_bidirectional_thresholds,
    load_risk_directions,
    load_variables,
)
from src.preprocessing.standardize import (
    apply_risk_direction,
    standardize_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = PROJECT_ROOT / "config" / "variables.yaml"


# =============================================================================
# apply_risk_direction — 단위 테스트
# =============================================================================


class TestApplyRiskDirection:
    """apply_risk_direction 단위 동작."""

    def test_positive_passthrough(self):
        """positive: z 그대로 반환."""
        z = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = apply_risk_direction(z, "positive", threshold=1.0)
        pd.testing.assert_series_equal(out, z)

    def test_negative_flips_sign(self):
        """negative: -z 반환 (양수↔음수 교환)."""
        z = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = apply_risk_direction(z, "negative", threshold=1.0)
        expected = pd.Series([2.0, 0.5, -0.0, -0.5, -2.0])
        pd.testing.assert_series_equal(out, expected)

    def test_positive_tail_clips_benign_side(self):
        """positive_tail: 양수 z만 남기고 음수 z는 0."""
        z = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = apply_risk_direction(z, "positive_tail", threshold=1.0)
        expected = pd.Series([0.0, 0.0, 0.0, 0.5, 2.0])
        pd.testing.assert_series_equal(out, expected)

    def test_bidirectional_within_band_returns_zero(self):
        """bidirectional: |z| ≤ threshold 구간은 모두 0."""
        z = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0])
        out = apply_risk_direction(z, "bidirectional", threshold=1.0)
        expected = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(out, expected)

    def test_bidirectional_positive_tail(self):
        """bidirectional: z > threshold이면 z - threshold."""
        z = pd.Series([1.5, 2.0, 3.79])  # BRENT 2026-03 시나리오 포함
        out = apply_risk_direction(z, "bidirectional", threshold=1.0)
        expected = pd.Series([0.5, 1.0, 2.79])
        pd.testing.assert_series_equal(out, expected)

    def test_bidirectional_negative_tail(self):
        """bidirectional: z < -threshold이면 |z| - threshold (양수)."""
        z = pd.Series([-1.5, -2.98, -4.24])  # BRENT 코로나, BEI 코로나 시나리오
        out = apply_risk_direction(z, "bidirectional", threshold=1.0)
        expected = pd.Series([0.5, 1.98, 3.24])
        pd.testing.assert_series_equal(out, expected)

    def test_bidirectional_output_nonnegative(self):
        """bidirectional 결과는 항상 ≥ 0."""
        rng = np.random.default_rng(42)
        z = pd.Series(rng.standard_normal(200) * 2.0)
        out = apply_risk_direction(z, "bidirectional", threshold=1.0)
        assert (out >= 0).all()

    def test_bidirectional_custom_threshold(self):
        """threshold 값을 변경하면 밴드 폭이 달라짐."""
        z = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = apply_risk_direction(z, "bidirectional", threshold=1.5)
        # |z|≤1.5는 0, |z|>1.5는 |z|-1.5
        expected = pd.Series([0.5, 0.0, 0.0, 0.0, 0.5])
        pd.testing.assert_series_equal(out, expected)

    def test_bidirectional_default_threshold(self):
        """threshold 미지정 시 DEFAULT_BIDIRECTIONAL_THRESHOLD(1.0) 사용."""
        z = pd.Series([-1.5, 0.0, 1.5])
        out = apply_risk_direction(z, "bidirectional")
        expected = pd.Series([0.5, 0.0, 0.5])
        pd.testing.assert_series_equal(out, expected)

    def test_nan_propagation(self):
        """NaN은 모든 방향에서 NaN으로 전파."""
        z = pd.Series([np.nan, 1.5, np.nan])
        for direction in ("positive", "negative", "positive_tail", "bidirectional"):
            out = apply_risk_direction(z, direction, threshold=1.0)
            assert pd.isna(out.iloc[0])
            assert pd.isna(out.iloc[2])

    def test_invalid_direction_raises(self):
        """지원하지 않는 risk_direction은 ValueError."""
        z = pd.Series([0.0, 1.0])
        with pytest.raises(ValueError, match="risk_direction"):
            apply_risk_direction(z, "unknown")
        with pytest.raises(ValueError):
            apply_risk_direction(z, "")
        with pytest.raises(ValueError):
            apply_risk_direction(z, "POSITIVE")  # 대소문자 엄격

    def test_valid_directions_constant(self):
        """VALID_RISK_DIRECTIONS 상수에 4종이 포함."""
        assert VALID_RISK_DIRECTIONS == frozenset(
            {"positive", "negative", "positive_tail", "bidirectional"}
        )


# =============================================================================
# standardize_panel — bidirectional 통합 동작
# =============================================================================


class TestStandardizePanelBidirectional:
    """standardize_panel이 양방향 변수를 올바르게 처리하는지."""

    @staticmethod
    def _make_panel(seed: int = 0, n: int = 1500) -> pd.DataFrame:
        """충분히 긴 영업일 패널(롤링 5년 충족) 생성."""
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2018-01-01", periods=n)
        return pd.DataFrame(
            {
                "VIX": 20 + rng.standard_normal(n) * 5,
                "ISM_PMI": 50 + rng.standard_normal(n) * 3,
                "BRENT": 70 + rng.standard_normal(n) * 10,
                "US_BEI_10Y": 2.0 + rng.standard_normal(n) * 0.3,
            },
            index=idx,
        )

    def test_mixed_directions(self):
        """3가지 방향 혼합 시 각각 올바른 변환."""
        panel = self._make_panel()
        directions = {
            "VIX": "positive",
            "ISM_PMI": "negative",
            "BRENT": "positive_tail",
            "US_BEI_10Y": "bidirectional",
        }
        thresholds = {"US_BEI_10Y": 1.0}
        out = standardize_panel(
            panel, risk_directions=directions, bidirectional_thresholds=thresholds
        )

        valid = out.dropna()
        assert not valid.empty

        # positive_tail/bidirectional 결과는 모두 ≥ 0
        assert (valid["BRENT"] >= 0).all()
        assert (valid["US_BEI_10Y"] >= 0).all()

        # BRENT 하락 꼬리는 0, BEI 밴드 내 평시(|z|≤1)는 0
        assert (valid["BRENT"] == 0).any()
        assert (valid["US_BEI_10Y"] == 0).any()

        # positive/negative는 음수도 정상적으로 나타남
        assert (valid["VIX"] < 0).any() and (valid["VIX"] > 0).any()
        assert (valid["ISM_PMI"] < 0).any() and (valid["ISM_PMI"] > 0).any()

    def test_missing_threshold_uses_default(self):
        """bidirectional_thresholds에 없는 변수는 DEFAULT(1.0) 사용."""
        panel = self._make_panel(seed=1)
        directions = {"US_BEI_10Y": "bidirectional"}
        # thresholds 매핑이 None → DEFAULT 사용
        out_default = standardize_panel(
            panel[["US_BEI_10Y"]],
            risk_directions=directions,
            bidirectional_thresholds=None,
        )
        out_explicit = standardize_panel(
            panel[["US_BEI_10Y"]],
            risk_directions=directions,
            bidirectional_thresholds={"US_BEI_10Y": DEFAULT_BIDIRECTIONAL_THRESHOLD},
        )
        pd.testing.assert_series_equal(
            out_default["US_BEI_10Y"].dropna(),
            out_explicit["US_BEI_10Y"].dropna(),
        )

    def test_custom_threshold_changes_signal(self):
        """더 큰 threshold는 더 작은 신호(또는 0)를 산출."""
        panel = self._make_panel(seed=2)
        directions = {"US_BEI_10Y": "bidirectional"}
        out_1 = standardize_panel(
            panel[["US_BEI_10Y"]],
            risk_directions=directions,
            bidirectional_thresholds={"US_BEI_10Y": 1.0},
        )
        out_2 = standardize_panel(
            panel[["US_BEI_10Y"]],
            risk_directions=directions,
            bidirectional_thresholds={"US_BEI_10Y": 2.0},
        )
        v1 = out_1["US_BEI_10Y"].dropna()
        v2 = out_2["US_BEI_10Y"].dropna().reindex(v1.index)
        # threshold 2.0 신호는 항상 1.0 신호 이하
        assert (v2 <= v1 + 1e-12).all()

    def test_invalid_direction_in_panel_raises(self):
        """패널 표준화에서 잘못된 방향이 있으면 ValueError 전파."""
        panel = self._make_panel(seed=3)
        directions = {"VIX": "positive", "BRENT": "wrong_direction"}
        with pytest.raises(ValueError, match="risk_direction"):
            standardize_panel(panel[["VIX", "BRENT"]], risk_directions=directions)

    def test_unknown_column_defaults_to_positive(self):
        """risk_directions 매핑에 없는 컬럼은 기본 positive로 처리."""
        panel = self._make_panel(seed=4)
        # 매핑 비어 있음 → 모두 positive로 동작
        out = standardize_panel(
            panel[["VIX"]],
            risk_directions={},
            bidirectional_thresholds=None,
        )
        # rolling_zscore 결과(positive 그대로)와 일치해야 함 → 음/양수 모두 존재
        valid = out["VIX"].dropna()
        assert (valid < 0).any() and (valid > 0).any()


# =============================================================================
# yaml 로딩 — 전역 기본값 / 변수별 오버라이드
# =============================================================================


class TestYamlBidirectionalLoading:
    """variables.yaml로부터 양방향 임계값을 올바르게 로드."""

    def test_default_threshold_from_yaml(self):
        """thresholds.bidirectional_threshold_default 값을 읽음."""
        val = load_bidirectional_threshold_default(YAML_PATH)
        assert val == 1.0  # v14 yaml 명시값

    def test_default_threshold_fallback_when_missing(self, tmp_path):
        """yaml에 키 없으면 DEFAULT_BIDIRECTIONAL_THRESHOLD."""
        yaml_text = (
            "variables: []\n"
            "target_variables: []\n"
            "channel_weights: {}\n"
            "thresholds: {}\n"
        )
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        assert load_bidirectional_threshold_default(p) == DEFAULT_BIDIRECTIONAL_THRESHOLD

    def test_default_threshold_fallback_when_invalid(self, tmp_path):
        """음수/문자열 등 잘못된 값이면 DEFAULT 사용."""
        yaml_text = (
            "variables: []\n"
            "target_variables: []\n"
            "channel_weights: {}\n"
            "thresholds:\n"
            "  bidirectional_threshold_default: -1.0\n"
        )
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        assert load_bidirectional_threshold_default(p) == DEFAULT_BIDIRECTIONAL_THRESHOLD

    def test_load_bidirectional_thresholds_includes_bei_only(self):
        """v22 yaml: US_BEI_10Y만 bidirectional 매핑에 포함."""
        thresholds = load_bidirectional_thresholds(YAML_PATH)
        assert "BRENT" not in thresholds
        assert "US_BEI_10Y" in thresholds
        assert thresholds["US_BEI_10Y"] == 1.0

    def test_load_bidirectional_thresholds_excludes_others(self):
        """positive/negative 변수는 thresholds 매핑에서 제외."""
        thresholds = load_bidirectional_thresholds(YAML_PATH)
        directions = load_risk_directions(YAML_PATH)
        for code, direction in directions.items():
            if direction != "bidirectional":
                assert code not in thresholds, (
                    f"{code}는 {direction}인데 thresholds에 포함됨"
                )

    def test_per_variable_override(self, tmp_path):
        """yaml의 변수별 bidirectional_threshold가 전역 기본보다 우선."""
        yaml_text = """
variables:
  - code: A_VAR
    name_kr: 테스트 A
    name_en: Test A
    source: test
    series_id: A
    channel: 1
    risk_direction: bidirectional
    bidirectional_threshold: 2.5
    unit: pct
    frequency: daily
    enabled: true
  - code: B_VAR
    name_kr: 테스트 B
    name_en: Test B
    source: test
    series_id: B
    channel: 1
    risk_direction: bidirectional
    unit: pct
    frequency: daily
    enabled: true
target_variables: []
channel_weights: {1: 1.0}
thresholds:
  bidirectional_threshold_default: 1.5
"""
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        thresholds = load_bidirectional_thresholds(p)
        assert thresholds["A_VAR"] == 2.5  # 개별 오버라이드
        assert thresholds["B_VAR"] == 1.5  # 전역 기본값

    def test_brent_positive_tail_and_bei_bidirectional_in_yaml(self):
        """v22 yaml: BRENT는 positive_tail, US_BEI_10Y는 bidirectional."""
        directions = load_risk_directions(YAML_PATH)
        assert directions.get("BRENT") == "positive_tail"
        assert directions.get("US_BEI_10Y") == "bidirectional"

    def test_other_variables_directions_preserved(self):
        """v14에서 KR_US_RATE_DIFF, DXY 등 다른 변수의 방향은 유지."""
        directions = load_risk_directions(YAML_PATH)
        # 명세: KR_US_RATE_DIFF, DXY는 변경 없음 (bidirectional 아님)
        if "KR_US_RATE_DIFF" in directions:
            assert directions["KR_US_RATE_DIFF"] != "bidirectional"
        if "DXY" in directions:
            assert directions["DXY"] != "bidirectional"

    def test_variable_bidirectional_threshold_field(self):
        """Variable 데이터클래스가 bidirectional_threshold 필드를 가지고
        파싱이 동작한다."""
        variables = {v.code: v for v in load_variables(YAML_PATH)}
        brent = variables.get("BRENT")
        bei = variables.get("US_BEI_10Y")
        assert brent is not None and bei is not None
        assert brent.bidirectional_threshold is None
        assert bei.bidirectional_threshold == 1.0
        # bidirectional이 아닌 변수는 None
        for code, v in variables.items():
            if v.risk_direction != "bidirectional":
                assert v.bidirectional_threshold is None, (
                    f"{code} (direction={v.risk_direction})에 bidirectional_threshold가 설정됨"
                )
