"""(v15) 종합점수 산출 함수 L2 norm 적용 테스트.

테스트 대상:
    - src.analysis.stress_index.compute_composite_score (신규, 단일 시점)
    - src.analysis.stress_index.compute_composite_index (method 분기 추가)
    - src.analysis.stress_index.build_stress_index_table (composite_method 전달)
    - src.config.VALID_COMPOSITE_METHODS / DEFAULT_COMPOSITE_METHOD
    - src.config.load_composite_method
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.stress_index import (
    build_stress_index_table,
    compute_composite_index,
    compute_composite_score,
)
from src.config import (
    DEFAULT_COMPOSITE_METHOD,
    VALID_COMPOSITE_METHODS,
    load_composite_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = PROJECT_ROOT / "config" / "variables.yaml"


# =============================================================================
# compute_composite_score — 단일 시점 단위 테스트 (의뢰서 명시 케이스)
# =============================================================================


class TestComputeCompositeScore:
    """compute_composite_score 단위 동작."""

    def test_default_method_is_l2_norm(self):
        """기본 method는 l2_norm이어야 한다 (v15)."""
        assert DEFAULT_COMPOSITE_METHOD == "l2_norm"

    def test_valid_methods_constant(self):
        """허용 값 집합이 정확히 {mean, l2_norm}."""
        assert VALID_COMPOSITE_METHODS == frozenset({"mean", "l2_norm"})

    def test_mean_all_ones(self):
        """mean: 모든 채널 1.0 → 1.0."""
        result = compute_composite_score(
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}, "mean"
        )
        assert result == pytest.approx(1.0)

    def test_l2_norm_all_ones(self):
        """l2_norm: 모든 채널 1.0 → 1.0 (대각선 케이스)."""
        result = compute_composite_score(
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}, "l2_norm"
        )
        assert result == pytest.approx(1.0)

    def test_mean_offsetting_cancels(self):
        """mean: {2, -2} → 0.0 (상쇄)."""
        result = compute_composite_score({1: 2.0, 2: -2.0}, "mean")
        assert result == pytest.approx(0.0)

    def test_l2_norm_offsetting_preserved(self):
        """l2_norm: {2, -2} → 2.0 (상쇄 안 됨, 부호 무관)."""
        result = compute_composite_score({1: 2.0, 2: -2.0}, "l2_norm")
        assert result == pytest.approx(2.0)

    def test_l2_norm_always_nonnegative(self):
        """l2_norm 결과는 어떤 입력에서도 항상 ≥0."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            sample = {i + 1: rng.standard_normal() * 3 for i in range(5)}
            result = compute_composite_score(sample, "l2_norm")
            assert result >= 0

    def test_nan_handling_l2_norm(self):
        """NaN은 제외하고 n 조정 — {1.0, NaN, 1.0}, l2_norm → 1.0."""
        result = compute_composite_score(
            {1: 1.0, 2: float("nan"), 3: 1.0}, "l2_norm"
        )
        assert result == pytest.approx(1.0)

    def test_nan_handling_mean(self):
        """NaN은 제외하고 n 조정 — {3.0, NaN, 1.0}, mean → 2.0."""
        result = compute_composite_score(
            {1: 3.0, 2: float("nan"), 3: 1.0}, "mean"
        )
        assert result == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        """모든 채널이 NaN이면 NaN."""
        result = compute_composite_score(
            {1: float("nan"), 2: float("nan")}, "l2_norm"
        )
        assert math.isnan(result)

    def test_empty_input_returns_nan(self):
        """빈 매핑 → NaN."""
        assert math.isnan(compute_composite_score({}, "l2_norm"))
        assert math.isnan(compute_composite_score({}, "mean"))

    def test_invalid_method_raises(self):
        """잘못된 method는 ValueError."""
        with pytest.raises(ValueError, match="composite_method"):
            compute_composite_score({1: 1.0}, "unknown")
        with pytest.raises(ValueError):
            compute_composite_score({1: 1.0}, "RMS")  # 대소문자 엄격
        with pytest.raises(ValueError):
            compute_composite_score({1: 1.0}, "")

    def test_pandas_series_input(self):
        """pd.Series 입력도 지원."""
        s = pd.Series({"S1": 2.0, "S2": -2.0})
        assert compute_composite_score(s, "l2_norm") == pytest.approx(2.0)
        assert compute_composite_score(s, "mean") == pytest.approx(0.0)


# =============================================================================
# 의뢰서 백테스트 표 재현 (v14 채널 점수 → v15 종합점수)
# =============================================================================


class TestBacktestReproduction:
    """의뢰서에 명시된 백테스트 예상값을 정확히 재현."""

    BACKTEST_CASES = [
        # (이름, [S1, S2, S3, S4, S5], 평균(v14), L2 norm(v15))
        ("China_shock", [2.51, 0.79, 1.96, 0.61, 0.60], 1.29, 1.51),
        ("미중",         [-0.64, 0.89, 0.79, 0.00, 0.76], 0.36, 0.69),
        ("코로나",       [2.01, -1.23, 4.58, 1.98, 0.68], 1.61, 2.49),
        ("Fed_매파",     [-0.28, 0.04, 0.46, 0.00, 1.37], 0.32, 0.66),
        ("2022-07",      [0.01, 1.83, 1.01, 0.00, 2.52], 1.07, 1.50),
        ("영국",         [0.04, 2.79, 1.30, 0.00, 2.83], 1.39, 1.89),
        ("2024-08",      [0.12, -0.26, 0.64, 0.00, -0.47], 0.01, 0.39),
        ("트럼프",       [0.17, -0.24, 1.01, 0.00, 0.07], 0.20, 0.47),
        ("중동",         [0.30, -0.48, 0.70, 2.79, 0.84], 0.83, 1.36),
    ]

    @pytest.mark.parametrize("name,channels,exp_mean,exp_l2", BACKTEST_CASES)
    def test_backtest_values(self, name, channels, exp_mean, exp_l2):
        """의뢰서 표 재현 — 평균/l2_norm 값 일치.

        의뢰서 기대값은 소수점 둘째자리로 반올된 표시값이므로
        절대 오차 0.05 허용 (예: 1.4642 → 1.50, 0.3775 → 0.39).
        """
        cs = {i + 1: v for i, v in enumerate(channels)}
        mean_val = compute_composite_score(cs, "mean")
        l2_val = compute_composite_score(cs, "l2_norm")
        assert mean_val == pytest.approx(exp_mean, abs=0.05), name
        assert l2_val == pytest.approx(exp_l2, abs=0.05), name

    def test_l2_norm_always_ge_abs_mean(self):
        """수학적 사실: RMS ≥ |mean|. 모든 백테스트 케이스에서 성립."""
        for name, channels, _, _ in self.BACKTEST_CASES:
            cs = {i + 1: v for i, v in enumerate(channels)}
            mean_val = compute_composite_score(cs, "mean")
            l2_val = compute_composite_score(cs, "l2_norm")
            assert l2_val >= abs(mean_val) - 1e-9, (
                f"{name}: l2={l2_val:.3f} vs |mean|={abs(mean_val):.3f}"
            )


# =============================================================================
# compute_composite_index — 시계열 집계 동작
# =============================================================================


class TestComputeCompositeIndex:
    """패널 단위 compute_composite_index 동작."""

    @staticmethod
    def _make_channel_panel(n: int = 10) -> pd.DataFrame:
        idx = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame(
            {
                "S1": [1.0, -1.0, 2.0, np.nan, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                "S2": [1.0, 1.0, -2.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                "S3": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                "S4": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
                "S5": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
            },
            index=idx,
        )

    def test_mean_default_weights(self):
        """동일 가중치 mean: 행별 평균."""
        panel = self._make_channel_panel()
        out = compute_composite_index(panel, channel_weights=None, method="mean")
        # 첫 행: 모두 1 → 1.0
        assert out.iloc[0] == pytest.approx(1.0)
        # 둘째 행: (-1+1+1+1+1)/5 = 0.6
        assert out.iloc[1] == pytest.approx(0.6)

    def test_l2_norm_default_weights(self):
        """동일 가중치 l2_norm: RMS."""
        panel = self._make_channel_panel()
        out = compute_composite_index(panel, channel_weights=None, method="l2_norm")
        # 첫 행: sqrt(5/5) = 1.0
        assert out.iloc[0] == pytest.approx(1.0)
        # 둘째 행: sqrt((1+1+1+1+1)/5) = 1.0 (부호 무관)
        assert out.iloc[1] == pytest.approx(1.0)
        # 셋째 행: sqrt((4+4+0+0+0)/5) = sqrt(1.6) ≈ 1.2649
        assert out.iloc[2] == pytest.approx(math.sqrt(1.6))

    def test_l2_norm_nonnegative(self):
        """l2_norm 시계열은 항상 ≥0 (NaN 제외)."""
        panel = self._make_channel_panel()
        out = compute_composite_index(panel, channel_weights=None, method="l2_norm")
        assert (out.dropna() >= 0).all()

    def test_nan_channel_reweighting_l2(self):
        """l2_norm: NaN 채널은 가중치 재정규화로 제외 — 넷째 행 (S1=NaN)."""
        panel = self._make_channel_panel()
        out = compute_composite_index(panel, channel_weights=None, method="l2_norm")
        # 넷째 행: S1=NaN 제외, 나머지 4개 (1,1,1,1) → sqrt(4/4) = 1.0
        assert out.iloc[3] == pytest.approx(1.0)

    def test_custom_weights_l2_norm(self):
        """채널 가중치 적용 — sqrt(Σ w_k * S_k²)."""
        idx = pd.bdate_range("2020-01-01", periods=1)
        panel = pd.DataFrame(
            {"S1": [2.0], "S2": [0.0], "S3": [0.0], "S4": [0.0], "S5": [0.0]},
            index=idx,
        )
        # 가중치 S1=0.5, 나머지 0.125씩
        weights = {1: 0.5, 2: 0.125, 3: 0.125, 4: 0.125, 5: 0.125}
        out = compute_composite_index(panel, channel_weights=weights, method="l2_norm")
        # sqrt(0.5*4 + 0+0+0+0) / sqrt(0.5+0.125*4) ... 코드는 w_eff 합으로 정규화
        # w_eff_sum = 1.0, sqrt(0.5*4/1.0) = sqrt(2) ≈ 1.414
        assert out.iloc[0] == pytest.approx(math.sqrt(2.0))

    def test_invalid_method_raises(self):
        """잘못된 method는 ValueError."""
        panel = self._make_channel_panel()
        with pytest.raises(ValueError, match="composite_method"):
            compute_composite_index(panel, method="bogus")

    def test_empty_panel(self):
        """빈 패널은 빈 Series."""
        empty = pd.DataFrame()
        out = compute_composite_index(empty, method="l2_norm")
        assert out.empty
        assert out.name == "composite"

    def test_method_changes_result(self):
        """동일 입력에서 mean과 l2_norm은 다른 결과를 낸다 (부호 상쇄 케이스)."""
        idx = pd.bdate_range("2020-01-01", periods=1)
        panel = pd.DataFrame(
            {"S1": [2.0], "S2": [-2.0], "S3": [0.0], "S4": [0.0], "S5": [0.0]},
            index=idx,
        )
        mean_out = compute_composite_index(panel, method="mean").iloc[0]
        l2_out = compute_composite_index(panel, method="l2_norm").iloc[0]
        assert mean_out == pytest.approx(0.0)
        assert l2_out == pytest.approx(math.sqrt(8.0 / 5.0))
        assert l2_out > mean_out


# =============================================================================
# build_stress_index_table 통합 — composite_method 전달 확인
# =============================================================================


class TestBuildStressIndexTable:
    """build_stress_index_table에 composite_method가 올바르게 전달되는지."""

    def test_l2_norm_path(self):
        """method=l2_norm 전달 시 composite ≥0."""
        z_panel = pd.DataFrame(
            {
                "VIX": [2.0, -2.0, 1.0],
                "ISM_PMI": [1.0, 1.0, 1.0],
            },
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        var_to_ch = {"VIX": 1, "ISM_PMI": 2}
        weights = {1: 0.5, 2: 0.5}
        table = build_stress_index_table(
            z_panel,
            variable_to_channel=var_to_ch,
            channel_weights=weights,
            composite_method="l2_norm",
        )
        assert "composite" in table.columns
        assert (table["composite"].dropna() >= 0).all()

    def test_mean_vs_l2_differ(self):
        """동일 입력에서 mean과 l2_norm은 명확히 다른 composite를 생성."""
        z_panel = pd.DataFrame(
            {"VIX": [3.0], "ISM_PMI": [-3.0]},
            index=pd.bdate_range("2020-01-01", periods=1),
        )
        var_to_ch = {"VIX": 1, "ISM_PMI": 2}
        weights = {1: 0.5, 2: 0.5}
        t_mean = build_stress_index_table(
            z_panel, var_to_ch, weights, composite_method="mean"
        )
        t_l2 = build_stress_index_table(
            z_panel, var_to_ch, weights, composite_method="l2_norm"
        )
        # 부호 상쇄로 mean은 0, l2는 3.0
        assert t_mean["composite"].iloc[0] == pytest.approx(0.0)
        assert t_l2["composite"].iloc[0] == pytest.approx(3.0)

    def test_default_method_is_l2_norm(self):
        """composite_method 미지정 시 l2_norm으로 동작."""
        z_panel = pd.DataFrame(
            {"VIX": [3.0], "ISM_PMI": [-3.0]},
            index=pd.bdate_range("2020-01-01", periods=1),
        )
        var_to_ch = {"VIX": 1, "ISM_PMI": 2}
        weights = {1: 0.5, 2: 0.5}
        t_default = build_stress_index_table(z_panel, var_to_ch, weights)
        t_l2 = build_stress_index_table(
            z_panel, var_to_ch, weights, composite_method="l2_norm"
        )
        assert t_default["composite"].iloc[0] == pytest.approx(
            t_l2["composite"].iloc[0]
        )


# =============================================================================
# yaml 로딩 — load_composite_method
# =============================================================================


class TestLoadCompositeMethod:
    """variables.yaml로부터 composite_method 로드."""

    def test_v15_yaml_returns_l2_norm(self):
        """v15 yaml에는 composite_method: l2_norm이 명시되어 있어야 함."""
        method = load_composite_method(YAML_PATH)
        assert method == "l2_norm"

    def test_missing_key_returns_default(self, tmp_path):
        """thresholds.composite_method 누락 시 DEFAULT_COMPOSITE_METHOD."""
        yaml_text = (
            "variables: []\n"
            "target_variables: []\n"
            "channel_weights: {}\n"
            "thresholds:\n"
            "  variable_alert: 1.5\n"
        )
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        assert load_composite_method(p) == DEFAULT_COMPOSITE_METHOD

    def test_empty_thresholds_section(self, tmp_path):
        """thresholds 섹션 자체가 없거나 비어 있어도 기본값."""
        yaml_text = "variables: []\ntarget_variables: []\nchannel_weights: {}\n"
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        assert load_composite_method(p) == DEFAULT_COMPOSITE_METHOD

    def test_explicit_mean(self, tmp_path):
        """yaml에 mean 명시 시 그대로."""
        yaml_text = (
            "variables: []\n"
            "target_variables: []\n"
            "channel_weights: {}\n"
            "thresholds:\n"
            "  composite_method: mean\n"
        )
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        assert load_composite_method(p) == "mean"

    def test_invalid_value_raises(self, tmp_path):
        """yaml에 잘못된 값이 있으면 ValueError."""
        yaml_text = (
            "variables: []\n"
            "target_variables: []\n"
            "channel_weights: {}\n"
            "thresholds:\n"
            "  composite_method: harmonic_mean\n"
        )
        p = tmp_path / "vars.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(ValueError, match="composite_method"):
            load_composite_method(p)
