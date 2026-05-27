"""src/pipeline.py 테스트 (mock loader 기반).

실제 외부 API를 호출하지 않고, dispatcher.fetch_all_variables를 mock으로
대체하여 `run_full_diagnosis()`의 흐름과 DiagnosisResult 조립을 검증.

테스트 전략:
    - variables.yaml의 실제 변수 코드를 사용하여 합성 시계열을 생성
    - `_fetch_all_variables` 의존성 주입으로 외부 호출 차단
    - 가벼운 통합 테스트 (정합/표준화/지수/패턴/유사 시점/진원지까지 한 번에)

[한계]
    실제 데이터의 통계적 특성을 재현하지는 않음. 파이프라인이 끝까지
    돌고 DiagnosisResult 필드가 채워지는지가 주된 검증 대상.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from src.config import Variable, load_target_variables, load_variables
from src.pipeline import (
    DiagnosisResult,
    run_full_diagnosis,
    z_to_percentile,
)


# =============================================================================
# 1. z_to_percentile 헬퍼 테스트
# =============================================================================
class TestZToPercentile:
    def test_z_zero_is_fifty(self):
        assert z_to_percentile(0) == 50.0

    def test_z_positive_25_is_hundred(self):
        assert z_to_percentile(2.5) == 100.0

    def test_z_negative_25_is_zero(self):
        assert z_to_percentile(-2.5) == 0.0

    def test_z_above_25_clipped_to_hundred(self):
        assert z_to_percentile(5.0) == 100.0
        assert z_to_percentile(10.0) == 100.0

    def test_z_below_neg25_clipped_to_zero(self):
        assert z_to_percentile(-5.0) == 0.0

    def test_intermediate_value(self):
        # z=1 → 70
        assert z_to_percentile(1.0) == 70.0
        # z=-1 → 30
        assert z_to_percentile(-1.0) == 30.0

    def test_nan_passthrough(self):
        assert math.isnan(z_to_percentile(float("nan")))


# =============================================================================
# 2. Mock loader 헬퍼
# =============================================================================
def _make_synthetic_series(
    code: str,
    start: str,
    end: str,
    freq: str = "B",
    seed: int = 0,
    trend: float = 0.0,
    base: float = 100.0,
    noise_scale: float = 1.0,
) -> pd.Series:
    """합성 시계열 생성. KOSPI 같이 가격 시계열은 기하 누적, 나머지는 노이즈."""
    rng = np.random.default_rng(seed + abs(hash(code)) % 10_000)
    idx = pd.date_range(start=start, end=end, freq=freq)
    if len(idx) == 0:
        return pd.Series(dtype="float64", name=code)
    noise = rng.normal(0.0, noise_scale, size=len(idx))
    drift = trend * np.arange(len(idx))
    values = base + drift + noise.cumsum() * 0.5
    s = pd.Series(values, index=idx, name=code, dtype="float64")
    s.index.name = "date"
    return s


def _make_mock_fetch(
    series_map: dict[str, pd.Series],
    fail_codes: Iterable[str] = (),
):
    """fetch_all_variables 시그니처를 따르는 mock 함수 생성.

    Args:
        series_map: 반환할 {code: Series} dict.
        fail_codes: 결과에서 제외할 코드 (실패 시뮬레이션).
    """
    fail_set = set(fail_codes)

    def _mock(variables, start_date, end_date, use_cache=True, include_disabled=False):
        result: dict[str, pd.Series] = {}
        for v in variables:
            if v.code in fail_set:
                continue
            if v.code in series_map:
                result[v.code] = series_map[v.code]
        return result

    return _mock


# =============================================================================
# 3. run_full_diagnosis 통합 테스트 (mock)
# =============================================================================
class TestRunFullDiagnosis:
    """변수 6년치 합성 데이터를 주입하여 파이프라인이 끝까지 도는지 검증."""

    @pytest.fixture
    def synthetic_series_map(self) -> dict[str, pd.Series]:
        """variables.yaml의 활성 변수 + target 변수에 대한 합성 시계열."""
        start, end = "2020-01-01", "2026-05-22"

        # 실제 yaml에서 변수 코드 목록을 가져옴
        all_vars = list(load_variables())
        target_vars = list(load_target_variables())

        smap: dict[str, pd.Series] = {}
        for i, v in enumerate(all_vars + target_vars):
            # KOSPI/KOSDAQ은 가격처럼 양수 + 약한 추세
            if v.code in ("KOSPI", "KOSDAQ"):
                s = _make_synthetic_series(
                    v.code, start, end,
                    seed=i, trend=0.05, base=2500.0, noise_scale=10.0,
                )
            else:
                s = _make_synthetic_series(
                    v.code, start, end,
                    seed=i, trend=0.0, base=50.0, noise_scale=2.0,
                )
            smap[v.code] = s
        return smap

    def test_pipeline_runs_end_to_end(self, synthetic_series_map):
        """전체 파이프라인이 예외 없이 끝까지 실행되고 DiagnosisResult를 반환."""
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        assert isinstance(result, DiagnosisResult)

    def test_result_fields_populated(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # as_of_date는 Timestamp
        assert isinstance(result.as_of_date, pd.Timestamp)
        # composite_score / percentile은 float
        assert isinstance(result.composite_score, float)
        assert isinstance(result.composite_percentile, float)
        # 백분위는 0~100 범위
        if not math.isnan(result.composite_percentile):
            assert 0.0 <= result.composite_percentile <= 100.0
        # 원자료 패널은 정합/표준화 전 값 확인용으로 보존
        assert not result.raw_panel.empty

    def test_pattern_label_is_valid_key(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        valid_patterns = {
            "normal", "rate_shock", "risk_premium_shock",
            "supply_shock", "real_recession", "system_crisis",
        }
        assert result.pattern_label in valid_patterns

    def test_channel_scores_keys_1_to_5(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # 5개 채널 모두 존재
        assert set(result.channel_scores.keys()) == {1, 2, 3, 4, 5}
        assert set(result.channel_percentiles.keys()) == {1, 2, 3, 4, 5}
        # 백분위는 0~100 범위 (NaN 허용)
        for ch, pct in result.channel_percentiles.items():
            if not math.isnan(pct):
                assert 0.0 <= pct <= 100.0

    def test_variable_z_scores_for_standardized_vars(self, synthetic_series_map):
        """target_variables는 z_panel에 포함되지 않으므로 variable_z_scores에도 없어야 함."""
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # KOSPI/KOSDAQ는 표준화 대상에서 제외
        assert "KOSPI" not in result.variable_z_scores
        assert "KOSDAQ" not in result.variable_z_scores
        # 일반 변수(예: VIX)는 포함
        assert "VIX" in result.variable_z_scores or "INDPRO" in result.variable_z_scores

    def test_similar_dates_has_required_columns(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        if not result.similar_dates.empty:
            cols = set(result.similar_dates.columns)
            # k-NN 결과 컬럼
            assert {"distance", "S1", "S2", "S3", "S4", "S5"}.issubset(cols)
            # forward returns 컬럼
            assert {"fwd_30d", "fwd_90d", "fwd_180d"}.issubset(cols)

    def test_forward_returns_summary_horizons(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # 30/90/180일 horizon 모두 키가 있어야 함
        assert set(result.forward_returns_summary.keys()) == {30, 90, 180}
        for h, stats in result.forward_returns_summary.items():
            assert "avg" in stats
            assert "median" in stats

    def test_data_period_within_request(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of=None,  # 자동: 마지막 영업일
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        period_start, period_end = result.data_period
        assert period_start is not None
        assert period_end is not None
        assert period_start >= pd.Timestamp("2020-01-01")
        assert period_end <= pd.Timestamp("2026-05-22")

    def test_as_of_none_uses_last_business_day(self, synthetic_series_map):
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of=None,
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # as_of_date는 data_period 안에 있어야 함
        assert result.data_period[0] <= result.as_of_date <= result.data_period[1]

    def test_failed_variables_tracked(self, synthetic_series_map):
        """일부 변수 수집을 실패시켰을 때 failed_variables에 기록되는지."""
        # VIX와 INDPRO 수집 실패 시뮬레이션
        mock_fetch = _make_mock_fetch(synthetic_series_map, fail_codes=["VIX", "INDPRO"])
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # 실패 변수가 두 개 모두 기록
        assert "VIX" in result.failed_variables
        assert "INDPRO" in result.failed_variables

    def test_origin_result_present(self, synthetic_series_map):
        """OriginResult 객체가 origin_result에 들어가는지."""
        mock_fetch = _make_mock_fetch(synthetic_series_map)
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2026-05-22",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        from src.analysis.origin_tracking import OriginResult
        assert isinstance(result.origin_result, OriginResult)

    def test_raises_when_no_variables_collected(self, synthetic_series_map):
        """모든 변수 수집 실패 시 ValueError."""
        # 모든 코드를 실패시킴
        all_codes = list(synthetic_series_map.keys())
        mock_fetch = _make_mock_fetch(synthetic_series_map, fail_codes=all_codes)
        with pytest.raises(ValueError, match="수집된 변수가 0개"):
            run_full_diagnosis(
                start_date="2020-01-01",
                end_date="2026-05-22",
                as_of="2026-05-22",
                use_cache=False,
                _fetch_all_variables=mock_fetch,
            )


# =============================================================================
# 4. 부분 데이터 / 엣지 케이스
# =============================================================================
class TestEdgeCases:
    def test_as_of_before_data_falls_back(self):
        """as_of가 데이터 시작 이전이면 첫 영업일로 폴백 (경고 후 진행)."""
        all_vars = list(load_variables())
        target_vars = list(load_target_variables())
        smap: dict[str, pd.Series] = {}
        for i, v in enumerate(all_vars + target_vars):
            if v.code in ("KOSPI", "KOSDAQ"):
                smap[v.code] = _make_synthetic_series(
                    v.code, "2020-01-01", "2026-05-22",
                    seed=i, base=2500.0, noise_scale=10.0,
                )
            else:
                smap[v.code] = _make_synthetic_series(
                    v.code, "2020-01-01", "2026-05-22", seed=i,
                )
        mock_fetch = _make_mock_fetch(smap)
        # as_of를 데이터 시작 이전으로 지정
        result = run_full_diagnosis(
            start_date="2020-01-01",
            end_date="2026-05-22",
            as_of="2018-01-01",
            use_cache=False,
            _fetch_all_variables=mock_fetch,
        )
        # 폴백으로 인해 as_of_date는 실제 데이터 첫 영업일
        assert result.as_of_date >= pd.Timestamp("2020-01-01")
