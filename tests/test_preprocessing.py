"""전처리 모듈 단위 테스트.

더미 데이터로 다음을 검증:
    1. alignment
       - 일별 거래일 인덱스 정렬
       - forward fill 한도 적용
       - 빈 시리즈 처리
       - 서로 다른 주기(일별 + 월별) 혼합
    2. standardize
       - z-score 평균 0, 표준편차 1에 근접
       - 위험방향 negative 변수는 부호 반전
       - 롤링 윈도우 부족 구간은 NaN
       - 상수 시리즈는 NaN (분모 0 회피)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.alignment import align_series, drop_long_gap_periods
from src.preprocessing.standardize import (
    rolling_percentile_rank,
    rolling_robust_zscore,
    rolling_zscore,
    standardize_panel,
    load_risk_directions_from_yaml,
    TRADING_DAYS_PER_YEAR,
)


# =============================================================================
# 더미 데이터 생성 헬퍼
# =============================================================================
def make_daily_series(start: str, end: str, value: float = 1.0, name: str = "X") -> pd.Series:
    """일별(달력일) 인덱스의 상수 시리즈."""
    idx = pd.date_range(start, end, freq="D")
    return pd.Series(value, index=idx, name=name)


def make_random_walk(start: str, end: str, seed: int = 42, name: str = "X") -> pd.Series:
    """거래일 인덱스의 랜덤워크."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="B")
    steps = rng.normal(0, 1, size=len(idx))
    return pd.Series(steps.cumsum(), index=idx, name=name)


def make_monthly_series(start: str, end: str, values=None, name: str = "M") -> pd.Series:
    """월말 발표 시리즈 (월별 인덱스)."""
    idx = pd.date_range(start, end, freq="ME")  # MonthEnd
    if values is None:
        values = np.arange(len(idx), dtype=float)
    return pd.Series(values, index=idx, name=name)


# =============================================================================
# alignment 테스트
# =============================================================================
class TestAlignment:

    def test_empty_dict_returns_empty_df(self):
        """[Unit] 빈 입력은 빈 DataFrame."""
        out = align_series({}, "2024-01-01", "2024-01-31")
        assert out.empty

    def test_single_daily_series_business_days(self):
        """[Unit] 일별 시리즈가 'B' freq로 정렬되어 주말이 제거된다."""
        s = make_daily_series("2024-01-01", "2024-01-14", value=10.0)
        out = align_series({"X": s}, "2024-01-01", "2024-01-14")
        # 2024-01-01 (월) ~ 2024-01-14 (일) 사이 거래일: 1,2,3,4,5,8,9,10,11,12 = 10일
        assert len(out) == 10
        # 모든 값이 10.0
        assert (out["X"] == 10.0).all()
        # 주말이 포함되지 않음
        assert (out.index.dayofweek < 5).all()

    def test_ffill_limit_enforced(self):
        """[Unit] forward fill이 limit 일수까지만 적용된다."""
        # 1월 1일에만 값 있는 시리즈
        s = pd.Series([100.0], index=[pd.Timestamp("2024-01-01")], name="X")
        out = align_series({"X": s}, "2024-01-01", "2024-03-31", ffill_limit=10)
        # 1월 1일 (월요일) 이후 10거래일까지 채워지고, 그 다음부터 NaN
        # 2024-01-01 (월) 부터 10거래일째 = 2024-01-12 (금)까지 채워짐
        # ffill(limit=10)은 첫 값 이후 추가로 10개를 채움 (총 11개 non-NaN)
        non_nan_count = out["X"].notna().sum()
        assert non_nan_count == 11  # 원본 1 + ffill 10
        # 그 이후는 NaN
        assert out["X"].iloc[11:].isna().all()

    def test_monthly_series_alignment(self):
        """[Unit] 월말 시리즈가 일별 거래일 인덱스에 ffill되는지 확인.

        모두 거래일인 월말 날자를 골라 인덱스에 존재함을 보장:
            2024-01-31 = 수, 2024-02-29 = 목, 2024-05-31 = 금
        """
        # 거래일만 골라서 수동 구성 (PMI 발표 날)
        publish_dates = [pd.Timestamp("2024-01-31"),  # 수
                         pd.Timestamp("2024-02-29"),  # 목
                         pd.Timestamp("2024-05-31")]  # 금
        m = pd.Series([50.0, 52.0, 48.0], index=pd.DatetimeIndex(publish_dates), name="PMI")
        out = align_series({"PMI": m}, "2024-01-31", "2024-06-05", ffill_limit=30)

        # 1/31(수) 발표 → 50.0
        assert out["PMI"].loc["2024-01-31"] == 50.0
        # 2/1(목) → ffill로 50.0
        assert out["PMI"].loc["2024-02-01"] == 50.0
        # 2/28(수) 발표 전이므로 여전히 50.0
        assert out["PMI"].loc["2024-02-28"] == 50.0
        # 2/29(목) 발표 → 52.0
        assert out["PMI"].loc["2024-02-29"] == 52.0
        # 3월 중순 (ffill_limit=30 이내) → 52.0 유지
        assert out["PMI"].loc["2024-03-15"] == 52.0
        # 4월 중순 (2/29 경과 30거래일 이상) → NaN
        # 2/29 이후 30거래일 = 대략 4/11. 그 이후 5/31이전까지는 NaN
        assert pd.isna(out["PMI"].loc["2024-04-30"])
        assert pd.isna(out["PMI"].loc["2024-05-15"])
        # 5/31(금) 발표 → 48.0
        assert out["PMI"].loc["2024-05-31"] == 48.0

    def test_multiple_series_columns_preserved(self):
        """[Unit] 여러 시리즈가 컬럼으로 보존된다."""
        a = make_daily_series("2024-01-01", "2024-01-31", value=1.0, name="A")
        b = make_daily_series("2024-01-01", "2024-01-31", value=2.0, name="B")
        out = align_series({"A": a, "B": b}, "2024-01-01", "2024-01-31")
        assert list(out.columns) == ["A", "B"]
        assert (out["A"] == 1.0).all()
        assert (out["B"] == 2.0).all()

    def test_empty_series_becomes_nan_column(self):
        """[Unit] 빈 시리즈는 NaN 컬럼으로 추가된다 (분석은 가능)."""
        empty = pd.Series(dtype="float64", name="E")
        valid = make_daily_series("2024-01-01", "2024-01-31", value=5.0, name="V")
        out = align_series({"E": empty, "V": valid}, "2024-01-01", "2024-01-31")
        assert "E" in out.columns
        assert out["E"].isna().all()
        assert (out["V"] == 5.0).all()

    def test_tz_aware_input_normalized(self):
        """[Unit] tz-aware 인덱스는 KST로 변환 후 tz-naive로 정규화."""
        idx_utc = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
        s = pd.Series(range(len(idx_utc)), index=idx_utc, name="X", dtype="float64")
        out = align_series({"X": s}, "2024-01-01", "2024-01-10")
        assert out.index.tz is None
        assert len(out) > 0

    def test_drop_long_gap_returns_copy(self):
        """[Unit] drop_long_gap_periods는 1차에서 copy만 반환 (식별 로직)."""
        df = pd.DataFrame({"X": [1.0, np.nan, 3.0]},
                          index=pd.date_range("2024-01-01", periods=3, freq="B"))
        out = drop_long_gap_periods(df)
        pd.testing.assert_frame_equal(out, df)
        assert out is not df  # copy


# =============================================================================
# standardize 테스트
# =============================================================================
class TestStandardize:

    def test_zscore_mean_zero_std_one_approx(self):
        """[Unit] 충분히 긴 정규분포 시리즈의 z-score는 평균 ~0, 표준편차 ~1."""
        rng = np.random.default_rng(123)
        # 5년 + 여유분
        n = TRADING_DAYS_PER_YEAR * 6
        idx = pd.bdate_range("2018-01-01", periods=n)
        s = pd.Series(rng.normal(10, 2, size=n), index=idx, name="X")

        z = rolling_zscore(s, window_years=5)
        # 마지막 1년은 윈도우가 완전히 채워진 구간
        z_tail = z.dropna().iloc[-TRADING_DAYS_PER_YEAR:]
        assert abs(z_tail.mean()) < 0.3, f"평균이 0에 가깝지 않음: {z_tail.mean()}"
        assert 0.5 < z_tail.std() < 1.5, f"표준편차가 1에 가깝지 않음: {z_tail.std()}"

    def test_zscore_constant_series_is_nan(self):
        """[Unit] 상수 시리즈는 σ=0이라 z-score가 NaN."""
        idx = pd.bdate_range("2018-01-01", periods=TRADING_DAYS_PER_YEAR * 6)
        s = pd.Series(5.0, index=idx, name="C")
        z = rolling_zscore(s, window_years=5)
        # σ=0인 윈도우는 NaN으로 처리됨
        assert z.dropna().empty or (z.abs() < 1e-10).all()

    def test_zscore_empty_series(self):
        """[Unit] 빈 시리즈는 빈 시리즈를 반환."""
        s = pd.Series(dtype="float64", name="X")
        z = rolling_zscore(s)
        assert z.empty

    def test_robust_zscore_less_distorted_by_outlier(self):
        """[Unit] robust z-score는 단발성 극단치 이후의 기준선 오염이 작다."""
        n = TRADING_DAYS_PER_YEAR * 6
        idx = pd.bdate_range("2018-01-01", periods=n)
        vals = np.zeros(n)
        vals[TRADING_DAYS_PER_YEAR * 5] = 100.0
        vals[TRADING_DAYS_PER_YEAR * 5 + 5] = 1.0
        s = pd.Series(vals, index=idx, name="X")

        classic = rolling_zscore(s, window_years=5, min_periods_ratio=0.5)
        robust = rolling_robust_zscore(s, window_years=5, min_periods_ratio=0.5)

        probe_idx = idx[TRADING_DAYS_PER_YEAR * 5 + 5]
        assert abs(robust.loc[probe_idx]) > abs(classic.loc[probe_idx])

    def test_empirical_percentile_rank_midrank(self):
        """[Unit] rolling empirical percentile는 선형 z 환산이 아니라 분포 내 순위."""
        n = TRADING_DAYS_PER_YEAR * 3
        idx = pd.bdate_range("2020-01-01", periods=n)
        s = pd.Series(np.arange(n, dtype=float), index=idx, name="X")
        pct = rolling_percentile_rank(s, window_years=1, min_periods_ratio=0.5)
        assert pct.dropna().iloc[-1] > 99.0

    def test_empirical_percentile_rank_flat_window_is_midrank(self):
        """[Unit] 값이 모두 같은 윈도우는 경험분포 midrank인 50."""
        n = TRADING_DAYS_PER_YEAR * 2
        idx = pd.bdate_range("2020-01-01", periods=n)
        s = pd.Series(np.zeros(n), index=idx, name="X")
        pct = rolling_percentile_rank(s, window_years=1, min_periods_ratio=0.5)
        assert pct.dropna().iloc[-1] == 50.0

    def test_zscore_insufficient_window_is_nan(self):
        """[Unit] 윈도우가 채워지지 않은 초기 구간은 NaN."""
        rng = np.random.default_rng(7)
        # 1년치만 → 5년 윈도우 못 채움
        idx = pd.bdate_range("2023-01-01", periods=TRADING_DAYS_PER_YEAR)
        s = pd.Series(rng.normal(0, 1, size=len(idx)), index=idx, name="X")
        z = rolling_zscore(s, window_years=5, min_periods_ratio=0.5)
        # min_periods = window * 0.5 = 5*252*0.5 = 630일 필요 → 1년치는 부족
        assert z.isna().all()

    def test_panel_negative_direction_inverts_sign(self):
        """[Unit] risk_direction='negative' 변수는 표준화 후 부호 반전."""
        rng = np.random.default_rng(99)
        n = TRADING_DAYS_PER_YEAR * 6
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(rng.normal(0, 1, size=n), index=idx, name="X")
        df = pd.DataFrame({"X_POS": x, "X_NEG": x})

        out = standardize_panel(
            df,
            risk_directions={"X_POS": "positive", "X_NEG": "negative"},
            window_years=5,
        )
        # 두 컬럼이 부호만 반대
        non_nan = out.dropna()
        np.testing.assert_array_almost_equal(
            non_nan["X_POS"].values, -non_nan["X_NEG"].values, decimal=10
        )

    def test_panel_positive_tail_clips_negative_values(self):
        """[Unit] risk_direction='positive_tail' 변수는 음수 z-score를 0 처리."""
        rng = np.random.default_rng(101)
        n = TRADING_DAYS_PER_YEAR * 6
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(rng.normal(0, 1, size=n), index=idx, name="X")
        df = pd.DataFrame({"X_TAIL": x})

        out = standardize_panel(
            df,
            risk_directions={"X_TAIL": "positive_tail"},
            window_years=5,
        )
        valid = out["X_TAIL"].dropna()
        assert not valid.empty
        assert (valid >= 0).all()
        assert (valid == 0).any()

    def test_panel_invalid_direction_raises(self):
        """[Unit] 잘못된 risk_direction은 ValueError."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]},
                          index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValueError):
            standardize_panel(df, risk_directions={"X": "unknown"})

    def test_panel_empty(self):
        """[Unit] 빈 DataFrame은 빈 DataFrame 반환."""
        out = standardize_panel(pd.DataFrame(), risk_directions={})
        assert out.empty

    def test_load_directions_from_yaml(self):
        """[Unit] variables.yaml에서 위험방향 매핑을 추출."""
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "variables.yaml"
        directions = load_risk_directions_from_yaml(str(yaml_path))

        # 활성 변수는 포함되어야 함
        assert "VIX" in directions
        assert directions["VIX"] == "positive"
        assert "INDPRO" in directions
        assert directions["INDPRO"] == "negative"
        assert "KRW_USD" in directions
        assert directions["KRW_USD"] == "positive"

        # 비활성 변수(SP500_EPS, MOVE)는 제외되어야 함
        assert "SP500_EPS" not in directions
        assert "MOVE" not in directions


# =============================================================================
# 통합: alignment → standardize 파이프라인
# =============================================================================
class TestPipeline:

    def test_align_then_standardize(self):
        """[Integration] alignment 결과를 standardize에 그대로 넘길 수 있어야 한다."""
        rng = np.random.default_rng(11)
        n_days = TRADING_DAYS_PER_YEAR * 6
        idx = pd.bdate_range("2018-01-01", periods=n_days)
        a = pd.Series(rng.normal(100, 5, size=n_days), index=idx, name="A")
        b = pd.Series(rng.normal(20, 2, size=n_days), index=idx, name="B")

        df = align_series({"A": a, "B": b},
                          start_date="2018-01-01",
                          end_date=idx[-1].strftime("%Y-%m-%d"))

        z = standardize_panel(
            df,
            risk_directions={"A": "positive", "B": "negative"},
        )

        # 같은 인덱스
        assert (z.index == df.index).all()
        # 같은 컬럼
        assert list(z.columns) == ["A", "B"]
        # 마지막 1년은 모두 유효한 값
        tail = z.dropna().iloc[-TRADING_DAYS_PER_YEAR:]
        assert len(tail) >= TRADING_DAYS_PER_YEAR - 5  # 끝부분 NaN 약간 허용
