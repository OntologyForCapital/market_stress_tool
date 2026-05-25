"""분석 모듈 단위 테스트.

더미 데이터로 4개 분석 모듈을 검증:
    1. stress_index      : 채널 점수 + 종합 지수
    2. pattern_diagnosis : 6개 패턴 규칙
    3. nearest_neighbors : k-NN 거리, 직전 60일 제외, 향후 수익률
    4. origin_tracking   : 가장 이른 발화 변수 식별, 전이 사슬
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.stress_index import (
    compute_channel_scores,
    compute_composite_index,
    build_stress_index_table,
    load_channel_mapping_from_yaml,
)
from src.analysis.pattern_diagnosis import (
    PatternThresholds,
    classify_at,
    classify_history,
    pattern_meta,
    PATTERN_META,
)
from src.analysis.nearest_neighbors import (
    find_similar_dates,
    compute_forward_returns,
    find_similar_with_forward_returns,
)
from src.analysis.origin_tracking import (
    track_origin,
    origin_result_to_dataframe,
)


# =============================================================================
# 헬퍼: 더미 z-score 패널
# =============================================================================
def make_zpanel(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    """5채널 15변수 더미 z-score 패널 생성."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = {}
    for ch in range(1, 6):
        for k in range(3):
            name = f"CH{ch}_V{k}"
            cols[name] = rng.normal(0, 1, size=n_days)
    return pd.DataFrame(cols, index=idx)


def make_channel_mapping() -> dict[str, int]:
    """위 더미 패널에 맞는 채널 매핑."""
    return {f"CH{ch}_V{k}": ch for ch in range(1, 6) for k in range(3)}


# =============================================================================
# 1. stress_index
# =============================================================================
class TestStressIndex:

    def test_channel_scores_mean_of_variables(self):
        """[Unit] 채널 점수 = 채널 내 변수들의 평균."""
        idx = pd.bdate_range("2024-01-01", periods=3)
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": [3.0, 4.0, 5.0],
            "C": [10.0, 20.0, 30.0],
        }, index=idx)
        mapping = {"A": 1, "B": 1, "C": 2}
        out = compute_channel_scores(df, mapping)
        # S1 = (A+B)/2
        np.testing.assert_array_almost_equal(out["S1"].values, [2.0, 3.0, 4.0])
        # S2 = C
        np.testing.assert_array_almost_equal(out["S2"].values, [10.0, 20.0, 30.0])

    def test_channel_scores_skip_nan(self):
        """[Unit] 채널 내 일부 NaN은 평균에서 제외."""
        idx = pd.bdate_range("2024-01-01", periods=2)
        df = pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 5.0]}, index=idx)
        out = compute_channel_scores(df, {"A": 1, "B": 1})
        # 첫날: (1+3)/2=2, 둘째날: 5만 있음 → 5
        np.testing.assert_array_almost_equal(out["S1"].values, [2.0, 5.0])

    def test_channel_scores_all_nan_row_is_nan(self):
        """[Unit] 채널 내 모든 변수가 NaN인 행은 NaN."""
        idx = pd.bdate_range("2024-01-01", periods=2)
        df = pd.DataFrame({"A": [np.nan, 1.0]}, index=idx)
        out = compute_channel_scores(df, {"A": 1})
        assert pd.isna(out["S1"].iloc[0])
        assert out["S1"].iloc[1] == 1.0

    def test_composite_equal_weight(self):
        """[Unit] mean 방식 가중치 None → 단순 평균.

        v15: compute_composite_index 기본값이 l2_norm으로 변경되었으므로
        mean 동작 검증은 method="mean" 명시.
        """
        idx = pd.bdate_range("2024-01-01", periods=2)
        ch = pd.DataFrame({
            "S1": [1.0, 2.0], "S2": [3.0, 4.0], "S3": [5.0, 6.0],
            "S4": [7.0, 8.0], "S5": [9.0, 10.0],
        }, index=idx)
        comp = compute_composite_index(ch, method="mean")
        # 평균 = (1+3+5+7+9)/5 = 5, (2+4+6+8+10)/5 = 6
        np.testing.assert_array_almost_equal(comp.values, [5.0, 6.0])

    def test_composite_custom_weights(self):
        """[Unit] mean 방식 사용자 가중치 적용 + 합 정규화."""
        idx = pd.bdate_range("2024-01-01", periods=1)
        ch = pd.DataFrame({"S1": [10.0], "S2": [0.0]}, index=idx)
        # 가중치 합이 1이 아님 (총 0.3) - 합 정규화 안 하더라도 sum/w_sum으로 정규화됨
        comp = compute_composite_index(
            ch, channel_weights={1: 0.2, 2: 0.1}, method="mean",
        )
        # 결과 = (10*0.2 + 0*0.1) / (0.2+0.1) = 2.0/0.3 ≈ 6.666...
        np.testing.assert_array_almost_equal(comp.values, [6.6666666], decimal=4)

    def test_composite_nan_reweighting(self):
        """[Unit] mean 방식 일부 채널 NaN → 남은 채널들로 재정규화."""
        idx = pd.bdate_range("2024-01-01", periods=1)
        ch = pd.DataFrame({"S1": [4.0], "S2": [np.nan], "S3": [2.0]}, index=idx)
        comp = compute_composite_index(ch, method="mean")  # 동일 가중치 1/3
        # S2가 NaN이므로 S1, S3만으로 평균 (가중치 1/3씩 → 정규화 후 1/2씩)
        # = (4+2)/2 = 3.0
        np.testing.assert_array_almost_equal(comp.values, [3.0])

    def test_composite_all_nan_row(self):
        """[Unit] 모든 채널 NaN이면 종합도 NaN (method 분기 무관)."""
        idx = pd.bdate_range("2024-01-01", periods=1)
        ch = pd.DataFrame({"S1": [np.nan], "S2": [np.nan]}, index=idx)
        # 양쪽 방식 모두 NaN 반환이어야 함
        assert pd.isna(compute_composite_index(ch, method="mean").iloc[0])
        assert pd.isna(compute_composite_index(ch, method="l2_norm").iloc[0])

    def test_build_stress_index_table_columns(self):
        """[Unit] build_stress_index_table은 S1..S5 + composite 컬럼."""
        z = make_zpanel(n_days=100)
        m = make_channel_mapping()
        out = build_stress_index_table(z, m)
        assert {"S1", "S2", "S3", "S4", "S5", "composite"} == set(out.columns)
        assert len(out) == 100

    def test_load_yaml_mapping(self):
        """[Unit] variables.yaml에서 매핑 추출."""
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "variables.yaml"
        var_to_ch, weights = load_channel_mapping_from_yaml(str(yaml_path))
        # 활성 변수 일부 검증
        assert var_to_ch["VIX"] == 3
        assert var_to_ch["ISM_PMI"] == 1
        assert var_to_ch["KRW_USD"] == 5
        # 비활성 제외
        assert "MOVE" not in var_to_ch
        assert "SP500_EPS" not in var_to_ch
        # 가중치 합
        assert abs(sum(weights.values()) - 1.0) < 1e-9


# =============================================================================
# 2. pattern_diagnosis
# =============================================================================
class TestPatternDiagnosis:

    def test_system_crisis_pattern(self):
        """[Unit] S1>1.5 & S2>1.5 & S3>2.0 & S5>1.5 → 시스템 위기형."""
        p = classify_at(s1=2.0, s2=2.0, s3=2.5, s4=0.0, s5=2.0,
                        ma30_s1=2.0, ma60_s1=1.5)
        assert p == "system_crisis"

    def test_risk_premium_shock(self):
        """[Unit] S3>2.0 AND S1<1.0 AND S2<1.0 → 위험프리미엄 충격형."""
        p = classify_at(s1=0.5, s2=0.5, s3=2.5, s4=0.0, s5=0.0,
                        ma30_s1=0.5, ma60_s1=0.4)
        assert p == "risk_premium_shock"

    def test_rate_shock(self):
        """[Unit] S2>1.5 AND S3<1.5 → 금리 충격형."""
        p = classify_at(s1=0.0, s2=2.0, s3=0.5, s4=0.0, s5=0.0,
                        ma30_s1=0.0, ma60_s1=0.0)
        assert p == "rate_shock"

    def test_real_recession(self):
        """[Unit] S1>1.5 AND MA30(S1)>MA60(S1) → 실물 침체형."""
        p = classify_at(s1=2.0, s2=0.0, s3=0.0, s4=0.0, s5=0.0,
                        ma30_s1=1.8, ma60_s1=1.0)
        assert p == "real_recession"

    def test_supply_shock(self):
        """[Unit] S4>2.0 AND S1<1.5 → 공급충격형."""
        p = classify_at(s1=0.5, s2=0.5, s3=0.5, s4=2.5, s5=0.5,
                        ma30_s1=0.5, ma60_s1=0.5)
        assert p == "supply_shock"

    def test_normal_default(self):
        """[Unit] 어떤 규칙도 만족 안 하면 정상."""
        p = classify_at(s1=0.0, s2=0.0, s3=0.0, s4=0.0, s5=0.0,
                        ma30_s1=0.0, ma60_s1=0.0)
        assert p == "normal"

    def test_priority_system_over_others(self):
        """[Unit] 시스템 위기와 다른 패턴 동시 매칭 시 시스템 우선."""
        # S1, S2, S3, S5 모두 매우 높음 → 시스템 위기 + 금리 충격에도 해당
        p = classify_at(s1=2.0, s2=2.0, s3=2.5, s4=0.0, s5=2.0,
                        ma30_s1=2.0, ma60_s1=1.0)
        assert p == "system_crisis"

    def test_classify_history_returns_series(self):
        """[Unit] classify_history는 인덱스 동일한 Series."""
        idx = pd.bdate_range("2024-01-01", periods=10)
        ch = pd.DataFrame({
            "S1": [0.0] * 10, "S2": [0.0] * 10, "S3": [0.0] * 10,
            "S4": [0.0] * 10, "S5": [0.0] * 10,
        }, index=idx)
        out = classify_history(ch)
        assert isinstance(out, pd.Series)
        assert len(out) == 10
        assert (out == "normal").all()

    def test_nan_safe_evaluation(self):
        """[Unit] NaN은 패턴 평가에서 False (정상으로 분류됨)."""
        p = classify_at(s1=np.nan, s2=np.nan, s3=np.nan, s4=np.nan, s5=np.nan,
                        ma30_s1=np.nan, ma60_s1=np.nan)
        assert p == "normal"

    def test_pattern_meta_has_all_keys(self):
        """[Unit] 모든 패턴 키에 메타데이터(name_kr, severity, color) 있음."""
        for key in ["system_crisis", "risk_premium_shock", "rate_shock",
                    "real_recession", "supply_shock", "normal"]:
            meta = pattern_meta(key)
            assert "name_kr" in meta
            assert "severity" in meta
            assert "color" in meta

    def test_thresholds_customizable(self):
        """[Unit] PatternThresholds로 임계값 조정 가능."""
        th = PatternThresholds(rate_s2=3.0)  # 금리 충격 더 엄격하게
        # S2=2.0이면 기본값(1.5)으론 매칭, 새 값(3.0)으론 미매칭
        p_default = classify_at(0, 2.0, 0.5, 0, 0, 0, 0)
        p_strict = classify_at(0, 2.0, 0.5, 0, 0, 0, 0, thresholds=th)
        assert p_default == "rate_shock"
        assert p_strict == "normal"


# =============================================================================
# 3. nearest_neighbors
# =============================================================================
class TestNearestNeighbors:

    def test_find_similar_returns_k_rows(self):
        """[Unit] k=10이면 10행 반환."""
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=500)
        df = pd.DataFrame(rng.normal(0, 1, size=(500, 5)),
                          columns=["S1", "S2", "S3", "S4", "S5"], index=idx)
        out = find_similar_dates(df, k=10, exclude_recent_days=60)
        assert len(out) == 10
        assert list(out.columns) == ["distance", "S1", "S2", "S3", "S4", "S5"]
        # 거리 오름차순
        assert (out["distance"].diff().dropna() >= 0).all()

    def test_exclude_recent_days(self):
        """[Unit] 직전 60일은 후보에서 제외된다."""
        idx = pd.bdate_range("2020-01-01", periods=500)
        # 모든 시점 동일 벡터 → 거리 0
        df = pd.DataFrame(np.zeros((500, 5)),
                          columns=["S1", "S2", "S3", "S4", "S5"], index=idx)
        out = find_similar_dates(df, k=5, exclude_recent_days=60)
        query_date = idx[-1]
        cutoff = query_date - pd.tseries.offsets.BDay(60)
        # 반환된 모든 시점은 cutoff 이전
        assert (out.index < cutoff).all()

    def test_self_is_closest_when_exclude_zero(self):
        """[Unit] exclude=0이면 query 자기 자신이 가장 가까움 (거리 0).

        주의: find_similar_dates는 query < cutoff 인 후보만 보므로
        exclude=0이어도 query 시점 자체는 제외됨. exclude=-1 같은 트릭 없이
        검증하려면 query_date를 더 앞으로 설정.
        """
        idx = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(np.arange(500).reshape(100, 5).astype(float),
                          columns=["S1", "S2", "S3", "S4", "S5"], index=idx)
        # query=중간, exclude=0
        out = find_similar_dates(df, query_date=idx[50], k=1, exclude_recent_days=0)
        assert len(out) == 1
        # 가장 가까운 시점은 query 직전 거래일 (값이 가장 비슷)
        assert out.index[0] == idx[49]

    def test_compute_forward_returns(self):
        """[Unit] 향후 수익률 계산."""
        idx = pd.bdate_range("2024-01-01", periods=300)
        # 매일 0.1% 상승하는 가상 가격
        price = pd.Series(np.cumprod(np.full(len(idx), 1.001)), index=idx)
        ret = compute_forward_returns(price, [idx[0]], horizons=[30])
        assert "fwd_30d" in ret.columns
        # 30일 후 (캘린더일) 가격 ≈ (1.001)^21 (대략 21거래일) → 약 +2.1%
        # 정확값보다는 양수인지만 확인
        assert ret["fwd_30d"].iloc[0] > 0

    def test_forward_return_nan_when_horizon_exceeds(self):
        """[Unit] horizon이 시리즈 범위를 넘으면 NaN."""
        idx = pd.bdate_range("2024-01-01", periods=10)
        price = pd.Series(range(10), index=idx, dtype="float64")
        # 마지막 날짜에서 100일 후는 범위 밖
        ret = compute_forward_returns(price, [idx[-1]], horizons=[100])
        assert pd.isna(ret["fwd_100d"].iloc[0])

    def test_find_similar_with_forward_returns_combines(self):
        """[Integration] find_similar + forward_returns 결합."""
        rng = np.random.default_rng(2)
        idx = pd.bdate_range("2020-01-01", periods=500)
        z = pd.DataFrame(rng.normal(0, 1, size=(500, 5)),
                         columns=["S1", "S2", "S3", "S4", "S5"], index=idx)
        price = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, size=500)), index=idx)
        out = find_similar_with_forward_returns(z, price, k=5, exclude_recent_days=60)
        assert len(out) == 5
        # 거리 + 5채널 + 3 horizon
        assert {"distance", "S1", "S2", "S3", "S4", "S5",
                "fwd_30d", "fwd_90d", "fwd_180d"}.issubset(out.columns)

    def test_raises_when_empty(self):
        """[Unit] 빈 입력은 ValueError."""
        with pytest.raises(ValueError):
            find_similar_dates(pd.DataFrame(columns=["S1", "S2", "S3", "S4", "S5"]))

    def test_raises_when_missing_channel_column(self):
        """[Unit] 필요한 채널 컬럼이 없으면 ValueError."""
        df = pd.DataFrame({"S1": [1.0], "S2": [1.0]},
                          index=pd.bdate_range("2024-01-01", periods=1))
        with pytest.raises(ValueError):
            find_similar_dates(df)  # S3/S4/S5 없음


# =============================================================================
# 4. origin_tracking
# =============================================================================
class TestOriginTracking:

    def test_identifies_earliest_breach(self):
        """[Unit] 가장 이른 임계 돌파 변수를 진원지로 식별."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        # VAR_A는 1/3일에 먼저 발화, VAR_B는 1/10일에 발화
        df = pd.DataFrame({
            "VAR_A": [0.0] * 30,
            "VAR_B": [0.0] * 30,
        }, index=idx)
        df.loc[idx[2], "VAR_A"] = 2.0   # 1/3 (수): 임계값(1.5) 돌파
        df.loc[idx[9], "VAR_B"] = 2.0   # 1/12 (금): 임계값 돌파

        result = track_origin(df, {"VAR_A": 1, "VAR_B": 2},
                              as_of=idx[-1], threshold=1.5, lookback_days=60)
        assert result.origin_variable == "VAR_A"
        assert result.origin_channel == 1
        assert result.origin_first_breach_date == idx[2]

    def test_transition_chain_time_diff(self):
        """[Unit] 전이 사슬: 진원지 채널부터 다른 채널 발화까지 일수."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({
            "A": [0.0] * 30,   # 채널 1
            "B": [0.0] * 30,   # 채널 2
            "C": [0.0] * 30,   # 채널 3
        }, index=idx)
        df.loc[idx[0], "A"] = 2.0   # 1일에 채널 1 발화
        df.loc[idx[5], "B"] = 2.0   # 6일후 채널 2 발화
        df.loc[idx[10], "C"] = 2.0  # 11일후 채널 3 발화

        result = track_origin(df, {"A": 1, "B": 2, "C": 3}, as_of=idx[-1])
        assert result.origin_variable == "A"
        # 전이 사슬: (1, 0일), (2, ~9일), (3, ~14일)
        chain = result.transition_chain
        assert chain[0][0] == 1
        assert chain[0][2] == 0   # 진원지 자기 자신은 0일
        assert chain[1][0] == 2
        assert chain[1][2] > 0
        assert chain[2][0] == 3
        assert chain[2][2] > chain[1][2]

    def test_no_breach_returns_none(self):
        """[Unit] 임계값을 넘는 변수가 없으면 origin_variable=None."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({"A": [0.5] * 30, "B": [0.5] * 30}, index=idx)
        result = track_origin(df, {"A": 1, "B": 2}, threshold=1.5)
        assert result.origin_variable is None
        assert result.origin_channel is None
        assert result.transition_chain == []

    def test_tie_breaker_alphabetic(self):
        """[Unit] 같은 시점 발화 시 변수 코드 알파벳 순으로 진원지 결정."""
        idx = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "ZZZ": [2.0] * 10,
            "AAA": [2.0] * 10,
        }, index=idx)
        result = track_origin(df, {"ZZZ": 1, "AAA": 2})
        assert result.origin_variable == "AAA"  # 알파벳 순 첫 번째

    def test_lookback_excludes_old_breaches(self):
        """[Unit] 룩백 기간 밖의 발화는 무시된다."""
        idx = pd.bdate_range("2024-01-01", periods=200)
        df = pd.DataFrame({"A": [0.0] * 200, "B": [0.0] * 200}, index=idx)
        df.loc[idx[5], "A"] = 2.0       # 매우 오래된 발화
        df.loc[idx[180], "B"] = 2.0     # 최근 발화

        # lookback 60일 (as_of=마지막날)
        result = track_origin(df, {"A": 1, "B": 2}, lookback_days=60)
        # A의 발화는 룩백 밖이므로 무시되고 B가 진원지
        assert result.origin_variable == "B"

    def test_origin_to_dataframe(self):
        """[Unit] OriginResult를 DataFrame으로 변환."""
        idx = pd.bdate_range("2024-01-01", periods=20)
        df = pd.DataFrame({"A": [0.0] * 20, "B": [0.0] * 20}, index=idx)
        df.loc[idx[0], "A"] = 2.0
        df.loc[idx[5], "B"] = 2.0
        result = track_origin(df, {"A": 1, "B": 2})
        out_df = origin_result_to_dataframe(result)
        assert list(out_df.columns) == ["channel", "first_breach_date", "days_from_origin"]
        assert len(out_df) == 2

    def test_empty_panel(self):
        """[Unit] 빈 패널은 origin=None."""
        result = track_origin(pd.DataFrame(), {})
        assert result.origin_variable is None
