"""Z-score 표준화 모듈.

각 변수의 시계열을 평균 0, 표준편차 1로 변환합니다.
부호 규칙: 표준화 후 **양수(+) = 위험 증가 방향**

방법:
    z_i(t) = (X_i(t) - μ_i(t)) / σ_i(t)
    μ, σ는 시점 t 기준 직전 5년 롤링 윈도우로 계산.

v18 기본값은 평균/표준편차가 아니라 median/MAD 기반 robust z-score입니다.
금융·거시 시계열은 위기 구간의 fat-tail과 단발성 API spike가 평균/표준편차를
오염시키기 쉬우므로, 평시 중심과 산포를 robust하게 추정합니다.

롤링 vs 전체 평균:
    - 전체 기간 평균은 미래 정보 누출 (앞 시점에서 미래 데이터를 사용)
    - 롤링 윈도우는 시점 t에서 t 이전 데이터만 사용 → look-ahead bias 없음
    - 윈도우 데이터가 부족하면 (min_periods 미만) NaN으로 둠

위험 방향 부호 반전:
    - risk_direction이 'negative'인 변수는 표준화 후 -1 곱함
    - 예: ISM_PMI는 낮을수록 위험 ↑ → z 계산 후 부호 반전하여 "양의 z = 위험"

양방향 위험 처리 (v14):
    - risk_direction이 'bidirectional'인 변수는 양쪽 꼬리가 모두 위험
    - 예: BRENT(유가 급등=공급쇼크, 급락=수요붕괴), US_BEI_10Y(인플레/디플레 모두 위험)
    - 신호 변환: signal = max(|z| - threshold, 0) — ±threshold σ 밴드 내 평시는 0,
      밴드를 벗어나면 두 방향 모두 양수 신호로 변환 ("양수=위험" 규약 유지)
    - threshold 기본 1.0σ. 변수별로 bidirectional_threshold 오버라이드 가능.

극단치 처리:
    - 1차 프로토타입에서는 별도 winsorization 안 함
    - 극단치도 그대로 신호로 인정 (위기 시점 z-score가 5~10에 달할 수 있음)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..config import VALID_RISK_DIRECTIONS, DEFAULT_BIDIRECTIONAL_THRESHOLD
from ..config import (
    DEFAULT_STANDARDIZATION_METHOD,
    DEFAULT_Z_CLIP_ABS,
    VALID_STANDARDIZATION_METHODS,
)


# 1년 거래일 수 (대략)
TRADING_DAYS_PER_YEAR = 252

# v13: 6개월 워밍아웃을 252/2이 아닌 126으로 고정 (의뢰서 명세)
HALFYEAR_TRADING_DAYS = 126

# 정규분포에서 MAD를 표준편차 추정량으로 맞추는 스케일.
MAD_TO_SIGMA = 1.4826


def rolling_zscore(
    series: pd.Series,
    window_years: int = 5,
    min_periods_ratio: float = 0.5,
) -> pd.Series:
    """단일 시리즈에 대해 롤링 z-score를 계산.

    Args:
        series: 표준화할 시리즈 (거래일 인덱스 권장).
        window_years: 롤링 윈도우 길이 (년). 기본 5년.
        min_periods_ratio: 윈도우 크기 대비 최소 유효 관측치 비율.
                           기본 0.5 → 윈도우가 5년이면 최소 2.5년치 데이터 필요.

    Returns:
        z-score 시리즈. 윈도우 부족 구간은 NaN.

    구현:
        - window = window_years * TRADING_DAYS_PER_YEAR (관측치 수 기준)
        - μ = rolling mean, σ = rolling std (ddof=0, population std)
        - σ = 0인 시점은 분모 0 회피를 위해 NaN으로 처리
    """
    if series.empty:
        return series.copy()

    window = window_years * TRADING_DAYS_PER_YEAR
    min_periods = max(2, int(window * min_periods_ratio))

    # closed='left'를 쓰면 현재 시점 제외(t 이전만)하지만, 일반적으로 t 포함
    # 1차 프로토타입에서는 t 포함 (당일 값으로 표준화) — 분석 단순성 우선
    mu = series.rolling(window=window, min_periods=min_periods).mean()
    sigma = series.rolling(window=window, min_periods=min_periods).std(ddof=0)

    # σ가 0이거나 너무 작으면 NaN (수치 안정성)
    sigma = sigma.where(sigma > 1e-12)

    z = (series - mu) / sigma
    z.name = series.name
    return z


def rolling_robust_zscore(
    series: pd.Series,
    window_years: int = 5,
    min_periods_ratio: float = 0.5,
) -> pd.Series:
    """단일 시리즈에 대해 rolling robust z-score를 계산.

    중앙값과 MAD(median absolute deviation)를 사용하므로 극단치가 평균과
    표준편차를 끌어당기는 문제를 줄인다. MAD가 0인 이산적/계단형 시계열은
    동일 윈도우의 일반 표준편차를 보조 분모로 사용한다.
    """
    if series.empty:
        return series.copy()

    window = window_years * TRADING_DAYS_PER_YEAR
    min_periods = max(2, int(window * min_periods_ratio))

    median = series.rolling(window=window, min_periods=min_periods).median()

    def _mad(values: np.ndarray) -> float:
        vals = values[~np.isnan(values)]
        if len(vals) == 0:
            return float("nan")
        med = np.median(vals)
        return float(np.median(np.abs(vals - med)))

    mad = series.rolling(window=window, min_periods=min_periods).apply(_mad, raw=True)
    robust_sigma = (mad * MAD_TO_SIGMA).where(lambda s: s > 1e-12)

    # 정책금리처럼 오래 평평하다가 계단식으로 움직이는 시계열은 MAD가 0이기 쉽다.
    # 이 경우에만 표준편차를 보조 분모로 사용해 신호 소실을 피한다.
    fallback_sigma = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    fallback_sigma = fallback_sigma.where(fallback_sigma > 1e-12)
    sigma = robust_sigma.fillna(fallback_sigma)

    z = (series - median) / sigma
    z.name = series.name
    return z


def rolling_percentile_rank(
    series: pd.Series,
    window_years: int = 5,
    min_periods_ratio: float = 0.5,
) -> pd.Series:
    """현재 값의 rolling empirical percentile rank를 0~100으로 계산.

    기존 선형 변환(50 + 20*z)은 정규분포를 암묵적으로 가정한다. 이 함수는
    각 시점의 값을 그 시점까지의 롤링 분포와 직접 비교하므로 fat-tail,
    비대칭, zero-inflated bidirectional 신호에 더 자연스럽다.
    """
    if series.empty:
        return series.copy()

    window = window_years * TRADING_DAYS_PER_YEAR
    min_periods = max(2, int(window * min_periods_ratio))

    def _percentile(values: np.ndarray) -> float:
        current = values[-1]
        if np.isnan(current):
            return float("nan")
        vals = values[~np.isnan(values)]
        if len(vals) < min_periods:
            return float("nan")
        span = float(np.nanmax(vals) - np.nanmin(vals))
        if span <= 1e-12:
            return 50.0
        less = float(np.sum(vals < current))
        equal = float(np.sum(vals == current))
        return ((less + 0.5 * equal) / len(vals)) * 100.0

    out = series.rolling(window=window, min_periods=min_periods).apply(
        _percentile, raw=True,
    )
    out.name = series.name
    return out


def apply_risk_direction(
    z: pd.Series,
    risk_direction: str,
    threshold: float = DEFAULT_BIDIRECTIONAL_THRESHOLD,
) -> pd.Series:
    """(v14) z-score에 위험 방향 규칙을 적용해 "양수=위험" 신호로 변환.

    Args:
        z: rolling_zscore 결과 시리즈.
        risk_direction: 'positive' | 'negative' | 'bidirectional'.
            - positive: 그대로 반환 (z가 클수록 위험).
            - negative: -z (작을수록 위험인 지표를 양수로 변환).
            - bidirectional: max(|z| - threshold, 0) — 양쪽 꼬리 모두 위험.
              ±threshold σ 밴드 내 평시는 0, 밴드 밖이면 두 방향 모두 양수 신호.
        threshold: bidirectional일 때만 사용하는 ±σ 밴드 폭. 기본 1.0σ.

    Returns:
        변환된 시리즈. NaN은 NaN으로 전파됨.

    Raises:
        ValueError: risk_direction이 VALID_RISK_DIRECTIONS에 없으면 발생.
    """
    if risk_direction not in VALID_RISK_DIRECTIONS:
        raise ValueError(
            f"지원하지 않는 risk_direction 값: {risk_direction!r}. "
            f"허용: {sorted(VALID_RISK_DIRECTIONS)}"
        )
    if risk_direction == "positive":
        return z
    if risk_direction == "negative":
        return -z
    # bidirectional
    return (z.abs() - threshold).clip(lower=0)


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """(v13) 변수별 사전 변환 적용. z-score 계산 이전 단계에서 호출.

    수준(level)만 표준화하면 후행적 지표(산업생산)·변곡점 지표(정책금리)·
    모멘텀 지표(환율·유가)의 신호가 무뎌지므로, 각 변수 특성에 맞는
    변환을 적용한 후 z-score를 산출한다.

    Args:
        series: 영업일 인덱스 시계열 (이미 align 단계 완료 상태).
        transform: 'level' | 'yoy_pct' | 'pct_change_6m' | 'diff_6m'.

    Returns:
        변환된 시계열. 'level'은 입력 그대로.
        나머지 변환은 앞쪽 N영업일이 NaN이 되므로 (252 또는 126),
        호출 측은 충분한 워밍아웃 기간을 포함해 데이터를 들여와야 한다.

    Raises:
        ValueError: 지원하지 않는 transform 값.
    """
    if transform == "level":
        return series
    if transform == "yoy_pct":
        # 1년(≈252영업일) 전 대비 % 변화
        return series.pct_change(
            periods=TRADING_DAYS_PER_YEAR, fill_method=None,
        ) * 100.0
    if transform == "pct_change_6m":
        # 6개월(≈126영업일) 전 대비 % 변화
        return series.pct_change(
            periods=HALFYEAR_TRADING_DAYS, fill_method=None,
        ) * 100.0
    if transform == "diff_6m":
        # 6개월 전 대비 단순 차분 (단위 유지)
        return series.diff(periods=HALFYEAR_TRADING_DAYS)
    raise ValueError(
        f"지원하지 않는 transform 값: {transform!r}. "
        "허용: 'level', 'yoy_pct', 'pct_change_6m', 'diff_6m'"
    )


def apply_transforms_panel(
    df: pd.DataFrame,
    transforms: Mapping[str, str],
) -> pd.DataFrame:
    """(v13) 패널(DataFrame) 전체에 변수별 transform 적용.

    Args:
        df: alignment.align_series 결과. 각 컬럼이 변수.
        transforms: {변수 코드: transform 명}. 매핑 없는 컬럼은 'level'로 처리.

    Returns:
        같은 인덱스/컬럼의 DataFrame. 변환 적용 후 앞쪽 워밍아웃 구간은 NaN.
    """
    if df.empty:
        return df.copy()

    out_cols: dict[str, pd.Series] = {}
    for col in df.columns:
        transform = transforms.get(col, "level")
        out_cols[col] = apply_transform(df[col], transform)
    result = pd.concat(out_cols, axis=1)
    result.index = df.index
    return result


def standardize_panel(
    df: pd.DataFrame,
    risk_directions: Mapping[str, str],
    window_years: int = 5,
    min_periods_ratio: float = 0.5,
    bidirectional_thresholds: Mapping[str, float] | None = None,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
    clip_abs: float | None = DEFAULT_Z_CLIP_ABS,
) -> pd.DataFrame:
    """패널(DataFrame) 전체에 대해 z-score 표준화 + 위험방향 규칙 적용.

    Args:
        df: alignment.align_series 결과. 각 컬럼이 변수.
        risk_directions: {변수 코드: 'positive' | 'negative' | 'bidirectional'}.
            각 방향별 처리는 apply_risk_direction 참고.
        window_years: 롤링 윈도우 (년).
        min_periods_ratio: 최소 유효 관측치 비율.
        bidirectional_thresholds: (v14) {변수 코드: ±σ 밴드 폭}. bidirectional
            변수에만 적용. 매핑에 없으면 DEFAULT_BIDIRECTIONAL_THRESHOLD(1.0)
            을 사용.
        method: 'classic'이면 평균/표준편차, 'robust'이면 median/MAD 기반.
        clip_abs: 위험방향 적용 전 z-score 절댓값 상한. None이면 비활성.

    Returns:
        같은 인덱스/컬럼의 DataFrame. 모든 값이 "양수=위험" 부호 규칙.

    Raises:
        ValueError: risk_directions 값이 VALID_RISK_DIRECTIONS 외이면 발생.
    """
    if df.empty:
        return df.copy()
    if method not in VALID_STANDARDIZATION_METHODS:
        raise ValueError(
            f"지원하지 않는 standardization_method 값: {method!r}. "
            f"허용: {sorted(VALID_STANDARDIZATION_METHODS)}"
        )

    thresholds = bidirectional_thresholds or {}

    out_cols: dict[str, pd.Series] = {}
    for col in df.columns:
        if method == "classic":
            z = rolling_zscore(
                df[col], window_years=window_years,
                min_periods_ratio=min_periods_ratio,
            )
        else:
            z = rolling_robust_zscore(
                df[col], window_years=window_years,
                min_periods_ratio=min_periods_ratio,
            )

        if clip_abs is not None:
            z = z.clip(lower=-float(clip_abs), upper=float(clip_abs))

        direction = risk_directions.get(col, "positive")
        threshold = thresholds.get(col, DEFAULT_BIDIRECTIONAL_THRESHOLD)
        out_cols[col] = apply_risk_direction(z, direction, threshold)

    result = pd.concat(out_cols, axis=1)
    result.index = df.index
    return result


def load_risk_directions_from_yaml(yaml_path: str) -> dict[str, str]:
    """variables.yaml에서 {코드: 위험방향} 매핑을 추출.

    enabled=False 변수는 매핑에 포함하지 않음 (분석에서 제외).
    computed 변수도 포함 (예: KR_US_RATE_DIFF).

    Args:
        yaml_path: variables.yaml 절대 경로.

    Returns:
        예: {'VIX': 'positive', 'ISM_PMI': 'negative', ...}
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    directions = {}
    for v in cfg.get("variables", []):
        if not v.get("enabled", True):
            continue
        directions[v["code"]] = v["risk_direction"]
    return directions
