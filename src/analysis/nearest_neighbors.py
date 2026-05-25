"""k-NN 유사 시점 분석 모듈.

현재 채널 점수 벡터(5차원)에 가장 가까운 과거 시점 K개를 찾고,
각 시점 이후 30/90/180일의 KOSPI 변화율을 함께 반환합니다.

거리:
    5차원 유클리드 거리 (S1..S5)

제외 기간:
    가장 최근 60거래일은 제외 (현재 충격이 아직 진행 중일 수 있어 자기 자신과 매우 유사)

목적:
    "지금과 비슷한 매크로 환경이 과거에 있었나? 그때 KOSPI는 어떻게 움직였나?"
    → 단일 확률 예측이 아닌 "분포"를 보여주기 위함
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class NeighborResult:
    """k-NN 결과 한 행."""
    date: pd.Timestamp
    distance: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    fwd_30d: float | None
    fwd_90d: float | None
    fwd_180d: float | None


def find_similar_dates(
    channel_scores: pd.DataFrame,
    query_date: pd.Timestamp | str | None = None,
    k: int = 20,
    exclude_recent_days: int = 60,
    channels: Sequence[str] = ("S1", "S2", "S3", "S4", "S5"),
) -> pd.DataFrame:
    """현재(또는 지정 시점)와 유사한 과거 시점 K개를 반환.

    Args:
        channel_scores: 컬럼 'S1'..'S5'를 포함하는 DataFrame (시계열).
        query_date: 비교 기준 시점. None이면 마지막 행 사용.
        k: 반환할 이웃 수. 기본 20.
        exclude_recent_days: query 기준 직전 N거래일을 후보에서 제외.
        channels: 거리 계산에 사용할 채널 컬럼.

    Returns:
        K행 DataFrame. 컬럼:
            distance, S1, S2, S3, S4, S5 (해당 과거 시점의 점수)
        인덱스는 해당 과거 날짜.

    Raises:
        ValueError: 데이터가 부족하거나 query_date에 결측치.
    """
    if channel_scores.empty:
        raise ValueError("channel_scores가 비어있습니다.")

    missing = [c for c in channels if c not in channel_scores.columns]
    if missing:
        raise ValueError(f"필요한 채널 컬럼이 없습니다: {missing}")

    df = channel_scores[list(channels)].dropna()
    if df.empty:
        raise ValueError("결측 제거 후 유효한 시점이 없습니다.")

    # 1) query 벡터 결정
    if query_date is None:
        query_idx = df.index[-1]
    else:
        query_idx = pd.Timestamp(query_date)
        if query_idx not in df.index:
            # query_date 이전의 가장 가까운 거래일 사용
            valid = df.index[df.index <= query_idx]
            if len(valid) == 0:
                raise ValueError(f"query_date 이전 데이터가 없습니다: {query_date}")
            query_idx = valid[-1]

    q = df.loc[query_idx].values  # (5,)

    # 2) 후보 집합: query 이전 + exclude_recent_days 제외
    cutoff = query_idx - pd.tseries.offsets.BDay(exclude_recent_days)
    candidates = df.loc[df.index < cutoff]
    if candidates.empty:
        raise ValueError(
            f"후보 시점이 없습니다. exclude_recent_days={exclude_recent_days}를 줄이거나 "
            f"데이터 기간을 늘리세요."
        )

    # 3) 유클리드 거리 계산
    diff = candidates.values - q  # (N, 5)
    dist = np.sqrt((diff ** 2).sum(axis=1))

    # 4) 상위 K
    order = np.argsort(dist)[:k]
    top = candidates.iloc[order].copy()
    top.insert(0, "distance", dist[order])
    top.index.name = "date"
    return top


def compute_forward_returns(
    price_series: pd.Series,
    dates: Sequence[pd.Timestamp],
    horizons: Sequence[int] = (30, 90, 180),
) -> pd.DataFrame:
    """지정 시점들로부터 H일 후 가격 변화율을 계산.

    Args:
        price_series: KOSPI 종가 시계열 (DatetimeIndex).
        dates: 기준 시점들.
        horizons: 일 단위 horizon 리스트 (캘린더일 기준).

    Returns:
        인덱스가 dates, 컬럼이 'fwd_30d', 'fwd_90d', ... 인 DataFrame.
        H일 후 가격이 시리즈 범위 밖이면 NaN.

    구현:
        각 기준 시점 t에 대해
            P(t)  = t 또는 t 이전의 가장 가까운 거래일 가격
            P(t+H) = t+H 또는 t+H 이전의 가장 가까운 거래일 가격
            return = P(t+H) / P(t) - 1
        H는 캘린더일이라 거래일 수와 다를 수 있음 (직관성 우선).
    """
    if price_series.empty:
        return pd.DataFrame(index=pd.Index(dates), columns=[f"fwd_{h}d" for h in horizons])

    # asof를 위해 정렬 보장
    ps = price_series.sort_index()
    rows = {}
    for h in horizons:
        col_name = f"fwd_{h}d"
        vals = []
        for d in dates:
            d = pd.Timestamp(d)
            # 기준일 가격: d 또는 그 이전
            p0_idx = ps.index.asof(d)
            if pd.isna(p0_idx):
                vals.append(np.nan)
                continue
            p0 = ps.loc[p0_idx]

            # H일 후 가격
            target = d + pd.Timedelta(days=h)
            # target이 시계열 마지막 날짜보다 미래면 NaN
            if target > ps.index[-1]:
                vals.append(np.nan)
                continue
            p1_idx = ps.index.asof(target)
            if pd.isna(p1_idx):
                vals.append(np.nan)
                continue
            p1 = ps.loc[p1_idx]

            if p0 == 0 or pd.isna(p0) or pd.isna(p1):
                vals.append(np.nan)
            else:
                vals.append(float(p1 / p0 - 1.0))
        rows[col_name] = vals

    return pd.DataFrame(rows, index=pd.Index([pd.Timestamp(d) for d in dates], name="date"))


def find_similar_with_forward_returns(
    channel_scores: pd.DataFrame,
    price_series: pd.Series,
    query_date: pd.Timestamp | str | None = None,
    k: int = 20,
    exclude_recent_days: int = 60,
    horizons: Sequence[int] = (30, 90, 180),
) -> pd.DataFrame:
    """편의 함수: 유사 시점 K개 + 각 시점 이후 가격 변화율을 결합.

    Returns:
        컬럼: distance, S1..S5, fwd_30d, fwd_90d, fwd_180d
    """
    neighbors = find_similar_dates(
        channel_scores,
        query_date=query_date,
        k=k,
        exclude_recent_days=exclude_recent_days,
    )
    fwd = compute_forward_returns(price_series, neighbors.index.tolist(), horizons=horizons)
    return neighbors.join(fwd, how="left")
