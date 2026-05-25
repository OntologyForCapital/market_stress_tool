"""진원지 추적 모듈.

룩백 기간 내에서 z-score가 임계값을 처음으로 넘은 변수를 식별하여
"충격의 진원지 후보"로 표시하고, 다른 채널로의 전이 시간차를 계산합니다.

[중요 한계 - 사용자에게 반드시 노출할 것]
    이 모듈이 식별하는 "진원지"는 **시간적 선후관계에 기반한 추정**일 뿐입니다.
    실제 인과관계는 다를 수 있습니다 (예: 공통 원인이 두 변수에 영향을 주었을 수 있음).
    Confounding 가능성을 항상 명시해야 합니다.

알고리즘:
    1. 룩백 기간 (기본 60거래일) 내 각 변수에 대해
       z(t) >= threshold를 처음 만족하는 시점 t_i 식별
    2. 가장 이른 t_i의 변수를 "진원지 후보"로 지정
    3. 진원지 변수가 속한 채널을 시작으로, 다른 채널의 "첫 발화 시점" 계산
    4. 시간차(일수)를 사슬로 출력
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class OriginResult:
    """진원지 추적 결과."""
    origin_variable: str | None              # 진원지 변수 코드 (없으면 None)
    origin_channel: int | None               # 진원지 채널 번호
    origin_first_breach_date: pd.Timestamp | None
    variable_first_breach: dict[str, pd.Timestamp] = field(default_factory=dict)
    channel_first_breach: dict[int, pd.Timestamp] = field(default_factory=dict)
    transition_chain: list[tuple[int, pd.Timestamp, int]] = field(default_factory=list)
    # transition_chain: [(채널, 발화시점, 진원지로부터 일수)]


def _first_breach_date(
    series: pd.Series,
    threshold: float,
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> pd.Timestamp | None:
    """시리즈에서 [lookback_start, lookback_end] 구간 중 처음으로 threshold를 넘는 날짜.

    None을 반환하는 경우:
        - 구간 내 데이터가 없음
        - 임계값을 한 번도 넘지 않음
    """
    window = series.loc[lookback_start:lookback_end].dropna()
    if window.empty:
        return None
    above = window[window >= threshold]
    if above.empty:
        return None
    return above.index[0]


def track_origin(
    z_panel: pd.DataFrame,
    variable_to_channel: Mapping[str, int],
    as_of: pd.Timestamp | str | None = None,
    threshold: float = 1.5,
    lookback_days: int = 60,
) -> OriginResult:
    """진원지 변수와 채널 전이 사슬을 추정.

    Args:
        z_panel: 표준화된 z-score DataFrame (양수=위험 부호 규칙).
        variable_to_channel: {변수 코드: 채널 번호}.
        as_of: 기준 시점. None이면 마지막 행.
        threshold: 발화 임계값 (기본 1.5σ).
        lookback_days: 룩백 기간 (거래일 기준, 기본 60).

    Returns:
        OriginResult.
            모든 변수가 임계값 미만이면 origin_variable=None.

    [한계]
        - 시간 순서 ≠ 인과관계
        - threshold는 임의값. 정규분포 가정 하 1.5σ는 약 6.7% 분위수
        - 동률 시점일 경우 변수 코드 알파벳 순으로 결정 (deterministic)
    """
    if z_panel.empty:
        return OriginResult(origin_variable=None, origin_channel=None,
                            origin_first_breach_date=None)

    # 1) as_of 결정
    if as_of is None:
        as_of_ts = z_panel.index[-1]
    else:
        as_of_ts = pd.Timestamp(as_of)
        # as_of 이전 가장 가까운 인덱스
        valid = z_panel.index[z_panel.index <= as_of_ts]
        if len(valid) == 0:
            return OriginResult(origin_variable=None, origin_channel=None,
                                origin_first_breach_date=None)
        as_of_ts = valid[-1]

    # 2) 룩백 시작 시점
    lookback_start = as_of_ts - pd.tseries.offsets.BDay(lookback_days)

    # 3) 변수별 첫 발화 시점
    var_first: dict[str, pd.Timestamp] = {}
    for var, ch in variable_to_channel.items():
        if var not in z_panel.columns:
            continue
        d = _first_breach_date(z_panel[var], threshold, lookback_start, as_of_ts)
        if d is not None:
            var_first[var] = d

    if not var_first:
        # 아무 변수도 임계값을 넘지 않음
        return OriginResult(origin_variable=None, origin_channel=None,
                            origin_first_breach_date=None,
                            variable_first_breach={},
                            channel_first_breach={},
                            transition_chain=[])

    # 4) 가장 이른 시점의 변수 = 진원지. 동률 시 변수 코드 알파벳 순.
    sorted_vars = sorted(var_first.items(), key=lambda kv: (kv[1], kv[0]))
    origin_var, origin_date = sorted_vars[0]
    origin_ch = variable_to_channel[origin_var]

    # 5) 채널별 첫 발화 시점 (그 채널의 변수들 중 가장 이른 시점)
    ch_first: dict[int, pd.Timestamp] = {}
    for var, d in var_first.items():
        ch = variable_to_channel[var]
        if ch not in ch_first or d < ch_first[ch]:
            ch_first[ch] = d

    # 6) 전이 사슬: 채널을 발화 시점 순으로 정렬
    chain_sorted = sorted(ch_first.items(), key=lambda kv: kv[1])
    chain = []
    for ch, d in chain_sorted:
        days_from_origin = (d - origin_date).days
        chain.append((ch, d, days_from_origin))

    return OriginResult(
        origin_variable=origin_var,
        origin_channel=origin_ch,
        origin_first_breach_date=origin_date,
        variable_first_breach=var_first,
        channel_first_breach=ch_first,
        transition_chain=chain,
    )


def origin_result_to_dataframe(result: OriginResult) -> pd.DataFrame:
    """OriginResult를 UI 표시용 DataFrame으로 변환.

    Returns:
        컬럼 ['channel', 'first_breach_date', 'days_from_origin'] DataFrame.
        전이 사슬 시간 순.
    """
    if not result.transition_chain:
        return pd.DataFrame(columns=["channel", "first_breach_date", "days_from_origin"])

    rows = [
        {"channel": ch, "first_breach_date": d, "days_from_origin": days}
        for ch, d, days in result.transition_chain
    ]
    return pd.DataFrame(rows)
