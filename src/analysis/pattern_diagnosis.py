"""충격 패턴 진단 모듈.

채널 점수 (S1~S5)를 입력 받아 규칙 기반으로 충격 패턴을 분류합니다.

분류 규칙 (variables.yaml의 pattern_rules와 일치):
    system_crisis        : S1>1.5 AND S2>1.5 AND S3>2.0 AND S5>1.5
    risk_premium_shock   : S3>2.0 AND S1<1.0 AND S2<1.0
    rate_shock           : S2>1.5 AND S3<1.5
    real_recession       : S1>1.5 AND MA30(S1) > MA60(S1)
    supply_shock         : S4>2.0 AND S1<1.5
    normal               : 그 외

우선순위:
    여러 규칙에 동시 해당 시 위 순서대로 첫 매칭을 채택.
    (시스템 위기 > 위험프리미엄 > 금리 > 실물침체 > 공급 > 정상)

임계값:
    함수 파라미터로 조정 가능 (Streamlit 설정 페이지에서 슬라이더 노출).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# 패턴 정의 (UI에서 사용할 메타데이터)
# =============================================================================
PATTERN_META: dict[str, dict] = {
    "system_crisis":      {"name_kr": "시스템 위기형",     "severity": "critical", "color": "#8B0000"},
    "risk_premium_shock": {"name_kr": "위험프리미엄 충격형", "severity": "high",     "color": "#FF4500"},
    "rate_shock":         {"name_kr": "금리 충격형",        "severity": "medium",   "color": "#FFA500"},
    "real_recession":     {"name_kr": "실물 침체형",        "severity": "high",     "color": "#DC143C"},
    "supply_shock":       {"name_kr": "공급충격형",         "severity": "medium",   "color": "#DAA520"},
    "normal":             {"name_kr": "정상",              "severity": "low",      "color": "#2E8B57"},
}


@dataclass
class PatternThresholds:
    """패턴 분류 임계값 (Streamlit에서 조정 가능)."""
    system_s1: float = 1.5
    system_s2: float = 1.5
    system_s3: float = 2.0
    system_s5: float = 1.5

    rp_s3: float = 2.0
    rp_s1_max: float = 1.0
    rp_s2_max: float = 1.0

    rate_s2: float = 1.5
    rate_s3_max: float = 1.5

    real_s1: float = 1.5
    real_ma_short: int = 30   # 거래일
    real_ma_long: int = 60

    supply_s4: float = 2.0
    supply_s1_max: float = 1.5


def _safe_gt(value: float, threshold: float) -> bool:
    """NaN-safe greater-than. NaN은 False로 평가."""
    return bool(pd.notna(value) and value > threshold)


def _safe_lt(value: float, threshold: float) -> bool:
    """NaN-safe less-than. NaN은 False로 평가."""
    return bool(pd.notna(value) and value < threshold)


def classify_at(
    s1: float, s2: float, s3: float, s4: float, s5: float,
    ma30_s1: float, ma60_s1: float,
    thresholds: PatternThresholds | None = None,
) -> str:
    """단일 시점에서의 패턴 분류.

    Args:
        s1~s5: 채널 점수 (이 시점).
        ma30_s1, ma60_s1: S1의 이동평균 (실물침체 판정용).
        thresholds: 임계값 객체. None이면 기본값 사용.

    Returns:
        패턴 키 (예: 'system_crisis', 'normal').
    """
    th = thresholds or PatternThresholds()

    # 1) 시스템 위기형 (최우선)
    if (_safe_gt(s1, th.system_s1) and _safe_gt(s2, th.system_s2)
            and _safe_gt(s3, th.system_s3) and _safe_gt(s5, th.system_s5)):
        return "system_crisis"

    # 2) 위험프리미엄 충격형
    if (_safe_gt(s3, th.rp_s3)
            and _safe_lt(s1, th.rp_s1_max) and _safe_lt(s2, th.rp_s2_max)):
        return "risk_premium_shock"

    # 3) 금리 충격형
    if _safe_gt(s2, th.rate_s2) and _safe_lt(s3, th.rate_s3_max):
        return "rate_shock"

    # 4) 실물 침체형
    if _safe_gt(s1, th.real_s1) and _safe_gt(ma30_s1, ma60_s1):
        return "real_recession"

    # 5) 공급충격형
    if _safe_gt(s4, th.supply_s4) and _safe_lt(s1, th.supply_s1_max):
        return "supply_shock"

    return "normal"


def classify_history(
    channel_scores: pd.DataFrame,
    thresholds: PatternThresholds | None = None,
) -> pd.Series:
    """전체 기간에 대해 매일 패턴을 분류.

    Args:
        channel_scores: 컬럼 'S1'..'S5'를 포함하는 DataFrame.
        thresholds: 임계값 객체.

    Returns:
        인덱스 동일, 값이 패턴 키 문자열인 pd.Series (이름 'pattern').
    """
    if channel_scores.empty:
        return pd.Series(dtype="object", name="pattern")

    # S1의 이동평균 미리 계산
    s1 = channel_scores.get("S1", pd.Series(np.nan, index=channel_scores.index))
    th = thresholds or PatternThresholds()
    ma30 = s1.rolling(window=th.real_ma_short, min_periods=1).mean()
    ma60 = s1.rolling(window=th.real_ma_long, min_periods=1).mean()

    # 컬럼 추출 (없으면 NaN 시리즈)
    def col(name: str) -> pd.Series:
        if name in channel_scores.columns:
            return channel_scores[name]
        return pd.Series(np.nan, index=channel_scores.index)

    s1s, s2s, s3s, s4s, s5s = col("S1"), col("S2"), col("S3"), col("S4"), col("S5")

    patterns = []
    for i in range(len(channel_scores)):
        patterns.append(classify_at(
            s1=s1s.iloc[i], s2=s2s.iloc[i], s3=s3s.iloc[i],
            s4=s4s.iloc[i], s5=s5s.iloc[i],
            ma30_s1=ma30.iloc[i], ma60_s1=ma60.iloc[i],
            thresholds=th,
        ))
    return pd.Series(patterns, index=channel_scores.index, name="pattern")


def pattern_meta(pattern_key: str) -> dict:
    """패턴 키에 대한 UI 메타데이터 (name_kr, severity, color) 반환."""
    return PATTERN_META.get(pattern_key, PATTERN_META["normal"])
