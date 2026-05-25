"""스트레스 지수 계산 모듈.

표준화된 z-score 패널에서 채널별 점수와 종합 지수를 만듭니다.

채널 점수:
    S_k(t) = mean( z_i(t) for i in channel k )
        - NaN은 제외하고 평균 (skipna=True)
        - 채널에 활성 변수가 모두 NaN인 시점은 그 채널 점수도 NaN

종합 지수:
    집계 방식은 thresholds.composite_method 설정으로 선택 (v15):
      - mean (종래): Stress(t) = Σ_k w_k * S_k(t)
            · 부호 보존 → 채널 상쇄 가능 (위험 채널 신호 묻힘).
      - l2_norm (v15 기본): Stress(t) = sqrt(Σ_k w_k * S_k(t)²)
            · 항상 ≥0 — "평시 상태로부터의 거리(위험 강도)" 해석.
            · 한 채널의 강한 신호가 다른 약한 채널에 의해 달라지지 않음.

    공통:
        - w_k는 채널 가중치 (variables.yaml의 channel_weights), 합 1.0 정규화.
        - 1차 프로토타입은 동일 가중치 0.20 × 5.
        - 각 시점에서 NaN 채널은 제외하고 남은 가중치로 재정규화.

부호 규칙:
    모든 입력 z-score는 이미 "양수=위험 증가" 규칙을 따른다고 가정
    (preprocessing/standardize.py가 보장).
    l2_norm은 제곱으로 인해 부호 정보를 잃으므로 종합점수는 "강도" 해석만 유효.
    채널 점수 S1~S5는 여전히 부호 정보를 유지하므로 패턴 분류에 사용 가능.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_COMPOSITE_METHOD,
    VALID_COMPOSITE_METHODS,
)


def compute_channel_scores(
    z_panel: pd.DataFrame,
    variable_to_channel: Mapping[str, int],
) -> pd.DataFrame:
    """채널별 점수를 계산.

    Args:
        z_panel: 표준화된 z-score DataFrame. 컬럼=변수 코드.
        variable_to_channel: {변수 코드: 채널 번호(int)}.

    Returns:
        인덱스가 동일하고 컬럼이 'S1'..'S5'인 DataFrame.
        채널에 속한 모든 변수가 NaN인 시점은 NaN.

    동작:
        - z_panel에 있는 컬럼 중 매핑이 없는 변수는 무시
        - 채널 번호는 1~5 가정
    """
    if z_panel.empty:
        return pd.DataFrame(index=z_panel.index)

    # 채널 -> 변수 리스트로 역매핑
    ch_to_vars: dict[int, list[str]] = {}
    for var, ch in variable_to_channel.items():
        if var in z_panel.columns:
            ch_to_vars.setdefault(ch, []).append(var)

    result = {}
    for ch in sorted(ch_to_vars.keys()):
        cols = ch_to_vars[ch]
        # axis=1 평균. skipna=True로 부분 결측 허용.
        result[f"S{ch}"] = z_panel[cols].mean(axis=1, skipna=True)

    df = pd.DataFrame(result, index=z_panel.index)
    return df


def compute_composite_score(
    channel_scores: Mapping[int, float] | pd.Series,
    method: str = DEFAULT_COMPOSITE_METHOD,
) -> float:
    """(v15) 단일 시점의 5채널 점수를 종합점수로 집계.

    Args:
        channel_scores: {채널 번호: z-score} 또는 동등한 Series.
            NaN 값은 자동 제외하고 조정된 n으로 계산.
        method: "mean" 또는 "l2_norm" (기본 l2_norm).
            - mean    : 산술평균 Σ S_k / n (부호 보존, 채널 가중 미적용).
            - l2_norm : RMS sqrt(Σ S_k² / n) (항상 ≥0, 위험 강도).

    Returns:
        스칼라 종합점수. 유효 채널이 0개면 NaN.

    Raises:
        ValueError: VALID_COMPOSITE_METHODS 외의 method.

    Note:
        이 함수는 단일 시점 계산용. 시계열 전체 집계는
        compute_composite_index가 가중치와 함께 처리.
    """
    if method not in VALID_COMPOSITE_METHODS:
        raise ValueError(
            f"지원하지 않는 composite_method 값: {method!r}. "
            f"허용: {sorted(VALID_COMPOSITE_METHODS)}"
        )

    if isinstance(channel_scores, pd.Series):
        iterable = channel_scores.values
    elif isinstance(channel_scores, Mapping):
        iterable = channel_scores.values()
    else:
        iterable = channel_scores

    values = [float(v) for v in iterable if pd.notna(v)]
    n = len(values)
    if n == 0:
        return float("nan")

    if method == "mean":
        return sum(values) / n
    # l2_norm
    return (sum(v * v for v in values) / n) ** 0.5


def compute_composite_index(
    channel_scores: pd.DataFrame,
    channel_weights: Mapping[int, float] | None = None,
    method: str = DEFAULT_COMPOSITE_METHOD,
) -> pd.Series:
    """종합 스트레스 지수 시계열을 계산.

    Args:
        channel_scores: compute_channel_scores 결과 (컬럼 S1..S5).
        channel_weights: {채널 번호: 가중치}. None이면 동일 가중치.
        method: (v15) "mean" 또는 "l2_norm" (기본 l2_norm).
            - mean    : Σ w_k * S_k (가중평균, 부호 보존).
            - l2_norm : sqrt(Σ w_k * S_k²)
                · 가중 제곱합의 제곱근 — 항상 ≥0, 위험 강도 해석.
                · 동일 가중치 시 sqrt(mean(S_k²)) (RMS)와 동치.

    Returns:
        종합 지수 시계열 (이름 'composite').

    동작:
        - 가중치 합이 1이 아니면 정규화
        - 채널 점수 NaN은 가중치 재정규화 (남은 채널들로만 계산)
            → 한두 채널이 일시적으로 NaN이어도 종합 점수 계산 가능
        - 모든 채널이 NaN인 시점은 NaN
    """
    if method not in VALID_COMPOSITE_METHODS:
        raise ValueError(
            f"지원하지 않는 composite_method 값: {method!r}. "
            f"허용: {sorted(VALID_COMPOSITE_METHODS)}"
        )

    if channel_scores.empty:
        return pd.Series(dtype="float64", name="composite")

    # 1) 가중치 준비
    channels_present = []
    for col in channel_scores.columns:
        # 'S1' -> 1
        try:
            ch = int(col.replace("S", ""))
            channels_present.append(ch)
        except ValueError:
            continue

    if channel_weights is None:
        weights = {ch: 1.0 / len(channels_present) for ch in channels_present}
    else:
        weights = {ch: channel_weights.get(ch, 0.0) for ch in channels_present}

    # 2) DataFrame 형태로 정렬
    w_array = np.array([weights[int(c.replace("S", ""))] for c in channel_scores.columns])

    # 3) NaN-aware 집계:
    #    각 행에서 NaN이 아닌 채널만으로 가중치 재정규화
    values = channel_scores.values  # (T, K)
    mask = ~np.isnan(values)
    # 가중치 행렬 (모든 행 동일)
    w_matrix = np.tile(w_array, (values.shape[0], 1))
    # NaN 위치 가중치는 0으로
    w_eff = w_matrix * mask
    w_sum = w_eff.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        # NaN을 0으로 치환 후 집계
        v_filled = np.where(mask, values, 0.0)
        if method == "mean":
            composite = (v_filled * w_eff).sum(axis=1) / w_sum
        else:  # l2_norm
            # 가중 제곱합의 제곱근.
            # 동일 가중치(1/n) 시 sqrt(Σ S_k² / n) = RMS와 동치.
            sq_weighted = (v_filled ** 2) * w_eff
            composite = np.sqrt(sq_weighted.sum(axis=1) / w_sum)

    out = pd.Series(composite, index=channel_scores.index, name="composite")
    # w_sum=0인 행은 NaN
    out = out.where(w_sum > 0)
    return out


def build_stress_index_table(
    z_panel: pd.DataFrame,
    variable_to_channel: Mapping[str, int],
    channel_weights: Mapping[int, float] | None = None,
    composite_method: str = DEFAULT_COMPOSITE_METHOD,
) -> pd.DataFrame:
    """편의 함수: 채널 점수 5개 + 종합 점수를 한 번에 반환.

    Args:
        z_panel: 표준화 완료 점수 패널.
        variable_to_channel: {변수: 채널} 매핑.
        channel_weights: {채널: 가중치}. None이면 동일 가중치.
        composite_method: (v15) 종합 점수 집계 방식 ("mean" | "l2_norm").

    Returns:
        컬럼이 'S1', 'S2', 'S3', 'S4', 'S5', 'composite'인 DataFrame.
    """
    ch_scores = compute_channel_scores(z_panel, variable_to_channel)
    comp = compute_composite_index(
        ch_scores, channel_weights, method=composite_method,
    )
    result = ch_scores.copy()
    result["composite"] = comp
    return result


def load_channel_mapping_from_yaml(yaml_path: str) -> tuple[dict[str, int], dict[int, float]]:
    """variables.yaml에서 {코드: 채널}, {채널: 가중치} 매핑을 추출.

    enabled=False 변수는 매핑에서 제외.

    Returns:
        (variable_to_channel, channel_weights) 튜플.
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    var_to_ch = {}
    for v in cfg.get("variables", []):
        if not v.get("enabled", True):
            continue
        var_to_ch[v["code"]] = int(v["channel"])

    # YAML의 channel_weights는 키가 int인지 str인지 yaml 파서에 따라 다름
    raw_weights = cfg.get("channel_weights", {})
    weights = {int(k): float(v) for k, v in raw_weights.items()}
    return var_to_ch, weights
