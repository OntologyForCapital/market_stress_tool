"""전체 진단 파이프라인 (Step 7-3).

`run_full_diagnosis()`는 데이터 수집 → 정합화 → 표준화 → 채널/종합 지수 산출 →
패턴 분류 → 유사 시점 탐색 → 진원지 추적까지 한 번에 수행하고,
UI에서 바로 사용할 수 있는 `DiagnosisResult` 객체를 반환합니다.

호출 흐름:
    1) config.load_variables() / load_target_variables() / 기타 매핑 로드
    2) dispatcher.fetch_all_variables() — variables + target_variables 통합 수집
    3) preprocessing.align_series() — 영업일 그리드 정합
    4) preprocessing.standardize_panel() — 롤링 robust z-score (target 제외)
    5) analysis.build_stress_index_table() — S1~S5 + composite
    6) analysis.classify_history() — 일자별 패턴 분류
    7) analysis.find_similar_with_forward_returns() — KOSPI 기준 유사 시점 + forward returns
    8) analysis.track_origin() — 진원지 변수/채널 추적
    9) DiagnosisResult 조립

데이터 출처 안내:
    KOSPI/KOSDAQ 타겟은 v20부터 KRX API가 아니라 Yahoo Finance
    (^KS11, ^KQ11)를 사용합니다.

[한계 및 주의]
    - 변수 수집 실패는 silent하게 격리되어 `failed_variables`에 기록됨
    - 종합 점수는 활성화된(channel이 있는) 변수만으로 계산. target은 제외.
    - 패턴 분류 임계값은 PatternThresholds 기본값을 사용.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.nearest_neighbors import find_similar_with_forward_returns
from src.analysis.origin_tracking import OriginResult, track_origin
from src.analysis.pattern_diagnosis import (
    PatternThresholds,
    classify_at,
    classify_history,
)
from src.analysis.stress_index import build_stress_index_table
from src.analysis.threshold_calibration import build_threshold_calibration
from src.config import (
    Variable,
    load_channel_mapping,
    load_channel_weights,
    load_bidirectional_thresholds,
    load_composite_method,
    load_percentile_method,
    load_risk_directions,
    load_standardization_method,
    load_target_variables,
    load_transform_map,
    load_variables,
    load_z_clip_abs,
)
from src.data_collection.dispatcher import fetch_all_variables
from src.preprocessing.alignment import align_series
from src.preprocessing.standardize import (
    apply_transforms_panel,
    rolling_percentile_rank,
    standardize_panel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 결과 dataclass
# =============================================================================
@dataclass
class DiagnosisResult:
    """전체 진단 결과 (UI/리포트에서 직접 소비).

    Fields (verbatim spec):
        as_of_date            : 진단 기준일 (None이면 데이터의 마지막 영업일 자동 사용)
        composite_score       : 현재 종합 스트레스 지수 (z-score)
        composite_percentile  : 0~100 백분위 변환값 (기본: rolling empirical)
        pattern_label         : 현재 패턴 분류 (예: 'normal', 'rate_shock', ...)
        channel_scores        : 현재 채널별 z-score {1: S1, 2: S2, ..., 5: S5}
        channel_percentiles   : 0~100 백분위 변환값 {1: pct, ..., 5: pct}
        variable_z_scores     : 변수별 z-score {코드: z}
        variable_percentiles  : 0~100 백분위 변환값 {코드: pct}
        origin_result         : OriginResult 객체 (진원지 추적)
        similar_dates         : k-NN 결과 DataFrame (distance, S1..S5, fwd_30d/90d/180d)
        forward_returns_summary: {30: {'avg': , 'median': }, 90: ..., 180: ...}
        failed_variables      : 수집 실패 변수 코드 목록
        data_period           : (실제 데이터 시작일, 끝일)
        composite_pct_series  : 종합 스트레스 백분위 시계열 (UI 시계열 차트용)
        channel_pct_panel     : 채널별 백분위 시계열 DataFrame (S1~S5 컬럼)
        raw_panel             : 수집 직후 원자료 시계열 DataFrame (정합/변환 전)
        aligned_panel         : 정합된 raw 값 시계열 DataFrame (변수별 원본값, 세부 내용 탭용)
        z_panel               : 변수별 z-score 시계열 DataFrame (세부 내용 탭용)
        stress_table          : 채널/종합 z-score 시계열 DataFrame
        variable_pct_panel    : 변수별 백분위 시계열 DataFrame
        standardization_method: 표준화 방식 ('robust' | 'classic')
        percentile_method     : 백분위 방식 ('empirical' | 'linear')
        z_clip_abs            : z-score 절대값 clip 한도
        calibration_summary   : 이벤트/변동성 regime 기반 임계값 보정 요약
        calibration_event_metrics: 이벤트 라벨 기준 threshold 성능표
        calibration_regime_thresholds: volatility regime별 경험적 분위수
        calibration_event_labels: 이벤트 윈도우 라벨 시계열
        calibration_regime_series: 날짜별 volatility regime 시계열
    """

    as_of_date: pd.Timestamp
    composite_score: float
    composite_percentile: float
    pattern_label: str
    channel_scores: dict[int, float] = field(default_factory=dict)
    channel_percentiles: dict[int, float] = field(default_factory=dict)
    variable_z_scores: dict[str, float] = field(default_factory=dict)
    variable_percentiles: dict[str, float] = field(default_factory=dict)
    origin_result: OriginResult | None = None
    similar_dates: pd.DataFrame = field(default_factory=pd.DataFrame)
    forward_returns_summary: dict[int, dict[str, float]] = field(default_factory=dict)
    failed_variables: list[str] = field(default_factory=list)
    data_period: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)
    composite_pct_series: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    channel_pct_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    aligned_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    z_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    stress_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    variable_pct_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    standardization_method: str = "robust"
    percentile_method: str = "empirical"
    z_clip_abs: float | None = None
    calibration_summary: dict[str, object] = field(default_factory=dict)
    calibration_event_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_regime_thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration_event_labels: pd.Series = field(default_factory=lambda: pd.Series(dtype="bool"))
    calibration_regime_series: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))


# =============================================================================
# 백분위 변환 헬퍼
# =============================================================================
def z_to_percentile(z: float) -> float:
    """z-score를 0~100 백분위로 선형 변환 (레거시 fallback).

    매핑:
        z = -2.5 → 0
        z =  0   → 50
        z = +2.5 → 100
        그 밖은 [0, 100]으로 clip.

    이 변환은 정규분포 가정 하의 정확한 CDF가 아니라 UI 시각화를 위한
    선형 근사입니다. v18 이후 파이프라인 기본값은 rolling empirical
    percentile rank이며, 이 함수는 테스트/폴백 호환성을 위해 유지합니다.

    NaN 입력은 NaN 그대로 반환.
    """
    if pd.isna(z):
        return float("nan")
    val = 50.0 + float(z) * 20.0
    return float(np.clip(val, 0.0, 100.0))


# =============================================================================
# 내부 헬퍼
# =============================================================================
def _resolve_as_of(z_panel: pd.DataFrame, as_of: pd.Timestamp | str | None) -> pd.Timestamp:
    """as_of를 z_panel의 실제 인덱스로 정규화.

    - None: 마지막 행
    - 지정: 해당 일자 이하의 가장 가까운 인덱스
    """
    if z_panel.empty:
        raise ValueError("z_panel이 비어 있어 as_of를 결정할 수 없습니다.")
    if as_of is None:
        return z_panel.index[-1]

    ts = pd.Timestamp(as_of)
    valid = z_panel.index[z_panel.index <= ts]
    if len(valid) == 0:
        # 데이터 시작 이전을 지정한 경우, 첫 인덱스 사용 (보수적 처리)
        logger.warning(
            "as_of=%s가 데이터 시작(%s) 이전입니다. 첫 영업일로 폴백.",
            as_of, z_panel.index[0],
        )
        return z_panel.index[0]
    return valid[-1]


def _safe_series_row(
    df: pd.DataFrame, ts: pd.Timestamp,
) -> pd.Series:
    """df에서 ts 행을 안전 추출. 정확 매치 없으면 asof로 직전 행."""
    if ts in df.index:
        return df.loc[ts]
    asof_idx = df.index.asof(ts)
    if pd.isna(asof_idx):
        return pd.Series(dtype="float64")
    return df.loc[asof_idx]


def _summarize_forward_returns(
    similar_df: pd.DataFrame,
    horizons: tuple[int, ...] = (30, 90, 180),
) -> dict[int, dict[str, float]]:
    """fwd_30d/90d/180d 컬럼에서 평균/중앙값을 추출.

    Returns:
        {30: {'avg': float, 'median': float, 'count': int}, 90: ..., 180: ...}
        해당 컬럼이 없거나 전부 NaN이면 평균/중앙값 NaN.
    """
    summary: dict[int, dict[str, float]] = {}
    for h in horizons:
        col = f"fwd_{h}d"
        if col not in similar_df.columns:
            summary[h] = {"avg": float("nan"), "median": float("nan"), "count": 0}
            continue
        vals = similar_df[col].dropna()
        if vals.empty:
            summary[h] = {"avg": float("nan"), "median": float("nan"), "count": 0}
        else:
            summary[h] = {
                "avg": float(vals.mean()),
                "median": float(vals.median()),
                "count": int(len(vals)),
            }
    return summary


def _percentile_series(
    series: pd.Series,
    method: str,
    window_years: int = 5,
    min_periods_ratio: float = 0.5,
) -> pd.Series:
    """시계열을 UI용 0~100 점수로 변환."""
    if method == "empirical":
        return rolling_percentile_rank(
            series,
            window_years=window_years,
            min_periods_ratio=min_periods_ratio,
        )
    return series.apply(z_to_percentile)


def _lookup_percentile(
    pct_series: pd.Series,
    ts: pd.Timestamp,
    fallback_value: float,
) -> float:
    """percentile 시계열에서 ts 값을 안전 조회하고 없으면 선형 fallback."""
    if pct_series.empty:
        return z_to_percentile(fallback_value)
    row = _safe_series_row(pct_series.to_frame("pct"), ts)
    pct = row.get("pct", float("nan")) if not row.empty else float("nan")
    if pd.notna(pct):
        return float(pct)
    return z_to_percentile(fallback_value)


# =============================================================================
# 메인 진단 파이프라인
# =============================================================================
def run_full_diagnosis(
    start_date: str,
    end_date: str,
    as_of: pd.Timestamp | str | None = None,
    use_cache: bool = True,
    yaml_path: Path | str | None = None,
    k_neighbors: int = 20,
    exclude_recent_days: int = 60,
    horizons: tuple[int, ...] = (30, 90, 180),
    origin_threshold: float = 1.5,
    origin_lookback_days: int = 60,
    # 의존성 주입(테스트용)
    _fetch_all_variables=None,
) -> DiagnosisResult:
    """전체 시장 스트레스 진단을 한 번에 실행.

    Args:
        start_date: 데이터 수집 시작일 ('YYYY-MM-DD'). 표준화 윈도우(기본 5년)
            + (v13) 변환 워밍아웃(최대 252영업일 ≈1년, yoy_pct 기준)을
            고려해 진단 기준일보다 **7년 이상 이전**을 권장.
            (기존 6년 권장 → v13에서 변환 적용으로 1년 더 필요)
        end_date: 데이터 수집 종료일 ('YYYY-MM-DD').
        as_of: 진단 기준일. None이면 정합된 패널의 마지막 영업일.
        use_cache: True면 디스크 캐시 사용. 운영 단계에서는 True 권장.
        yaml_path: variables.yaml 경로. None이면 기본값.
        k_neighbors: 유사 시점 K개.
        exclude_recent_days: 유사 탐색 시 query 직전 N영업일 제외.
        horizons: forward return 일수 (캘린더일).
        origin_threshold: 진원지 추적 임계 z (기본 1.5σ).
        origin_lookback_days: 진원지 추적 룩백 영업일.
        _fetch_all_variables: 테스트용 의존성 주입 (signature는 dispatcher와 동일).
            None이면 실제 dispatcher.fetch_all_variables 사용.

    Returns:
        DiagnosisResult.

    Raises:
        ValueError: variables.yaml 로드 실패, 또는 수집된 변수가 0개.
    """
    logger.info(
        "run_full_diagnosis 시작: start=%s, end=%s, as_of=%s, use_cache=%s",
        start_date, end_date, as_of, use_cache,
    )

    # -----------------------------------------------------------------
    # 1) 설정 로드
    # -----------------------------------------------------------------
    all_vars: list[Variable] = list(load_variables(yaml_path))
    target_vars: list[Variable] = list(load_target_variables(yaml_path))
    variable_to_channel: dict[str, int] = load_channel_mapping(yaml_path)
    risk_directions: dict[str, str] = load_risk_directions(yaml_path)
    channel_weights: dict[int, float] = load_channel_weights(yaml_path, normalize=True)
    # v13: 변수별 사전 변환 매핑 로드 (yaml의 transform 필드)
    transform_map: dict[str, str] = load_transform_map(yaml_path)
    # v14: 양방향(bidirectional) 변수에 적용할 ±σ 임계값 매핑
    bidirectional_thresholds: dict[str, float] = load_bidirectional_thresholds(yaml_path)
    # v15: 종합점수 집계 방식 ("mean" | "l2_norm", 기본 l2_norm)
    composite_method: str = load_composite_method(yaml_path)
    # v18: fat-tail을 고려한 robust 표준화 + 경험적 백분위.
    standardization_method: str = load_standardization_method(yaml_path)
    percentile_method: str = load_percentile_method(yaml_path)
    z_clip_abs: float | None = load_z_clip_abs(yaml_path)

    # 활성화 변수 코드 (channel 매핑이 있는 변수 = standardize/지수 계산 대상)
    target_codes = {v.code for v in target_vars}
    logger.info(
        "config 로드: variables=%d, target=%d, channel_mapping=%d, risk_dir=%d, bidir=%d",
        len(all_vars), len(target_vars), len(variable_to_channel),
        len(risk_directions), len(bidirectional_thresholds),
    )

    # 수집 대상: variables + target_variables (둘 다 align 필요)
    fetch_list = list(all_vars) + list(target_vars)

    # -----------------------------------------------------------------
    # 2) 데이터 수집 (dispatcher)
    # -----------------------------------------------------------------
    fetch_fn = _fetch_all_variables if _fetch_all_variables is not None else fetch_all_variables
    series_dict: dict[str, pd.Series] = fetch_fn(
        fetch_list, start_date, end_date, use_cache=use_cache,
    )

    # 실패 변수 = 요청했지만 결과에 없는 + 결과가 빈 시리즈
    requested_codes = {v.code for v in fetch_list if v.enabled or v.section == "target_variables"}
    returned_codes = set(series_dict.keys())
    failed_variables: list[str] = sorted(requested_codes - returned_codes)
    # 빈 시리즈도 실패로 간주 (정합 단계에서 사실상 무의미)
    for code, s in list(series_dict.items()):
        if s is None or s.empty:
            failed_variables.append(code)
            series_dict.pop(code, None)
    failed_variables = sorted(set(failed_variables))
    logger.info(
        "수집 결과: 성공=%d, 실패=%d (%s)",
        len(series_dict), len(failed_variables),
        ", ".join(failed_variables[:5]) + ("..." if len(failed_variables) > 5 else ""),
    )

    if not series_dict:
        raise ValueError(
            "수집된 변수가 0개입니다. variables.yaml 또는 네트워크/API 키를 확인하세요."
        )

    # 수집 직후 원자료 패널. 영업일 정합/forward-fill/변환 전 상태를 UI에서 확인하기 위함.
    raw_panel = pd.concat(
        [s.rename(code) for code, s in series_dict.items()],
        axis=1,
    ).sort_index()

    # -----------------------------------------------------------------
    # 3) 정합 (영업일 그리드)
    # -----------------------------------------------------------------
    aligned: pd.DataFrame = align_series(
        series_dict, start_date=start_date, end_date=end_date,
    )
    if aligned.empty:
        raise ValueError("align_series 결과가 비어 있습니다. 기간/데이터를 확인하세요.")
    logger.info(
        "align 완료: shape=%s, period=%s ~ %s",
        aligned.shape, aligned.index[0].date(), aligned.index[-1].date(),
    )

    # -----------------------------------------------------------------
    # 4) 표준화 (target 제외)
    # -----------------------------------------------------------------
    # standardize_panel은 risk_directions 매핑에 있는 컬럼만 z-score 화함.
    # target_variables는 risk_directions에 없으므로 자동 제외되지만,
    # 명시적으로 분리하여 가독성 확보.
    standardize_cols = [c for c in aligned.columns if c not in target_codes]
    if not standardize_cols:
        raise ValueError("표준화 대상 변수가 없습니다 (모두 target?).")
    panel_for_z = aligned[standardize_cols]

    # v13: z-score 계산 이전에 변수별 사전 변환 적용
    # (level은 그대로, yoy_pct/pct_change_6m/diff_6m은 앞쪽 N일 NaN)
    panel_for_z = apply_transforms_panel(panel_for_z, transforms=transform_map)
    logger.info(
        "transform 적용: 변수별 변환 종류=%s",
        {t: sum(1 for v in transform_map.values() if v == t)
         for t in ("level", "yoy_pct", "pct_change_6m", "diff_6m")},
    )

    z_panel: pd.DataFrame = standardize_panel(
        panel_for_z,
        risk_directions=risk_directions,
        bidirectional_thresholds=bidirectional_thresholds,
        method=standardization_method,
        clip_abs=z_clip_abs,
    )
    logger.info(
        "standardize 완료: z_panel shape=%s, method=%s, clip_abs=%s, bidirectional 변수=%s",
        z_panel.shape, standardization_method, z_clip_abs,
        sorted(bidirectional_thresholds.keys()),
    )

    # -----------------------------------------------------------------
    # 5) 채널/종합 스트레스 지수
    # -----------------------------------------------------------------
    stress_table: pd.DataFrame = build_stress_index_table(
        z_panel,
        variable_to_channel=variable_to_channel,
        channel_weights=channel_weights,
        composite_method=composite_method,
    )
    if stress_table.empty:
        raise ValueError("build_stress_index_table 결과가 비어 있습니다.")
    logger.info(
        "stress_index 완료: columns=%s, composite_method=%s",
        list(stress_table.columns), composite_method,
    )

    # 종합 백분위 시계열 (UI 시계열 차트용).
    # v18: 기본값은 정규분포 선형 근사가 아니라 과거 rolling 분포 내 empirical rank.
    composite_pct_series = _percentile_series(
        stress_table["composite"], method=percentile_method,
    )
    composite_pct_series.name = "composite_pct"

    # 채널 백분위 시계열 (클릭 시 분해 차트용)
    channel_pct_panel = stress_table[["S1", "S2", "S3", "S4", "S5"]].apply(
        lambda col: _percentile_series(col, method=percentile_method),
        axis=0,
    )

    # 변수별 백분위 시계열 (현재값 조회용)
    variable_pct_panel = z_panel.apply(
        lambda col: _percentile_series(col, method=percentile_method),
        axis=0,
    )

    # -----------------------------------------------------------------
    # 6) 패턴 분류 (전체 히스토리)
    # -----------------------------------------------------------------
    thresholds = PatternThresholds()
    pattern_series: pd.Series = classify_history(stress_table, thresholds=thresholds)

    # -----------------------------------------------------------------
    # 7) 진단 기준일(as_of) 결정 및 현재값 추출
    # -----------------------------------------------------------------
    as_of_ts = _resolve_as_of(z_panel, as_of)
    logger.info("as_of 결정: %s", as_of_ts.date())

    # 7-a) 채널/종합 점수
    stress_row = _safe_series_row(stress_table, as_of_ts)
    channel_scores: dict[int, float] = {}
    channel_percentiles: dict[int, float] = {}
    for ch in (1, 2, 3, 4, 5):
        col = f"S{ch}"
        if col in stress_row.index:
            v = float(stress_row[col]) if pd.notna(stress_row[col]) else float("nan")
        else:
            v = float("nan")
        channel_scores[ch] = v
        pct_col = channel_pct_panel[col] if col in channel_pct_panel.columns else pd.Series(dtype="float64")
        channel_percentiles[ch] = _lookup_percentile(pct_col, as_of_ts, v)

    composite_raw = stress_row.get("composite", float("nan")) if not stress_row.empty else float("nan")
    composite_score = float(composite_raw) if pd.notna(composite_raw) else float("nan")
    composite_percentile = _lookup_percentile(
        composite_pct_series, as_of_ts, composite_score,
    )

    # 7-b) 변수별 z-score / 백분위
    z_row = _safe_series_row(z_panel, as_of_ts)
    variable_z_scores: dict[str, float] = {}
    variable_percentiles: dict[str, float] = {}
    for code in z_panel.columns:
        if code in z_row.index:
            zv = float(z_row[code]) if pd.notna(z_row[code]) else float("nan")
        else:
            zv = float("nan")
        variable_z_scores[code] = zv
        pct_col = variable_pct_panel[code] if code in variable_pct_panel.columns else pd.Series(dtype="float64")
        variable_percentiles[code] = _lookup_percentile(pct_col, as_of_ts, zv)

    # 7-c) 현재 패턴 (classify_at으로 직접 계산 - history에 의존하지 않음)
    s1 = channel_scores.get(1, float("nan"))
    # MA30/MA60 of S1: stress_table에서 as_of_ts 이하 구간을 사용
    s1_series = stress_table.get("S1", pd.Series(dtype="float64", index=stress_table.index))
    s1_window = s1_series.loc[:as_of_ts]
    if len(s1_window) >= 1:
        ma30 = float(s1_window.tail(thresholds.real_ma_short).mean())
        ma60 = float(s1_window.tail(thresholds.real_ma_long).mean())
    else:
        ma30 = float("nan")
        ma60 = float("nan")

    pattern_label = classify_at(
        s1=channel_scores.get(1, float("nan")),
        s2=channel_scores.get(2, float("nan")),
        s3=channel_scores.get(3, float("nan")),
        s4=channel_scores.get(4, float("nan")),
        s5=channel_scores.get(5, float("nan")),
        ma30_s1=ma30,
        ma60_s1=ma60,
        thresholds=thresholds,
    )
    logger.info(
        "현재 진단: pattern=%s, composite=%.3f (pct=%.1f)",
        pattern_label, composite_score, composite_percentile,
    )

    # -----------------------------------------------------------------
    # 7-d) 이벤트/변동성 regime 기반 임계값 보정
    # -----------------------------------------------------------------
    kospi_series = aligned.get("KOSPI") if "KOSPI" in aligned.columns else None
    calibration = build_threshold_calibration(
        stress_table,
        price_series=kospi_series,
        as_of=as_of_ts,
    )

    # -----------------------------------------------------------------
    # 8) 유사 시점 K개 + forward returns (KOSPI 기준)
    # -----------------------------------------------------------------
    # KOSPI 시리즈를 가격 시계열로 사용. 없으면 빈 결과로 폴백.
    kospi_code = "KOSPI"
    similar_df: pd.DataFrame
    if kospi_code in aligned.columns and not aligned[kospi_code].dropna().empty:
        price_series = aligned[kospi_code].dropna()
        try:
            similar_df = find_similar_with_forward_returns(
                channel_scores=stress_table[["S1", "S2", "S3", "S4", "S5"]],
                price_series=price_series,
                query_date=as_of_ts,
                k=k_neighbors,
                exclude_recent_days=exclude_recent_days,
                horizons=horizons,
            )
        except ValueError as e:
            logger.warning("유사 시점 탐색 실패: %s. 빈 결과로 폴백.", e)
            similar_df = pd.DataFrame(
                columns=["distance", "S1", "S2", "S3", "S4", "S5"]
                + [f"fwd_{h}d" for h in horizons]
            )
    else:
        logger.warning(
            "KOSPI 시리즈가 없어 유사 시점 탐색을 건너뜁니다. "
            "(변수 코드 'KOSPI'가 target_variables에 있는지 확인)"
        )
        similar_df = pd.DataFrame(
            columns=["distance", "S1", "S2", "S3", "S4", "S5"]
            + [f"fwd_{h}d" for h in horizons]
        )

    forward_returns_summary = _summarize_forward_returns(similar_df, horizons=horizons)

    # -----------------------------------------------------------------
    # 9) 진원지 추적
    # -----------------------------------------------------------------
    origin_res: OriginResult = track_origin(
        z_panel,
        variable_to_channel=variable_to_channel,
        as_of=as_of_ts,
        threshold=origin_threshold,
        lookback_days=origin_lookback_days,
    )
    logger.info(
        "진원지: variable=%s, channel=%s, first_breach=%s",
        origin_res.origin_variable, origin_res.origin_channel,
        origin_res.origin_first_breach_date,
    )

    # -----------------------------------------------------------------
    # 10) 결과 조립
    # -----------------------------------------------------------------
    data_period = (
        aligned.index[0] if len(aligned.index) else None,
        aligned.index[-1] if len(aligned.index) else None,
    )
    result = DiagnosisResult(
        as_of_date=as_of_ts,
        composite_score=composite_score,
        composite_percentile=composite_percentile,
        pattern_label=pattern_label,
        channel_scores=channel_scores,
        channel_percentiles=channel_percentiles,
        variable_z_scores=variable_z_scores,
        variable_percentiles=variable_percentiles,
        origin_result=origin_res,
        similar_dates=similar_df,
        forward_returns_summary=forward_returns_summary,
        failed_variables=failed_variables,
        data_period=data_period,
        composite_pct_series=composite_pct_series,
        channel_pct_panel=channel_pct_panel,
        raw_panel=raw_panel,
        aligned_panel=aligned,
        z_panel=z_panel,
        stress_table=stress_table,
        variable_pct_panel=variable_pct_panel,
        standardization_method=standardization_method,
        percentile_method=percentile_method,
        z_clip_abs=z_clip_abs,
        calibration_summary=calibration.summary,
        calibration_event_metrics=calibration.event_metrics,
        calibration_regime_thresholds=calibration.regime_thresholds,
        calibration_event_labels=calibration.event_label_series,
        calibration_regime_series=calibration.regime_series,
    )
    logger.info("run_full_diagnosis 완료.")
    return result


__all__ = [
    "DiagnosisResult",
    "run_full_diagnosis",
    "z_to_percentile",
]
