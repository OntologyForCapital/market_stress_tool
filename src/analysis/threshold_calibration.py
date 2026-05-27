"""Historical threshold calibration helpers.

The core dashboard still reports interpretable z-scores, but alert thresholds
should not rely only on fixed sigma cutoffs. This module adds two empirical
checks:

1. Event-label calibration: known stress dates are expanded into local event
   windows, then composite thresholds are scored by precision/recall/F1.
2. Volatility-regime thresholds: composite/channel cutoffs are estimated from
   the observed distribution inside low/mid/high KOSPI volatility regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationEvent:
    """A historical stress event used as weak supervision."""

    date: str
    label: str


@dataclass
class ThresholdCalibrationResult:
    """Container returned by build_threshold_calibration."""

    summary: dict[str, object]
    event_metrics: pd.DataFrame
    event_label_series: pd.Series
    regime_thresholds: pd.DataFrame
    regime_series: pd.Series


DEFAULT_CALIBRATION_EVENTS: tuple[CalibrationEvent, ...] = (
    CalibrationEvent("2015-08-24", "China shock"),
    CalibrationEvent("2018-10-29", "미중 무역전쟁"),
    CalibrationEvent("2020-03-19", "코로나"),
    CalibrationEvent("2022-01-27", "Fed 매파 전환"),
    CalibrationEvent("2022-07-04", "Fed 75bp + 인플레 정점"),
    CalibrationEvent("2022-09-30", "영국 미니예산/강달러"),
    CalibrationEvent("2024-08-05", "일본 캐리 청산"),
    CalibrationEvent("2025-04-09", "트럼프 상호관세"),
    CalibrationEvent("2026-03-30", "지정학/정책 스트레스"),
)


REGIME_LABELS_KR: dict[str, str] = {
    "all": "전체",
    "low_vol": "저변동",
    "mid_vol": "중변동",
    "high_vol": "고변동",
    "unknown": "미분류",
}


def _nearest_index_on_or_before(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int | None:
    """Return integer location of the nearest index value on or before ts."""
    valid = index[index <= ts]
    if len(valid) == 0:
        return None
    return int(index.get_loc(valid[-1]))


def label_event_windows(
    index: pd.DatetimeIndex,
    events: Iterable[CalibrationEvent] = DEFAULT_CALIBRATION_EVENTS,
    window_days: int = 10,
) -> pd.Series:
    """Mark business-day windows around known stress events.

    Args:
        index: Sorted DatetimeIndex to label.
        events: Stress events.
        window_days: Number of index rows before/after each event to label True.

    Returns:
        Boolean Series indexed like ``index``.
    """
    labels = pd.Series(False, index=index, name="event_label")
    if len(index) == 0:
        return labels

    for event in events:
        loc = _nearest_index_on_or_before(index, pd.Timestamp(event.date))
        if loc is None:
            continue
        start = max(0, loc - window_days)
        end = min(len(index) - 1, loc + window_days)
        labels.iloc[start:end + 1] = True
    return labels


def _score_binary_threshold(
    score: pd.Series,
    labels: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification metrics for score >= threshold."""
    pred = score >= threshold
    truth = labels.astype(bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    alert_rate = float(pred.mean()) if len(pred) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": alert_rate,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def calibrate_event_thresholds(
    stress_table: pd.DataFrame,
    events: Iterable[CalibrationEvent] = DEFAULT_CALIBRATION_EVENTS,
    score_col: str = "composite",
    event_window_days: int = 10,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Score composite thresholds against historical event windows.

    The returned DataFrame is sorted by best F1 first. Threshold candidates are
    empirical quantiles, which avoids assuming a normal distribution.
    """
    if stress_table.empty or score_col not in stress_table.columns:
        return pd.DataFrame()

    score = stress_table[score_col].dropna()
    if len(score) < min_observations:
        return pd.DataFrame()

    labels = label_event_windows(
        pd.DatetimeIndex(score.index),
        events=events,
        window_days=event_window_days,
    )
    labels = labels.reindex(score.index).fillna(False)
    if int(labels.sum()) == 0:
        return pd.DataFrame()

    quantile_grid = np.linspace(0.50, 0.99, 50)
    candidates = np.unique(score.quantile(quantile_grid).dropna().to_numpy())
    rows: list[dict[str, float]] = []
    for threshold in candidates:
        row = _score_binary_threshold(score, labels, float(threshold))
        row["threshold_percentile"] = float((score <= threshold).mean() * 100.0)
        row["positive_days"] = float(labels.sum())
        row["total_days"] = float(len(score))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def classify_volatility_regime(
    price_series: pd.Series | None,
    target_index: pd.DatetimeIndex,
    lookback_days: int = 63,
) -> pd.Series:
    """Classify each date into low/mid/high realized-volatility regimes."""
    regime = pd.Series("unknown", index=target_index, name="vol_regime")
    if price_series is None or price_series.empty or len(target_index) == 0:
        return regime

    price = price_series.sort_index().reindex(target_index).ffill()
    returns = price.pct_change(fill_method=None)
    min_periods = max(10, lookback_days // 2)
    vol = returns.rolling(lookback_days, min_periods=min_periods).std() * np.sqrt(252.0)
    valid = vol.dropna()
    if len(valid) < min_periods:
        return regime

    q_low, q_high = valid.quantile([1 / 3, 2 / 3])
    regime.loc[vol <= q_low] = "low_vol"
    regime.loc[(vol > q_low) & (vol <= q_high)] = "mid_vol"
    regime.loc[vol > q_high] = "high_vol"
    return regime


def compute_regime_thresholds(
    stress_table: pd.DataFrame,
    price_series: pd.Series | None,
    score_cols: tuple[str, ...] = ("composite", "S1", "S2", "S3", "S4", "S5"),
    lookback_days: int = 63,
    quantiles: tuple[float, ...] = (0.67, 0.80, 0.90, 0.95),
) -> tuple[pd.DataFrame, pd.Series]:
    """Estimate empirical score thresholds by volatility regime."""
    if stress_table.empty:
        return pd.DataFrame(), pd.Series(dtype="object", name="vol_regime")

    cols = [c for c in score_cols if c in stress_table.columns]
    regime_series = classify_volatility_regime(
        price_series,
        pd.DatetimeIndex(stress_table.index),
        lookback_days=lookback_days,
    )

    rows: list[dict[str, object]] = []
    regime_masks = {"all": pd.Series(True, index=stress_table.index)}
    for regime_key in ("low_vol", "mid_vol", "high_vol", "unknown"):
        regime_masks[regime_key] = regime_series == regime_key

    for regime_key, mask in regime_masks.items():
        subset = stress_table.loc[mask, cols]
        if subset.dropna(how="all").empty:
            continue
        for metric in cols:
            s = subset[metric].dropna()
            if s.empty:
                continue
            row: dict[str, object] = {
                "regime": regime_key,
                "regime_kr": REGIME_LABELS_KR.get(regime_key, regime_key),
                "metric": metric,
                "count": int(len(s)),
            }
            for q in quantiles:
                row[f"q{int(round(q * 100))}"] = float(s.quantile(q))
            rows.append(row)

    return pd.DataFrame(rows), regime_series


def _current_row_asof(df: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    if as_of is None:
        return df.iloc[-1]
    valid = df.index[df.index <= pd.Timestamp(as_of)]
    if len(valid) == 0:
        return df.iloc[0]
    return df.loc[valid[-1]]


def build_threshold_calibration(
    stress_table: pd.DataFrame,
    price_series: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
    events: Iterable[CalibrationEvent] = DEFAULT_CALIBRATION_EVENTS,
    event_window_days: int = 10,
    vol_lookback_days: int = 63,
) -> ThresholdCalibrationResult:
    """Build a combined event/regime calibration report."""
    if stress_table.empty:
        empty_labels = pd.Series(dtype="bool", name="event_label")
        return ThresholdCalibrationResult({}, pd.DataFrame(), empty_labels, pd.DataFrame(), pd.Series(dtype="object"))

    event_metrics = calibrate_event_thresholds(
        stress_table,
        events=events,
        event_window_days=event_window_days,
    )
    event_labels = label_event_windows(
        pd.DatetimeIndex(stress_table.index),
        events=events,
        window_days=event_window_days,
    )
    regime_thresholds, regime_series = compute_regime_thresholds(
        stress_table,
        price_series=price_series,
        lookback_days=vol_lookback_days,
    )

    current_row = _current_row_asof(stress_table, as_of)
    current_score = float(current_row.get("composite", np.nan))
    if as_of is None:
        as_of_ts = stress_table.index[-1]
    else:
        valid = stress_table.index[stress_table.index <= pd.Timestamp(as_of)]
        as_of_ts = valid[-1] if len(valid) else stress_table.index[0]

    current_regime = "unknown"
    if not regime_series.empty:
        current_regime = str(regime_series.loc[as_of_ts])

    label_threshold = float("nan")
    label_precision = float("nan")
    label_recall = float("nan")
    label_f1 = float("nan")
    if not event_metrics.empty:
        best = event_metrics.iloc[0]
        label_threshold = float(best["threshold"])
        label_precision = float(best["precision"])
        label_recall = float(best["recall"])
        label_f1 = float(best["f1"])

    regime_row = pd.DataFrame()
    if not regime_thresholds.empty:
        mask = (
            (regime_thresholds["regime"] == current_regime)
            & (regime_thresholds["metric"] == "composite")
        )
        regime_row = regime_thresholds.loc[mask]
        if regime_row.empty:
            regime_row = regime_thresholds.loc[
                (regime_thresholds["regime"] == "all")
                & (regime_thresholds["metric"] == "composite")
            ]

    q80 = q90 = q95 = float("nan")
    if not regime_row.empty:
        q80 = float(regime_row.iloc[0].get("q80", np.nan))
        q90 = float(regime_row.iloc[0].get("q90", np.nan))
        q95 = float(regime_row.iloc[0].get("q95", np.nan))

    calibrated_level = "unknown"
    if pd.notna(current_score):
        if pd.notna(label_threshold) and current_score >= label_threshold:
            calibrated_level = "event_alert"
        elif pd.notna(q95) and current_score >= q95:
            calibrated_level = "tail_alert"
        elif pd.notna(q90) and current_score >= q90:
            calibrated_level = "high_regime"
        elif pd.notna(q80) and current_score >= q80:
            calibrated_level = "watch"
        else:
            calibrated_level = "normal"

    summary: dict[str, object] = {
        "event_window_days": int(event_window_days),
        "event_positive_days": int(event_labels.sum()),
        "event_total_days": int(event_labels.notna().sum()),
        "current_score": current_score,
        "current_regime": current_regime,
        "current_regime_kr": REGIME_LABELS_KR.get(current_regime, current_regime),
        "label_threshold": label_threshold,
        "label_precision": label_precision,
        "label_recall": label_recall,
        "label_f1": label_f1,
        "regime_q80": q80,
        "regime_q90": q90,
        "regime_q95": q95,
        "calibrated_level": calibrated_level,
    }

    return ThresholdCalibrationResult(
        summary=summary,
        event_metrics=event_metrics,
        event_label_series=event_labels,
        regime_thresholds=regime_thresholds,
        regime_series=regime_series,
    )


__all__ = [
    "CalibrationEvent",
    "DEFAULT_CALIBRATION_EVENTS",
    "REGIME_LABELS_KR",
    "ThresholdCalibrationResult",
    "build_threshold_calibration",
    "calibrate_event_thresholds",
    "classify_volatility_regime",
    "compute_regime_thresholds",
    "label_event_windows",
]
