"""Threshold calibration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.threshold_calibration import (
    CalibrationEvent,
    build_threshold_calibration,
    calibrate_event_thresholds,
    classify_volatility_regime,
    compute_regime_thresholds,
    label_event_windows,
)


def test_label_event_windows_marks_nearest_business_days():
    idx = pd.bdate_range("2020-01-01", periods=20)
    events = [CalibrationEvent(idx[10].strftime("%Y-%m-%d"), "test")]

    labels = label_event_windows(idx, events=events, window_days=2)

    assert labels.sum() == 5
    assert labels.iloc[8]
    assert labels.iloc[12]
    assert not labels.iloc[7]


def test_event_threshold_calibration_prefers_event_spikes():
    idx = pd.bdate_range("2020-01-01", periods=260)
    score = pd.Series(0.5, index=idx)
    event_date = idx[130]
    score.iloc[128:133] = 3.0
    stress = pd.DataFrame({"composite": score})
    events = [CalibrationEvent(event_date.strftime("%Y-%m-%d"), "spike")]

    metrics = calibrate_event_thresholds(
        stress,
        events=events,
        event_window_days=2,
        min_observations=30,
    )

    assert not metrics.empty
    assert metrics.iloc[0]["f1"] > 0
    assert metrics.iloc[0]["threshold"] >= 0.5


def test_volatility_regime_and_thresholds_are_computed():
    idx = pd.bdate_range("2020-01-01", periods=260)
    returns = np.r_[np.full(90, 0.001), np.full(80, -0.002), np.linspace(-0.02, 0.02, 90)]
    price = pd.Series(100 * (1 + returns).cumprod(), index=idx)
    stress = pd.DataFrame({
        "composite": np.linspace(0, 3, len(idx)),
        "S1": np.linspace(0, 1, len(idx)),
    }, index=idx)

    regimes = classify_volatility_regime(price, idx, lookback_days=40)
    thresholds, regimes_2 = compute_regime_thresholds(
        stress,
        price_series=price,
        score_cols=("composite", "S1"),
        lookback_days=40,
    )

    assert len(regimes) == len(idx)
    assert len(regimes_2) == len(idx)
    assert not thresholds.empty
    assert {"regime", "metric", "q90"}.issubset(thresholds.columns)


def test_build_threshold_calibration_summary_has_current_level():
    idx = pd.bdate_range("2020-01-01", periods=260)
    stress = pd.DataFrame({
        "composite": np.linspace(0, 3, len(idx)),
        "S1": np.linspace(0, 1, len(idx)),
        "S2": np.linspace(0, 1, len(idx)),
        "S3": np.linspace(0, 1, len(idx)),
        "S4": np.linspace(0, 1, len(idx)),
        "S5": np.linspace(0, 1, len(idx)),
    }, index=idx)
    price = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    events = [CalibrationEvent(idx[-20].strftime("%Y-%m-%d"), "late stress")]

    result = build_threshold_calibration(
        stress,
        price_series=price,
        as_of=idx[-1],
        events=events,
        event_window_days=5,
        vol_lookback_days=40,
    )

    assert "calibrated_level" in result.summary
    assert not result.event_metrics.empty
    assert not result.regime_thresholds.empty
