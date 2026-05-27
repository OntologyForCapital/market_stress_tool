"""Plotly 차트 구성 테스트."""

from __future__ import annotations

import pandas as pd

from src.ui.charts import make_composite_timeseries


def test_composite_timeseries_draws_visible_event_annotations():
    idx = pd.date_range("2020-03-16", "2020-03-23", freq="B")
    series = pd.Series([40, 45, 52, 85, 72, 60], index=idx)

    fig = make_composite_timeseries(
        series,
        event_annotations=[
            ("2020-03-19", "코로나"),
            ("2021-01-01", "범위 밖 사건"),
        ],
    )

    annotation_texts = [str(a.text) for a in fig.layout.annotations]
    assert any("코로나" in text for text in annotation_texts)
    assert all("범위 밖 사건" not in text for text in annotation_texts)
    assert fig.layout.margin.b == 128
    assert any(
        str(shape.x0).startswith("2020-03-19") and shape.yref == "paper"
        for shape in fig.layout.shapes
    )


def test_composite_timeseries_keeps_default_margin_without_visible_events():
    idx = pd.date_range("2020-03-16", "2020-03-23", freq="B")
    series = pd.Series([40, 45, 52, 85, 72, 60], index=idx)

    fig = make_composite_timeseries(
        series,
        event_annotations=[("2021-01-01", "범위 밖 사건")],
    )

    annotation_texts = [str(a.text) for a in fig.layout.annotations]
    assert all("범위 밖 사건" not in text for text in annotation_texts)
    assert fig.layout.margin.b == 40


def test_composite_timeseries_staggers_event_label_heights():
    idx = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    series = pd.Series(range(len(idx)), index=idx)

    fig = make_composite_timeseries(
        series,
        event_annotations=[
            ("2022-01-27", "Fed 매파 전환"),
            ("2022-07-04", "Fed 75bp + 인플레 정점"),
            ("2022-09-30", "영국 미니예산/강달러"),
        ],
    )

    event_annotations = [
        a for a in fig.layout.annotations
        if any(token in str(a.text) for token in ("Fed", "영국"))
    ]
    event_y_positions = [a.y for a in event_annotations]
    assert event_y_positions == [-0.14, -0.28, -0.42]
    assert fig.layout.margin.b == 204
