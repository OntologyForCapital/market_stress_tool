"""시각화 유틸 (plotly 기반).

제공 함수:
    - make_channel_bar_chart   : 5개 채널 가로 막대 (백분위 색상)
    - make_geo_risk_map        : 변수별 지리 마커 지도 (scattergeo)
    - make_composite_timeseries: 종합 백분위 시계열 (33/67 점선 임계)
    - make_channel_breakdown   : 특정 일자의 5채널 분해 막대
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui import labels as L
from src.ui.labels import (
    CHANNEL_LABELS_KR,
    VARIABLE_GEO_COORDS,
    LEVEL_COLORS,
)


# =============================================================================
# 1) 채널별 가로 막대 (메인 좌측)
# =============================================================================
def make_channel_bar_chart(
    channel_percentiles: dict[int, float],
    title: str | None = None,
    height: int = 320,
) -> go.Figure:
    """5개 채널(S1~S5)의 백분위를 가로 막대로 표시.

    Args:
        channel_percentiles: {1: pct, 2: pct, ..., 5: pct}.
        title: 차트 제목.
        height: 차트 높이(px).

    Returns:
        plotly Figure.
    """
    # 위에서 아래 S1→S5 순서로 보이려면 y축 카테고리는 역순으로 입력
    channels_top_down = [1, 2, 3, 4, 5]
    y_labels = [CHANNEL_LABELS_KR[c] for c in channels_top_down]
    x_values = [channel_percentiles.get(c, float("nan")) for c in channels_top_down]
    colors = [L.percentile_to_color(v) for v in x_values]
    text_labels = [
        f"{v:.1f}점" if (v is not None and not math.isnan(v)) else "—"
        for v in x_values
    ]

    fig = go.Figure(
        go.Bar(
            x=x_values,
            y=y_labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.15)", width=1)),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>백분위: %{x:.1f}점<extra></extra>",
        )
    )
    fig.update_layout(
        title=title if title else "",
        xaxis=dict(
            title="백분위 (0~100)",
            range=[0, 105],
            tickvals=[0, 33, 50, 67, 100],
            ticktext=["0", "33", "50", "67", "100"],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=y_labels[::-1],  # 위→아래로 S1→S5
            autorange="reversed",
        ),
        height=height,
        margin=dict(l=10, r=30, t=50, b=40),
        plot_bgcolor="white",
        showlegend=False,
    )
    # 33점, 67점 임계 점선 (정상/주의/위험 경계선)
    for x_thr, line_color in [(33, "#4CAF50"), (67, "#F44336")]:
        fig.add_vline(
            x=x_thr, line_width=1.5, line_dash="dash",
            line_color=line_color, opacity=0.5,
        )
    return fig


# =============================================================================
# 2) 변수별 위험 지도 (지리적 배치)
# =============================================================================
def make_geo_risk_map(
    variable_z_scores: dict[str, float],
    variable_percentiles: dict[str, float],
    variable_to_channel: dict[str, int],
    variables_meta: dict | None = None,
    height: int = 500,
) -> go.Figure:
    """변수별 z-score/백분위를 지구본에 마커로 표시.

    Args:
        variable_z_scores: {code: z}.
        variable_percentiles: {code: pct} (0~100).
        variable_to_channel: {code: channel_int}.
        variables_meta: {code: {'name_kr': ..., ...}} - yaml에서 로드한 메타.
        height: 차트 높이(px).

    Returns:
        plotly Figure (scattergeo).
    """
    lats: list[float] = []
    lons: list[float] = []
    sizes: list[float] = []
    colors: list[str] = []
    hover_texts: list[str] = []
    short_labels: list[str] = []

    for code, (lat, lon) in VARIABLE_GEO_COORDS.items():
        if code not in variable_z_scores and code not in variable_percentiles:
            continue
        z = variable_z_scores.get(code, float("nan"))
        pct = variable_percentiles.get(code, float("nan"))

        # NaN은 작은 회색 마커로
        if math.isnan(pct):
            size = 8.0
            color = "#BDBDBD"
        else:
            # 마커 크기: |z| 절댓값 (8~30)
            abs_z = abs(z) if not math.isnan(z) else 0.5
            size = float(np.clip(8.0 + abs_z * 8.0, 8.0, 30.0))
            color = L.percentile_to_color(pct)

        ch = variable_to_channel.get(code)
        ch_kr = CHANNEL_LABELS_KR.get(ch, "—") if ch is not None else "—"
        name_kr = L.variable_to_korean(code, variables_meta=variables_meta)

        z_str = f"{z:+.2f}σ" if not math.isnan(z) else "—"
        pct_str = f"{pct:.1f}점" if not math.isnan(pct) else "—"

        hover_texts.append(
            f"<b>{name_kr}</b> ({code})<br>"
            f"채널: {ch_kr}<br>"
            f"z-score: {z_str}<br>"
            f"백분위: {pct_str}"
        )
        short_labels.append(
            f"{code} {pct:.0f}" if not math.isnan(pct) else code
        )

        lats.append(lat)
        lons.append(lon)
        sizes.append(size)
        colors.append(color)

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=short_labels,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
            mode="markers+text",
            textposition="top right",
            textfont=dict(size=10, color="#333333"),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                opacity=0.85,
            ),
        )
    )
    fig.update_geos(
        projection_type="equirectangular",
        showcountries=True,
        countrycolor="rgba(0,0,0,0.3)",
        showcoastlines=True,
        coastlinecolor="rgba(0,0,0,0.3)",
        showland=True,
        landcolor="#F5F5F5",
        showocean=True,
        oceancolor="#EAF4FB",
        showframe=False,
        lataxis=dict(range=[-20, 70]),
        lonaxis=dict(range=[-130, 160]),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    return fig


# =============================================================================
# 3) 종합 백분위 시계열
# =============================================================================
def make_composite_timeseries(
    composite_pct_series: pd.Series,
    title: str | None = None,
    height: int = 360,
    show_threshold_lines: bool = True,
) -> go.Figure:
    """종합 스트레스 백분위 시계열.

    Args:
        composite_pct_series: 인덱스가 날짜인 Series (값: 0~100 백분위).
        title: 차트 제목.
        height: 차트 높이.
        show_threshold_lines: True면 33/67 가로 점선 표시.

    Returns:
        plotly Figure.
    """
    s = composite_pct_series.dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            line=dict(color="#1F77B4", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>종합 백분위: %{y:.1f}점<extra></extra>",
            name="종합 백분위",
        )
    )

    if show_threshold_lines:
        fig.add_hline(
            y=33, line_width=1, line_dash="dot",
            line_color=LEVEL_COLORS["low"],
            annotation_text=L.UI_TEXTS["ts_low_threshold_label"],
            annotation_position="bottom right",
            annotation_font_size=10,
        )
        fig.add_hline(
            y=67, line_width=1, line_dash="dot",
            line_color=LEVEL_COLORS["high"],
            annotation_text=L.UI_TEXTS["ts_high_threshold_label"],
            annotation_position="top right",
            annotation_font_size=10,
        )

    fig.update_layout(
        title=title if title else "",
        height=height,
        margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="white",
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(
            title="백분위 (0~100)",
            range=[-5, 105],
            tickvals=[0, 33, 50, 67, 100],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


# =============================================================================
# 4) 특정 일자의 5채널 분해 (시계열 클릭 시 표시)
# =============================================================================
def make_channel_breakdown(
    z_to_pct_fn,
    stress_row: pd.Series,
    date_str: str,
    height: int = 260,
) -> go.Figure:
    """특정 일자의 5채널 + 종합을 가로 막대로 분해 표시.

    Args:
        z_to_pct_fn: z-score(float) → percentile(float) 함수 (`pipeline.z_to_percentile`).
        stress_row: stress_table의 한 행 (S1~S5, composite 컬럼 포함).
        date_str: 표시용 날짜 문자열.
        height: 차트 높이.

    Returns:
        plotly Figure.
    """
    channels_top_down = [1, 2, 3, 4, 5]
    y_labels = [CHANNEL_LABELS_KR[c] for c in channels_top_down]
    z_values = [float(stress_row.get(f"S{c}", float("nan"))) for c in channels_top_down]
    pct_values = [z_to_pct_fn(z) for z in z_values]
    colors = [L.percentile_to_color(p) for p in pct_values]
    text_labels = [
        f"{p:.1f}점 (z={z:+.2f})" if not math.isnan(p) else "—"
        for p, z in zip(pct_values, z_values)
    ]

    fig = go.Figure(
        go.Bar(
            x=pct_values,
            y=y_labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.15)", width=1)),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{date_str} 채널 분해",
        xaxis=dict(
            title="백분위", range=[0, 110],
            tickvals=[0, 33, 50, 67, 100],
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(autorange="reversed"),
        height=height,
        margin=dict(l=10, r=30, t=40, b=40),
        plot_bgcolor="white",
        showlegend=False,
    )
    for x_thr in (33, 67):
        fig.add_vline(
            x=x_thr, line_width=1, line_dash="dot",
            line_color="rgba(0,0,0,0.35)",
        )
    return fig


# =============================================================================
# 5) SVG 추상 지도 (plotly scattergeo 대체용 — 가벼움/렉 없음)
# =============================================================================
def make_geo_risk_map_svg(
    variable_z_scores: dict[str, float],
    variable_percentiles: dict[str, float],
    variable_to_channel: dict[str, int],
    variables_meta: dict | None = None,
    width: int = 720,
    height: int = 420,
) -> str:
    """변수별 위험도를 추상 그리드 지도(정적 SVG, HTML 문자열)로 렌더.

    plotly scattergeo는 큰 GeoJSON을 그릴 때 렉이 심하므로,
    위경도를 간단한 직사각형 좌표계(equirectangular)로 매핑하여
    경량 SVG를 직접 만든다.

    Args:
        variable_z_scores: {code: z}.
        variable_percentiles: {code: pct} (0~100).
        variable_to_channel: {code: channel_int}.
        variables_meta: {code: {'name_kr': ..., ...}}.
        width: SVG 폭(px).
        height: SVG 높이(px).

    Returns:
        HTML 문자열 (`st.markdown(..., unsafe_allow_html=True)` 또는
        `st.components.v1.html`에 그대로 넘기면 됨).
    """
    # 표시 범위 (메인 탭 지도와 동일)
    LAT_MIN, LAT_MAX = -20.0, 70.0
    LON_MIN, LON_MAX = -130.0, 160.0

    def project(lat: float, lon: float) -> tuple[float, float]:
        """위경도 → SVG 픽셀 좌표."""
        x = (lon - LON_MIN) / (LON_MAX - LON_MIN) * width
        y = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * height
        return x, y

    # ── 배경: 추상 "대륙" 윤곽 (간단한 폴리곤; 자유 라이선스, 자체 작성) ──
    # 좌표는 (lat, lon) 페어 리스트. equirectangular 투영으로 정확하진 않지만,
    # "미국/유럽/아시아/한국/일본" 지역감을 주기에 충분.
    continents = {
        "north_america": [
            (60, -130), (70, -100), (60, -75), (45, -65), (25, -80),
            (15, -95), (30, -115), (50, -125),
        ],
        "south_america": [
            (10, -75), (5, -55), (-20, -45), (-40, -65), (-30, -75),
            (-5, -80),
        ],
        "europe": [
            (60, -10), (70, 30), (60, 50), (45, 40), (40, 15), (50, 0),
        ],
        "africa": [
            (35, -10), (35, 30), (10, 45), (-10, 40), (-35, 20),
            (-20, 10), (5, 0),
        ],
        "asia": [
            (70, 50), (70, 140), (50, 145), (35, 130), (25, 110),
            (10, 100), (25, 75), (40, 55),
        ],
        "oceania": [
            (-10, 115), (-15, 145), (-35, 150), (-35, 115),
        ],
    }

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'style="width:100%; height:auto; max-height:{height}px; '
        f'background:#EAF4FB; border-radius:6px;">'
    ]
    # 대륙 폴리곤
    for _name, coords in continents.items():
        points = " ".join(f"{project(la, lo)[0]:.1f},{project(la, lo)[1]:.1f}" for la, lo in coords)
        svg_parts.append(
            f'<polygon points="{points}" fill="#F5F5F5" '
            f'stroke="rgba(0,0,0,0.25)" stroke-width="1"/>'
        )

    # 격자선 (위도/경도 30도 간격)
    for lon in range(-120, 161, 30):
        x, _ = project(0, lon)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
            f'stroke="rgba(0,0,0,0.06)" stroke-width="1"/>'
        )
    for lat in range(-20, 71, 20):
        _, y = project(lat, 0)
        svg_parts.append(
            f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" '
            f'stroke="rgba(0,0,0,0.06)" stroke-width="1"/>'
        )

    # ── 마커 ──
    # 같은 좌표에 마커가 겹치는 일을 방지하기 위해, 이미 사용된 픽셀 좌표 근처에는
    # 살짝 오프셋을 준다 (단순한 grid scatter).
    used_positions: list[tuple[float, float]] = []

    def find_free_position(x: float, y: float, min_dist: float = 26.0) -> tuple[float, float]:
        cur_x, cur_y = x, y
        for _ in range(8):
            if all(((cur_x - ux) ** 2 + (cur_y - uy) ** 2) ** 0.5 >= min_dist for ux, uy in used_positions):
                return cur_x, cur_y
            cur_x += min_dist * 0.6
            cur_y += min_dist * 0.4
        return cur_x, cur_y

    markers_html: list[str] = []
    for code, (lat, lon) in VARIABLE_GEO_COORDS.items():
        if code not in variable_z_scores and code not in variable_percentiles:
            continue
        z = variable_z_scores.get(code, float("nan"))
        pct = variable_percentiles.get(code, float("nan"))

        if math.isnan(pct):
            radius = 6.0
            color = "#BDBDBD"
        else:
            abs_z = abs(z) if not math.isnan(z) else 0.5
            # 직경 8~30px 범위 → 반지름 4~15
            radius = float(np.clip(4.0 + abs_z * 4.0, 4.0, 15.0))
            color = L.percentile_to_color(pct)

        ch = variable_to_channel.get(code)
        ch_kr = CHANNEL_LABELS_KR.get(ch, "—") if ch is not None else "—"
        name_kr = L.variable_to_korean(code, variables_meta=variables_meta)
        z_str = f"{z:+.2f}σ" if not math.isnan(z) else "—"
        pct_str = f"{pct:.0f}점" if not math.isnan(pct) else "—"

        px, py = project(lat, lon)
        px, py = find_free_position(px, py)
        used_positions.append((px, py))

        tooltip = f"{name_kr} ({code}) · {ch_kr} · z {z_str} · 백분위 {pct_str}"
        # 이스케이프
        tooltip = tooltip.replace('"', "'")

        markers_html.append(
            f'<g><circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.85" '
            f'stroke="rgba(0,0,0,0.4)" stroke-width="1">'
            f'<title>{tooltip}</title></circle>'
            f'<text x="{px + radius + 3:.1f}" y="{py + 3:.1f}" '
            f'font-size="10" fill="#333" font-family="sans-serif">{code}</text></g>'
        )

    svg_parts.extend(markers_html)

    # 범례 (좌하단)
    legend_y = height - 18
    svg_parts.append(
        f'<g font-size="11" font-family="sans-serif" fill="#333">'
        f'<circle cx="14" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["low"]}"/>'
        f'<text x="24" y="{legend_y + 4}">정상 (0~33)</text>'
        f'<circle cx="110" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["mid"]}"/>'
        f'<text x="120" y="{legend_y + 4}">주의 (34~66)</text>'
        f'<circle cx="220" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["high"]}"/>'
        f'<text x="230" y="{legend_y + 4}">위험 (67~100)</text>'
        f'<text x="330" y="{legend_y + 4}" fill="#666">· 원 크기 = |z-score|</text>'
        f'</g>'
    )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


__all__ = [
    "make_channel_bar_chart",
    "make_geo_risk_map",
    "make_geo_risk_map_svg",
    "make_composite_timeseries",
    "make_channel_breakdown",
]
