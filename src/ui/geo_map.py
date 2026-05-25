"""실세계 지도 기반 위험도 SVG 렌더링.

Natural Earth 1:110m admin 0 countries GeoJSON을 사용해 정확한 국가 윤곽을
그린 뒤, VARIABLE_GEO_COORDS의 (위도, 경도)에 색상/크기 마커를 얹는다.

데이터 출처: Natural Earth (https://www.naturalearthdata.com/)
            - public domain
            - 파일: src/ui/world_110m.geojson (820KB, 177국가)
            - 출처 GitHub: nvkelso/natural-earth-vector

렌더 방식:
    - plotly choropleth_mapbox / scattergeo는 매번 데이터를 재전송하여
      Streamlit 환경에서 렉이 큼.
    - 정적 SVG로 한 번 그리면 브라우저가 캐싱하므로 빠르다.
    - 폴리곤 simplification은 1:110m 해상도가 이미 충분히 단순화돼 있어
      추가 가공 없이 사용.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.ui import labels as L
from src.ui.labels import (
    CHANNEL_LABELS_KR,
    LEVEL_COLORS,
    VARIABLE_GEO_COORDS,
)

_GEOJSON_PATH = Path(__file__).parent / "world_110m.geojson"


@lru_cache(maxsize=1)
def _load_geojson() -> dict:
    """GeoJSON을 한 번만 읽고 캐싱."""
    with _GEOJSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _project(lat: float, lon: float, *,
             width: float, height: float,
             lat_min: float, lat_max: float,
             lon_min: float, lon_max: float) -> tuple[float, float]:
    """equirectangular: (lat, lon) → (px, py)."""
    x = (lon - lon_min) / (lon_max - lon_min) * width
    y = (lat_max - lat) / (lat_max - lat_min) * height
    return x, y


def _ring_to_path(
    ring: list[list[float]],
    *,
    width: float,
    height: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """GeoJSON 링([[lon, lat], ...]) → SVG path 'M x y L x y ... Z'."""
    pts = []
    for lon, lat in ring:
        if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
            # 뷰포트 밖이라도 일단 클램핑 (윤곽이 잘릴 뿐 망가지지는 않음)
            lon_c = max(lon_min, min(lon_max, lon))
            lat_c = max(lat_min, min(lat_max, lat))
        else:
            lon_c, lat_c = lon, lat
        x, y = _project(
            lat_c, lon_c,
            width=width, height=height,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        pts.append(f"{x:.1f},{y:.1f}")
    if not pts:
        return ""
    return "M " + " L ".join(pts) + " Z"


def make_world_geo_map_svg(
    variable_z_scores: dict[str, float],
    variable_percentiles: dict[str, float],
    variable_to_channel: dict[str, int],
    variables_meta: dict | None = None,
    width: int = 720,
    height: int = 420,
    lat_range: tuple[float, float] = (-20.0, 70.0),
    lon_range: tuple[float, float] = (-130.0, 160.0),
) -> str:
    """Natural Earth GeoJSON 기반 실세계 지도 + 변수 마커를 SVG로 렌더.

    Args:
        variable_z_scores: {code: z}.
        variable_percentiles: {code: pct} (0~100).
        variable_to_channel: {code: channel_int}.
        variables_meta: {code: {'name_kr': ..., ...}}.
        width: SVG 폭(px).
        height: SVG 높이(px).
        lat_range: 표시 위도 범위.
        lon_range: 표시 경도 범위.

    Returns:
        HTML 문자열. `st.markdown(..., unsafe_allow_html=True)` 또는
        `st.components.v1.html`에 넘기면 렌더됨.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # 강조국 (스트레스 도구에서 중요한 5개 지역)
    HIGHLIGHTED_ISO = {
        "USA": "#FFFFFF",   # 미국
        "KOR": "#FFFFFF",   # 한국
        "JPN": "#FFFFFF",   # 일본
        "CHN": "#FFFFFF",   # 중국
        "SAU": "#FFFFFF",   # 사우디 (중동 대표)
    }

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'style="width:100%; height:auto; max-height:{height}px; '
        f'background:#EAF4FB; border-radius:6px;">'
    ]

    # 경도 격자선 (30도 간격, 옅게)
    for lon in range(int(lon_min) - int(lon_min) % 30, int(lon_max) + 1, 30):
        x, _ = _project(
            0, lon, width=width, height=height,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
            f'stroke="rgba(0,0,0,0.05)" stroke-width="1"/>'
        )
    for lat in range(int(lat_min) - int(lat_min) % 20, int(lat_max) + 1, 20):
        _, y = _project(
            lat, 0, width=width, height=height,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        parts.append(
            f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" '
            f'stroke="rgba(0,0,0,0.05)" stroke-width="1"/>'
        )

    # GeoJSON 국가 렌더
    geojson = _load_geojson()
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        iso_a3 = props.get("ISO_A3") or props.get("ADM0_A3") or ""
        # 강조국 vs 일반국
        if iso_a3 in HIGHLIGHTED_ISO:
            fill = "#FFFFFF"  # 강조국은 흰색 (마커가 눈에 띄도록)
            stroke = "rgba(31, 119, 180, 0.7)"
            stroke_w = 1.0
        else:
            fill = "#EEEEEE"
            stroke = "rgba(0,0,0,0.25)"
            stroke_w = 0.6

        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []

        # MultiPolygon은 [[[ring],[hole]], ...], Polygon은 [[ring],[hole]]
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords
        else:
            continue

        for poly in polygons:
            # poly[0] = outer ring, poly[1:] = holes (1:110m은 hole 거의 없음, 무시)
            if not poly:
                continue
            path_d = _ring_to_path(
                poly[0],
                width=width, height=height,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
            )
            if not path_d:
                continue
            parts.append(
                f'<path d="{path_d}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{stroke_w}" '
                f'fill-rule="evenodd"/>'
            )

    # ── 변수 마커 ──
    # 충돌 회피용: 이미 쓴 픽셀 좌표 근처면 살짝 이동
    used_positions: list[tuple[float, float]] = []

    def find_free(x: float, y: float, min_dist: float = 26.0) -> tuple[float, float]:
        cx, cy = x, y
        for _ in range(8):
            if all(((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5 >= min_dist for ux, uy in used_positions):
                return cx, cy
            cx += min_dist * 0.6
            cy += min_dist * 0.5
        return cx, cy

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
            radius = float(np.clip(4.0 + abs_z * 4.0, 4.0, 15.0))
            color = L.percentile_to_color(pct)

        ch = variable_to_channel.get(code)
        ch_kr = CHANNEL_LABELS_KR.get(ch, "—") if ch is not None else "—"
        name_kr = L.variable_to_korean(code, variables_meta=variables_meta)
        z_str = f"{z:+.2f}σ" if not math.isnan(z) else "—"
        pct_str = f"{pct:.0f}점" if not math.isnan(pct) else "—"

        px, py = _project(
            lat, lon, width=width, height=height,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        px, py = find_free(px, py)
        used_positions.append((px, py))

        tooltip = (
            f"{name_kr} ({code}) · {ch_kr} · z {z_str} · 백분위 {pct_str}"
        ).replace('"', "'")

        parts.append(
            f'<g><circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.9" '
            f'stroke="rgba(0,0,0,0.5)" stroke-width="1.2">'
            f'<title>{tooltip}</title></circle>'
            f'<text x="{px + radius + 3:.1f}" y="{py + 3:.1f}" '
            f'font-size="10" fill="#222" font-family="sans-serif" '
            f'style="paint-order:stroke; stroke:rgba(255,255,255,0.85); '
            f'stroke-width:2px; font-weight:600;">{code}</text></g>'
        )

    # 범례 (좌하단)
    legend_y = height - 18
    parts.append(
        f'<g font-size="11" font-family="sans-serif" fill="#333">'
        f'<rect x="6" y="{legend_y - 12}" width="430" height="22" '
        f'fill="rgba(255,255,255,0.85)" rx="3" ry="3" '
        f'stroke="rgba(0,0,0,0.15)" stroke-width="0.5"/>'
        f'<circle cx="20" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["low"]}"/>'
        f'<text x="30" y="{legend_y + 4}">정상 (0~33)</text>'
        f'<circle cx="116" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["mid"]}"/>'
        f'<text x="126" y="{legend_y + 4}">주의 (34~66)</text>'
        f'<circle cx="226" cy="{legend_y}" r="6" fill="{LEVEL_COLORS["high"]}"/>'
        f'<text x="236" y="{legend_y + 4}">위험 (67~100)</text>'
        f'<text x="336" y="{legend_y + 4}" fill="#666">· 원 크기 = |z-score|</text>'
        f'</g>'
    )

    # 데이터 출처 (작게)
    parts.append(
        f'<text x="{width - 6}" y="{height - 4}" '
        f'font-size="9" fill="rgba(0,0,0,0.45)" text-anchor="end" '
        f'font-family="sans-serif">지도: Natural Earth 1:110m (public domain)</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


__all__ = ["make_world_geo_map_svg"]
