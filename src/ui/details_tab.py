"""세부 내용 탭 — 변수별 상세 정보를 채널별로 그룹화하여 표시.

각 변수에 대해:
    1. 변수 코드 + 한국어 이름
    2. 출처 (FRED/KRX/ECOS/yfinance) + series_id
    3. 채널 (1~5) + 채널 한국어 이름
    4. 위험 방향 (positive/negative) + 설명
    5. 기준일 raw 값
    6. 5년 롤링 z-score
    7. 백분위 (0~100)
    8. 채널 점수 기여도
    9. 1년 시계열 그래프 (z-score 시계열)

본 모듈은 `DiagnosisResult.aligned_panel`(raw 값 시계열)과 `z_panel`(z-score 시계열)을
필요로 한다.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import DiagnosisResult, z_to_percentile
from src.ui import labels as L
from src.ui.labels import (
    CHANNEL_LABELS_KR,
    LEVEL_COLORS,
    RISK_DIRECTION_LABELS_KR,
    SOURCE_LABELS_KR,
    UI_TEXTS,
)


def _last_valid_on_or_before(
    series: pd.Series,
    as_of: pd.Timestamp,
) -> tuple[float, pd.Timestamp | None]:
    """Return the last valid value on or before as_of."""
    if series.empty:
        return float("nan"), None
    prior = series.loc[:as_of].dropna()
    if prior.empty:
        return float("nan"), None
    return float(prior.iloc[-1]), prior.index[-1]


def _mini_timeseries(
    series: pd.Series,
    height: int = 180,
    is_zscore: bool = True,
) -> go.Figure:
    """변수 1개의 최근 1년 시계열 미니 차트."""
    color = "#1F77B4"
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color=color, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}<extra></extra>",
        )
    )
    if is_zscore:
        # +1.5σ / -1.5σ 가이드라인
        fig.add_hline(y=1.5, line_dash="dot", line_color="rgba(244,67,54,0.4)", line_width=1)
        fig.add_hline(y=-1.5, line_dash="dot", line_color="rgba(244,67,54,0.4)", line_width=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(0,0,0,0.2)", line_width=1)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="z-score" if is_zscore else None, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        plot_bgcolor="white",
    )
    return fig


def _format_value(value: float, fmt: str = "{:+.3f}") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return fmt.format(value)


def _color_chip(pct: float) -> str:
    """백분위 → 색칠된 작은 칩 HTML."""
    if math.isnan(pct):
        return '<span style="color:#999;">—</span>'
    color = L.percentile_to_color(pct)
    level_kr = L.percentile_to_level_label_kr(pct)
    return (
        f'<span style="background-color:{color}; color:white; '
        f'padding:2px 8px; border-radius:10px; font-weight:600;">'
        f'{pct:.0f}점 ({level_kr})</span>'
    )


def render_details_tab(
    result: DiagnosisResult,
    variables_meta: dict[str, dict],
    title: str | None = None,
    intro: str | None = None,
    key_prefix: str = "details",
) -> None:
    """세부 내용 탭 렌더.

    Args:
        result: DiagnosisResult — `aligned_panel`, `z_panel`, `variable_z_scores`,
                `variable_percentiles`, `channel_scores` 필드 사용.
        variables_meta: {code: {name_kr, channel, risk_direction, source, ...}}.
    """
    st.markdown(f"### {title or UI_TEXTS['details_title']}")
    st.markdown(intro or UI_TEXTS["details_intro"])

    aligned_panel = result.aligned_panel
    z_panel = result.z_panel

    if (aligned_panel is None or aligned_panel.empty) and (z_panel is None or z_panel.empty):
        st.info("패널 데이터가 없어 시계열을 표시할 수 없습니다.")
        return

    # 1년 윈도우
    as_of = result.as_of_date
    one_year_ago = as_of - pd.Timedelta(days=365)

    # variables.yaml에서 변수 메타 + series_id 가져오기
    # variables_meta는 yaml에서 로드된 dict인데 series_id가 없을 수도 있어 try/get
    # series_id는 variables.yaml의 source_config.series_id 또는 ticker
    # → 메타에 직접 들어있지 않으면 — 표기
    # (메타 구성은 app.load_variables_meta에서 만들어진 것)
    from src.config import load_variables, load_target_variables, load_channel_mapping
    all_vars = list(load_variables()) + list(load_target_variables())
    code_to_obj = {v.code: v for v in all_vars}
    variable_to_channel = load_channel_mapping()

    # 채널 1~5 grouping (target_variables 등 채널 없는 변수는 별도)
    channel_to_codes: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    no_channel_codes: list[str] = []
    for code, var in code_to_obj.items():
        ch = variable_to_channel.get(code)
        if ch in channel_to_codes:
            channel_to_codes[ch].append(code)
        else:
            no_channel_codes.append(code)

    # 채널별로 expander 묶음
    for ch in (1, 2, 3, 4, 5):
        ch_name = CHANNEL_LABELS_KR[ch]
        codes = sorted(channel_to_codes[ch])
        if not codes:
            continue

        # 채널 헤더 (펼쳐서 보이게)
        with st.expander(f"**채널 {ch} · {ch_name}** ({len(codes)}개 변수)", expanded=False):
            for code in codes:
                _render_variable_block(
                    code=code,
                    var_obj=code_to_obj.get(code),
                    result=result,
                    aligned_panel=aligned_panel,
                    z_panel=z_panel,
                    one_year_ago=one_year_ago,
                    variable_to_channel=variable_to_channel,
                    key_prefix=key_prefix,
                )

    # 채널 외 변수 (예: KOSPI 같은 타깃)
    if no_channel_codes:
        with st.expander(f"**타깃·기타 변수** ({len(no_channel_codes)}개)", expanded=False):
            for code in sorted(no_channel_codes):
                _render_variable_block(
                    code=code,
                    var_obj=code_to_obj.get(code),
                    result=result,
                    aligned_panel=aligned_panel,
                    z_panel=z_panel,
                    one_year_ago=one_year_ago,
                    variable_to_channel=variable_to_channel,
                    key_prefix=key_prefix,
                )


def _render_variable_block(
    code: str,
    var_obj: Any,
    result: DiagnosisResult,
    aligned_panel: pd.DataFrame | None,
    z_panel: pd.DataFrame | None,
    one_year_ago: pd.Timestamp,
    variable_to_channel: dict[str, int],
    key_prefix: str,
) -> None:
    """변수 한 개의 상세 블록 렌더."""
    # 메타 정보
    if var_obj is not None:
        name_kr = getattr(var_obj, "name_kr", code)
        source = getattr(var_obj, "source", None) or "—"
        risk_dir = getattr(var_obj, "risk_direction", None)
        # v17: 사전 변환 종류 (v13 도입, yaml의 transform 필드). 기본값 'level'.
        transform = getattr(var_obj, "transform", None)
        # 작업 3 (v12): Variable 데이터클래스는 series_id를 최상위 속성으로 노출.
        # ECOS 변수는 item_code를 병기하는 경우가 있으므로 fallback 구성.
        sid = getattr(var_obj, "series_id", None)
        item_code = getattr(var_obj, "item_code", None)
        fallback_sid = getattr(var_obj, "fallback_series_id", None)
        if sid and item_code:
            series_id = f"{sid} / {item_code}"
        else:
            series_id = sid or item_code or fallback_sid or "—"
    else:
        name_kr = code
        source = "—"
        risk_dir = None
        transform = None
        series_id = "—"

    ch = variable_to_channel.get(code)
    ch_kr = CHANNEL_LABELS_KR.get(ch, "—") if ch is not None else "—"
    source_kr = SOURCE_LABELS_KR.get(source, source)
    risk_dir_kr = (
        RISK_DIRECTION_LABELS_KR.get(risk_dir, risk_dir or "—")
        if risk_dir
        else "—"
    )
    # v17: transform 한국어 라벨 (메타 없으면 '—').
    transform_kr = L.transform_to_korean(transform) if transform else "—"

    # 현재 값 + z + 백분위
    z_val = result.variable_z_scores.get(code, float("nan"))
    pct_val = result.variable_percentiles.get(code, float("nan"))

    # 작업 3 (v12): 월별 변수(INDPRO, KR_EXPORT 등)은 ffill 30일 한도 만료로
    # variable_z_scores에 NaN이 들어갈 수 있다. z_panel의 마지막 유효값으로 보관.
    z_is_stale = False
    last_valid_z_date: pd.Timestamp | None = None
    as_of = result.as_of_date
    if math.isnan(z_val) and z_panel is not None and not z_panel.empty and code in z_panel.columns:
        z_val, last_valid_z_date = _last_valid_on_or_before(z_panel[code], as_of)
        if last_valid_z_date is not None:
            z_is_stale = True
            if math.isnan(pct_val):
                pct_panel = getattr(result, "variable_pct_panel", None)
                if pct_panel is not None and not pct_panel.empty and code in pct_panel.columns:
                    pct_val, _ = _last_valid_on_or_before(pct_panel[code], as_of)
                if math.isnan(pct_val):
                    pct_val = float(z_to_percentile(z_val))

    # raw 값: 기준일 이하 마지막 유효 정합값
    raw_val = float("nan")
    last_valid_raw_date: pd.Timestamp | None = None
    if aligned_panel is not None and not aligned_panel.empty and code in aligned_panel.columns:
        raw_val, last_valid_raw_date = _last_valid_on_or_before(aligned_panel[code], as_of)

    # 채널 점수 기여도: 채널 내 변수 개수로 균등 가중 + 위험방향 부호
    # 작업 3 (v12): z_val이 stale이더라도 의미있는 수치면 기여도 계산.
    contrib = float("nan")
    if ch is not None and not math.isnan(z_val):
        ch_codes = [c for c, cc in variable_to_channel.items() if cc == ch]
        n = len(ch_codes)
        if n > 0:
            sign = -1.0 if risk_dir == "negative" else 1.0
            contrib = sign * z_val / n

    # ─ 레이아웃 ─
    st.markdown(f"##### `{code}` · {name_kr}")

    # 메타 5필드 — 2열
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"- **{UI_TEXTS['details_field_source']}**: {source_kr}  \n"
            f"- **{UI_TEXTS['details_field_series_id']}**: `{series_id}`  \n"
            f"- **{UI_TEXTS['details_field_channel']}**: 채널 {ch if ch else '—'} · {ch_kr}"
        )
    with col_b:
        # v17: 사전 변환(transform) 필드 추가.
        st.markdown(
            f"- **{UI_TEXTS['details_field_risk_direction']}**: {risk_dir_kr}  \n"
            f"- **{UI_TEXTS['details_field_transform']}**: {transform_kr}  \n"
            f"- **{UI_TEXTS['details_field_current_raw']}**: `{_format_value(raw_val, '{:,.4f}')}`  \n"
            f"- **{UI_TEXTS['details_field_current_z']}**: `{_format_value(z_val, '{:+.3f}')}σ`"
        )

    # 백분위 + 기여도 — 1줄
    pct_chip = _color_chip(pct_val)
    contrib_text = (
        UI_TEXTS["details_contribution_fmt"].format(contrib=contrib)
        if not math.isnan(contrib)
        else "—"
    )
    st.markdown(
        f"**{UI_TEXTS['details_field_current_pct']}**: {pct_chip} &nbsp;&nbsp; "
        f"**{UI_TEXTS['details_field_contribution']}**: {contrib_text}",
        unsafe_allow_html=True,
    )

    # 작업 3 (v12): 월별 변수의 stale z-score 안내 메시지
    if z_is_stale and last_valid_z_date is not None:
        st.caption(
            f"⚠️ 기준일에서 직접 산출된 z-score는 없습니다 (월별 데이터 정렬 한계). "
            f"마지막 유효 발표일: {last_valid_z_date.strftime('%Y-%m-%d')} 기준값을 표시했습니다."
        )
    elif math.isnan(z_val):
        st.caption("기준일 z-score 산출 불가 (해당 변수의 유효 시계열이 부족합니다).")
    if last_valid_raw_date is not None and last_valid_raw_date < as_of:
        st.caption(f"raw 값은 {last_valid_raw_date.strftime('%Y-%m-%d')} 발표/관측값입니다.")

    # 시계열 (최근 1년)
    if z_panel is not None and not z_panel.empty and code in z_panel.columns:
        ts = z_panel[code]
        ts_view = ts[(ts.index >= one_year_ago) & (ts.index <= result.as_of_date)].dropna()
        if not ts_view.empty:
            st.markdown(f"**{UI_TEXTS['details_field_timeseries']}** (z-score, 점선: ±1.5σ)")
            fig = _mini_timeseries(ts_view, height=180, is_zscore=True)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"{key_prefix}_ts_{code}",
            )
        else:
            st.caption("최근 1년 z-score 시계열 데이터 없음.")
    elif aligned_panel is not None and not aligned_panel.empty and code in aligned_panel.columns:
        ts = aligned_panel[code]
        ts_view = ts[(ts.index >= one_year_ago) & (ts.index <= result.as_of_date)].dropna()
        if not ts_view.empty:
            st.markdown(f"**{UI_TEXTS['details_field_timeseries']}** (raw)")
            fig = _mini_timeseries(ts_view, height=180, is_zscore=False)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"{key_prefix}_ts_{code}",
            )

    st.markdown("---")


__all__ = ["render_details_tab"]
