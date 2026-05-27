"""Streamlit 메인 엔트리.

실행:
    streamlit run app.py

탭 구성:
    1) 메인 진단        — 현재 시장 스트레스 상태 + 채널/지도/시계열
    2) 세부 내용        — 현재 기준 변수별 상세
    3) 과거 데이터 조회 — 임의 날짜 진단 + 유사 시점 상세
    4) 과거 세부 내용   — 과거 기준일의 변수별 상세
    5) 설명             — 도구의 가정·5채널·5패턴·한계
    6) 원자료 시계열    — 정합/변환 전 원자료 시계열
    7) 통계 검증        — 이벤트 라벨/regime 기반 threshold 검증

[데이터 출처 안내]
    KOSPI/KOSDAQ 타겟은 v20부터 KRX API가 아니라 Yahoo Finance
    (^KS11, ^KQ11)를 사용합니다.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.config import (
    load_channel_mapping,
    load_variables,
    load_target_variables,
)
from src.analysis.threshold_calibration import (
    DEFAULT_CALIBRATION_EVENTS,
    REGIME_LABELS_KR,
)
from src.pipeline import DiagnosisResult, run_full_diagnosis, z_to_percentile
from src.ui import labels as L
from src.ui.charts import (
    make_channel_bar_chart,
    make_channel_breakdown,
    make_composite_timeseries,
    make_geo_risk_map,
)
from src.ui.date_defaults import previous_business_day, today_kst
from src.ui.details_tab import render_details_tab
from src.ui.geo_map import make_world_geo_map_svg
from src.ui.diagnosis_sentence import generate_diagnosis_sentence
from src.ui.labels import (
    CHANNEL_LABELS_KR,
    KNN_COLUMN_LABELS_KR,
    PATTERN_CONDITIONS_KR,
    PATTERN_DESCRIPTIONS_KR,
    PATTERN_LABELS_KR,
    RISK_DIRECTION_LABELS_KR,
    UI_TEXTS,
)

logger = logging.getLogger(__name__)

# 플로틀리 클릭 이벤트 (선택 의존성). 부재 시 인터랙션만 비활성화.
try:
    from streamlit_plotly_events import plotly_events
    _HAS_PLOTLY_EVENTS = True
except ImportError:  # noqa: BLE001
    plotly_events = None
    _HAS_PLOTLY_EVENTS = False

# 가로 탭 메뉴 (선택 의존성). 부재 시 st.radio로 폴백.
try:
    from streamlit_option_menu import option_menu
    _HAS_OPTION_MENU = True
except ImportError:  # noqa: BLE001
    option_menu = None
    _HAS_OPTION_MENU = False


# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title=UI_TEXTS["app_title"],
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# 기본 기간 설정
# =============================================================================
# 기준일은 페이지 실행 시점의 한국 날짜 기준 직전 영업일
DEFAULT_END_DATE = previous_business_day()
# 표준화 5년 윈도우를 위해 6년 이상 이전부터 수집
DEFAULT_START_DATE = date(DEFAULT_END_DATE.year - 6, 1, 1)


# =============================================================================
# 캐싱된 진단 호출
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def run_diagnosis_cached(
    start_date_iso: str,
    end_date_iso: str,
    as_of_iso: str | None,
    use_cache: bool = True,
) -> DiagnosisResult:
    """진단 결과를 1시간 캐싱.

    Streamlit의 cache_data는 인자를 hashable로 받아야 하므로 datetime 대신
    ISO 문자열을 사용. DiagnosisResult는 dataclass라 cacheable.
    """
    return run_full_diagnosis(
        start_date=start_date_iso,
        end_date=end_date_iso,
        as_of=as_of_iso,
        use_cache=use_cache,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def load_variables_meta() -> dict[str, dict]:
    """variables.yaml에서 {code: {name_kr, channel, ...}} 딕셔너리 생성."""
    meta: dict[str, dict] = {}
    for v in list(load_variables()) + list(load_target_variables()):
        meta[v.code] = {
            "name_kr": getattr(v, "name_kr", v.code),
            "channel": getattr(v, "channel", None),
            "risk_direction": getattr(v, "risk_direction", None),
            "section": getattr(v, "section", None),
            "source": getattr(v, "source", None),
        }
    return meta


@st.cache_data(ttl=86400, show_spinner=False)
def load_channels_raw() -> dict[int, dict]:
    """variables.yaml에서 channels 섹션 원본 dict 로드 (이름·설명·변수 목록 표시용)."""
    yaml_path = Path(__file__).parent / "config" / "variables.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    channels = raw.get("channels", {})
    # YAML에서 키가 문자열일 수 있어 int로 정규화
    return {int(k): v for k, v in channels.items()}


# =============================================================================
# 컴포넌트: 상단 헤더 (3등분)
# =============================================================================
def render_top_header(result: DiagnosisResult) -> None:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f"### {UI_TEXTS['header_title']}")
        st.latex(r"P = \frac{E[CF_{\text{실질}}]}{r_{\text{실질}} + \pi_{\text{위험}}}")
        st.caption(UI_TEXTS["asset_formula_caption"])

    with col2:
        pct = result.composite_percentile
        color = L.percentile_to_color(pct) if not math.isnan(pct) else "#9E9E9E"
        pct_str = f"{pct:.0f}" if not math.isnan(pct) else "—"
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:4.5rem; font-weight:800; color:{color}; "
            f"line-height:1.0;'>{pct_str}</div>"
            f"<div style='font-size:1.0rem; color:#555;'>{UI_TEXTS['composite_label']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        pattern_kr = L.pattern_to_korean(result.pattern_label)
        st.markdown(
            f"<div style='text-align:right;'>"
            f"<div style='font-size:1.1rem; color:#666;'>현재 패턴</div>"
            f"<div style='font-size:2.2rem; font-weight:700; color:#333;'>{pattern_kr}</div>"
            f"<div style='font-size:0.9rem; color:#888;'>"
            f"{UI_TEXTS['as_of_caption_fmt'].format(date=result.as_of_date.date().isoformat())}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# 컴포넌트: 진단 한 줄
# =============================================================================
def render_diagnosis_sentence(result: DiagnosisResult, title: str = "오늘의 진단") -> None:
    sentence = generate_diagnosis_sentence(result)
    level = L.percentile_to_level(result.composite_percentile)
    bg_color = {
        "low":  "#E8F5E9",
        "mid":  "#FFF8E1",
        "high": "#FFEBEE",
    }[level]
    border_color = L.percentile_to_color(result.composite_percentile)
    st.markdown(
        f"<div style='background-color:{bg_color}; border-left:5px solid {border_color}; "
        f"padding:14px 18px; border-radius:6px; margin:10px 0; font-size:1.1rem;'>"
        f"<b>{title}</b> · {sentence}"
        f"</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 컴포넌트: 하단 카드 (진원지 / 유사 시점)
# =============================================================================
def render_origin_card(result: DiagnosisResult, variables_meta: dict) -> None:
    st.markdown(f"##### {UI_TEXTS['origin_card_title']}")
    origin = result.origin_result
    if origin is None or origin.origin_variable is None:
        st.info(UI_TEXTS["origin_card_empty"])
        return
    var_kr = L.variable_to_korean(origin.origin_variable, variables_meta=variables_meta)
    ch_kr = L.channel_to_korean(origin.origin_channel) if origin.origin_channel else "—"
    breach_date = origin.origin_first_breach_date
    if isinstance(breach_date, pd.Timestamp):
        breach_str = breach_date.date().isoformat()
    else:
        breach_str = str(breach_date) if breach_date else "—"
    st.markdown(
        UI_TEXTS["origin_card_fmt"].format(
            date=breach_str, variable_kr=var_kr, channel_kr=ch_kr,
        )
    )


def render_similar_card(result: DiagnosisResult, key_suffix: str = "main") -> None:
    st.markdown(f"##### {UI_TEXTS['similar_card_title']}")
    df = result.similar_dates
    if df is None or df.empty:
        st.info(UI_TEXTS["similar_card_empty"])
        return

    # 1위 시점
    top = df.iloc[0]
    top_date = df.index[0]
    if isinstance(top_date, pd.Timestamp):
        top_date_str = top_date.date().isoformat()
    else:
        top_date_str = str(top_date)
    f30 = float(top.get("fwd_30d", float("nan")))
    f90 = float(top.get("fwd_90d", float("nan")))
    f180 = float(top.get("fwd_180d", float("nan")))

    st.markdown(UI_TEXTS["similar_card_date_fmt"].format(date=top_date_str))
    st.markdown(
        UI_TEXTS["similar_card_returns_fmt"].format(
            f30=0.0 if math.isnan(f30) else f30,
            f90=0.0 if math.isnan(f90) else f90,
            f180=0.0 if math.isnan(f180) else f180,
        )
    )
    if st.button(UI_TEXTS["similar_card_more"], key=f"goto_history_{key_suffix}"):
        # 작업 1: 과거 탭 자동 이동 + 날짜 자동 입력 + 자동 실행
        st.session_state["active_tab"] = "history"
        st.session_state["history_target_date"] = (
            top_date.date() if isinstance(top_date, pd.Timestamp) else top_date
        )
        st.session_state["history_auto_run"] = True
        st.session_state["history_show_knn"] = True
        st.rerun()


def render_knn_table(result: DiagnosisResult) -> None:
    """k-NN 유사 시점 상세 테이블 + 의미 설명 expander.

    메인 탭과 과거 탭 둘 다에서 호출 가능.
    """
    st.markdown(f"#### {UI_TEXTS['history_knn_title']}")

    with st.expander(UI_TEXTS["history_knn_help_title"], expanded=False):
        st.markdown(f"- {UI_TEXTS['history_knn_help_distance']}")
        st.markdown(f"- {UI_TEXTS['history_knn_help_zscore']}")
        st.markdown(f"- {UI_TEXTS['history_knn_help_returns']}")
        st.markdown(f"- {UI_TEXTS['history_knn_help_caveat']}")

    knn_df = result.similar_dates
    if knn_df is None or knn_df.empty:
        st.info("유사 시점 결과가 없습니다.")
        return

    view_df = knn_df.copy()
    view_df.insert(0, "date", view_df.index.strftime("%Y-%m-%d"))
    for col in view_df.columns:
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(view_df[col]):
            if col.startswith("fwd_"):
                view_df[col] = view_df[col].apply(
                    lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
                )
            elif col == "distance":
                view_df[col] = view_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )
            else:
                view_df[col] = view_df[col].apply(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
                )
    view_df = view_df.rename(columns=KNN_COLUMN_LABELS_KR)
    st.dataframe(view_df, use_container_width=True, hide_index=True)


# =============================================================================
# 컴포넌트: 푸터
# =============================================================================
def render_footer() -> None:
    st.markdown("---")
    st.caption(UI_TEXTS["footer_sources"])
    st.caption(UI_TEXTS["footer_disclaimer"])


# =============================================================================
# 메인 탭
# =============================================================================
def render_main_tab(result: DiagnosisResult, variables_meta: dict) -> None:
    # 작업 5: 풀 헤더는 메인 탭에서만 렌더
    render_top_header(result)
    st.markdown("---")

    # 1) 오늘의 진단 한 줄
    render_diagnosis_sentence(result)

    # 3) 본체: 좌 1/3 (채널 막대) + 우 2/3 (지도)
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown(f"#### {UI_TEXTS['channel_section_title']}")
        fig_bar = make_channel_bar_chart(result.channel_percentiles, height=380)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    with col_right:
        st.markdown(f"#### {UI_TEXTS['geo_section_title']}")
        variable_to_channel = load_channel_mapping()
        # 작업 3: SVG 추상 지도 (plotly scattergeo 대체)
        svg_html = make_world_geo_map_svg(
            variable_z_scores=result.variable_z_scores,
            variable_percentiles=result.variable_percentiles,
            variable_to_channel=variable_to_channel,
            variables_meta=variables_meta,
            width=720,
            height=420,
        )
        st.markdown(svg_html, unsafe_allow_html=True)

    # 4) 종합 시계열
    st.markdown("---")
    st.markdown(f"#### {UI_TEXTS['ts_section_title']}")
    render_timeseries_section(result)

    # 5) 하단 카드
    st.markdown("---")
    col_o, col_s = st.columns([1, 1])
    with col_o:
        render_origin_card(result, variables_meta)
    with col_s:
        render_similar_card(result, key_suffix="main")

    # k-NN 상세 테이블 (메인 탭으로 이동)
    st.markdown("---")
    render_knn_table(result)

    render_footer()


def render_timeseries_section(result: DiagnosisResult) -> None:
    """종합 백분위 시계열 + 기간 선택 + 클릭 시 채널 분해."""
    # 기간 선택 (기본: 최근 1년)
    data_start, data_end = result.data_period
    if data_start is None or data_end is None:
        st.warning("데이터 기간 정보가 없어 시계열을 표시할 수 없습니다.")
        return

    # 종합 백분위 시계열 가져오기
    composite_ts = result.composite_pct_series
    if composite_ts is None or composite_ts.empty:
        st.info("종합 백분위 시계열 데이터가 없습니다.")
        return

    default_ts_start = max(data_start, data_end - pd.Timedelta(days=365))

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        ts_start = st.date_input(
            UI_TEXTS["ts_period_start"],
            value=default_ts_start.date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="ts_start",
        )
    with col_b:
        ts_end = st.date_input(
            UI_TEXTS["ts_period_end"],
            value=data_end.date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="ts_end",
        )
    with col_c:
        st.write("")  # 정렬용 빈 줄
        st.write("")
        st.button(UI_TEXTS["ts_query_button"], key="ts_query")

    # 기간 필터
    mask = (composite_ts.index >= pd.Timestamp(ts_start)) & (
        composite_ts.index <= pd.Timestamp(ts_end)
    )
    ts_view = composite_ts[mask]
    if ts_view.empty:
        st.warning("선택된 기간에 데이터가 없습니다.")
        return

    event_annotations = _event_annotations_for_timeseries(ts_view)
    event_label_lanes = min(len(event_annotations), 3)
    chart_height = 360 if not event_annotations else 390 + event_label_lanes * 50
    fig = make_composite_timeseries(
        ts_view,
        height=chart_height,
        event_annotations=event_annotations,
    )

    if _HAS_PLOTLY_EVENTS:
        st.caption(UI_TEXTS["ts_breakdown_caption"])
        events = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            key="ts_click", override_height=chart_height,
        )
        if events:
            clicked_x = events[0].get("x")
            if clicked_x:
                try:
                    clicked_date = pd.Timestamp(clicked_x)
                    _render_breakdown_for_date(clicked_date, result)
                except Exception:  # noqa: BLE001
                    pass
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "그래프 클릭 인터랙션을 사용하려면 `streamlit-plotly-events`를 설치하세요."
        )


def _event_annotations_for_timeseries(
    ts_view: pd.Series,
) -> list[tuple[pd.Timestamp, str]]:
    """현재 조회 기간 안에 들어오는 주요 사건 라벨만 반환."""
    if ts_view.empty:
        return []

    start = pd.Timestamp(ts_view.index.min())
    end = pd.Timestamp(ts_view.index.max())
    events: list[tuple[pd.Timestamp, str]] = []
    for event in DEFAULT_CALIBRATION_EVENTS:
        event_date = pd.Timestamp(event.date)
        if start <= event_date <= end:
            events.append((event_date, event.label))
    return events


def _render_breakdown_for_date(target_date: pd.Timestamp, result: DiagnosisResult) -> None:
    """클릭된 날짜의 5채널 분해 차트."""
    channel_panel = result.channel_pct_panel
    if channel_panel is None or channel_panel.empty:
        st.info(f"{target_date.date().isoformat()} 시점의 채널 분해 데이터가 없습니다.")
        return

    # 가장 가까운 영업일 찾기
    valid_idx = channel_panel.index[channel_panel.index <= target_date]
    if len(valid_idx) == 0:
        st.info(f"{target_date.date().isoformat()} 시점의 데이터를 찾을 수 없습니다.")
        return
    nearest_date = valid_idx[-1]
    row = channel_panel.loc[nearest_date]

    # 5채널 분해 차트
    import math as _math
    from src.ui.charts import make_channel_bar_chart
    channel_pct = {int(c[1]): float(row[c]) if not _math.isnan(row[c]) else float("nan")
                   for c in ["S1", "S2", "S3", "S4", "S5"]}
    fig_breakdown = make_channel_bar_chart(
        channel_pct,
        title=f"{nearest_date.date().isoformat()} 채널 분해",
        height=260,
    )
    st.plotly_chart(fig_breakdown, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# 설명 탭
# =============================================================================
def render_explain_tab(variables_meta: dict) -> None:
    st.markdown(f"### {UI_TEXTS['explain_title']}")

    st.markdown("#### 도구의 본질")
    st.markdown(
        "이 도구는 현재 시장 스트레스 수준과 그 원인을 근거 기반으로 추리합니다. "
        "대상은 거시경제 입문자입니다."
    )

    st.markdown("#### 자산가격 모형")
    st.latex(r"P = \frac{E[CF_\text{실질}]}{r_\text{실질} + \pi_\text{위험}}")
    st.markdown(
        "- **분자** `E[CF실질]`: 기업이 미래에 벌어들일 것으로 기대되는 실질 현금흐름. "
        "글로벌 수요·한국 수출·산업생산 같은 실물 변수가 신호를 줍니다.\n"
        "- **분모** `r실질 + π위험`: 미래 현금흐름을 현재가치로 환산할 때 쓰는 할인율. "
        "무위험 실질금리(r)와 위험 보상(π)으로 나뉩니다.\n"
        "- **조정의 본질**: 분모가 갑자기 커지거나(금리/위험 프리미엄 충격), "
        "분자가 갑자기 줄어들면(실물 침체·공급 충격) 자산가격은 빠르게 하락합니다."
    )

    st.markdown("#### 5개 채널과 변수")
    channels_raw = load_channels_raw()
    variable_to_channel = load_channel_mapping()

    # 채널별 변수 그루핑
    channel_to_vars: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for code, ch in variable_to_channel.items():
        if ch in channel_to_vars:
            channel_to_vars[ch].append(code)

    for ch in (1, 2, 3, 4, 5):
        name_kr = CHANNEL_LABELS_KR.get(ch, f"채널 {ch}")
        desc = channels_raw.get(ch, {}).get("description", "")
        with st.expander(f"**채널 {ch} · {name_kr}**", expanded=False):
            if desc:
                st.markdown(desc)
            st.markdown("**포함 변수:**")
            for code in sorted(channel_to_vars.get(ch, [])):
                meta = variables_meta.get(code, {})
                name = meta.get("name_kr", code)
                risk_dir = meta.get("risk_direction")
                # v17: 중앙 라벨 사전(labels.RISK_DIRECTION_LABELS_KR) 사용.
                # v14 이후 'bidirectional' 값도 자동 대응.
                risk_dir_kr = (
                    RISK_DIRECTION_LABELS_KR.get(risk_dir, risk_dir)
                    if risk_dir
                    else "—"
                )
                st.markdown(f"- `{code}` · {name} · *{risk_dir_kr}*")

    # v17: 종합 스트레스 합성 방법 설명 (v15 L2 norm).
    st.markdown("#### 종합 스트레스 지수의 계산 방법")
    st.markdown(
        "5개 채널의 z-score(S1~S5)를 하나의 종합 점수로 합치는 방법입니다.\n\n"
        "- **합성 방식**: 5채널 z-score의 **L2 norm** (제곱 평균의 제곱근).\n"
        "- **의미**: 거시 환경이 평시 상태에서 얼마나 떨어져 있는지(위험 강도)를 측정합니다. "
        "한 채널의 강한 신호가 다른 채널의 약한 신호에 묻히지 않습니다."
    )
    st.latex(r"\text{composite\_z} = \sqrt{\frac{1}{5}\sum_{i=1}^{5} S_i^{\,2}}")
    st.markdown(
        "단순 평균과 달리 제곱을 거치므로 음수 값이 양수 값을 상쇄시키지 않습니다. "
        "\"한 채널이 +2.5σ, 세 채널이 -0.5σ\"인 분산된 신호도 제대로 집계합니다."
    )

    st.markdown("#### 통계 처리")
    st.markdown(
        "- **표준화**: 평균/표준편차 대신 median/MAD 기반 robust z-score를 기본값으로 사용합니다.\n"
        "- **극단치 처리**: 위험방향 적용 전 z-score를 ±6σ로 제한해 단일 spike의 지수 지배를 완화합니다.\n"
        "- **백분위**: 정규분포 선형 근사가 아니라 5년 rolling empirical percentile rank로 표시합니다."
    )

    st.markdown("#### 임계값 보정")
    st.markdown(
        "- **이벤트 라벨 보정**: 과거 주요 스트레스 시점 주변을 event window로 두고, "
        "종합 z-score threshold의 precision/recall/F1을 계산합니다.\n"
        "- **regime별 분위수**: KOSPI 63영업일 실현변동성으로 저·중·고변동 구간을 나눈 뒤, "
        "각 regime 안에서 q80/q90/q95 경험적 임계값을 산출합니다."
    )

    st.markdown("#### 5가지 패턴 분류")
    for key, name_kr in PATTERN_LABELS_KR.items():
        if key == "system_crisis":
            continue  # 시스템 위기는 최후에 별도 노출
        with st.expander(f"**{name_kr}** ({key})", expanded=False):
            st.markdown(PATTERN_DESCRIPTIONS_KR.get(key, ""))
            cond = PATTERN_CONDITIONS_KR.get(key)
            if cond:
                st.code(cond, language="text")
    # 시스템 위기
    with st.expander(f"**{PATTERN_LABELS_KR['system_crisis']}** (system_crisis)", expanded=False):
        st.markdown(PATTERN_DESCRIPTIONS_KR["system_crisis"])
        st.code(PATTERN_CONDITIONS_KR["system_crisis"], language="text")

    st.markdown("#### 도구의 한계")
    st.markdown(
        "1. **분자 충격 진단 약함** — 분자(실질 현금흐름) 측정 변수가 적습니다 "
        "(BDI · INDPRO · KR_EXPORT 정도). 분자 신호가 잡힐 즈음에는 이미 가격에 반영된 경우가 많습니다.\n"
        "2. **한국 시장 고유 위험 측정 부분적** — 환율(KRW_USD)은 보지만, "
        "외국인 매매·한국 실질금리·CDS 등은 1차에서 빠져 있습니다.\n"
        "3. **한미 시차 미보정** — 같은 캘린더 날짜로 정렬하므로 미국 야간 충격이 "
        "한국 다음 날에 반영되는 시차를 직접 모형화하지 않습니다.\n"
        "4. **자산 가격 변수는 수익률 변환이 더 적합** — 현재는 수준값을 표준화합니다. "
        "장기 추세가 있는 시리즈는 수익률/로그차분이 더 통계적으로 안정적입니다.\n"
        "5. **단일 라벨 분류** — 동시에 여러 충격이 일어나는 경우 한 라벨로 단순화됩니다. "
        "채널 막대를 함께 확인하세요.\n"
        "6. **임계값 보정의 표본 한계** — 이벤트 라벨과 regime 분위수로 보정하지만, "
        "위기 표본 수가 적어 과최적화 가능성이 남습니다.\n"
        "7. **시간 순서 ≠ 인과** — '진원지'는 1.5σ를 가장 먼저 돌파한 변수일 뿐, "
        "도구가 보지 못하는 외부 사건(지정학·정책 발표 등)이 진짜 원인일 수 있습니다.\n"
        "8. **k-NN은 분모 환경 유사성만 측정** — 유사 시점의 향후 수익률은 참고치이며, "
        "분자(미래 현금흐름) 환경이 다르면 결과는 달라질 수 있습니다."
    )

    render_footer()


# =============================================================================
# 과거 데이터 조회 탭
# =============================================================================
def render_history_tab(variables_meta: dict) -> None:
    st.markdown(f"### {UI_TEXTS['history_title']}")

    # 작업 1: 메인 탭에서 전달된 자동 날짜가 있으면 초기값으로 사용
    initial_target = st.session_state.pop("history_target_date", None)
    if initial_target is None:
        initial_target = st.session_state.get("history_date", DEFAULT_END_DATE)

    auto_run = st.session_state.pop("history_auto_run", False)

    target_date = st.date_input(
        UI_TEXTS["history_date_label"],
        value=initial_target,
        min_value=date(2010, 1, 1),
        max_value=DEFAULT_END_DATE,
        key="history_date",
    )

    show_knn = st.session_state.get("history_show_knn", False)

    # 날짜가 선택되어 있으면 항상 진단 실행 (버튼 없이 자동)
    if target_date:
        with st.spinner("진단을 실행하고 있습니다... (최초 호출 시 데이터 수집으로 시간이 걸립니다)"):
            try:
                hist_result = run_diagnosis_cached(
                    start_date_iso=(pd.Timestamp(target_date) - pd.DateOffset(years=7)).date().isoformat(),
                    end_date_iso=DEFAULT_END_DATE.isoformat(),
                    as_of_iso=target_date.isoformat(),
                    use_cache=True,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"진단 실행 실패: {e}")
                return

        # 작업 5: 간소 헤더 한 줄 (풀 헤더 대신 종합 점수 요약)
        pct = hist_result.composite_percentile
        level_kr = (
            L.percentile_to_level_label_kr(pct)
            if not math.isnan(pct)
            else "—"
        )
        color = L.percentile_to_color(pct) if not math.isnan(pct) else "#9E9E9E"
        summary_text = UI_TEXTS["history_summary_fmt"].format(
            pct=pct if not math.isnan(pct) else 0.0,
            level_kr=level_kr,
        )
        st.markdown(
            f"<div style='background-color:#F5F5F5; padding:10px 14px; "
            f"border-left:4px solid {color}; border-radius:4px; margin:8px 0;'>"
            f"<span style='font-size:1.05rem;'><b>{summary_text}</b></span>"
            f"<span style='color:#888; margin-left:14px;'>· 기준일: "
            f"{hist_result.as_of_date.date().isoformat()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        render_diagnosis_sentence(hist_result, title="그날의 진단")

        col_l, col_r = st.columns([1, 2])
        with col_l:
            fig_bar = make_channel_bar_chart(
                hist_result.channel_percentiles, height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        with col_r:
            variable_to_channel = load_channel_mapping()
            # 작업 3: SVG 추상 지도
            svg_html = make_world_geo_map_svg(
                variable_z_scores=hist_result.variable_z_scores,
                variable_percentiles=hist_result.variable_percentiles,
                variable_to_channel=variable_to_channel,
                variables_meta=variables_meta,
                width=720,
                height=420,
            )
            st.markdown(svg_html, unsafe_allow_html=True)

        st.markdown("---")
        col_o, col_s = st.columns([1, 1])
        with col_o:
            render_origin_card(hist_result, variables_meta)
        with col_s:
            render_similar_card(hist_result, key_suffix="history")

        # k-NN 상세 테이블 + 의미 expander
        st.markdown("---")
        render_knn_table(hist_result)

        # 조회 후 자동 플래그 클리어
        st.session_state["history_show_knn"] = False

    render_footer()


def _format_numeric_frame(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Return a display-friendly rounded DataFrame."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def _make_history_line_chart(df: pd.DataFrame, title: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=str(col),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=35),
        plot_bgcolor="white",
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0.08)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def render_history_detail_tab(result: DiagnosisResult, variables_meta: dict) -> None:
    """특정 과거 기준일의 변수별 세부 내용."""
    st.markdown(f"### {UI_TEXTS['history_detail_title']}")

    data_start, data_end = result.data_period
    min_d = data_start.date() if data_start is not None else date(2010, 1, 1)
    max_d = min(DEFAULT_END_DATE, data_end.date()) if data_end is not None else DEFAULT_END_DATE

    initial_target = st.session_state.get("history_detail_date", DEFAULT_END_DATE)
    target_date = st.date_input(
        "조회 기준일",
        value=initial_target,
        min_value=min_d,
        max_value=max_d,
        key="history_detail_date",
    )

    with st.spinner("과거 기준일의 세부 내용을 계산하고 있습니다..."):
        try:
            hist_result = run_diagnosis_cached(
                start_date_iso=(pd.Timestamp(target_date) - pd.DateOffset(years=7)).date().isoformat(),
                end_date_iso=DEFAULT_END_DATE.isoformat(),
                as_of_iso=target_date.isoformat(),
                use_cache=True,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"과거 세부내용 계산 실패: {e}")
            render_footer()
            return

    pct = hist_result.composite_percentile
    level_kr = L.percentile_to_level_label_kr(pct) if not math.isnan(pct) else "—"
    st.caption(
        f"기준일 {hist_result.as_of_date.date().isoformat()} · "
        f"종합 {pct:.1f}점 ({level_kr}) · 패턴 {L.pattern_to_korean(hist_result.pattern_label)}"
    )
    render_details_tab(
        hist_result,
        variables_meta,
        title=f"{hist_result.as_of_date.date().isoformat()} 기준 변수별 세부 정보",
        intro="선택한 과거 기준일에 각 변수가 어떤 raw 값, z-score, 백분위를 가졌는지 확인합니다.",
        key_prefix="history_details",
    )
    render_footer()


def render_raw_timeseries_tab(result: DiagnosisResult, variables_meta: dict) -> None:
    """정합/변환 전 원자료 시계열 탭."""
    st.markdown(f"### {UI_TEXTS['raw_timeseries_title']}")
    st.caption("API/로더에서 수집된 원자료입니다. 영업일 정합, forward-fill, 사전 변환, z-score 표준화가 적용되기 전 값입니다.")

    raw_panel = result.raw_panel
    if raw_panel is None or raw_panel.empty:
        st.info("원자료 패널이 없습니다.")
        render_footer()
        return

    data_start = raw_panel.index.min()
    data_end = raw_panel.index.max()
    default_start = max(data_start, data_end - pd.Timedelta(days=730))

    from src.config import load_channel_mapping
    variable_to_channel = load_channel_mapping()

    channel_options = ["전체", "타깃·기타"] + [
        f"채널 {ch} · {CHANNEL_LABELS_KR[ch]}" for ch in (1, 2, 3, 4, 5)
    ]
    col_a, col_b, col_c = st.columns([1, 1, 1.2])
    with col_a:
        start = st.date_input(
            "시작일",
            value=default_start.date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="raw_ts_start",
        )
    with col_b:
        end = st.date_input(
            "종료일",
            value=data_end.date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="raw_ts_end",
        )
    with col_c:
        channel_filter = st.selectbox("채널", channel_options, key="raw_ts_channel")

    candidate_cols = list(raw_panel.columns)
    if channel_filter.startswith("채널"):
        ch = int(channel_filter.split()[1])
        candidate_cols = [c for c in candidate_cols if variable_to_channel.get(c) == ch]
    elif channel_filter == "타깃·기타":
        candidate_cols = [c for c in candidate_cols if variable_to_channel.get(c) is None]

    label_to_col = {
        f"{code} · {L.variable_to_korean(str(code), variables_meta)}": str(code)
        for code in candidate_cols
    }
    default_labels = list(label_to_col.keys())[: min(6, len(label_to_col))]
    selected_labels = st.multiselect(
        "표시 변수",
        options=list(label_to_col.keys()),
        default=default_labels,
        key="raw_ts_columns",
    )
    selected_cols = [label_to_col[label] for label in selected_labels if label in label_to_col]
    if not selected_cols:
        st.info("표시할 변수를 선택하세요.")
        render_footer()
        return

    mask = (raw_panel.index >= pd.Timestamp(start)) & (raw_panel.index <= pd.Timestamp(end))
    view = raw_panel.loc[mask, selected_cols].copy()
    if view.empty:
        st.warning("선택한 기간에 원자료가 없습니다.")
        render_footer()
        return

    st.plotly_chart(
        _make_history_line_chart(view, "원자료 시계열", height=420),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("#### 변수별 원자료")
    for code in selected_cols:
        series = view[code].dropna()
        if series.empty:
            continue
        name = L.variable_to_korean(code, variables_meta)
        latest_date = series.index[-1].strftime("%Y-%m-%d")
        latest_val = series.iloc[-1]
        with st.expander(f"`{code}` · {name}", expanded=False):
            st.metric("선택 기간 마지막 관측값", f"{latest_val:,.4f}", help=latest_date)
            fig = _make_history_line_chart(series.to_frame(code), f"{code} · {name}", height=220)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"raw_ts_block_{code}",
            )
            tail = series.tail(20).to_frame("raw")
            tail.insert(0, "date", tail.index.strftime("%Y-%m-%d"))
            st.dataframe(_format_numeric_frame(tail), use_container_width=True, hide_index=True)

    render_footer()


def _calibration_level_to_kr(level: str) -> str:
    return {
        "normal": "정상권",
        "watch": "주의권",
        "high_regime": "고위험권",
        "tail_alert": "꼬리위험권",
        "event_alert": "이벤트 경보권",
        "unknown": "미분류",
    }.get(level, level)


def _finite(value: object) -> bool:
    try:
        return value is not None and not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _make_calibration_chart(result: DiagnosisResult) -> go.Figure:
    stress = result.stress_table
    fig = go.Figure()
    if stress is None or stress.empty or "composite" not in stress.columns:
        return fig

    score = stress["composite"].dropna()
    fig.add_trace(
        go.Scatter(
            x=score.index,
            y=score.values,
            mode="lines",
            line=dict(color="#1F77B4", width=1.8),
            name="composite z",
            hovertemplate="%{x|%Y-%m-%d}<br>z=%{y:.3f}<extra></extra>",
        )
    )

    event_labels = result.calibration_event_labels
    if event_labels is not None and not event_labels.empty:
        event_mask = event_labels.reindex(score.index).fillna(False).astype(bool)
        points = score[event_mask]
        if not points.empty:
            fig.add_trace(
                go.Scatter(
                    x=points.index,
                    y=points.values,
                    mode="markers",
                    marker=dict(color="#D32F2F", size=5, opacity=0.65),
                    name="event window",
                    hovertemplate="%{x|%Y-%m-%d}<br>event z=%{y:.3f}<extra></extra>",
                )
            )

    summary = result.calibration_summary or {}
    hlines = [
        ("label_threshold", "라벨 보정", "#D32F2F"),
        ("regime_q90", "regime q90", "#F9A825"),
        ("regime_q95", "regime q95", "#6A1B9A"),
    ]
    for key, label, color in hlines:
        value = summary.get(key)
        if _finite(value):
            fig.add_hline(
                y=float(value),
                line_width=1.2,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{label}: {float(value):.2f}",
                annotation_position="top left",
                annotation_font_size=10,
            )

    fig.update_layout(
        title="종합 스트레스 z-score와 보정 임계값",
        height=380,
        margin=dict(l=10, r=10, t=50, b=35),
        plot_bgcolor="white",
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="composite z", gridcolor="rgba(0,0,0,0.08)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    return fig


def render_calibration_tab(result: DiagnosisResult) -> None:
    """이벤트 라벨/변동성 regime 기반 통계 검증 탭."""
    st.markdown(f"### {UI_TEXTS['calibration_title']}")

    summary = result.calibration_summary or {}
    if not summary:
        st.info("보정 결과가 없습니다.")
        render_footer()
        return

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        st.metric("현재 보정 레벨", _calibration_level_to_kr(str(summary.get("calibrated_level", "unknown"))))
    with col_2:
        st.metric("현재 composite z", f"{float(summary.get('current_score', float('nan'))):.2f}")
    with col_3:
        regime = str(summary.get("current_regime", "unknown"))
        st.metric("현재 변동성 regime", REGIME_LABELS_KR.get(regime, regime))
    with col_4:
        threshold = summary.get("label_threshold")
        st.metric("라벨 보정 임계값", f"{float(threshold):.2f}" if _finite(threshold) else "—")

    st.plotly_chart(
        _make_calibration_chart(result),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 이벤트 라벨 기준")
        event_metrics = result.calibration_event_metrics
        if event_metrics is None or event_metrics.empty:
            st.info("이벤트 라벨 기준 threshold 성능표가 없습니다.")
        else:
            show_cols = [
                "threshold", "threshold_percentile", "precision", "recall",
                "f1", "alert_rate", "tp", "fp", "fn",
            ]
            view = event_metrics[show_cols].head(12).copy()
            for col in ("threshold", "threshold_percentile", "precision", "recall", "f1", "alert_rate"):
                view[col] = view[col].astype(float).round(3)
            for col in ("tp", "fp", "fn"):
                view[col] = view[col].astype(int)
            view = view.rename(columns={
                "threshold": "임계값",
                "threshold_percentile": "분포 위치",
                "precision": "정밀도",
                "recall": "재현율",
                "f1": "F1",
                "alert_rate": "경보 빈도",
                "tp": "TP",
                "fp": "FP",
                "fn": "FN",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### 변동성 regime별 분위수")
        regime_thresholds = result.calibration_regime_thresholds
        if regime_thresholds is None or regime_thresholds.empty:
            st.info("regime별 threshold 표가 없습니다.")
        else:
            metric_options = [m for m in ["composite", "S1", "S2", "S3", "S4", "S5"]
                              if m in set(regime_thresholds["metric"])]
            metric = st.selectbox("지표", metric_options, key="calibration_metric")
            view = regime_thresholds[regime_thresholds["metric"] == metric].copy()
            keep_cols = ["regime_kr", "count", "q67", "q80", "q90", "q95"]
            view = view[keep_cols].rename(columns={
                "regime_kr": "regime",
                "count": "관측수",
                "q67": "q67",
                "q80": "q80",
                "q90": "q90",
                "q95": "q95",
            })
            st.dataframe(_format_numeric_frame(view, digits=3), use_container_width=True, hide_index=True)

    st.markdown("#### 보정 이벤트")
    period_start, period_end = result.data_period
    event_rows = [
        {"date": e.date, "label": e.label}
        for e in DEFAULT_CALIBRATION_EVENTS
        if period_start is None
        or period_end is None
        or (period_start <= pd.Timestamp(e.date) <= period_end)
    ]
    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
    render_footer()


def render_sidebar() -> tuple[date, date, date]:
    with st.sidebar:
        current_date = today_kst()
        st.markdown(f"### {UI_TEXTS['app_title']}")
        st.markdown("##### 진단 기간")
        end_d = st.date_input(
            "기준일 (as_of)",
            value=DEFAULT_END_DATE,
            min_value=date(2010, 1, 1),
            max_value=current_date,
            key="sidebar_as_of",
        )
        # 수집 시작일: 표준화 5년 윈도우 + 여유 1년
        start_d = date(2010, 1, 4)
        st.caption(f"데이터 수집 시작: {start_d.isoformat()}")
        st.caption(f"데이터 수집 종료: {end_d.isoformat()}")

        st.markdown("---")
        use_cache = st.checkbox("로더 캐시 사용", value=True, key="sidebar_use_cache")

        st.markdown("---")
        st.caption(UI_TEXTS["footer_sources"])

    return start_d, end_d, end_d


# =============================================================================
# 메인 라우팅
# =============================================================================
def main() -> None:
    # 사이드바 (기간 설정)
    start_d, end_d, as_of_d = render_sidebar()
    use_cache = st.session_state.get("sidebar_use_cache", True)

    # 메타 로드
    variables_meta = load_variables_meta()

    # 진단 실행 (메인/세부 탭 공용)
    with st.spinner("진단을 실행하고 있습니다... (최초 호출 시 데이터 수집으로 시간이 걸립니다)"):
        try:
            result = run_diagnosis_cached(
                start_date_iso=start_d.isoformat(),
                end_date_iso=end_d.isoformat(),
                as_of_iso=as_of_d.isoformat(),
                use_cache=use_cache,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"진단 실행 실패: {e}")
            logger.exception("진단 실행 실패")
            return

    # 작업 1+4+5: st.radio 기반 탭 라우팅
    # 탭 순서: 메인 / 세부 내용 / 과거 데이터 조회 / 과거 세부 내용 / 설명 / 원자료 시계열 / 통계 검증
    tab_keys = ["main", "details", "history", "history_detail", "explain", "raw_timeseries", "calibration"]
    tab_labels = [
        UI_TEXTS["tab_main"],
        UI_TEXTS["tab_details"],
        UI_TEXTS["tab_history"],
        UI_TEXTS["tab_history_detail"],
        UI_TEXTS["tab_explain"],
        UI_TEXTS["tab_raw_timeseries"],
        UI_TEXTS["tab_calibration"],
    ]
    key_to_label = dict(zip(tab_keys, tab_labels))
    label_to_key = dict(zip(tab_labels, tab_keys))

    # 프로그래마틱 탭 전환이 요청되었으면 라디오 초기값으로 적용
    pending_tab = st.session_state.pop("active_tab", None)
    if pending_tab and pending_tab in tab_keys:
        st.session_state["_tab_radio"] = key_to_label[pending_tab]

    # 작업 2 (v12): streamlit-option-menu 기반 가로 탭 (라디오 버튼 외관 개선)
    if _HAS_OPTION_MENU:
        _default_label = st.session_state.get("_tab_radio", tab_labels[0])
        if _default_label not in tab_labels:
            _default_label = tab_labels[0]
        selected_label = option_menu(
            menu_title=None,
            options=tab_labels,
            default_index=tab_labels.index(_default_label),
            orientation="horizontal",
            key="_tab_radio",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#FAFAFA",
                    "border-radius": "8px",
                },
                "icon": {"display": "none"},
                "nav-link": {
                    "font-size": "15px",
                    "font-weight": "500",
                    "text-align": "center",
                    "margin": "0px",
                    "padding": "10px 18px",
                    "color": "#555555",
                    "--hover-color": "#EEEEEE",
                },
                "nav-link-selected": {
                    "background-color": "#1E88E5",
                    "color": "#FFFFFF",
                    "font-weight": "600",
                },
            },
        )
    else:
        selected_label = st.radio(
            "탭 선택",
            options=tab_labels,
            horizontal=True,
            label_visibility="collapsed",
            key="_tab_radio",
        )
    selected_key = label_to_key[selected_label]

    st.markdown("---")

    if selected_key == "main":
        render_main_tab(result, variables_meta)
    elif selected_key == "details":
        # 작업 4: 세부 내용 탭
        render_details_tab(result, variables_meta)
        render_footer()
    elif selected_key == "history_detail":
        render_history_detail_tab(result, variables_meta)
    elif selected_key == "raw_timeseries":
        render_raw_timeseries_tab(result, variables_meta)
    elif selected_key == "calibration":
        render_calibration_tab(result)
    elif selected_key == "explain":
        render_explain_tab(variables_meta)
    elif selected_key == "history":
        render_history_tab(variables_meta)



if __name__ == "__main__":
    main()
