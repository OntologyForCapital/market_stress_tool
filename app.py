"""Streamlit 메인 엔트리.

실행:
    streamlit run app.py

탭 구성:
    1) 메인 진단        — 현재 시장 스트레스 상태 + 채널/지도/시계열
    2) 설명             — 도구의 가정·5채널·5패턴·한계
    3) 과거 데이터 조회 — 임의 날짜 진단 + 유사 시점 상세

[KRX 약관 안내]
    KRX 일별 데이터 사용에는 출처 표기 의무가 있으므로,
    하단 푸터에 "한국거래소 통계정보"를 반드시 노출합니다.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.config import (
    load_channel_mapping,
    load_variables,
    load_target_variables,
)
from src.pipeline import DiagnosisResult, run_full_diagnosis, z_to_percentile
from src.ui import labels as L
from src.ui.charts import (
    make_channel_bar_chart,
    make_channel_breakdown,
    make_composite_timeseries,
    make_geo_risk_map,
)
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
# 기준일은 한국 영업일 기준 어제 (실데이터 의존, 사용자가 사이드바에서 조정 가능)
DEFAULT_END_DATE = date(2026, 5, 22)
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

    fig = make_composite_timeseries(ts_view, height=360)

    if _HAS_PLOTLY_EVENTS:
        st.caption(UI_TEXTS["ts_breakdown_caption"])
        events = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            key="ts_click", override_height=360,
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
        "6. **임계값 임의성** — 1.5σ, 2.0σ 같은 임계는 통계적 관습일 뿐 "
        "이론적 절대 기준이 아닙니다.\n"
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



def render_sidebar() -> tuple[date, date, date]:
    with st.sidebar:
        st.markdown(f"### {UI_TEXTS['app_title']}")
        st.markdown("##### 진단 기간")
        end_d = st.date_input(
            "기준일 (as_of)",
            value=DEFAULT_END_DATE,
            min_value=date(2010, 1, 1),
            max_value=date.today(),
            key="sidebar_as_of",
        )
        # 수집 시작일: 표준화 5년 윈도우 + 여유 1년
        start_d = date(end_d.year - 18, 1, 1)
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
    # 탭 순서: 메인 / 세부 내용 / 설명 / 과거 데이터 조회
    tab_keys = ["main", "details", "explain", "history"]
    tab_labels = [
        UI_TEXTS["tab_main"],
        UI_TEXTS["tab_details"],
        UI_TEXTS["tab_explain"],
        UI_TEXTS["tab_history"],
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
    elif selected_key == "explain":
        render_explain_tab(variables_meta)
    elif selected_key == "history":
        render_history_tab(variables_meta)



if __name__ == "__main__":
    main()
