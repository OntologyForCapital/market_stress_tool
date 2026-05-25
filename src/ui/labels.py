"""UI에 노출되는 모든 한국어 라벨/문구를 한 곳에 모은 모듈.

[수정 가이드]
    - 채널 이름, 패턴 이름, 위험도 라벨, 변수 단축 표기 등 UI에 보이는
      모든 한국어 문자열은 이 파일의 dict에만 정의합니다.
    - 새 변수가 variables.yaml에 추가되면 `VARIABLE_GEO_COORDS`에 좌표만
      추가하면 됩니다 (이름은 yaml의 name_kr를 자동 사용).
    - 색상 임계값(0~33/34~66/67~100)을 바꾸려면 `LEVEL_THRESHOLDS`만 수정.
"""

from __future__ import annotations

# =============================================================================
# 채널 라벨 (1~5)
# =============================================================================
CHANNEL_LABELS_KR: dict[int, str] = {
    1: "기업 펀더멘털",
    2: "무위험 실질금리",
    3: "위험 프리미엄",
    4: "공급환경",
    5: "환율 및 자본흐름",
}

# 채널 한 줄 설명 (설명 탭/툴팁용)
CHANNEL_DESCRIPTIONS_KR: dict[int, str] = {
    1: "기업의 미래 현금흐름 기대치. 글로벌 제조업 사이클과 한국 수출이 핵심 신호입니다.",
    2: "할인율의 무위험 성분. 미국 실질금리·기대인플레이션·정책금리·한미 금리차로 측정합니다.",
    3: "할인율의 위험 보상 성분. VIX·MOVE·하이일드 스프레드·달러 강세로 측정합니다.",
    4: "원자재 가격 충격에 의한 비용 인플레이션 채널. 1차 도구는 브렌트 유가로 대표합니다.",
    5: "한국 시장 특유 채널. 원/달러·엔/달러로 자본 유출 압력을 측정합니다.",
}


# =============================================================================
# 패턴 라벨
# =============================================================================
PATTERN_LABELS_KR: dict[str, str] = {
    "normal": "정상",
    "supply_shock": "공급 충격",
    "rate_shock": "금리 충격",
    "risk_premium_shock": "위험 프리미엄 충격",
    "real_recession": "실물 침체",
    "system_crisis": "시스템 위기",
}

PATTERN_DESCRIPTIONS_KR: dict[str, str] = {
    "normal": "특별한 충격 신호가 관측되지 않는 평상 상태입니다.",
    "supply_shock": "원자재 가격 급등이 비용 인플레이션 충격으로 작용하는 상태입니다. "
                    "S4(공급환경)가 2.0σ 이상이고 실물 침체 신호는 약합니다.",
    "rate_shock": "할인율 무위험 성분(실질금리·정책금리 기대)이 급격히 상승하는 상태입니다. "
                  "S2(무위험 실질금리)가 1.5σ 이상이고 위험 프리미엄 충격은 동반되지 않습니다.",
    "risk_premium_shock": "안전자산 선호가 급증하여 위험 프리미엄이 급등하는 상태입니다. "
                          "S3(위험 프리미엄)이 2.0σ 이상이나 실물·금리 동반 충격은 약합니다.",
    "real_recession": "실물 경기 둔화가 본격화되는 상태입니다. "
                      "S1(기업 펀더멘털)이 1.5σ 이상이고 30일 이동평균이 60일 이동평균을 상회합니다.",
    "system_crisis": "여러 채널이 동시에 임계를 돌파한 복합 위기 상태입니다 "
                     "(S1·S2·S5 모두 1.5σ↑, S3은 2.0σ↑).",
}


# =============================================================================
# 위험도 레벨 (백분위 → 라벨)
# =============================================================================
LEVEL_LABELS_KR: dict[str, str] = {
    "low": "정상",       # 0~33점
    "mid": "주의",       # 34~66점
    "high": "위험",      # 67~100점
}

# 임계값 (백분위)
LEVEL_THRESHOLDS = {
    "low_max": 33.0,     # 0~33: low
    "mid_max": 66.0,     # 34~66: mid
    # 67~100: high
}

# 위험도별 색상 (hex)
LEVEL_COLORS = {
    "low":  "#4CAF50",   # 녹색
    "mid":  "#FFC107",   # 노랑
    "high": "#F44336",   # 빨강
}


# =============================================================================
# 변수 단축 라벨 (지도/막대 텍스트가 길어지는 변수만 여기에 추가)
# 그 외는 variables.yaml의 name_kr 그대로 사용.
# =============================================================================
VARIABLE_SHORT_LABELS_KR: dict[str, str] = {
    "VIX": "VIX (공포지수)",
    "KRW_USD": "원/달러",
    "JPY_USD": "엔/달러",
    "KOSPI": "코스피",
    "KOSDAQ": "코스닥",
    "BRENT": "브렌트유",
    "DXY": "달러인덱스",
    "BDI": "BDI(해상운임)",
    "MOVE": "MOVE(국채변동성)",
    "INDPRO": "美 산업생산",
    "KR_EXPORT": "한국 수출",
    "US_TIPS_10Y": "美 10년 실질금리",
    "US_BEI_10Y": "美 기대인플레",
    "US_POLICY_RATE_EXPECT": "美 정책금리",
    "US_HY_SPREAD": "美 하이일드 스프레드",
    "KR_US_RATE_DIFF": "한미 금리차",
    "SP500_EPS": "S&P500 EPS",
    "FOREIGN_NET_BUY": "외국인 순매수",
}


# =============================================================================
# 변수 지리적 좌표 (지도 마커용)
# (위도, 경도) — 같은 지역 변수는 살짝씩 흩뿌려서 마커가 겹치지 않게 배치.
# =============================================================================
VARIABLE_GEO_COORDS: dict[str, tuple[float, float]] = {
    # ===== 미국 (40°N 근처, 경도 -95°W ~ -110°W에 분산) =====
    "VIX":                    (38.0,  -95.0),
    "MOVE":                   (40.0,  -97.0),
    "US_TIPS_10Y":            (42.0,  -99.0),
    "US_BEI_10Y":             (44.0, -101.0),
    "US_POLICY_RATE_EXPECT":  (36.0, -103.0),
    "US_HY_SPREAD":           (34.0, -105.0),
    "INDPRO":                 (46.0,  -93.0),
    "SP500_EPS":              (32.0,  -97.0),
    "DXY":                    (40.0, -110.0),  # 글로벌 의미지만 미국 발 통화로 미국 서쪽에 배치

    # ===== 한국 (37.5°N, 127.5°E 근처에 분산) =====
    "KOSPI":             (37.5, 127.0),
    "KOSDAQ":            (36.5, 127.5),
    "KRW_USD":           (38.5, 128.5),
    "KR_EXPORT":         (35.5, 129.5),
    "KR_US_RATE_DIFF":   (34.5, 126.5),
    "FOREIGN_NET_BUY":   (37.0, 130.0),

    # ===== 일본 =====
    "JPY_USD": (35.7, 139.7),

    # ===== 중동 (브렌트 유가) =====
    "BRENT": (25.0, 50.0),

    # ===== 해상 (글로벌 해운) =====
    "BDI": (0.0, 70.0),  # 인도양
}


# =============================================================================
# 패턴 분류 조건 설명 (설명 탭용 — pattern_diagnosis.PatternThresholds와 동기화)
# =============================================================================
PATTERN_CONDITIONS_KR: dict[str, str] = {
    "system_crisis":       "S1>1.5 AND S2>1.5 AND S3>2.0 AND S5>1.5",
    "risk_premium_shock":  "S3>2.0 AND S1<1.0 AND S2<1.0",
    "rate_shock":          "S2>1.5 AND S3<1.5",
    "real_recession":      "S1>1.5 AND MA30(S1) > MA60(S1)",
    "supply_shock":        "S4>2.0 AND S1<1.5",
    "normal":              "위 조건 모두 미해당",
}


# =============================================================================
# k-NN 테이블 한국어 헤더
# =============================================================================
KNN_COLUMN_LABELS_KR: dict[str, str] = {
    "date":     "날짜",
    "distance": "유사도 (Euclidean 거리)",
    "S1":       "기업 펀더멘털 (z)",
    "S2":       "무위험 실질금리 (z)",
    "S3":       "위험 프리미엄 (z)",
    "S4":       "공급환경 (z)",
    "S5":       "환율 및 자본흐름 (z)",
    "fwd_30d":  "30일 후 코스피 변화율",
    "fwd_90d":  "90일 후 코스피 변화율",
    "fwd_180d": "180일 후 코스피 변화율",
}


# =============================================================================
# 위험 방향 라벨
# v14: bidirectional 추가 (BRENT, US_BEI_10Y 등 양쪽 꼬리가 모두 위험인 변수).
# =============================================================================
RISK_DIRECTION_LABELS_KR: dict[str, str] = {
    "positive": "↑ 값이 클수록 위험 증가",
    "negative": "↓ 값이 클수록 위험 감소",
    "bidirectional": "↕ 적정 밴드 ±1.0σ 벗어난 정도가 위험 신호",
}


# =============================================================================
# 변수 사전 변환 라벨 (v13)
# z-score 산출 전 원변수에 적용되는 변환 종류를 사용자에게 표시.
# =============================================================================
TRANSFORM_LABELS_KR: dict[str, str] = {
    "level": "수준",
    "yoy_pct": "전년 동월 대비 % 변화",
    "pct_change_6m": "6개월 변화율",
    "diff_6m": "6개월 차분",
}


# =============================================================================
# 데이터 소스 라벨
# =============================================================================
SOURCE_LABELS_KR: dict[str, str] = {
    "fred":     "FRED (미국 연준)",
    "krx":      "한국거래소 통계정보",
    "ecos":     "한국은행 ECOS",
    "yfinance": "Yahoo Finance",
    "pykrx":    "pykrx (한국거래소)",
}


# =============================================================================
# UI 일반 문구
# =============================================================================
UI_TEXTS: dict[str, str] = {
    "app_title": "시장 스트레스 진단 도구",
    "tab_main": "메인 진단",
    "tab_explain": "설명",
    "tab_history": "과거 데이터 조회",

    "header_title": "",
    "asset_formula": "P = E[CF실질] / (r실질 + π위험)",
    "asset_formula_caption": "분자 = 미래 현금흐름 기대값  ·  분모 = 할인율(무위험 실질금리 + 위험 프리미엄)",

    "composite_label": "종합 스트레스 지수",
    "as_of_caption_fmt": "{date}, 한국 영업일 기준",

    "channel_section_title": "채널별 스트레스 (백분위)",
    "geo_section_title": "변수별 위험 지도",
    "ts_section_title": "종합 스트레스 시계열",
    "ts_period_start": "시작일",
    "ts_period_end": "종료일",
    "ts_query_button": "조회",
    "ts_breakdown_caption": "그래프의 한 시점을 클릭하면 그 날짜의 5채널 분해가 아래에 표시됩니다.",
    "ts_low_threshold_label": "정상 상한 (33점)",
    "ts_high_threshold_label": "위험 하한 (67점)",

    "origin_card_title": "충격 진원지",
    "origin_card_empty": "60일 룩백 내 1.5σ를 돌파한 변수가 없습니다 (시장 평온).",
    "origin_card_fmt": "{date}: {variable_kr}가 1.5σ를 처음 돌파했습니다 ({channel_kr} 채널).",

    "similar_card_title": "가장 비슷한 과거",
    "similar_card_date_fmt": "가장 비슷한 과거: {date}",
    "similar_card_returns_fmt": "이후 코스피: 30일 {f30:+.1%}  ·  90일 {f90:+.1%}  ·  180일 {f180:+.1%}",
    "similar_card_empty": "유사 시점을 찾지 못했습니다 (데이터 부족).",
    "similar_card_more": "→ 자세히 보기",

    "footer_sources": "데이터 출처: FRED · 한국거래소 통계정보 · Yahoo Finance · 한국은행 ECOS",
    "footer_disclaimer": "이 도구는 분석 보조용이며 투자 권유가 아닙니다.",

    "history_title": "임의 날짜 진단 조회",
    "history_date_label": "조회 기준일",
    "history_run_button": "그날 진단 보기",
    "history_knn_title": "유사 시점 상세 (k-NN 결과)",
    "history_knn_help_title": "테이블 컬럼이 의미하는 것",
    "history_knn_help_distance":
        "유사도: 5채널 z-score 벡터(S1~S5)의 Euclidean 거리. "
        "작을수록 그날과 비슷한 스트레스 상태입니다.",
    "history_knn_help_zscore":
        "채널별 z-score: 각 채널이 장기 평균(5년 롤링)에 비해 몇 표준편차 떨어져 있는지. "
        "+1.5σ 이상은 보통 충격 신호로 간주합니다.",
    "history_knn_help_returns":
        "30/90/180일 후 코스피 변화율: 그날 종가 → N일 후 종가의 단순 변화율입니다. "
        "인과관계가 아니라 통계적 동반 관측이며, 과거 패턴이 미래에 반복된다는 보장은 없습니다.",
    "history_knn_help_caveat":
        "한계: k-NN은 분모(할인율) 환경 유사성에 기반합니다. "
        "분자(기업 펀더멘털) 환경 차이는 반영하지 않으므로, "
        "유사 시점의 향후 수익률은 참고치로만 활용하세요.",

    "explain_title": "도구 설명",

    # 세부 내용 탭
    "tab_details": "세부 내용",
    "details_title": "변수별 세부 정보",
    "details_intro":
        "채널별로 묶어, 각 변수의 출처·현재값·z-score·백분위·1년 시계열을 "
        "확인할 수 있습니다. 각 변수 이름을 클릭하면 펼쳐집니다.",
    "details_field_source": "출처",
    "details_field_series_id": "시리즈 ID",
    "details_field_channel": "소속 채널",
    "details_field_risk_direction": "위험 방향",
    "details_field_transform": "사전 변환",
    "details_field_current_raw": "현재 값 (raw)",
    "details_field_current_z": "현재 z-score (5년 롤링)",
    "details_field_current_pct": "백분위 (0~100)",
    "details_field_contribution": "채널 점수 기여도",
    "details_field_timeseries": "최근 1년 시계열",
    "details_contribution_fmt":
        "이 변수는 채널 z-score에 약 {contrib:+.2f}σ 영향을 줬습니다 "
        "(채널 내 가중치 균등, 위험방향 부호 반영).",

    # 과거 조회 간소 헤더
    "history_summary_fmt": "종합 점수: {pct:.0f}점 ({level_kr})",
}


# =============================================================================
# 헬퍼 함수
# =============================================================================
def percentile_to_level(pct: float) -> str:
    """백분위 → 'low' | 'mid' | 'high' 레벨 키."""
    import math
    if math.isnan(pct):
        return "low"  # NaN은 기본 정상 표기
    if pct <= LEVEL_THRESHOLDS["low_max"]:
        return "low"
    if pct <= LEVEL_THRESHOLDS["mid_max"]:
        return "mid"
    return "high"


def percentile_to_color(pct: float) -> str:
    """백분위 → hex 색상."""
    return LEVEL_COLORS[percentile_to_level(pct)]


def percentile_to_level_label_kr(pct: float) -> str:
    """백분위 → 한국어 레벨 라벨 ('정상' | '주의' | '위험')."""
    return LEVEL_LABELS_KR[percentile_to_level(pct)]


def pattern_to_korean(label: str) -> str:
    """영문 패턴 키 → 한국어 라벨."""
    return PATTERN_LABELS_KR.get(label, label)


def channel_to_korean(channel: int) -> str:
    """채널 번호(1~5) → 한국어 라벨."""
    return CHANNEL_LABELS_KR.get(channel, f"채널 {channel}")


def transform_to_korean(transform: str | None) -> str:
    """transform 키 (v13) → 한국어 설명.

    알 수 없는 값이면 원값을 그대로 돌려주고,
    None이면 기본값 'level' 라벨을 돌려준다.
    """
    if not transform:
        return TRANSFORM_LABELS_KR["level"]
    return TRANSFORM_LABELS_KR.get(transform, transform)


def variable_to_korean(code: str, variables_meta: dict | None = None) -> str:
    """변수 코드 → 한국어 표시명.

    우선순위:
        1) `VARIABLE_SHORT_LABELS_KR` (단축 표기)
        2) `variables_meta[code]['name_kr']` (variables.yaml에서 로드한 메타)
        3) `code` 그대로
    """
    if code in VARIABLE_SHORT_LABELS_KR:
        return VARIABLE_SHORT_LABELS_KR[code]
    if variables_meta is not None and code in variables_meta:
        nm = variables_meta[code].get("name_kr")
        if nm:
            return nm
    return code
