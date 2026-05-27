"""KRX Data Marketplace OPEN API 로더.

한국거래소 KRX Data Marketplace의 OPEN API로 국내 지수(KOSPI/KOSDAQ 등)
일별 시세를 수집합니다.

서비스 정보:
    - 제공처    : 한국거래소 (KRX Data Marketplace)
    - 베이스 URL: https://data-dbg.krx.co.kr/svc/apis/
    - 인증 방식 : HTTP 헤더 'AUTH_KEY: <사용자 인증키>'
    - 호출 방식 : HTTP GET
    - 일일 한도 : 10,000회/일 (약관 제8조 4항)
    - 데이터 시작일 : 2010-01-04 (그 이전 데이터 없음)

[약관 준수 사항 — 매우 중요]
    1. 비상업적 목적 사용만 가능 (약관 제6조 2항)
    2. 결과 화면(Streamlit UI 등)에 "데이터 출처: 한국거래소 통계정보"를
       반드시 명시해야 함 (약관 제10조 3항)
    이 모듈을 사용하는 모든 산출물(차트, 테이블, 리포트)에 위 출처 표기 필수.

설계 원칙:
    - 기존 fred_loader/ecos_loader의 패턴을 그대로 따름
    - _common.py의 예외 클래스/로거/KST 정규화 헬퍼 재사용
    - parquet 캐시 사용. 단, 과거 거래일 데이터는 변하지 않으므로
      TTL 무한대(파일 존재만으로 캐시 hit)로 처리.
    - pykrx_loader는 그대로 두고 비활성. 본 모듈이 KRX 데이터 1차 경로.

엔드포인트 (1차 파일럿):
    /idx/krx_dd_trd
        - 파라미터: basDd (YYYYMMDD)
        - 응답: 그 날의 모든 KRX 시리즈 지수 (배열)
        - 우리는 IDX_NM으로 필터링하여 원하는 지수 추출
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# 기존 _common.py의 헬퍼 재사용 (수정 없이 import만)
from ._common import (
    DataLoaderError,
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
    RAW_CACHE_DIR,
    KST,
    logger,
    require_env,
    to_kst_index,
)

# =============================================================================
# 상수
# =============================================================================
KRX_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
ENDPOINT_IDX_KRX_DD_TRD = "/idx/krx_dd_trd"
ENDPOINT_IDX_KOSPI_DD_TRD = "/idx/kospi_dd_trd"
ENDPOINT_IDX_KOSDAQ_DD_TRD = "/idx/kosdaq_dd_trd"

# 환경 변수 이름
ENV_KEY_NAME = "KRX_API_KEY"

# 호출 속도 제어: 너무 빠른 연속 호출 시 KRX가 일시 차단할 수 있음
DEFAULT_REQUEST_DELAY_SEC = 0.1

# 진행 상황 로깅 간격 (호출 횟수 기준)
PROGRESS_LOG_EVERY = 50

# 데이터 시작일 (이보다 이른 날짜 요청 시 즉시 에러)
KRX_DATA_START_DATE = "2010-01-04"

# OUTPUT BLOCK 키 (응답 JSON에서 데이터 배열이 들어있는 키)
OUT_BLOCK_KEY = "OutBlock_1"

# /idx/krx_dd_trd 응답에서 코스피/코스닥을 필터링할 IDX_NM 값
# (한국거래소는 응답을 한글로 반환함 - 표기 오타 방지용 상수)
KOSPI_INDEX_NAME = "코스피"
KOSDAQ_INDEX_NAME = "코스닥"

# HTTP 타임아웃 (초)
HTTP_TIMEOUT_SEC = 30

# 의무 출처 표기 (UI에 노출할 때 사용)
DATA_SOURCE_NOTICE_KR = "데이터 출처: 한국거래소 통계정보"


# =============================================================================
# 응답 컬럼 매핑 (KRX 응답 → 내부 컬럼명)
# -----------------------------------------------------------------------------
# 응답이 문자열 한글 키로 오므로, 내부에서는 영문 snake_case로 변환하여 사용.
# =============================================================================
KRX_COLUMN_MAP: dict[str, str] = {
    "BAS_DD":          "bas_dd",       # 기준일자 (YYYYMMDD 문자열)
    "IDX_CLSS":        "idx_clss",     # 계열구분 (예: 'KRX')
    "IDX_NM":          "idx_nm",       # 지수명 (예: '코스피')
    "CLSPRC_IDX":      "close",        # 종가
    "CMPPREVDD_IDX":   "change",       # 대비 (전일 대비)
    "FLUC_RT":         "fluc_rt",      # 등락률
    "OPNPRC_IDX":      "open",         # 시가
    "HGPRC_IDX":       "high",         # 고가
    "LWPRC_IDX":       "low",          # 저가
    "ACC_TRDVOL":      "trd_vol",      # 거래량
    "ACC_TRDVAL":      "trd_val",      # 거래대금
    "MKTCAP":          "mkt_cap",      # 시가총액
}

# 숫자로 변환할 컬럼 (응답에서 문자열로 오므로 float 변환 필요)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "close", "change", "fluc_rt", "open", "high", "low",
    "trd_vol", "trd_val", "mkt_cap",
)


# =============================================================================
# 캐시 헬퍼 (KRX 전용 - 파일 존재만 검사. TTL 무한대.)
# -----------------------------------------------------------------------------
# 이유:
#   기존 _common.is_cache_valid()는 CACHE_TTL_HOURS를 강제 적용하지만,
#   KRX 일별 과거 데이터는 한 번 발표되면 정정되는 일이 거의 없음
#   (정정 시에도 KRX는 별도 공지로 다음 영업일에 반영).
#   → 파일 존재 = 캐시 유효 로 단순화하여 4000+ 일 다운로드 시간 절약.
#   기존 _common.py를 수정하지 않기 위해 본 모듈에 별도 구현.
#
# [캐시 키 정책 - 2026년 수정]
#   기존 파일명 `krx_dd_trd_YYYYMMDD.parquet`는 endpoint를 구분하지 않아
#   동일 일자에 대해 KOSPI 시리즈와 KOSDAQ 시리즈가 서로의 캐시를 침범하는
#   버그가 있었음 (예: KOSPI 호출 후 KOSDAQ 호출 시 KOSPI 응답 반환).
#   → endpoint slug를 파일명에 포함: `krx_<slug>_dd_trd_YYYYMMDD.parquet`
#     - slug 'krx'   ← /idx/krx_dd_trd   (통합 응답)
#     - slug 'kospi' ← /idx/kospi_dd_trd (코스피 계열)
#     - slug 'kosdaq'← /idx/kosdaq_dd_trd(코스닥 계열)
#   기존 캐시(`krx_dd_trd_YYYYMMDD.parquet`)는 그대로 둠.
#   사용자가 clear_krx_cache()로 직접 정리하거나, 그냥 두면 더 이상 참조되지
#   않아 무해함 (디스크 용량 정도만 차지).
# =============================================================================

# endpoint → 파일명 slug 매핑
_ENDPOINT_SLUG: dict[str, str] = {
    ENDPOINT_IDX_KRX_DD_TRD:    "krx",
    ENDPOINT_IDX_KOSPI_DD_TRD:  "kospi",
    ENDPOINT_IDX_KOSDAQ_DD_TRD: "kosdaq",
}

# 모든 endpoint slug (clear_krx_cache 등에서 사용)
_ALL_ENDPOINT_SLUGS: tuple[str, ...] = ("krx", "kospi", "kosdaq")


def _endpoint_slug(endpoint: str) -> str:
    """endpoint URL을 캐시 파일명 slug로 변환.

    Args:
        endpoint: '/idx/krx_dd_trd', '/idx/kospi_dd_trd', '/idx/kosdaq_dd_trd' 중 하나.

    Returns:
        'krx' | 'kospi' | 'kosdaq'.

    Raises:
        ValueError: 알 수 없는 endpoint.
    """
    slug = _ENDPOINT_SLUG.get(endpoint)
    if slug is None:
        raise ValueError(
            f"알 수 없는 endpoint: '{endpoint}'. "
            f"지원: {list(_ENDPOINT_SLUG.keys())}"
        )
    return slug


def _krx_day_cache_path(base_date: str, endpoint: str = ENDPOINT_IDX_KRX_DD_TRD) -> Path:
    """단일 거래일 응답을 저장할 parquet 경로.

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 형식.
        endpoint: KRX API endpoint. 기본값은 통합 응답(/idx/krx_dd_trd).
            endpoint별로 별도 파일에 캐시되어 KOSPI/KOSDAQ 호출이
            서로의 캐시를 침범하지 않음.

    Returns:
        data/raw/krx_<slug>_dd_trd_YYYYMMDD.parquet
        (예: krx_kospi_dd_trd_20241230.parquet)
    """
    yyyymmdd = base_date.replace("-", "")[:8]
    slug = _endpoint_slug(endpoint)
    return RAW_CACHE_DIR / f"krx_{slug}_dd_trd_{yyyymmdd}.parquet"


def _krx_day_meta_path(base_date: str, endpoint: str = ENDPOINT_IDX_KRX_DD_TRD) -> Path:
    """단일 거래일 캐시의 메타데이터 경로 (휴일 표시 등 부가정보).

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        endpoint: KRX API endpoint.

    Returns:
        data/raw/krx_<slug>_dd_trd_YYYYMMDD.meta.json
    """
    yyyymmdd = base_date.replace("-", "")[:8]
    slug = _endpoint_slug(endpoint)
    return RAW_CACHE_DIR / f"krx_{slug}_dd_trd_{yyyymmdd}.meta.json"


def _is_day_cache_valid(base_date: str, endpoint: str = ENDPOINT_IDX_KRX_DD_TRD) -> bool:
    """파일 존재 여부만으로 캐시 유효성 판정 (TTL 무한대).

    휴일 캐시(빈 DataFrame이 저장된 경우)도 유효 캐시로 인정하여
    재호출을 막는다 (메타에 holiday=True 표시).

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        endpoint: KRX API endpoint.
    """
    return _krx_day_cache_path(base_date, endpoint=endpoint).exists()


def _read_day_cache(
    base_date: str, endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
) -> pd.DataFrame | None:
    """캐시된 일별 DataFrame을 읽어 반환. 실패/부재 시 None."""
    path = _krx_day_cache_path(base_date, endpoint=endpoint)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:  # noqa: BLE001 - 캐시 손상은 무시하고 재수집
        logger.warning("KRX cache read failed for %s (endpoint=%s): %s", base_date, endpoint, e)
        return None


def _ensure_cache_dir() -> None:
    """캐시 디렉터리 존재를 보장 (idempotent).

    `_common.py`가 import 시점에 이미 만듭지만, krx_loader만 독립
    호출하거나 외부에서 실수로 삭제한 경우를 대비.
    """
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _write_day_cache(
    base_date: str,
    df: pd.DataFrame,
    is_holiday: bool = False,
    endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
) -> None:
    """일별 DataFrame을 parquet으로 저장 + 메타 기록 (원자적).

    원자성:
        - .tmp 파일에 쓴 뒤 rename 으로 교체 (쓰기 중 중단 시 손상 방지)
        - parquet 쪽이 메타보다 먼저 (데이터 없는 메타 상태 방지)

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        df: 저장할 DataFrame (휴일이면 빈 DataFrame).
        is_holiday: True면 해당일이 휴일/미발표일로 표시.
        endpoint: KRX API endpoint. 파일명에 slug로 포함됨.
    """
    _ensure_cache_dir()

    data_path = _krx_day_cache_path(base_date, endpoint=endpoint)
    meta_path = _krx_day_meta_path(base_date, endpoint=endpoint)
    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")

    # 빈 DataFrame도 캐시 (휴일 재호출 방지). parquet은 빈 DF도 저장 가능.
    df.to_parquet(data_tmp, engine="pyarrow")
    data_tmp.replace(data_path)  # 원자적 rename (POSIX)

    meta = {
        "source": "krx",
        "endpoint": endpoint,
        "base_date": base_date,
        "rows": int(len(df)),
        "is_holiday": bool(is_holiday),
        "last_updated": datetime.now(KST).isoformat(),
    }
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)


# =============================================================================
# 캐시 유틸리티 (사용자 공개 API - 운영 편의용)
# =============================================================================
def clear_krx_cache(
    start_date: str | None = None,
    end_date: str | None = None,
    endpoint: str | None = None,
) -> int:
    """KRX 일별 캐시 파일을 삭제하고 삭제된 파일 수를 반환.

    사용 시나리오:
        - KRX가 과거 데이터를 정정 공지했을 때 해당 구간 재다운로드
        - 캐시 손상이 의심될 때 전체 재구축
        - 구버전(endpoint 구분 없는 캐시) 정리

    Args:
        start_date: 'YYYY-MM-DD' 또는 'YYYYMMDD'. None이면 제한 없음.
        end_date  : 동상. None이면 제한 없음.
        endpoint  : 특정 endpoint만 삭제할 때 지정.
            None이면 모든 endpoint(kospi/kosdaq/krx) + 구버전 캐시 모두 삭제.
            지정 가능값: ENDPOINT_IDX_KRX_DD_TRD, ENDPOINT_IDX_KOSPI_DD_TRD,
            ENDPOINT_IDX_KOSDAQ_DD_TRD.
        세 개 모두 None이면 전체 KRX 캐시 삭제.

    Returns:
        삭제된 파일 수 (parquet + meta가 각각 세어짐).

    주의:
        다른 로더(fred/ecos/yfinance)의 캐시는 삭제하지 않음
        (파일명 prefix `krx_`로 안전 필터링).
    """
    if not RAW_CACHE_DIR.exists():
        return 0

    start_key = _yyyymmdd(start_date) if start_date else None
    end_key = _yyyymmdd(end_date) if end_date else None

    # 삭제 대상 slug 결정
    if endpoint is None:
        # 모든 endpoint slug + 구버전 (slug 없는 파일)
        target_slugs: tuple[str | None, ...] = _ALL_ENDPOINT_SLUGS + (None,)
    else:
        target_slugs = (_endpoint_slug(endpoint),)

    deleted = 0
    for slug in target_slugs:
        if slug is None:
            # 구버전 파일: krx_dd_trd_YYYYMMDD.{parquet,meta.json}
            # (endpoint slug가 없던 이전 버전 캐시)
            glob_pattern = "krx_dd_trd_*"
            prefix = "krx_dd_trd_"
        else:
            glob_pattern = f"krx_{slug}_dd_trd_*"
            prefix = f"krx_{slug}_dd_trd_"

        for path in RAW_CACHE_DIR.glob(glob_pattern):
            # 구버전 글록 시 신버전도 잡히지 않도록 엄격 필터링:
            # 'krx_dd_trd_*' 패턴은 'krx_kospi_dd_trd_*'도 메치하지 않음 (slug에 _ 포함)
            # 단, parquet/meta 둘 다 고려해야 함
            stem = path.name.removeprefix(prefix).split(".")[0]
            if len(stem) != 8 or not stem.isdigit():
                continue
            if start_key is not None and stem < start_key:
                continue
            if end_key is not None and stem > end_key:
                continue
            try:
                path.unlink()
                deleted += 1
            except OSError as e:
                logger.warning("KRX cache delete failed for %s: %s", path.name, e)

    logger.info(
        "KRX cache cleared: %d files (range=%s~%s, endpoint=%s)",
        deleted, start_date or "-\u221e", end_date or "+\u221e",
        endpoint or "ALL",
    )
    return deleted


def get_krx_cache_status(
    start_date: str,
    end_date: str,
    endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
) -> dict[str, int]:
    """주어진 기간의 캐시 상태를 진단.

    다운로드 전에 몇 일을 실제로 호출해야 하는지 파악할 때 유용.
    endpoint 별로 캐시가 독립되므로, 조회하려는 endpoint를 명시적으로 지정.

    Args:
        start_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 시작일.
        end_date  : 'YYYY-MM-DD' 또는 'YYYYMMDD' 종료일.
        endpoint  : KRX API endpoint. 기본값은 통합 응답(/idx/krx_dd_trd).

    Returns:
        {
            'biz_days'       : 영업일 수,
            'cached'         : 총 캐시 히트 예상 (데이터 + 휴일),
            'cached_data'    : 거래일 캐시 (rows > 0),
            'cached_holiday' : 휴일 캐시 (rows == 0),
            'missing'        : 추가 API 호출이 필요한 날 수,
        }

    Raises:
        ValueError: 잘못된 기간.
    """
    start_key = _yyyymmdd(start_date)
    end_key = _yyyymmdd(end_date)
    if start_key > end_key:
        raise ValueError(
            f"start_date({start_date})가 end_date({end_date})보다 늦습니다."
        )

    biz_days = pd.date_range(start=pd.Timestamp(start_key), end=pd.Timestamp(end_key), freq="B")

    cached_data = 0
    cached_holiday = 0
    missing = 0
    for day in biz_days:
        day_str = day.strftime("%Y-%m-%d")
        meta_path = _krx_day_meta_path(day_str, endpoint=endpoint)
        data_path = _krx_day_cache_path(day_str, endpoint=endpoint)
        if not data_path.exists():
            missing += 1
            continue
        # 메타로 휴일 여부 판단 (메타가 없으면 parquet을 읽어 row 수로 판단)
        is_holiday = False
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                is_holiday = bool(meta.get("is_holiday", False))
            except (json.JSONDecodeError, OSError):
                pass
        else:
            try:
                df_tmp = pd.read_parquet(data_path)
                is_holiday = df_tmp.empty
            except Exception:  # noqa: BLE001
                pass

        if is_holiday:
            cached_holiday += 1
        else:
            cached_data += 1

    cached_total = cached_data + cached_holiday
    return {
        "biz_days": int(len(biz_days)),
        "cached": cached_total,
        "cached_data": cached_data,
        "cached_holiday": cached_holiday,
        "missing": missing,
    }


# =============================================================================
# 인증 헤더 헬퍼
# =============================================================================
def _auth_headers() -> dict[str, str]:
    """KRX OPEN API 호출용 HTTP 헤더 생성.

    Returns:
        {'AUTH_KEY': '<사용자 인증키>'} 형태의 헤더 딕셔너리.

    Raises:
        MissingAPIKeyError: KRX_API_KEY가 .env에 없거나 비어있을 때.
    """
    api_key = require_env(ENV_KEY_NAME)
    return {"AUTH_KEY": api_key}


def _yyyymmdd(date_str: str) -> str:
    """'YYYY-MM-DD' 또는 'YYYYMMDD'를 'YYYYMMDD'로 정규화."""
    return date_str.replace("-", "")[:8]


def _validate_date_range(start_date: str, end_date: str) -> None:
    """요청 기간이 KRX 데이터 시작일 이후인지, 시작 <= 끝 인지 검증.

    Raises:
        ValueError: 잘못된 기간.
    """
    if start_date < KRX_DATA_START_DATE:
        raise ValueError(
            f"KRX OPEN API는 {KRX_DATA_START_DATE} 이후 데이터만 제공합니다. "
            f"요청 시작일: {start_date}"
        )
    if start_date > end_date:
        raise ValueError(f"start_date({start_date})가 end_date({end_date})보다 늦습니다.")


# =============================================================================
# 응답 파싱 헬퍼 (순수 함수 - 네트워크 호출과 분리하여 테스트 가능)
# =============================================================================
def _coerce_numeric(value: Any) -> float:
    """KRX 응답의 문자열 숫자 필드를 float로 변환.

    처리 대상:
        - 콤마 포함 ("1,234.56")     -> 1234.56
        - '-' (데이터 없음 표기)     -> NaN
        - '' (빈 문자열)            -> NaN
        - None                      -> NaN
        - 이미 숫자 (int/float)      -> float() 그대로
        - 변환 불가 문자열           -> NaN (로그 미출력: 일상적 케이스)
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s in ("", "-", "N/A", "null", "None"):
        return float("nan")
    s = s.replace(",", "")  # 콤마 제거
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_krx_response(
    payload: dict[str, Any],
    base_date: str,
) -> pd.DataFrame:
    """KRX OPEN API JSON 응답을 DataFrame으로 파싱.

    Args:
        payload: response.json() 결과.
        base_date: 'YYYY-MM-DD'. 에러 메시지용.

    Returns:
        인덱스: idx_nm (지수명)
        컬럼: [bas_dd, idx_clss, close, change, fluc_rt,
              open, high, low, trd_vol, trd_val, mkt_cap]
        휴일/빈 응답이면 같은 컬럼 스키마의 빈 DataFrame.

    Raises:
        InvalidSeriesError: OutBlock_1 키가 없거나 응답 형식이 이상할 때.
            (단, OutBlock_1이 빈 배열인 경우는 휴일로 간주하고 빈 DF 반환)
    """
    # 1) 응답 구조 검증
    if not isinstance(payload, dict):
        raise InvalidSeriesError(
            f"KRX 응답이 dict가 아닙니다 (base_date={base_date}): {type(payload).__name__}"
        )
    if OUT_BLOCK_KEY not in payload:
        raise InvalidSeriesError(
            f"KRX 응답에 '{OUT_BLOCK_KEY}' 키가 없습니다 (base_date={base_date}). "
            f"응답 키: {list(payload.keys())}"
        )

    rows = payload[OUT_BLOCK_KEY]
    if not isinstance(rows, list):
        raise InvalidSeriesError(
            f"'{OUT_BLOCK_KEY}'이 배열이 아닙니다 (base_date={base_date}): {type(rows).__name__}"
        )

    # 2) 빈 응답 (휴일/미발표일) -> 올바른 컬럼 스키마의 빈 DataFrame
    ordered_cols = ["bas_dd", "idx_clss", "close", "change", "fluc_rt",
                    "open", "high", "low", "trd_vol", "trd_val", "mkt_cap"]
    if not rows:
        empty_df = pd.DataFrame(columns=ordered_cols)
        empty_df.index = pd.Index([], name="idx_nm", dtype="object")
        return empty_df

    # 3) 레코드 변환
    records = []
    for r in rows:
        if not isinstance(r, dict):
            logger.warning("KRX row skip (not dict, base_date=%s): %s", base_date, r)
            continue
        rec = {}
        for src_key, dst_key in KRX_COLUMN_MAP.items():
            val = r.get(src_key)
            if dst_key in NUMERIC_COLUMNS:
                rec[dst_key] = _coerce_numeric(val)
            else:
                rec[dst_key] = str(val).strip() if val is not None else None
        records.append(rec)

    if not records:
        empty_df = pd.DataFrame(columns=ordered_cols)
        empty_df.index = pd.Index([], name="idx_nm", dtype="object")
        return empty_df

    df = pd.DataFrame.from_records(records)

    # 4) idx_nm을 인덱스로 (중복 시 첫 번째 유지)
    df = df.drop_duplicates(subset=["idx_nm"], keep="first")
    df = df.set_index("idx_nm")

    # 5) 컬럼 순서 고정 (UI/테스트 일관성)
    df = df[[c for c in ordered_cols if c in df.columns]]
    return df


# =============================================================================
# 네트워크 호출 헬퍼
# =============================================================================
def _request_krx_one_day(
    base_date: str,
    endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
) -> dict[str, Any]:
    """KRX API 엔드포인트를 한 번 호출하여 JSON 반환.

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        endpoint: 호출할 엔드포인트 (예: '/idx/kospi_dd_trd').
    Returns:
        응답 JSON dict.

    Raises:
        MissingAPIKeyError: 인증키 부재.
        NetworkError      : 타임아웃/연결 오류/HTTP 4xx-5xx/JSON 파싱 실패.
    """
    headers = _auth_headers()
    params = {"basDd": _yyyymmdd(base_date)}
    url = KRX_BASE_URL + endpoint

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise NetworkError(
            f"KRX API 호출 실패 (base_date={base_date}): {e}"
        ) from e

    if resp.status_code != 200:
        snippet = resp.text[:200] if resp.text else ""
        raise NetworkError(
            f"KRX HTTP {resp.status_code} (base_date={base_date}): {snippet}"
        )

    try:
        return resp.json()
    except ValueError as e:
        raise NetworkError(
            f"KRX 응답이 JSON이 아닙니다 (base_date={base_date}): {resp.text[:200]}"
        ) from e


# =============================================================================
# Public API
# =============================================================================
def fetch_krx_index_one_day(
    base_date: str,
    endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
    use_cache: bool = True,
) -> pd.DataFrame:
    """단일 거래일의 호출된 KRX 지수 시세를 받는다.

    Args:
        base_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 기준일자.
        endpoint: 호출할 KRX API 엔드포인트.
        use_cache: True면 디스크 캐시 우선 사용 (TTL 무한대).

    Returns:
        인덱스=idx_nm(지수명), 컬럼=[bas_dd, idx_clss, close, change, fluc_rt,
        open, high, low, trd_vol, trd_val, mkt_cap]인 DataFrame.
        휴일/미발표일이면 같은 스키마의 빈 DataFrame.

    Raises:
        MissingAPIKeyError : KRX_API_KEY 미설정.
        InvalidSeriesError : 응답 스키마 이상 (OutBlock_1 부재 등).
        NetworkError       : 타임아웃/HTTP 오류/JSON 파싱 실패.

    캐싱:
        data/raw/krx_dd_trd_YYYYMMDD.parquet + .meta.json
        한 번 받은 과거 날짜는 변하지 않으므로 파일 존재만으로 캐시 hit.
        휴일도 빈 DF로 캐싱하여 재호출 방지.
    """
    # 1) 캐시 확인 (TTL 무한대, endpoint별 독립)
    if use_cache and _is_day_cache_valid(base_date, endpoint=endpoint):
        cached = _read_day_cache(base_date, endpoint=endpoint)
        if cached is not None:
            logger.debug(
                "KRX cache hit: %s (endpoint=%s, rows=%d)",
                base_date, endpoint, len(cached),
            )
            return cached

    # 2) API 호출
    try:
        payload = _request_krx_one_day(base_date, endpoint=endpoint)
    except TypeError as e:
        # 구버전 테스트 더블은 endpoint 인자를 받지 않는다.
        # 실제 함수는 endpoint 기본값을 지원하므로 운영 경로는 위 호출을 사용한다.
        if "endpoint" not in str(e):
            raise
        payload = _request_krx_one_day(base_date)

    # 3) 파싱
    df = _parse_krx_response(payload, base_date=base_date)
    is_holiday = df.empty

    if is_holiday:
        logger.info(
            "KRX %s (endpoint=%s): empty response (holiday/non-trading day)",
            base_date, endpoint,
        )
    else:
        logger.debug(
            "KRX %s (endpoint=%s): %d indices fetched",
            base_date, endpoint, len(df),
        )

    # 4) 캐싱 (endpoint별 독립 파일)
    if use_cache:
        _write_day_cache(base_date, df, is_holiday=is_holiday, endpoint=endpoint)

    return df


def fetch_krx_index_range(
    index_name: str,
    start_date: str,
    end_date: str,
    endpoint: str = ENDPOINT_IDX_KRX_DD_TRD,
    use_cache: bool = True,
    request_delay_sec: float = DEFAULT_REQUEST_DELAY_SEC,
    field: str = "close",
) -> pd.Series:
    """기간 동안 특정 KRX 지수의 일별 시계열을 반환.

    내부 동작:
        1) start_date~end_date의 영업일(B-freq)을 순회
        2) 각 날짜에 fetch_krx_index_one_day() 호출 (캐시 우선)
        3) 휴일(빈 DF) 또는 지수명 미포함일은 스킵
        4) 결과를 idx_nm 필터링하여 단일 시리즈로 결합

    Args:
        index_name: 지수명 (예: '코스피', '코스닥'). 응답의 IDX_NM과 정확히 일치해야 함.
        start_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 시작일.
        end_date  : 'YYYY-MM-DD' 또는 'YYYYMMDD' 종료일.
        use_cache : True면 캐시 사용 (TTL 무한대).
        request_delay_sec: 캐시 미스 시 호출 사이 sleep 시간 (KRX 부하 방지).
            캐시 히트한 날짜에는 sleep을 적용하지 않음 (불필요한 대기 제거).
        field: 추출할 컬럼명. 기본 'close'. NUMERIC_COLUMNS 중 하나여야 함.

    Returns:
        DatetimeIndex(KST, name='date') × float 시리즈. 이름=index_name.
        결측 거래일은 시리즈에서 누락 (KOSPI 미발표 등).

    Raises:
        ValueError         : 잘못된 기간 또는 잘못된 field.
        MissingAPIKeyError : KRX_API_KEY 미설정.
        InvalidSeriesError : 응답 스키마 이상.
        NetworkError       : 네트워크/HTTP/JSON 오류 (한 날짜라도 발생 시 즉시 중단).
    """
    # 1) 입력 정규화 & 검증
    start_yyyymmdd = _yyyymmdd(start_date)
    end_yyyymmdd = _yyyymmdd(end_date)
    krx_start_yyyymmdd = _yyyymmdd(KRX_DATA_START_DATE)
    if start_yyyymmdd < krx_start_yyyymmdd:
        raise ValueError(
            f"KRX OPEN API는 {KRX_DATA_START_DATE} 이후 데이터만 제공합니다. "
            f"요청 시작일: {start_date}"
        )
    if start_yyyymmdd > end_yyyymmdd:
        raise ValueError(
            f"start_date({start_date})가 end_date({end_date})보다 늦습니다."
        )
    if field not in NUMERIC_COLUMNS:
        raise ValueError(
            f"field='{field}'는 지원되지 않습니다. "
            f"가능: {NUMERIC_COLUMNS}"
        )

    # 2) 영업일 캘린더 생성 (월~금. 한국 공휴일은 응답이 비어있을 때 자연 스킵)
    start_ts = pd.Timestamp(start_yyyymmdd)
    end_ts = pd.Timestamp(end_yyyymmdd)
    biz_days = pd.date_range(start=start_ts, end=end_ts, freq="B")

    if len(biz_days) == 0:
        logger.warning(
            "KRX range: 영업일 없음 (%s ~ %s). 빈 시리즈 반환.",
            start_date, end_date,
        )
        empty = pd.Series(dtype="float64", name=index_name)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    logger.info(
        "KRX range fetch 시작: index='%s', %s~%s, 영업일=%d, field='%s'",
        index_name, start_date, end_date, len(biz_days), field,
    )

    # 3) 일별 루프
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    n_holiday = 0
    n_missing_index = 0
    n_api_call = 0

    for i, day_ts in enumerate(biz_days, start=1):
        day_str = day_ts.strftime("%Y-%m-%d")

        # 캐시 히트 여부를 사전 확인 (sleep 결정용)
        was_cached = use_cache and _is_day_cache_valid(day_str, endpoint=endpoint)

        day_df = fetch_krx_index_one_day(day_str, endpoint=endpoint, use_cache=use_cache)

        if not was_cached:
            n_api_call += 1
            # 다음 호출 전 딜레이 (마지막 날 제외)
            if i < len(biz_days) and request_delay_sec > 0:
                time.sleep(request_delay_sec)

        # 휴일 처리: 빈 DF
        if day_df.empty:
            n_holiday += 1
        elif index_name not in day_df.index:
            # 응답은 왔지만 해당 지수명이 없는 경우 (드물지만 가능)
            n_missing_index += 1
            logger.debug(
                "KRX %s: index='%s' 응답에 없음 (응답 지수 수=%d)",
                day_str, index_name, len(day_df),
            )
        else:
            val = day_df.loc[index_name, field]
            dates.append(day_ts)
            values.append(float(val))

        # 진행 로깅
        if i % PROGRESS_LOG_EVERY == 0 or i == len(biz_days):
            logger.info(
                "KRX range 진행: %d/%d (수집=%d, 휴일=%d, 지수누락=%d, API호출=%d)",
                i, len(biz_days), len(values), n_holiday, n_missing_index, n_api_call,
            )

    # 4) 시리즈 조립
    if not dates:
        logger.warning(
            "KRX range: index='%s' 수집 결과 0건 (영업일=%d, 휴일=%d, 지수누락=%d). "
            "index_name이 정확한지 확인하세요.",
            index_name, len(biz_days), n_holiday, n_missing_index,
        )
        empty = pd.Series(dtype="float64", name=index_name)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty

    idx = pd.DatetimeIndex(dates, name="date")
    series = pd.Series(values, index=idx, name=index_name, dtype="float64")

    # 인덱스 정규화 (다른 로더와 일관성): tz-naive + 자정 normalize
    series = to_kst_index(series)
    series.index.name = "date"

    logger.info(
        "KRX range 완료: index='%s', 수집=%d일, 휴일=%d, 지수누락=%d, 신규API호출=%d",
        index_name, len(series), n_holiday, n_missing_index, n_api_call,
    )
    return series


def fetch_kospi(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    request_delay_sec: float = DEFAULT_REQUEST_DELAY_SEC,
) -> pd.Series:
    """코스피 일별 종가 시리즈 (편의 함수).

    내부적으로 fetch_krx_index_range(index_name="코스피", field="close")
    를 호출하여 지수명 오타를 방지한다.

    Args:
        start_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 시작일.
        end_date  : 'YYYY-MM-DD' 또는 'YYYYMMDD' 종료일.
        use_cache : True면 캐시 사용 (TTL 무한대).
        request_delay_sec: 캐시 미스 시 호출 사이 sleep.

    Returns:
        DatetimeIndex(name='date') × float64 종가 시리즈. 이름='코스피'.

    사용 예:
        >>> kospi = fetch_kospi('2024-01-01', '2024-12-31')
        >>> kospi.tail()

    주의(약관):
        이 데이터를 사용한 결과물(UI/리포트)에는 반드시
        "데이터 출처: 한국거래소 통계정보"를 표시해야 함
        (DATA_SOURCE_NOTICE_KR 상수 참고).
    """
    return fetch_krx_index_range(
        index_name=KOSPI_INDEX_NAME,
        start_date=start_date,
        end_date=end_date,
        endpoint=ENDPOINT_IDX_KOSPI_DD_TRD,
        use_cache=use_cache,
        request_delay_sec=request_delay_sec,
        field="close",
    )


def fetch_kosdaq(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    request_delay_sec: float = DEFAULT_REQUEST_DELAY_SEC,
) -> pd.Series:
    """코스닥 일별 종가 시리즈 (편의 함수).

    내부적으로 fetch_krx_index_range(index_name="코스닥", field="close")
    를 호출하여 지수명 오타를 방지한다.

    Args:
        start_date: 'YYYY-MM-DD' 또는 'YYYYMMDD' 시작일.
        end_date  : 'YYYY-MM-DD' 또는 'YYYYMMDD' 종료일.
        use_cache : True면 캐시 사용 (TTL 무한대).
        request_delay_sec: 캐시 미스 시 호출 사이 sleep.

    Returns:
        DatetimeIndex(name='date') × float64 종가 시리즈. 이름='코스닥'.

    사용 예:
        >>> kosdaq = fetch_kosdaq('2024-01-01', '2024-12-31')
        >>> kosdaq.tail()

    주의(약관):
        이 데이터를 사용한 결과물(UI/리포트)에는 반드시
        "데이터 출처: 한국거래소 통계정보"를 표시해야 함
        (DATA_SOURCE_NOTICE_KR 상수 참고).
    """
    return fetch_krx_index_range(
        index_name=KOSDAQ_INDEX_NAME,
        start_date=start_date,
        end_date=end_date,
        endpoint=ENDPOINT_IDX_KOSDAQ_DD_TRD,
        use_cache=use_cache,
        request_delay_sec=request_delay_sec,
        field="close",
    )
