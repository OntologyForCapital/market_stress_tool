"""한국은행 ECOS (Economic Statistics System) 로더.

ECOS REST API를 requests로 직접 호출합니다 (라이브러리 의존성 최소화).
API 문서: https://ecos.bok.or.kr/api/

이 도구에서 사용하는 ECOS 시리즈:
    722Y001  : 한국 기준금리 (일별, 통계코드 - 한국은행 기준금리)
    901Y009  : 한국 CPI (월별, 보조 변수)
    403Y001  : 통관기준 수출입 (월별, KR_EXPORT 후보)

ECOS URL 형식:
    https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/{start}/{end}/{stat_code}/{cycle}/{from_date}/{to_date}/{item_code1}/.../{item_code4}

[한계]
    - ECOS는 일일 호출 한도가 있음 (10,000회/일, 1회당 최대 10만건)
    - 시리즈마다 cycle(D/M/Q/A)이 다르므로 호출 전 확인 필요
    - 시계열에 따라 발표 지연 있음 (월별: 보통 익월 중순)
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import requests

from ._common import (
    DataLoaderError,
    InvalidSeriesError,
    NetworkError,
    is_cache_valid,
    logger,
    read_cache,
    require_env,
    to_kst_index,
    write_cache,
    _cache_paths,
)

# ECOS API 베이스 URL
ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"

# 주기 코드 (cycle)
CYCLE_DAILY = "D"
CYCLE_MONTHLY = "M"
CYCLE_QUARTERLY = "Q"
CYCLE_ANNUAL = "A"


def _format_date_for_cycle(date_str: str, cycle: str) -> str:
    """'YYYY-MM-DD'를 ECOS의 주기별 포맷으로 변환.

        D -> YYYYMMDD
        M -> YYYYMM
        Q -> YYYYQ#  (ECOS는 분기를 YYYYQ1~Q4 로 받음)
        A -> YYYY
    """
    d = pd.to_datetime(date_str)
    if cycle == CYCLE_DAILY:
        return d.strftime("%Y%m%d")
    if cycle == CYCLE_MONTHLY:
        return d.strftime("%Y%m")
    if cycle == CYCLE_QUARTERLY:
        q = (d.month - 1) // 3 + 1
        return f"{d.year}Q{q}"
    if cycle == CYCLE_ANNUAL:
        return d.strftime("%Y")
    raise ValueError(f"Unsupported cycle: {cycle}")


def _parse_ecos_time(time_str: str, cycle: str) -> pd.Timestamp:
    """ECOS 응답의 TIME 필드를 pandas Timestamp로 변환.

    응답 형식:
        D -> 'YYYYMMDD'
        M -> 'YYYYMM'
        Q -> 'YYYYQ#'
        A -> 'YYYY'
    """
    if cycle == CYCLE_DAILY:
        return pd.Timestamp(time_str[:4] + "-" + time_str[4:6] + "-" + time_str[6:8])
    if cycle == CYCLE_MONTHLY:
        # 월 시리즈는 그 달 1일로 정규화
        return pd.Timestamp(time_str[:4] + "-" + time_str[4:6] + "-01")
    if cycle == CYCLE_QUARTERLY:
        m = re.match(r"(\d{4})Q(\d)", time_str)
        if not m:
            raise ValueError(f"Bad quarterly time: {time_str}")
        year, q = int(m.group(1)), int(m.group(2))
        month = (q - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)
    if cycle == CYCLE_ANNUAL:
        return pd.Timestamp(year=int(time_str), month=1, day=1)
    raise ValueError(f"Unsupported cycle: {cycle}")


def fetch_ecos(
    stat_code: str,
    item_code: Optional[str],
    start_date: str,
    end_date: str,
    cycle: str = CYCLE_DAILY,
    use_cache: bool = True,
    max_rows: int = 100000,
) -> pd.Series:
    """ECOS에서 단일 시리즈를 가져옴.

    Args:
        stat_code: 통계표 코드 (예: '722Y001' = 한국은행 기준금리).
        item_code: 통계항목 코드. 통계표가 단일 시리즈만 갖는 경우 None 가능.
                   다중 항목 통계표(예: 환율표)는 반드시 지정.
        start_date: 'YYYY-MM-DD' 시작일.
        end_date: 'YYYY-MM-DD' 종료일.
        cycle: 'D' | 'M' | 'Q' | 'A'. 기본 일별.
        use_cache: True면 캐시 우선 사용.
        max_rows: 한 번에 받아올 최대 행수 (ECOS 최대 100,000).

    Returns:
        KST 일별 인덱스의 pd.Series. 이름은 '{stat_code}_{item_code}'.

    Raises:
        MissingAPIKeyError: ECOS_API_KEY 미설정.
        InvalidSeriesError: 시리즈 코드가 잘못되거나 응답이 빈 경우.
        NetworkError: 네트워크/타임아웃.
    """
    cache_key = f"{stat_code}_{item_code or 'NA'}_{cycle}"
    _, meta_path = _cache_paths("ecos", cache_key)

    # 1) 캐시 확인
    if use_cache and is_cache_valid(meta_path, start_date, end_date):
        cached = read_cache("ecos", cache_key)
        if cached is not None:
            logger.debug("ECOS cache hit: %s", cache_key)
            return cached.loc[start_date:end_date]

    # 2) API 키
    api_key = require_env("ECOS_API_KEY")

    # 3) URL 구성
    from_d = _format_date_for_cycle(start_date, cycle)
    to_d = _format_date_for_cycle(end_date, cycle)

    parts = [
        ECOS_BASE,
        api_key,
        "json",
        "kr",
        "1",                  # start row
        str(max_rows),        # end row
        stat_code,
        cycle,
        from_d,
        to_d,
    ]
    if item_code:
        parts.append(item_code)
    url = "/".join(parts)

    # 4) 호출
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise NetworkError(f"ECOS API 호출 실패 (stat_code={stat_code}): {e}") from e

    if resp.status_code != 200:
        raise NetworkError(
            f"ECOS HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        payload = resp.json()
    except ValueError as e:
        raise NetworkError(f"ECOS 응답이 JSON이 아님: {resp.text[:200]}") from e

    # 5) ECOS 에러 응답 처리
    # 정상: {"StatisticSearch": {"list_total_count": N, "row": [...]}}
    # 오류: {"RESULT": {"CODE": "INFO-200", "MESSAGE": "..."}}
    if "RESULT" in payload:
        result = payload["RESULT"]
        code = result.get("CODE", "")
        msg = result.get("MESSAGE", "")
        # INFO-200 = 해당 자료 없음
        if code.startswith("INFO-200"):
            raise InvalidSeriesError(
                f"ECOS 시리즈 ({stat_code}, {item_code})에 데이터가 없습니다: {msg}"
            )
        # INFO-100 = 인증 실패
        if code.startswith("INFO-100"):
            raise InvalidSeriesError(f"ECOS 인증 실패: {msg}")
        raise InvalidSeriesError(f"ECOS 오류 [{code}]: {msg}")

    if "StatisticSearch" not in payload:
        raise InvalidSeriesError(
            f"ECOS 응답 형식이 예상과 다릅니다: {list(payload.keys())}"
        )

    rows = payload["StatisticSearch"].get("row", [])
    if not rows:
        raise InvalidSeriesError(
            f"ECOS ({stat_code}, {item_code}) 응답이 비어있습니다."
        )

    # 6) DataFrame 구성
    records = []
    for r in rows:
        try:
            ts = _parse_ecos_time(r["TIME"], cycle)
            val = float(r["DATA_VALUE"])
            records.append((ts, val))
        except (KeyError, ValueError) as e:
            logger.debug("ECOS row skip: %s (%s)", r, e)

    if not records:
        raise InvalidSeriesError(f"ECOS ({stat_code}) 응답에서 유효한 데이터를 추출할 수 없습니다.")

    df = pd.DataFrame(records, columns=["date", "value"]).set_index("date")
    series = df["value"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = cache_key
    series = to_kst_index(series)

    if use_cache:
        write_cache("ecos", cache_key, series, start_date, end_date)

    logger.info("Fetched ECOS %s: %d rows (%s ~ %s)",
                cache_key, len(series), start_date, end_date)
    return series


def fetch_kr_base_rate(start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
    """한국 기준금리 편의 함수 (일별).

    통계표 722Y001 - 한국은행 기준금리. 단일 항목 시리즈로 item_code 불필요.
    """
    return fetch_ecos(
        stat_code="722Y001",
        item_code="0101000",   # 기준금리 항목 코드 (일반적 표기)
        start_date=start_date,
        end_date=end_date,
        cycle=CYCLE_DAILY,
        use_cache=use_cache,
    )
