"""pykrx (한국거래소) 로더.

pykrx 라이브러리로 KOSPI/KOSDAQ 지수 및 외국인 매매 데이터를 수집합니다.

이 도구에서 사용:
    KOSPI 일별 종가  : k-NN 유사 시점 이후 성과 측정 타겟
    KOSDAQ 일별 종가 : 보조 타겟
    외국인 순매수 (옵셔널, 1차 비활성)

[한계]
    - pykrx는 KRX 페이지를 파싱하므로 사이트 변경 시 깨질 수 있음
    - 첫 호출이 느림 (5~10초). 캐시 권장.
    - 거래일만 데이터 존재 (주말/공휴일 비어있음)
"""

from __future__ import annotations

import pandas as pd

from ._common import (
    DataLoaderError,
    InvalidSeriesError,
    NetworkError,
    is_cache_valid,
    logger,
    read_cache,
    to_kst_index,
    write_cache,
    _cache_paths,
)

# pykrx 지수 코드
KOSPI_CODE = "1001"
KOSDAQ_CODE = "2001"


def _yyyymmdd(date_str: str) -> str:
    """'YYYY-MM-DD' -> 'YYYYMMDD' (pykrx 입력 형식)."""
    return pd.to_datetime(date_str).strftime("%Y%m%d")


def fetch_krx_index(
    index_code: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.Series:
    """KRX 지수 종가 시리즈를 가져옴.

    Args:
        index_code: '1001' (KOSPI) | '2001' (KOSDAQ) 등.
        start_date: 'YYYY-MM-DD'.
        end_date: 'YYYY-MM-DD'.
        use_cache: 캐시 사용 여부.

    Returns:
        거래일 인덱스의 pd.Series. 이름은 'KOSPI'/'KOSDAQ' 등.

    Raises:
        InvalidSeriesError: 지수 코드가 잘못되거나 응답이 빈 경우.
        NetworkError: pykrx 호출 실패.
    """
    cache_key = f"index_{index_code}"
    _, meta_path = _cache_paths("pykrx", cache_key)

    # 1) 캐시 확인
    if use_cache and is_cache_valid(meta_path, start_date, end_date):
        cached = read_cache("pykrx", cache_key)
        if cached is not None:
            logger.debug("pykrx cache hit: %s", cache_key)
            return cached.loc[start_date:end_date]

    # 2) pykrx 임포트 (호출 시점)
    try:
        from pykrx import stock
    except ImportError as e:
        raise DataLoaderError("pykrx가 설치되지 않았습니다. pip install pykrx") from e

    # 3) 호출
    try:
        df = stock.get_index_ohlcv_by_date(
            fromdate=_yyyymmdd(start_date),
            todate=_yyyymmdd(end_date),
            ticker=index_code,
        )
    except Exception as e:  # noqa: BLE001
        raise NetworkError(f"pykrx 호출 실패 (index_code={index_code}): {e}") from e

    if df is None or df.empty:
        raise InvalidSeriesError(
            f"pykrx 지수 '{index_code}' 응답이 비어있습니다. "
            f"코드와 기간을 확인하세요."
        )

    # 4) 종가 컬럼 추출 (pykrx는 '종가' 컬럼 사용)
    if "종가" not in df.columns:
        raise InvalidSeriesError(
            f"pykrx 응답에 '종가' 컬럼이 없습니다. 컬럼: {df.columns.tolist()}"
        )

    series = df["종가"].copy()
    name_map = {KOSPI_CODE: "KOSPI", KOSDAQ_CODE: "KOSDAQ"}
    series.name = name_map.get(index_code, f"KRX_{index_code}")
    series = to_kst_index(series).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if use_cache:
        write_cache("pykrx", cache_key, series, start_date, end_date)

    logger.info("Fetched pykrx %s: %d rows (%s ~ %s)",
                series.name, len(series), start_date, end_date)
    return series


def fetch_kospi(start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
    """KOSPI 일별 종가 편의 함수."""
    return fetch_krx_index(KOSPI_CODE, start_date, end_date, use_cache=use_cache)


def fetch_kosdaq(start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
    """KOSDAQ 일별 종가 편의 함수."""
    return fetch_krx_index(KOSDAQ_CODE, start_date, end_date, use_cache=use_cache)
