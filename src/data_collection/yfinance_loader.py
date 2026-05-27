"""Yahoo Finance 로더.

yfinance 라이브러리로 시장 가격 시계열을 수집합니다.
이 도구에서 사용하는 티커:
    ^BDIY      : Baltic Dry Index (1차 시도, 안 되면 BDRY ETF로 폴백)
    BDRY       : Baltic Dry Index ETF (폴백)
    DX-Y.NYB   : 달러 인덱스 (DXY)
    ^KS11      : KOSPI Composite
    ^KQ11      : KOSDAQ Composite
    ^MOVE      : 미국채 변동성 (1차 비활성)

[한계]
    - yfinance는 비공식 API이므로 안정성이 보장되지 않음
    - 일부 티커(^BDIY, ^MOVE)는 시점에 따라 조회 불가
    - 무료 사용 시 rate limit 가능
"""

from __future__ import annotations

import pandas as pd

from ._common import (
    InvalidSeriesError,
    NetworkError,
    DataLoaderError,
    is_cache_valid,
    logger,
    read_cache,
    to_kst_index,
    write_cache,
    _cache_paths,
)


def fetch_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    field: str = "Close",
) -> pd.Series:
    """Yahoo Finance에서 단일 티커의 가격 시계열을 가져옴.

    Args:
        ticker: yfinance 티커 (예: 'DX-Y.NYB', '^BDIY').
        start_date: 'YYYY-MM-DD' 형식 시작일.
        end_date: 'YYYY-MM-DD' 형식 종료일.
        use_cache: True면 캐시 우선 사용.
        field: 'Close' (기본) 또는 'Adj Close', 'Open', 'High', 'Low', 'Volume'.

    Returns:
        거래일 인덱스(KST 기준 일별)의 pd.Series. 이름은 ticker.

    Raises:
        InvalidSeriesError: 티커가 잘못되었거나 데이터가 빈 경우.
        NetworkError: yfinance 호출 실패.
    """
    # 1) 캐시 확인
    cache_key = f"{ticker}_{field}"
    _, meta_path = _cache_paths("yfinance", cache_key)
    if use_cache and is_cache_valid(meta_path, start_date, end_date):
        cached = read_cache("yfinance", cache_key)
        if cached is not None:
            logger.debug("yfinance cache hit: %s", ticker)
            return cached.loc[start_date:end_date]

    # 2) yfinance 임포트 (호출 시점에)
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataLoaderError(
            "yfinance가 설치되지 않았습니다. pip install yfinance"
        ) from e

    # 3) 다운로드
    try:
        # auto_adjust=False로 원본 Close 유지 (Adj Close 별도 컬럼)
        # progress=False로 stdout 노이즈 제거
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:  # noqa: BLE001
        raise NetworkError(f"yfinance 호출 실패 (ticker={ticker}): {e}") from e

    if df is None or df.empty:
        raise InvalidSeriesError(
            f"yfinance 티커 '{ticker}' 응답이 비어있습니다. "
            f"티커가 유효한지, 해당 기간에 데이터가 있는지 확인하세요."
        )

    # 4) 멀티인덱스 컬럼 처리 (yfinance >= 0.2.x는 MultiIndex 반환 가능)
    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', ticker) 구조에서 field만 추출
        if field not in df.columns.get_level_values(0):
            raise InvalidSeriesError(
                f"필드 '{field}'가 응답에 없습니다. 사용 가능: {df.columns.get_level_values(0).unique().tolist()}"
            )
        series = df[field].squeeze()
    else:
        if field not in df.columns:
            raise InvalidSeriesError(
                f"필드 '{field}'가 응답에 없습니다. 사용 가능: {df.columns.tolist()}"
            )
        series = df[field]

    series.name = ticker
    series = to_kst_index(series)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()

    if series.empty:
        raise InvalidSeriesError(
            f"yfinance 티커 '{ticker}' {field} 시리즈가 모두 NaN입니다."
        )

    # 5) 캐시 저장
    if use_cache:
        write_cache("yfinance", cache_key, series, start_date, end_date)

    logger.info("Fetched yfinance %s.%s: %d rows (%s ~ %s)",
                ticker, field, len(series), start_date, end_date)
    return series


def fetch_yfinance_with_fallback(
    primary_ticker: str,
    fallback_ticker: str | None,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.Series:
    """주요 티커가 실패하면 폴백 티커로 재시도.

    BDI처럼 ^BDIY가 자주 실패하는 시리즈를 위해 BDRY ETF로 자동 폴백.

    Args:
        primary_ticker: 1차 티커.
        fallback_ticker: 폴백 티커. None이면 폴백 없음.
        start_date, end_date: 기간.
        use_cache: 캐시 사용 여부.

    Returns:
        성공한 시리즈. 시리즈 이름은 실제 사용된 티커로 설정.

    Raises:
        모든 시도가 실패하면 마지막 예외를 그대로 raise.
    """
    try:
        return fetch_yfinance(primary_ticker, start_date, end_date, use_cache=use_cache)
    except (InvalidSeriesError, NetworkError) as e:
        if not fallback_ticker:
            raise
        logger.warning(
            "Primary ticker %s failed (%s). Trying fallback %s.",
            primary_ticker, type(e).__name__, fallback_ticker,
        )
        return fetch_yfinance(fallback_ticker, start_date, end_date, use_cache=use_cache)
