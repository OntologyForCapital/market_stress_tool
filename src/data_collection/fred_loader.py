"""FRED (Federal Reserve Economic Data) 로더.

미국 연준이 제공하는 거시 경제 시계열을 fredapi 라이브러리로 수집합니다.
이 도구에서 사용하는 FRED 시리즈 (총 10개):
    NAPM           : ISM 제조업 PMI (월별)
    DFII10         : 미국 10년 TIPS 실질금리 (일별)
    T10YIE         : 미국 10년 BEI (일별)
    DFEDTARU       : Fed Funds 상단 목표 금리 (일별)
    DFF            : Fed Funds Effective Rate (일별)
    VIXCLS         : VIX 종가 (일별)
    BAMLH0A0HYM2   : 미국 HY 스프레드 OAS (일별)
    DCOILBRENTEU   : 브렌트 유가 (일별)
    DEXKOUS        : 원/달러 환율 (일별)
    DEXJPUS        : 엔/달러 환율 (일별)

[한계]
    - FRED는 무료지만 일부 시리즈는 갱신 지연 있음 (특히 월별 시리즈)
    - 시리즈 코드가 폐기되거나 변경될 수 있음 → 실패 시 코드 점검 필요
"""

from __future__ import annotations

import pandas as pd

from ._common import (
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
    is_cache_valid,
    logger,
    read_cache,
    require_env,
    to_kst_index,
    write_cache,
    _cache_paths,
)


def fetch_fred(
    series_id: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.Series:
    """FRED에서 단일 시리즈를 가져옴.

    Args:
        series_id: FRED 시리즈 코드 (예: 'VIXCLS', 'DFII10').
        start_date: 'YYYY-MM-DD' 형식 시작일.
        end_date: 'YYYY-MM-DD' 형식 종료일.
        use_cache: True면 캐시 우선 사용.

    Returns:
        인덱스가 KST 기준 일별 DatetimeIndex인 pd.Series.
        시리즈 이름은 series_id로 설정됨.

    Raises:
        MissingAPIKeyError: FRED_API_KEY 미설정.
        InvalidSeriesError: 시리즈 코드가 잘못되었거나 데이터가 빈 경우.
        NetworkError: 네트워크/타임아웃/서버 오류.

    캐싱:
        data/raw/fred_{series_id}.parquet + .meta.json
        TTL은 CACHE_TTL_HOURS 환경변수 (기본 24시간).
    """
    # 1) 캐시 확인 - 메타가 유효하면 캐시 데이터 사용
    _, meta_path = _cache_paths("fred", series_id)
    if use_cache and is_cache_valid(meta_path, start_date, end_date):
        cached = read_cache("fred", series_id)
        if cached is not None:
            logger.debug("FRED cache hit: %s", series_id)
            # 요청 범위로 자르기
            return cached.loc[start_date:end_date]

    # 2) API 키 확인
    api_key = require_env("FRED_API_KEY")

    # 3) fredapi 임포트 - 모듈 임포트 시점이 아닌 호출 시점에 (테스트 모킹 용이)
    try:
        from fredapi import Fred
    except ImportError as e:
        raise DataLoaderError(  # noqa: F821
            "fredapi가 설치되지 않았습니다. pip install fredapi"
        ) from e

    fred = Fred(api_key=api_key)

    # 4) 실제 호출 - 오류 종류 구분
    try:
        raw = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except ValueError as e:
        # fredapi는 잘못된 series_id에 ValueError("Bad Request") 발생
        raise InvalidSeriesError(
            f"FRED 시리즈 코드 '{series_id}'를 찾을 수 없습니다: {e}"
        ) from e
    except Exception as e:  # noqa: BLE001
        # 그 외는 네트워크 오류로 간주 (requests.ConnectionError, Timeout 등)
        raise NetworkError(f"FRED API 호출 실패 (series_id={series_id}): {e}") from e

    if raw is None or len(raw) == 0:
        raise InvalidSeriesError(
            f"FRED 시리즈 '{series_id}' 응답이 비어있습니다. 코드 또는 기간을 확인하세요."
        )

    # 5) Series 정리
    series = pd.Series(raw, name=series_id)
    series = to_kst_index(series)
    series = series.sort_index()
    # 중복 인덱스 제거 (혹시 모를 데이터 정합성 문제)
    series = series[~series.index.duplicated(keep="last")]

    # 6) 캐시 저장
    if use_cache:
        write_cache("fred", series_id, series, start_date, end_date)

    logger.info("Fetched FRED %s: %d rows (%s ~ %s)",
                series_id, len(series), start_date, end_date)
    return series


# DataLoaderError를 forward-import (위 try/except 블록에서 사용)
from ._common import DataLoaderError  # noqa: E402
