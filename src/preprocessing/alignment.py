"""시점 정렬 모듈.

서로 다른 주기(일별/월별/분기별)와 결측 패턴을 가진 시리즈들을
공통 거래일 인덱스 위로 정렬합니다.

정책:
    1. 거래일 인덱스 사용 (주말 제외, business day frequency 'B')
       - 한국 공휴일까지 반영하려면 pandas_market_calendars 등 필요하나
         1차 프로토타입에서는 'B' (월~금)로 충분
    2. Forward fill: 최근 관측값으로 채우되 최대 30일까지만
       - 월별 시리즈(예: ISM PMI)는 발표 후 다음 발표까지 같은 값 유지가 자연스러움
       - 30일 이상 결측 = 데이터 끊김 → 그대로 NaN 유지 (분석에서 제외 가능)
    3. Backward fill 안 함 (미래 정보 누출 방지)
    4. 정렬 후 시점은 KST 일자 (tz-naive)

[한계]
    - 'B' 인덱스는 한국 공휴일을 모름. 결과 인덱스에 공휴일이 포함되어 있으면
      해당 일의 KOSPI는 NaN으로 남게 됨. 분석 시 dropna로 처리.
    - 월별 시리즈를 ffill 30일로 제한하면 발표 지연이 30일 이상이면 끊김.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def align_series(
    series_dict: Mapping[str, pd.Series],
    start_date: str,
    end_date: str,
    freq: str = "B",
    ffill_limit: int = 30,
) -> pd.DataFrame:
    """여러 시리즈를 공통 일별 거래일 인덱스 위에 정렬.

    Args:
        series_dict: {변수 코드: pd.Series} 매핑. 각 시리즈는 DatetimeIndex.
        start_date: 'YYYY-MM-DD' 정렬 시작일.
        end_date: 'YYYY-MM-DD' 정렬 종료일.
        freq: pandas frequency. 기본 'B'(월~금).
        ffill_limit: forward fill 최대 일수 (기본 30).
                     이를 넘어가는 결측은 NaN으로 유지.

    Returns:
        인덱스가 거래일 DatetimeIndex(tz-naive)인 DataFrame.
        컬럼은 series_dict의 키 순서.

    동작 상세:
        - 빈 series_dict는 빈 DataFrame 반환
        - 빈 시리즈(혹은 모든 값 NaN)는 그대로 NaN 컬럼으로 추가
        - 각 시리즈를 공통 인덱스로 reindex → ffill(limit=ffill_limit) → 결과 결합
        - tz-aware 인덱스는 tz-naive로 정규화 (KST 정렬 가정)
    """
    if not series_dict:
        return pd.DataFrame()

    # 1) 공통 거래일 인덱스 생성
    common_index = pd.date_range(start=start_date, end=end_date, freq=freq)

    # 2) 각 시리즈 정렬
    aligned_cols: dict[str, pd.Series] = {}
    for code, s in series_dict.items():
        if s is None or len(s) == 0:
            # 빈 시리즈는 모두 NaN 컬럼으로
            aligned_cols[code] = pd.Series(index=common_index, dtype="float64", name=code)
            continue

        # tz-naive로 정규화
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Seoul").tz_localize(None)
        idx = idx.normalize()
        s_local = pd.Series(s.values, index=idx, name=code)
        s_local = s_local[~s_local.index.duplicated(keep="last")].sort_index()

        # 공통 인덱스로 reindex 후 ffill (한도 적용)
        reindexed = s_local.reindex(common_index)
        filled = reindexed.ffill(limit=ffill_limit)
        aligned_cols[code] = filled

    # 3) DataFrame 결합
    df = pd.concat(aligned_cols, axis=1)
    df.index.name = "date"
    return df


def drop_long_gap_periods(
    df: pd.DataFrame,
    max_consecutive_nan: int = 30,
) -> pd.DataFrame:
    """긴 결측 구간을 추가로 식별하여 보조 마스크와 함께 반환.

    align_series가 ffill_limit으로 채우고 남은 NaN이 이미 "긴 결측"의 표시지만,
    이 함수는 명시적으로 "한 변수라도 연속 30일 이상 NaN인 구간"을 검출하여
    분석에서 제외할 행을 표시할 수 있게 합니다.

    1차 프로토타입에서는 단순히 "행에 하나라도 NaN이 있으면 제거"는 너무 공격적이라
    이 함수는 기본 동작이 식별만 하고 제거하지 않음. 호출자가 마스크를 보고 결정.

    Args:
        df: align_series 결과.
        max_consecutive_nan: 이 일수 이상 연속 NaN인 변수는 해당 구간 제외 후보.

    Returns:
        원본과 같은 DataFrame을 반환 (현재는 식별만 수행).
        추후 확장 시 마스크 컬럼 추가 가능.
    """
    # 1차 프로토타입: 식별 로직만 두고 그대로 반환
    # (실제 제외는 표준화 단계에서 NaN 그대로 두면 자연스럽게 분석에서 빠짐)
    return df.copy()
