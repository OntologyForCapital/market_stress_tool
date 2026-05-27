"""변수 source 필드를 보고 적절한 로더로 라우팅하는 dispatcher.

이 모듈은 `Variable` dataclass (src.config) 를 받아서 해당하는 로더로
실제 데이터 수집을 위임한다.

설계 원칙:
    - 단일 변수 실패가 전체 파이프라인을 중단시키지 않음
      (실패 변수는 로그 + 결과 dict에서 제외)
    - 빈 시리즈는 경고 로그 + 결과에 포함 (이후 alignment 단계에서 처리)
    - source별 라우팅:
        fred      → fred_loader.fetch_fred
        yfinance  → yfinance_loader.fetch_yfinance_with_fallback
        ecos      → ecos_loader.fetch_ecos (item_code 사용)
        krx       → legacy alias. KOSPI/KOSDAQ도 yfinance로 우회
        computed  → 하위 components를 재귀 수집 후 산식 적용
        pykrx     → NotImplementedError (1차 비활성)

사용 예:
    from src.config import load_variables, load_target_variables
    from src.data_collection.dispatcher import fetch_all_variables

    all_vars = load_variables() + load_target_variables()
    series_dict = fetch_all_variables(all_vars, "2020-01-01", "2025-12-31")
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.config import ComponentRef, Variable
from src.data_collection._common import (
    DataLoaderError,
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
)
from src.data_collection import (
    fred_loader,
    yfinance_loader,
    ecos_loader,
)

logger = logging.getLogger(__name__)

# 로더 호출 시 처리하는 "예상 가능한" 예외들 (실패 격리 대상)
LOADER_EXCEPTIONS = (
    DataLoaderError,
    InvalidSeriesError,
    MissingAPIKeyError,
    NetworkError,
    NotImplementedError,
    ValueError,
)

YFINANCE_INDEX_TICKERS: dict[str, str] = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
}


# =============================================================================
# computed 변수: KR_US_RATE_DIFF 같은 사용자 정의 산식
# =============================================================================
def _fetch_component(
    component: ComponentRef,
    start_date: str,
    end_date: str,
    use_cache: bool,
) -> pd.Series:
    """computed 변수의 한 컴포넌트를 source 기반으로 수집.

    components 항목은 variables.yaml의 단순화된 dict 형태로,
    Variable과는 다른 작은 스키마를 가진다.
    """
    src = component.source
    if src == "fred":
        if not component.series_id:
            raise InvalidSeriesError(
                f"computed component '{component.name}'(fred)에 series_id 누락"
            )
        return fred_loader.fetch_fred(
            component.series_id, start_date, end_date, use_cache=use_cache,
        )

    if src == "ecos":
        if not component.stat_code:
            raise InvalidSeriesError(
                f"computed component '{component.name}'(ecos)에 stat_code 누락"
            )
        # 한국 기준금리(722Y001)는 단일 항목 통계표지만 item_code='0101000'이 필요
        # → 명시된 item_code가 없으면 fetch_kr_base_rate 편의 함수 사용
        if component.stat_code == "722Y001" and not component.item_code:
            return ecos_loader.fetch_kr_base_rate(
                start_date, end_date, use_cache=use_cache,
            )
        return ecos_loader.fetch_ecos(
            stat_code=component.stat_code,
            item_code=component.item_code,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

    if src == "yfinance":
        if not component.series_id:
            raise InvalidSeriesError(
                f"computed component '{component.name}'(yfinance)에 series_id 누락"
            )
        return yfinance_loader.fetch_yfinance(
            component.series_id, start_date, end_date, use_cache=use_cache,
        )

    raise InvalidSeriesError(
        f"computed component source '{src}' (component={component.name}) 미지원"
    )


def _fetch_computed_variable(
    variable: Variable,
    start_date: str,
    end_date: str,
    use_cache: bool,
) -> pd.Series:
    """computed 변수의 산식 적용 (현재는 KR_US_RATE_DIFF 패턴만 지원).

    KR_US_RATE_DIFF 산식:
        kr_rate - us_rate (한국 기준금리 - 미국 FFR)

    향후 다른 computed 변수가 추가되면 code 매칭으로 분기 추가.
    """
    if not variable.components:
        raise InvalidSeriesError(
            f"computed 변수 {variable.code}에 components가 정의되지 않음"
        )

    # 모든 컴포넌트 수집
    comp_series: dict[str, pd.Series] = {}
    for name, comp in variable.components.items():
        s = _fetch_component(comp, start_date, end_date, use_cache=use_cache)
        comp_series[name] = s

    # 산식 적용 (변수별)
    if variable.code == "KR_US_RATE_DIFF":
        if "kr_rate" not in comp_series or "us_rate" not in comp_series:
            raise InvalidSeriesError(
                "KR_US_RATE_DIFF는 kr_rate와 us_rate 컴포넌트가 모두 필요합니다."
            )
        # 두 시리즈를 outer join 후 forward fill (저빈도 → 일별 연결)
        df = pd.concat(
            [comp_series["kr_rate"], comp_series["us_rate"]],
            axis=1,
            keys=["kr_rate", "us_rate"],
        ).sort_index()
        df = df.ffill()
        diff = df["kr_rate"] - df["us_rate"]
        diff = diff.dropna()
        diff.name = variable.code
        return diff

    raise NotImplementedError(
        f"computed 변수 {variable.code}의 산식이 dispatcher에 구현되지 않음. "
        f"_fetch_computed_variable()에 분기 추가 필요."
    )


# =============================================================================
# source별 라우팅
# =============================================================================
def fetch_variable(
    variable: Variable,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.Series:
    """단일 변수 수집.

    Args:
        variable: src.config.Variable dataclass 인스턴스.
        start_date, end_date: 'YYYY-MM-DD'.
        use_cache: 캐시 사용 여부.

    Returns:
        pd.Series. 이름은 variable.code로 정규화됨.

    Raises:
        LOADER_EXCEPTIONS: 각 로더에서 발생한 예외를 그대로 전파.
            (fetch_all_variables에서 격리됨)
    """
    src = variable.source

    if src == "fred":
        if not variable.series_id:
            raise InvalidSeriesError(f"{variable.code}: fred series_id 누락")
        s = fred_loader.fetch_fred(
            variable.series_id, start_date, end_date, use_cache=use_cache,
        )

    elif src == "yfinance":
        if not variable.series_id:
            raise InvalidSeriesError(f"{variable.code}: yfinance series_id 누락")
        s = yfinance_loader.fetch_yfinance_with_fallback(
            primary_ticker=variable.series_id,
            fallback_ticker=variable.fallback_series_id,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

    elif src == "ecos":
        if not variable.series_id:
            raise InvalidSeriesError(f"{variable.code}: ecos series_id(stat_code) 누락")
        # frequency → ECOS cycle 변환
        freq_to_cycle = {"daily": "D", "monthly": "M", "quarterly": "Q", "annual": "A"}
        cycle = freq_to_cycle.get(variable.frequency, "D")
        try:
            s = ecos_loader.fetch_ecos(
                stat_code=variable.series_id,
                item_code=variable.item_code,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
                cycle=cycle,
            )
        except TypeError as e:
            # 구버전 테스트 더블/로더는 cycle 인자를 받지 않을 수 있다.
            # 실제 로더는 cycle을 지원하므로 운영 경로는 위 호출을 사용한다.
            if "cycle" not in str(e):
                raise
            s = ecos_loader.fetch_ecos(
                stat_code=variable.series_id,
                item_code=variable.item_code,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )

    elif src == "krx":
        # v20: KRX Data Marketplace API 제약을 피하기 위해 KOSPI/KOSDAQ legacy
        # source도 yfinance 지수 티커로 우회한다.
        ticker = YFINANCE_INDEX_TICKERS.get(variable.code)
        if ticker is None:
            raise InvalidSeriesError(
                f"legacy krx source는 KOSPI/KOSDAQ yfinance 우회만 지원. code={variable.code}"
            )
        s = yfinance_loader.fetch_yfinance_with_fallback(
            primary_ticker=ticker,
            fallback_ticker=None,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

    elif src == "computed":
        s = _fetch_computed_variable(variable, start_date, end_date, use_cache=use_cache)

    elif src == "pykrx":
        raise NotImplementedError(
            f"pykrx source는 현재 비활성화되어 있습니다. ({variable.code}) "
            f"variables.yaml에서 source: yfinance로 변경하거나 enabled: false 처리하세요."
        )

    else:
        raise InvalidSeriesError(
            f"{variable.code}: 알 수 없는 source '{src}'. "
            f"지원: fred | yfinance | ecos | computed | krx(legacy alias)"
        )

    # 시리즈 이름을 variable.code로 정규화 (각 로더가 다른 이름을 쓰므로)
    s = s.rename(variable.code)
    return s


# =============================================================================
# 일괄 수집
# =============================================================================
def fetch_all_variables(
    variables: Iterable[Variable],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    include_disabled: bool = False,
) -> dict[str, pd.Series]:
    """여러 변수를 한 번에 수집.

    Args:
        variables: Variable 이터러블 (load_variables() + load_target_variables() 등 결합 가능).
        start_date, end_date: 'YYYY-MM-DD'.
        use_cache: 캐시 사용 여부.
        include_disabled: True면 enabled=False 변수도 시도.
            기본 False — variables 섹션의 비활성 변수는 자동 스킵.
            target_variables는 enabled 키가 없어 항상 True 취급.

    Returns:
        {variable.code: pd.Series} dict. 실패한 변수는 제외됨.

    동작:
        - 각 변수 수집을 독립 try/except로 감쌈
        - 성공 → dict에 추가 (빈 시리즈여도 포함, 경고 로그)
        - 실패 → 로그만 남기고 dict에서 제외
        - 한 변수 실패가 다른 변수 수집을 막지 않음
    """
    result: dict[str, pd.Series] = {}
    skipped_disabled: list[str] = []
    failed: list[tuple[str, str]] = []

    var_list = list(variables)
    logger.info(
        "fetch_all_variables 시작: %d개 변수, %s ~ %s",
        len(var_list), start_date, end_date,
    )

    for variable in var_list:
        if not variable.enabled and not include_disabled:
            skipped_disabled.append(variable.code)
            logger.debug("[%s] enabled=False → 스킵", variable.code)
            continue

        try:
            s = fetch_variable(variable, start_date, end_date, use_cache=use_cache)
        except LOADER_EXCEPTIONS as e:
            failed.append((variable.code, f"{type(e).__name__}: {e}"))
            logger.error(
                "[%s] 수집 실패 (%s): %s",
                variable.code, type(e).__name__, e,
            )
            continue
        except Exception as e:  # noqa: BLE001 - 예상치 못한 예외도 격리
            failed.append((variable.code, f"{type(e).__name__}: {e}"))
            logger.exception("[%s] 예상치 못한 예외", variable.code)
            continue

        # 빈 시리즈는 경고 + 포함 (alignment 단계에서 처리)
        if s.empty:
            logger.warning(
                "[%s] 수집 결과가 빈 시리즈입니다. (source=%s)",
                variable.code, variable.source,
            )

        result[variable.code] = s
        logger.info(
            "[%s] 수집 완료: %d개 관측치 (source=%s)",
            variable.code, len(s), variable.source,
        )

    # 요약 로그
    logger.info(
        "fetch_all_variables 완료: 성공=%d, 실패=%d, 비활성스킵=%d",
        len(result), len(failed), len(skipped_disabled),
    )
    if failed:
        logger.warning(
            "실패한 변수: %s",
            ", ".join(f"{code}({err.split(':')[0]})" for code, err in failed),
        )

    return result
