"""데이터 수집 공통 유틸리티.

이 모듈은 모든 로더가 공유하는 기능을 제공합니다:
    1. 디스크 캐시 (parquet 기반, last_updated 메타데이터 포함)
    2. 캐시 TTL 검사 (기본 24시간)
    3. 공통 예외 클래스 (API 키 누락, 시리즈 코드 오류, 네트워크 오류 구분)
    4. KST 타임존 정규화

캐시 정책:
    - 같은 (source, series_id) 요청 시 디스크 캐시(.parquet) 재사용
    - 캐시 파일 옆에 .meta.json 파일로 last_updated, start, end, source 저장
    - TTL이 지나면 캐시를 무시하고 다시 받아옴
    - 캐시 디렉토리: data/raw/{source}_{series_id}.parquet

부호 규칙:
    - 이 모듈은 raw 데이터만 다룹니다. 위험 방향 부호 반전은 preprocessing/standardize.py 담당.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------------------------------------------------------
# 경로 / 환경
# -----------------------------------------------------------------------------
# 프로젝트 루트: src/data_collection/_common.py 기준 두 단계 상위
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# KST = UTC+9
KST = timezone(timedelta(hours=9))

# 로거 (앱 전체에서 공유)
logger = logging.getLogger("market_stress_tool")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


# -----------------------------------------------------------------------------
# 공통 예외
# -----------------------------------------------------------------------------
class DataLoaderError(Exception):
    """데이터 로더의 모든 예외의 베이스 클래스."""


class MissingAPIKeyError(DataLoaderError):
    """API 키가 .env에 없거나 비어있을 때 발생."""


class InvalidSeriesError(DataLoaderError):
    """시리즈 코드가 잘못되어 데이터 소스가 4xx 또는 빈 응답을 반환할 때 발생."""


class NetworkError(DataLoaderError):
    """네트워크 연결/타임아웃/5xx 등 일시적 오류."""


# -----------------------------------------------------------------------------
# 캐시 키/경로 헬퍼
# -----------------------------------------------------------------------------
def _cache_paths(source: str, series_id: str) -> tuple[Path, Path]:
    """캐시 데이터 파일과 메타데이터 파일 경로를 반환.

    Args:
        source: 데이터 출처 이름 (fred, yfinance, ecos, pykrx 등)
        series_id: 시리즈 식별자. 슬래시/특수문자는 언더스코어로 변환.

    Returns:
        (data_path, meta_path) 튜플.
            data_path: data/raw/{source}_{safe_id}.parquet
            meta_path: data/raw/{source}_{safe_id}.meta.json
    """
    # 파일 시스템에서 안전한 문자만 남김
    safe = series_id.replace("/", "_").replace("\\", "_").replace(":", "_").replace("^", "")
    safe = safe.replace(".", "_")
    base = RAW_CACHE_DIR / f"{source}_{safe}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


def _cache_ttl_hours() -> float:
    """환경변수 CACHE_TTL_HOURS를 읽어 캐시 유효 시간을 반환 (기본 24시간)."""
    try:
        return float(os.environ.get("CACHE_TTL_HOURS", "24"))
    except ValueError:
        return 24.0


def is_cache_valid(meta_path: Path, start: str, end: str) -> bool:
    """캐시 메타데이터를 보고 캐시가 아직 유효한지 판단.

    유효 조건:
        1. meta_path 파일이 존재함
        2. 메타의 last_updated가 현재로부터 CACHE_TTL_HOURS 이내
        3. 메타의 start <= 요청 start, 메타의 end >= 요청 end
           (캐시가 요청 범위를 포함하면 재사용 가능)

    Returns:
        True면 캐시 사용 가능, False면 다시 받아와야 함.
    """
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    # 1) TTL 검사
    last_updated_str = meta.get("last_updated")
    if not last_updated_str:
        return False
    try:
        last_updated = datetime.fromisoformat(last_updated_str)
    except ValueError:
        return False
    age_hours = (datetime.now(KST) - last_updated).total_seconds() / 3600
    if age_hours > _cache_ttl_hours():
        return False

    # 2) 범위 검사 - 캐시가 요청 범위를 포함하는지
    cached_start = meta.get("start")
    cached_end = meta.get("end")
    if not cached_start or not cached_end:
        return False
    if cached_start > start or cached_end < end:
        # 캐시 범위가 요청보다 좁으면 갱신 필요
        return False
    return True


def read_cache(source: str, series_id: str) -> Optional[pd.Series]:
    """캐시에서 시리즈를 읽어 반환. 캐시 파일이 없거나 손상되면 None."""
    data_path, _ = _cache_paths(source, series_id)
    if not data_path.exists():
        return None
    try:
        df = pd.read_parquet(data_path)
        # 단일 컬럼 DataFrame -> Series 변환
        if df.shape[1] != 1:
            logger.warning("Cache for %s has unexpected shape %s", series_id, df.shape)
            return None
        s = df.iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index)
        return s
    except Exception as e:  # noqa: BLE001 - 캐시 손상 시 무시하고 재수집
        logger.warning("Cache read failed for %s: %s", series_id, e)
        return None


def write_cache(
    source: str,
    series_id: str,
    series: pd.Series,
    start: str,
    end: str,
) -> None:
    """시리즈를 parquet으로 저장하고 메타데이터 JSON도 함께 기록.

    Args:
        source: 데이터 출처
        series_id: 시리즈 식별자
        series: 저장할 pd.Series (인덱스: DatetimeIndex)
        start, end: 이 캐시가 커버하는 데이터 범위 (YYYY-MM-DD)
    """
    if series.empty:
        # 빈 시리즈는 캐시하지 않음 (다음 호출에서 다시 시도하도록)
        logger.warning("Skipping cache write for empty series: %s", series_id)
        return

    data_path, meta_path = _cache_paths(source, series_id)
    # Series -> DataFrame 으로 저장 (parquet은 Series 직접 저장 미지원)
    df = series.to_frame(name=series.name or series_id)
    df.to_parquet(data_path, engine="pyarrow")

    meta = {
        "source": source,
        "series_id": series_id,
        "start": start,
        "end": end,
        "rows": int(len(series)),
        "last_updated": datetime.now(KST).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Cached %s (%d rows) -> %s", series_id, len(series), data_path)


# -----------------------------------------------------------------------------
# 타임존 정규화
# -----------------------------------------------------------------------------
def to_kst_index(series: pd.Series) -> pd.Series:
    """시리즈의 인덱스를 KST 기준 일별 DatetimeIndex로 통일.

    동작:
        - tz-aware 인덱스: KST로 변환
        - tz-naive 인덱스: KST를 부여한 뒤 tz-naive로 normalize
          (대부분의 매크로/금융 시계열은 날짜 단위라 시각 정보는 의미가 적음)
        - 최종적으로 시각 부분은 제거 (날짜만 유지, tz-naive로 반환)

    Returns:
        인덱스가 일별(date-only) DatetimeIndex인 시리즈.
    """
    if series.empty:
        return series

    idx = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        idx = idx.tz_convert(KST).tz_localize(None)
    # 시각 부분 제거 → 자정 (00:00:00) 으로 정규화
    idx = idx.normalize()
    out = series.copy()
    out.index = idx
    return out


# -----------------------------------------------------------------------------
# .env / Streamlit Cloud secrets 로드 헬퍼 (v17)
# -----------------------------------------------------------------------------
def load_env() -> None:
    """프로젝트 루트의 .env 파일을 로드 (이미 로드되어 있으면 no-op).

    python-dotenv가 설치되어 있어야 함. 없으면 경고만 출력하고 진행.
    """
    try:
        from dotenv import load_dotenv

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            logger.debug(".env file not found at %s", env_path)
    except ImportError:
        logger.warning("python-dotenv not installed. Skipping .env load.")


def _try_streamlit_secret(key: str) -> str:
    """Streamlit Cloud 배포 환경일 때 st.secrets에서 값을 읽으려 시도.

    로컬 실행 환경(streamlit 컨텍스트 없음)이나 streamlit 미설치 시 빈 문자열 반환.
    secrets.toml 미존재·키 미등록 시에도 조용히 빈 문자열 반환.
    """
    try:
        import streamlit as st  # noqa: WPS433 — lazy import (배포 환경 전용).
    except ImportError:
        return ""
    try:
        # st.secrets는 secrets.toml 미존재 시에도 lazy attribute이므로
        # 접근 자체는 운행 가능하나 키 접근에서 예외 발생 가능.
        value = st.secrets[key]  # type: ignore[index]
    except Exception:  # noqa: BLE001 — streamlit 내부 예외 종류 다양.
        return ""
    return str(value).strip() if value is not None else ""


def require_env(key: str) -> str:
    """환경 변수를 읽어 반환. 없거나 비어있으면 MissingAPIKeyError 발생.

    조회 순서 (v17):
        1) OS 환경 변수 (.env 이미 로드된 경우 포함).
        2) Streamlit Cloud secrets (st.secrets[key]) — Cloud 배포 전용.
        3) .env 로드 후 OS 환경 변수 재조회 — 로컬 fallback.

    이 순서 덕에 로컬(streamlit run + .env)·Streamlit Cloud(st.secrets) 둘 다
    수정 없이 동작한다.
    """
    # 1) 이미 설정된 OS env (예: 세션에서 export 했거나 알맞은 .env가 로드된 경우).
    value = os.environ.get(key, "").strip()
    if value and not value.startswith("your_"):
        return value

    # 2) Streamlit Cloud secrets (경우에 따라는 키 부재·secrets.toml 미설정이면 빈 문자열).
    secret_value = _try_streamlit_secret(key)
    if secret_value and not secret_value.startswith("your_"):
        # 후속 로직이 os.environ을 기대하는 경우를 대비해 주입.
        os.environ[key] = secret_value
        return secret_value

    # 3) .env 로드 후 재조회 (로컬 실행 fallback).
    load_env()
    value = os.environ.get(key, "").strip()
    if not value or value.startswith("your_"):
        raise MissingAPIKeyError(
            f"환경 변수 '{key}'가 설정되지 않았습니다. "
            f"로컬 실행: .env 파일에 {key}를 설정하세요 (.env.example 참고). "
            f"Streamlit Cloud: 앱 Settings > Secrets에 {key}를 입력하세요."
        )
    return value
