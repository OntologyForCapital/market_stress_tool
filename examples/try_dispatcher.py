"""dispatcher 동작 검증 예시.

사용 전 준비:
    1. .env에 FRED_API_KEY, ECOS_API_KEY, KRX_API_KEY 설정
    2. pip install -r requirements.txt

실행:
    cd market_stress_tool
    python -m examples.try_dispatcher
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (스크립트로 직접 실행 시)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# .env 자동 로드 (python-dotenv 설치되어 있을 경우)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    print("주의: python-dotenv가 없습니다. 환경변수를 수동으로 export 하세요.")

# 로그 출력 활성화 (dispatcher 동작 추적용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import (
    load_variables,
    load_target_variables,
    get_enabled_variables,
    load_channel_mapping,
    load_channel_weights,
    load_risk_directions,
)
from src.data_collection.dispatcher import fetch_variable, fetch_all_variables


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =============================================================================
# 1) variables.yaml 파싱 확인
# =============================================================================
section("1) config.py: variables.yaml 파싱 확인")

all_vars = load_variables()
enabled = get_enabled_variables()
targets = load_target_variables()

print(f"전체 variables  : {len(all_vars)}개")
print(f"enabled=True    : {len(enabled)}개")
print(f"  → {[v.code for v in enabled]}")
print(f"target_variables: {len(targets)}개")
print(f"  → {[v.code for v in targets]}")
print(f"채널 매핑       : {load_channel_mapping()}")
print(f"위험 방향       : {load_risk_directions()}")
print(f"채널 가중치(정규화) : {load_channel_weights()}")

# =============================================================================
# 2) 단일 변수 수집: VIX (FRED)
# =============================================================================
section("2) fetch_variable: VIX (FRED) 1년치")

vix_var = next(v for v in all_vars if v.code == "VIX")
print(f"Variable: {vix_var.code} (source={vix_var.source}, series_id={vix_var.series_id})")

try:
    vix = fetch_variable(vix_var, "2024-01-01", "2025-01-01")
    print(f"수집 성공: {len(vix)}개 관측치, name={vix.name!r}")
    print(vix.tail(3))
except Exception as e:
    print(f"수집 실패: {type(e).__name__}: {e}")

# =============================================================================
# 3) KOSPI (target_variable, source=krx, 한글 series_id)
# =============================================================================
section("3) fetch_variable: KOSPI (KRX) 1개월치")

kospi_var = next(v for v in targets if v.code == "KOSPI")
print(f"Variable: {kospi_var.code} (source={kospi_var.source}, series_id={kospi_var.series_id!r})")

try:
    kospi = fetch_variable(kospi_var, "2024-12-01", "2024-12-31")
    print(f"수집 성공: {len(kospi)}개 관측치, name={kospi.name!r}")
    print(kospi.tail(3))
except Exception as e:
    print(f"수집 실패: {type(e).__name__}: {e}")

# =============================================================================
# 4) KR_US_RATE_DIFF (computed: ECOS + FRED)
# =============================================================================
section("4) fetch_variable: KR_US_RATE_DIFF (computed)")

diff_var = next(v for v in all_vars if v.code == "KR_US_RATE_DIFF")
print(f"Variable: {diff_var.code} (source={diff_var.source})")
print(f"Components: {list(diff_var.components.keys())}")
for name, comp in diff_var.components.items():
    print(f"  - {name}: source={comp.source}, "
          f"series_id={comp.series_id}, stat_code={comp.stat_code}")

try:
    diff = fetch_variable(diff_var, "2024-01-01", "2024-12-31")
    print(f"수집 성공: {len(diff)}개 관측치, name={diff.name!r}")
    print(diff.tail(5))
except Exception as e:
    print(f"수집 실패: {type(e).__name__}: {e}")

# =============================================================================
# 5) 전체 enabled + target 일괄 수집 (작업 3 검증)
# =============================================================================
section("5) fetch_all_variables: enabled + targets 일괄 수집")

all_to_fetch = enabled + targets
print(f"수집 대상: {len(all_to_fetch)}개 변수")
print(f"  → {[v.code for v in all_to_fetch]}")

result = fetch_all_variables(
    all_to_fetch,
    start_date="2024-01-01",
    end_date="2024-12-31",
    use_cache=True,
)

print(f"\n성공한 변수 ({len(result)}/{len(all_to_fetch)}): {sorted(result.keys())}")
print(f"\n핵심 확인:")
print(f"  KRW_USD 포함 여부 : {'KRW_USD' in result}  (작업 3 요구사항)")
print(f"  KOSPI 포함 여부   : {'KOSPI' in result}")
print(f"  KOSDAQ 포함 여부  : {'KOSDAQ' in result}")

# 각 시리즈 요약
print("\n변수별 요약:")
print(f"  {'코드':<25s} {'관측치':>6s}  기간")
for code in sorted(result.keys()):
    s = result[code]
    if len(s) > 0:
        date_range = f"{s.index.min().date()} ~ {s.index.max().date()}"
    else:
        date_range = "(빈 시리즈)"
    print(f"  {code:<25s} {len(s):>6d}  {date_range}")

# 실패한 변수
failed_codes = {v.code for v in all_to_fetch} - set(result.keys()) - {
    v.code for v in all_to_fetch if not v.enabled
}
if failed_codes:
    print(f"\n실패한 변수: {sorted(failed_codes)}")

print("\n검증 완료.")
