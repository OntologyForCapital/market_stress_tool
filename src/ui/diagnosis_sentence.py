"""오늘의 진단 한 줄 생성기 (1차 단순 버전).

[설계 노트]
    - 1차 도구에서는 패턴 + 종합 백분위 + 위험도 라벨만 조합한 단순 문장을 사용.
    - 사용자가 추후 패턴별/강도별로 정성스러운 메시지로 교체할 예정이므로,
      이 모듈은 그 자리만 잡아두는 역할.
    - 확장 시: PATTERN_LABELS_KR 키를 분기해 패턴별 맞춤 문구를 구성하면 됨.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline import DiagnosisResult

from src.ui import labels as L


def generate_diagnosis_sentence(result: "DiagnosisResult") -> str:
    """진단 결과 → 한 줄 요약 문장.

    1차 형식 (사용자 후속 다듬기 전):
        "패턴: {패턴}. 종합 스트레스 {백분위}점 ({위험도})."

    Args:
        result: `run_full_diagnosis()`의 반환값.

    Returns:
        한 줄 한국어 문장.
    """
    pattern_kr = L.pattern_to_korean(result.pattern_label)
    pct = result.composite_percentile

    if pct is None or math.isnan(pct):
        return f"패턴: {pattern_kr}. 종합 스트레스 지수 산출 불가 (데이터 부족)."

    level_kr = L.percentile_to_level_label_kr(pct)
    return f"패턴: {pattern_kr}. 종합 스트레스 {pct:.0f}점 ({level_kr})."


__all__ = ["generate_diagnosis_sentence"]
