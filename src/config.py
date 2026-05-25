"""variables.yaml 파싱 헬퍼.

이 모듈은 config/variables.yaml의 구조를 dataclass로 변환하여
타입 안전하게 다른 모듈에서 사용할 수 있게 한다.

설계 원칙:
    - 기존 standardize.py / stress_index.py의 YAML 파싱 코드는 그대로 둠 (점진 이행)
    - 향후 dispatcher.py / pipeline.py / app.py가 이 모듈만 사용하면 됨
    - 알 수 없는 필드는 무시 (forward compatibility)
    - enabled=False 변수도 dataclass에는 포함하되 별도 헬퍼로 필터링

주요 함수:
    load_variables       : variables 섹션 → list[Variable] (enabled 무관)
    load_target_variables: target_variables 섹션 → list[Variable]
    load_channel_mapping : {코드: 채널번호} (enabled=True만)
    load_risk_directions : {코드: 'positive'|'negative'} (enabled=True만)
    load_channel_weights : {채널번호: 가중치} (정규화 포함)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 기본 yaml 경로 (프로젝트 루트 기준)
DEFAULT_VARIABLES_YAML = Path(__file__).resolve().parent.parent / "config" / "variables.yaml"

# v13: 변수별 사전 변환 종류.
# z-score 계산 이전에 적용되며, yaml의 transform 필드를 통해 지정.
#   level         : 변환 없음 (기존 동작, 기본값)
#   yoy_pct       : 1년 전 대비 % 변화 (252영업일)
#   pct_change_6m : 6개월 전 대비 % 변화 (126영업일)
#   diff_6m       : 6개월 전 대비 단순 차분 (126영업일)
VALID_TRANSFORMS: frozenset[str] = frozenset({"level", "yoy_pct", "pct_change_6m", "diff_6m"})

# v14: risk_direction 허용 값.
#   positive      : z 그대로 (수치 ↑ = 위험 ↑)
#   negative      : -z (수치 ↓ = 위험 ↑)
#   bidirectional : (|z| - threshold).clip(lower=0) — 양방향 임계값 초과 부분만 신호
VALID_RISK_DIRECTIONS: frozenset[str] = frozenset({"positive", "negative", "bidirectional"})

# v15: 종합점수(composite) 산출 방식.
#   mean    : 종래 가중평균 (부호 보존, 채널 상쇄 가능)
#   l2_norm : RMS = sqrt(mean(S_k²)) — 항상 ≥0, 위험 강도(평시 상태로부터의 거리)
VALID_COMPOSITE_METHODS: frozenset[str] = frozenset({"mean", "l2_norm"})

# v15: composite_method 전역 기본값. yaml 미설정 시 fallback.
DEFAULT_COMPOSITE_METHOD: str = "l2_norm"

# v14: bidirectional 전역 기본 임계값 (단위: σ). yaml 미설정 시 fallback.
DEFAULT_BIDIRECTIONAL_THRESHOLD: float = 1.0


@dataclass
class ComponentRef:
    """KR_US_RATE_DIFF 같은 computed 변수의 한 컴포넌트.

    예시 YAML:
        components:
          kr_rate: { source: ecos, stat_code: "722Y001" }
          us_rate: { source: fred, series_id: "DFF" }
    """
    name: str                          # 컴포넌트 이름 (예: 'kr_rate', 'us_rate')
    source: str                        # 'fred' | 'ecos' | 'yfinance' | 'krx'
    series_id: str | None = None       # FRED/yfinance/KRX 시리즈
    stat_code: str | None = None       # ECOS 통계표 코드
    item_code: str | None = None       # ECOS 통계항목 코드 (있을 수도 없을 수도)


@dataclass
class Variable:
    """variables.yaml의 단일 변수 엔트리 (variables/target_variables/auxiliary 공통).

    필드는 YAML의 슈퍼셋. 섹션별로 사용되지 않는 필드는 기본값으로 둔다.

    Args:
        code            : 내부 식별자 (예: 'VIX')
        name_kr         : 한국어 표시명
        name_en         : 영어 표시명
        source          : 'fred' | 'yfinance' | 'ecos' | 'krx' | 'pykrx' | 'computed'
        series_id       : 출처별 시리즈 코드 (computed면 None)
        channel         : 1~5 (target/auxiliary는 None)
        risk_direction  : 'positive' | 'negative' (target/auxiliary는 None)
        unit            : 단위
        frequency       : 'daily' | 'weekly' | 'monthly'
        enabled         : 1차 프로토타입 사용 여부 (target_variables는 항상 True 취급)
        notes           : 한계/주의사항
        item_code       : ECOS 다중항목 통계표용
        fallback_series_id: yfinance 1차 실패 시 대체 티커
        components      : computed 변수의 하위 시리즈
        section         : 어느 YAML 섹션에서 왔는지 ('variables'|'target_variables'|'auxiliary_variables')
        transform       : (v13) z-score 산출 전 사전 변환 종류
                          'level' (기본) | 'yoy_pct' | 'pct_change_6m' | 'diff_6m'
        bidirectional_threshold:
                          (v14) risk_direction='bidirectional' 일 때 적정 밴드 임계값 (σ).
                          None이면 전역 기본값 사용. positive/negative에서는 무시.
    """
    code: str
    name_kr: str = ""
    name_en: str = ""
    source: str = ""
    series_id: str | None = None
    channel: int | None = None
    risk_direction: str | None = None
    unit: str = ""
    frequency: str = "daily"
    enabled: bool = True
    notes: str = ""
    item_code: str | None = None
    fallback_series_id: str | None = None
    components: dict[str, ComponentRef] = field(default_factory=dict)
    section: str = "variables"
    transform: str = "level"  # v13: 사전 변환 종류. yaml에 없으면 'level' 기본값.
    bidirectional_threshold: float | None = None  # v14: bidirectional 변수별 임계값 (σ). None이면 전역 기본값.


# =============================================================================
# 내부 헬퍼
# =============================================================================
def _read_yaml(path: Path | str | None) -> dict[str, Any]:
    """YAML 파일을 dict로 로드."""
    yaml_path = Path(path) if path else DEFAULT_VARIABLES_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"variables.yaml not found at {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_components(raw: dict[str, Any] | None) -> dict[str, ComponentRef]:
    """`components:` 섹션을 ComponentRef dict로 변환."""
    if not raw:
        return {}
    result: dict[str, ComponentRef] = {}
    for comp_name, comp_data in raw.items():
        if not isinstance(comp_data, dict):
            continue
        result[comp_name] = ComponentRef(
            name=comp_name,
            source=comp_data.get("source", ""),
            series_id=comp_data.get("series_id"),
            stat_code=comp_data.get("stat_code"),
            item_code=comp_data.get("item_code"),
        )
    return result


def _parse_variable_entry(entry: dict[str, Any], section: str, default_enabled: bool = True) -> Variable:
    """YAML의 단일 변수 dict를 Variable dataclass로 변환.

    Args:
        entry: YAML의 한 항목 dict.
        section: 'variables' | 'target_variables' | 'auxiliary_variables'.
        default_enabled: enabled 필드가 없을 때 기본값.
            (target_variables는 enabled 키를 안 가지므로 True가 적절)

    Raises:
        ValueError: transform/risk_direction 필드가 허용 범위를 벗어나거나,
                    bidirectional_threshold가 음수이면 발생.
    """
    code_val = entry.get("code")

    # v13: transform 메타데이터 파싱 + 검증.
    transform_val = entry.get("transform", "level")
    if transform_val not in VALID_TRANSFORMS:
        raise ValueError(
            f"변수 '{code_val}'의 transform 값이 잘못되었습니다: "
            f"{transform_val!r}. 허용 값: {sorted(VALID_TRANSFORMS)}"
        )

    # v14: risk_direction 검증 (None 허용 — target/auxiliary 변수는 설정하지 않음).
    rd_val = entry.get("risk_direction")
    if rd_val is not None and rd_val not in VALID_RISK_DIRECTIONS:
        raise ValueError(
            f"변수 '{code_val}'의 risk_direction 값이 잘못되었습니다: "
            f"{rd_val!r}. 허용 값: {sorted(VALID_RISK_DIRECTIONS)}"
        )

    # v14: bidirectional_threshold 파싱 및 검증.
    bidir_thr_raw = entry.get("bidirectional_threshold")
    bidir_thr_val: float | None = None
    if bidir_thr_raw is not None:
        try:
            bidir_thr_val = float(bidir_thr_raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"변수 '{code_val}'의 bidirectional_threshold를 float로 변환 불가: {bidir_thr_raw!r}"
            ) from e
        if bidir_thr_val < 0:
            raise ValueError(
                f"변수 '{code_val}'의 bidirectional_threshold는 0 이상이어야 합니다: {bidir_thr_val}"
            )

    return Variable(
        code=entry["code"],
        name_kr=entry.get("name_kr", ""),
        name_en=entry.get("name_en", ""),
        source=entry.get("source", ""),
        series_id=entry.get("series_id"),
        channel=entry.get("channel"),
        risk_direction=rd_val,
        unit=entry.get("unit", ""),
        frequency=entry.get("frequency", "daily"),
        enabled=bool(entry.get("enabled", default_enabled)),
        notes=entry.get("notes", "") or "",
        item_code=entry.get("item_code"),
        fallback_series_id=entry.get("fallback_series_id"),
        components=_parse_components(entry.get("components")),
        section=section,
        transform=transform_val,
        bidirectional_threshold=bidir_thr_val,
    )


# =============================================================================
# Public API
# =============================================================================
def load_variables(yaml_path: Path | str | None = None) -> list[Variable]:
    """`variables:` 섹션을 list[Variable]로 반환.

    enabled=False 변수도 포함된다. 필터링이 필요하면 호출 측에서
    `[v for v in load_variables() if v.enabled]` 로 처리.
    """
    cfg = _read_yaml(yaml_path)
    raw_list = cfg.get("variables", []) or []
    return [_parse_variable_entry(e, section="variables", default_enabled=True) for e in raw_list]


def load_target_variables(yaml_path: Path | str | None = None) -> list[Variable]:
    """`target_variables:` 섹션을 list[Variable]로 반환.

    이 섹션은 enabled 키가 없으므로 기본 True 처리.
    """
    cfg = _read_yaml(yaml_path)
    raw_list = cfg.get("target_variables", []) or []
    return [_parse_variable_entry(e, section="target_variables", default_enabled=True) for e in raw_list]


def load_auxiliary_variables(yaml_path: Path | str | None = None) -> list[Variable]:
    """`auxiliary_variables:` 섹션을 list[Variable]로 반환."""
    cfg = _read_yaml(yaml_path)
    raw_list = cfg.get("auxiliary_variables", []) or []
    return [_parse_variable_entry(e, section="auxiliary_variables", default_enabled=False) for e in raw_list]


def load_channel_mapping(yaml_path: Path | str | None = None) -> dict[str, int]:
    """enabled=True 변수의 {코드: 채널번호} 매핑.

    target_variables/auxiliary는 채널이 없으므로 자동 제외.
    """
    variables = load_variables(yaml_path)
    return {
        v.code: v.channel
        for v in variables
        if v.enabled and v.channel is not None
    }


def load_risk_directions(yaml_path: Path | str | None = None) -> dict[str, str]:
    """enabled=True 변수의 {코드: risk_direction} 매핑."""
    variables = load_variables(yaml_path)
    return {
        v.code: v.risk_direction
        for v in variables
        if v.enabled and v.risk_direction is not None
    }


def load_transform_map(yaml_path: Path | str | None = None) -> dict[str, str]:
    """(v13) enabled=True 변수의 {코드: transform} 매핑.

    z-score 계산 이전 단계에서 apply_transform()에 전달하는 용도.
    yaml에 transform 필드가 없는 변수는 'level'로 동작 (기존 호환).
    """
    variables = load_variables(yaml_path)
    return {v.code: v.transform for v in variables if v.enabled}


def load_bidirectional_threshold_default(yaml_path: Path | str | None = None) -> float:
    """(v14) thresholds.bidirectional_threshold_default 반환. 없으면 1.0.

    bidirectional 변수가 개별 bidirectional_threshold를 명시하지 않은 경우
    이 값이 적용됨. yaml에 해당 키가 없거나 파싱 불가하면
    모듈 상수 DEFAULT_BIDIRECTIONAL_THRESHOLD(=1.0) 사용.
    """
    cfg = _read_yaml(yaml_path)
    raw = (cfg.get("thresholds") or {}).get("bidirectional_threshold_default")
    if raw is None:
        return DEFAULT_BIDIRECTIONAL_THRESHOLD
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BIDIRECTIONAL_THRESHOLD
    if val < 0:
        return DEFAULT_BIDIRECTIONAL_THRESHOLD
    return val


def load_bidirectional_thresholds(
    yaml_path: Path | str | None = None,
) -> dict[str, float]:
    """(v14) bidirectional 변수의 {코드: 임계값} 매핑.

    개별 bidirectional_threshold가 설정되었으면 그 값,
    아니면 전역 기본값을 사용. risk_direction이 bidirectional이 아닌
    변수는 매핑에서 제외.

    Returns:
        예: {'BRENT': 1.0, 'US_BEI_10Y': 1.0}
    """
    default_thr = load_bidirectional_threshold_default(yaml_path)
    variables = load_variables(yaml_path)
    out: dict[str, float] = {}
    for v in variables:
        if not v.enabled:
            continue
        if v.risk_direction != "bidirectional":
            continue
        out[v.code] = v.bidirectional_threshold if v.bidirectional_threshold is not None else default_thr
    return out


def load_channel_weights(yaml_path: Path | str | None = None, normalize: bool = True) -> dict[int, float]:
    """`channel_weights:` 섹션을 {채널번호: 가중치}로 반환.

    Args:
        yaml_path: variables.yaml 경로.
        normalize: True면 합이 1.0이 되도록 정규화.

    Returns:
        {1: 0.20, 2: 0.20, ...}
    """
    cfg = _read_yaml(yaml_path)
    raw = cfg.get("channel_weights", {}) or {}
    # YAML의 정수 키는 yaml.safe_load가 자동으로 int로 반환
    weights = {int(k): float(v) for k, v in raw.items()}
    if normalize and weights:
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
    return weights


def load_thresholds(yaml_path: Path | str | None = None) -> dict[str, Any]:
    """`thresholds:` 섹션 반환 (분석 모듈 기본값)."""
    cfg = _read_yaml(yaml_path)
    return dict(cfg.get("thresholds", {}) or {})


def load_composite_method(yaml_path: Path | str | None = None) -> str:
    """(v15) thresholds.composite_method 반환. 없으면 전역 기본값.

    종합점수(composite_z)를 산출할 때 적용할 집계 방식을 결정.
        - "mean"    : 종래 가중평균 (부호 보존).
        - "l2_norm" : RMS = sqrt(mean(S_k²)) — 위험 강도 해석.

    yaml에 해당 키가 없거나 값이 None이면 DEFAULT_COMPOSITE_METHOD가 돌아감.

    Raises:
        ValueError: VALID_COMPOSITE_METHODS 외의 값이 설정되어 있으면 발생.
    """
    cfg = _read_yaml(yaml_path)
    raw = (cfg.get("thresholds") or {}).get("composite_method")
    if raw is None:
        return DEFAULT_COMPOSITE_METHOD
    method = str(raw).strip()
    if method not in VALID_COMPOSITE_METHODS:
        raise ValueError(
            f"지원하지 않는 composite_method 값: {method!r}. "
            f"허용: {sorted(VALID_COMPOSITE_METHODS)}"
        )
    return method


# =============================================================================
# 편의 함수
# =============================================================================
def get_enabled_variables(yaml_path: Path | str | None = None) -> list[Variable]:
    """enabled=True인 variables 섹션 변수만 반환."""
    return [v for v in load_variables(yaml_path) if v.enabled]


def find_variable(code: str, yaml_path: Path | str | None = None) -> Variable | None:
    """코드로 변수를 찾음 (variables + target_variables + auxiliary 모두 검색).

    None을 반환하면 어디에도 없음.
    """
    for loader in (load_variables, load_target_variables, load_auxiliary_variables):
        for v in loader(yaml_path):
            if v.code == code:
                return v
    return None
