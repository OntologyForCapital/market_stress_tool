"""v16: Dynamics 검증 인프라.

9개 폭락 시점 각각에 대해 T-60일 ~ T+120일 (181 영업일) 윈도우의
일별 composite_z + 채널별 z-score 시계열을 추출하고 시각화한다.

목적
----
backtest_run.py는 단일 시점 진단만 검증한다. 본 스크립트는 그 옆에
시간적 dynamics를 더해 다음을 본다:
    - 위기 진입: 폭락 전 신호 선행성
    - 위기 한복판: 신호 안정성
    - 회복기: 신호 해제 sensitivity
    - 평시: false positive 선택성

방식
----
도구 코드는 일절 건드리지 않고, 각 시점마다 run_full_diagnosis를 1회
호출(총 9회)한 뒤, DiagnosisResult.z_panel에 도구가 채워둔 표준화 후
변수별 z-score 시계열을 사용해 stress_table(S1~S5, composite)을
스크립트 내에서 재구성한다.
    - z_panel은 백분위 clip이 없는 원본 z를 보존
    - compute_channel_scores / compute_composite_index 사용
    - thresholds.composite_method (v15, 기본 l2_norm) 그대로 적용

산출물
------
- dynamics_composite_long.csv
- dynamics_channels_long.csv
- dynamics_metrics.csv
- dynamics_composite_grid.png
- dynamics_channels_grid.png

실행
----
    python backtest_dynamics.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")  # headless 환경 안전
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.stress_index import (
    build_stress_index_table,
)
from src.config import (
    load_channel_mapping,
    load_channel_weights,
    load_composite_method,
)
from src.pipeline import run_full_diagnosis


# =============================================================================
# 설정
# =============================================================================

# backtest_run.py와 동일한 9개 시점.
TARGET_DATES: list[tuple[str, str]] = [
    ("2015-08-24", "China shock"),
    ("2018-10-29", "미중 무역전쟁"),
    ("2020-03-19", "코로나"),
    ("2022-01-27", "Fed 매파 전환"),
    ("2022-07-04", "Fed 75bp + 인플레 정점"),
    ("2022-09-30", "영국 미니예산/강달러"),
    ("2024-08-05", "일본 캐리 청산"),
    ("2025-04-09", "트럼프 상호관세"),
    ("2026-03-30", "미국-이스라엘-이란 전쟁"),
]

# 윈도우 (영업일 기준): T-60 ~ T+120.
WIN_BEFORE = 60
WIN_AFTER = 120

# peak 탐색 윈도우 (T 주변).
PEAK_WIN = 30

# 그래프 자원 — 한국어 깨짐 방지: 영문 라벨 + 한글 fallback 시도.
plt.rcParams["axes.unicode_minus"] = False
for font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"):
    try:
        plt.rcParams["font.family"] = font_name
        break
    except Exception:  # pragma: no cover - 환경 의존
        continue


CHANNEL_COLORS = {
    1: "#1f77b4",  # 신용/은행
    2: "#ff7f0e",  # 실물경제
    3: "#2ca02c",  # 위험프리미엄
    4: "#d62728",  # 원자재/무역
    5: "#9467bd",  # 환율/캐리
}


# =============================================================================
# 유틸
# =============================================================================


@dataclass
class WindowResult:
    """단일 시점 윈도우 추출 결과."""

    date_target: pd.Timestamp
    label: str
    stress_table: pd.DataFrame   # 컬럼: S1..S5, composite. 인덱스 = 영업일.
    days_from_t0: pd.Series      # 영업일 차이 (정수, 인덱스=영업일).


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _yaml_path() -> Path:
    return _project_root() / "config" / "variables.yaml"


def _rebuild_stress_table_from_zpanel(z_panel: pd.DataFrame) -> pd.DataFrame:
    """z_panel + yaml 매핑 → stress_table(S1..S5 + composite) 재구성.

    pipeline.run_full_diagnosis가 내부에서 만든 stress_table은
    DiagnosisResult에 직접 노출되지 않지만, z_panel은 노출되므로
    동일 yaml 설정으로 build_stress_index_table을 다시 호출하면
    같은 결과를 얻을 수 있다 (도구 코드 무변경 원칙).
    """
    yaml_path = _yaml_path()
    variable_to_channel = load_channel_mapping(yaml_path)
    channel_weights = load_channel_weights(yaml_path, normalize=True)
    composite_method = load_composite_method(yaml_path)
    return build_stress_index_table(
        z_panel,
        variable_to_channel=variable_to_channel,
        channel_weights=channel_weights,
        composite_method=composite_method,
    )


def _slice_window(
    stress_table: pd.DataFrame,
    t0: pd.Timestamp,
    before: int = WIN_BEFORE,
    after: int = WIN_AFTER,
) -> tuple[pd.DataFrame, pd.Series]:
    """t0 기준 [-before, +after] 영업일 윈도우 추출.

    Returns:
        (sliced_stress_table, days_from_t0).
        days_from_t0는 sliced의 인덱스에 정렬된 정수 시리즈
        (예: t0이면 0, 직전 영업일이면 -1).
    """
    idx = stress_table.index
    # t0를 데이터에 맞춰 정규화 (정확 일치 없으면 직전 영업일).
    if t0 in idx:
        t0_pos = idx.get_loc(t0)
    else:
        prior = idx[idx <= t0]
        if len(prior) == 0:
            raise ValueError(f"t0={t0} 이전 영업일이 없음")
        t0_pos = idx.get_loc(prior[-1])

    start = max(0, t0_pos - before)
    end = min(len(idx) - 1, t0_pos + after)
    sliced = stress_table.iloc[start:end + 1].copy()

    # days_from_t0 (영업일 오프셋, t0 기준 0).
    offsets = np.arange(start - t0_pos, end - t0_pos + 1, dtype=int)
    days_series = pd.Series(offsets, index=sliced.index, name="days_from_t0")

    return sliced, days_series


# =============================================================================
# 시점별 추출 (run_full_diagnosis × 9)
# =============================================================================


def collect_windows() -> list[WindowResult]:
    """9개 시점 각각에 대해 run_full_diagnosis 1회 → 윈도우 추출."""
    results: list[WindowResult] = []
    for date_str, label in TARGET_DATES:
        t0 = pd.Timestamp(date_str)
        # 5년 롤링 표준화 + 워밍아웃 + T+120일 여유를 위해 넉넉히 8년.
        start = (t0 - pd.DateOffset(years=8)).strftime("%Y-%m-%d")
        end = (t0 + pd.DateOffset(days=WIN_AFTER + 80)).strftime("%Y-%m-%d")

        print(f"\n=== {date_str} ({label}) ===")
        try:
            diag = run_full_diagnosis(
                start_date=start, end_date=end, as_of=t0, use_cache=True,
            )
        except Exception as e:  # pragma: no cover - 실 API/캐시 환경
            print(f"  실패: {e}")
            traceback.print_exc()
            continue

        try:
            stress_table = _rebuild_stress_table_from_zpanel(diag.z_panel)
        except Exception as e:
            print(f"  stress_table 재구성 실패: {e}")
            traceback.print_exc()
            continue

        sliced, days = _slice_window(stress_table, t0)
        n_obs = len(sliced)
        nan_pct = sliced["composite"].isna().mean() * 100
        t0_mask = (days == 0)
        if t0_mask.any():
            t0_score = sliced.loc[days[t0_mask].index[0], "composite"]
            t0_str = f"{t0_score:.2f}" if pd.notna(t0_score) else "NaN"
        else:
            t0_str = "미존재"
        print(f"  윈도우 {n_obs}영업일 (NaN composite: {nan_pct:.1f}%), t0 종합={t0_str}")
        results.append(
            WindowResult(
                date_target=t0, label=label,
                stress_table=sliced, days_from_t0=days,
            )
        )
    return results


# =============================================================================
# CSV 출력
# =============================================================================


def write_csvs(windows: list[WindowResult]) -> None:
    """3개 CSV 생성: composite_long, channels_long, metrics."""
    comp_rows: list[dict] = []
    chan_rows: list[dict] = []

    for w in windows:
        for date_obs, row in w.stress_table.iterrows():
            offset = int(w.days_from_t0.loc[date_obs])
            comp_rows.append({
                "date_target": w.date_target.strftime("%Y-%m-%d"),
                "label": w.label,
                "days_from_t0": offset,
                "date_obs": date_obs.strftime("%Y-%m-%d"),
                "composite_z": round(float(row["composite"]), 4)
                    if pd.notna(row["composite"]) else None,
            })
            for ch in (1, 2, 3, 4, 5):
                col = f"S{ch}"
                if col not in w.stress_table.columns:
                    continue
                val = row[col]
                chan_rows.append({
                    "date_target": w.date_target.strftime("%Y-%m-%d"),
                    "label": w.label,
                    "days_from_t0": offset,
                    "date_obs": date_obs.strftime("%Y-%m-%d"),
                    "channel": ch,
                    "z_score": round(float(val), 4) if pd.notna(val) else None,
                })

    pd.DataFrame(comp_rows).to_csv("dynamics_composite_long.csv", index=False)
    pd.DataFrame(chan_rows).to_csv("dynamics_channels_long.csv", index=False)

    # Metrics
    metric_rows: list[dict] = []
    for w in windows:
        offset_series = w.days_from_t0
        comp = w.stress_table["composite"]

        def _at_offset(offset: int) -> float:
            """offset에 가장 가까운 (≤ offset) 영업일의 composite."""
            mask = offset_series <= offset
            if not mask.any():
                return float("nan")
            # offset에 정확히 맞으면 그 행, 아니면 그 이하 중 최댓값.
            candidates = offset_series[mask]
            chosen_idx = candidates.idxmax()
            return float(comp.loc[chosen_idx])

        t0_score = _at_offset(0)
        t_plus_60 = _at_offset(60)

        # peak 탐색: -30 ~ +30 윈도우.
        peak_mask = (offset_series >= -PEAK_WIN) & (offset_series <= PEAK_WIN)
        peak_window = comp[peak_mask].dropna()
        if peak_window.empty:
            peak_score = float("nan")
            peak_offset = None
        else:
            peak_idx = peak_window.idxmax()
            peak_score = float(peak_window.loc[peak_idx])
            peak_offset = int(offset_series.loc[peak_idx])

        if pd.notna(t0_score) and pd.notna(t_plus_60) and t0_score != 0:
            recovery_pct = (t0_score - t_plus_60) / t0_score * 100.0
        else:
            recovery_pct = float("nan")

        metric_rows.append({
            "label": w.label,
            "date_target": w.date_target.strftime("%Y-%m-%d"),
            "t0_score": round(t0_score, 4) if pd.notna(t0_score) else None,
            "peak_score": round(peak_score, 4) if pd.notna(peak_score) else None,
            "peak_offset_days": peak_offset,
            "t_plus_60_score": round(t_plus_60, 4) if pd.notna(t_plus_60) else None,
            "recovery_pct": round(recovery_pct, 2)
                if pd.notna(recovery_pct) else None,
            "peak_before_t0": (peak_offset is not None and peak_offset < 0),
        })
    pd.DataFrame(metric_rows).to_csv("dynamics_metrics.csv", index=False)
    print("\nCSV 3개 저장: dynamics_composite_long.csv, "
          "dynamics_channels_long.csv, dynamics_metrics.csv")


# =============================================================================
# 그래프
# =============================================================================


def _plot_composite_grid(windows: list[WindowResult]) -> None:
    n = len(windows)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharey=False)
    axes = np.atleast_2d(axes)

    for i, w in enumerate(windows):
        ax = axes[i // cols, i % cols]
        x = w.days_from_t0.values
        y = w.stress_table["composite"].values
        # NaN 구간은 끊어진 라인.
        ax.plot(x, y, color="#222", lw=1.4)
        ax.axvline(0, color="crimson", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="#999", lw=0.5, alpha=0.5)
        # T=0 점 강조
        try:
            t0_idx = int(np.where(x == 0)[0][0])
            if pd.notna(y[t0_idx]):
                ax.plot(0, y[t0_idx], "o", color="crimson", ms=4)
        except IndexError:
            pass
        ax.set_title(f"{w.label}\n({w.date_target.strftime('%Y-%m-%d')})",
                     fontsize=10)
        ax.set_xlabel("days_from_t0", fontsize=8)
        ax.set_ylabel("composite_z", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-WIN_BEFORE, WIN_AFTER)

    # 빈 subplot 숨김
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig.suptitle(
        "Composite z dynamics (T-60 ~ T+120) — v16",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    fig.savefig("dynamics_composite_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_channels_grid(windows: list[WindowResult]) -> None:
    n = len(windows)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharey=False)
    axes = np.atleast_2d(axes)

    handles_for_legend = []
    labels_for_legend = []

    for i, w in enumerate(windows):
        ax = axes[i // cols, i % cols]
        x = w.days_from_t0.values
        for ch in (1, 2, 3, 4, 5):
            col = f"S{ch}"
            if col not in w.stress_table.columns:
                continue
            y = w.stress_table[col].values
            line, = ax.plot(
                x, y,
                color=CHANNEL_COLORS[ch], lw=1.1,
                label=f"S{ch}",
            )
            if i == 0:
                handles_for_legend.append(line)
                labels_for_legend.append(f"S{ch}")

        ax.axvline(0, color="crimson", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="#999", lw=0.5, alpha=0.5)
        ax.set_title(f"{w.label}\n({w.date_target.strftime('%Y-%m-%d')})",
                     fontsize=10)
        ax.set_xlabel("days_from_t0", fontsize=8)
        ax.set_ylabel("channel z", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-WIN_BEFORE, WIN_AFTER)

    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")

    if handles_for_legend:
        fig.legend(
            handles_for_legend, labels_for_legend,
            loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, -0.01),
            fontsize=10, frameon=False,
        )
    fig.suptitle(
        "Channel z dynamics (T-60 ~ T+120) — v16",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    fig.savefig("dynamics_channels_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_plots(windows: list[WindowResult]) -> None:
    _plot_composite_grid(windows)
    _plot_channels_grid(windows)
    print("PNG 2개 저장: dynamics_composite_grid.png, "
          "dynamics_channels_grid.png")


# =============================================================================
# main
# =============================================================================


def main() -> None:
    windows = collect_windows()
    if not windows:
        print("어떤 시점도 추출 못 함. 실패.")
        sys.exit(1)
    write_csvs(windows)
    write_plots(windows)

    # 요약 출력
    metrics = pd.read_csv("dynamics_metrics.csv")
    print("\n=== Dynamics metrics ===")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
