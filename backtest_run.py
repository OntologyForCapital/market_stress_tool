# backtest_run.py
import sys
sys.path.insert(0, "src")
import pandas as pd
from pipeline import run_full_diagnosis

TODAY = pd.Timestamp.today().normalize()

target_dates = [
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

results = []
var_z_records = []  # 변수별 z-score long format

for date_str, label in target_dates:
    as_of = pd.Timestamp(date_str)
    start = (as_of - pd.DateOffset(years=7)).strftime("%Y-%m-%d")
    end_ts = min(as_of + pd.DateOffset(days=200), TODAY)
    end = end_ts.strftime("%Y-%m-%d")

    print(f"\n=== {date_str} ({label}) ===")
    try:
        r = run_full_diagnosis(
            start_date=start, end_date=end, as_of=as_of, use_cache=True,
        )

        knn_top1 = r.similar_dates.index[0] if len(r.similar_dates) > 0 else None

        origins = []
        if r.origin_result is not None and r.origin_result.origin_variable is not None:
            origins.append(
                f"{r.origin_result.origin_variable}(ch{r.origin_result.origin_channel})"
            )
            breach_vars = list(r.origin_result.variable_first_breach.keys())
            for v in breach_vars[:5]:
                if v != r.origin_result.origin_variable:
                    origins.append(v)
                if len(origins) >= 4:
                    break

        results.append({
            "date": date_str, "label": label,
            "composite_z": round(r.composite_score, 2),
            "composite_pct": round(r.composite_percentile, 1),
            "S1": round(r.channel_scores.get(1, float("nan")), 2),
            "S2": round(r.channel_scores.get(2, float("nan")), 2),
            "S3": round(r.channel_scores.get(3, float("nan")), 2),
            "S4": round(r.channel_scores.get(4, float("nan")), 2),
            "S5": round(r.channel_scores.get(5, float("nan")), 2),
            "pattern": r.pattern_label,
            "knn_top1": str(knn_top1)[:10] if knn_top1 is not None else None,
            "origins": ", ".join(origins) if origins else "",
            "failed": ", ".join(r.failed_variables),
        })

        # 변수별 z-score 기록
        for var_code, z_val in r.variable_z_scores.items():
            var_z_records.append({
                "date": date_str, "label": label,
                "variable": var_code,
                "z_score": round(z_val, 2) if pd.notna(z_val) else None,
            })

        print(f"  z={r.composite_score:.2f}, pattern={r.pattern_label}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append({"date": date_str, "label": label, "error": str(e)})

if any(row.get("error") for row in results):
    df_error = pd.DataFrame(results)
    print("\n\n=== 오류가 있어 CSV 저장을 건너뜁니다 ===")
    print(df_error.to_string())
    print("네트워크/API 키/캐시를 보강한 뒤 다시 실행하세요.")
    sys.exit(1)

# 채널 점수 CSV
df = pd.DataFrame(results)
print("\n\n=== 채널 점수 ===")
print(df.to_string())
df.to_csv("backtest_results.csv", index=False)

# 변수별 z-score CSV (long + wide)
df_var = pd.DataFrame(var_z_records)
df_var.to_csv("backtest_var_zscores_long.csv", index=False)

df_var_wide = df_var.pivot_table(
    index=["date", "label"], columns="variable", values="z_score"
)
df_var_wide.to_csv("backtest_var_zscores_wide.csv")

print("\n=== 변수별 z-score (분석용 wide) ===")
print(df_var_wide.to_string())

print("\n3개 CSV 저장 완료.")
