from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
FINAL = PROJECT_ROOT / "results/experiments/top500_final15_factor_cumulative"
REPORT_DIR = PROJECT_ROOT / "reports/model_results/top500_final15_cumulative"
SEGMENTS = ("2005_2009", "2010_2014", "2015_2019", "2020_2024")
METRICS = (
    "annualized_net_return",
    "annualized_volatility",
    "net_sharpe",
    "average_turnover",
    "average_leverage",
    "annualized_trading_cost",
    "annualized_ex_ante_utility_flow",
)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FINAL / "returns.parquet", FINAL / "monthly_returns.parquet")
    metrics = _comparison_metrics()
    inference = _comparison_inference()
    risk = _comparison_risk()
    metrics.to_csv(FINAL / "top500_core10_vs_final15_metrics.csv", index=False)
    inference.to_csv(FINAL / "top500_core10_vs_final15_utility.csv", index=False)
    risk.to_csv(FINAL / "top500_core10_vs_final15_risk.csv", index=False)
    (REPORT_DIR / "top500_core10_vs_final15.md").write_text(
        _comparison_report(metrics, inference, risk), encoding="utf-8"
    )
    runtime_payload, runtime_markdown = _runtime_report()
    (FINAL / "runtime_summary.json").write_text(
        json.dumps(runtime_payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORT_DIR / "runtime_report.md").write_text(runtime_markdown, encoding="utf-8")
    _annotate_summary(runtime_payload)
    print(f"comparison={REPORT_DIR / 'top500_core10_vs_final15.md'}")
    print(f"runtime={REPORT_DIR / 'runtime_report.md'}")
    print(f"guarded_acceptance={runtime_payload['guarded_acceptance_passed']}")
    return 0


def _comparison_metrics() -> pd.DataFrame:
    core = pd.read_csv(CORE / "metrics.csv")
    final = pd.read_csv(FINAL / "metrics.csv")
    return _wide_compare(core, final, ["method"], list(METRICS))


def _comparison_inference() -> pd.DataFrame:
    columns = [
        "annualized_mean_difference",
        "newey_west_t_stat",
        "bootstrap_annualized_ci_lower",
        "bootstrap_annualized_ci_upper",
        "probability_delta_positive",
        "annual_years_won",
    ]
    core = pd.read_csv(CORE / "statistical_inference.csv")
    final = pd.read_csv(FINAL / "statistical_inference.csv")
    core = core[core["outcome"].eq("utility_flow")]
    final = final[final["outcome"].eq("utility_flow")]
    return _wide_compare(
        core,
        final,
        ["candidate_method", "benchmark_method", "outcome"],
        columns,
    )


def _comparison_risk() -> pd.DataFrame:
    columns = [
        "yearly_ratio_median",
        "yearly_ratio_p10",
        "yearly_ratio_p90",
        "yearly_ratio_max",
        "fraction_years_within_0_6_1_5",
        "years_ratio_above_2",
        "quality_gate_passed",
    ]
    core = pd.read_csv(CORE / "risk_calibration.csv")
    final = pd.read_csv(FINAL / "risk_calibration.csv")
    return _wide_compare(
        core[core["scope"].eq("summary")],
        final[final["scope"].eq("summary")],
        ["method"],
        columns,
    )


def _wide_compare(
    core: pd.DataFrame,
    final: pd.DataFrame,
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    merged = core[keys + values].merge(
        final[keys + values],
        on=keys,
        how="inner",
        suffixes=("_core10", "_final15"),
        validate="one_to_one",
    )
    for value in values:
        if value == "quality_gate_passed":
            continue
        merged[f"{value}_delta_final15_minus_core10"] = (
            pd.to_numeric(merged[f"{value}_final15"], errors="coerce")
            - pd.to_numeric(merged[f"{value}_core10"], errors="coerce")
        )
    return merged


def _comparison_report(
    metrics: pd.DataFrame,
    inference: pd.DataFrame,
    risk: pd.DataFrame,
) -> str:
    compact_metrics = metrics[
        [
            "method",
            "annualized_net_return_core10",
            "annualized_net_return_final15",
            "annualized_net_return_delta_final15_minus_core10",
            "annualized_volatility_core10",
            "annualized_volatility_final15",
            "net_sharpe_core10",
            "net_sharpe_final15",
            "annualized_ex_ante_utility_flow_core10",
            "annualized_ex_ante_utility_flow_final15",
            "annualized_ex_ante_utility_flow_delta_final15_minus_core10",
        ]
    ]
    return (
        "# Top500 Core10 versus Final15\n\n"
        "Both experiments use the same universe, cumulative OOS validation, "
        "factor risk, full matrix m, AUM, gamma, transaction costs and small RF "
        "grid. Differences below are Final15 minus Core10.\n\n"
        "## Metrics\n\n"
        + _markdown(compact_metrics)
        + "\n\n## Utility inference\n\n"
        + _markdown(inference)
        + "\n\n## Risk calibration\n\n"
        + _markdown(risk)
        + "\n\n## Conclusion\n\n"
        "Final15 modestly improves Portfolio-ML point estimates, but does not "
        "strengthen HAC/bootstrap evidence relative to Core10. Static portfolios "
        "take materially more leverage and realized risk, and the guarded risk "
        "gate fails for Static-ML. The Final15 result is therefore informative but "
        "does not yet justify frontier, AUM sensitivity or feature-importance work.\n"
    )


def _runtime_report() -> tuple[dict[str, object], str]:
    rows: list[dict[str, object]] = []
    cache_totals = {
        name: {"requests": 0, "hits": 0}
        for name in (
            "covariance",
            "matrix_m",
            "random_features",
            "transaction_cost",
            "return_predictions",
            "validation_history",
        )
    }
    for segment in SEGMENTS:
        root = FINAL / "segments" / segment
        runtime = json.loads((root / "runtime_profile.json").read_text(encoding="utf-8"))
        gate = json.loads((root / "safety_gate.json").read_text(encoding="utf-8"))
        cache = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "segment": segment,
                "seconds": float(runtime["wall_clock_seconds"]),
                "minutes": float(runtime["wall_clock_seconds"]) / 60.0,
                "peak_working_set_gib": float(runtime["peak_working_set_gib"]),
                "estimated_full_hours": float(
                    runtime["full_runtime_estimate"]["estimated_full_hours"]
                ),
                "safety_gate_passed": bool(gate["passed"]),
            }
        )
        for name in cache_totals:
            requests = int(cache[name]["requests"])
            hits = int(cache[name]["hits"])
            cache_totals[name]["requests"] += requests
            cache_totals[name]["hits"] += hits
    cache_rows = []
    for name, counts in cache_totals.items():
        cache_rows.append(
            {
                "cache": name,
                **counts,
                "hit_rate": counts["hits"] / counts["requests"]
                if counts["requests"]
                else np.nan,
            }
        )
    preflight = json.loads((FINAL / "preflight_diagnostics.json").read_text(encoding="utf-8"))
    final_runtime = json.loads((FINAL / "runtime_profile.json").read_text(encoding="utf-8"))
    final_gate = json.loads((FINAL / "safety_gate.json").read_text(encoding="utf-8"))
    segment_frame = pd.DataFrame(rows)
    total_seconds = float(segment_frame["seconds"].sum()) + float(
        final_runtime["wall_clock_seconds"]
    )
    peak = float(segment_frame["peak_working_set_gib"].max())
    payload = {
        "preflight_passed": bool(preflight["passed"]),
        "preflight_seconds": float(preflight["elapsed_seconds"]),
        "segments": rows,
        "merge_seconds": float(final_runtime["wall_clock_seconds"]),
        "total_full_seconds_including_merge": total_seconds,
        "total_full_hours_including_merge": total_seconds / 3600.0,
        "peak_working_set_gib": peak,
        "cache_hit_rates": cache_rows,
        "guarded_acceptance_passed": bool(final_gate["passed"]),
        "failed_full_checks": [
            name for name, passed in final_gate["checks"].items() if not passed
        ],
        "final_safety_gate": final_gate,
    }
    markdown = (
        "# Top500/Final15 runtime report\n\n"
        f"Preflight: **{'PASS' if payload['preflight_passed'] else 'FAIL'}**. "
        f"Final guarded acceptance: **{'PASS' if payload['guarded_acceptance_passed'] else 'FAIL'}**.\n\n"
        "The completed calculation files are retained, but a failed guarded "
        "acceptance means no next-stage analysis may start automatically.\n\n"
        "## Segment runtime\n\n"
        + _markdown(segment_frame)
        + "\n\n## Cache hit rates\n\n"
        + _markdown(pd.DataFrame(cache_rows))
        + "\n\n## Totals\n\n"
        f"- Segment plus merge runtime: {total_seconds / 3600.0:.3f} hours.\n"
        f"- Peak working set: {peak:.3f} GiB.\n"
        f"- Failed full checks: {payload['failed_full_checks']}.\n"
        "- Failure reason: Static-ML has three full-period years with "
        "realized/predicted volatility above 2; the 2020-2024 segment contains "
        "two such years, exceeding the protected gate allowance of one.\n"
    )
    return payload, markdown


def _annotate_summary(runtime: dict[str, object]) -> None:
    path = REPORT_DIR / "summary.md"
    text = path.read_text(encoding="utf-8")
    marker = "\n## Guarded acceptance\n"
    text = text.split(marker, maxsplit=1)[0].rstrip()
    status = "PASS" if runtime["guarded_acceptance_passed"] else "FAIL"
    text += (
        marker
        + f"\nStatus: **{status}**. Failed checks: "
        + str(runtime["failed_full_checks"])
        + ". The result files are complete, but no next-stage experiment is "
        "authorized by this run. See `runtime_report.md`.\n"
    )
    path.write_text(text, encoding="utf-8")


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    data = frame.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
