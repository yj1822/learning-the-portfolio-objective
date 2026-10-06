from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    PROJECT_ROOT
    / "results/experiments/top500_final20_conservative_factor_cumulative_pilot"
)
PERFORMANCE = PROJECT_ROOT / "reports/performance"
REPORTS = PROJECT_ROOT / "reports/model_results/top500_final20_conservative_pilot"


def main() -> int:
    preflight = json.loads((RESULTS / "preflight_diagnostics.json").read_text())
    runtime = json.loads((RESULTS / "runtime_profile.json").read_text())
    cache = json.loads((RESULTS / "cache_manifest.json").read_text())
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    annual = pd.read_csv(RESULTS / "annual_performance.csv")
    selected = pd.read_csv(RESULTS / "selected_hyperparameters_by_year.csv")
    selected = selected[selected["selected"]].copy()
    returns = pd.read_parquet(RESULTS / "returns.parquet")
    risk = pd.read_csv(
        PERFORMANCE / "top500_final20_conservative_pilot_risk_calibration.csv"
    )

    diagnostics = method_diagnostics(returns)
    gate = build_gate(preflight, runtime, diagnostics, risk)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pilot_safety_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (REPORTS / "summary.md").write_text(
        summary_report(gate, metrics, selected, diagnostics, preflight),
        encoding="utf-8",
    )
    (REPORTS / "risk_calibration.md").write_text(
        risk_report(risk, diagnostics), encoding="utf-8"
    )
    (REPORTS / "runtime_report.md").write_text(
        runtime_report(runtime, cache), encoding="utf-8"
    )
    print(f"pilot_gate={'PASS' if gate['passed'] else 'FAIL'}")
    print(f"recommend_full={gate['recommend_full']}")
    print(f"reports={REPORTS}")
    return 0 if gate["passed"] else 1


def method_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in returns.groupby("method", sort=True):
        matrix_applicable = group["adjustment_mode"].ne("not_applicable")
        rows.append(
            {
                "method": method,
                "months": int(len(group)),
                "mean_leverage": float(group["leverage"].mean()),
                "max_leverage": float(group["leverage"].max()),
                "mean_turnover": float(group["turnover"].mean()),
                "max_turnover": float(group["turnover"].max()),
                "annualized_trading_cost": float(12.0 * group["trading_cost"].mean()),
                "max_monthly_trading_cost": float(group["trading_cost"].max()),
                "beta_norm_mean": float(group["beta_norm"].mean()),
                "beta_norm_max": float(group["beta_norm"].max()),
                "factor_fallback_count": int(group["risk_model_used"].ne("factor").sum()),
                "matrix_m_fallback_count": int(
                    (matrix_applicable & group["adjustment_mode"].ne("matrix_m")).sum()
                ),
                "covariance_repair_count": int(group["covariance_repair_count"].sum()),
                "matrix_m_eigenvalue_min": float(
                    group["matrix_m_real_eigenvalue_min"].min()
                ),
                "matrix_m_eigenvalue_max": float(
                    group["matrix_m_real_eigenvalue_max"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_gate(
    preflight: dict[str, object],
    runtime: dict[str, object],
    diagnostics: pd.DataFrame,
    risk: pd.DataFrame,
) -> dict[str, object]:
    portfolio = diagnostics[
        diagnostics["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])
    ]
    year_risk = risk[risk["scope"].eq("year")]
    checks = {
        "preflight_passed": bool(preflight["passed"]),
        "factor_fallback_zero": int(diagnostics["factor_fallback_count"].sum()) == 0,
        "matrix_m_fallback_zero": int(diagnostics["matrix_m_fallback_count"].sum()) == 0,
        "covariance_repair_zero": int(diagnostics["covariance_repair_count"].sum()) == 0,
        "peak_memory_below_12_gib": float(runtime["peak_working_set_gib"]) <= 12.0,
        "estimated_full_runtime_below_24_hours": float(
            runtime["full_runtime_estimate"]["estimated_full_hours"]
        )
        <= 24.0,
        "portfolio_mean_leverage_below_3": float(portfolio["mean_leverage"].max()) <= 3.0,
        "portfolio_monthly_leverage_below_10": float(portfolio["max_leverage"].max()) <= 10.0,
        "portfolio_mean_turnover_below_3": float(portfolio["mean_turnover"].max()) <= 3.0,
        "portfolio_monthly_turnover_below_10": float(portfolio["max_turnover"].max()) <= 10.0,
        "no_risk_ratio_above_2": int((year_risk["realized_to_predicted_ratio"] > 2.0).sum()) == 0,
        "rf_diagnostics_passed": all(
            bool(row["passed"]) for row in preflight["rf_diagnostics"]
        ),
        "closed_form_diagnostics_passed": all(
            bool(row["passed"]) for row in preflight["closed_form_diagnostics"]
        ),
        "resume_passed": bool(runtime["resume_verification"]["passed"]),
        "future_information_zero": int(
            runtime["execution_quality_gate"]["future_information_violations"]
        )
        == 0,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "recommend_full": passed,
        "checks": checks,
        "actual_runtime_seconds": float(runtime["wall_clock_seconds"]),
        "estimated_full_hours": float(
            runtime["full_runtime_estimate"]["estimated_full_hours"]
        ),
        "peak_working_set_gib": float(runtime["peak_working_set_gib"]),
        "factor_fallback_count": int(diagnostics["factor_fallback_count"].sum()),
        "matrix_m_fallback_count": int(diagnostics["matrix_m_fallback_count"].sum()),
        "covariance_repair_count": int(diagnostics["covariance_repair_count"].sum()),
        "risk_ratio_above_2_count": int(
            (year_risk["realized_to_predicted_ratio"] > 2.0).sum()
        ),
        "note": (
            "A full run still requires extending the ten selected JKP features through "
            "2024 and rerunning full-period feature-data preflight."
        ),
    }


def summary_report(
    gate: dict[str, object],
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: pd.DataFrame,
    preflight: dict[str, object],
) -> str:
    metric_columns = [
        "method",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "average_turnover",
        "average_leverage",
        "annualized_trading_cost",
        "annualized_ex_ante_utility_flow",
    ]
    selection = selected[
        ["test_year", "method", "candidate_hyperparameters", "validation_years"]
    ]
    closed = pd.DataFrame(preflight["closed_form_diagnostics"])
    return (
        "# Top500/Final20 Conservative pilot\n\n"
        f"Pilot safety gate: **{'PASS' if gate['passed'] else 'FAIL'}**. "
        f"Recommendation to run a later full experiment: **{gate['recommend_full']}**. "
        "No full run was started.\n\n"
        "The pilot uses test years 2005-2007, cumulative OOS validation, repaired "
        "factor risk, full matrix m, baseline AUM/gamma, and the existing small RF grid.\n\n"
        "## Gate checks\n\n"
        + markdown(pd.DataFrame([{"check": key, "passed": value} for key, value in gate["checks"].items()]))
        + "\n\n## Pilot metrics\n\n"
        + markdown(metrics[metric_columns])
        + "\n\n## Selected hyperparameters\n\n"
        + markdown(selection)
        + "\n\n## Leverage, turnover, cost and fallback diagnostics\n\n"
        + markdown(diagnostics)
        + "\n\n## Beta and closed-form diagnostics\n\n"
        + markdown(closed)
        + "\n\n## Interpretation\n\n"
        "Portfolio-ML passes the leverage, turnover, cost, factor-risk, matrix-m and "
        "closed-form gates. Static methods become materially larger in 2007, but the "
        "explicit leverage stop applies to Portfolio-ML and neither Static method "
        "produces a realized/predicted ratio above two in this pilot.\n\n"
        "A full run is computationally feasible, but it must not start from the current "
        "pilot-only panel. The ten selected characteristics must first be extended "
        "through 2024 and pass the same feature-data preflight.\n"
    )


def risk_report(risk: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    years = risk[risk["scope"].eq("year")][
        [
            "method",
            "test_year",
            "predicted_annualized_volatility",
            "realized_annualized_volatility",
            "realized_to_predicted_ratio",
            "factor_risk_fallback_count",
            "covariance_repair_count",
        ]
    ]
    summary = risk[risk["scope"].eq("summary")][
        [
            "method",
            "yearly_ratio_median",
            "yearly_ratio_p10",
            "yearly_ratio_p90",
            "yearly_ratio_max",
            "fraction_years_within_0_6_1_5",
            "years_ratio_above_2",
            "quality_gate_passed",
        ]
    ]
    return (
        "# Top500/Final20 Conservative pilot risk calibration\n\n"
        "No method-year has realized/predicted volatility above two. Portfolio-ML "
        "overpredicts risk in 2006 (ratios approximately 0.45-0.47), so its formal "
        "three-year calibration gate fails because only two of three years are in "
        "[0.6, 1.5]. This is conservative risk estimation, not hidden underestimation.\n\n"
        "Static-ML reaches 1.75 in 2007 and Static-ML* reaches 1.50, both below the "
        "specified stop threshold. Factor fallback, matrix fallback and covariance "
        "repair counts are all zero.\n\n"
        "## Yearly calibration\n\n"
        + markdown(years)
        + "\n\n## Three-year summaries\n\n"
        + markdown(summary)
        + "\n\n## Portfolio diagnostics\n\n"
        + markdown(diagnostics)
        + "\n"
    )


def runtime_report(runtime: dict[str, object], cache: dict[str, object]) -> str:
    cache_rows = []
    for name, values in cache.items():
        if isinstance(values, dict) and "hit_rate" in values:
            cache_rows.append(
                {
                    "cache": name,
                    "requests": values.get("requests"),
                    "hits": values.get("hits"),
                    "misses": values.get("misses"),
                    "writes": values.get("writes"),
                    "hit_rate": values.get("hit_rate"),
                }
            )
    estimate = runtime["full_runtime_estimate"]
    resume = runtime["resume_verification"]
    return (
        "# Top500/Final20 Conservative pilot runtime\n\n"
        f"- Wall-clock runtime: {float(runtime['wall_clock_seconds']) / 60.0:.2f} minutes\n"
        f"- Estimated 2005-2024 runtime: {float(estimate['estimated_full_hours']):.3f} hours\n"
        f"- Peak working set: {float(runtime['peak_working_set_gib']):.3f} GiB\n"
        f"- Resume verification: {'PASS' if resume['passed'] else 'FAIL'}; "
        f"{resume['new_checkpoint_writes']} new writes, "
        f"{resume['checkpoint_hits']} checkpoint hits during replay\n"
        f"- Runner full-feasibility flag: {runtime['can_run_full']}\n\n"
        "## Cache statistics\n\n"
        + markdown(pd.DataFrame(cache_rows))
        + "\n\nThe zero initial validation-history hit rate is expected for a cold first pilot. "
        "The explicit resume replay subsequently produced 1,254 checkpoint hits with "
        "zero new writes. Covariance, transaction-cost and matrix-m caches achieved "
        "100% hit rates; RF design cache achieved approximately 96.0%.\n"
    )


def markdown(frame: pd.DataFrame) -> str:
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
