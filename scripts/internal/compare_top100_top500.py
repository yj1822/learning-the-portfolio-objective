from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOP100 = PROJECT_ROOT / "results/experiments/top100_core10_factor_cumulative"
TOP500 = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT = (
    PROJECT_ROOT
    / "reports/model_results/top500_core10_cumulative/top100_vs_top500.md"
)

METRIC_COLUMNS = (
    "annualized_net_return",
    "annualized_volatility",
    "net_sharpe",
    "average_turnover",
    "average_leverage",
    "annualized_trading_cost",
    "annualized_ex_ante_utility_flow",
)

INFERENCE_COLUMNS = (
    "annualized_mean_difference",
    "newey_west_t_stat",
    "bootstrap_annualized_ci_lower",
    "bootstrap_annualized_ci_upper",
    "probability_delta_positive",
)


def main() -> int:
    top100_metrics = pd.read_csv(TOP100 / "metrics.csv")
    top500_metrics = pd.read_csv(TOP500 / "metrics.csv")
    metrics = _compare(
        top100_metrics,
        top500_metrics,
        keys=["method"],
        values=list(METRIC_COLUMNS),
    )
    metrics.to_csv(TOP500 / "top100_vs_top500_metrics.csv", index=False)

    top100_inference = pd.read_csv(TOP100 / "statistical_inference.csv")
    top500_inference = pd.read_csv(TOP500 / "statistical_inference.csv")
    utility100 = top100_inference[top100_inference["outcome"].eq("utility_flow")]
    utility500 = top500_inference[top500_inference["outcome"].eq("utility_flow")]
    inference = _compare(
        utility100,
        utility500,
        keys=["candidate_method", "benchmark_method", "outcome"],
        values=list(INFERENCE_COLUMNS),
    )
    inference.to_csv(TOP500 / "top100_vs_top500_utility_inference.csv", index=False)

    top100_risk = _risk_summary(TOP100)
    top500_risk = _risk_summary(TOP500)
    risk = _compare(
        top100_risk,
        top500_risk,
        keys=["method"],
        values=[
            "yearly_ratio_median",
            "fraction_years_within_0_6_1_5",
            "yearly_ratio_max",
            "quality_gate_passed",
        ],
    )
    risk.to_csv(TOP500 / "top100_vs_top500_risk_calibration.csv", index=False)

    metadata100 = json.loads((TOP100 / "metadata.json").read_text(encoding="utf-8"))
    metadata500 = json.loads((TOP500 / "metadata.json").read_text(encoding="utf-8"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Top100 vs Top500 Core10\n\n"
        "Both experiments use cumulative OOS validation, repaired factor risk, "
        "full matrix m, the same Core10 features, gamma, AUM, transaction-cost "
        "model, and 2005-2024 test years. Markowitz-ML is excluded from the "
        "cross-universe comparison.\n\n"
        "## Full-period metrics\n\n"
        + _markdown(metrics)
        + "\n\n## Utility inference\n\n"
        + _markdown(inference)
        + "\n\n## Risk calibration\n\n"
        + _markdown(risk)
        + "\n\n## Fallbacks\n\n"
        + _markdown(
            pd.DataFrame(
                [
                    {
                        "universe": "top100",
                        "factor_fallback_count": metadata100.get("factor_fallback_count"),
                        "matrix_m_fallback_count": metadata100.get("matrix_m_fallback_count"),
                    },
                    {
                        "universe": "top500",
                        "factor_fallback_count": metadata500.get("factor_fallback_count"),
                        "matrix_m_fallback_count": metadata500.get("matrix_m_fallback_count"),
                    },
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"metrics={TOP500 / 'top100_vs_top500_metrics.csv'}")
    print(f"utility={TOP500 / 'top100_vs_top500_utility_inference.csv'}")
    print(f"report={REPORT}")
    return 0


def _risk_summary(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "risk_calibration.csv")
    return frame[frame["scope"].eq("summary")].copy()


def _compare(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: list[str],
    values: list[str],
) -> pd.DataFrame:
    left_columns = keys + [value for value in values if value in left]
    right_columns = keys + [value for value in values if value in right]
    merged = left[left_columns].merge(
        right[right_columns],
        on=keys,
        how="inner",
        suffixes=("_top100", "_top500"),
        validate="one_to_one",
    )
    for value in values:
        left_name = f"{value}_top100"
        right_name = f"{value}_top500"
        if left_name not in merged or right_name not in merged:
            continue
        if pd.api.types.is_bool_dtype(merged[left_name]) or value == "quality_gate_passed":
            continue
        merged[f"{value}_delta_top500_minus_top100"] = (
            pd.to_numeric(merged[right_name], errors="coerce")
            - pd.to_numeric(merged[left_name], errors="coerce")
        )
    return merged


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    data = frame.copy()
    for column in data.select_dtypes(include="number").columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
