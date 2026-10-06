from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare diagonal and factor-risk experiments")
    parser.add_argument("--diagonal-experiment", required=True)
    parser.add_argument("--factor-experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    diagonal_root = Path(args.diagonal_experiment).resolve()
    factor_root = Path(args.factor_experiment).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods = ["static_ml", "portfolio_ml_linear", "portfolio_ml_rf"]
    columns = [
        "method",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "annualized_trading_cost",
        "average_turnover",
        "average_leverage",
        "empirical_utility",
    ]
    diagonal = pd.read_csv(diagonal_root / "metrics_raw.csv")[columns]
    factor = pd.read_csv(factor_root / "metrics_raw.csv")[columns]
    diagonal = diagonal[diagonal["method"].isin(methods)]
    factor = factor[factor["method"].isin(methods)]
    comparison = diagonal.merge(factor, on="method", suffixes=("_diagonal", "_factor"))
    for column in columns[1:]:
        comparison[f"{column}_factor_minus_diagonal"] = (
            comparison[f"{column}_factor"] - comparison[f"{column}_diagonal"]
        )
    comparison.to_csv(output_root / "diagonal_vs_factor_metrics.csv", index=False)
    diagonal_meta = json.loads((diagonal_root / "metadata.json").read_text(encoding="utf-8"))
    factor_meta = json.loads((factor_root / "metadata.json").read_text(encoding="utf-8"))
    display = comparison.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    headers = display.columns.tolist()
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    table.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    lines = [
        "# Diagonal Versus Repaired Factor Risk",
        "",
        f"- Diagonal experiment: `{diagonal_meta['experiment_id']}`",
        f"- Factor experiment: `{factor_meta['experiment_id']}`",
        f"- Factor risk successful: `{factor_meta['factor_risk_successful']}`",
        f"- Factor fallback count: `{factor_meta['factor_risk_fallback_reason_count']}`",
        "",
        *table,
        "",
    ]
    (output_root / "diagonal_vs_factor_risk.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
