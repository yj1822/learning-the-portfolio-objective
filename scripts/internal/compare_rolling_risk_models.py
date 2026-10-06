from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.metrics import (  # noqa: E402
    annual_performance,
    performance_metrics,
)
from implementable_frontier.models.rolling_robustness import (  # noqa: E402
    RollingRobustnessResult,
    rolling_robustness_analysis,
)


METHOD_ORDER = [
    "return_ml_rank",
    "markowitz_ml",
    "static_ml",
    "static_ml_star",
    "portfolio_ml_linear",
    "portfolio_ml_rf",
]
CONFIG_KEYS_REQUIRED_EQUAL = [
    "universe",
    "feature_set",
    "risk_model",
    "protocol",
    "rolling_yearly",
    "baseline_gamma",
    "baseline_aum",
    "random_seed",
    "return_ml",
    "markowitz_ml",
    "static_ml",
    "portfolio_ml",
    "risk",
    "accounting",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Static-ML* into the diagonal rolling run and compare factor risk"
    )
    parser.add_argument(
        "--diagonal-base",
        default="results/experiments/exp_top100_core10_rolling_2005_2024",
    )
    parser.add_argument(
        "--diagonal-star",
        default="results/experiments/exp_top100_core10_diagonal_static_star_only_2005_2024",
    )
    parser.add_argument(
        "--factor",
        default="results/experiments/exp_top100_core10_factor_rolling_2005_2024_static_star",
    )
    parser.add_argument(
        "--diagonal-output",
        default="results/experiments/exp_top100_core10_diagonal_rolling_2005_2024_static_star_merged",
    )
    parser.add_argument(
        "--comparison-output",
        default="reports/model_results/rolling_factor_vs_diagonal",
    )
    args = parser.parse_args()

    diagonal_base = _resolve(args.diagonal_base)
    diagonal_star = _resolve(args.diagonal_star)
    factor_dir = _resolve(args.factor)
    diagonal_output = _resolve(args.diagonal_output)
    comparison_output = _resolve(args.comparison_output)
    _require_empty_or_missing(diagonal_output)
    comparison_output.mkdir(parents=True, exist_ok=True)

    current_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "model_top100_core10_rolling.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_config = yaml.safe_load(
        (diagonal_base / "config_used.yaml").read_text(encoding="utf-8")
    )
    for key in CONFIG_KEYS_REQUIRED_EQUAL:
        if base_config.get(key) != current_config.get(key):
            raise ValueError(f"Diagonal base/current configuration mismatch for {key}")

    base_returns = pd.read_parquet(diagonal_base / "returns.parquet")
    star_returns = pd.read_parquet(diagonal_star / "returns.parquet")
    factor_returns = pd.read_parquet(factor_dir / "returns.parquet")
    diagonal_returns = pd.concat([base_returns, star_returns], ignore_index=True)
    _validate_complete(diagonal_returns, "diagonal")
    _validate_complete(factor_returns, "factor")

    base_selected = pd.read_csv(diagonal_base / "selected_hyperparameters_by_year.csv")
    star_selected_all = pd.read_csv(
        diagonal_star / "selected_hyperparameters_by_year.csv"
    )
    _validate_return_selection_consistency(base_selected, star_selected_all)
    star_selected = star_selected_all[
        star_selected_all["method"].eq("static_ml_star")
    ].copy()
    if len(star_selected) != 20:
        raise ValueError("Static-ML* must have exactly 20 selected annual parameter rows")
    diagonal_selected = pd.concat([base_selected, star_selected], ignore_index=True)

    base_validation = pd.read_parquet(diagonal_base / "validation_selection.parquet")
    star_validation = pd.read_parquet(diagonal_star / "validation_selection.parquet")
    star_validation = star_validation[star_validation["method"].eq("static_ml_star")]
    if len(star_validation) != 20 * 60:
        raise ValueError("Static-ML* validation must contain 60 candidates for each year")
    diagonal_validation = pd.concat(
        [base_validation, star_validation], ignore_index=True
    )

    gamma = float(current_config["baseline_gamma"])
    diagonal_metrics = _ordered(performance_metrics(diagonal_returns, gamma))
    diagonal_annual = _ordered(
        annual_performance(diagonal_returns, gamma), by_year=True
    )
    diagonal_robustness = rolling_robustness_analysis(
        diagonal_returns,
        gamma=gamma,
        seed=int(current_config["random_seed"]),
        newey_west_lags=int(current_config["robustness"]["newey_west_lags"]),
        bootstrap_block_months=int(
            current_config["robustness"]["bootstrap_block_months"]
        ),
        bootstrap_replications=int(
            current_config["robustness"]["bootstrap_replications"]
        ),
    )
    _write_merged_diagonal(
        diagonal_output,
        diagonal_base,
        diagonal_star,
        current_config,
        diagonal_returns,
        diagonal_metrics,
        diagonal_annual,
        diagonal_selected,
        diagonal_validation,
        diagonal_robustness,
    )

    factor_metrics = _ordered(pd.read_csv(factor_dir / "metrics_raw.csv"))
    factor_annual = _ordered(
        pd.read_csv(factor_dir / "annual_performance.csv"), by_year=True
    )
    factor_selected = pd.read_csv(
        factor_dir / "selected_hyperparameters_by_year.csv"
    )
    factor_robustness = _read_robustness(factor_dir)

    full_comparison = _compare_frames(
        diagonal_metrics,
        factor_metrics,
        keys=["method"],
        value_columns=[
            "annualized_net_return",
            "annualized_volatility",
            "net_sharpe",
            "empirical_utility",
            "annualized_ex_ante_utility_flow",
            "annualized_trading_cost",
            "average_turnover",
            "average_leverage",
        ],
    )
    annual_comparison = _compare_frames(
        diagonal_annual,
        factor_annual,
        keys=["method", "test_year"],
        value_columns=[
            "annualized_net_return",
            "annualized_volatility",
            "net_sharpe",
            "empirical_utility",
            "annualized_ex_ante_utility_flow",
        ],
    )
    risk_full_comparison = _compare_frames(
        diagonal_robustness.risk_calibration_full,
        factor_robustness.risk_calibration_full,
        keys=["method"],
        value_columns=[
            "predicted_annualized_volatility",
            "realized_annualized_gross_volatility",
            "realized_annualized_net_volatility",
            "realized_to_predicted_gross_ratio",
            "gross_volatility_calibration_error",
        ],
    )
    risk_annual_comparison = _compare_frames(
        diagonal_robustness.risk_calibration_by_year,
        factor_robustness.risk_calibration_by_year,
        keys=["method", "test_year"],
        value_columns=[
            "predicted_annualized_volatility",
            "realized_annualized_gross_volatility",
            "realized_to_predicted_gross_ratio",
            "gross_volatility_calibration_error",
        ],
    )

    full_comparison.to_csv(
        comparison_output / "full_period_metrics_diagonal_vs_factor.csv", index=False
    )
    annual_comparison.to_csv(
        comparison_output / "annual_performance_diagonal_vs_factor.csv", index=False
    )
    risk_full_comparison.to_csv(
        comparison_output / "risk_calibration_full_diagonal_vs_factor.csv", index=False
    )
    risk_annual_comparison.to_csv(
        comparison_output / "risk_calibration_by_year_diagonal_vs_factor.csv",
        index=False,
    )
    diagonal_selected.to_csv(
        comparison_output / "selected_hyperparameters_diagonal.csv", index=False
    )
    factor_selected.to_csv(
        comparison_output / "selected_hyperparameters_factor.csv", index=False
    )
    factor_robustness.difference_statistics.to_csv(
        comparison_output / "factor_method_difference_statistics.csv", index=False
    )
    factor_robustness.advantage_by_year.to_csv(
        comparison_output / "factor_portfolio_ml_advantage_by_year.csv", index=False
    )
    factor_robustness.advantage_concentration.to_csv(
        comparison_output / "factor_advantage_concentration.csv", index=False
    )
    factor_robustness.crisis_period_analysis.to_csv(
        comparison_output / "factor_crisis_period_analysis.csv", index=False
    )

    summary = _comparison_markdown(
        diagonal_metrics,
        factor_metrics,
        full_comparison,
        factor_annual,
        factor_robustness,
        risk_full_comparison,
        diagonal_output,
        factor_dir,
    )
    summary_path = comparison_output / "rolling_factor_vs_diagonal_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"diagonal_merged={diagonal_output}")
    print(f"comparison_report={summary_path}")
    print(full_comparison.to_string(index=False))
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_empty_or_missing(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_complete(returns: pd.DataFrame, risk_model: str) -> None:
    counts = returns.groupby("method")["eom"].nunique()
    if set(counts.index) != set(METHOD_ORDER) or not counts.eq(240).all():
        raise ValueError(f"Incomplete methods/months for {risk_model}: {counts.to_dict()}")
    years = sorted(pd.to_numeric(returns["test_year"], errors="raise").astype(int).unique())
    if years != list(range(2005, 2025)):
        raise ValueError(f"Unexpected test years for {risk_model}: {years}")
    if set(returns["risk_model_used"].dropna().unique()) != {risk_model}:
        raise ValueError(f"Unexpected risk_model_used values for {risk_model}")


def _validate_return_selection_consistency(
    base_selected: pd.DataFrame, star_selected: pd.DataFrame
) -> None:
    columns = ["year", "candidate_hyperparameters", "validation_utility"]
    base = base_selected[base_selected["method"].eq("return_ml")][columns].reset_index(
        drop=True
    )
    star = star_selected[star_selected["method"].eq("return_ml")][columns].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(base, star, check_exact=False, rtol=1.0e-12)


def _ordered(frame: pd.DataFrame, *, by_year: bool = False) -> pd.DataFrame:
    result = frame.copy()
    result["_method_order"] = pd.Categorical(
        result["method"], categories=METHOD_ORDER, ordered=True
    )
    sort_columns = ["_method_order", "test_year"] if by_year else ["_method_order"]
    return result.sort_values(sort_columns).drop(columns="_method_order").reset_index(drop=True)


def _write_merged_diagonal(
    output: Path,
    base: Path,
    star: Path,
    config: dict[str, object],
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    selected: pd.DataFrame,
    validation: pd.DataFrame,
    robustness: RollingRobustnessResult,
) -> None:
    weights = pd.concat(
        [pd.read_parquet(base / "weights.parquet"), pd.read_parquet(star / "weights.parquet")],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            pd.read_parquet(base / "predictions.parquet"),
            pd.read_parquet(star / "predictions.parquet"),
        ],
        ignore_index=True,
    )
    (output / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": output.name,
                "construction": "exact merge of configuration-equivalent diagonal rolling runs",
                "base_experiment": str(base),
                "static_ml_star_experiment": str(star),
                "methods": METHOD_ORDER,
                "risk_model_requested": "diagonal",
                "risk_models_used": ["diagonal"],
                "test_years": [2005, 2024],
                "months_per_method": 240,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    returns.to_parquet(output / "returns.parquet", index=False)
    weights.to_parquet(output / "weights.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    validation.to_parquet(output / "validation_selection.parquet", index=False)
    metrics.to_csv(output / "metrics_raw.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    annual.to_csv(output / "annual_performance.csv", index=False)
    selected.to_csv(output / "selected_hyperparameters_by_year.csv", index=False)
    _write_robustness(output, robustness)
    lines = [
        "# Top100/Core10 Diagonal Rolling-Year Summary",
        "",
        "- Test years: 2005-2024",
        "- Markowitz-ML is an unstable diagnostic and is excluded from conclusions.",
        "- Static-ML* uses u/v/k selected only on each test year's prior validation window.",
        "",
        "## Full-Period Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Monthly Difference Inference",
        "",
        _markdown_table(robustness.difference_statistics),
        "",
        "## Risk Calibration",
        "",
        _markdown_table(robustness.risk_calibration_full),
        "",
    ]
    (output / "rolling_yearly_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_robustness(output: Path, result: RollingRobustnessResult) -> None:
    result.monthly_differences.to_parquet(
        output / "monthly_method_differences.parquet", index=False
    )
    result.difference_statistics.to_csv(
        output / "method_difference_statistics.csv", index=False
    )
    result.advantage_by_year.to_csv(
        output / "portfolio_ml_advantage_by_year.csv", index=False
    )
    result.advantage_concentration.to_csv(
        output / "advantage_concentration.csv", index=False
    )
    result.crisis_period_analysis.to_csv(
        output / "crisis_period_analysis.csv", index=False
    )
    result.risk_calibration_full.to_csv(
        output / "risk_calibration_full.csv", index=False
    )
    result.risk_calibration_by_year.to_csv(
        output / "risk_calibration_by_year.csv", index=False
    )


def _read_robustness(path: Path) -> RollingRobustnessResult:
    return RollingRobustnessResult(
        pd.read_parquet(path / "monthly_method_differences.parquet"),
        pd.read_csv(path / "method_difference_statistics.csv"),
        pd.read_csv(path / "portfolio_ml_advantage_by_year.csv"),
        pd.read_csv(path / "advantage_concentration.csv"),
        pd.read_csv(path / "crisis_period_analysis.csv"),
        pd.read_csv(path / "risk_calibration_full.csv"),
        pd.read_csv(path / "risk_calibration_by_year.csv"),
    )


def _compare_frames(
    diagonal: pd.DataFrame,
    factor: pd.DataFrame,
    *,
    keys: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    left = diagonal[[*keys, *value_columns]].rename(
        columns={column: f"diagonal_{column}" for column in value_columns}
    )
    right = factor[[*keys, *value_columns]].rename(
        columns={column: f"factor_{column}" for column in value_columns}
    )
    result = left.merge(right, on=keys, how="inner", validate="one_to_one")
    for column in value_columns:
        result[f"factor_minus_diagonal_{column}"] = (
            result[f"factor_{column}"] - result[f"diagonal_{column}"]
        )
    return result


def _comparison_markdown(
    diagonal_metrics: pd.DataFrame,
    factor_metrics: pd.DataFrame,
    full_comparison: pd.DataFrame,
    factor_annual: pd.DataFrame,
    factor_robustness: RollingRobustnessResult,
    risk_comparison: pd.DataFrame,
    diagonal_dir: Path,
    factor_dir: Path,
) -> str:
    star = factor_metrics.set_index("method").loc["static_ml_star"]
    linear = factor_metrics.set_index("method").loc["portfolio_ml_linear"]
    rf = factor_metrics.set_index("method").loc["portfolio_ml_rf"]
    annual = factor_annual.set_index(["method", "test_year"])
    years = list(range(2005, 2025))
    linear_wins = int(
        (
            annual.loc[("portfolio_ml_linear", years), "annualized_net_return"].to_numpy()
            > annual.loc[("static_ml_star", years), "annualized_net_return"].to_numpy()
        ).sum()
    )
    rf_wins = int(
        (
            annual.loc[("portfolio_ml_rf", years), "annualized_net_return"].to_numpy()
            > annual.loc[("static_ml_star", years), "annualized_net_return"].to_numpy()
        ).sum()
    )
    lines = [
        "# Top100/Core10 Rolling Factor Versus Diagonal",
        "",
        f"- Diagonal merged experiment: `{diagonal_dir}`",
        f"- Factor experiment: `{factor_dir}`",
        "- Window: 2005-2024, 240 months per method",
        "- Markowitz-ML remains an unstable diagnostic and is excluded from conclusions.",
        "",
        "## Factor Full-Period Metrics",
        "",
        _markdown_table(factor_metrics),
        "",
        "## Diagonal Full-Period Metrics",
        "",
        _markdown_table(diagonal_metrics),
        "",
        "## Factor Minus Diagonal",
        "",
        _markdown_table(full_comparison),
        "",
        "## Portfolio-ML Monthly Difference Inference Under Factor Risk",
        "",
        _markdown_table(factor_robustness.difference_statistics),
        "",
        "## Factor-Risk Advantage Concentration",
        "",
        _markdown_table(factor_robustness.advantage_concentration),
        "",
        "## Factor-Risk Crisis Periods",
        "",
        _markdown_table(factor_robustness.crisis_period_analysis),
        "",
        "## Ex-Ante Versus Realized Risk",
        "",
        _markdown_table(risk_comparison),
        "",
        "## Decision",
        "",
        f"- Portfolio-ML Linear factor-risk net return minus Static-ML*: {(linear['annualized_net_return'] - star['annualized_net_return']):.2%}; annual wins {linear_wins}/20.",
        f"- Portfolio-ML RF factor-risk net return minus Static-ML*: {(rf['annualized_net_return'] - star['annualized_net_return']):.2%}; annual wins {rf_wins}/20.",
        f"- Linear is more robust than RF under factor risk: net Sharpe {linear['net_sharpe']:.3f} versus {rf['net_sharpe']:.3f}, empirical utility {linear['empirical_utility']:.2%} versus {rf['empirical_utility']:.2%}.",
        "- Portfolio-ML does not beat Static-ML* under factor-risk rolling; do not expand to top300/core10 yet.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: f"{value:.6g}" if pd.notna(value) else ""
        )
    rows = [list(map(str, display.columns)), ["---"] * len(display.columns)]
    rows.extend(
        [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
