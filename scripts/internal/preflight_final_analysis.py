from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.formal_analysis import (  # noqa: E402
    monthly_utility_differences,
)
from implementable_frontier.models.metrics import performance_metrics  # noqa: E402


RESULT_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
FINAL_ROOT = PROJECT_ROOT / "reports/final_results"
OUTPUT_ROOT = PROJECT_ROOT / "reports/final_analysis/preflight"
WORKBOOK_EXTRACT = (
    PROJECT_ROOT
    / ".artifact_tool_final_analysis/final_performance_workbook_extract.json"
)
DATA_ROOT = PROJECT_ROOT / "data/processed/us_equity_ml"
TOLERANCE = 1.0e-6

METHOD_LABELS = {
    "return_ml_rank": "Return-ML Rank",
    "static_ml": "Static-ML",
    "static_ml_star": "Static-ML*",
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}
LABEL_METHODS = {value: key for key, value in METHOD_LABELS.items()}
METRIC_MAP = {
    "annualized_net_return": "Net return",
    "annualized_volatility": "Volatility",
    "net_sharpe": "Sharpe",
    "average_turnover": "Turnover",
    "average_leverage": "Leverage",
    "annualized_trading_cost": "Trading cost",
    "annualized_ex_ante_utility_flow": "Utility",
}
INFERENCE_PAIRS = {
    "Portfolio-ML Linear vs Static-ML": (
        "portfolio_ml_linear",
        "static_ml",
    ),
    "Portfolio-ML Linear vs Static-ML*": (
        "portfolio_ml_linear",
        "static_ml_star",
    ),
    "Portfolio-ML RF vs Static-ML": ("portfolio_ml_rf", "static_ml"),
    "Portfolio-ML RF vs Static-ML*": (
        "portfolio_ml_rf",
        "static_ml_star",
    ),
    "Portfolio-ML RF vs Portfolio-ML Linear": (
        "portfolio_ml_rf",
        "portfolio_ml_linear",
    ),
}


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    returns = pd.read_parquet(RESULT_ROOT / "returns.parquet")
    weights = pd.read_parquet(RESULT_ROOT / "weights.parquet")
    validation = pd.read_parquet(RESULT_ROOT / "validation_selection.parquet")
    reported_metrics = pd.read_csv(RESULT_ROOT / "metrics.csv")
    reported_inference = pd.read_csv(RESULT_ROOT / "statistical_inference.csv")
    metadata = json.loads((RESULT_ROOT / "metadata.json").read_text(encoding="utf-8"))
    workbook = json.loads(WORKBOOK_EXTRACT.read_text(encoding="utf-8"))
    config = yaml.safe_load((RESULT_ROOT / "config_used.yaml").read_text(encoding="utf-8"))
    data_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/data_us_equity_ml.yaml").read_text(encoding="utf-8")
    )

    _check_locked_spec(config, checks)
    metric_comparisons = _check_metrics(
        returns, reported_metrics, workbook, checks
    )
    inference_comparisons = _check_inference(
        returns, reported_inference, workbook, checks
    )
    markdown_checks = _check_markdown_display(
        reported_metrics, reported_inference, checks
    )
    safety = _check_safety(returns, validation, metadata, checks)
    availability = _check_file_availability(
        returns, weights, data_config, checks
    )

    passed = bool(all(bool(item["passed"]) for item in checks))
    payload = {
        "passed": passed,
        "tolerance": TOLERANCE,
        "specification": {
            "universe": "top500",
            "feature_set": "core10",
            "risk_model": "factor",
            "protocol": "cumulative_oos_validation",
            "gamma": 10.0,
            "aum": 1_000_000_000.0,
        },
        "metric_comparisons": metric_comparisons,
        "utility_inference_comparisons": inference_comparisons,
        "markdown_display_checks": markdown_checks,
        "safety": safety,
        "availability": availability,
        "checks": checks,
        "failed_checks": [item["check"] for item in checks if not item["passed"]],
    }
    (OUTPUT_ROOT / "final_analysis_preflight.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (OUTPUT_ROOT / "final_analysis_preflight.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "failed_checks": payload["failed_checks"]}, indent=2))
    return 0 if passed else 2


def _check_locked_spec(config: dict[str, object], checks: list[dict[str, object]]) -> None:
    expected = {
        "universe": "top500",
        "feature_set": "core10",
        "risk_model": "factor",
        "protocol": "cumulative_oos_validation",
        "baseline_gamma": 10,
        "baseline_aum": 1_000_000_000,
    }
    for key, value in expected.items():
        actual = config.get(key)
        _record(checks, f"locked_config_{key}", actual == value, actual=actual, expected=value)
    _record(
        checks,
        "no_diagonal_factor_fallback",
        config["risk"].get("allow_factor_fallback") is False,
        actual=config["risk"].get("allow_factor_fallback"),
        expected=False,
    )
    _record(
        checks,
        "no_scalar_rho_fallback",
        config["portfolio_ml"].get("allow_scalar_rho_fallback") is False,
        actual=config["portfolio_ml"].get("allow_scalar_rho_fallback"),
        expected=False,
    )


def _check_metrics(
    returns: pd.DataFrame,
    reported: pd.DataFrame,
    workbook: dict[str, object],
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    recomputed = performance_metrics(returns, gamma=10.0).set_index("method")
    source = reported.set_index("method")
    workbook_rows = workbook["main_metrics"]
    workbook_headers = workbook_rows[0]
    workbook_frame = pd.DataFrame(workbook_rows[1:], columns=workbook_headers)
    workbook_frame["method"] = workbook_frame["Method"].map(LABEL_METHODS)
    workbook_frame = workbook_frame.set_index("method")
    comparisons: list[dict[str, object]] = []
    for method in METHOD_LABELS:
        for source_column, workbook_column in METRIC_MAP.items():
            source_value = float(source.loc[method, source_column])
            recomputed_value = float(recomputed.loc[method, source_column])
            workbook_value = float(workbook_frame.loc[method, workbook_column])
            source_error = abs(recomputed_value - source_value)
            workbook_error = abs(workbook_value - source_value)
            row = {
                "method": method,
                "metric": source_column,
                "reported": source_value,
                "recomputed": recomputed_value,
                "workbook": workbook_value,
                "recomputed_absolute_error": source_error,
                "workbook_absolute_error": workbook_error,
                "passed": source_error <= TOLERANCE and workbook_error <= TOLERANCE,
            }
            comparisons.append(row)
    max_error = max(
        max(item["recomputed_absolute_error"], item["workbook_absolute_error"])
        for item in comparisons
    )
    _record(
        checks,
        "baseline_metrics_match_recomputed_and_excel",
        all(item["passed"] for item in comparisons),
        max_absolute_error=max_error,
        tolerance=TOLERANCE,
    )
    _record(
        checks,
        "final_results_workbook_formula_errors_zero",
        "matched 0 entries" in workbook["formula_error_scan"],
        scan=workbook["formula_error_scan"],
    )
    return comparisons


def _check_inference(
    returns: pd.DataFrame,
    reported: pd.DataFrame,
    workbook: dict[str, object],
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    utility = reported[reported["outcome"].eq("utility_flow")].set_index(
        ["candidate_method", "benchmark_method"]
    )
    monthly = monthly_utility_differences(returns, gamma=10.0)
    workbook_rows = workbook["main_utility_inference"]
    workbook_frame = pd.DataFrame(workbook_rows[1:], columns=workbook_rows[0]).set_index(
        "Comparison"
    )
    comparisons: list[dict[str, object]] = []
    for label, pair in INFERENCE_PAIRS.items():
        group = monthly[
            monthly["candidate_method"].eq(pair[0])
            & monthly["benchmark_method"].eq(pair[1])
        ]
        recomputed = 12.0 * float(group["delta_utility_flow"].mean())
        source = float(utility.loc[pair, "annualized_mean_difference"])
        workbook_value = float(
            workbook_frame.loc[label, "Annual utility difference"]
        )
        accounting_error = float(
            np.nanmax(
                np.abs(
                    group["delta_utility_flow"]
                    - (group["delta_net_return"] - group["delta_risk_penalty"])
                )
            )
        )
        row = {
            "comparison": label,
            "reported": source,
            "recomputed": recomputed,
            "workbook": workbook_value,
            "recomputed_absolute_error": abs(recomputed - source),
            "workbook_absolute_error": abs(workbook_value - source),
            "utility_accounting_max_error": accounting_error,
        }
        row["passed"] = bool(
            row["recomputed_absolute_error"] <= TOLERANCE
            and row["workbook_absolute_error"] <= TOLERANCE
            and accounting_error <= 1.0e-10
        )
        comparisons.append(row)
    max_error = max(
        max(item["recomputed_absolute_error"], item["workbook_absolute_error"])
        for item in comparisons
    )
    _record(
        checks,
        "utility_inference_matches_recomputed_and_excel",
        all(item["passed"] for item in comparisons),
        max_absolute_error=max_error,
        tolerance=TOLERANCE,
    )
    return comparisons


def _check_markdown_display(
    metrics: pd.DataFrame,
    inference: pd.DataFrame,
    checks: list[dict[str, object]],
) -> dict[str, object]:
    text = (FINAL_ROOT / "final_performance_tables.md").read_text(encoding="utf-8")
    expected_metric_rows = []
    source = metrics.set_index("method")
    for method, label in METHOD_LABELS.items():
        row = source.loc[method]
        expected_metric_rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{row['annualized_net_return']:.2%}",
                    f"{row['annualized_volatility']:.2%}",
                    f"{row['net_sharpe']:.3f}",
                    f"{row['average_turnover']:.2%}",
                    f"{row['average_leverage']:.3f}",
                    f"{row['annualized_trading_cost']:.2%}",
                    f"{row['annualized_ex_ante_utility_flow']:.2%}",
                ]
            )
            + " |"
        )
    utility = inference[inference["outcome"].eq("utility_flow")].set_index(
        ["candidate_method", "benchmark_method"]
    )
    expected_inference_rows = []
    for label, pair in INFERENCE_PAIRS.items():
        row = utility.loc[pair]
        expected_inference_rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{row['annualized_mean_difference']:.2%}",
                    f"{row['newey_west_t_stat']:.3f}",
                    f"{row['bootstrap_annualized_ci_lower']:.2%}",
                    f"{row['bootstrap_annualized_ci_upper']:.2%}",
                    f"{row['probability_delta_positive']:.2%}",
                    f"{int(row['annual_years_won'])}/{int(row['annual_years_total'])}",
                ]
            )
            + " |"
        )
    missing = [
        row for row in [*expected_metric_rows, *expected_inference_rows] if row not in text
    ]
    passed = not missing
    _record(
        checks,
        "final_performance_markdown_matches_display_precision",
        passed,
        missing_rows=missing,
    )
    return {
        "passed": passed,
        "metric_rows_checked": len(expected_metric_rows),
        "inference_rows_checked": len(expected_inference_rows),
        "missing_rows": missing,
        "note": "Markdown is presentation-rounded; Excel is the exact-value tolerance source.",
    }


def _check_safety(
    returns: pd.DataFrame,
    validation: pd.DataFrame,
    metadata: dict[str, object],
    checks: list[dict[str, object]],
) -> dict[str, object]:
    future = int(validation["future_information_used"].fillna(False).sum())
    cutoff = int(
        (
            pd.to_numeric(validation["latest_validation_year"], errors="coerce")
            >= pd.to_numeric(validation["test_year"], errors="coerce")
        ).sum()
    )
    wrong_risk = int((~returns["risk_model_used"].astype(str).eq("factor")).sum())
    diagonal = int(returns["factor_covariance_diagonal_fallback"].fillna(False).sum())
    repairs = int(pd.to_numeric(returns["covariance_repair_count"], errors="coerce").sum())
    pml = returns[returns["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])]
    wrong_m = int((~pml["adjustment_mode"].astype(str).eq("matrix_m")).sum())
    factor_fallback = max(int(metadata.get("factor_fallback_count", 0)), wrong_risk, diagonal)
    matrix_fallback = max(int(metadata.get("matrix_m_fallback_count", 0)), wrong_m)
    values = {
        "future_information_violations": future,
        "validation_cutoff_violations": cutoff,
        "factor_fallback_count": factor_fallback,
        "matrix_m_fallback_count": matrix_fallback,
        "covariance_repair_count": repairs,
    }
    for key, value in values.items():
        _record(checks, f"{key}_zero", value == 0, actual=value, expected=0)
    return values


def _check_file_availability(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    data_config: dict[str, object],
    checks: list[dict[str, object]],
) -> dict[str, object]:
    core_features = list(data_config["feature_sets"]["core10"])
    feature_panel = DATA_ROOT / "master_panel.parquet"
    cost_inputs = DATA_ROOT / "cost_inputs.parquet"
    risk_root = DATA_ROOT / "risk_repaired"
    required_risk = [
        risk_root / "factor_exposures.parquet",
        risk_root / "factor_cov_monthly.parquet",
        risk_root / "idio_var_monthly.parquet",
        risk_root / "factor_returns_daily.parquet",
    ]
    panel_columns = set(pd.read_parquet(feature_panel).columns)
    cost_columns = set(pd.read_parquet(cost_inputs).columns)
    availability = {
        "monthly_weights": _columns(weights, ["eom", "permno", "method", "weight"]),
        "grown_inherited_weights": _columns(weights, ["inherited_weight"])
        and int(weights["inherited_weight"].notna().sum()) == len(weights),
        "aim_portfolios_portfolio_ml": _columns(weights, ["aim_weight"])
        and int(
            weights[weights["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])][
                "aim_weight"
            ].notna().sum()
        )
        == int(weights["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"]).sum()),
        "monthly_net_gross_returns": _columns(
            returns, ["gross_return", "net_return", "trading_cost", "utility_flow"]
        ),
        "predicted_and_realized_risk_inputs": _columns(
            returns, ["ex_ante_monthly_variance", "net_return"]
        ),
        "feature_panel": feature_panel.exists()
        and all(f"rank_{feature}" in panel_columns for feature in core_features),
        "factor_risk_inputs": all(path.exists() for path in required_risk),
        "adv_lambda_inputs": cost_inputs.exists()
        and {"adv_6m", "lambda_adv", "in_top500"}.issubset(cost_columns),
        "stock_level_trade_details": _columns(weights, ["trade"]),
        "monthly_liquidity_diagnostics": _columns(
            returns, ["weighted_lambda_exposure", "max_lambda_adv"]
        ),
    }
    for key, value in availability.items():
        _record(checks, f"available_{key}", bool(value), actual=bool(value), expected=True)
    availability.update(
        {
            "core10_features": core_features,
            "weights_rows": int(len(weights)),
            "returns_rows": int(len(returns)),
            "aim_portfolio_scope": "Portfolio-ML Linear/RF only; not defined for Rank or Static methods",
            "risk_files": [str(path.relative_to(PROJECT_ROOT)) for path in required_risk],
        }
    )
    return availability


def _columns(frame: pd.DataFrame, required: list[str]) -> bool:
    return set(required).issubset(frame.columns)


def _record(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    **details: object,
) -> None:
    checks.append({"check": name, "passed": bool(passed), **details})


def _markdown_report(payload: dict[str, object]) -> str:
    metric_summary = pd.DataFrame(payload["metric_comparisons"])[
        [
            "method",
            "metric",
            "reported",
            "recomputed_absolute_error",
            "workbook_absolute_error",
            "passed",
        ]
    ]
    inference_summary = pd.DataFrame(payload["utility_inference_comparisons"])[
        [
            "comparison",
            "reported",
            "recomputed_absolute_error",
            "workbook_absolute_error",
            "utility_accounting_max_error",
            "passed",
        ]
    ]
    checks = pd.DataFrame(payload["checks"])[["check", "passed"]]
    availability = pd.DataFrame(
        [
            {"input": key, "available": value}
            for key, value in payload["availability"].items()
            if isinstance(value, bool)
        ]
    )
    return (
        "# Final analysis preflight\n\n"
        f"Status: **{'PASS' if payload['passed'] else 'FAIL'}**. "
        "This preflight performs no model training, portfolio reconstruction or feature selection.\n\n"
        "## Locked specification\n\n"
        "Top500/Core10; repaired factor risk; cumulative OOS validation; full matrix m; "
        "ADV quadratic costs; gamma 10; baseline AUM USD 1 billion.\n\n"
        "## Baseline metric reconciliation\n\n"
        + _markdown(metric_summary)
        + "\n\nExact-value tolerance is 1e-6. Excel stores exact source values; the Markdown presentation is separately checked at its declared display precision.\n\n"
        "## Utility inference reconciliation\n\n"
        + _markdown(inference_summary)
        + "\n\n## Safety checks\n\n"
        + _markdown(checks)
        + "\n\n## Input availability\n\n"
        + _markdown(availability)
        + "\n\nAim portfolios are available for Portfolio-ML Linear/RF. They are not defined for Return-ML Rank or Static methods. Grown inherited positions and stock-level trades are available for every method.\n"
    )


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    data = frame.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.8g}"
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
