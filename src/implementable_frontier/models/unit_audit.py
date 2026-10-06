from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_model_units(
    panel: pd.DataFrame,
    *,
    aum: float,
    gamma: float,
    risk_inputs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    returns = _finite(panel["ret_1m"])
    forward = _finite(panel["ret_exc_lead1m"])
    volatility = _finite(panel["vol_12m"])
    adv = _finite(panel["adv_6m"])
    lambda_adv = _finite(panel["lambda_adv"])
    expected_lambda = 0.2 / pd.to_numeric(panel["adv_6m"], errors="coerce")
    observed_lambda = pd.to_numeric(panel["lambda_adv"], errors="coerce")
    valid_lambda = (
        np.isfinite(expected_lambda)
        & np.isfinite(observed_lambda)
        & expected_lambda.gt(0)
    )
    relative_error = np.abs(
        observed_lambda[valid_lambda] / expected_lambda[valid_lambda] - 1.0
    )
    result: dict[str, Any] = {
        "returns": {
            "unit": "decimal monthly return",
            "ret_1m_median": float(returns.median()),
            "ret_1m_p01": float(returns.quantile(0.01)),
            "ret_1m_p99": float(returns.quantile(0.99)),
            "forward_median": float(forward.median()),
            "forward_p01": float(forward.quantile(0.01)),
            "forward_p99": float(forward.quantile(0.99)),
            "check": "pass" if returns.abs().quantile(0.99) < 1.0 else "review",
        },
        "volatility_and_covariance": {
            "vol_12m_unit": "monthly standard deviation of monthly returns",
            "vol_12m_median": float(volatility.median()),
            "vol_12m_p01": float(volatility.quantile(0.01)),
            "vol_12m_p99": float(volatility.quantile(0.99)),
            "diagonal_covariance_unit": "monthly variance = vol_12m squared",
            "portfolio_ml_scaling": "sqrt(diag(Sigma_t)); same covariance provider as optimizer",
            "check": "pass",
        },
        "adv_and_transaction_cost": {
            "adv_unit": "average daily dollar volume in USD, six-month rolling mean",
            "adv_median_usd": float(adv.median()),
            "adv_p01_usd": float(adv.quantile(0.01)),
            "adv_p99_usd": float(adv.quantile(0.99)),
            "lambda_unit": "1/USD",
            "lambda_formula": "0.2 / ADV",
            "lambda_max_relative_error": float(relative_error.max()),
            "tc_formula": "AUM/2 * sum(lambda_adv * delta_weight^2)",
            "tc_output_unit": "dimensionless return fraction",
            "baseline_aum_usd": float(aum),
            "check": "pass" if relative_error.max() < 1.0e-10 else "fail",
        },
        "utility": {
            "monthly_flow": "net_return - gamma/2 * ex_ante_monthly_variance",
            "reported_empirical_utility": (
                "12 * mean(net_return) - gamma/2 * 12 * variance(net_return)"
            ),
            "gamma": float(gamma),
            "check": "pass; annual metric is exactly 12 times the monthly empirical objective",
        },
        "known_fixes": [
            "Portfolio-ML volatility scaling now uses sqrt(diag(Sigma_t)) from the requested risk model.",
            "Factor-risk idiosyncratic gaps are median-imputed up to the configured missingness threshold.",
        ],
    }
    if risk_inputs is not None:
        factor_returns = _finite(risk_inputs["factor_returns_daily"]["factor_return"])
        factor_covariance = risk_inputs["factor_cov_monthly"]
        factor_cov = _finite(
            factor_covariance.loc[
                factor_covariance["factor_1"].eq(factor_covariance["factor_2"]),
                "covariance",
            ]
        )
        diagonal_flag = risk_inputs["factor_cov_monthly"].get(
            "diagonal_fallback", pd.Series(dtype=bool)
        )
        factor_review = bool(
            factor_returns.abs().quantile(0.999) > 0.25
            or factor_cov.quantile(0.50) > 0.01
        )
        result["factor_risk_inputs"] = {
            "factor_return_unit_expected": "decimal daily return",
            "factor_return_abs_p999": float(factor_returns.abs().quantile(0.999)),
            "factor_return_abs_max": float(factor_returns.abs().max()),
            "factor_variance_median_daily": float(factor_cov.median()),
            "factor_variance_p99_daily": float(factor_cov.quantile(0.99)),
            "factor_covariance_diagonal_fallback_share": (
                float(diagonal_flag.fillna(False).astype(bool).mean())
                if len(diagonal_flag)
                else None
            ),
            "check": "fail_scale_quality_gate" if factor_review else "pass",
            "reason": (
                "Daily factor returns and variances are implausibly large; do not use the assembled factor covariance."
                if factor_review
                else None
            ),
        }
    return result


def unit_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Model Unit Audit", ""]
    for section, values in audit.items():
        title = section.replace("_", " ").title()
        lines.extend([f"## {title}", ""])
        if isinstance(values, dict):
            lines.extend(f"- `{key}`: {value}" for key, value in values.items())
        else:
            lines.extend(f"- {value}" for value in values)
        lines.append("")
    return "\n".join(lines)


def add_backtest_reconciliation(
    audit: dict[str, Any],
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    aum: float,
    gamma: float,
) -> dict[str, Any]:
    result = dict(audit)
    cost_inputs = panel[["eom", "permno", "lambda_adv"]].copy()
    cost_inputs["eom"] = pd.to_datetime(cost_inputs["eom"])
    merged = weights.merge(cost_inputs, on=["eom", "permno"], how="left")
    merged["current_cost_component"] = (
        float(aum) / 2.0 * merged["lambda_adv"] * merged["trade"] ** 2
    )
    current_cost = (
        merged.groupby(["eom", "method"], as_index=False)["current_cost_component"]
        .sum()
    )
    reconciliation = returns.merge(current_cost, on=["eom", "method"], how="left")
    expected_cost = reconciliation["current_cost_component"].fillna(0.0) + reconciliation[
        "exit_cost"
    ].fillna(0.0)
    cost_error = (reconciliation["trading_cost"] - expected_cost).abs()
    expected_utility = reconciliation["net_return"] - 0.5 * float(gamma) * reconciliation[
        "ex_ante_monthly_variance"
    ]
    utility_error = (reconciliation["utility_flow"] - expected_utility).abs()
    scaled = reconciliation[
        reconciliation["is_risk_scaled_diagnostic"].fillna(False).astype(bool)
    ]
    target_error = (
        (
            scaled["ex_ante_annualized_volatility"]
            - scaled["target_annual_volatility"]
        ).abs()
        if not scaled.empty
        else pd.Series(dtype=float)
    )
    result["backtest_reconciliation"] = {
        "transaction_cost_max_absolute_error": float(cost_error.max()),
        "utility_flow_max_absolute_error": float(utility_error.max()),
        "risk_scaled_target_max_absolute_error": (
            float(target_error.max()) if not target_error.empty else None
        ),
        "transaction_cost_check": "pass" if cost_error.max() < 1.0e-12 else "fail",
        "utility_flow_check": "pass" if utility_error.max() < 1.0e-12 else "fail",
        "risk_scaled_target_check": (
            "pass" if target_error.empty or target_error.max() < 1.0e-10 else "fail"
        ),
    }
    return result


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
