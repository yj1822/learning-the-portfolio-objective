from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from implementable_frontier.models.accounting import PortfolioAccounting
from implementable_frontier.models.portfolio_ml import PortfolioMLModel
from implementable_frontier.models.portfolios import (
    equal_weight,
    markowitz_weights,
    rank_long_short,
    static_ml_star_weights,
    static_ml_precision_matrix,
    static_ml_star_precision_matrix,
    static_ml_weights,
)
from implementable_frontier.models.risk import RiskModelProvider


@dataclass
class BacktestState:
    weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    forward_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    lambda_adv: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    market_growth: float | None = None


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    weights: pd.DataFrame
    returns: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    risk_models_used: set[str] = field(default_factory=set)
    adjustment_modes: set[str] = field(default_factory=set)
    final_state: BacktestState = field(default_factory=BacktestState)


def run_backtest(
    frame: pd.DataFrame,
    *,
    method: str,
    risk_provider: RiskModelProvider,
    gamma: float,
    aum: float,
    accounting_settings: dict[str, object],
    method_settings: dict[str, object],
    predictions: np.ndarray | None = None,
    portfolio_model: PortfolioMLModel | None = None,
    test_year: int | None = None,
    initial_state: BacktestState | None = None,
    output_method: str | None = None,
    target_annual_volatility: float | None = None,
    beta_norm: float | None = None,
    transaction_cache: Any | None = None,
    static_precision_cache: Any | None = None,
) -> BacktestResult:
    data = frame.sort_values(["eom", "permno"]).reset_index(drop=True).copy()
    if predictions is not None:
        if len(predictions) != len(data):
            raise ValueError("Prediction count does not match backtest rows")
        data["_mu_hat"] = np.asarray(predictions, dtype=float)
    accounting = PortfolioAccounting(
        aum=aum,
        gamma=gamma,
        **accounting_settings,
    )
    state = initial_state or BacktestState()
    previous_weights = state.weights.copy()
    previous_returns = state.forward_returns.copy()
    previous_lambda = state.lambda_adv.copy()
    previous_market_growth = state.market_growth
    prediction_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    risk_models_used: set[str] = set()
    adjustment_modes: set[str] = set()
    reported_method = output_method or method

    for eom, month in data.groupby("eom", sort=True):
        month = month.sort_values("permno").reset_index(drop=True)
        target = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce")
        if not target.notna().any():
            warnings.append(f"{pd.Timestamp(eom).date()}: skipped month with no forward returns")
            continue
        permnos = month["permno"].astype(int).to_numpy()
        raw_lambdas = pd.to_numeric(
            month["lambda_adv"], errors="coerce"
        ).to_numpy(float)
        if transaction_cache is not None:
            lambda_series = transaction_cache.get(
                universe=risk_provider.universe,
                eom=pd.Timestamp(eom),
                aum=aum,
                cost_model_version=str(
                    method_settings.get("cost_model_version", "adv_quadratic_v1")
                ),
                permnos=permnos,
                values=raw_lambdas,
            )
        else:
            lambda_series = pd.Series(raw_lambdas, index=permnos)
        if previous_weights.empty and str(
            method_settings.get("initial_position", "zero")
        ) == "value_weighted":
            cap_column = "lag_mktcap" if "lag_mktcap" in month else "mktcap"
            caps = (
                pd.to_numeric(month[cap_column], errors="coerce").to_numpy(float)
                if cap_column in month
                else np.full(len(month), np.nan)
            )
            valid_caps = np.isfinite(caps) & (caps > 0)
            if valid_caps.any() and caps[valid_caps].sum() > 0:
                initial = np.zeros(len(month))
                initial[valid_caps] = caps[valid_caps] / caps[valid_caps].sum()
            else:
                initial = equal_weight(len(month))
            previous_weights = pd.Series(initial, index=permnos)
            previous_returns = pd.Series(0.0, index=permnos)
            previous_market_growth = 1.0
        inherited = accounting.compute_inherited_weights(
            previous_weights,
            permnos,
            previous_returns,
            wealth_growth=previous_market_growth,
        )
        covariance_result = risk_provider.get_covariance(eom, permnos)
        covariance = covariance_result.covariance
        risk_models_used.add(covariance_result.used_model)
        if covariance_result.fallback_reason:
            warnings.append(
                f"{pd.Timestamp(eom).date()}: risk fallback: {covariance_result.fallback_reason}"
            )
        if covariance_result.missing_assets:
            warnings.append(
                f"{pd.Timestamp(eom).date()}: risk volatility fallback for "
                f"{len(covariance_result.missing_assets)} names"
            )
        if covariance_result.condition_number > float(
            method_settings.get("covariance_condition_warning", 1.0e10)
        ):
            warnings.append(
                f"{pd.Timestamp(eom).date()}: covariance condition number "
                f"{covariance_result.condition_number:.4g}"
            )
        mu_hat = (
            pd.to_numeric(month["_mu_hat"], errors="coerce").fillna(0.0).to_numpy(float)
            if "_mu_hat" in month
            else None
        )
        aim = np.full(len(month), np.nan)
        adjustment_mode = "not_applicable"
        adjustment_eigenvalues = np.array([], dtype=float)
        adjustment_real_eigenvalues = np.array([], dtype=float)
        adjustment_max_imaginary = np.nan

        if method == "equal_weight":
            weights = equal_weight(len(month))
        elif method == "return_ml_rank":
            _require_predictions(mu_hat, method)
            weights = rank_long_short(
                mu_hat,
                gross_leverage=float(method_settings.get("rank_gross_leverage", 1.0)),
            )
        elif method == "markowitz_ml":
            _require_predictions(mu_hat, method)
            solution = markowitz_weights(
                mu_hat,
                covariance,
                gamma=gamma,
                ridge=float(method_settings.get("covariance_ridge", 1.0e-8)),
                max_leverage=method_settings.get("max_leverage"),
            )
            weights = solution.weights
            if solution.used_pseudoinverse:
                warnings.append(f"{pd.Timestamp(eom).date()}: Markowitz used pseudoinverse")
        elif method == "static_ml":
            _require_predictions(mu_hat, method)
            precision = None
            if static_precision_cache is not None:
                precision = static_precision_cache.get_or_compute(
                    (
                        method,
                        pd.Timestamp(eom),
                        float(gamma),
                        float(aum),
                        float(method_settings["phi"]),
                        float(method_settings.get("covariance_ridge", 1.0e-8)),
                        tuple(int(value) for value in permnos),
                    ),
                    lambda: static_ml_precision_matrix(
                        covariance,
                        lambda_series.to_numpy(float),
                        gamma=gamma,
                        aum=aum,
                        phi=float(method_settings["phi"]),
                        ridge=float(method_settings.get("covariance_ridge", 1.0e-8)),
                    ),
                )
            solution = static_ml_weights(
                mu_hat,
                covariance,
                inherited.current.reindex(permnos).to_numpy(float),
                lambda_series.to_numpy(float),
                gamma=gamma,
                aum=aum,
                phi=float(method_settings["phi"]),
                mu_shrinkage=float(method_settings["mu_shrinkage"]),
                ridge=float(method_settings.get("covariance_ridge", 1.0e-8)),
                precision_inverse=(precision.matrix if precision is not None else None),
                precision_used_pseudoinverse=(
                    precision.used_pseudoinverse if precision is not None else False
                ),
            )
            weights = solution.weights
            if solution.used_pseudoinverse:
                warnings.append(f"{pd.Timestamp(eom).date()}: Static-ML used pseudoinverse")
        elif method == "static_ml_star":
            _require_predictions(mu_hat, method)
            precision = None
            if static_precision_cache is not None:
                precision = static_precision_cache.get_or_compute(
                    (
                        method,
                        pd.Timestamp(eom),
                        float(gamma),
                        float(aum),
                        float(method_settings["v"]),
                        float(method_settings["k"]),
                        float(method_settings.get("covariance_ridge", 1.0e-8)),
                        tuple(int(value) for value in permnos),
                    ),
                    lambda: static_ml_star_precision_matrix(
                        covariance,
                        lambda_series.to_numpy(float),
                        gamma=gamma,
                        aum=aum,
                        v=float(method_settings["v"]),
                        k=float(method_settings["k"]),
                        ridge=float(method_settings.get("covariance_ridge", 1.0e-8)),
                    ),
                )
            solution = static_ml_star_weights(
                mu_hat,
                covariance,
                inherited.current.reindex(permnos).to_numpy(float),
                lambda_series.to_numpy(float),
                gamma=gamma,
                aum=aum,
                u=float(method_settings["u"]),
                v=float(method_settings["v"]),
                k=float(method_settings["k"]),
                ridge=float(method_settings.get("covariance_ridge", 1.0e-8)),
                precision_inverse=(precision.matrix if precision is not None else None),
                precision_used_pseudoinverse=(
                    precision.used_pseudoinverse if precision is not None else False
                ),
            )
            weights = solution.weights
            if solution.used_pseudoinverse:
                warnings.append(
                    f"{pd.Timestamp(eom).date()}: Static-ML* used pseudoinverse"
                )
        elif method in {"portfolio_ml_linear", "portfolio_ml_rf"}:
            if portfolio_model is None:
                raise ValueError(f"{method} requires a fitted PortfolioMLModel")
            aim = portfolio_model.aim_weights(month)
            adjustment = portfolio_model.adjustment_matrix(
                covariance, lambda_series.to_numpy(float), eom=pd.Timestamp(eom)
            )
            adjustment_mode = adjustment.mode
            adjustment_modes.add(adjustment.mode)
            if adjustment.fallback_reason:
                warnings.append(
                    f"{pd.Timestamp(eom).date()}: matrix m fallback: {adjustment.fallback_reason}"
                )
            inherited_values = inherited.current.reindex(permnos).to_numpy(float)
            weights = adjustment.matrix @ inherited_values + (
                np.eye(len(month)) - adjustment.matrix
            ) @ aim
            raw_adjustment_eigenvalues = (
                adjustment.eigenvalues
                if adjustment.eigenvalues is not None
                else np.linalg.eigvals(adjustment.matrix)
            )
            adjustment_eigenvalues = np.abs(raw_adjustment_eigenvalues)
            adjustment_real_eigenvalues = np.real(raw_adjustment_eigenvalues)
            adjustment_max_imaginary = float(
                np.max(np.abs(np.imag(raw_adjustment_eigenvalues)))
            )
        else:
            raise ValueError(f"Unsupported backtest method: {method}")

        pre_scale_variance = max(float(weights @ covariance @ weights), 0.0)
        pre_scale_annual_volatility = float(np.sqrt(12.0 * pre_scale_variance))
        risk_scale_factor = 1.0
        if target_annual_volatility is not None:
            risk_scale_factor = (
                float(target_annual_volatility) / pre_scale_annual_volatility
                if pre_scale_annual_volatility > 0
                else 0.0
            )
            weights = weights * risk_scale_factor
            aim = aim * risk_scale_factor
        weights_series = pd.Series(weights, index=permnos)
        forward_returns = pd.Series(target.to_numpy(float), index=permnos)
        ex_ante_variance = float(weights @ covariance @ weights)
        result = accounting.evaluate(
            weights_series,
            lambda_series,
            forward_returns,
            inherited,
            previous_lambda=previous_lambda,
            ex_ante_variance=ex_ante_variance,
        )
        absolute_weights = np.abs(weights)
        max_abs_weight = float(absolute_weights.max()) if len(weights) else 0.0
        top10_abs_weight = float(np.sort(absolute_weights)[-10:].sum())
        top10_concentration = (
            top10_abs_weight / result.leverage if result.leverage > 0 else 0.0
        )
        inherited_leverage = float(
            inherited.current.abs().sum() + inherited.dropped.abs().sum()
        )
        aim_leverage = float(np.nansum(np.abs(aim))) if np.isfinite(aim).any() else np.nan
        if max_abs_weight > float(method_settings.get("max_abs_weight_warning", 5.0)):
            warnings.append(
                f"{pd.Timestamp(eom).date()}: {reported_method} max abs weight {max_abs_weight:.4g}"
            )
        if result.leverage > float(method_settings.get("max_leverage_warning", 20.0)):
            warnings.append(
                f"{pd.Timestamp(eom).date()}: {reported_method} leverage {result.leverage:.4g}"
            )
        for position, permno in enumerate(permnos):
            weight_rows.append(
                {
                    "eom": pd.Timestamp(eom),
                    "permno": int(permno),
                    "method": reported_method,
                    "weight": float(weights[position]),
                    "aim_weight": float(aim[position]) if np.isfinite(aim[position]) else np.nan,
                    "inherited_weight": float(result.inherited.loc[permno]),
                    "trade": float(result.trade.loc[permno]),
                    "test_year": test_year,
                }
            )
            if mu_hat is not None or np.isfinite(aim[position]):
                prediction_rows.append(
                    {
                        "eom": pd.Timestamp(eom),
                        "permno": int(permno),
                        "method": reported_method,
                        "mu_hat": float(mu_hat[position]) if mu_hat is not None else np.nan,
                        "aim_weight": float(aim[position]) if np.isfinite(aim[position]) else np.nan,
                        "test_year": test_year,
                    }
                )
        return_rows.append(
            {
                "eom": pd.Timestamp(eom),
                "method": reported_method,
                "gross_return": result.gross_return,
                "trading_cost": result.trading_cost,
                "net_return": result.net_return,
                "utility_flow": result.utility_flow,
                "turnover": result.turnover,
                "leverage": result.leverage,
                "actual_portfolio_leverage": result.leverage,
                "long_exposure": result.long_exposure,
                "short_exposure": result.short_exposure,
                "max_abs_single_name_weight": max_abs_weight,
                "top10_abs_weight": top10_abs_weight,
                "top10_abs_weight_concentration": top10_concentration,
                "aim_portfolio_leverage": aim_leverage,
                "inherited_position_leverage": inherited_leverage,
                "n_names": result.n_names,
                "exit_cost": result.exit_cost,
                "dropped_names": int(len(result.dropped_positions)),
                "wealth_growth": result.wealth_growth,
                "weighted_lambda_exposure": result.weighted_lambda_exposure,
                "max_lambda_adv": result.max_lambda_adv,
                "risk_model_used": covariance_result.used_model,
                "covariance_min_eigenvalue": covariance_result.min_eigenvalue,
                "covariance_condition_number": covariance_result.condition_number,
                "covariance_diagonal_min": covariance_result.diagonal_min,
                "covariance_diagonal_median": covariance_result.diagonal_median,
                "covariance_diagonal_max": covariance_result.diagonal_max,
                "missing_risk_assets": len(covariance_result.missing_assets),
                "idio_imputed_assets": len(covariance_result.idio_imputed_assets),
                "factor_covariance_diagonal_fallback": (
                    covariance_result.factor_covariance_diagonal_fallback
                ),
                "covariance_repair_count": covariance_result.covariance_repair_count,
                "risk_fallback_reason": covariance_result.fallback_reason,
                "adjustment_mode": adjustment_mode,
                "matrix_m_eigenvalue_min": (
                    float(adjustment_eigenvalues.min()) if adjustment_eigenvalues.size else np.nan
                ),
                "matrix_m_eigenvalue_median": (
                    float(np.median(adjustment_eigenvalues)) if adjustment_eigenvalues.size else np.nan
                ),
                "matrix_m_eigenvalue_max": (
                    float(adjustment_eigenvalues.max()) if adjustment_eigenvalues.size else np.nan
                ),
                "trading_speed_min": (
                    float((1.0 - adjustment_eigenvalues).min())
                    if adjustment_eigenvalues.size
                    else np.nan
                ),
                "trading_speed_median": (
                    float(np.median(1.0 - adjustment_eigenvalues))
                    if adjustment_eigenvalues.size
                    else np.nan
                ),
                "trading_speed_max": (
                    float((1.0 - adjustment_eigenvalues).max())
                    if adjustment_eigenvalues.size
                    else np.nan
                ),
                "matrix_m_real_eigenvalue_min": (
                    float(adjustment_real_eigenvalues.min())
                    if adjustment_real_eigenvalues.size
                    else np.nan
                ),
                "matrix_m_real_eigenvalue_median": (
                    float(np.median(adjustment_real_eigenvalues))
                    if adjustment_real_eigenvalues.size
                    else np.nan
                ),
                "matrix_m_real_eigenvalue_max": (
                    float(adjustment_real_eigenvalues.max())
                    if adjustment_real_eigenvalues.size
                    else np.nan
                ),
                "matrix_m_max_imaginary_eigenvalue": adjustment_max_imaginary,
                "beta_norm": float(beta_norm) if beta_norm is not None else np.nan,
                "ex_ante_monthly_variance": ex_ante_variance,
                "ex_ante_annualized_volatility": float(np.sqrt(max(12.0 * ex_ante_variance, 0.0))),
                "pre_scale_annualized_volatility": pre_scale_annual_volatility,
                "risk_scale_factor": risk_scale_factor,
                "target_annual_volatility": target_annual_volatility,
                "is_risk_scaled_diagnostic": target_annual_volatility is not None,
                "test_year": test_year,
            }
        )
        previous_weights = weights_series
        previous_returns = forward_returns
        previous_lambda = lambda_series
        if "lag_mktcap" in month:
            caps = pd.to_numeric(month["lag_mktcap"], errors="coerce").to_numpy(float)
            realized = target.to_numpy(float)
            valid_market = np.isfinite(caps) & (caps > 0) & np.isfinite(realized)
            previous_market_growth = (
                max(
                    1.0
                    + float(
                        np.average(realized[valid_market], weights=caps[valid_market])
                    ),
                    0.1,
                )
                if valid_market.any() and caps[valid_market].sum() > 0
                else result.wealth_growth
            )
        else:
            previous_market_growth = result.wealth_growth

    return BacktestResult(
        predictions=pd.DataFrame(prediction_rows),
        weights=pd.DataFrame(weight_rows),
        returns=pd.DataFrame(return_rows),
        warnings=warnings,
        risk_models_used=risk_models_used,
        adjustment_modes=adjustment_modes,
        final_state=BacktestState(
            previous_weights,
            previous_returns,
            previous_lambda,
            previous_market_growth,
        ),
    )


def _require_predictions(values: np.ndarray | None, method: str) -> None:
    if values is None:
        raise ValueError(f"{method} requires Return-ML predictions")
