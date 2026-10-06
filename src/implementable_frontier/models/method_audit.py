from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from implementable_frontier.models.accounting import PortfolioAccounting
from implementable_frontier.models.features import (
    RandomFourierFeatures,
    monthly_cross_sectional_standardize,
    volatility_scale_design,
)
from implementable_frontier.models.portfolio_ml import PortfolioMLModel, PortfolioMLSpec
from implementable_frontier.models.dynamic_adjustment import adjustment_matrix_from_risk
from implementable_frontier.models.portfolios import (
    static_ml_star_weights,
)
from implementable_frontier.models.risk import CovarianceResult


@dataclass(frozen=True)
class ParityCheck:
    name: str
    absolute_error: float
    relative_error: float
    tolerance: float
    passed: bool
    note: str = ""


class SyntheticRiskProvider:
    requested_model = "factor"
    universe = "top100"
    factor_model_version = "small-sample-v1"

    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel

    def get_covariance(
        self, eom: str | pd.Timestamp, permnos: Sequence[int], risk_model: str | None = None
    ) -> CovarianceResult:
        ids = np.asarray(permnos, dtype=float)
        centered = (ids - ids.mean()) / max(ids.std(), 1.0)
        loading = np.column_stack([np.ones(len(ids)), centered])
        factor_cov = np.array([[1.2e-4, 2.0e-5], [2.0e-5, 8.0e-5]])
        covariance = loading @ factor_cov @ loading.T + np.eye(len(ids)) * 3.0e-4
        eigenvalues = np.linalg.eigvalsh(covariance)
        diagonal = np.diag(covariance)
        return CovarianceResult(
            covariance=covariance,
            requested_model="factor",
            used_model="factor",
            min_eigenvalue=float(eigenvalues.min()),
            condition_number=float(eigenvalues.max() / eigenvalues.min()),
            missing_assets=(),
            diagonal_min=float(diagonal.min()),
            diagonal_median=float(np.median(diagonal)),
            diagonal_max=float(diagonal.max()),
        )

    def get_volatility(
        self, eom: str | pd.Timestamp, permnos: Sequence[int]
    ) -> np.ndarray:
        return np.sqrt(np.diag(self.get_covariance(eom, permnos).covariance))


def synthetic_parity_panel(
    *, months: int = 18, names_per_month: int = 25, n_features: int = 4, seed: int = 19
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    universe = np.arange(10001, 10001 + names_per_month + 5)
    rows: list[dict[str, object]] = []
    for month_index, eom in enumerate(dates):
        active = np.roll(universe, month_index % len(universe))[:names_per_month]
        active = np.sort(active)
        market_return = 0.004 + 0.002 * np.sin(month_index / 3.0)
        for position, permno in enumerate(active):
            features = rng.uniform(0.0, 1.0, n_features)
            id_signal = (position - names_per_month / 2.0) / names_per_month
            realized = market_return + 0.012 * id_signal + rng.normal(0.0, 0.02)
            row: dict[str, object] = {
                "eom": eom,
                "permno": int(permno),
                "ret_1m": realized,
                "ret_exc_lead1m": realized,
                "lambda_adv": 1.0e-9 * (1.0 + position / 50.0),
                "lag_mktcap": 1.0e9 * (1.0 + position / 20.0),
            }
            for feature_index, value in enumerate(features):
                row[f"rank_x{feature_index + 1}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["eom", "permno"]).reset_index(drop=True)


def run_small_sample_parity(seed: int = 19) -> dict[str, object]:
    panel = synthetic_parity_panel(seed=seed)
    feature_columns = [column for column in panel if column.startswith("rank_x")]
    provider = SyntheticRiskProvider(panel)
    gamma = 10.0
    aum = 1.0e9
    settings = {
        "rf_p_grid": [16, 32],
        "use_matrix_m": True,
        "matrix_m_iterations": 10,
        "expected_market_return": 0.007,
        "risk_free_rate": 0.0,
        "allow_scalar_rho_fallback": False,
        "rho_grid_if_fallback": [0.5],
        "volatility_floor": 0.01,
        "coefficient_ridge_floor": 1.0e-10,
        "add_constant": True,
    }
    spec = PortfolioMLSpec("rf", 10.0, rf_p=16, rf_eta=0.2, seed=seed)
    model = PortfolioMLModel(
        spec,
        feature_columns,
        provider,
        gamma=gamma,
        aum=aum,
        settings=settings,
        adjustment_cache={},
    ).fit(panel.iloc[: 12 * 25].copy())

    checks: list[ParityCheck] = []
    first_month = panel[panel["eom"].eq(panel["eom"].min())].copy()
    raw = first_month[feature_columns].to_numpy(float)
    rff = RandomFourierFeatures(4, 16, 0.2, seed, max_features=32)
    explicit_rf = np.concatenate(
        [
            np.cos(raw @ rff.weights[:, :8]),
            np.sin(raw @ rff.weights[:, :8]),
        ],
        axis=1,
    ) / np.sqrt(16.0)
    checks.append(_check("rf_matrix", rff.transform(raw), explicit_rf, 1.0e-13))

    permnos = first_month["permno"].to_numpy(int)
    covariance = provider.get_covariance(first_month["eom"].iloc[0], permnos).covariance
    lambdas = first_month["lambda_adv"].to_numpy(float)
    production_m = adjustment_matrix_from_risk(
        covariance,
        lambdas,
        gamma=gamma,
        aum=aum,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
    )
    reference_m = reference_matrix_m(
        covariance,
        lambdas,
        gamma=gamma,
        aum=aum,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
    )
    checks.append(_check("matrix_m", production_m, reference_m, 1.0e-10))

    reference_numerator, reference_denominator = reference_portfolio_sufficient_statistics(
        panel.iloc[: 12 * 25].copy(),
        feature_columns,
        spec,
        provider,
        settings,
        gamma=gamma,
        aum=aum,
    )
    checks.append(
        _check("mean_r_tilde", model.mean_numerator_, reference_numerator, 1.0e-10)
    )
    checks.append(
        _check(
            "mean_sigma_tilde",
            model.mean_denominator_,
            reference_denominator,
            1.0e-9,
        )
    )
    reference_hessian = (
        reference_denominator + reference_denominator.T
    ) / 2.0 + model.ridge_lambda_ * np.eye(len(reference_numerator))
    reference_beta = np.linalg.solve(reference_hessian, reference_numerator)
    checks.append(_check("beta", model.beta_, reference_beta, 1.0e-9))

    gradient = model.gradient()
    hessian_eigenvalues = np.linalg.eigvalsh(model.quadratic_hessian())
    objective_beta = model.objective()
    objective_zero = model.objective(np.zeros_like(model.beta_))
    iterative_beta = direct_quadratic_solver(
        model.mean_numerator_, model.quadratic_hessian()
    )
    checks.append(_check("closed_form_vs_iterative_solver", model.beta_, iterative_beta, 1.0e-7))
    checks.append(
        ParityCheck(
            "gradient_norm",
            float(np.linalg.norm(gradient)),
            float(np.linalg.norm(gradient)),
            1.0e-8,
            bool(np.linalg.norm(gradient) <= 1.0e-8),
        )
    )
    checks.append(
        ParityCheck(
            "hessian_psd",
            float(max(-hessian_eigenvalues.min(), 0.0)),
            float(max(-hessian_eigenvalues.min(), 0.0)),
            1.0e-10,
            bool(hessian_eigenvalues.min() >= -1.0e-10),
        )
    )
    checks.append(
        ParityCheck(
            "objective_dominates_zero",
            float(max(objective_zero - objective_beta, 0.0)),
            float(max(objective_zero - objective_beta, 0.0)),
            1.0e-12,
            bool(objective_beta >= objective_zero - 1.0e-12),
        )
    )

    rng = np.random.default_rng(seed + 1)
    perturbation_gains = []
    for _ in range(200):
        direction = rng.normal(size=len(model.beta_))
        direction /= max(np.linalg.norm(direction), 1.0e-12)
        perturbed = model.beta_ + direction * 1.0e-5
        perturbation_gains.append(model.objective(perturbed) - objective_beta)
    max_gain = float(max(perturbation_gains))
    checks.append(
        ParityCheck(
            "local_perturbations",
            max(max_gain, 0.0),
            max(max_gain, 0.0),
            1.0e-10,
            max_gain <= 1.0e-10,
        )
    )

    test_month = panel[panel["eom"].eq(panel["eom"].unique()[12])].copy()
    aim = model.aim_weights(test_month)
    covariance_test = provider.get_covariance(
        test_month["eom"].iloc[0], test_month["permno"].to_numpy(int)
    ).covariance
    lambda_test = test_month["lambda_adv"].to_numpy(float)
    m_test = adjustment_matrix_from_risk(
        covariance_test,
        lambda_test,
        gamma=gamma,
        aum=aum,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
    )
    inherited = np.zeros(len(test_month))
    actual = m_test @ inherited + (np.eye(len(test_month)) - m_test) @ aim
    accounting = PortfolioAccounting(aum=aum, gamma=gamma)
    inherited_positions = accounting.compute_inherited_weights(
        pd.Series(inherited, index=test_month["permno"].to_numpy(int)),
        test_month["permno"].to_numpy(int),
        pd.Series(0.0, index=test_month["permno"].to_numpy(int)),
    )
    accounting_result = accounting.evaluate(
        pd.Series(actual, index=test_month["permno"].to_numpy(int)),
        pd.Series(lambda_test, index=test_month["permno"].to_numpy(int)),
        pd.Series(
            test_month["ret_exc_lead1m"].to_numpy(float),
            index=test_month["permno"].to_numpy(int),
        ),
        inherited_positions,
        ex_ante_variance=float(actual @ covariance_test @ actual),
    )

    static = static_ml_star_weights(
        np.linspace(-0.01, 0.01, len(test_month)),
        covariance_test,
        inherited,
        lambda_test,
        gamma=gamma,
        aum=aum,
        u=0.5,
        v=0.25,
        k=2.0,
        ridge=1.0e-8,
    )
    static_reference = reference_static_ml_star(
        np.linspace(-0.01, 0.01, len(test_month)),
        covariance_test,
        inherited,
        lambda_test,
        gamma=gamma,
        aum=aum,
        u=0.5,
        v=0.25,
        k=2.0,
        ridge=1.0e-8,
    )
    checks.append(_check("static_ml_star", static.weights, static_reference, 1.0e-11))

    eigenvalues_m = np.linalg.eigvals(m_test)
    return {
        "sample": {
            "months": int(panel["eom"].nunique()),
            "names_per_month": int(panel.groupby("eom").size().median()),
            "features": len(feature_columns),
            "seed": seed,
            "gamma": gamma,
            "aum": aum,
            "ridge_lambda": spec.ridge_lambda,
            "rf_p": spec.rf_p,
            "rf_eta": spec.rf_eta,
        },
        "checks": [check.__dict__ for check in checks],
        "passed": all(check.passed for check in checks),
        "portfolio_ml": {
            "beta": model.beta_.tolist(),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "hessian_eigenvalue_min": float(hessian_eigenvalues.min()),
            "hessian_eigenvalue_median": float(np.median(hessian_eigenvalues)),
            "hessian_eigenvalue_max": float(hessian_eigenvalues.max()),
            "objective_at_beta": objective_beta,
            "objective_at_zero": objective_zero,
            "max_local_perturbation_gain": max_gain,
            "aim_weights": aim.tolist(),
            "actual_weights": actual.tolist(),
            "monthly_utility": accounting_result.utility_flow,
        },
        "matrix_m": {
            "eigenvalue_real_min": float(np.real(eigenvalues_m).min()),
            "eigenvalue_real_median": float(np.median(np.real(eigenvalues_m))),
            "eigenvalue_real_max": float(np.real(eigenvalues_m).max()),
            "max_imaginary_part": float(np.abs(np.imag(eigenvalues_m)).max()),
            "trading_speed_min": float(np.real(1.0 - eigenvalues_m).min()),
            "trading_speed_median": float(np.median(np.real(1.0 - eigenvalues_m))),
            "trading_speed_max": float(np.real(1.0 - eigenvalues_m).max()),
            "condition_number": float(np.linalg.cond(m_test)),
            "failure_count": 0,
            "fallback_count": 0,
        },
        "rf": {
            "column_means": rff.transform(raw).mean(axis=0).tolist(),
            "column_norms": np.linalg.norm(rff.transform(raw), axis=0).tolist(),
            "singular_values": np.linalg.svd(rff.transform(raw), compute_uv=False).tolist(),
            "effective_rank": int(np.linalg.matrix_rank(rff.transform(raw))),
            "condition_number": float(np.linalg.cond(rff.transform(raw))),
            "random_seed": seed,
            "rf_matrix_hash": hashlib.sha256(
                np.ascontiguousarray(rff.transform(raw)).tobytes()
            ).hexdigest(),
        },
        "accounting": {
            "inherited_weights": inherited_positions.current.tolist(),
            "trades": accounting_result.trade.tolist(),
            "trading_cost": accounting_result.trading_cost,
            "gross_return": accounting_result.gross_return,
            "net_return": accounting_result.net_return,
            "utility": accounting_result.utility_flow,
        },
    }


def reference_portfolio_sufficient_statistics(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: PortfolioMLSpec,
    provider: SyntheticRiskProvider,
    settings: dict[str, object],
    *,
    gamma: float,
    aum: float,
) -> tuple[np.ndarray, np.ndarray]:
    rff = RandomFourierFeatures(
        len(feature_columns),
        int(spec.rf_p or 0),
        float(spec.rf_eta or 0),
        spec.seed,
        max_features=max(int(value) for value in settings["rf_p_grid"]),
    )
    numerator = None
    denominator = None
    observations = 0
    previous_b: pd.DataFrame | None = None
    previous_returns = pd.Series(dtype=float)
    previous_lambda = pd.Series(dtype=float)
    previous_market_growth: float | None = None
    for eom, month in frame.sort_values(["eom", "permno"]).groupby("eom", sort=True):
        month = month.sort_values("permno").reset_index(drop=True)
        ids = month["permno"].to_numpy(int)
        raw = month[feature_columns].to_numpy(float)
        transformed = rff.transform(raw)
        standardized = monthly_cross_sectional_standardize(transformed, add_constant=True)
        covariance = provider.get_covariance(eom, ids).covariance
        design, _ = volatility_scale_design(
            standardized, np.sqrt(np.diag(covariance)), floor=0.01
        )
        lambdas = month["lambda_adv"].to_numpy(float)
        m = reference_matrix_m(
            covariance,
            lambdas,
            gamma=gamma,
            aum=aum,
            expected_market_return=0.007,
            risk_free_rate=0.0,
            iterations=10,
        )
        if previous_b is None:
            inherited = np.zeros((len(ids), design.shape[1]))
            dropped = pd.DataFrame()
        else:
            previous_aligned_returns = previous_returns.reindex(previous_b.index).fillna(0.0)
            growth = max(
                float(previous_market_growth)
                if previous_market_growth is not None
                else 1.0 + float(previous_aligned_returns.mean()),
                0.1,
            )
            grown = previous_b.mul((1.0 + previous_aligned_returns) / growth, axis=0)
            inherited_frame = grown.reindex(ids).fillna(0.0)
            dropped = grown.loc[~grown.index.isin(ids)]
            inherited = inherited_frame.to_numpy(float)
        b_matrix = m @ inherited + (np.eye(len(ids)) - m) @ design
        delta = b_matrix - inherited
        denom = gamma * (b_matrix.T @ covariance @ b_matrix)
        denom += aum * (delta.T @ (lambdas[:, None] * delta))
        if not dropped.empty:
            dropped_lambda = previous_lambda.reindex(dropped.index).fillna(
                float(np.median(lambdas))
            )
            dropped_values = dropped.to_numpy(float)
            denom += aum * (
                dropped_values.T @ (dropped_lambda.to_numpy()[:, None] * dropped_values)
            )
        targets = month["ret_exc_lead1m"].to_numpy(float)
        r_tilde = b_matrix.T @ targets
        numerator = r_tilde if numerator is None else numerator + r_tilde
        denominator = denom if denominator is None else denominator + denom
        observations += 1
        previous_b = pd.DataFrame(b_matrix, index=ids)
        previous_returns = pd.Series(targets, index=ids)
        previous_lambda = pd.Series(lambdas, index=ids)
        caps = month["lag_mktcap"].to_numpy(float)
        previous_market_growth = max(
            1.0 + float(np.average(targets, weights=caps)), 0.1
        )
    if numerator is None or denominator is None:
        raise ValueError("Reference statistics are empty")
    return numerator / observations, denominator / observations


def reference_matrix_m(
    covariance: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    expected_market_return: float,
    risk_free_rate: float,
    iterations: int,
) -> np.ndarray:
    """Audit the thesis fixed point via a spectral initial value and linear solves.

    The scalar spectral map uses the rationalized positive root of the thesis
    quadratic equation. It provides a separate numerical path from the
    production matrix-square-root implementation.
    """
    risk = np.asarray(covariance, dtype=float)
    impact = np.asarray(lambda_adv, dtype=float)
    n_assets = len(impact)
    if aum == 0:
        return np.zeros((n_assets, n_assets), dtype=float)
    impact_scale = np.sqrt(impact)
    scaled_risk = (gamma / aum) * risk / np.outer(impact_scale, impact_scale)
    eigenvalues, basis = np.linalg.eigh((scaled_risk + scaled_risk.T) / 2.0)
    shifted = eigenvalues + 2.0
    initial_values = 2.0 / (shifted + np.sqrt(np.maximum(shifted**2 - 4.0, 0.0)))
    state = (basis * initial_values) @ basis.T

    market_growth = 1.0 + risk_free_rate + expected_market_return
    growth_covariance = np.ones_like(risk) + risk / market_growth**2
    diagonal_term = 1.0 + np.diag(growth_covariance)
    identity = np.eye(n_assets)
    for _ in range(iterations):
        coefficient = scaled_risk - state * growth_covariance
        coefficient[np.diag_indices(n_assets)] += diagonal_term
        state = np.linalg.solve(coefficient, identity)
    return state * (impact_scale[None, :] / impact_scale[:, None])


def reference_static_ml_star(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    inherited: np.ndarray,
    lambdas: np.ndarray,
    *,
    gamma: float,
    aum: float,
    u: float,
    v: float,
    k: float,
    ridge: float,
) -> np.ndarray:
    sigma = (covariance + covariance.T) / 2.0
    sigma_star = sigma + v * np.diag(np.diag(sigma)) + ridge * np.eye(len(sigma))
    impact = aum * k * np.diag(lambdas)
    return np.linalg.solve(
        gamma * sigma_star + impact,
        u * expected_returns + impact @ inherited,
    )


def direct_quadratic_solver(
    numerator: np.ndarray, hessian: np.ndarray, iterations: int = 20_000
) -> np.ndarray:
    values = np.zeros_like(numerator)
    step = 1.0 / max(float(np.linalg.eigvalsh(hessian).max()), 1.0e-12)
    for _ in range(iterations):
        gradient = hessian @ values - numerator
        updated = values - step * gradient
        if np.linalg.norm(updated - values) <= 1.0e-12:
            values = updated
            break
        values = updated
    return values


def _check(name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float) -> ParityCheck:
    actual_values = np.asarray(actual, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    absolute = float(np.max(np.abs(actual_values - expected_values)))
    scale = max(float(np.max(np.abs(expected_values))), 1.0e-15)
    relative = absolute / scale
    return ParityCheck(name, absolute, relative, tolerance, absolute <= tolerance)
