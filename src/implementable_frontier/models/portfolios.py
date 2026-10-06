from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from implementable_frontier.models.dynamic_adjustment import adjustment_matrix_from_risk


@dataclass(frozen=True)
class PortfolioSolution:
    weights: np.ndarray
    leverage: float
    max_abs_weight: float
    used_pseudoinverse: bool


@dataclass(frozen=True)
class AdjustmentMatrixResult:
    matrix: np.ndarray
    mode: str
    spectral_radius: float
    fallback_reason: str | None = None
    eigenvalues: np.ndarray | None = None


def equal_weight(n_assets: int) -> np.ndarray:
    if n_assets < 1:
        return np.empty(0)
    return np.full(n_assets, 1.0 / n_assets)


def rank_long_short(predictions: np.ndarray, gross_leverage: float = 1.0) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float)
    centered = ranks - ranks.mean()
    gross = np.abs(centered).sum()
    return centered * (gross_leverage / gross) if gross > 0 else np.zeros_like(centered)


def top_bottom_portfolio(predictions: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    count = max(1, int(np.floor(len(values) * fraction)))
    order = np.argsort(values, kind="mergesort")
    weights = np.zeros(len(values))
    weights[order[:count]] = -0.5 / count
    weights[order[-count:]] = 0.5 / count
    return weights


def markowitz_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    gamma: float,
    ridge: float,
    max_leverage: float | None = None,
) -> PortfolioSolution:
    mu = np.asarray(expected_returns, dtype=float)
    sigma = _regularized_covariance(covariance, ridge)
    used_pseudoinverse = False
    try:
        weights = np.linalg.solve(gamma * sigma, mu)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gamma * sigma, hermitian=True) @ mu
        used_pseudoinverse = True
    weights = _limit_leverage(weights, max_leverage)
    return _solution(weights, used_pseudoinverse)


def static_ml_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    inherited_weights: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    phi: float,
    mu_shrinkage: float,
    ridge: float,
    precision_inverse: np.ndarray | None = None,
    precision_used_pseudoinverse: bool = False,
) -> PortfolioSolution:
    mu = np.asarray(expected_returns, dtype=float) * float(mu_shrinkage)
    inherited = np.asarray(inherited_weights, dtype=float)
    lambdas = np.asarray(lambda_adv, dtype=float)
    if (lambdas <= 0).any():
        raise ValueError("lambda_adv must be positive")
    tc_precision = aum / phi * np.diag(lambdas)
    rhs = mu + tc_precision @ inherited
    used_pseudoinverse = bool(precision_used_pseudoinverse)
    if precision_inverse is not None:
        weights = np.asarray(precision_inverse, dtype=float) @ rhs
    else:
        lhs = static_ml_precision_matrix(
            covariance,
            lambdas,
            gamma=gamma,
            aum=aum,
            phi=phi,
            ridge=ridge,
        )
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs, hermitian=True) @ rhs
            used_pseudoinverse = True
    return _solution(weights, used_pseudoinverse)


def static_ml_star_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    inherited_weights: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    u: float,
    v: float,
    k: float,
    ridge: float,
    precision_inverse: np.ndarray | None = None,
    precision_used_pseudoinverse: bool = False,
) -> PortfolioSolution:
    """Solve the Static-ML* quadratic objective used in the thesis.

    ``u`` scales expected returns, ``v`` adds diagonal covariance shrinkage,
    and ``k`` scales quadratic trading impact.
    """
    mu = np.asarray(expected_returns, dtype=float) * float(u)
    inherited = np.asarray(inherited_weights, dtype=float)
    lambdas = np.asarray(lambda_adv, dtype=float)
    if (lambdas <= 0).any():
        raise ValueError("lambda_adv must be positive")
    tc_precision = float(aum) * float(k) * np.diag(lambdas)
    rhs = mu + tc_precision @ inherited
    used_pseudoinverse = bool(precision_used_pseudoinverse)
    if precision_inverse is not None:
        weights = np.asarray(precision_inverse, dtype=float) @ rhs
    else:
        lhs = static_ml_star_precision_matrix(
            covariance,
            lambdas,
            gamma=gamma,
            aum=aum,
            v=v,
            k=k,
            ridge=ridge,
        )
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs, hermitian=True) @ rhs
            used_pseudoinverse = True
    return _solution(weights, used_pseudoinverse)


def static_ml_precision_matrix(
    covariance: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    phi: float,
    ridge: float,
) -> np.ndarray:
    lambdas = np.asarray(lambda_adv, dtype=float)
    return (
        float(gamma) * _regularized_covariance(covariance, ridge)
        + float(aum) / float(phi) * np.diag(lambdas)
    )


def static_ml_star_precision_matrix(
    covariance: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    v: float,
    k: float,
    ridge: float,
) -> np.ndarray:
    sigma = np.asarray(covariance, dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    sigma_adjusted = sigma + float(v) * np.diag(np.diag(sigma))
    return (
        float(gamma) * _regularized_covariance(sigma_adjusted, ridge)
        + float(aum) * float(k) * np.diag(np.asarray(lambda_adv, dtype=float))
    )


def compute_adjustment_matrix(
    covariance: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    expected_market_return: float,
    risk_free_rate: float,
    iterations: int,
    allow_scalar_fallback: bool,
    scalar_rho: float,
) -> AdjustmentMatrixResult:
    try:
        matrix = adjustment_matrix_from_risk(
            covariance,
            lambda_adv,
            gamma=gamma,
            aum=aum,
            expected_market_return=expected_market_return,
            risk_free_rate=risk_free_rate,
            iterations=iterations,
        )
        eigenvalues = np.linalg.eigvals(matrix)
        radius = float(np.max(np.abs(eigenvalues)))
        if not np.isfinite(matrix).all() or radius > 1.5:
            raise np.linalg.LinAlgError(f"matrix m is unstable (spectral radius={radius:.4g})")
        return AdjustmentMatrixResult(
            matrix, "matrix_m", radius, eigenvalues=eigenvalues
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        if not allow_scalar_fallback:
            raise
        matrix = np.eye(len(lambda_adv)) * float(scalar_rho)
        return AdjustmentMatrixResult(
            matrix,
            "scalar_rho_fallback",
            float(scalar_rho),
            str(exc),
            np.full(len(lambda_adv), float(scalar_rho)),
        )


def _regularized_covariance(covariance: np.ndarray, ridge: float) -> np.ndarray:
    sigma = np.asarray(covariance, dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    return sigma + np.eye(sigma.shape[0]) * float(ridge)


def _limit_leverage(weights: np.ndarray, max_leverage: float | None) -> np.ndarray:
    if max_leverage is None:
        return weights
    leverage = float(np.abs(weights).sum())
    return weights * (max_leverage / leverage) if leverage > max_leverage else weights


def _solution(weights: np.ndarray, used_pseudoinverse: bool) -> PortfolioSolution:
    return PortfolioSolution(
        weights=np.asarray(weights, dtype=float),
        leverage=float(np.abs(weights).sum()),
        max_abs_weight=float(np.max(np.abs(weights))) if len(weights) else 0.0,
        used_pseudoinverse=used_pseudoinverse,
    )
