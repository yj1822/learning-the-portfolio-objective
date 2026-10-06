"""Monthly portfolio adjustment implied by the thesis trading objective.

The equations implemented here are stated in Jiang (2026), Section 3.7.1,
following the dynamic Portfolio-ML framework of Jensen et al. (2024).
"""

from __future__ import annotations

import numpy as np


def adjustment_matrix_from_risk(
    covariance: np.ndarray,
    lambda_adv: np.ndarray,
    *,
    gamma: float,
    aum: float,
    expected_market_return: float,
    risk_free_rate: float,
    iterations: int,
) -> np.ndarray:
    """Return the matrix that carries inherited holdings into the next decision.

    ``lambda_adv`` is the diagonal of the quadratic impact matrix. The fixed
    number of updates comes from the model configuration, not a convergence
    rule chosen after observing portfolio results.
    """
    risk = np.asarray(covariance, dtype=float)
    impact = np.asarray(lambda_adv, dtype=float)
    count = len(impact)
    if risk.shape != (count, count):
        raise ValueError("covariance/lambda shape mismatch")
    if (impact <= 0).any() or gamma <= 0 or aum < 0:
        raise ValueError("lambda_adv and gamma must be positive; aum must be non-negative")
    if aum == 0:
        return np.zeros((count, count), dtype=float)

    market_growth = 1.0 + risk_free_rate + expected_market_return
    if np.allclose(risk, np.diag(np.diag(risk)), rtol=0.0, atol=1.0e-14):
        return _diagonal_adjustment(
            np.diag(risk), impact, gamma, aum, market_growth, iterations
        )

    impact_scale = np.sqrt(impact)
    scaled_risk = (gamma / aum) * risk / impact_scale[:, None] / impact_scale[None, :]
    growth_covariance = np.ones_like(risk) + risk / market_growth**2
    diagonal_term = 1.0 + np.diag(growth_covariance)
    policy = _initial_dense_policy(scaled_risk)

    for _ in range(int(iterations)):
        system = scaled_risk - policy * growth_covariance
        system[np.diag_indices(count)] += diagonal_term
        policy = np.linalg.inv(system)

    return policy * (impact_scale[None, :] / impact_scale[:, None])


def _diagonal_adjustment(
    variances: np.ndarray,
    impact: np.ndarray,
    gamma: float,
    aum: float,
    market_growth: float,
    iterations: int,
) -> np.ndarray:
    """Evaluate the same fixed point asset by asset for diagonal covariance."""
    scaled_risk = gamma * variances / (aum * impact)
    growth_variance = 1.0 + variances / market_growth**2
    shifted = scaled_risk + 2.0
    initial_root = np.sqrt(np.maximum(shifted**2 - 4.0, 0.0))
    diagonal_policy = (shifted - initial_root) / 2.0
    for _ in range(int(iterations)):
        diagonal_policy = 1.0 / (
            scaled_risk + 1.0 + growth_variance - diagonal_policy * growth_variance
        )
    return np.diag(diagonal_policy)


def _initial_dense_policy(scaled_risk: np.ndarray) -> np.ndarray:
    """Use the positive semidefinite matrix root for the initial policy."""
    identity = np.eye(len(scaled_risk))
    shifted = scaled_risk + 2.0 * identity
    square_root_input = shifted @ shifted - 4.0 * identity
    square_root_input = (square_root_input + square_root_input.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(square_root_input)
    if eigenvalues.min() < -1.0e-7:
        raise np.linalg.LinAlgError(
            f"matrix square-root argument is not PSD (min eigenvalue={eigenvalues.min():.4g})"
        )
    square_root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    return (shifted - square_root) / 2.0
