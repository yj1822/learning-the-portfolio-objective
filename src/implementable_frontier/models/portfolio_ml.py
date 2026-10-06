from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import MutableMapping
from typing import Any, Sequence

import numpy as np
import pandas as pd

from implementable_frontier.models.features import (
    RandomFourierFeatures,
    monthly_cross_sectional_standardize,
    volatility_scale_design,
)
from implementable_frontier.models.portfolios import (
    AdjustmentMatrixResult,
    compute_adjustment_matrix,
)
from implementable_frontier.models.risk import RiskModelProvider


AdjustmentCache = MutableMapping[tuple[object, ...], AdjustmentMatrixResult]


@dataclass(frozen=True)
class PortfolioMLSpec:
    kind: str
    ridge_lambda: float
    rf_p: int | None = None
    rf_eta: float | None = None
    seed: int = 0

    def metadata(self) -> dict[str, object]:
        return asdict(self)


class PortfolioMLModel:
    """Closed-form net-utility Portfolio-ML with dynamic-universe B recursion."""

    def __init__(
        self,
        spec: PortfolioMLSpec,
        feature_columns: Sequence[str],
        risk_provider: RiskModelProvider,
        *,
        gamma: float,
        aum: float,
        settings: dict[str, object],
        adjustment_cache: AdjustmentCache | None = None,
        design_cache: Any | None = None,
        transaction_cache: Any | None = None,
    ) -> None:
        self.spec = spec
        self.feature_columns = list(feature_columns)
        self.risk_provider = risk_provider
        self.gamma = float(gamma)
        self.aum = float(aum)
        self.settings = settings
        self.adjustment_cache = adjustment_cache if adjustment_cache is not None else {}
        self.design_cache = design_cache
        self.transaction_cache = transaction_cache
        self.rff = (
            RandomFourierFeatures(
                len(self.feature_columns),
                int(spec.rf_p or 0),
                float(spec.rf_eta or 0),
                spec.seed,
                max_features=max(int(value) for value in settings["rf_p_grid"]),
            )
            if spec.kind == "rf"
            else None
        )
        self.diagnostics: list[str] = []
        self.adjustment_modes: set[str] = set()

    def fit(self, frame: pd.DataFrame) -> "PortfolioMLModel":
        data = frame.sort_values(["eom", "permno"]).copy()
        numerator: np.ndarray | None = None
        denominator: np.ndarray | None = None
        observations = 0
        previous_b: pd.DataFrame | None = None
        previous_returns = pd.Series(dtype=float)
        previous_lambda = pd.Series(dtype=float)
        previous_market_growth: float | None = None
        for eom, month in data.groupby("eom", sort=True):
            month = month.sort_values("permno").reset_index(drop=True)
            permnos = month["permno"].astype(int).to_numpy()
            covariance = self.risk_provider.get_covariance(eom, permnos)
            raw_lambdas = pd.to_numeric(
                month["lambda_adv"], errors="coerce"
            ).to_numpy(float)
            if self.transaction_cache is not None:
                lambda_series = self.transaction_cache.get(
                    universe=self.risk_provider.universe,
                    eom=pd.Timestamp(eom),
                    aum=self.aum,
                    cost_model_version=str(
                        self.settings.get("cost_model_version", "adv_quadratic_v1")
                    ),
                    permnos=permnos,
                    values=raw_lambdas,
                )
                lambdas = lambda_series.to_numpy(float)
            else:
                lambdas = raw_lambdas
            if not np.isfinite(lambdas).all() or (lambdas <= 0).any():
                raise ValueError(f"Portfolio-ML has invalid lambda_adv at {pd.Timestamp(eom).date()}")
            design = self.design_matrix(month)
            adjustment = self.adjustment_matrix(
                covariance.covariance, lambdas, eom=pd.Timestamp(eom)
            )
            self.adjustment_modes.add(adjustment.mode)
            if adjustment.fallback_reason:
                self.diagnostics.append(
                    f"{pd.Timestamp(eom).date()}: matrix m fallback: {adjustment.fallback_reason}"
                )
            inherited, dropped = _inherit_loading_matrix(
                previous_b,
                previous_returns,
                permnos,
                n_columns=design.shape[1],
                floor=0.1,
                wealth_growth=previous_market_growth,
            )
            identity_minus_m = np.eye(len(month)) - adjustment.matrix
            b_matrix = adjustment.matrix @ inherited + identity_minus_m @ design
            delta = b_matrix - inherited
            sigma_tilde = self.gamma * (
                b_matrix.T @ covariance.covariance @ b_matrix
            ) + self.aum * (delta.T @ (lambdas[:, None] * delta))
            if not dropped.empty:
                dropped_lambda = previous_lambda.reindex(dropped.index)
                fallback = float(np.nanmedian(lambdas))
                dropped_lambda = dropped_lambda.fillna(fallback).to_numpy(float)
                dropped_values = dropped.to_numpy(float)
                sigma_tilde += self.aum * (
                    dropped_values.T @ (dropped_lambda[:, None] * dropped_values)
                )
            targets = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce").to_numpy(float)
            finite = np.isfinite(targets)
            if finite.any():
                r_tilde = b_matrix[finite].T @ targets[finite]
                numerator = r_tilde if numerator is None else numerator + r_tilde
                denominator = sigma_tilde if denominator is None else denominator + sigma_tilde
                observations += 1
            previous_b = pd.DataFrame(b_matrix, index=permnos)
            previous_returns = pd.Series(targets, index=permnos)
            previous_lambda = pd.Series(lambdas, index=permnos)
            previous_market_growth = _market_growth(month, targets)
        if observations == 0 or numerator is None or denominator is None:
            raise ValueError("Portfolio-ML training period has no usable monthly targets")
        mean_denominator = denominator / observations
        mean_numerator = numerator / observations
        self.mean_denominator_ = mean_denominator
        self.mean_numerator_ = mean_numerator
        self.solve_beta(float(self.spec.ridge_lambda))
        self.train_start_ = pd.to_datetime(data["eom"]).min()
        self.train_end_ = pd.to_datetime(data["eom"]).max()
        self.training_months_ = observations
        self.final_b_ = previous_b
        self.final_returns_ = previous_returns
        self.final_lambda_ = previous_lambda
        self.final_market_growth_ = previous_market_growth
        return self

    def solve_beta(self, ridge_lambda: float) -> "PortfolioMLModel":
        if not hasattr(self, "mean_denominator_"):
            raise RuntimeError("Portfolio-ML sufficient statistics are unavailable")
        ridge = max(
            float(ridge_lambda),
            float(self.settings.get("coefficient_ridge_floor", 1.0e-10)),
        )
        lhs = (self.mean_denominator_ + self.mean_denominator_.T) / 2.0 + ridge * np.eye(
            len(self.mean_numerator_)
        )
        try:
            self.beta_ = np.linalg.solve(lhs, self.mean_numerator_)
            self.used_pseudoinverse_ = False
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.pinv(lhs, hermitian=True) @ self.mean_numerator_
            self.used_pseudoinverse_ = True
        self.denominator_condition_ = float(np.linalg.cond(lhs))
        self.ridge_lambda_ = ridge
        return self

    def quadratic_hessian(self) -> np.ndarray:
        if not hasattr(self, "mean_denominator_"):
            raise RuntimeError("Portfolio-ML sufficient statistics are unavailable")
        symmetric = (self.mean_denominator_ + self.mean_denominator_.T) / 2.0
        return symmetric + float(self.ridge_lambda_) * np.eye(len(self.mean_numerator_))

    def objective(self, beta: np.ndarray | None = None) -> float:
        values = self.beta_ if beta is None else np.asarray(beta, dtype=float)
        hessian = self.quadratic_hessian()
        return float(self.mean_numerator_ @ values - 0.5 * values @ hessian @ values)

    def gradient(self, beta: np.ndarray | None = None) -> np.ndarray:
        values = self.beta_ if beta is None else np.asarray(beta, dtype=float)
        return self.mean_numerator_ - self.quadratic_hessian() @ values

    def design_matrix(self, month: pd.DataFrame) -> np.ndarray:
        raw = month[self.feature_columns].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)
        volatility = self.risk_provider.get_volatility(
            pd.Timestamp(month["eom"].iloc[0]), month["permno"].astype(int).to_numpy()
        )
        floor = float(self.settings.get("volatility_floor", 0.01))
        missing = int((~(np.isfinite(volatility) & (volatility >= floor))).sum())
        def compute() -> np.ndarray:
            transformed = self.rff.transform(raw) if self.rff is not None else raw
            standardized = monthly_cross_sectional_standardize(
                transformed,
                add_constant=bool(self.settings.get("add_constant", True)),
            )
            scaled, _ = volatility_scale_design(
                standardized,
                volatility,
                floor=floor,
            )
            return scaled

        if self.design_cache is not None:
            eom = pd.Timestamp(month["eom"].iloc[0])
            key = (
                str(self.settings.get("feature_set", "unknown")),
                self.risk_provider.universe,
                eom,
                int(self.spec.rf_p or 0),
                float(self.spec.rf_eta or 0.0),
                int(self.spec.seed),
                str(self.settings.get("rf_version", "official_cos_sin_v1")),
            )
            scaled = self.design_cache.get_or_compute(
                key,
                month["permno"].astype(int).to_numpy(),
                compute,
            )
        else:
            scaled = compute()
        if missing:
            self.diagnostics.append(
                f"{pd.Timestamp(month['eom'].iloc[0]).date()}: volatility fallback for {missing} names"
            )
        return scaled

    def aim_weights(self, month: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "beta_"):
            raise RuntimeError("Portfolio-ML model must be fitted before prediction")
        return self.design_matrix(month) @ self.beta_

    def adjustment_matrix(
        self,
        covariance: np.ndarray,
        lambda_adv: np.ndarray,
        *,
        eom: pd.Timestamp | None = None,
    ) -> AdjustmentMatrixResult:
        if not bool(self.settings.get("use_matrix_m", True)):
            rho = float(self.settings.get("rho_grid_if_fallback", [0.5])[0])
            return AdjustmentMatrixResult(
                np.eye(len(lambda_adv)) * rho,
                "scalar_rho_configured",
                rho,
                "matrix m disabled by configuration",
                np.full(len(lambda_adv), rho),
            )
        key = (
            self.risk_provider.universe,
            pd.Timestamp(eom) if eom is not None else pd.Timestamp.min,
            self.risk_provider.requested_model,
            self.risk_provider.factor_model_version,
            self.aum,
            self.gamma,
            str(self.settings.get("cost_model_version", "adv_quadratic_v1")),
        )
        if eom is not None:
            try:
                return self.adjustment_cache[key]
            except KeyError:
                pass
        result = compute_adjustment_matrix(
            covariance,
            lambda_adv,
            gamma=self.gamma,
            aum=self.aum,
            expected_market_return=float(self.settings["expected_market_return"]),
            risk_free_rate=float(self.settings["risk_free_rate"]),
            iterations=int(self.settings["matrix_m_iterations"]),
            allow_scalar_fallback=bool(self.settings["allow_scalar_rho_fallback"]),
            scalar_rho=float(self.settings.get("rho_grid_if_fallback", [0.5])[0]),
        )
        if eom is not None:
            self.adjustment_cache[key] = result
        return result


def portfolio_ml_specs(config: dict[str, object], kind: str, seed: int) -> list[PortfolioMLSpec]:
    if kind == "linear":
        return [
            PortfolioMLSpec("linear", float(value), seed=seed)
            for value in config["lambda_grid"]
        ]
    return [
        PortfolioMLSpec("rf", float(ridge), int(p), float(eta), seed)
        for p in config["rf_p_grid"]
        for eta in config["rf_eta_grid"]
        for ridge in config["lambda_grid"]
    ]


def _inherit_loading_matrix(
    previous_b: pd.DataFrame | None,
    previous_returns: pd.Series,
    current_permnos: np.ndarray,
    *,
    n_columns: int,
    floor: float,
    wealth_growth: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if previous_b is None:
        return np.zeros((len(current_permnos), n_columns)), pd.DataFrame()
    returns = previous_returns.reindex(previous_b.index).fillna(0.0)
    benchmark_growth = max(
        float(wealth_growth)
        if wealth_growth is not None
        else 1.0 + float(returns.mean()),
        floor,
    )
    grown = previous_b.mul((1.0 + returns) / benchmark_growth, axis=0)
    inherited = grown.reindex(current_permnos).fillna(0.0)
    dropped = grown.loc[~grown.index.isin(current_permnos)]
    return inherited.to_numpy(float), dropped


def _market_growth(month: pd.DataFrame, returns: np.ndarray) -> float:
    if "lag_mktcap" in month:
        caps = pd.to_numeric(month["lag_mktcap"], errors="coerce").to_numpy(float)
        valid = np.isfinite(caps) & (caps > 0) & np.isfinite(returns)
        if valid.any() and caps[valid].sum() > 0:
            return max(1.0 + float(np.average(returns[valid], weights=caps[valid])), 0.1)
    finite = np.isfinite(returns)
    return max(1.0 + float(np.mean(returns[finite])) if finite.any() else 1.0, 0.1)
