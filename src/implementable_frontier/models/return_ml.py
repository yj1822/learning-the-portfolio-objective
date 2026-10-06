from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from implementable_frontier.models.features import RandomFourierFeatures


@dataclass(frozen=True)
class ReturnModelSpec:
    kind: str
    ridge_lambda: float
    rf_p: int | None = None
    rf_eta: float | None = None
    seed: int = 0
    rf_max_p: int | None = None

    def metadata(self) -> dict[str, object]:
        return asdict(self)


class RidgeReturnModel:
    def __init__(
        self,
        spec: ReturnModelSpec,
        feature_columns: Sequence[str],
        *,
        design_cache: Any | None = None,
        cache_context: dict[str, object] | None = None,
    ) -> None:
        self.spec = spec
        self.feature_columns = list(feature_columns)
        self.rff: RandomFourierFeatures | None = None
        self.design_cache = design_cache
        self.cache_context = cache_context or {}

    def fit(self, frame: pd.DataFrame, target_column: str = "ret_exc_lead1m") -> "RidgeReturnModel":
        valid_target = pd.to_numeric(frame[target_column], errors="coerce").notna()
        train = frame.loc[valid_target].reset_index(drop=True)
        if train.empty:
            raise ValueError("Return-ML training data has no finite targets")
        design = self._raw_design(train, fit=True)
        if self.spec.kind == "rf_ridge":
            # Match the thesis's unstandardized random-feature ridge design;
            # the intercept remains unpenalized.
            self.mean_ = np.zeros(design.shape[1])
            self.scale_ = np.ones(design.shape[1])
        else:
            self.mean_ = design.mean(axis=0)
            self.scale_ = design.std(axis=0)
            self.scale_[self.scale_ < 1.0e-12] = 1.0
        standardized = (design - self.mean_) / self.scale_
        x = np.column_stack([np.ones(len(standardized)), standardized])
        y = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)
        penalty = np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.base_gram_ = x.T @ x / len(x)
        self.penalty_ = penalty
        self.rhs_ = x.T @ y / len(x)
        self.solve_ridge(self.spec.ridge_lambda)
        self.train_start_ = pd.to_datetime(train["eom"]).min()
        self.train_end_ = pd.to_datetime(train["eom"]).max()
        return self

    def solve_ridge(self, ridge_lambda: float) -> "RidgeReturnModel":
        if not hasattr(self, "base_gram_"):
            raise RuntimeError("Return-ML sufficient statistics are unavailable")
        gram = self.base_gram_ + float(ridge_lambda) * self.penalty_
        self.coef_ = np.linalg.pinv(gram, hermitian=True) @ self.rhs_
        self.ridge_lambda_ = float(ridge_lambda)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model must be fitted before predict")
        design = self._raw_design(frame.reset_index(drop=True), fit=False)
        standardized = (design - self.mean_) / self.scale_
        x = np.column_stack([np.ones(len(standardized)), standardized])
        return x @ self.coef_

    def _raw_design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if self.spec.kind == "ridge":
            values = frame[self.feature_columns].to_numpy(dtype=float)
            return np.nan_to_num(values, nan=0.5)
        if self.spec.kind != "rf_ridge":
            raise ValueError(f"Unsupported Return-ML model: {self.spec.kind}")
        if fit:
            self.rff = RandomFourierFeatures(
                input_dim=len(self.feature_columns),
                n_features=int(self.spec.rf_p or 0),
                eta=float(self.spec.rf_eta or 0),
                seed=self.spec.seed,
                max_features=int(self.spec.rf_max_p or self.spec.rf_p or 0),
            )
        if self.rff is None:
            raise RuntimeError("Random feature map is unavailable")
        if self.design_cache is None or "eom" not in frame or "permno" not in frame:
            raw = frame[self.feature_columns].to_numpy(dtype=float)
            raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)
            return self.rff.transform(raw)
        output = np.zeros((len(frame), int(self.spec.rf_p or 0)), dtype=float)
        dates = pd.to_datetime(frame["eom"])
        for eom, positions in frame.groupby(dates, sort=True).indices.items():
            position_array = np.asarray(positions, dtype=int)
            month = frame.iloc[position_array]
            raw = month[self.feature_columns].to_numpy(dtype=float)
            raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)
            key = (
                str(self.cache_context.get("feature_set", "unknown")),
                str(self.cache_context.get("universe", "unknown")),
                pd.Timestamp(eom),
                int(self.spec.rf_p or 0),
                float(self.spec.rf_eta or 0.0),
                int(self.spec.seed),
                str(self.cache_context.get("rf_version", "official_cos_sin_v1")),
                "return_ml",
            )
            output[position_array] = self.design_cache.get_or_compute(
                key,
                month["permno"].astype(int).to_numpy(),
                lambda raw=raw: self.rff.transform(raw),
            )
        return output


def return_model_specs(config: dict[str, object], seed: int) -> list[ReturnModelSpec]:
    max_p = max(int(value) for value in config["rf_p_grid"])
    specs = [
        ReturnModelSpec("ridge", float(ridge_lambda), seed=seed)
        for ridge_lambda in config["lambda_grid"]
    ]
    if "rf_ridge" in config.get("models", []):
        specs.extend(
            ReturnModelSpec(
                "rf_ridge",
                float(ridge_lambda),
                rf_p=int(p),
                rf_eta=float(eta),
                seed=seed,
                rf_max_p=max_p,
            )
            for p in config["rf_p_grid"]
            for eta in config["rf_eta_grid"]
            for ridge_lambda in config["lambda_grid"]
        )
    return specs
