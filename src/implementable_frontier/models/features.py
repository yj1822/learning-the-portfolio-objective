from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RandomFourierFeatures:
    input_dim: int
    n_features: int
    eta: float
    seed: int
    max_features: int | None = None

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.n_features < 2 or self.n_features % 2:
            raise ValueError("n_features must be an even integer")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        bank_features = int(self.max_features or self.n_features)
        if bank_features < self.n_features or bank_features % 2:
            raise ValueError("max_features must be even and at least n_features")
        rng = np.random.default_rng(self.seed)
        weights = rng.normal(
            loc=0.0,
            scale=self.eta,
            size=(self.input_dim, bank_features // 2),
        )
        object.__setattr__(self, "weights", weights)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected a 2D array with {self.input_dim} columns, got {x.shape}"
            )
        projection = x @ self.weights[:, : self.n_features // 2]
        return np.concatenate([np.cos(projection), np.sin(projection)], axis=1) / np.sqrt(
            self.n_features
        )

    def matrix_hash(self) -> str:
        active = np.ascontiguousarray(self.weights[:, : self.n_features // 2])
        return hashlib.sha256(active.tobytes()).hexdigest()


def monthly_cross_sectional_standardize(
    values: np.ndarray,
    *,
    add_constant: bool,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Demean and scale each signal column to unit cross-sectional L2 norm."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be a 2D array")
    centered = x - np.nanmean(x, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.sqrt(np.sum(centered**2, axis=0))
    scaled = np.divide(
        centered,
        norms,
        out=np.zeros_like(centered),
        where=norms > epsilon,
    )
    if not add_constant:
        return scaled
    constant = np.full((len(scaled), 1), 1.0 / np.sqrt(max(len(scaled), 1)))
    return np.concatenate([constant, scaled], axis=1)


def transform_panel_monthly(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    rff: RandomFourierFeatures | None = None,
    add_constant: bool = False,
    date_column: str = "eom",
) -> np.ndarray:
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    output_width = (rff.n_features if rff is not None else len(feature_columns)) + int(
        add_constant
    )
    output = np.zeros((len(frame), output_width), dtype=float)
    dates = pd.to_datetime(frame[date_column])
    for _, positions in frame.groupby(dates, sort=True).indices.items():
        position_array = np.asarray(positions, dtype=int)
        raw = frame.iloc[position_array][list(feature_columns)].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.5)
        design = rff.transform(raw) if rff is not None else raw
        output[position_array] = monthly_cross_sectional_standardize(
            design, add_constant=add_constant
        )
    return output


def volatility_scale_design(
    design: np.ndarray,
    volatility: Sequence[float],
    *,
    floor: float,
) -> tuple[np.ndarray, int]:
    vol = np.asarray(volatility, dtype=float)
    valid = np.isfinite(vol) & (vol >= floor)
    fallback = float(np.nanmedian(vol[valid])) if valid.any() else max(floor, 0.05)
    cleaned = np.where(valid, vol, fallback)
    return design / cleaned[:, None], int((~valid).sum())
