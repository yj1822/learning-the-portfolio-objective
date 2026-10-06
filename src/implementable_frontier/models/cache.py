from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from implementable_frontier.models.portfolios import AdjustmentMatrixResult


@dataclass(frozen=True)
class PrecisionInverseResult:
    matrix: np.ndarray
    used_pseudoinverse: bool


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    disk_hits: int = 0
    writes: int = 0

    def as_dict(self) -> dict[str, float | int]:
        requests = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "disk_hits": self.disk_hits,
            "writes": self.writes,
            "requests": requests,
            "hit_rate": self.hits / requests if requests else 0.0,
        }


class StaticPrecisionCache:
    """Year-local inverse cache for repeated Static-ML precision systems."""

    def __init__(self) -> None:
        self.memory: dict[tuple[object, ...], PrecisionInverseResult] = {}
        self.stats = CacheStats()

    def get_or_compute(
        self,
        key: tuple[object, ...],
        compute: Callable[[], np.ndarray],
    ) -> PrecisionInverseResult:
        if key in self.memory:
            self.stats.hits += 1
            return self.memory[key]
        self.stats.misses += 1
        precision = np.asarray(compute(), dtype=float)
        try:
            inverse = np.linalg.inv(precision)
            result = PrecisionInverseResult(inverse, False)
        except np.linalg.LinAlgError:
            result = PrecisionInverseResult(
                np.linalg.pinv(precision, hermitian=True), True
            )
        self.memory[key] = result
        return result

    def clear(self) -> None:
        self.memory.clear()

    def info(self) -> dict[str, float | int]:
        return self.stats.as_dict()


class PersistentAdjustmentCache(
    MutableMapping[tuple[object, ...], AdjustmentMatrixResult]
):
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.memory: dict[tuple[object, ...], AdjustmentMatrixResult] = {}
        self.stats = CacheStats()

    def __getitem__(self, key: tuple[object, ...]) -> AdjustmentMatrixResult:
        if key in self.memory:
            self.stats.hits += 1
            return self.memory[key]
        path = self._path(key)
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as values:
                fallback = str(values["fallback_reason"].item())
                matrix = values["matrix"]
                eigenvalues = (
                    values["eigenvalues"]
                    if "eigenvalues" in values.files
                    else np.linalg.eigvals(matrix)
                )
                result = AdjustmentMatrixResult(
                    matrix=matrix,
                    mode=str(values["mode"].item()),
                    spectral_radius=float(values["spectral_radius"].item()),
                    fallback_reason=fallback or None,
                    eigenvalues=eigenvalues,
                )
            self.memory[key] = result
            self.stats.hits += 1
            self.stats.disk_hits += 1
            return result
        self.stats.misses += 1
        raise KeyError(key)

    def __setitem__(
        self, key: tuple[object, ...], value: AdjustmentMatrixResult
    ) -> None:
        self.memory[key] = value
        path = self._path(key)
        if path is not None:
            np.savez_compressed(
                path,
                matrix=value.matrix,
                mode=np.array(value.mode),
                spectral_radius=np.array(value.spectral_radius),
                fallback_reason=np.array(value.fallback_reason or ""),
                eigenvalues=(
                    value.eigenvalues
                    if value.eigenvalues is not None
                    else np.linalg.eigvals(value.matrix)
                ),
            )
            self.stats.writes += 1

    def __delitem__(self, key: tuple[object, ...]) -> None:
        del self.memory[key]
        path = self._path(key)
        if path is not None and path.exists():
            path.unlink()

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        return iter(self.memory)

    def __len__(self) -> int:
        return len(self.memory)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple):
            return False
        if key in self.memory:
            return True
        path = self._path(key)
        return bool(path is not None and path.exists())

    def info(self) -> dict[str, float | int]:
        return self.stats.as_dict()

    def _path(self, key: tuple[object, ...]) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{_key_hash(key)}.npz"


class DesignMatrixCache:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.memory: dict[tuple[object, ...], tuple[np.ndarray, tuple[int, ...]]] = {}
        self.stats = CacheStats()

    def get_or_compute(
        self,
        key: tuple[object, ...],
        permnos: np.ndarray,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        ids = tuple(int(value) for value in permnos)
        # Return-ML fitting can legitimately use a finite-label subset of a
        # month while prediction uses the full month. Keep those designs apart.
        storage_key = (*key, _id_fingerprint(ids))
        if storage_key in self.memory:
            values, stored_ids = self.memory[storage_key]
            if stored_ids != ids:
                raise ValueError("RF cache permno order mismatch")
            self.stats.hits += 1
            return values
        path = self._path(storage_key)
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as stored:
                stored_ids = tuple(int(value) for value in stored["permnos"])
                if stored_ids != ids:
                    raise ValueError("Persisted RF cache permno order mismatch")
                values = stored["design"]
            self.memory[storage_key] = (values, stored_ids)
            self.stats.hits += 1
            self.stats.disk_hits += 1
            return values
        self.stats.misses += 1
        values = np.asarray(compute(), dtype=float)
        self.memory[storage_key] = (values, ids)
        if path is not None:
            np.savez_compressed(path, design=values, permnos=np.asarray(ids, dtype=np.int64))
            self.stats.writes += 1
        return values

    def info(self) -> dict[str, float | int]:
        return self.stats.as_dict()

    def _path(self, key: tuple[object, ...]) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{_key_hash(key)}.npz"


class TransactionCostCache:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.memory: dict[tuple[object, ...], tuple[pd.Series, tuple[int, ...]]] = {}
        self.stats = CacheStats()

    def get(
        self,
        *,
        universe: str,
        eom: pd.Timestamp,
        aum: float,
        cost_model_version: str,
        permnos: np.ndarray,
        values: np.ndarray,
    ) -> pd.Series:
        key = (universe, pd.Timestamp(eom), float(aum), cost_model_version)
        ids = tuple(int(value) for value in permnos)
        if key in self.memory:
            series, stored_ids = self.memory[key]
            if stored_ids != ids:
                raise ValueError("Transaction-cost cache permno order mismatch")
            self.stats.hits += 1
            return series.copy()
        path = self._path(key)
        if path is not None and path.exists():
            frame = pd.read_parquet(path)
            stored_ids = tuple(frame["permno"].astype(int))
            if stored_ids != ids:
                raise ValueError("Persisted transaction-cost permno order mismatch")
            series = pd.Series(frame["lambda_adv"].to_numpy(float), index=ids)
            self.memory[key] = (series, ids)
            self.stats.hits += 1
            self.stats.disk_hits += 1
            return series.copy()
        self.stats.misses += 1
        series = pd.Series(np.asarray(values, dtype=float), index=ids)
        self.memory[key] = (series, ids)
        if path is not None:
            pd.DataFrame(
                {"permno": np.asarray(ids, dtype=np.int64), "lambda_adv": series.to_numpy()}
            ).to_parquet(path, index=False)
            self.stats.writes += 1
        return series.copy()

    def info(self) -> dict[str, float | int]:
        return self.stats.as_dict()

    def _path(self, key: tuple[object, ...]) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{_key_hash(key)}.parquet"


def _key_hash(key: tuple[object, ...]) -> str:
    serialized = json.dumps(key, default=str, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _id_fingerprint(ids: tuple[int, ...]) -> str:
    values = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()[:20]
