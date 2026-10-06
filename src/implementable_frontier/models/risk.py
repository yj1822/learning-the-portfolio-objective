from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from implementable_frontier.data.loaders import load_risk_inputs
from implementable_frontier.models.cache import CacheStats, _key_hash


@dataclass(frozen=True)
class CovarianceResult:
    covariance: np.ndarray
    requested_model: str
    used_model: str
    min_eigenvalue: float
    condition_number: float
    missing_assets: tuple[int, ...]
    fallback_reason: str | None = None
    diagonal_min: float = float("nan")
    diagonal_median: float = float("nan")
    diagonal_max: float = float("nan")
    factor_covariance_diagonal_fallback: bool = False
    idio_imputed_assets: tuple[int, ...] = ()
    diagnostic_notes: tuple[str, ...] = ()
    covariance_repair_count: int = 0


class RiskModelProvider:
    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        universe: str,
        requested_model: str,
        settings: dict[str, object],
        risk_inputs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.panel = panel.copy()
        self.panel["eom"] = pd.to_datetime(self.panel["eom"])
        self.universe = universe
        self.requested_model = requested_model
        self.settings = settings
        self.risk_inputs = risk_inputs
        self.factor_model_version = str(
            settings.get("factor_model_version", "risk_repaired_v1")
        )
        self._cache: dict[tuple[object, ...], CovarianceResult] = {}
        persistent_root = settings.get("persistent_cache_dir")
        self._persistent_cache_dir = (
            Path(str(persistent_root)).resolve() if persistent_root else None
        )
        if self._persistent_cache_dir is not None:
            self._persistent_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_stats = CacheStats()
        self._panel_months = {
            pd.Timestamp(eom): group.set_index("permno", drop=False)
            for eom, group in self.panel.groupby("eom", sort=True)
        }

    def get_covariance(
        self,
        eom: str | pd.Timestamp,
        permnos: Sequence[int],
        risk_model: str | None = None,
    ) -> CovarianceResult:
        requested = risk_model or self.requested_model
        timestamp = pd.Timestamp(eom)
        ids = tuple(int(value) for value in permnos)
        cache_key = (
            self.universe,
            timestamp,
            requested,
            self.factor_model_version,
            ids,
        )
        if cache_key in self._cache:
            self.cache_stats.hits += 1
            return self._cache[cache_key]
        persistent_path = self._persistent_path(cache_key)
        if persistent_path is not None and persistent_path.exists():
            result = _load_covariance_result(persistent_path)
            self._cache[cache_key] = result
            self.cache_stats.hits += 1
            self.cache_stats.disk_hits += 1
            return result
        self.cache_stats.misses += 1
        if requested == "factor":
            try:
                result = self._factor_covariance(timestamp, ids)
            except (FileNotFoundError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
                if not bool(self.settings.get("allow_factor_fallback", True)):
                    raise
                diagonal = self._diagonal_covariance(timestamp, ids)
                result = CovarianceResult(
                    covariance=diagonal.covariance,
                    requested_model="factor",
                    used_model="diagonal",
                    min_eigenvalue=diagonal.min_eigenvalue,
                    condition_number=diagonal.condition_number,
                    missing_assets=diagonal.missing_assets,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    diagonal_min=diagonal.diagonal_min,
                    diagonal_median=diagonal.diagonal_median,
                    diagonal_max=diagonal.diagonal_max,
                    diagnostic_notes=("Full factor model failed; used vol_12m diagonal covariance.",),
                )
        else:
            result = self._diagonal_covariance(timestamp, ids)
        self._cache[cache_key] = result
        if persistent_path is not None:
            _save_covariance_result(persistent_path, result)
            self.cache_stats.writes += 1
        return result

    def cache_info(self) -> dict[str, float | int | str]:
        return {
            **self.cache_stats.as_dict(),
            "universe": self.universe,
            "risk_model": self.requested_model,
            "factor_model_version": self.factor_model_version,
        }

    def _persistent_path(self, key: tuple[object, ...]) -> Path | None:
        if self._persistent_cache_dir is None:
            return None
        return self._persistent_cache_dir / f"{_key_hash(key)}.npz"

    def get_volatility(
        self, eom: str | pd.Timestamp, permnos: Sequence[int]
    ) -> np.ndarray:
        result = self.get_covariance(eom, permnos)
        return np.sqrt(np.diag(result.covariance))

    def _diagonal_covariance(
        self, eom: pd.Timestamp, permnos: Sequence[int]
    ) -> CovarianceResult:
        month = self._panel_months.get(eom)
        if month is None:
            raise KeyError(f"No panel risk inputs for {eom.date()}")
        source = str(self.settings.get("diagonal_source", "vol_12m"))
        if source not in month:
            raise KeyError(f"Diagonal volatility source is missing: {source}")
        values = pd.to_numeric(month[source], errors="coerce").reindex(permnos)
        floor = float(self.settings.get("variance_floor", 1.0e-6))
        valid = values.notna() & np.isfinite(values) & values.gt(0)
        fallback = float(values[valid].median()) if valid.any() else float(np.sqrt(floor))
        cleaned = values.where(valid, fallback).to_numpy(dtype=float)
        variances = np.maximum(cleaned**2, floor)
        missing = tuple(int(permno) for permno, ok in zip(permnos, valid, strict=True) if not ok)
        return CovarianceResult(
            covariance=np.diag(variances),
            requested_model="diagonal",
            used_model="diagonal",
            min_eigenvalue=float(variances.min()),
            condition_number=float(variances.max() / variances.min()),
            missing_assets=missing,
            diagonal_min=float(variances.min()),
            diagonal_median=float(np.median(variances)),
            diagonal_max=float(variances.max()),
            diagnostic_notes=("vol_12m is a monthly standard deviation; covariance is monthly.",),
        )

    def _factor_covariance(
        self, eom: pd.Timestamp, permnos: Sequence[int]
    ) -> CovarianceResult:
        if self.risk_inputs is None:
            self.risk_inputs = load_risk_inputs(universe=self.universe)
        exposures = self.risk_inputs["factor_exposures"]
        exposure_month = exposures[pd.to_datetime(exposures["eom"]) == eom].set_index("permno")
        missing = tuple(int(value) for value in permnos if value not in exposure_month.index)
        if missing:
            raise ValueError(f"Factor exposures missing {len(missing)} requested assets")
        factor_columns = [
            column for column in exposure_month.columns if column not in {"eom", "universe"}
        ]
        cov_frame = self.risk_inputs["factor_cov_monthly"]
        cov_month = cov_frame[pd.to_datetime(cov_frame["eom"]) == eom]
        if cov_month.empty:
            raise ValueError(f"Factor covariance is missing for {eom.date()}")
        factors = sorted(set(cov_month["factor_1"]) | set(cov_month["factor_2"]))
        factors = [factor for factor in factors if factor in factor_columns]
        if not factors:
            raise ValueError("No common factor exposure/covariance columns")
        factor_index = {factor: index for index, factor in enumerate(factors)}
        factor_cov = np.zeros((len(factors), len(factors)), dtype=float)
        for row in cov_month.itertuples(index=False):
            if row.factor_1 in factor_index and row.factor_2 in factor_index:
                i, j = factor_index[row.factor_1], factor_index[row.factor_2]
                factor_cov[i, j] = float(row.covariance)
                factor_cov[j, i] = float(row.covariance)
        multiplier = float(self.settings.get("factor_daily_to_monthly", 21.0))
        factor_cov *= multiplier
        loadings = exposure_month.reindex(permnos)[factors].fillna(0.0).to_numpy(dtype=float)
        idio = self.risk_inputs["idio_var_monthly"]
        idio_month = idio[pd.to_datetime(idio["eom"]) == eom].set_index("permno")
        idio_values = pd.to_numeric(idio_month["idio_var"], errors="coerce").reindex(permnos)
        valid_idio = idio_values.notna() & np.isfinite(idio_values) & idio_values.gt(0)
        imputed = tuple(
            int(permno)
            for permno, valid in zip(permnos, valid_idio, strict=True)
            if not valid
        )
        missing_fraction = len(imputed) / max(len(permnos), 1)
        max_missing_fraction = float(self.settings.get("factor_max_idio_missing_fraction", 0.10))
        if missing_fraction > max_missing_fraction or not valid_idio.any():
            raise ValueError(
                "Idiosyncratic variance missing for "
                f"{len(imputed)}/{len(permnos)} assets; allowed fraction is "
                f"{max_missing_fraction:.1%}"
            )
        idio_fallback = float(idio_values[valid_idio].median())
        idio_values = idio_values.where(valid_idio, idio_fallback)
        covariance = loadings @ factor_cov @ loadings.T + np.diag(
            idio_values.to_numpy(dtype=float) * multiplier
        )
        covariance = (covariance + covariance.T) / 2.0
        floor = float(self.settings.get("variance_floor", 1.0e-6))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        repair_count = int((eigenvalues < floor).sum())
        covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
        adjusted = np.linalg.eigvalsh(covariance)
        diagonal = np.diag(covariance)
        diagonal_reference = self._diagonal_covariance(eom, permnos)
        variance_ratio = float(
            np.median(diagonal) / max(diagonal_reference.diagonal_median, floor)
        )
        annualized_asset_vol = float(np.sqrt(12.0 * np.median(diagonal)))
        max_ratio = float(self.settings.get("factor_variance_ratio_max", 25.0))
        max_asset_vol = float(
            self.settings.get("factor_max_annualized_asset_volatility", 2.0)
        )
        if variance_ratio > max_ratio or annualized_asset_vol > max_asset_vol:
            raise ValueError(
                "Factor covariance failed scale quality gate: median variance is "
                f"{variance_ratio:.1f}x the vol_12m diagonal reference and implies "
                f"{annualized_asset_vol:.1%} annualized median asset volatility"
            )
        diagonal_flag = bool(
            "diagonal_fallback" in cov_month
            and cov_month["diagonal_fallback"].fillna(False).astype(bool).any()
        )
        notes = [
            "Daily factor and idiosyncratic variances multiplied by the configured daily-to-monthly factor."
        ]
        if diagonal_flag:
            notes.append("Factor covariance input is diagonal, but shared exposures create asset correlations.")
        if imputed:
            notes.append(
                f"Imputed daily idiosyncratic variance for {len(imputed)} assets with the monthly cross-sectional median."
            )
        return CovarianceResult(
            covariance=covariance,
            requested_model="factor",
            used_model="factor",
            min_eigenvalue=float(adjusted.min()),
            condition_number=float(adjusted.max() / adjusted.min()),
            missing_assets=imputed,
            diagonal_min=float(diagonal.min()),
            diagonal_median=float(np.median(diagonal)),
            diagonal_max=float(diagonal.max()),
            factor_covariance_diagonal_fallback=diagonal_flag,
            idio_imputed_assets=imputed,
            diagnostic_notes=tuple(notes),
            covariance_repair_count=repair_count,
        )


def _save_covariance_result(path: Path, result: CovarianceResult) -> None:
    np.savez_compressed(
        path,
        covariance=result.covariance,
        requested_model=np.array(result.requested_model),
        used_model=np.array(result.used_model),
        min_eigenvalue=np.array(result.min_eigenvalue),
        condition_number=np.array(result.condition_number),
        missing_assets=np.asarray(result.missing_assets, dtype=np.int64),
        fallback_reason=np.array(result.fallback_reason or ""),
        diagonal_min=np.array(result.diagonal_min),
        diagonal_median=np.array(result.diagonal_median),
        diagonal_max=np.array(result.diagonal_max),
        factor_covariance_diagonal_fallback=np.array(
            result.factor_covariance_diagonal_fallback
        ),
        idio_imputed_assets=np.asarray(result.idio_imputed_assets, dtype=np.int64),
        diagnostic_notes=np.asarray(result.diagnostic_notes, dtype=str),
        covariance_repair_count=np.array(result.covariance_repair_count),
    )


def _load_covariance_result(path: Path) -> CovarianceResult:
    with np.load(path, allow_pickle=False) as values:
        fallback = str(values["fallback_reason"].item())
        return CovarianceResult(
            covariance=values["covariance"],
            requested_model=str(values["requested_model"].item()),
            used_model=str(values["used_model"].item()),
            min_eigenvalue=float(values["min_eigenvalue"].item()),
            condition_number=float(values["condition_number"].item()),
            missing_assets=tuple(int(value) for value in values["missing_assets"]),
            fallback_reason=fallback or None,
            diagonal_min=float(values["diagonal_min"].item()),
            diagonal_median=float(values["diagonal_median"].item()),
            diagonal_max=float(values["diagonal_max"].item()),
            factor_covariance_diagonal_fallback=bool(
                values["factor_covariance_diagonal_fallback"].item()
            ),
            idio_imputed_assets=tuple(
                int(value) for value in values["idio_imputed_assets"]
            ),
            diagnostic_notes=tuple(str(value) for value in values["diagnostic_notes"]),
            covariance_repair_count=int(values["covariance_repair_count"].item()),
        )
