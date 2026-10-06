from __future__ import annotations

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from implementable_frontier.data.config import DataConfig


class TradingDataHubUnavailableError(RuntimeError):
    """Raised when the configured trading-data-hub checkout cannot be imported."""


@dataclass(frozen=True)
class HubIdentity:
    version: str
    module_path: str
    config_path: str | None


class TradingDataHubAdapter:
    """Thin orchestration layer around trading-data-hub's existing interfaces."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.project_root = self._resolve_project_root(project_root)
        self.config_path = self._resolve_config_path(config_path)
        if self.config_path is not None:
            os.environ["DATAHUB_CONFIG"] = str(self.config_path)
        self.module = self._import_public_module()

    @property
    def identity(self) -> HubIdentity:
        return HubIdentity(
            version=str(getattr(self.module, "__version__", "unknown")),
            module_path=str(Path(self.module.__file__).resolve()),
            config_path=str(self.config_path) if self.config_path is not None else None,
        )

    def load_master(self, config: DataConfig) -> pd.DataFrame:
        return self.module.load_us_equity_ml_panel(
            start=str(config.values["start_date"]),
            end=str(config.values["end_date"]),
            universe="top500",
            feature_set="final15",
        )

    def load_risk(self, config: DataConfig) -> dict[str, pd.DataFrame]:
        combined: dict[str, list[pd.DataFrame]] = {}
        for universe in config.values["universe_variants"]:
            outputs = self.module.load_us_equity_risk_inputs(
                start=str(config.values["start_date"]),
                end=str(config.values["end_date"]),
                universe=str(universe),
            )
            for name, frame in outputs.items():
                combined.setdefault(name, []).append(frame)
        return {
            name: pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
            for name, frames in combined.items()
        }

    def build_master(self, config: DataConfig) -> pd.DataFrame:
        builder_module = self._import_internal("datahub.us_equity.master_panel")
        builder = builder_module.build_us_equity_ml_master_panel
        with tempfile.TemporaryDirectory(prefix="ief-feature-aliases-") as directory:
            alias_path = Path(directory) / "feature_aliases.yaml"
            alias_path.write_text(
                yaml.safe_dump(config.feature_aliases, sort_keys=False), encoding="utf-8"
            )
            return builder(
                start_date=str(config.values["start_date"]),
                end_date=str(config.values["end_date"]),
                max_universe_n=int(config.values["max_universe_n"]),
                features=list(config.final15),
                min_price=float(config.values["min_price"]),
                rank_features=bool(config.values["rank_features"]),
                feature_aliases_path=directory,
                country=str(config.values["data_hub"].get("country", "USA")),
            )

    def build_risk(self, config: DataConfig, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
        builder_module = self._import_internal("datahub.us_equity.risk")
        return builder_module.build_us_equity_risk_inputs(
            str(config.values["start_date"]),
            str(config.values["end_date"]),
            panel=panel,
            ewma_span=int(config.values["data_hub"].get("risk_ewma_span", 60)),
        )

    def download_wrds(self, config: DataConfig) -> dict[str, Any]:
        connector_module = self._import_internal("datahub.sources.wrds")
        connector = connector_module.WRDSConnector()
        start = pd.Timestamp(config.values["start_date"])
        end = pd.Timestamp(config.values["end_date"])
        country = str(config.values["data_hub"].get("country", "USA"))
        totals = {"jkp_rows": 0, "crsp_monthly_rows": 0, "crsp_daily_rows": 0}
        partition_status: dict[str, dict[str, str]] = {}
        download_features: list[str] | None = None
        unresolved_features: list[str] = []
        try:
            for year, window_start, window_end in _year_windows(start, end):
                year_status: dict[str, str] = {}
                jkp_path = _partition_path(
                    connector.raw_root, "jkp_global_factor", year, country=country
                )
                if jkp_path.exists():
                    jkp = pd.read_parquet(jkp_path, columns=["permno"])
                    totals["jkp_rows"] += _parquet_row_count(jkp_path)
                    year_status["jkp"] = "existing"
                    if download_features is None:
                        download_features, unresolved_features = _resolve_download_features(
                            pq.read_schema(jkp_path).names, config
                        )
                        _feature_progress(download_features, unresolved_features)
                else:
                    _progress(year, "jkp", "downloading")
                    jkp = connector.download_jkp_global_factors(
                        window_start,
                        window_end,
                        country=country,
                        features=download_features,
                    )
                    totals["jkp_rows"] += int(len(jkp))
                    year_status["jkp"] = "downloaded"
                    if download_features is None:
                        download_features, unresolved_features = _resolve_download_features(
                            jkp.columns, config
                        )
                        _feature_progress(download_features, unresolved_features)
                permnos = _unique_permnos(jkp)

                monthly_path = _partition_path(connector.raw_root, "crsp_monthly", year)
                if monthly_path.exists():
                    totals["crsp_monthly_rows"] += _parquet_row_count(monthly_path)
                    year_status["crsp_monthly"] = "existing"
                else:
                    _progress(year, "crsp_monthly", "downloading")
                    monthly = connector.download_crsp_monthly(
                        window_start, window_end, permnos=permnos
                    )
                    totals["crsp_monthly_rows"] += int(len(monthly))
                    year_status["crsp_monthly"] = "downloaded"

                daily_path = _partition_path(connector.raw_root, "crsp_daily", year)
                if daily_path.exists():
                    totals["crsp_daily_rows"] += _parquet_row_count(daily_path)
                    year_status["crsp_daily"] = "existing"
                else:
                    _progress(year, "crsp_daily", "downloading")
                    daily = connector.download_crsp_daily(
                        window_start, window_end, permnos=permnos
                    )
                    totals["crsp_daily_rows"] += int(len(daily))
                    year_status["crsp_daily"] = "downloaded"
                partition_status[str(year)] = year_status
                _progress(year, "year", "complete")
            return {
                **totals,
                "years": partition_status,
                "storage_root": str(connector.raw_root),
                "jkp_download_features": download_features or [],
                "unresolved_features": unresolved_features,
                "fama_french": "not_exposed_by_current_trading_data_hub",
            }
        except Exception as exc:  # noqa: BLE001 - convert upstream errors into one actionable boundary
            raise RuntimeError(f"WRDS download through trading-data-hub failed: {exc}") from exc
        finally:
            connector.close()

    def _resolve_project_root(self, value: str | Path | None) -> Path | None:
        configured = value or os.getenv("TRADING_DATA_HUB_ROOT")
        if not configured:
            return None
        root = Path(configured).expanduser().resolve()
        source_root = root / "src"
        if not source_root.exists():
            raise TradingDataHubUnavailableError(
                f"TRADING_DATA_HUB_ROOT does not contain src/: {root}"
            )
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        return root

    def _resolve_config_path(self, value: str | Path | None) -> Path | None:
        configured = value or os.getenv("DATAHUB_CONFIG")
        if configured:
            path = Path(configured).expanduser().resolve()
        elif self.project_root is not None:
            path = self.project_root / "config" / "sources.yaml"
        else:
            return None
        if not path.exists():
            raise TradingDataHubUnavailableError(f"trading-data-hub config not found: {path}")
        return path

    def _import_public_module(self) -> ModuleType:
        try:
            return importlib.import_module("trading_data_hub")
        except ImportError as exc:
            raise TradingDataHubUnavailableError(
                "trading-data-hub is not importable. Install its checkout with `pip install -e <path>` "
                "or set TRADING_DATA_HUB_ROOT to the repository root."
            ) from exc

    @staticmethod
    def _import_internal(name: str) -> ModuleType:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise TradingDataHubUnavailableError(
                f"The installed trading-data-hub does not expose required module {name!r}."
            ) from exc


def _unique_permnos(frame: pd.DataFrame) -> list[int]:
    if "permno" not in frame:
        return []
    values = pd.to_numeric(frame["permno"], errors="coerce").dropna().astype(int).unique()
    return sorted(int(value) for value in values)


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, str, str]]:
    windows: list[tuple[int, str, str]] = []
    for year in range(start.year, end.year + 1):
        window_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        window_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        windows.append((year, str(window_start.date()), str(window_end.date())))
    return windows


def _partition_path(
    raw_root: Path, dataset: str, year: int, *, country: str | None = None
) -> Path:
    root = raw_root / "wrds" / dataset
    if country is not None:
        root = root / f"excntry={country}"
    return root / f"year={year}" / "part.parquet"


def _parquet_row_count(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def _progress(year: int, dataset: str, status: str) -> None:
    print(f"[WRDS] year={year} dataset={dataset} status={status}", flush=True)


def _resolve_download_features(
    available_columns: Any, config: DataConfig
) -> tuple[list[str], list[str]]:
    available = {str(column).lower() for column in available_columns}
    selected: list[str] = []
    unresolved: list[str] = []
    for feature in config.final15:
        source = next(
            (
                candidate.lower()
                for candidate in config.feature_aliases.get(feature, [])
                if candidate.lower() in available
            ),
            None,
        )
        if source is None:
            if feature not in {"log_mcap", "adv_6m"}:
                unresolved.append(feature)
        elif source not in selected:
            selected.append(source)
    return selected, unresolved


def _feature_progress(selected: list[str], unresolved: list[str]) -> None:
    print(f"[WRDS] JKP selected feature columns: {','.join(selected)}", flush=True)
    if unresolved:
        print(f"[WRDS] JKP unresolved feature themes: {','.join(unresolved)}", flush=True)
