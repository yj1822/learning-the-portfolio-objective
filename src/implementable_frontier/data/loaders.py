from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from implementable_frontier.data.config import VALID_UNIVERSES, DataConfig, load_data_config


UNIVERSE_FLAGS = {"top100": "in_top100", "top300": "in_top300", "top500": "in_top500"}
RISK_FILES = {
    "factor_exposures": "factor_exposures.parquet",
    "factor_returns_daily": "factor_returns_daily.parquet",
    "factor_cov_monthly": "factor_cov_monthly.parquet",
    "idio_var_monthly": "idio_var_monthly.parquet",
}
BASE_PANEL_COLUMNS = [
    "permno",
    "gvkey",
    "eom",
    "ret_1m",
    "ret_exc_lead1m",
    "price",
    "mktcap",
    "lag_mktcap",
    "adv_6m",
    "lambda_adv",
    "vol_12m",
    "beta_mkt",
    "idio_vol",
    "sic",
    "siccd",
    "universe_rank_mcap",
    "in_top100",
    "in_top300",
    "in_top500",
]


def load_master_panel(
    start: str | None = None,
    end: str | None = None,
    universe: str = "top500",
    feature_set: str = "final15",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read and filter the thesis-local processed panel without rebuilding it."""
    config = _runtime_config()
    explicit = os.getenv("IEF_MASTER_PANEL_PATH")
    path = Path(explicit).expanduser().resolve() if explicit else config.master_panel_path
    available = _parquet_columns(path)
    universe_name, universe_flag = _universe(universe)
    feature_name = feature_set.strip().lower()
    if feature_name not in config.feature_sets:
        raise ValueError(f"feature_set must be one of {sorted(config.feature_sets)}")

    if columns is None:
        feature_columns = [f"rank_{feature}" for feature in config.feature_sets[feature_name]]
        missing_features = set(feature_columns) - set(available)
        if missing_features:
            raise ValueError(f"Processed panel is missing configured rank features: {sorted(missing_features)}")
        selected = [column for column in BASE_PANEL_COLUMNS if column in available]
        selected.extend(column for column in feature_columns if column not in selected)
    else:
        selected = _validated_columns(columns, available)

    internal = [column for column in ("eom", universe_flag) if column not in selected]
    frame = pd.read_parquet(path, columns=[*selected, *internal])
    frame["eom"] = pd.to_datetime(frame["eom"], errors="coerce")
    frame = frame[frame[universe_flag].fillna(False)]
    frame = _filter_dates(frame, "eom", start, end)
    ordering = [column for column in ("eom", "universe_rank_mcap", "permno") if column in frame]
    frame = frame.sort_values(ordering).reset_index(drop=True) if ordering else frame.reset_index(drop=True)
    return frame[selected]


def load_cost_inputs(
    start: str | None = None,
    end: str | None = None,
    universe: str = "top500",
) -> pd.DataFrame:
    """Read the precomputed transaction-cost inputs only."""
    config = _runtime_config()
    path = _processed_root(config) / "cost_inputs.parquet"
    universe_name, universe_flag = _universe(universe)
    available = _parquet_columns(path)
    if universe_flag not in available:
        raise ValueError(f"Cost inputs do not contain {universe_flag}")
    frame = pd.read_parquet(path)
    frame["eom"] = pd.to_datetime(frame["eom"], errors="coerce")
    frame = frame[frame[universe_flag].fillna(False)]
    frame = _filter_dates(frame, "eom", start, end)
    return frame.sort_values(["eom", "permno"]).reset_index(drop=True)


def load_risk_inputs(
    start: str | None = None,
    end: str | None = None,
    universe: str = "top500",
    root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Read the four copied risk inputs; this function never estimates risk."""
    config = _runtime_config()
    universe_name, _ = _universe(universe)
    risk_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else _processed_root(config) / "risk"
    )
    missing = [filename for filename in RISK_FILES.values() if not (risk_root / filename).exists()]
    if missing:
        detail = _risk_manifest_detail(_processed_root(config) / "risk_manifest.json")
        raise FileNotFoundError(
            f"Risk inputs are unavailable ({', '.join(missing)}). {detail}"
        )
    result: dict[str, pd.DataFrame] = {}
    for name, filename in RISK_FILES.items():
        frame = pd.read_parquet(risk_root / filename)
        if "universe" in frame:
            frame = frame[frame["universe"].astype(str).str.lower() == universe_name]
        date_column = "date" if "date" in frame else "eom" if "eom" in frame else None
        if date_column is not None:
            frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
            frame = _filter_dates(frame, date_column, start, end)
        result[name] = frame.reset_index(drop=True)
    return result


def _runtime_config() -> DataConfig:
    return load_data_config(os.getenv("IEF_DATA_CONFIG"))


def _processed_root(config: DataConfig) -> Path:
    override = os.getenv("IEF_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else config.processed_dir


def _parquet_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pq.read_schema(path).names


def _validated_columns(columns: Sequence[str], available: Sequence[str]) -> list[str]:
    selected = [str(column) for column in columns]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("columns must be non-empty and unique")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown processed columns: {sorted(unknown)}")
    return selected


def _universe(value: str) -> tuple[str, str]:
    name = value.strip().lower()
    if name not in VALID_UNIVERSES:
        raise ValueError(f"universe must be one of {sorted(VALID_UNIVERSES)}")
    return name, UNIVERSE_FLAGS[name]


def _filter_dates(
    frame: pd.DataFrame, column: str, start: str | None, end: str | None
) -> pd.DataFrame:
    result = frame
    if start is not None:
        result = result[result[column] >= pd.Timestamp(start)]
    if end is not None:
        result = result[result[column] <= pd.Timestamp(end)]
    return result


def _risk_manifest_detail(path: Path) -> str:
    if not path.exists():
        return "No risk_manifest.json was generated; run the build command first."
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Inspect the unreadable diagnostic manifest at {path}."
    return f"Risk status: {manifest.get('status', 'unknown')}; reason: {manifest.get('reason', 'not recorded')}."
