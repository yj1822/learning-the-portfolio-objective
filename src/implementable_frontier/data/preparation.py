from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from implementable_frontier import __version__
from implementable_frontier.data.config import DataConfig
from implementable_frontier.data.hub import TradingDataHubAdapter
from implementable_frontier.data.io import atomic_write_json, atomic_write_parquet, atomic_write_text
from implementable_frontier.data.validation import validate_us_equity_ml_dataset


RISK_FILES = {
    "factor_exposures": "factor_exposures.parquet",
    "factor_returns_daily": "factor_returns_daily.parquet",
    "factor_cov_monthly": "factor_cov_monthly.parquet",
    "idio_var_monthly": "idio_var_monthly.parquet",
}
COST_COLUMNS = [
    "permno",
    "eom",
    "adv_6m",
    "lambda_adv",
    "in_top100",
    "in_top300",
    "in_top500",
]


class DataPreparationError(RuntimeError):
    """Raised when the upstream data contract cannot produce the thesis dataset."""


@dataclass(frozen=True)
class BuildResult:
    master_panel_path: str
    cost_inputs_path: str
    row_count: int
    month_count: int
    risk_status: str
    warnings: tuple[str, ...]
    validation_passed: bool


def build_us_equity_ml_dataset(
    config: DataConfig,
    hub: TradingDataHubAdapter,
    *,
    download: bool = False,
) -> BuildResult:
    """Orchestrate upstream hub interfaces and persist the thesis-local copy."""
    build_warnings: list[str] = []
    download_summary: dict[str, Any] | None = None
    if download:
        download_summary = hub.download_wrds(config)
        build_warnings.append(
            "trading-data-hub currently has no Fama-French download interface; beta/idio availability "
            "depends on JKP aliases and is diagnosed below."
        )

    try:
        upstream = hub.load_master(config)
    except (FileNotFoundError, ValueError) as load_error:
        if not bool(config.values["data_hub"].get("build_master_if_missing", True)):
            raise DataPreparationError(f"Upstream master panel is unavailable: {load_error}") from load_error
        try:
            hub.build_master(config)
            upstream = hub.load_master(config)
        except Exception as build_error:  # noqa: BLE001 - preserve one actionable project boundary
            raise DataPreparationError(
                "trading-data-hub could not load or build its processed master panel. "
                "Run this command with --download and valid WRDS credentials, or build the upstream "
                f"hub datasets first. Upstream error: {build_error}"
            ) from build_error

    panel = _normalize_upstream_schema(upstream)
    _assert_master_contract(panel, config)
    panel = panel.sort_values(["eom", "universe_rank_mcap", "permno"]).reset_index(drop=True)
    atomic_write_parquet(panel, config.master_panel_path)
    atomic_write_parquet(panel[COST_COLUMNS].copy(), config.cost_inputs_path)

    risk_manifest = _copy_or_diagnose_risk(config, hub, panel, build_warnings)
    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = _metadata(panel, config, generated_at)
    identity = hub.identity
    manifest = {
        "dataset": config.values.get("dataset_name", "us_equity_ml"),
        "generated_at_utc": generated_at,
        "thesis_package_version": __version__,
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "config_path": str(config.path),
        "trading_data_hub": asdict(identity),
        "download": download_summary,
        "outputs": {
            "master_panel": str(config.master_panel_path),
            "cost_inputs": str(config.cost_inputs_path),
            "risk_manifest": str(config.processed_dir / "risk_manifest.json"),
        },
        "source_contract": {
            "master": "trading_data_hub.load_us_equity_ml_panel",
            "risk": "trading_data_hub.load_us_equity_risk_inputs",
            "label": "JKP ret_exc_lead1m; not recomputed in the thesis project",
            "fama_french": "No dedicated upstream connector in trading-data-hub 0.1.0",
        },
        "warnings": build_warnings,
    }
    atomic_write_json(metadata, config.processed_dir / "metadata.json")
    atomic_write_json(manifest, config.processed_dir / "manifest.json")
    atomic_write_text(
        yaml.safe_dump(config.values, sort_keys=False, allow_unicode=True),
        config.processed_dir / "effective_config.yaml",
    )
    report = validate_us_equity_ml_dataset(config=config, panel=panel)
    return BuildResult(
        master_panel_path=str(config.master_panel_path),
        cost_inputs_path=str(config.cost_inputs_path),
        row_count=int(len(panel)),
        month_count=int(panel["eom"].nunique()),
        risk_status=str(risk_manifest["status"]),
        warnings=tuple(build_warnings),
        validation_passed=bool(report["passed"]),
    )


def export_sample(
    config: DataConfig,
    *,
    universe: str,
    feature_set: str,
    start: str,
    end: str,
) -> Path:
    """Export a filtered view of the existing master panel without reconstruction."""
    import os

    from implementable_frontier.data.loaders import load_master_panel

    os.environ["IEF_DATA_CONFIG"] = str(config.path)
    os.environ["IEF_DATA_ROOT"] = str(config.processed_dir)
    sample = load_master_panel(
        start=start, end=end, universe=universe, feature_set=feature_set
    )
    start_year, end_year = pd.Timestamp(start).year, pd.Timestamp(end).year
    output = config.samples_dir / (
        f"us_equity_ml_{universe.lower()}_{feature_set.lower()}_{start_year}_{end_year}.parquet"
    )
    return atomic_write_parquet(sample, output)


def _normalize_upstream_schema(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data.columns = [str(column).lower() for column in data.columns]
    aliases = {
        "ret_1m": ["delisting_adjusted_return", "return", "ret"],
        "mktcap": ["me", "lag_mktcap"],
        "sic": ["siccd", "sic"],
        "beta_mkt": ["raw_beta_mkt"],
        "idio_vol": ["raw_idio_vol"],
    }
    for target, candidates in aliases.items():
        if target in data:
            continue
        source = next((candidate for candidate in candidates if candidate in data), None)
        data[target] = data[source] if source is not None else np.nan
    data["eom"] = pd.to_datetime(data["eom"], errors="coerce")
    return data


def _assert_master_contract(panel: pd.DataFrame, config: DataConfig) -> None:
    required = {
        "permno",
        "eom",
        "ret_1m",
        "ret_exc_lead1m",
        "price",
        "mktcap",
        "lag_mktcap",
        "adv_6m",
        "lambda_adv",
        "vol_12m",
        "universe_rank_mcap",
        "in_top100",
        "in_top300",
        "in_top500",
        *[f"raw_{feature}" for feature in config.final15],
        *[f"rank_{feature}" for feature in config.final15],
    }
    missing = required - set(panel)
    if missing:
        raise DataPreparationError(
            "trading-data-hub master panel does not satisfy the thesis schema; missing columns: "
            f"{sorted(missing)}. Check feature_aliases and rebuild the upstream master panel."
        )
    multiplier = float(config.values["transaction_cost"]["lambda_multiplier"])
    adv = pd.to_numeric(panel["adv_6m"], errors="coerce")
    observed = pd.to_numeric(panel["lambda_adv"], errors="coerce")
    expected = multiplier / adv
    valid = adv.gt(0) & observed.notna()
    if valid.any() and not np.allclose(observed[valid], expected[valid], rtol=1e-10, atol=0.0):
        raise DataPreparationError(
            "Upstream lambda_adv does not match transaction_cost.lambda_multiplier / adv_6m"
        )


def _copy_or_diagnose_risk(
    config: DataConfig,
    hub: TradingDataHubAdapter,
    panel: pd.DataFrame,
    build_warnings: list[str],
) -> dict[str, Any]:
    reason: str | None = None
    outputs: dict[str, pd.DataFrame] | None = None
    try:
        outputs = hub.load_risk(config)
    except (FileNotFoundError, ValueError) as load_error:
        if bool(config.values["data_hub"].get("build_risk_if_missing", True)):
            try:
                hub.build_risk(config, panel)
                outputs = hub.load_risk(config)
            except Exception as build_error:  # noqa: BLE001 - fallback metadata is the required behavior
                reason = str(build_error)
        else:
            reason = str(load_error)

    if outputs is not None:
        for name, filename in RISK_FILES.items():
            if name not in outputs:
                reason = f"trading-data-hub risk output omitted {name}"
                outputs = None
                break
            atomic_write_parquet(outputs[name], config.risk_dir / filename)

    if outputs is None:
        message = (
            "Risk inputs are unavailable from trading-data-hub; no thesis-local risk model was "
            f"substituted. Diagnostic: {reason or 'unknown upstream error'}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        build_warnings.append(message)
        status = "unavailable"
        rows: dict[str, int] = {}
    else:
        rows = {name: int(len(frame)) for name, frame in outputs.items()}
        status = "available" if any(rows.values()) else "empty_fallback"
        if status == "empty_fallback":
            build_warnings.append(
                "trading-data-hub wrote schema-stable but empty risk inputs; inspect CRSP daily coverage."
            )
    manifest = {
        "status": status,
        "reason": reason,
        "rows": rows,
        "expected_files": list(RISK_FILES.values()),
        "implementation": "trading-data-hub lightweight factor risk; diagonal fallback may be flagged",
        "thesis_side_estimation": False,
    }
    atomic_write_json(manifest, config.processed_dir / "risk_manifest.json")
    return manifest


def _metadata(panel: pd.DataFrame, config: DataConfig, generated_at: str) -> dict[str, Any]:
    raw_missing = {
        feature: float(panel[f"raw_{feature}"].isna().mean()) for feature in config.final15
    }
    return {
        "generated_at_utc": generated_at,
        "row_unit": "stock-month",
        "primary_key": ["eom", "permno"],
        "row_count": int(len(panel)),
        "month_count": int(panel["eom"].nunique()),
        "date_min": panel["eom"].min().isoformat() if not panel.empty else None,
        "date_max": panel["eom"].max().isoformat() if not panel.empty else None,
        "columns": {column: str(dtype) for column, dtype in panel.dtypes.items()},
        "feature_sets": {name: list(values) for name, values in config.feature_sets.items()},
        "raw_feature_missing_rate": raw_missing,
        "universe_definition": "monthly descending rank of lagged market cap after min-price eligibility",
        "transaction_cost": config.values["transaction_cost"],
        "baseline_aum": config.values["baseline_aum"],
        "sample_periods": config.values.get("sample_periods", {}),
    }
