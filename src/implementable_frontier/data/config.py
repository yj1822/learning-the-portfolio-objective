from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_us_equity_ml.yaml"
VALID_UNIVERSES = {"top100", "top300", "top500"}


class DataConfigError(ValueError):
    """Raised when the thesis data configuration is internally inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    path: Path
    project_root: Path
    values: dict[str, Any]

    @property
    def final15(self) -> tuple[str, ...]:
        return tuple(self.values["feature_sets"]["final15"])

    @property
    def core10(self) -> tuple[str, ...]:
        return tuple(self.values["feature_sets"]["core10"])

    @property
    def feature_sets(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(features) for name, features in self.values["feature_sets"].items()}

    @property
    def feature_aliases(self) -> dict[str, list[str]]:
        return {
            name: list(candidates) for name, candidates in self.values["feature_aliases"].items()
        }

    @property
    def processed_dir(self) -> Path:
        return self._output_path("processed_dir")

    @property
    def samples_dir(self) -> Path:
        return self._output_path("samples_dir")

    @property
    def validation_dir(self) -> Path:
        return self._output_path("validation_dir")

    @property
    def master_panel_path(self) -> Path:
        explicit = self.values["outputs"].get("master_panel_path")
        return self._resolve_project_path(explicit) if explicit else self.processed_dir / "master_panel.parquet"

    @property
    def cost_inputs_path(self) -> Path:
        return self.processed_dir / "cost_inputs.parquet"

    @property
    def risk_dir(self) -> Path:
        return self.processed_dir / "risk"

    def _output_path(self, key: str) -> Path:
        return self._resolve_project_path(self.values["outputs"][key])

    def _resolve_project_path(self, raw_value: object) -> Path:
        value = Path(str(raw_value))
        if value.is_absolute():
            raise DataConfigError(f"Output paths must be project-relative, got: {value}")
        resolved = (self.project_root / value).resolve()
        if self.project_root not in resolved.parents and resolved != self.project_root:
            raise DataConfigError(f"outputs.{key} resolves outside the project: {resolved}")
        return resolved


def load_data_config(path: str | Path | None = None) -> DataConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Thesis data config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise DataConfigError("The thesis data config must be a YAML mapping")
    project_root = _project_root_for(config_path)
    _validate(values)
    return DataConfig(path=config_path, project_root=project_root, values=values)


def _project_root_for(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _validate(values: dict[str, Any]) -> None:
    required = {
        "start_date",
        "end_date",
        "analysis_start",
        "oos_start",
        "max_universe_n",
        "universe_variants",
        "min_price",
        "rank_features",
        "missing_feature_value",
        "baseline_aum",
        "transaction_cost",
        "feature_sets",
        "feature_aliases",
        "data_hub",
        "outputs",
        "validation",
    }
    missing = required - set(values)
    if missing:
        raise DataConfigError(f"Missing config keys: {sorted(missing)}")

    dates = [pd.Timestamp(values[name]) for name in ("start_date", "analysis_start", "oos_start", "end_date")]
    if dates != sorted(dates):
        raise DataConfigError("Expected start_date <= analysis_start <= oos_start <= end_date")
    if int(values["max_universe_n"]) != 500:
        raise DataConfigError("max_universe_n must be 500 so later experiments only filter the fixed panel")
    universes = {str(value).lower() for value in values["universe_variants"]}
    if universes != VALID_UNIVERSES:
        raise DataConfigError(f"universe_variants must be exactly {sorted(VALID_UNIVERSES)}")
    if float(values["min_price"]) < 0 or float(values["baseline_aum"]) <= 0:
        raise DataConfigError("min_price must be non-negative and baseline_aum must be positive")
    missing_value = float(values["missing_feature_value"])
    if not 0.0 <= missing_value <= 1.0:
        raise DataConfigError("missing_feature_value must be in [0, 1]")
    if not bool(values["rank_features"]):
        raise DataConfigError("rank_features must remain true for the research-ready panel")

    feature_sets = values["feature_sets"]
    if not {"core10", "final15"}.issubset(feature_sets):
        raise DataConfigError("feature_sets must define at least core10 and final15")
    core10 = _feature_names(feature_sets["core10"], "core10")
    final15 = _feature_names(feature_sets["final15"], "final15")
    if len(core10) != 10 or len(final15) != 15:
        raise DataConfigError("core10 and final15 must contain 10 and 15 unique features")
    if not set(core10).issubset(final15):
        raise DataConfigError("core10 must be a subset of final15")
    all_features: set[str] = set()
    for name, configured_features in feature_sets.items():
        parsed = _feature_names(configured_features, str(name))
        if name not in {"core10", "final15"} and not set(core10).issubset(parsed):
            raise DataConfigError(f"core10 must be a subset of feature_sets.{name}")
        all_features.update(parsed)
    missing_aliases = all_features - set(values["feature_aliases"])
    if missing_aliases:
        raise DataConfigError(f"Missing feature aliases: {sorted(missing_aliases)}")
    for feature in all_features:
        _feature_names(values["feature_aliases"][feature], f"feature_aliases.{feature}")

    transaction_cost = values["transaction_cost"]
    if transaction_cost.get("model") != "adv_quadratic":
        raise DataConfigError("transaction_cost.model must be adv_quadratic")
    if float(transaction_cost.get("lambda_multiplier", 0)) <= 0:
        raise DataConfigError("transaction_cost.lambda_multiplier must be positive")
    if int(transaction_cost.get("adv_window_months", 0)) <= 0:
        raise DataConfigError("transaction_cost.adv_window_months must be positive")

    for key in ("processed_dir", "samples_dir", "validation_dir"):
        if key not in values["outputs"]:
            raise DataConfigError(f"Missing outputs.{key}")


def _feature_names(values: Any, label: str) -> list[str]:
    if isinstance(values, str) or not isinstance(values, list):
        raise DataConfigError(f"{label} must be a YAML list")
    names = [str(value).strip().lower() for value in values if str(value).strip()]
    if not names or len(names) != len(set(names)):
        raise DataConfigError(f"{label} must contain unique non-empty names")
    return names
