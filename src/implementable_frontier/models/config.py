from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from implementable_frontier.data.config import PROJECT_ROOT


class ModelConfigError(ValueError):
    """Raised when a model experiment configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    project_root: Path
    values: dict[str, Any]

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.values["methods"])

    @property
    def feature_columns(self) -> list[str]:
        data_path = self.resolve_path(self.values["data_config"])
        with data_path.open("r", encoding="utf-8") as handle:
            data_values = yaml.safe_load(handle) or {}
        feature_set = str(self.values["feature_set"])
        return [f"rank_{name}" for name in data_values["feature_sets"][feature_set]]

    @property
    def experiments_dir(self) -> Path:
        return self.resolve_path(self.values["outputs"]["experiments_dir"])

    @property
    def reports_dir(self) -> Path:
        return self.resolve_path(self.values["outputs"]["reports_dir"])

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ModelConfigError("Model config must be a YAML mapping")
    _validate(values)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT
    return ModelConfig(path=config_path, project_root=project_root, values=values)


def _validate(values: dict[str, Any]) -> None:
    required = {
        "data_config",
        "universe",
        "feature_set",
        "risk_model",
        "methods",
        "protocol",
        "baseline_gamma",
        "baseline_aum",
        "random_seed",
        "return_ml",
        "static_ml",
        "portfolio_ml",
        "risk",
        "accounting",
        "outputs",
    }
    missing = required - set(values)
    if missing:
        raise ModelConfigError(f"Missing model config keys: {sorted(missing)}")
    if values["universe"] not in {"top100", "top300", "top500"}:
        raise ModelConfigError("universe must be top100, top300, or top500")
    if values["feature_set"] not in {"core10", "final15", "final20_conservative"}:
        raise ModelConfigError(
            "feature_set must be core10, final15, or final20_conservative"
        )
    if values["risk_model"] not in {"diagonal", "factor"}:
        raise ModelConfigError("risk_model must be diagonal or factor")
    if values["protocol"] not in {
        "fast_split",
        "rolling_yearly",
        "cumulative_oos_validation",
    }:
        raise ModelConfigError(
            "protocol must be fast_split, rolling_yearly, or cumulative_oos_validation"
        )
    gamma = float(values["baseline_gamma"])
    aum = float(values["baseline_aum"])
    allow_zero_aum = bool(values.get("allow_zero_aum_diagnostic", False))
    if gamma <= 0 or aum < 0 or (aum == 0 and not allow_zero_aum):
        raise ModelConfigError(
            "baseline_gamma must be positive and baseline_aum must be positive "
            "unless allow_zero_aum_diagnostic is explicitly enabled"
        )
    if "static_ml_star" in values["methods"]:
        settings = values.get("static_ml_star")
        if not isinstance(settings, dict):
            raise ModelConfigError("static_ml_star settings are required when the method is enabled")
        required_grids = {"u_grid", "v_grid", "k_grid"}
        missing_grids = required_grids - set(settings)
        if missing_grids:
            raise ModelConfigError(
                f"Missing Static-ML* grids: {sorted(missing_grids)}"
            )
        if any(float(value) <= 0 for value in settings["u_grid"]):
            raise ModelConfigError("Static-ML* u_grid values must be positive")
        if any(float(value) < 0 for value in settings["v_grid"]):
            raise ModelConfigError("Static-ML* v_grid values must be non-negative")
        if any(float(value) <= 0 for value in settings["k_grid"]):
            raise ModelConfigError("Static-ML* k_grid values must be positive")
    p_values = [*values["return_ml"]["rf_p_grid"], *values["portfolio_ml"]["rf_p_grid"]]
    if any(int(value) <= 0 or int(value) % 2 for value in p_values):
        raise ModelConfigError("Random-feature dimensions must be positive even integers")
    if values["protocol"] == "fast_split":
        split = values["fast_split"]
        dates = [
            pd.Timestamp(split[name])
            for name in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        ]
        if dates != sorted(dates) or dates[1] >= dates[2] or dates[3] >= dates[4]:
            raise ModelConfigError("fast_split train/validation/test periods must be ordered and disjoint")
    if values["protocol"] == "cumulative_oos_validation":
        cumulative = values.get("cumulative_oos_validation")
        if not isinstance(cumulative, dict):
            raise ModelConfigError("cumulative_oos_validation settings are required")
        years = [
            int(cumulative[name])
            for name in (
                "validation_oos_start_year",
                "first_test_year",
                "last_test_year",
            )
        ]
        if years != sorted(years) or years[0] >= years[1]:
            raise ModelConfigError(
                "cumulative validation years must be ordered with validation before testing"
            )
