from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs/model_top500_core10_cumulative.yaml"
CONFIG_ROOT = PROJECT_ROOT / "configs/final_analysis"
BASELINE_CHECKPOINT = PROJECT_ROOT / "results/checkpoints/top500_core10_cumulative"
SHARED_ROOT = PROJECT_ROOT / "results/checkpoints/final_analysis/shared"

GAMMAS = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
AUM_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare locked Core10 sensitivity configs")
    parser.add_argument("--stage", choices=["frontier", "aum", "all"], default="all")
    args = parser.parse_args()
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if args.stage in {"frontier", "all"}:
        for gamma in GAMMAS:
            paths.append(_write_config(base, "frontier", gamma=gamma))
    if args.stage in {"aum", "all"}:
        baseline = float(base["baseline_aum"])
        for multiplier in AUM_MULTIPLIERS:
            paths.append(
                _write_config(
                    base,
                    "aum",
                    aum=baseline * multiplier,
                    multiplier=multiplier,
                )
            )
    manifest = {
        "locked_specification": "Top500/Core10",
        "base_config": str(BASE_CONFIG.relative_to(PROJECT_ROOT)),
        "gamma_grid": GAMMAS,
        "aum_multipliers": AUM_MULTIPLIERS,
        "configs": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
        "model_selection_use": False,
        "analysis_labels": ["mechanism analysis", "sensitivity analysis"],
    }
    path = CONFIG_ROOT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)
    return 0


def _write_config(
    base: dict[str, object],
    stage: str,
    *,
    gamma: float | None = None,
    aum: float | None = None,
    multiplier: float | None = None,
) -> Path:
    values = copy.deepcopy(base)
    if stage == "frontier":
        assert gamma is not None
        slug = _slug(gamma)
        values["baseline_gamma"] = float(gamma)
        values["experiment_name"] = f"top500_core10_frontier_gamma_{slug}"
        scenario = f"gamma_{slug}"
    else:
        assert aum is not None and multiplier is not None
        slug = _slug(multiplier)
        values["baseline_aum"] = float(aum)
        values["allow_zero_aum_diagnostic"] = bool(aum == 0)
        values["experiment_name"] = f"top500_core10_aum_multiplier_{slug}"
        scenario = f"multiplier_{slug}"
    checkpoint_root = f"results/checkpoints/final_analysis/{stage}/{scenario}"
    result_root = f"results/final_analysis/{stage}/{scenario}"
    values["checkpoint"] = {
        **values["checkpoint"],
        "root": checkpoint_root,
        "resume": True,
    }
    values["cache"] = {
        **values["cache"],
        "root": str((SHARED_ROOT / stage).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "covariance_root": str(
            (BASELINE_CHECKPOINT / "cache/covariance").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "rf_design_root": str(
            (BASELINE_CHECKPOINT / "cache/rf_design").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "return_prediction_root": str(
            (BASELINE_CHECKPOINT / "return_predictions").relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "transaction_cost_root": f"{checkpoint_root}/cache/transaction_cost",
        "matrix_m_root": f"{checkpoint_root}/cache/matrix_m",
    }
    values["outputs"] = {
        **values["outputs"],
        "formal_dir": result_root,
        "pilot_dir": f"{result_root}_pilot",
        "formal_reports_dir": f"reports/final_analysis/{stage}/scenario_reports/{scenario}",
    }
    values["analysis_label"] = (
        "controlled gamma-grid mechanism analysis"
        if stage == "frontier"
        else "endogenous AUM sensitivity analysis"
    )
    values["feature_set_search"] = False
    path = CONFIG_ROOT / f"{stage}_{scenario}.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
