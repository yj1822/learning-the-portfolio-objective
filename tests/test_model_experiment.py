from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from implementable_frontier.models.experiment import ExperimentResult, run_model_experiment
from implementable_frontier.models.experiment import _concat_frames


def _synthetic_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    for month_index, eom in enumerate(dates):
        for permno in range(1, 5):
            signal = (permno + month_index % 3) / 7.0
            rows.append(
                {
                    "eom": eom,
                    "permno": permno,
                    "rank_x": signal,
                    "rank_y": 1.0 - signal,
                    "ret_1m": 0.01 * (signal - 0.4),
                    "ret_exc_lead1m": 0.01 * (signal - 0.4),
                    "lambda_adv": 1.0e-9 * (1.0 + permno / 10.0),
                    "adv_6m": 0.2 / (1.0e-9 * (1.0 + permno / 10.0)),
                    "vol_12m": 0.15 + permno / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_concat_frames_preserves_all_null_diagnostic_columns() -> None:
    frame = pd.DataFrame({"method": ["static_ml_star"], "aim": [np.nan]})
    result = _concat_frames([frame])
    assert list(result.columns) == ["method", "aim"]


def test_small_experiment_writes_complete_outputs(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    data_config = {
        "feature_sets": {"core10": ["x", "y"], "final15": ["x", "y"]}
    }
    (configs / "data.yaml").write_text(yaml.safe_dump(data_config), encoding="utf-8")
    model_config = {
        "version": 1,
        "experiment_name": "integration",
        "data_config": "configs/data.yaml",
        "universe": "top100",
        "feature_set": "core10",
        "risk_model": "diagonal",
        "methods": [
            "equal_weight",
            "return_ml_rank",
            "static_ml",
            "static_ml_star",
            "portfolio_ml_linear",
        ],
        "protocol": "fast_split",
        "fast_split": {
            "train_start": "2000-01-31",
            "train_end": "2000-06-30",
            "validation_start": "2000-07-31",
            "validation_end": "2000-09-30",
            "test_start": "2000-10-31",
            "test_end": "2000-12-31",
        },
        "rolling_yearly": {
            "first_test_year": 2005,
            "last_test_year": 2005,
            "analysis_start": "1995-01-31",
            "validation_years": 3,
        },
        "gamma_grid": [10],
        "baseline_gamma": 10,
        "aum_grid": [100000000],
        "baseline_aum": 100000000,
        "random_seed": 7,
        "return_ml": {
            "models": ["ridge"],
            "lambda_grid": [0.1],
            "rf_p_grid": [4],
            "rf_eta_grid": [0.5],
            "rank_gross_leverage": 1.0,
        },
        "markowitz_ml": {"covariance_ridge": 1.0e-8, "max_leverage": None},
        "static_ml": {
            "phi_grid": [3],
            "mu_shrinkage_grid": [0.25],
            "covariance_ridge": 1.0e-8,
        },
        "static_ml_star": {
            "u_grid": [0.25],
            "v_grid": [0.5],
            "k_grid": [1.0],
            "covariance_ridge": 1.0e-8,
        },
        "portfolio_ml": {
            "lambda_grid": [0.1],
            "rf_p_grid": [4],
            "rf_eta_grid": [0.5],
            "use_matrix_m": True,
            "matrix_m_iterations": 3,
            "expected_market_return": 0.007,
            "risk_free_rate": 0.0,
            "allow_scalar_rho_fallback": True,
            "rho_grid_if_fallback": [0.5],
            "volatility_floor": 0.01,
            "coefficient_ridge_floor": 1.0e-10,
            "add_constant": True,
        },
        "risk": {
            "diagonal_source": "vol_12m",
            "variance_floor": 1.0e-6,
            "allow_factor_fallback": True,
        },
        "accounting": {
            "charge_exit_costs": True,
            "missing_exit_return": 0.0,
            "wealth_growth_floor": 0.1,
        },
        "diagnostics": {"max_abs_weight_warning": 5.0, "max_leverage_warning": 20.0},
        "diagnostic_risk_scaling": {
            "enabled": True,
            "target_annual_volatility": 0.10,
            "methods": ["static_ml", "portfolio_ml_linear"],
        },
        "outputs": {
            "experiments_dir": str(tmp_path / "experiments"),
            "reports_dir": str(tmp_path / "reports"),
            "save_plots": False,
        },
    }
    config_path = configs / "model.yaml"
    config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    monkeypatch.setattr(
        "implementable_frontier.models.experiment.load_master_panel",
        lambda **_: _synthetic_panel(),
    )
    result = run_model_experiment(config_path, experiment_id="integration")
    assert isinstance(result, ExperimentResult)
    for filename in (
        "config_used.yaml",
        "metadata.json",
        "predictions.parquet",
        "weights.parquet",
        "returns.parquet",
        "metrics.csv",
        "validation_selection.parquet",
        "diagnostics.md",
        "diagnostics_summary.csv",
        "metrics_raw.csv",
        "metrics_risk_scaled.csv",
        "unit_audit.json",
        "unit_audit.md",
    ):
        assert (result.output_dir / filename).exists()
    metrics = pd.read_csv(result.output_dir / "metrics.csv")
    assert set(model_config["methods"]).issubset(set(metrics["method"]))
    assert any(metrics["method"].str.contains("scaled_10pct_vol"))
    numeric = metrics.select_dtypes(include=[np.number])
    assert numeric.notna().any(axis=1).all()
