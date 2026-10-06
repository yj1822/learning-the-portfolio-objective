from __future__ import annotations

from pathlib import Path

import pandas as pd

from implementable_frontier.data.io import atomic_write_parquet
from implementable_frontier.data.loaders import (
    load_cost_inputs,
    load_master_panel,
    load_risk_inputs,
)


def test_master_loader_filters_universe_and_feature_set(data_config, synthetic_panel) -> None:
    atomic_write_parquet(synthetic_panel, data_config.master_panel_path)
    top100 = load_master_panel(universe="top100", feature_set="core10")
    top300 = load_master_panel(universe="top300", feature_set="core10")
    top500 = load_master_panel(universe="top500", feature_set="final15")
    assert [len(top100), len(top300), len(top500)] == [300, 900, 1500]
    assert "rank_momentum_12_1" in top100
    assert "rank_momentum_6_1" not in top100
    assert "rank_momentum_6_1" in top500
    assert not any(column.startswith("raw_") for column in top100)


def test_master_loader_honors_dates_and_explicit_columns(data_config, synthetic_panel) -> None:
    atomic_write_parquet(synthetic_panel, data_config.master_panel_path)
    result = load_master_panel(
        start="2020-02-29",
        end="2020-02-29",
        universe="top100",
        columns=["permno", "eom", "lambda_adv"],
    )
    assert len(result) == 100
    assert list(result.columns) == ["permno", "eom", "lambda_adv"]
    assert result["eom"].eq(pd.Timestamp("2020-02-29")).all()


def test_cost_loader_reads_precomputed_inputs(data_config, synthetic_panel) -> None:
    columns = [
        "permno",
        "eom",
        "adv_6m",
        "lambda_adv",
        "in_top100",
        "in_top300",
        "in_top500",
    ]
    atomic_write_parquet(synthetic_panel[columns], data_config.cost_inputs_path)
    result = load_cost_inputs(universe="top300")
    assert len(result) == 900
    assert (result["lambda_adv"] == 0.2 / result["adv_6m"]).all()


def test_risk_loader_reports_fallback_manifest(data_config) -> None:
    data_config.processed_dir.mkdir(parents=True, exist_ok=True)
    (data_config.processed_dir / "risk_manifest.json").write_text(
        '{"status": "unavailable", "reason": "synthetic diagnostic"}', encoding="utf-8"
    )
    try:
        load_risk_inputs(universe="top500")
    except FileNotFoundError as exc:
        assert "synthetic diagnostic" in str(exc)
    else:
        raise AssertionError("Expected missing risk inputs to raise a clear error")


def test_risk_loader_filters_prebuilt_risk_files(data_config) -> None:
    risk = data_config.risk_dir
    atomic_write_parquet(
        pd.DataFrame(
            {
                "eom": pd.to_datetime(["2020-01-31", "2020-01-31"]),
                "permno": [1, 2],
                "universe": ["top100", "top500"],
                "rank_log_mcap": [0.2, 0.8],
            }
        ),
        risk / "factor_exposures.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "universe": ["top100", "top500"],
                "factor": ["size", "size"],
                "factor_return": [0.01, 0.02],
            }
        ),
        risk / "factor_returns_daily.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame(
            {
                "eom": pd.to_datetime(["2020-01-31", "2020-01-31"]),
                "universe": ["top100", "top500"],
                "factor_1": ["size", "size"],
                "factor_2": ["size", "size"],
                "covariance": [0.1, 0.2],
                "diagonal_fallback": [True, True],
            }
        ),
        risk / "factor_cov_monthly.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame(
            {
                "eom": pd.to_datetime(["2020-01-31", "2020-01-31"]),
                "universe": ["top100", "top500"],
                "permno": [1, 2],
                "idio_var": [0.03, 0.04],
            }
        ),
        risk / "idio_var_monthly.parquet",
    )
    result = load_risk_inputs(universe="top100")
    assert set(result) == {
        "factor_exposures",
        "factor_returns_daily",
        "factor_cov_monthly",
        "idio_var_monthly",
    }
    assert all(len(frame) == 1 for frame in result.values())
    assert result["factor_exposures"]["permno"].tolist() == [1]
