from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from implementable_frontier.data.hub import HubIdentity
from implementable_frontier.data.loaders import load_master_panel
from implementable_frontier.data.preparation import build_us_equity_ml_dataset, export_sample


class FakeHub:
    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel
        self.identity = HubIdentity(
            version="test", module_path="synthetic://trading_data_hub", config_path=None
        )

    def load_master(self, config):
        return self.panel.copy()

    def build_master(self, config):
        raise AssertionError("The available synthetic master must be loaded, not rebuilt")

    def load_risk(self, config):
        raise FileNotFoundError("synthetic risk is intentionally unavailable")

    def build_risk(self, config, panel):
        raise RuntimeError("synthetic risk builder unavailable")

    def download_wrds(self, config):
        raise AssertionError("Synthetic build must not request WRDS")


def test_build_writes_research_copy_manifests_and_validation(data_config, synthetic_panel) -> None:
    with pytest.warns(RuntimeWarning, match="Risk inputs are unavailable"):
        result = build_us_equity_ml_dataset(data_config, FakeHub(synthetic_panel))
    assert result.row_count == len(synthetic_panel)
    assert result.risk_status == "unavailable"
    assert result.validation_passed is True
    assert data_config.master_panel_path.exists()
    assert data_config.cost_inputs_path.exists()
    assert (data_config.processed_dir / "manifest.json").exists()
    assert (data_config.processed_dir / "metadata.json").exists()
    risk = json.loads((data_config.processed_dir / "risk_manifest.json").read_text(encoding="utf-8"))
    assert risk["thesis_side_estimation"] is False


def test_sample_is_filtered_from_existing_master(data_config, synthetic_panel) -> None:
    with pytest.warns(RuntimeWarning, match="Risk inputs are unavailable"):
        build_us_equity_ml_dataset(data_config, FakeHub(synthetic_panel))
    output = export_sample(
        data_config,
        universe="top100",
        feature_set="core10",
        start="2020-01-31",
        end="2020-02-29",
    )
    sample = pd.read_parquet(output)
    loaded = load_master_panel(
        start="2020-01-31", end="2020-02-29", universe="top100", feature_set="core10"
    )
    pd.testing.assert_frame_equal(sample, loaded)
    assert output.name == "us_equity_ml_top100_core10_2020_2020.parquet"
