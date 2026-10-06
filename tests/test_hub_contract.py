from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from implementable_frontier.data.config import load_data_config
from implementable_frontier.data.hub import (
    TradingDataHubAdapter,
    TradingDataHubUnavailableError,
)


def test_unavailable_hub_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADING_DATA_HUB_ROOT", raising=False)
    monkeypatch.delenv("DATAHUB_CONFIG", raising=False)
    original = importlib.import_module

    def missing(name: str, *args, **kwargs):
        if name == "trading_data_hub":
            raise ImportError("synthetic missing package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(TradingDataHubUnavailableError, match="TRADING_DATA_HUB_ROOT"):
        TradingDataHubAdapter()


def test_adapter_combines_all_configured_risk_universes(data_config) -> None:
    class FakeModule:
        @staticmethod
        def load_us_equity_risk_inputs(start, end, universe):
            return {
                "factor_exposures": pd.DataFrame(
                    {
                        "eom": [pd.Timestamp("2020-01-31")],
                        "permno": [1],
                        "universe": [universe],
                    }
                )
            }

    adapter = object.__new__(TradingDataHubAdapter)
    adapter.module = FakeModule()
    result = adapter.load_risk(data_config)
    assert set(result["factor_exposures"]["universe"]) == {
        "top100",
        "top300",
        "top500",
    }


@pytest.mark.integration
def test_upstream_builder_contract_without_wrds(tmp_path: Path) -> None:
    module = pytest.importorskip("datahub.us_equity.master_panel")
    months = pd.to_datetime(["2020-01-31", "2020-02-29"])
    jkp_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for eom in months:
        eom = pd.Timestamp(eom)
        for stock_index in range(505):
            permno = 10000 + stock_index
            jkp_rows.append(
                {
                    "id": f"id-{permno}",
                    "eom": eom,
                    "excntry": "USA",
                    "gvkey": str(permno),
                    "permno": permno,
                    "me": float(1_000_000 - stock_index),
                    "ret_exc_lead1m": 0.01,
                    "common": 1,
                    "exch_main": 1,
                    "primary_sec": 1,
                    "obs_main": 1,
                    "feature_a": float(stock_index),
                    "feature_b": np.nan if stock_index == 10 else float(505 - stock_index),
                }
            )
            monthly_rows.append(
                {
                    "permno": permno,
                    "date": eom,
                    "prc": 10.0,
                    "ret": 0.01,
                    "dlret": np.nan,
                    "vol": 1000.0,
                    "shrout": 100.0,
                    "shrcd": 10,
                    "exchcd": 1,
                    "siccd": 2000,
                }
            )
            daily_rows.append(
                {"permno": permno, "date": eom - pd.offsets.Day(1), "prc": 10.0, "vol": 1000.0}
            )
    output = tmp_path / "master_panel.parquet"
    panel = module.build_us_equity_ml_master_panel(
        "2020-01-01",
        "2020-02-29",
        max_universe_n=500,
        features=["feature_a", "feature_b"],
        jkp_data=pd.DataFrame(jkp_rows),
        crsp_monthly=pd.DataFrame(monthly_rows),
        crsp_daily=pd.DataFrame(daily_rows),
        raw_root=tmp_path / "raw",
        output_path=output,
        feature_aliases_path=tmp_path,
    )
    assert len(panel) == 1000
    assert panel["in_top100"].sum() == 200
    assert panel["in_top300"].sum() == 600
    assert panel["in_top500"].all()
    feature_ranks = panel[["rank_feature_a", "rank_feature_b"]]
    assert feature_ranks.apply(lambda column: column.between(0, 1).all()).all()
    assert panel.loc[panel["raw_feature_b"].isna(), "rank_feature_b"].eq(0.5).all()
    assert np.allclose(panel["lambda_adv"], 0.2 / panel["adv_6m"])


@pytest.mark.integration
def test_upstream_wrds_missing_credentials_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = pytest.importorskip("datahub.sources.wrds")
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    connector = module.WRDSConnector(raw_root=tmp_path, username=None, password=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="WRDS credentials are not configured"):
            connector.download_crsp_monthly("2020-01-01", "2020-01-31")


def test_wrds_download_is_year_partitioned_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeConnector:
        calls: list[tuple[str, str]] = []

        def __init__(self) -> None:
            self.raw_root = tmp_path / "raw"
            self.last_written_paths: list[Path] = []

        def download_jkp_global_factors(self, start, end, country, features):
            self.calls.append(("jkp", start))
            year = pd.Timestamp(start).year
            frame = pd.DataFrame({"permno": [1, 2], "eom": pd.to_datetime([f"{year}-01-31"] * 2)})
            path = self.raw_root / "wrds" / "jkp_global_factor" / f"excntry={country}" / f"year={year}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            return frame

        def download_crsp_monthly(self, start, end, permnos):
            self.calls.append(("monthly", start))
            return self._write("crsp_monthly", start, "date")

        def download_crsp_daily(self, start, end, permnos):
            self.calls.append(("daily", start))
            return self._write("crsp_daily", start, "date")

        def _write(self, dataset, start, date_column):
            year = pd.Timestamp(start).year
            frame = pd.DataFrame({"permno": [1, 2], date_column: pd.to_datetime([f"{year}-01-02"] * 2)})
            path = self.raw_root / "wrds" / dataset / f"year={year}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            return frame

        def close(self):
            return None

    class FakeModule:
        WRDSConnector = FakeConnector

    values = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "data_us_equity_ml.yaml").read_text(
            encoding="utf-8"
        )
    )
    values["start_date"] = "2020-01-01"
    values["analysis_start"] = "2020-01-01"
    values["oos_start"] = "2021-01-01"
    values["end_date"] = "2021-12-31"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data_us_equity_ml.yaml"
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    config = load_data_config(config_path)

    adapter = object.__new__(TradingDataHubAdapter)
    monkeypatch.setattr(adapter, "_import_internal", lambda name: FakeModule())
    first = adapter.download_wrds(config)
    first_call_count = len(FakeConnector.calls)
    second = adapter.download_wrds(config)
    assert first_call_count == 6
    assert len(FakeConnector.calls) == first_call_count
    assert first["crsp_daily_rows"] == second["crsp_daily_rows"] == 4
    assert second["years"]["2020"] == {
        "jkp": "existing",
        "crsp_monthly": "existing",
        "crsp_daily": "existing",
    }
