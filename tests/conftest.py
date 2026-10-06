from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
hub_root = os.getenv("TRADING_DATA_HUB_ROOT")
if hub_root:
    hub_source = Path(hub_root).expanduser().resolve() / "src"
    if hub_source.exists() and str(hub_source) not in sys.path:
        sys.path.insert(0, str(hub_source))

from implementable_frontier.data.config import DataConfig, load_data_config  # noqa: E402


@pytest.fixture
def data_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataConfig:
    values = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "data_us_equity_ml.yaml").read_text(encoding="utf-8")
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data_us_equity_ml.yaml"
    config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    config = load_data_config(config_path)
    monkeypatch.setenv("IEF_DATA_CONFIG", str(config_path))
    monkeypatch.setenv("IEF_DATA_ROOT", str(config.processed_dir))
    return config


@pytest.fixture
def synthetic_panel(data_config: DataConfig) -> pd.DataFrame:
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows: list[dict[str, object]] = []
    for month_index, eom in enumerate(months):
        for stock_index in range(500):
            rank = stock_index + 1
            adv = 1_000_000.0 + 10_000.0 * stock_index
            row: dict[str, object] = {
                "permno": 10000 + stock_index,
                "gvkey": str(20000 + stock_index),
                "eom": eom,
                "ret_1m": 0.001 * month_index + 0.00001 * stock_index,
                "ret_exc_lead1m": 0.001 * (month_index + 1) + 0.00001 * stock_index,
                "price": 10.0 + stock_index / 10,
                "mktcap": float(10_000_000 - stock_index * 10_000),
                "lag_mktcap": float(10_000_000 - stock_index * 10_000),
                "adv_6m": adv,
                "lambda_adv": 0.2 / adv,
                "vol_12m": 0.2,
                "beta_mkt": 1.0,
                "idio_vol": 0.1,
                "sic": 2000,
                "siccd": 2000,
                "universe_rank_mcap": float(rank),
                "in_top100": rank <= 100,
                "in_top300": rank <= 300,
                "in_top500": True,
                "accounting_lag_months": 6,
            }
            for feature_index, feature in enumerate(data_config.final15):
                value = (stock_index + feature_index) / (499 + feature_index)
                row[f"raw_{feature}"] = np.nan if stock_index == feature_index else value
                row[f"rank_{feature}"] = 0.5 if stock_index == feature_index else value
            rows.append(row)
    return pd.DataFrame(rows)

