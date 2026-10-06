from __future__ import annotations

import json

from implementable_frontier.data.validation import validate_us_equity_ml_dataset


def test_validation_report_covers_synthetic_invariants(data_config, synthetic_panel) -> None:
    report = validate_us_equity_ml_dataset(config=data_config, panel=synthetic_panel)
    assert report["passed"] is True
    assert report["checks"]["nested_universes"]["passed"] is True
    assert report["checks"]["rank_features_in_zero_one"]["passed"] is True
    assert report["checks"]["missing_features_use_neutral_rank"]["passed"] is True
    assert report["checks"]["transaction_cost_lambda"]["passed"] is True
    json_path = data_config.validation_dir / "us_equity_ml_validation.json"
    markdown_path = data_config.validation_dir / "us_equity_ml_validation.md"
    assert json_path.exists() and markdown_path.exists()
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["row_count"] == len(synthetic_panel)


def test_validation_detects_subset_rank_and_lambda_failures(data_config, synthetic_panel) -> None:
    broken = synthetic_panel.copy()
    broken.loc[0, "in_top300"] = False
    broken.loc[1, "rank_momentum_12_1"] = 1.5
    broken.loc[2, "lambda_adv"] *= 2
    report = validate_us_equity_ml_dataset(
        config=data_config, panel=broken, write_reports=False
    )
    assert report["passed"] is False
    assert report["checks"]["nested_universes"]["passed"] is False
    assert report["checks"]["rank_features_in_zero_one"]["passed"] is False
    assert report["checks"]["transaction_cost_lambda"]["passed"] is False

