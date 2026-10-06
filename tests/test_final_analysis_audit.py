from __future__ import annotations

import pandas as pd
import pytest

from scripts.internal.audit_final_analysis_checkpoints import _scenario_progress
from scripts.run_final_analysis_scenario import _portfolio_leverage_summary


@pytest.mark.parametrize(
    "column",
    ["leverage", "gross_leverage", "actual_portfolio_leverage"],
)
def test_quality_wrapper_accepts_monthly_leverage_aliases(column: str) -> None:
    frame = pd.DataFrame({column: [0.5, 1.0, 1.5]})

    average, maximum, mapping = _portfolio_leverage_summary(frame)

    assert average == pytest.approx(1.0)
    assert maximum == pytest.approx(1.5)
    assert mapping == {"average": column, "maximum": column}


def test_quality_wrapper_accepts_aggregated_leverage_fields() -> None:
    frame = pd.DataFrame(
        {"average_leverage": [0.4, 0.6], "max_leverage": [0.8, 1.2]}
    )

    average, maximum, mapping = _portfolio_leverage_summary(frame)

    assert average == pytest.approx(0.5)
    assert maximum == pytest.approx(1.2)
    assert mapping == {"average": "average_leverage", "maximum": "max_leverage"}


def test_superseded_checkpoint_error_is_historical_non_blocking(tmp_path) -> None:
    boundary = tmp_path / "validation" / "candidate" / "1997"
    boundary.mkdir(parents=True)
    (boundary / "error.json").write_text(
        '{"error_type":"ArrowInvalid","failed_at_utc":"2026-08-07T01:18:23Z"}',
        encoding="utf-8",
    )
    (boundary / "complete.json").write_text(
        '{"completed_at_utc":"2026-08-07T01:20:11Z"}', encoding="utf-8"
    )
    for name in ("returns.parquet", "state.parquet", "state_metadata.json"):
        (boundary / name).write_bytes(b"complete")

    progress = _scenario_progress(tmp_path)

    assert progress["error_records"] == 1
    assert len(progress["historical_non_blocking_error_records"]) == 1
    assert progress["blocking_error_records"] == []
