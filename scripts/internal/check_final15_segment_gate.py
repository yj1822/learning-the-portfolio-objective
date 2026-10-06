from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.config import load_model_config  # noqa: E402
from implementable_frontier.models.formal_analysis import (  # noqa: E402
    risk_calibration_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply post-segment Final15 safety gates")
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument(
        "--config", default="configs/model_top500_final15_cumulative.yaml"
    )
    parser.add_argument("--first-segment", action="store_true")
    args = parser.parse_args()
    segment_dir = Path(args.segment_dir)
    if not segment_dir.is_absolute():
        segment_dir = PROJECT_ROOT / segment_dir
    config = load_model_config(args.config)
    safety = config.values["safety"]
    returns = pd.read_parquet(segment_dir / "returns.parquet")
    selection = pd.read_parquet(segment_dir / "validation_selection.parquet")
    runtime = json.loads((segment_dir / "runtime_profile.json").read_text(encoding="utf-8"))
    portfolio = returns[returns["method"].astype(str).str.startswith("portfolio_ml")]
    risk = risk_calibration_report(returns)
    risk_year = risk[risk["scope"].eq("year")].copy()
    ratio_counts = (
        risk_year.assign(over_two=risk_year["realized_to_predicted_ratio"].gt(2.0))
        .groupby("method", sort=True)
        .agg(years=("test_year", "size"), years_over_two=("over_two", "sum"))
        .reset_index()
    )
    ratio_counts["allowed_over_two"] = ratio_counts["years"].map(
        lambda count: max(1, int(np.ceil(0.1 * count)))
    )
    factor_fallback = int(
        (
            ~returns["risk_model_used"].astype(str).eq("factor")
            | returns["risk_fallback_reason"].notna()
        ).sum()
    )
    matrix_fallback = int((portfolio["adjustment_mode"] != "matrix_m").sum())
    covariance_repairs = int(returns["covariance_repair_count"].sum())
    future_violations = int(selection["future_information_used"].fillna(False).sum())
    cutoff_violations = int(
        (
            pd.to_numeric(selection["latest_validation_year"], errors="coerce")
            >= pd.to_numeric(selection["test_year"], errors="coerce")
        ).sum()
    )
    estimated_hours = float(runtime["full_runtime_estimate"]["estimated_full_hours"])
    checks = {
        "factor_fallback_zero": factor_fallback == 0,
        "matrix_m_fallback_zero": matrix_fallback == 0,
        "covariance_repair_zero": covariance_repairs == 0,
        "peak_memory_within_limit": float(runtime["peak_working_set_gib"])
        <= float(safety["max_peak_memory_gib"]),
        "portfolio_mean_leverage_within_limit": float(portfolio["leverage"].mean())
        <= float(safety["max_portfolio_mean_leverage"]),
        "portfolio_monthly_leverage_within_limit": float(portfolio["leverage"].max())
        <= float(safety["max_portfolio_monthly_leverage"]),
        "portfolio_mean_turnover_within_limit": float(portfolio["turnover"].mean())
        <= float(safety["max_portfolio_mean_turnover"]),
        "portfolio_monthly_turnover_within_limit": float(portfolio["turnover"].max())
        <= float(safety["max_portfolio_monthly_turnover"]),
        "primary_method_risk_ratio_not_widely_above_two": bool(
            (ratio_counts["years_over_two"] <= ratio_counts["allowed_over_two"]).all()
        ),
        "future_information_zero": future_violations == 0,
        "validation_cutoff_violations_zero": cutoff_violations == 0,
        "first_segment_runtime_within_24h": (
            estimated_hours <= float(safety["max_full_runtime_hours"])
            if args.first_segment
            else True
        ),
    }
    payload = {
        "passed": all(checks.values()),
        "segment_dir": str(segment_dir),
        "checks": checks,
        "factor_fallback_count": factor_fallback,
        "matrix_m_fallback_count": matrix_fallback,
        "covariance_repair_count": covariance_repairs,
        "peak_working_set_gib": float(runtime["peak_working_set_gib"]),
        "estimated_full_hours": estimated_hours,
        "portfolio_mean_leverage": float(portfolio["leverage"].mean()),
        "portfolio_max_leverage": float(portfolio["leverage"].max()),
        "portfolio_mean_turnover": float(portfolio["turnover"].mean()),
        "portfolio_max_turnover": float(portfolio["turnover"].max()),
        "risk_ratio_summary": ratio_counts.to_dict(orient="records"),
        "future_information_violations": future_violations,
        "validation_cutoff_violations": cutoff_violations,
    }
    path = segment_dir / "safety_gate.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
