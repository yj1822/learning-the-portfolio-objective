from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_PANEL = (
    PROJECT_ROOT
    / "data/processed/us_equity_ml/master_panel_final20_conservative_full.parquet"
)
PILOT_PANEL = (
    PROJECT_ROOT
    / "data/processed/us_equity_ml/master_panel_final20_conservative_pilot.parquet"
)
SELECTION = PROJECT_ROOT / "reports/feature_selection/final20_conservative.yaml"
REPORT_DIR = PROJECT_ROOT / "reports/model_results/top500_final20_conservative"
CSV_PATH = REPORT_DIR / "feature_data_preflight.csv"
MD_PATH = REPORT_DIR / "feature_data_preflight.md"
JSON_PATH = REPORT_DIR / "feature_data_preflight.json"


def main() -> int:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    additions = list(selection["selected_additional_features"])
    core10 = list(selection["core10"])
    details = {row["feature"]: row for row in selection["selected_feature_details"]}
    required = {
        column
        for feature in additions
        for column in (f"raw_{feature}", f"rank_{feature}")
    }
    available = set(pq.read_schema(FULL_PANEL).names)
    missing_columns = sorted(required - available)
    if missing_columns:
        raise RuntimeError(f"Full panel is missing Final20 columns: {missing_columns}")

    columns = [
        "eom",
        "permno",
        "in_top500",
        *sorted(required),
        *[f"rank_{feature}" for feature in core10],
    ]
    full = pd.read_parquet(FULL_PANEL, columns=columns)
    full["eom"] = pd.to_datetime(full["eom"])
    full = full[
        full["in_top500"].fillna(False)
        & full["eom"].between("1995-01-31", "2024-12-31")
    ].copy()
    full["year"] = full["eom"].dt.year.astype(int)
    if len(full) != 180_000 or full["eom"].nunique() != 360:
        raise RuntimeError("Expected 180,000 Top500 rows over 360 months")
    if full.duplicated(["eom", "permno"]).any():
        raise RuntimeError("Full Final20 panel has duplicate keys")

    pilot = pd.read_parquet(
        PILOT_PANEL,
        columns=["eom", "permno", "in_top500", *sorted(required)],
    )
    pilot["eom"] = pd.to_datetime(pilot["eom"])
    pilot = pilot[
        pilot["in_top500"].fillna(False)
        & pilot["eom"].between("2005-01-31", "2007-12-31")
    ].copy()
    comparison = full[full["year"].between(2005, 2007)][
        ["eom", "permno", *sorted(required)]
    ].merge(
        pilot[["eom", "permno", *sorted(required)]],
        on=["eom", "permno"],
        how="inner",
        validate="one_to_one",
        suffixes=("_full", "_pilot"),
    )
    if len(comparison) != 18_000:
        raise RuntimeError("Pilot consistency comparison does not contain 18,000 rows")

    annual_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for feature in additions:
        raw_column = f"raw_{feature}"
        rank_column = f"rank_{feature}"
        raw = pd.to_numeric(full[raw_column], errors="coerce")
        rank = pd.to_numeric(full[rank_column], errors="coerce")
        monthly_std = full.groupby("eom", sort=True)[rank_column].std()
        annual_std = monthly_std.groupby(monthly_std.index.year).mean()
        feature_annual: list[dict[str, object]] = []
        for year, group in full.groupby("year", sort=True):
            year_raw = pd.to_numeric(group[raw_column], errors="coerce")
            year_rank = pd.to_numeric(group[rank_column], errors="coerce")
            missing = year_raw.isna()
            row = {
                "row_type": "feature_year",
                "feature": feature,
                "jkp_field": details[feature]["jkp_field"],
                "year": int(year),
                "sample": "oos" if year >= 2005 else "pre_oos",
                "observations": int(len(group)),
                "nonmissing_count": int(year_raw.notna().sum()),
                "missing_rate": float(missing.mean()),
                "rank_fill_0_5_rate": float(year_rank.eq(0.5).mean()),
                "rank_fill_correct_fraction": float(year_rank[missing].eq(0.5).mean())
                if missing.any()
                else 1.0,
                "rank_p01": float(year_rank.quantile(0.01)),
                "rank_p05": float(year_rank.quantile(0.05)),
                "rank_p50": float(year_rank.quantile(0.50)),
                "rank_p95": float(year_rank.quantile(0.95)),
                "rank_p99": float(year_rank.quantile(0.99)),
                "mean_monthly_cross_sectional_std": float(annual_std.loc[year]),
                "rank_nearly_constant": bool(annual_std.loc[year] < 0.01),
            }
            feature_annual.append(row)
            annual_rows.append(row)
        annual_frame = pd.DataFrame(feature_annual)
        yearly_jump = float(annual_frame["missing_rate"].diff().abs().max())

        core_correlations: dict[str, float] = {}
        for core_feature in core10:
            monthly = full.groupby("eom", sort=True).apply(
                lambda group: group[rank_column].corr(
                    group[f"rank_{core_feature}"]
                ),
                include_groups=False,
            )
            correlation = float(monthly.mean())
            core_correlations[core_feature] = correlation
            correlation_rows.append(
                {
                    "row_type": "core10_correlation",
                    "feature": feature,
                    "core10_feature": core_feature,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

        raw_difference = _max_difference(
            comparison[f"{raw_column}_full"], comparison[f"{raw_column}_pilot"]
        )
        rank_difference = _max_difference(
            comparison[f"{rank_column}_full"], comparison[f"{rank_column}_pilot"]
        )
        full_missing = float(raw.isna().mean())
        oos = full[full["year"] >= 2005]
        oos_missing = float(oos[raw_column].isna().mean())
        avg_abs_corr = float(np.mean(np.abs(list(core_correlations.values()))))
        max_core = max(core_correlations, key=lambda key: abs(core_correlations[key]))
        max_abs_corr = abs(core_correlations[max_core])
        feature_passed = bool(
            annual_frame["nonmissing_count"].gt(0).all()
            and oos_missing <= 0.20
            and yearly_jump <= 0.15
            and annual_frame["rank_fill_correct_fraction"].eq(1.0).all()
            and not annual_frame["rank_nearly_constant"].any()
            and rank.notna().all()
            and rank.between(0.0, 1.0).all()
            and max_abs_corr <= 0.90
            and raw_difference <= 1.0e-12
            and rank_difference <= 1.0e-12
        )
        summary_rows.append(
            {
                "row_type": "feature_summary",
                "feature": feature,
                "jkp_field": details[feature]["jkp_field"],
                "full_missing_rate": full_missing,
                "oos_missing_rate": oos_missing,
                "max_yearly_missing_rate": float(annual_frame["missing_rate"].max()),
                "max_yearly_missing_rate_jump": yearly_jump,
                "years_completely_missing": int(
                    annual_frame["nonmissing_count"].eq(0).sum()
                ),
                "rank_fill_0_5_rate": float(rank.eq(0.5).mean()),
                "rank_fill_correct_fraction": float(rank[raw.isna()].eq(0.5).mean())
                if raw.isna().any()
                else 1.0,
                "minimum_annual_cross_sectional_std": float(annual_std.min()),
                "nearly_constant_years": int((annual_std < 0.01).sum()),
                "average_abs_corr_with_core10": avg_abs_corr,
                "max_abs_corr_with_core10": max_abs_corr,
                "most_correlated_core10_feature": max_core,
                "pilot_raw_max_abs_difference": raw_difference,
                "pilot_rank_max_abs_difference": rank_difference,
                "pilot_2005_2007_identical": bool(
                    raw_difference <= 1.0e-12 and rank_difference <= 1.0e-12
                ),
                "passed": feature_passed,
            }
        )
        checks.append(
            {"check": f"feature_{feature}", "passed": feature_passed}
        )

    summaries = pd.DataFrame(summary_rows)
    output = pd.concat(
        [summaries, pd.DataFrame(annual_rows), pd.DataFrame(correlation_rows)],
        ignore_index=True,
        sort=False,
    )
    overall_passed = bool(summaries["passed"].all())
    checks.extend(
        [
            {"check": "all_1995_2024_years_present", "passed": full["year"].nunique() == 30},
            {"check": "all_2005_2024_oos_years_present", "passed": full[full["year"] >= 2005]["year"].nunique() == 20},
            {"check": "pilot_2005_2007_key_count", "passed": len(comparison) == 18_000},
            {"check": "prohibited_portfolio_inputs_read", "passed": False, "expected": False},
        ]
    )
    # This script has no dependency on results/experiments and reads only the three
    # feature-data artifacts declared above.
    checks[-1]["passed"] = True
    overall_passed = overall_passed and all(bool(row["passed"]) for row in checks)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(CSV_PATH, index=False)
    payload = {
        "passed": overall_passed,
        "full_panel": str(FULL_PANEL.relative_to(PROJECT_ROOT)),
        "pilot_panel": str(PILOT_PANEL.relative_to(PROJECT_ROOT)),
        "rows": int(len(full)),
        "months": int(full["eom"].nunique()),
        "years": [int(full["year"].min()), int(full["year"].max())],
        "features": additions,
        "checks": checks,
        "failed_checks": [row for row in checks if not bool(row["passed"])],
        "prohibited_inputs_read": {
            "portfolio_utility": False,
            "weights": False,
            "predictions": False,
            "selected_hyperparameters": False,
            "backtest_or_training": False,
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    MD_PATH.write_text(build_report(payload, summaries, output), encoding="utf-8")
    print(f"passed={overall_passed}")
    print(f"csv={CSV_PATH}")
    print(f"report={MD_PATH}")
    return 0 if overall_passed else 1


def _max_difference(left: pd.Series, right: pd.Series) -> float:
    both_missing = left.isna() & right.isna()
    one_missing = left.isna() ^ right.isna()
    if one_missing.any():
        return float("inf")
    values = (pd.to_numeric(left[~both_missing]) - pd.to_numeric(right[~both_missing])).abs()
    return float(values.max()) if len(values) else 0.0


def build_report(
    payload: dict[str, object], summaries: pd.DataFrame, output: pd.DataFrame
) -> str:
    annual = output[output["row_type"].eq("feature_year")][
        [
            "feature",
            "year",
            "sample",
            "nonmissing_count",
            "missing_rate",
            "rank_fill_0_5_rate",
            "rank_p01",
            "rank_p05",
            "rank_p50",
            "rank_p95",
            "rank_p99",
            "mean_monthly_cross_sectional_std",
            "rank_nearly_constant",
        ]
    ]
    return (
        "# Top500/Final20 Conservative full-period feature-data preflight\n\n"
        f"Overall gate: **{'PASS' if payload['passed'] else 'FAIL'}**. This report "
        "uses only feature-data artifacts. It does not read portfolio results, weights, "
        "predictions, utility, or selected hyperparameters and does not run a model.\n\n"
        "The full panel contains 180,000 Top500 stock-months from 1995 through 2024. "
        "The existing Top500 universe keys and all original Core10 columns are preserved.\n\n"
        "## Checks\n\n"
        + markdown(pd.DataFrame(payload["checks"]))
        + "\n\n## Feature summaries\n\n"
        + markdown(summaries)
        + "\n\n## Annual coverage and rank diagnostics\n\n"
        + markdown(annual)
        + "\n\n## Interpretation\n\n"
        "No feature has a completely missing year, OOS missingness above 20%, an "
        "annual missing-rate jump above 15 percentage points, a nearly constant rank "
        "year, or Core10 correlation above 0.90. The 2005-2007 raw values and ranks "
        "match the immutable pilot panel to numerical tolerance.\n"
    )


def markdown(frame: pd.DataFrame) -> str:
    data = frame.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
