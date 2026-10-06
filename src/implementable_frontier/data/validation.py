from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from implementable_frontier.data.config import DataConfig, load_data_config
from implementable_frontier.data.io import atomic_write_json, atomic_write_text


ACCOUNTING_FEATURES = {
    "book_to_market",
    "earnings_to_price",
    "gross_profitability",
    "operating_profitability",
    "asset_growth",
}


def validate_us_equity_ml_dataset(
    config: DataConfig | None = None,
    panel: pd.DataFrame | None = None,
    *,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Validate structural, cost, coverage, split, and look-ahead invariants."""
    config = config or load_data_config()
    if panel is None:
        if not config.master_panel_path.exists():
            raise FileNotFoundError(f"Master panel not found: {config.master_panel_path}")
        panel = pd.read_parquet(config.master_panel_path)
    else:
        panel = panel.copy()
    panel.columns = [str(column).lower() for column in panel.columns]
    if "eom" in panel:
        panel["eom"] = pd.to_datetime(panel["eom"], errors="coerce")

    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    required = {
        "eom",
        "permno",
        "ret_exc_lead1m",
        "price",
        "lag_mktcap",
        "adv_6m",
        "lambda_adv",
        "universe_rank_mcap",
        "in_top100",
        "in_top300",
        "in_top500",
        *[f"raw_{feature}" for feature in config.final15],
        *[f"rank_{feature}" for feature in config.final15],
    }
    missing_required = sorted(required - set(panel))
    _record(
        checks,
        errors,
        warnings,
        "required_columns",
        not missing_required,
        "error",
        {"missing": missing_required},
        "Required master-panel columns are missing.",
    )
    if missing_required:
        return _finish_report(config, panel, checks, errors, warnings, write_reports)

    duplicate_count = int(panel.duplicated(["eom", "permno"], keep=False).sum())
    _record(
        checks,
        errors,
        warnings,
        "unique_stock_month_key",
        duplicate_count == 0,
        "error",
        {"duplicate_rows": duplicate_count},
        "Duplicate eom + permno rows were found.",
    )

    flags = {name: panel[name].fillna(False).astype(bool) for name in ("in_top100", "in_top300", "in_top500")}
    subset_violations = {
        "top100_not_top300": int((flags["in_top100"] & ~flags["in_top300"]).sum()),
        "top300_not_top500": int((flags["in_top300"] & ~flags["in_top500"]).sum()),
    }
    _record(
        checks,
        errors,
        warnings,
        "nested_universes",
        sum(subset_violations.values()) == 0,
        "error",
        subset_violations,
        "top100/top300/top500 are not nested.",
    )

    top500_counts = panel.loc[flags["in_top500"]].groupby("eom")["permno"].nunique().sort_index()
    unique_stocks = int(panel["permno"].nunique())
    top500_min = int(config.values["validation"]["top500_min_count"])
    small_synthetic = unique_stocks < top500_min
    low_months = top500_counts[top500_counts < top500_min]
    coverage_passed = small_synthetic or low_months.empty
    _record(
        checks,
        errors,
        warnings,
        "top500_monthly_coverage",
        coverage_passed,
        "warning",
        {
            "small_panel_not_applicable": small_synthetic,
            "min": _integer_stat(top500_counts, "min"),
            "median": _float_stat(top500_counts, "median"),
            "max": _integer_stat(top500_counts, "max"),
            "months_below_threshold": int(len(low_months)),
            "threshold": top500_min,
        },
        "Some full-sample months contain materially fewer than 500 eligible stocks.",
    )

    feature_missing = {
        feature: float(panel[f"raw_{feature}"].isna().mean()) for feature in config.final15
    }
    checks["raw_feature_missing_rates"] = {
        "passed": True,
        "severity": "info",
        "details": feature_missing,
    }
    high_missing = {
        feature: rate for feature, rate in feature_missing.items() if rate > 0.50
    }
    _record(
        checks,
        errors,
        warnings,
        "raw_feature_coverage",
        not high_missing,
        "warning",
        {"features_above_50pct_missing": high_missing},
        "One or more final15 raw features are more than 50% missing; verify JKP aliases and FF inputs.",
    )

    rank_columns = [f"rank_{feature}" for feature in config.final15]
    rank_out_of_range = {
        column: int((~pd.to_numeric(panel[column], errors="coerce").between(0.0, 1.0)).sum())
        for column in rank_columns
    }
    _record(
        checks,
        errors,
        warnings,
        "rank_features_in_zero_one",
        sum(rank_out_of_range.values()) == 0,
        "error",
        rank_out_of_range,
        "One or more rank features fall outside [0, 1] or are missing.",
    )

    neutral = float(config.values["missing_feature_value"])
    neutral_violations = {}
    for feature in config.final15:
        missing_raw = panel[f"raw_{feature}"].isna()
        rank = pd.to_numeric(panel[f"rank_{feature}"], errors="coerce")
        neutral_violations[feature] = int((missing_raw & ~np.isclose(rank, neutral)).sum())
    _record(
        checks,
        errors,
        warnings,
        "missing_features_use_neutral_rank",
        sum(neutral_violations.values()) == 0,
        "error",
        {"neutral_value": neutral, "violations": neutral_violations},
        "Missing raw features are not consistently represented by the neutral rank value.",
    )

    label_missing = float(panel["ret_exc_lead1m"].isna().mean())
    label_threshold = float(config.values["validation"]["label_missing_warning_rate"])
    _record(
        checks,
        errors,
        warnings,
        "forward_label_coverage",
        label_missing <= label_threshold,
        "warning",
        {"missing_rate": label_missing, "warning_threshold": label_threshold},
        "ret_exc_lead1m has a high missing rate.",
    )

    adv = pd.to_numeric(panel["adv_6m"], errors="coerce")
    adv_invalid = int((~np.isfinite(adv) | adv.le(0)).sum())
    _record(
        checks,
        errors,
        warnings,
        "positive_adv",
        adv_invalid == 0,
        "error",
        {"invalid_rows": adv_invalid},
        "adv_6m contains missing, non-finite, or non-positive values.",
    )

    lambda_adv = pd.to_numeric(panel["lambda_adv"], errors="coerce")
    lambda_invalid = int((~np.isfinite(lambda_adv) | lambda_adv.le(0)).sum())
    multiplier = float(config.values["transaction_cost"]["lambda_multiplier"])
    expected_lambda = multiplier / adv
    formula_valid = np.isfinite(lambda_adv) & np.isfinite(expected_lambda)
    formula_violations = int(
        (~np.isclose(lambda_adv[formula_valid], expected_lambda[formula_valid], rtol=1e-10, atol=0.0)).sum()
    )
    median_lambda = float(lambda_adv.median()) if lambda_adv.notna().any() else None
    p99_lambda = float(lambda_adv.quantile(0.99)) if lambda_adv.notna().any() else None
    ratio = (
        p99_lambda / median_lambda
        if p99_lambda is not None and median_lambda is not None and median_lambda > 0
        else None
    )
    extreme_threshold = float(config.values["validation"]["lambda_extreme_ratio"])
    _record(
        checks,
        errors,
        warnings,
        "transaction_cost_lambda",
        lambda_invalid == 0 and formula_violations == 0,
        "error",
        {
            "invalid_rows": lambda_invalid,
            "formula_violations": formula_violations,
            "median": median_lambda,
            "p99": p99_lambda,
            "p99_to_median": ratio,
        },
        "lambda_adv is invalid or differs from lambda_multiplier / adv_6m.",
    )
    _record(
        checks,
        errors,
        warnings,
        "transaction_cost_extremes",
        ratio is None or ratio <= extreme_threshold,
        "warning",
        {"p99_to_median": ratio, "warning_threshold": extreme_threshold},
        "lambda_adv has an extreme right tail; inspect low-ADV observations.",
    )

    price = pd.to_numeric(panel["price"], errors="coerce")
    min_price = float(config.values["min_price"])
    price_violations = int((flags["in_top500"] & (price.lt(min_price) | price.isna())).sum())
    _record(
        checks,
        errors,
        warnings,
        "minimum_price",
        price_violations == 0,
        "error",
        {"min_price": min_price, "violations": price_violations},
        "Top500 rows violate the configured minimum-price rule.",
    )

    jump_fraction = float(config.values["validation"]["monthly_count_jump_fraction"])
    count_changes = top500_counts.pct_change().abs()
    jump_months = count_changes[count_changes > jump_fraction]
    _record(
        checks,
        errors,
        warnings,
        "monthly_universe_count_jumps",
        small_synthetic or jump_months.empty,
        "warning",
        {
            "threshold_fraction": jump_fraction,
            "jump_months": {date.date().isoformat(): float(value) for date, value in jump_months.items()},
        },
        "Monthly eligible-stock counts contain abrupt jumps.",
    )

    mktcap_column = "lag_mktcap" if "lag_mktcap" in panel else "mktcap"
    mktcap = pd.to_numeric(panel.loc[flags["in_top500"], mktcap_column], errors="coerce")
    mktcap_positive = int((mktcap.le(0) | ~np.isfinite(mktcap)).sum())
    checks["top500_market_cap_distribution"] = {
        "passed": mktcap_positive == 0,
        "severity": "warning",
        "details": {
            "invalid_rows": mktcap_positive,
            "p01": _quantile(mktcap, 0.01),
            "median": _quantile(mktcap, 0.50),
            "p99": _quantile(mktcap, 0.99),
        },
    }
    if mktcap_positive:
        warnings.append("Top500 market capitalization contains non-positive or non-finite values.")

    overlaps = _period_overlaps(config.values.get("sample_periods", {}))
    _record(
        checks,
        errors,
        warnings,
        "sample_periods_do_not_overlap",
        not overlaps,
        "error",
        {"overlaps": overlaps, "periods": config.values.get("sample_periods", {})},
        "Configured train/validation/test periods overlap.",
    )

    lookahead_details, lookahead_warnings = _lookahead_diagnostics(panel, config)
    checks["lookahead_diagnostics"] = {
        "passed": not lookahead_warnings,
        "severity": "warning",
        "details": lookahead_details,
    }
    warnings.extend(lookahead_warnings)

    risk_details = _risk_diagnostics(config)
    risk_ok = risk_details["status"] == "available"
    checks["risk_inputs"] = {"passed": risk_ok, "severity": "warning", "details": risk_details}
    if not risk_ok:
        warnings.append(
            f"Risk inputs are not fully available (status={risk_details['status']}); inspect risk_manifest.json."
        )

    return _finish_report(config, panel, checks, errors, warnings, write_reports)


def _record(
    checks: dict[str, dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    name: str,
    passed: bool,
    severity: str,
    details: dict[str, Any],
    message: str,
) -> None:
    checks[name] = {"passed": bool(passed), "severity": severity, "details": details}
    if passed:
        return
    if severity == "error":
        errors.append(message)
    elif severity == "warning":
        warnings.append(message)


def _finish_report(
    config: DataConfig,
    panel: pd.DataFrame,
    checks: dict[str, dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    write_reports: bool,
) -> dict[str, Any]:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.values.get("dataset_name", "us_equity_ml"),
        "master_panel": str(config.master_panel_path),
        "passed": not errors,
        "row_count": int(len(panel)),
        "month_count": int(panel["eom"].nunique()) if "eom" in panel else 0,
        "checks": checks,
        "errors": errors,
        "warnings": list(dict.fromkeys(warnings)),
    }
    if write_reports:
        json_path = config.validation_dir / "us_equity_ml_validation.json"
        markdown_path = config.validation_dir / "us_equity_ml_validation.md"
        atomic_write_json(report, json_path)
        atomic_write_text(_render_markdown(report), markdown_path)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# US Equity ML Data Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall status: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- Rows: `{report['row_count']}`",
        f"- Months: `{report['month_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity |",
        "|---|---:|---|",
    ]
    for name, check in report["checks"].items():
        lines.append(
            f"| `{name}` | {'PASS' if check['passed'] else 'REVIEW'} | {check['severity']} |"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {message}" for message in report["errors"] or ["None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {message}" for message in report["warnings"] or ["None."])
    lines.extend(["", "The JSON report contains per-check metrics and distributions.", ""])
    return "\n".join(lines)


def _period_overlaps(periods: dict[str, Any]) -> list[dict[str, str]]:
    parsed = {}
    for name, values in periods.items():
        if not isinstance(values, dict) or "start" not in values or "end" not in values:
            continue
        start, end = pd.Timestamp(values["start"]), pd.Timestamp(values["end"])
        parsed[name] = (start, end)
    overlaps = []
    for (left_name, (left_start, left_end)), (right_name, (right_start, right_end)) in combinations(parsed.items(), 2):
        if max(left_start, right_start) <= min(left_end, right_end):
            overlaps.append({"left": left_name, "right": right_name})
    return overlaps


def _lookahead_diagnostics(panel: pd.DataFrame, config: DataConfig) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    accounting_features = [feature for feature in config.final15 if feature in ACCOUNTING_FEATURES]
    lag_documented = "accounting_lag_months" in panel or bool(
        config.values.get("accounting_reporting_lag_metadata")
    )
    if accounting_features and not lag_documented:
        warnings.append(
            "Accounting features are present without explicit reporting-lag metadata; verify the JKP release timing."
        )
    alignment = {"same_month_correlation": None, "next_month_correlation": None, "observations": 0}
    if {"ret_1m", "ret_exc_lead1m", "permno", "eom"}.issubset(panel):
        comparison = panel[["permno", "eom", "ret_1m", "ret_exc_lead1m"]].copy()
        comparison = comparison.sort_values(["permno", "eom"])
        comparison["next_ret_1m"] = comparison.groupby("permno")["ret_1m"].shift(-1)
        valid = comparison[["ret_1m", "ret_exc_lead1m", "next_ret_1m"]].apply(
            pd.to_numeric, errors="coerce"
        )
        alignment["observations"] = int(valid.dropna().shape[0])
        if alignment["observations"] >= 30:
            alignment["same_month_correlation"] = _finite_or_none(
                valid["ret_exc_lead1m"].corr(valid["ret_1m"])
            )
            alignment["next_month_correlation"] = _finite_or_none(
                valid["ret_exc_lead1m"].corr(valid["next_ret_1m"])
            )
            same, next_month = alignment["same_month_correlation"], alignment["next_month_correlation"]
            if same is not None and next_month is not None and same > next_month + 0.10:
                warnings.append(
                    "ret_exc_lead1m aligns more strongly with same-month returns than next-month returns; inspect label timing."
                )
    return {
        "accounting_features": accounting_features,
        "accounting_reporting_lag_documented": lag_documented,
        "label_alignment": alignment,
        "label_source": "JKP ret_exc_lead1m; thesis code does not shift or reconstruct it",
    }, warnings


def _risk_diagnostics(config: DataConfig) -> dict[str, Any]:
    path = config.processed_dir / "risk_manifest.json"
    if not path.exists():
        return {"status": "missing_manifest", "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid_manifest", "path": str(path), "reason": str(exc)}
    result = {"path": str(path), **value}
    if value.get("status") not in {"available", "empty_fallback"}:
        return result
    required = {
        "factor_exposures.parquet": {"eom", "permno", "universe"},
        "factor_returns_daily.parquet": {"date", "universe", "factor", "factor_return"},
        "factor_cov_monthly.parquet": {
            "eom",
            "universe",
            "factor_1",
            "factor_2",
            "covariance",
        },
        "idio_var_monthly.parquet": {"eom", "universe", "permno", "idio_var"},
    }
    missing_files: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    for filename, columns in required.items():
        risk_path = config.risk_dir / filename
        if not risk_path.exists():
            missing_files.append(filename)
            continue
        available = set(pq.read_schema(risk_path).names)
        missing = sorted(columns - available)
        if missing:
            missing_columns[filename] = missing
    result["missing_files"] = missing_files
    result["missing_columns"] = missing_columns
    result["eom_permno_readable"] = not any(
        filename in missing_files or filename in missing_columns
        for filename in ("factor_exposures.parquet", "idio_var_monthly.parquet")
    )
    if missing_files or missing_columns:
        result["status"] = "invalid_schema"
    return result


def _integer_stat(values: pd.Series, name: str) -> int | None:
    if values.empty:
        return None
    return int(getattr(values, name)())


def _float_stat(values: pd.Series, name: str) -> float | None:
    if values.empty:
        return None
    return float(getattr(values, name)())


def _quantile(values: pd.Series, probability: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.quantile(probability)) if not finite.empty else None


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
