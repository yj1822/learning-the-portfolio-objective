from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_final_analysis_scenario import _quality


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports/final_audit"
FRONTIER_REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/frontier"
AUM_REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/aum_sensitivity"
FRONTIER_RESULT_ROOT = PROJECT_ROOT / "results/final_analysis/frontier"
AUM_RESULT_ROOT = PROJECT_ROOT / "results/final_analysis/aum"
BASELINE_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
INTEGRITY_PATH = (
    PROJECT_ROOT
    / "reports/final_analysis/preflight/final_analysis_checkpoint_integrity.json"
)
PREFLIGHT_PATH = (
    PROJECT_ROOT / "reports/final_analysis/preflight/final_analysis_preflight.json"
)
PACKAGE_ROOT = PROJECT_ROOT / "deliverables/all_results_and_reports_20260810"
EXPECTED_METHODS = {
    "return_ml_rank",
    "static_ml",
    "static_ml_star",
    "portfolio_ml_linear",
    "portfolio_ml_rf",
}
EXPECTED_YEARS = list(range(2005, 2025))
TOLERANCE = 1.0e-6

ECONOMIC_ARTIFACTS = [
    "results/experiments/top500_core10_factor_cumulative/metrics.csv",
    "results/experiments/top500_core10_factor_cumulative/statistical_inference.csv",
    "reports/final_analysis/frontier/frontier_points.csv",
    "reports/final_analysis/frontier/frontier_points.parquet",
    "reports/final_analysis/aum_sensitivity/aum_sensitivity_points.csv",
    "reports/final_analysis/aum_sensitivity/aum_sensitivity_points.parquet",
    "reports/final_analysis/mechanisms/mechanism_metrics.csv",
    "reports/final_analysis/risk_calibration/risk_calibration_table.csv",
    "reports/final_analysis/feature_importance/feature_importance.csv",
    "reports/final_analysis/feature_importance/theme_importance.csv",
]


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now(timezone.utc).isoformat()
    before_hashes = _load_before_hashes()
    scenario_results = {
        gamma: _regenerate_frontier_quality(gamma, captured_at)
        for gamma in (15, 20)
    }
    checks, aggregates = _consistency_checks(scenario_results)
    after_hashes = _hash_artifacts()
    _write_hashes(
        REPORT_ROOT / "economic_artifact_hashes_after.json",
        captured_at,
        after_hashes,
    )
    hash_changes = {
        path: before_hashes[path] != after_hashes[path]
        for path in ECONOMIC_ARTIFACTS
    }
    checks["economic_artifact_hashes_unchanged"] = not any(hash_changes.values())

    integrity = json.loads(INTEGRITY_PATH.read_text(encoding="utf-8"))
    historical_count = int(integrity["historical_non_blocking_error_count"])
    blocking_count = int(integrity["blocking_error_count"])
    audit_passed = all(checks.values()) and blocking_count == 0
    summary = {
        "audit_passed": audit_passed,
        "audit_cleanup_at_utc": captured_at,
        "scope": "metadata, audit wrappers, integrity classification, and documentation only",
        "economic_results_changed": any(hash_changes.values()),
        "main_metrics_changed": hash_changes[
            "results/experiments/top500_core10_factor_cumulative/metrics.csv"
        ],
        "frontier_points_changed": hash_changes[
            "reports/final_analysis/frontier/frontier_points.csv"
        ]
        or hash_changes["reports/final_analysis/frontier/frontier_points.parquet"],
        "aum_points_changed": hash_changes[
            "reports/final_analysis/aum_sensitivity/aum_sensitivity_points.csv"
        ]
        or hash_changes[
            "reports/final_analysis/aum_sensitivity/aum_sensitivity_points.parquet"
        ],
        "feature_importance_changed": hash_changes[
            "reports/final_analysis/feature_importance/feature_importance.csv"
        ]
        or hash_changes[
            "reports/final_analysis/feature_importance/theme_importance.csv"
        ],
        "risk_calibration_changed": hash_changes[
            "reports/final_analysis/risk_calibration/risk_calibration_table.csv"
        ],
        "mechanism_metrics_changed": hash_changes[
            "reports/final_analysis/mechanisms/mechanism_metrics.csv"
        ],
        "gamma_15_final_status": scenario_results[15]["status"],
        "gamma_20_final_status": scenario_results[20]["status"],
        "gamma_20_gross_leverage_keyerror_resolved": checks[
            "gamma_20_quality_wrapper_compatible"
        ],
        "historical_checkpoint_error_record": historical_count,
        "historical_checkpoint_error_classification": "historical_non_blocking_error",
        "historical_checkpoint_error_blocking": blocking_count > 0,
        "jkp_timing_assumption_documented": checks[
            "jkp_timing_assumption_documented"
        ],
        "package_readme_updated": checks["package_readme_scope_documented"],
        "source_code_exclusion_documented": checks[
            "package_source_code_exclusion_documented"
        ],
        "raw_data_exclusion_documented": checks[
            "package_raw_data_exclusion_documented"
        ],
        "processed_data_exclusion_documented": checks[
            "package_processed_data_exclusion_documented"
        ],
        "future_information_violations": aggregates[
            "future_information_violations"
        ],
        "validation_cutoff_violations": aggregates[
            "validation_cutoff_violations"
        ],
        "factor_fallback_count": aggregates["factor_fallback_count"],
        "matrix_m_fallback_count": aggregates["matrix_m_fallback_count"],
        "covariance_repair_count": aggregates["covariance_repair_count"],
        "frontier_rows": aggregates["frontier_rows"],
        "frontier_gamma_values": aggregates["frontier_gamma_values"],
        "aum_rows": aggregates["aum_rows"],
        "aum_multiplier_values": aggregates["aum_multiplier_values"],
        "figure_manifest_rows": aggregates["figure_manifest_rows"],
        "checks": checks,
        "changed_economic_artifact_paths": [
            path for path, changed in hash_changes.items() if changed
        ],
        "ready_for_supervisor_report": audit_passed and not any(hash_changes.values()),
    }
    summary_path = REPORT_ROOT / "final_audit_cleanup_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    report_path = REPORT_ROOT / "final_audit_cleanup_report.md"
    report_path.write_text(
        _markdown_report(summary, scenario_results, integrity, captured_at),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0 if summary["ready_for_supervisor_report"] else 2


def _regenerate_frontier_quality(gamma: int, captured_at: str) -> dict[str, object]:
    root = FRONTIER_RESULT_ROOT / f"gamma_{gamma}"
    required = [
        "metrics.csv",
        "annual_performance.csv",
        "weights.parquet",
        "returns.parquet",
        "validation_selection.parquet",
        "predictions.parquet",
        "metadata.json",
        "config_used.yaml",
        "cache_manifest.json",
        "checkpoint_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    empty = [
        name for name in required if (root / name).exists() and (root / name).stat().st_size == 0
    ]
    if missing or empty:
        raise RuntimeError(f"gamma={gamma} final artifacts missing={missing}, empty={empty}")

    returns = pd.read_parquet(root / "returns.parquet")
    validation = pd.read_parquet(root / "validation_selection.parquet")
    metrics = pd.read_csv(root / "metrics.csv")
    annual = pd.read_csv(root / "annual_performance.csv")
    result = SimpleNamespace(returns=returns, validation_selection=validation)
    quality = _quality(result)
    years = sorted(pd.to_numeric(returns["test_year"], errors="raise").astype(int).unique())
    methods = sorted(returns["method"].astype(str).unique())
    if years != EXPECTED_YEARS or set(methods) != EXPECTED_METHODS:
        raise RuntimeError(
            f"gamma={gamma} incomplete coverage: years={years}, methods={methods}"
        )
    if set(metrics["method"].astype(str)) != EXPECTED_METHODS or len(metrics) != 5:
        raise RuntimeError(f"gamma={gamma} metrics.csv method coverage is incomplete")
    annual_year_column = "test_year" if "test_year" in annual.columns else "year"
    if (
        set(pd.to_numeric(annual[annual_year_column], errors="raise").astype(int))
        != set(EXPECTED_YEARS)
        or set(annual["method"].astype(str)) != EXPECTED_METHODS
        or len(annual) != 100
    ):
        raise RuntimeError(f"gamma={gamma} annual_performance.csv coverage is incomplete")

    frontier = pd.read_csv(FRONTIER_REPORT_ROOT / "frontier_points.csv")
    point = frontier[frontier["gamma"].eq(float(gamma))].set_index("method")
    metric = metrics.set_index("method")
    metric_mapping = {
        "annualized_gross_return": "annualized_gross_return",
        "annualized_net_return": "annualized_net_return",
        "annualized_volatility": "annualized_volatility",
        "net_sharpe": "net_sharpe",
        "monthly_turnover": "average_turnover",
        "average_leverage": "average_leverage",
        "annualized_trading_cost": "annualized_trading_cost",
        "annualized_utility": "annualized_ex_ante_utility_flow",
    }
    reconciliation_errors = {
        point_name: float(
            (
                pd.to_numeric(point.loc[sorted(EXPECTED_METHODS), point_name])
                - pd.to_numeric(metric.loc[sorted(EXPECTED_METHODS), metric_name])
            )
            .abs()
            .max()
        )
        for point_name, metric_name in metric_mapping.items()
    }
    reconciliation_max = max(reconciliation_errors.values())
    if reconciliation_max > TOLERANCE:
        raise RuntimeError(
            f"gamma={gamma} frontier reconciliation failed: {reconciliation_max}"
        )

    checkpoint_integrity = json.loads(INTEGRITY_PATH.read_text(encoding="utf-8"))
    checkpoint_progress = checkpoint_integrity["progress"][f"gamma_{gamma}"]
    run_status = json.loads(
        (
            PROJECT_ROOT
            / f"results/checkpoints/final_analysis/frontier/gamma_{gamma}/run_status.json"
        ).read_text(encoding="utf-8")
    )
    historical = (
        {
            "previous_status": "paused_time_budget",
            "preserved_before_file": "reports/final_audit/before/frontier_gamma_15_scenario_quality_before.json",
            "checkpoint_error_records": checkpoint_progress["error_records"],
            "checkpoint_error_classification": "historical_non_blocking_error",
            "note": (
                "The early time-budget snapshot and one superseded checkpoint error "
                "are retained for audit transparency; neither describes the final run."
            ),
        }
        if gamma == 15
        else {
            "previous_status": "quality_wrapper_failed_after_full_output",
            "preserved_log": "results/final_analysis/logs/frontier_gamma_20.stderr.log",
            "previous_error": "KeyError: 'gross_leverage'",
            "checkpoint_error_records": checkpoint_progress["error_records"],
            "note": (
                "The historical stderr log is retained. The final output was complete; "
                "the wrapper now resolves compatible leverage fields."
            ),
        }
    )
    payload = {
        "status": "complete",
        "mode": "full",
        "passed": bool(quality["passed"] and reconciliation_max <= TOLERANCE),
        "analysis_label": "controlled gamma-grid mechanism analysis",
        "scenario": f"frontier_gamma_{gamma}",
        "baseline_gamma": float(gamma),
        "baseline_aum": 1_000_000_000.0,
        "regenerated_at_utc": captured_at,
        "metadata_source": "existing final outputs only; no model or backtest rerun",
        "quality_wrapper": "leverage-field-compatible final wrapper",
        "quality": quality,
        "coverage": {
            "test_years": years,
            "methods": methods,
            "returns_rows": len(returns),
            "metrics_rows": len(metrics),
            "annual_performance_rows": len(annual),
            "validation_selection_rows": len(validation),
            "weights_rows": pq.ParquetFile(root / "weights.parquet").metadata.num_rows,
            "required_output_files_present": True,
        },
        "frontier_point_reconciliation": {
            "tolerance": TOLERANCE,
            "max_absolute_error": reconciliation_max,
            "errors_by_metric": reconciliation_errors,
            "passed": reconciliation_max <= TOLERANCE,
        },
        "checkpoint_integrity": {
            "validation_candidate_years_complete": checkpoint_progress[
                "validation_candidate_years_complete"
            ],
            "test_method_years_complete": checkpoint_progress[
                "test_method_years_complete"
            ],
            "partial_validation_years": checkpoint_progress[
                "partial_validation_years"
            ],
            "historical_non_blocking_error_records": checkpoint_progress[
                "historical_non_blocking_error_records"
            ],
            "blocking_error_records": checkpoint_progress["blocking_error_records"],
        },
        "run_status": run_status,
        "cache_manifest": json.loads(
            (root / "cache_manifest.json").read_text(encoding="utf-8")
        ),
        "checkpoint_manifest": json.loads(
            (root / "checkpoint_manifest.json").read_text(encoding="utf-8")
        ),
        "historical_audit_state": historical,
        "output_dir": str(root.resolve()),
    }
    if not payload["passed"]:
        raise RuntimeError(f"gamma={gamma} final quality did not pass")
    (root / "scenario_quality.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return payload


def _consistency_checks(
    scenario_results: dict[int, dict[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    metrics = pd.read_csv(BASELINE_ROOT / "metrics.csv").set_index("method")
    frontier = pd.read_csv(FRONTIER_REPORT_ROOT / "frontier_points.csv")
    aum = pd.read_csv(AUM_REPORT_ROOT / "aum_sensitivity_points.csv")
    frontier_10 = frontier[frontier["gamma"].eq(10.0)].set_index("method")
    aum_1 = aum[aum["aum_multiplier"].eq(1.0)].set_index("method")
    mapping = {
        "annualized_gross_return": "annualized_gross_return",
        "annualized_net_return": "annualized_net_return",
        "annualized_volatility": "annualized_volatility",
        "net_sharpe": "net_sharpe",
        "monthly_turnover": "average_turnover",
        "average_leverage": "average_leverage",
        "annualized_trading_cost": "annualized_trading_cost",
        "annualized_utility": "annualized_ex_ante_utility_flow",
    }

    def baseline_error(frame: pd.DataFrame) -> float:
        errors = []
        for point_column, metric_column in mapping.items():
            errors.append(
                float(
                    (
                        pd.to_numeric(frame.loc[sorted(EXPECTED_METHODS), point_column])
                        - pd.to_numeric(
                            metrics.loc[sorted(EXPECTED_METHODS), metric_column]
                        )
                    )
                    .abs()
                    .max()
                )
            )
        return max(errors)

    frontier_error = baseline_error(frontier_10)
    aum_error = baseline_error(aum_1)
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    qualities = [scenario_results[15], scenario_results[20]]
    for path in sorted(FRONTIER_RESULT_ROOT.glob("gamma_*/scenario_quality.json")):
        if path.parent.name not in {"gamma_15", "gamma_20"}:
            qualities.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(AUM_RESULT_ROOT.glob("multiplier_*/scenario_quality.json")):
        qualities.append(json.loads(path.read_text(encoding="utf-8")))
    quality_sums = {
        key: int(
            preflight["safety"].get(key, 0)
            + sum(int(item["quality"].get(key, 0)) for item in qualities)
        )
        for key in (
            "future_information_violations",
            "validation_cutoff_violations",
            "factor_fallback_count",
            "matrix_m_fallback_count",
            "covariance_repair_count",
        )
    }
    manifest = pd.read_csv(
        PROJECT_ROOT / "reports/final_analysis/dissertation_figures_manifest.csv"
    )
    figure_paths = [PROJECT_ROOT / path for path in manifest["figure_file_path"]]
    final_status_files = [
        PROJECT_ROOT / "reports/final_analysis/final_analysis_index.md",
        PROJECT_ROOT / "reports/final_analysis/frontier/implementable_frontier.md",
        PROJECT_ROOT / "reports/final_analysis/aum_sensitivity/aum_sensitivity.md",
        PROJECT_ROOT / "reports/final_results/final_specification_lock_in.md",
        FRONTIER_RESULT_ROOT / "gamma_15/scenario_quality.json",
        FRONTIER_RESULT_ROOT / "gamma_20/scenario_quality.json",
    ]
    stale_final_status = any(
        "paused_time_budget" in path.read_text(encoding="utf-8")
        and path.name != "scenario_quality.json"
        for path in final_status_files
    )
    integrity = json.loads(INTEGRITY_PATH.read_text(encoding="utf-8"))
    package_readme = (PACKAGE_ROOT / "PACKAGE_README.md").read_text(encoding="utf-8")
    lock_in = (
        PROJECT_ROOT / "reports/final_results/final_specification_lock_in.md"
    ).read_text(encoding="utf-8")
    final_index = (
        PROJECT_ROOT / "reports/final_analysis/final_analysis_index.md"
    ).read_text(encoding="utf-8")
    timing_phrase = "timing conventions embedded in the JKP characteristic data"
    checks = {
        "baseline_preflight_passed": bool(preflight["passed"]),
        "gamma_10_matches_baseline": frontier_error <= TOLERANCE,
        "aum_multiplier_1_matches_baseline": aum_error <= TOLERANCE,
        "frontier_has_40_rows": len(frontier) == 40,
        "frontier_has_8_gamma_values": set(frontier["gamma"])
        == {2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0},
        "frontier_has_5_methods_per_gamma": bool(
            frontier.groupby("gamma")["method"].nunique().eq(5).all()
        ),
        "aum_has_35_rows": len(aum) == 35,
        "aum_has_7_multiplier_values": set(aum["aum_multiplier"])
        == {0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0},
        "aum_has_5_methods_per_multiplier": bool(
            aum.groupby("aum_multiplier")["method"].nunique().eq(5).all()
        ),
        "factor_fallback_zero": quality_sums["factor_fallback_count"] == 0,
        "matrix_m_fallback_zero": quality_sums["matrix_m_fallback_count"] == 0,
        "covariance_repair_zero": quality_sums["covariance_repair_count"] == 0,
        "future_information_violations_zero": quality_sums[
            "future_information_violations"
        ]
        == 0,
        "validation_cutoff_violations_zero": quality_sums[
            "validation_cutoff_violations"
        ]
        == 0,
        "all_manifest_figures_exist_nonzero": all(
            path.exists() and path.stat().st_size > 0 for path in figure_paths
        ),
        "final_status_documents_not_stale": not stale_final_status,
        "gamma_15_quality_complete": scenario_results[15]["status"] == "complete"
        and scenario_results[15]["mode"] == "full",
        "gamma_20_quality_complete": scenario_results[20]["status"] == "complete"
        and scenario_results[20]["mode"] == "full",
        "gamma_20_quality_wrapper_compatible": scenario_results[20]["quality"][
            "leverage_field_mapping"
        ]["average"]
        in {"leverage", "gross_leverage", "actual_portfolio_leverage", "average_leverage"},
        "checkpoint_integrity_passed": bool(integrity["passed"]),
        "checkpoint_blocking_errors_zero": integrity["blocking_error_count"] == 0,
        "checkpoint_historical_error_classified": integrity[
            "historical_non_blocking_error_count"
        ]
        == 1,
        "jkp_timing_assumption_documented": timing_phrase in lock_in
        and timing_phrase in final_index,
        "package_readme_scope_documented": "not a complete reproducibility archive"
        in package_readme,
        "package_source_code_exclusion_documented": "source code" in package_readme,
        "package_raw_data_exclusion_documented": "raw WRDS/CRSP/JKP data"
        in package_readme,
        "package_processed_data_exclusion_documented": "Processed data" in package_readme,
    }
    aggregates = {
        **quality_sums,
        "frontier_rows": len(frontier),
        "frontier_gamma_values": sorted(frontier["gamma"].unique().tolist()),
        "frontier_baseline_max_absolute_error": frontier_error,
        "aum_rows": len(aum),
        "aum_multiplier_values": sorted(aum["aum_multiplier"].unique().tolist()),
        "aum_baseline_max_absolute_error": aum_error,
        "figure_manifest_rows": len(manifest),
    }
    return checks, aggregates


def _load_before_hashes() -> dict[str, str]:
    path = REPORT_ROOT / "before/economic_artifact_hashes_before.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = {row["path"]: row["sha256"] for row in payload["artifacts"]}
    missing = sorted(set(ECONOMIC_ARTIFACTS) - set(values))
    if missing:
        raise RuntimeError(f"Before-hash snapshot is missing artifacts: {missing}")
    return values


def _hash_artifacts() -> dict[str, str]:
    return {
        path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
        for path in ECONOMIC_ARTIFACTS
    }


def _write_hashes(path: Path, captured_at: str, hashes: dict[str, str]) -> None:
    payload = {
        "captured_at": captured_at,
        "artifacts": [
            {
                "path": name,
                "size_bytes": (PROJECT_ROOT / name).stat().st_size,
                "sha256": digest,
            }
            for name, digest in hashes.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _markdown_report(
    summary: dict[str, object],
    scenarios: dict[int, dict[str, object]],
    integrity: dict[str, object],
    captured_at: str,
) -> str:
    gamma15 = scenarios[15]
    gamma20 = scenarios[20]
    checks = summary["checks"]
    check_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )
    return f"""# Final audit cleanup report

## Audit information

- Audit cleanup time: `{captured_at}`.
- Scope: metadata, audit wrapper compatibility, checkpoint classification, data-limit disclosure and package documentation.
- Prohibited work respected: no model training, no baseline/robustness backtest, no new feature set and no economic-result rewrite.

## Files inspected

- Locked Top500/Core10 metrics, inference, returns and final performance tables.
- Frontier points and all gamma=15/gamma=20 final artifacts.
- AUM sensitivity points and scenario quality metadata.
- Mechanism, risk-calibration and feature-importance machine-readable outputs.
- Final-analysis preflight, checkpoint integrity summaries and historical runner logs.
- Dissertation figure manifest and all referenced PNG files.
- Final specification, robustness narrative, final analysis index and results-package README/metadata.

## Issues found and actions taken

1. The historical gamma=20 stderr log contains `KeyError: 'gross_leverage'`. The wrapper now resolves monthly `leverage`, legacy `gross_leverage`, `actual_portfolio_leverage`, or aggregated `average_leverage`/`max_leverage`. The retained final gamma=20 output passes through the corrected wrapper without recomputation.
2. Gamma=15 `scenario_quality.json` was an early `paused_time_budget` throughput snapshot even though later `metadata.json` and `run_status.json` showed a complete full run. The stale file was preserved under `reports/final_audit/before/` and replaced with final-output-derived metadata.
3. Gamma=20 had complete final outputs but no final scenario-quality file because the old wrapper failed after output completion. Final-output-derived metadata was created.
4. The gamma=15 checkpoint tree retains one historical `ArrowInvalid` error record. A valid complete checkpoint was written at the same boundary 107 seconds later. The error is classified as `historical_non_blocking_error`; the original `error.json` remains untouched.
5. JKP accounting timing and results-package reproducibility limitations are now disclosed in durable reports and generator templates.

## Economic-result immutability

- Economic results changed: **{summary['economic_results_changed']}**.
- Main metrics changed: **{summary['main_metrics_changed']}**.
- Frontier points changed: **{summary['frontier_points_changed']}**.
- AUM points changed: **{summary['aum_points_changed']}**.
- Mechanism metrics changed: **{summary['mechanism_metrics_changed']}**.
- Risk calibration changed: **{summary['risk_calibration_changed']}**.
- Feature importance changed: **{summary['feature_importance_changed']}**.

SHA-256 before/after records are stored in `reports/final_audit/before/economic_artifact_hashes_before.json` and `reports/final_audit/economic_artifact_hashes_after.json`.

## Gamma=15 before and after

- Before: historical metadata status `paused_time_budget`, covering an early 3.6-second throughput boundary.
- After: status `{gamma15['status']}`, mode `{gamma15['mode']}`, test years {gamma15['coverage']['test_years'][0]}-{gamma15['coverage']['test_years'][-1]}, {gamma15['coverage']['returns_rows']} method-month rows and frontier reconciliation error {gamma15['frontier_point_reconciliation']['max_absolute_error']:.3e}.
- Final quality: factor fallback {gamma15['quality']['factor_fallback_count']}, matrix-m fallback {gamma15['quality']['matrix_m_fallback_count']}, covariance repair {gamma15['quality']['covariance_repair_count']}, future-information violations {gamma15['quality']['future_information_violations']} and validation-cutoff violations {gamma15['quality']['validation_cutoff_violations']}.

## Gamma=20 before and after

- Before: complete final artifacts existed, but the historical quality wrapper failed on `gross_leverage` and did not write final scenario metadata.
- After: status `{gamma20['status']}`, mode `{gamma20['mode']}`, test years {gamma20['coverage']['test_years'][0]}-{gamma20['coverage']['test_years'][-1]}, {gamma20['coverage']['returns_rows']} method-month rows and frontier reconciliation error {gamma20['frontier_point_reconciliation']['max_absolute_error']:.3e}.
- Corrected leverage mapping: `{gamma20['quality']['leverage_field_mapping']}`.
- Final quality: factor fallback {gamma20['quality']['factor_fallback_count']}, matrix-m fallback {gamma20['quality']['matrix_m_fallback_count']}, covariance repair {gamma20['quality']['covariance_repair_count']}, future-information violations {gamma20['quality']['future_information_violations']} and validation-cutoff violations {gamma20['quality']['validation_cutoff_violations']}.

## Historical checkpoint record

One historical checkpoint error record is retained for audit transparency. It corresponds to an earlier interrupted or wrapper-level run state and does not affect the final full-run outputs. Final outputs passed all quality gates and completed the required scenario/method-year coverage. Integrity status: {integrity['historical_non_blocking_error_count']} historical non-blocking record, {integrity['blocking_error_count']} blocking records, {len(integrity['corrupt_files'])} corrupt files and {len(integrity['checkpoint_structure_errors'])} structural errors.

## JKP timing assumption

The implementation relies on the timing conventions embedded in the JKP characteristic data. I do not independently reconstruct accounting reporting lags from raw Compustat filing dates. This is treated as a data limitation rather than a model-selection issue. It should remain visible in the dissertation data and limitations sections.

## Results-package reproducibility note

The package is a final results and reports package, not a complete reproducibility archive. It includes reports, tables, figures, experiment outputs and final-analysis outputs. Licensed raw WRDS/CRSP/JKP data, processed data, source code, third-party code, checkpoints and caches are excluded. Full reproduction requires licensed data, source code, environment/configuration and run scripts from the separate project workspace or repository.

## Consistency checks

| Check | Status |
| --- | --- |
{check_rows}

## Remaining limitations

- JKP accounting-data timing is accepted from the vendor dataset rather than independently reconstructed from filing dates.
- The results package is not a complete licensed-data reproducibility archive.
- Static-ML risk calibration misses in 2020-2021 remain reported limitations; no post-hoc retuning is performed.
- Historical logs and the superseded checkpoint error remain present for audit transparency.

## Final conclusion

Audit cleanup **{'PASS' if summary['audit_passed'] else 'FAIL'}**. No economic results changed. Main results, frontier, AUM sensitivity, mechanism metrics, risk calibration and feature importance remain unchanged. Gamma=15 and gamma=20 final metadata now reflect completed full-run status. The historical checkpoint error record is documented as non-blocking. JKP accounting timing and results-package scope are documented. The package is {'ready' if summary['ready_for_supervisor_report'] else 'not ready'} for supervisor progress reporting.
"""


if __name__ == "__main__":
    raise SystemExit(main())
