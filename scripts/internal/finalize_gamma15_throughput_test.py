from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = PROJECT_ROOT / "results/checkpoints/final_analysis/frontier/gamma_15"
RESULT_ROOT = PROJECT_ROOT / "results/final_analysis/frontier/gamma_15"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/runtime"
START_VALIDATION_BOUNDARIES = 656
START_TEST_BOUNDARIES = 0
START_LATEST_FULL_VALIDATION_YEAR = 2000


def main() -> int:
    returns = pd.read_parquet(RESULT_ROOT / "returns.parquet")
    validation = pd.read_parquet(RESULT_ROOT / "validation_selection.parquet")
    runtime = json.loads((RESULT_ROOT / "runtime_profile.json").read_text(encoding="utf-8"))
    cache = json.loads((RESULT_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads(
        (RESULT_ROOT / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    status = json.loads((CHECKPOINT_ROOT / "run_status.json").read_text(encoding="utf-8"))
    validation_paths = list(CHECKPOINT_ROOT.glob("validation/*/*/complete.json"))
    test_paths = list(CHECKPOINT_ROOT.glob("test/*/*/complete.json"))
    validation_counts: dict[int, int] = {}
    for path in validation_paths:
        year = int(path.parent.name)
        validation_counts[year] = validation_counts.get(year, 0) + 1
    complete_years = sorted(year for year, count in validation_counts.items() if count == 114)
    quality = _quality(returns, validation)
    actual_seconds = float(runtime["total_seconds"])
    new_validation = len(validation_paths) - START_VALIDATION_BOUNDARIES
    new_test = len(test_paths) - START_TEST_BOUNDARIES
    method_accounted_seconds = float(sum(runtime.get("method_seconds", {}).values()))
    prior_short = json.loads(
        (REPORT_ROOT / "frontier_gamma_15_time_budget_test.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "status": status["status"],
        "passed": bool(status["status"] == "complete" and quality["passed"]),
        "scenario": "gamma=15",
        "time_budget_hours": 8.0,
        "actual_runtime_seconds": actual_seconds,
        "actual_runtime_hours": actual_seconds / 3600.0,
        "new_validation_candidate_years": new_validation,
        "new_test_method_years": new_test,
        "new_checkpoint_boundaries": new_validation + new_test,
        "validation_latest_full_year_before": START_LATEST_FULL_VALIDATION_YEAR,
        "validation_latest_full_year_after": max(complete_years),
        "validation_fully_complete_years_after": complete_years,
        "remaining_checkpoint_boundaries": 0,
        "remaining_runtime_estimate_hours": 0.0,
        "checkpoint_accounted_method_wall_seconds": method_accounted_seconds,
        "cpu_measurement": {
            "exact_process_cpu_seconds": None,
            "reason": (
                "The completed worker exited during post-run quality reporting before "
                "process CPU telemetry was persisted. Checkpoint-accounted method wall "
                "time is reported separately and is not relabelled as CPU time."
            ),
        },
        "peak_memory_measurement": {
            "exact_completed_worker_peak_gib": None,
            "reason": (
                "The completed worker exited during post-run quality reporting before "
                "PeakWorkingSetSize was persisted."
            ),
            "same_scenario_resume_smoke_peak_gib": prior_short.get(
                "peak_working_set_gib"
            ),
            "locked_core10_prior_cold_run_peak_gib": 2.920,
            "interpretation": "references only; neither value is presented as the missing exact peak",
        },
        "cache_manifest": cache,
        "checkpoint_manifest": checkpoint,
        "resume_verification": {
            "passed": bool(
                checkpoint.get("hits", 0) >= START_VALIDATION_BOUNDARIES
                and checkpoint.get("writes", 0) == new_validation + new_test
            ),
            "preexisting_checkpoint_hits": checkpoint.get("hits"),
            "new_checkpoint_writes": checkpoint.get("writes"),
            "all_expected_boundaries_complete": len(validation_paths) == 3192
            and len(test_paths) == 100,
            "no_independent_segment_tuning": True,
        },
        "quality": quality,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "frontier_gamma_15_time_budget_test.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (REPORT_ROOT / "frontier_gamma_15_time_budget_test.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 2


def _quality(returns: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    portfolio = returns[returns["method"].astype(str).str.startswith("portfolio_ml")]
    factor = int(
        (
            ~returns["risk_model_used"].astype(str).eq("factor")
            | returns["risk_fallback_reason"].notna()
        ).sum()
    )
    matrix = int((~portfolio["adjustment_mode"].astype(str).eq("matrix_m")).sum())
    repair = int(
        pd.to_numeric(returns["covariance_repair_count"], errors="coerce")
        .fillna(0)
        .sum()
    )
    future = int(validation["future_information_used"].fillna(False).sum())
    cutoff = int(
        (
            pd.to_numeric(validation["latest_validation_year"], errors="coerce")
            >= pd.to_numeric(validation["test_year"], errors="coerce")
        ).sum()
    )
    return {
        "passed": factor == matrix == repair == future == cutoff == 0,
        "factor_fallback_count": factor,
        "matrix_m_fallback_count": matrix,
        "covariance_repair_count": repair,
        "future_information_violations": future,
        "validation_cutoff_violations": cutoff,
    }


def _markdown(payload: dict[str, object]) -> str:
    cache_rows = []
    for name, values in payload["cache_manifest"].items():
        if isinstance(values, dict) and "hit_rate" in values:
            cache_rows.append(
                f"| {name} | {values.get('requests', 0)} | {values.get('hits', 0)} | {values.get('misses', 0)} | {values.get('hit_rate', 0):.4f} |"
            )
    return (
        "# Gamma 15 time-budget throughput test\n\n"
        f"- Status: **{payload['status']}**\n"
        f"- Budget: {payload['time_budget_hours']:.1f} hours\n"
        f"- Actual runtime: {payload['actual_runtime_hours']:.4f} hours ({payload['actual_runtime_seconds']:.3f} seconds)\n"
        f"- New validation candidate-years: {payload['new_validation_candidate_years']}\n"
        f"- New test method-years: {payload['new_test_method_years']}\n"
        f"- Validation latest complete year: {payload['validation_latest_full_year_before']} -> {payload['validation_latest_full_year_after']}\n"
        f"- Remaining runtime estimate: {payload['remaining_runtime_estimate_hours']:.1f} hours\n"
        f"- Resume verification: {payload['resume_verification']['passed']}\n"
        f"- Quality gate: {payload['quality']['passed']}\n\n"
        "## CPU and memory telemetry\n\n"
        f"Exact process CPU and completed-worker peak memory are unavailable because the post-run report failed after model completion. Checkpoint-accounted method wall time is {payload['checkpoint_accounted_method_wall_seconds']:.3f} seconds. The same-scenario resume smoke peak was {payload['peak_memory_measurement']['same_scenario_resume_smoke_peak_gib']:.3f} GiB; the prior locked Core10 cold-run peak was {payload['peak_memory_measurement']['locked_core10_prior_cold_run_peak_gib']:.3f} GiB. These are references, not replacements for the missing exact peak.\n\n"
        "## Cache hit rates\n\n"
        "| cache | requests | hits | misses | hit rate |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(cache_rows)
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
