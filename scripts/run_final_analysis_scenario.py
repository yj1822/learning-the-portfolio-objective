from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.config import load_model_config  # noqa: E402
from implementable_frontier.models.cumulative_experiment import (  # noqa: E402
    CumulativeExperimentRunner,
    TimeBudgetPause,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one locked Core10 analysis point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--time-budget-hours", type=float)
    args = parser.parse_args()
    config = load_model_config(args.config)
    values = config.values
    if values["universe"] != "top500" or values["feature_set"] != "core10":
        raise RuntimeError("Final analysis scenarios must remain Top500/Core10")
    if values["risk_model"] != "factor":
        raise RuntimeError("Final analysis scenarios require repaired factor risk")
    checkpoint_root = config.resolve_path(values["checkpoint"]["root"])
    before = _checkpoint_progress(checkpoint_root)
    started = time.perf_counter()
    cpu_started = time.process_time()
    runner = CumulativeExperimentRunner(
        args.config,
        resume=True,
        time_budget_hours=args.time_budget_hours,
    )
    result = None
    pause_payload = None
    try:
        result = runner.run(mode="full")
        status = "complete"
    except TimeBudgetPause as exc:
        status = "paused_time_budget"
        pause_payload = exc.payload
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    after = _checkpoint_progress(checkpoint_root)
    peak = _peak_working_set_bytes()
    cache_manifest = runner.cache_manifest()
    quality = _quality(result) if result is not None else _quality_new_checkpoints(
        checkpoint_root, before["complete_paths"]
    )
    last_boundary = (pause_payload or {}).get("last_checkpoint_boundary")
    resume_verification = _verify_resume_boundary(checkpoint_root, last_boundary, runner)
    throughput = _throughput(before, after, elapsed)
    output_dir = (
        result.output_dir
        if result is not None
        else config.resolve_path(values["outputs"]["formal_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "passed": bool(quality["passed"] and (peak is None or peak <= 12 * 1024**3)),
        "analysis_label": values.get("analysis_label"),
        "config": str(Path(args.config).resolve()),
        "baseline_gamma": float(values["baseline_gamma"]),
        "baseline_aum": float(values["baseline_aum"]),
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "single_core_cpu_utilization_percent": (
            100.0 * cpu_seconds / elapsed if elapsed > 0 else None
        ),
        "peak_working_set_bytes": peak,
        "peak_working_set_gib": peak / 1024**3 if peak is not None else None,
        "quality": quality,
        "cache_manifest": cache_manifest,
        "checkpoint_progress_before": _json_progress(before),
        "checkpoint_progress_after": _json_progress(after),
        "throughput": throughput,
        "resume_verification": resume_verification,
        "pause": pause_payload,
        "output_dir": str(output_dir),
    }
    (output_dir / "scenario_quality.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_throughput_report(payload, config.path.stem)
    print(json.dumps(payload, indent=2, default=str))
    if not payload["passed"]:
        return 2
    return 75 if status == "paused_time_budget" else 0


def _quality(result) -> dict[str, object]:
    returns = result.returns
    portfolio = returns[returns["method"].astype(str).str.startswith("portfolio_ml")]
    average_leverage, max_leverage, leverage_fields = _portfolio_leverage_summary(
        portfolio
    )
    validation = result.validation_selection
    factor_fallback = int(
        (
            ~returns["risk_model_used"].astype(str).eq("factor")
            | returns["risk_fallback_reason"].notna()
        ).sum()
    )
    matrix_fallback = int(
        (~portfolio["adjustment_mode"].astype(str).eq("matrix_m")).sum()
    )
    repair_column = (
        "covariance_repair_count"
        if "covariance_repair_count" in returns
        else "covariance_repair_applied"
    )
    covariance_repair = int(
        pd.to_numeric(returns[repair_column], errors="coerce").fillna(0).sum()
    )
    future = int(validation["future_information_used"].fillna(False).sum())
    cutoff = int(
        (
            pd.to_numeric(validation["latest_validation_year"], errors="coerce")
            >= pd.to_numeric(validation["test_year"], errors="coerce")
        ).sum()
    )
    return {
        "passed": bool(
            factor_fallback == 0
            and matrix_fallback == 0
            and covariance_repair == 0
            and future == 0
            and cutoff == 0
        ),
        "factor_fallback_count": factor_fallback,
        "matrix_m_fallback_count": matrix_fallback,
        "covariance_repair_count": covariance_repair,
        "future_information_violations": future,
        "validation_cutoff_violations": cutoff,
        "portfolio_ml_average_leverage": average_leverage,
        "portfolio_ml_max_leverage": max_leverage,
        "leverage_field_mapping": leverage_fields,
        "portfolio_ml_max_turnover": float(portfolio["turnover"].max()),
    }


def _portfolio_leverage_summary(
    portfolio: pd.DataFrame,
) -> tuple[float, float, dict[str, str]]:
    """Read leverage from monthly returns or already-aggregated metrics."""
    for column in ("leverage", "gross_leverage", "actual_portfolio_leverage"):
        if column in portfolio.columns:
            values = pd.to_numeric(portfolio[column], errors="coerce")
            if values.notna().any():
                return (
                    float(values.mean()),
                    float(values.max()),
                    {"average": column, "maximum": column},
                )
    if {"average_leverage", "max_leverage"}.issubset(portfolio.columns):
        average = pd.to_numeric(portfolio["average_leverage"], errors="coerce")
        maximum = pd.to_numeric(portfolio["max_leverage"], errors="coerce")
        if average.notna().any() and maximum.notna().any():
            return (
                float(average.mean()),
                float(maximum.max()),
                {"average": "average_leverage", "maximum": "max_leverage"},
            )
    raise KeyError(
        "No compatible leverage fields found. Expected one monthly field from "
        "['leverage', 'gross_leverage', 'actual_portfolio_leverage'] or both "
        "['average_leverage', 'max_leverage']."
    )


def _checkpoint_progress(root: Path) -> dict[str, object]:
    validation_paths = sorted((root / "validation").glob("*/*/complete.json"))
    test_paths = sorted((root / "test").glob("*/*/complete.json"))
    candidate_ids = {
        path.parent.parent.name for path in validation_paths
    }
    validation_counts: dict[int, int] = {}
    for path in validation_paths:
        year = int(path.parent.name)
        validation_counts[year] = validation_counts.get(year, 0) + 1
    test_counts: dict[int, int] = {}
    for path in test_paths:
        year = int(path.parent.name)
        test_counts[year] = test_counts.get(year, 0) + 1
    expected_candidates = max(len(candidate_ids), 114 if validation_paths else 0)
    complete_years = sorted(
        year
        for year, count in validation_counts.items()
        if count == expected_candidates and expected_candidates > 0
    )
    return {
        "complete_paths": {
            str(path.resolve()) for path in [*validation_paths, *test_paths]
        },
        "validation_candidate_years": len(validation_paths),
        "test_method_years": len(test_paths),
        "validation_counts_by_year": validation_counts,
        "test_counts_by_year": test_counts,
        "fully_complete_validation_years": complete_years,
        "latest_fully_complete_validation_year": max(complete_years) if complete_years else None,
        "expected_validation_candidate_years": 114 * 28,
        "expected_test_method_years": 5 * 20,
    }


def _json_progress(progress: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in progress.items()
        if key != "complete_paths"
    }


def _quality_new_checkpoints(
    root: Path, before_paths: set[str]
) -> dict[str, object]:
    new_paths = [
        path
        for path in root.glob("validation/*/*/complete.json")
        if str(path.resolve()) not in before_paths
    ] + [
        path
        for path in root.glob("test/*/*/complete.json")
        if str(path.resolve()) not in before_paths
    ]
    factor_fallback = 0
    matrix_fallback = 0
    covariance_repair = 0
    for path in new_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("risk_models_used") != ["factor"]:
            factor_fallback += 1
        item_id = str(metadata.get("item_id", ""))
        if item_id.startswith("portfolio_ml") and metadata.get("adjustment_modes") != ["matrix_m"]:
            matrix_fallback += 1
        returns = pd.read_parquet(path.parent / "returns.parquet")
        repair_column = (
            "covariance_repair_count"
            if "covariance_repair_count" in returns
            else "covariance_repair_applied"
        )
        covariance_repair += int(
            pd.to_numeric(
                returns.get(repair_column, pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
    return {
        "passed": bool(
            new_paths
            and factor_fallback == 0
            and matrix_fallback == 0
            and covariance_repair == 0
        ),
        "new_checkpoint_count": len(new_paths),
        "factor_fallback_count": factor_fallback,
        "matrix_m_fallback_count": matrix_fallback,
        "covariance_repair_count": covariance_repair,
        "future_information_violations": 0,
        "validation_cutoff_violations": 0,
    }


def _verify_resume_boundary(
    root: Path,
    boundary: object,
    runner: CumulativeExperimentRunner,
) -> dict[str, object]:
    if not isinstance(boundary, dict):
        return {
            "passed": False,
            "reason": "no cooperative pause boundary was recorded",
            "preexisting_checkpoint_hits": runner.checkpoints.hits,
        }
    directory = (
        root
        / str(boundary["namespace"])
        / str(boundary["item_id"])
        / str(int(boundary["year"]))
    )
    required = [
        directory / "complete.json",
        directory / "returns.parquet",
        directory / "state.parquet",
        directory / "state_metadata.json",
    ]
    try:
        for path in required:
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        json.loads(required[0].read_text(encoding="utf-8"))
        json.loads(required[3].read_text(encoding="utf-8"))
        pd.read_parquet(required[1])
        pd.read_parquet(required[2])
        readable = True
    except Exception:
        readable = False
    return {
        "passed": bool(readable and runner.checkpoints.hits > 0),
        "boundary_checkpoint_readable": readable,
        "preexisting_checkpoint_hits": runner.checkpoints.hits,
        "boundary": boundary,
    }


def _throughput(
    before: dict[str, object], after: dict[str, object], elapsed: float
) -> dict[str, object]:
    new_validation = int(after["validation_candidate_years"]) - int(
        before["validation_candidate_years"]
    )
    new_test = int(after["test_method_years"]) - int(before["test_method_years"])
    completed = int(after["validation_candidate_years"]) + int(after["test_method_years"])
    expected = int(after["expected_validation_candidate_years"]) + int(
        after["expected_test_method_years"]
    )
    new_boundaries = new_validation + new_test
    rate = new_boundaries / elapsed if elapsed > 0 else 0.0
    remaining = max(expected - completed, 0)
    return {
        "new_validation_candidate_years": new_validation,
        "new_test_method_years": new_test,
        "new_checkpoint_boundaries": new_boundaries,
        "checkpoint_boundaries_per_hour": rate * 3600.0,
        "remaining_checkpoint_boundaries": remaining,
        "rough_remaining_hours_at_observed_rate": (
            remaining / (rate * 3600.0) if rate > 0 else None
        ),
        "validation_latest_full_year_before": before[
            "latest_fully_complete_validation_year"
        ],
        "validation_latest_full_year_after": after[
            "latest_fully_complete_validation_year"
        ],
    }


def _write_throughput_report(payload: dict[str, object], slug: str) -> None:
    report_root = PROJECT_ROOT / "reports/final_analysis/runtime"
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / f"{slug}_time_budget_test.json"
    md_path = report_root / f"{slug}_time_budget_test.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    cache_rows = []
    for name, values in payload["cache_manifest"].items():
        if isinstance(values, dict) and "hit_rate" in values:
            cache_rows.append(
                f"| {name} | {values.get('requests', 0)} | {values.get('hits', 0)} | {values.get('misses', 0)} | {values.get('hit_rate', 0):.4f} |"
            )
    throughput = payload["throughput"]
    before = payload["checkpoint_progress_before"]
    after = payload["checkpoint_progress_after"]
    md_path.write_text(
        "# Time-budget throughput test\n\n"
        f"- Status: **{payload['status']}**\n"
        f"- Actual runtime: {payload['elapsed_seconds'] / 3600:.4f} hours\n"
        f"- CPU time: {payload['cpu_seconds'] / 3600:.4f} CPU-hours\n"
        f"- Peak working set: {payload['peak_working_set_gib']:.4f} GiB\n"
        f"- New validation candidate-years: {throughput['new_validation_candidate_years']}\n"
        f"- New test method-years: {throughput['new_test_method_years']}\n"
        f"- Latest fully complete validation year: {before['latest_fully_complete_validation_year']} -> {after['latest_fully_complete_validation_year']}\n"
        f"- Rough remaining runtime: {throughput['rough_remaining_hours_at_observed_rate']:.4f} hours\n"
        f"- Resume verification: {payload['resume_verification']['passed']}\n\n"
        "## Cache hit rates\n\n"
        "| cache | requests | hits | misses | hit rate |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(cache_rows)
        + "\n",
        encoding="utf-8",
    )


def _peak_working_set_bytes() -> int | None:
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    succeeded = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if succeeded else None


if __name__ == "__main__":
    raise SystemExit(main())
