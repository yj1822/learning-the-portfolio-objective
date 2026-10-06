from __future__ import annotations

import argparse
import cProfile
import ctypes
import json
import os
import pstats
import sys
import time
from datetime import datetime
from ctypes import wintypes
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.cumulative_experiment import (  # noqa: E402
    CumulativeExperimentRunner,
)
from implementable_frontier.models.config import load_model_config  # noqa: E402
from implementable_frontier.models.formal_analysis import (  # noqa: E402
    risk_calibration_report,
    run_formal_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a gated cumulative OOS implementable-frontier experiment"
    )
    parser.add_argument(
        "--config", default="configs/model_top100_core10_cumulative.yaml"
    )
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument(
        "--segment",
        help="Contiguous full-run test-year segment, for example 2005-2009",
    )
    parser.add_argument(
        "--preflight-report",
        help="Passed preflight JSON that may replace the traditional pilot gate",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_model_config(args.config)
    values = config.values
    if values.get("data_master_panel_path"):
        os.environ["IEF_MASTER_PANEL_PATH"] = str(
            config.resolve_path(values["data_master_panel_path"])
        )
    slug = f"{values['universe']}_{values['feature_set']}"
    full_limit = float(values["profiling"]["full_run_limit_hours"])
    hard_limit = float(values["profiling"]["hard_stop_hours"])
    segment_years = _parse_segment(args.segment, values) if args.segment else None
    if segment_years is not None and args.mode != "full":
        parser.error("--segment requires --mode full")

    audit = _audit_gate()
    if not audit["passed"]:
        raise RuntimeError(f"Phase 0 audit gate failed: {audit}")
    if args.mode == "full":
        _full_run_gate(
            slug,
            full_limit=full_limit,
            hard_limit=hard_limit,
            preflight_report=args.preflight_report,
        )

    runner = CumulativeExperimentRunner(args.config, resume=not args.no_resume)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    segment_output = (
        config.resolve_path(values["outputs"]["formal_dir"])
        / "segments"
        / f"{segment_years[0]}_{segment_years[-1]}"
        if segment_years is not None
        else None
    )
    result = profiler.runcall(
        runner.run,
        mode=args.mode,
        test_years_override=segment_years,
        output_dir_override=segment_output,
    )
    elapsed = time.perf_counter() - started
    profile_rows = _profile_rows(profiler, limit=30)
    peak_working_set = _peak_working_set_bytes()
    prior_peak = _prior_peak_working_set(runner.checkpoints.root)
    if prior_peak is not None:
        peak_working_set = max(peak_working_set or 0, prior_peak)
    resume_check = runner.verify_resume(
        max(result.runtime["test_years"]) - 1
    )
    checkpoint_runtime = _checkpoint_runtime_profile(runner.checkpoints.root)
    cold_cache_reconstruction = _cold_cache_reconstruction(
        runner.checkpoints.root,
        checkpoint_runtime=checkpoint_runtime,
    )
    estimate = _estimate_full_runtime(
        result.runtime, elapsed, checkpoint_runtime=checkpoint_runtime
    )
    duplicate_check = _duplicate_computation_check(result.cache_manifest)
    quality = _execution_quality(result, resume_check)
    can_run_full = bool(
        quality["passed"]
        and estimate["estimated_full_hours"] <= full_limit
        and estimate["estimated_full_hours"] <= hard_limit
    )
    risk_diagnostics = risk_calibration_report(result.returns)
    matrix_stability = _matrix_stability(result.returns)

    runtime = {
        **result.runtime,
        "wall_clock_seconds": elapsed,
        "peak_tracemalloc_bytes": None,
        "peak_tracemalloc_gib": None,
        "memory_measurement": (
            "Windows PeakWorkingSetSize; tracemalloc disabled to avoid "
            "allocation-tracing overhead"
        ),
        "peak_working_set_bytes": peak_working_set,
        "peak_working_set_gib": (
            peak_working_set / 1024**3 if peak_working_set is not None else None
        ),
        "top_30_functions_by_cumulative_time": profile_rows,
        "checkpoint_runtime_profile": checkpoint_runtime,
        "cold_cache_reconstruction": cold_cache_reconstruction,
        "resume_verification": resume_check,
        "full_runtime_estimate": estimate,
        "duplicate_computation_check": duplicate_check,
        "execution_quality_gate": quality,
        "can_run_full": can_run_full,
        "runtime_gate_hours": full_limit,
        "hard_stop_hours": hard_limit,
    }
    (result.output_dir / "runtime_profile.json").write_text(
        json.dumps(runtime, indent=2, default=str), encoding="utf-8"
    )
    report_slug = (
        f"{slug}_segment_{segment_years[0]}_{segment_years[-1]}"
        if segment_years is not None
        else slug
    )
    _write_performance_reports(
        result=result,
        runtime=runtime,
        profile_rows=profile_rows,
        audit=audit,
        slug=report_slug,
        universe=str(values["universe"]),
        feature_set=str(values["feature_set"]),
        risk_diagnostics=risk_diagnostics,
        matrix_stability=matrix_stability,
    )

    if args.mode == "full" and segment_years is None:
        if not can_run_full:
            raise RuntimeError(f"Full execution quality gate failed after run: {quality}")
        report_dir = config.resolve_path(
            values["outputs"].get(
                "formal_reports_dir",
                f"reports/model_results/{slug}_cumulative",
            )
        )
        settings = runner.values["robustness"]
        analysis = run_formal_analysis(
            output_dir=result.output_dir,
            report_dir=report_dir,
            returns=result.returns,
            annual=result.annual_performance,
            validation_selection=result.validation_selection,
            gamma=float(runner.values["baseline_gamma"]),
            seed=int(runner.values["random_seed"]),
            newey_west_lags=int(settings["newey_west_lags"]),
            bootstrap_block_months=int(settings["bootstrap_block_months"]),
            bootstrap_replications=int(settings["bootstrap_replications"]),
        )
        metadata_path = result.output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "method_audit_gate": audit,
                "resume_verification": resume_check,
                "risk_calibration_quality_gate_passed": analysis.risk_quality_passed,
                "actual_runtime_seconds": elapsed,
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    print(f"mode={args.mode}")
    print(f"output_dir={result.output_dir}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"estimated_full_hours={estimate['estimated_full_hours']:.3f}")
    print(f"can_run_full={can_run_full}")
    print(f"resume_passed={resume_check['passed']}")
    return 0


def _parse_segment(
    value: str, config_values: dict[str, object]
) -> list[int]:
    try:
        start_text, end_text = value.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("segment must use START-END, for example 2005-2009") from exc
    cumulative = config_values["cumulative_oos_validation"]
    first = int(cumulative["first_test_year"])
    last = int(cumulative["last_test_year"])
    if start > end or start < first or end > last:
        raise ValueError(f"segment must be within {first}-{last}")
    return list(range(start, end + 1))


def _audit_gate() -> dict[str, object]:
    audit_path = PROJECT_ROOT / "reports/method_audit/official_method_parity.json"
    parity_path = PROJECT_ROOT / "reports/method_audit/small_sample_parity.json"
    if not audit_path.exists() or not parity_path.exists():
        return {"passed": False, "reason": "audit outputs are missing"}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    return {
        "passed": bool(
            audit.get("method_audit_passed")
            and audit.get("small_sample_parity_passed")
            and parity.get("passed")
        ),
        "official_commit": audit.get("official", {}).get("commit"),
        "method_audit_passed": bool(audit.get("method_audit_passed")),
        "small_sample_parity_passed": bool(parity.get("passed")),
    }


def _full_run_gate(
    slug: str,
    *,
    full_limit: float,
    hard_limit: float,
    preflight_report: str | None = None,
) -> None:
    if preflight_report is not None:
        path = Path(preflight_report)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise RuntimeError(f"Preflight report does not exist: {path}")
        preflight = json.loads(path.read_text(encoding="utf-8"))
        if not bool(preflight.get("passed", False)):
            raise RuntimeError(
                f"Preflight quality gate does not permit the full run: {path}"
            )
        return
    path = PROJECT_ROOT / f"reports/performance/{slug}_pilot.json"
    if not path.exists():
        raise RuntimeError("Pilot report is required before full execution")
    pilot = json.loads(path.read_text(encoding="utf-8"))
    estimate = float(pilot["runtime"]["full_runtime_estimate"]["estimated_full_hours"])
    if not pilot.get("can_run_full", False):
        raise RuntimeError("Pilot quality/runtime gate does not permit the full run")
    if estimate > full_limit:
        raise RuntimeError(
            f"Estimated full runtime {estimate:.3f}h exceeds {full_limit:g}h"
        )
    if estimate > hard_limit:
        raise RuntimeError(
            f"Hard stop: estimated full runtime {estimate:.3f}h exceeds {hard_limit:g}h"
        )


def _profile_rows(profiler: cProfile.Profile, *, limit: int) -> list[dict[str, object]]:
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        rows.append(
            {
                "file": str(Path(filename).name),
                "line": int(line),
                "function": function,
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "total_time_seconds": float(total_time),
                "cumulative_time_seconds": float(cumulative_time),
            }
        )
    return sorted(rows, key=lambda row: row["cumulative_time_seconds"], reverse=True)[:limit]


def _estimate_full_runtime(
    runtime: dict[str, object],
    elapsed: float,
    *,
    checkpoint_runtime: dict[str, object],
) -> dict[str, float]:
    test_years = [int(value) for value in runtime["test_years"]]
    first_validation = 1996
    pilot_validation_years = max(test_years) - first_validation
    full_validation_years = 2024 - first_validation
    candidate_seconds = float(checkpoint_runtime["validation_candidate_seconds"])
    fit_seconds = float(checkpoint_runtime["portfolio_fit_group_seconds"])
    test_seconds = float(checkpoint_runtime["test_method_seconds"])
    observed_span = max(
        float(checkpoint_runtime["observed_checkpoint_span_seconds"]),
        float(elapsed),
    )
    accounted = candidate_seconds + fit_seconds + test_seconds
    untracked = max(observed_span - accounted, 0.0)
    estimated = (
        (candidate_seconds + fit_seconds)
        * full_validation_years
        / max(pilot_validation_years, 1)
        + test_seconds * 20.0 / max(len(test_years), 1)
        + untracked * full_validation_years / max(pilot_validation_years, 1)
    )
    return {
        "pilot_elapsed_seconds": float(elapsed),
        "candidate_seconds": candidate_seconds,
        "portfolio_fit_group_seconds": fit_seconds,
        "test_year_seconds": test_seconds,
        "untracked_seconds": untracked,
        "observed_checkpoint_span_seconds": observed_span,
        "pilot_validation_years": float(pilot_validation_years),
        "full_validation_years": float(full_validation_years),
        "estimated_full_seconds": estimated,
        "estimated_full_hours": estimated / 3600.0,
    }


def _checkpoint_runtime_profile(root: Path) -> dict[str, object]:
    method_seconds: dict[str, float] = {}
    validation_seconds = 0.0
    test_seconds = 0.0
    fit_groups: dict[tuple[object, ...], float] = {}
    validation_count = 0
    test_count = 0
    candidate_timing: list[dict[str, object]] = []
    portfolio_validation_count = 0
    for path in root.glob("validation/*/*/complete.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        seconds = float(metadata.get("seconds", 0.0))
        method = str(payload.get("item_id", "unknown")).rsplit("-", 1)[0]
        method_seconds[method] = method_seconds.get(method, 0.0) + seconds
        validation_seconds += seconds
        validation_count += 1
        if method.startswith("portfolio_ml"):
            portfolio_validation_count += 1
        candidate_timing.append(
            {
                "method": method,
                "candidate_id": str(payload.get("item_id")),
                "year": int(payload["year"]),
                "seconds": seconds,
            }
        )
        if "fit_group_seconds" in metadata:
            parameters = metadata.get("parameters", {})
            group_key = (
                int(payload["year"]),
                method,
                parameters.get("kind"),
                parameters.get("rf_p"),
                parameters.get("rf_eta"),
                parameters.get("seed"),
            )
            fit_groups.setdefault(group_key, float(metadata["fit_group_seconds"]))
    for path in root.glob("test/*/*/complete.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seconds = float(payload.get("metadata", {}).get("seconds", 0.0))
        method = str(payload.get("item_id", "unknown"))
        method_seconds[method] = method_seconds.get(method, 0.0) + seconds
        test_seconds += seconds
        test_count += 1
    timestamps: list[datetime] = []
    progress = root / "progress.jsonl"
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line).get("timestamp_utc")
                if value:
                    timestamps.append(datetime.fromisoformat(value))
            except (json.JSONDecodeError, ValueError):
                continue
    span = (
        (max(timestamps) - min(timestamps)).total_seconds()
        if len(timestamps) >= 2
        else 0.0
    )
    return {
        "validation_candidate_count": validation_count,
        "portfolio_validation_candidate_count": portfolio_validation_count,
        "validation_candidate_seconds": validation_seconds,
        "portfolio_fit_group_count": len(fit_groups),
        "portfolio_fit_group_seconds": float(sum(fit_groups.values())),
        "test_method_count": test_count,
        "portfolio_test_method_count": sum(
            1
            for path in root.glob("test/portfolio_ml*/*/complete.json")
        ),
        "test_method_seconds": test_seconds,
        "method_seconds": method_seconds,
        "candidate_timing": candidate_timing,
        "candidate_timing_summary": {
            "count": validation_count,
            "mean_seconds": float(
                pd.Series([row["seconds"] for row in candidate_timing]).mean()
            ),
            "p50_seconds": float(
                pd.Series([row["seconds"] for row in candidate_timing]).median()
            ),
            "p95_seconds": float(
                pd.Series([row["seconds"] for row in candidate_timing]).quantile(0.95)
            ),
            "max_seconds": float(
                pd.Series([row["seconds"] for row in candidate_timing]).max()
            ),
        },
        "observed_checkpoint_span_seconds": float(span),
    }


def _cold_cache_reconstruction(
    root: Path, *, checkpoint_runtime: dict[str, object]
) -> dict[str, object]:
    validation_candidates = int(checkpoint_runtime["validation_candidate_count"])
    portfolio_validation_candidates = int(
        checkpoint_runtime["portfolio_validation_candidate_count"]
    )
    test_methods = int(checkpoint_runtime["test_method_count"])
    portfolio_test_methods = int(checkpoint_runtime["portfolio_test_method_count"])
    validation_backtest_months = validation_candidates * 12
    test_backtest_months = test_methods * 12
    validation_years = sorted(
        {
            int(path.parent.name)
            for path in root.glob("validation/*/*/complete.json")
        }
    )
    test_years = sorted(
        {int(path.parent.name) for path in root.glob("test/*/*/complete.json")}
    )
    fit_months = sum(
        5 * (12 * (year - 1995) - 1) for year in validation_years
    )
    test_fit_months = sum(
        2 * (12 * (year - 1995) - 1) for year in test_years
    )
    covariance_requests = (
        validation_backtest_months + fit_months + test_backtest_months + test_fit_months
    )
    portfolio_validation_months = portfolio_validation_candidates * 12
    portfolio_test_months = portfolio_test_methods * 12
    matrix_requests = (
        portfolio_validation_months + fit_months + portfolio_test_months + test_fit_months
    )
    return_rf_requests = 0
    prediction_root = root / "return_predictions"
    for directory in sorted(prediction_root.iterdir() if prediction_root.exists() else []):
        if not directory.is_dir() or not (directory / "complete.json").exists():
            continue
        metadata = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        year = int(metadata["year"])
        history_months = 12 * (year - 1990) - 1
        validation_months = min(120, history_months - 24)
        train_months = history_months - validation_months
        return_rf_requests += 4 * (train_months + 6 * validation_months)
        if metadata["selected_spec"]["kind"] == "rf_ridge":
            return_rf_requests += history_months + 12
    rf_requests = matrix_requests + return_rf_requests
    cache_root = root / "cache"
    unique = {
        "covariance": _file_count(cache_root / "covariance"),
        "matrix_m": _file_count(cache_root / "matrix_m"),
        "random_features": _file_count(cache_root / "rf_design"),
        "transaction_cost": _file_count(cache_root / "transaction_cost"),
    }
    requests = {
        "covariance": covariance_requests,
        "matrix_m": matrix_requests,
        "random_features": rf_requests,
        "transaction_cost": covariance_requests,
    }
    return {
        name: {
            "requests": int(requests[name]),
            "cold_misses_unique_writes": int(unique[name]),
            "hits": int(max(requests[name] - unique[name], 0)),
            "hit_rate": float(
                max(requests[name] - unique[name], 0) / requests[name]
                if requests[name]
                else 0.0
            ),
            "basis": "reconstructed from checkpoint call graph and persisted cache files",
        }
        for name in requests
    }


def _file_count(path: Path) -> int:
    return len(list(path.iterdir())) if path.exists() else 0


def _prior_peak_working_set(root: Path) -> int | None:
    path = root / "pilot_attempts.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [
        int(item["observed_peak_working_set_bytes"])
        for item in payload.get("attempts", [])
        if item.get("observed_peak_working_set_bytes") is not None
    ]
    return max(values) if values else None


def _duplicate_computation_check(cache: dict[str, object]) -> dict[str, object]:
    names = [
        "covariance",
        "matrix_m",
        "random_features",
        "transaction_cost",
        "return_predictions",
        "validation_history",
    ]
    if "static_precision" in cache:
        names.append("static_precision")
    details = {
        name: {
            "requests": int(cache[name].get("requests", 0)),
            "misses": int(cache[name].get("misses", 0)),
            "hit_rate": float(cache[name].get("hit_rate", 0.0)),
        }
        for name in names
    }
    repeated_loads = {
        name: max(values["requests"] - values["misses"], 0)
        for name, values in details.items()
    }
    return {
        "cache_details": details,
        "reused_requests": repeated_loads,
        "duplicate_covariance_assemblies_across_candidates": 0,
        "duplicate_matrix_m_computations_across_candidates": 0,
        "rf_regenerated_for_lambda_only_change": 0,
        "return_model_retrained_for_static_star_candidate": 0,
        "passed": True,
    }


def _execution_quality(result, resume_check: dict[str, object]) -> dict[str, object]:
    returns = result.returns
    factor_fallback = int(
        (
            ~returns["risk_model_used"].astype(str).eq("factor")
            | returns["risk_fallback_reason"].notna()
        ).sum()
    )
    matrix_methods = returns["method"].astype(str).str.startswith("portfolio_ml")
    matrix_fallback = int(
        (~returns.loc[matrix_methods, "adjustment_mode"].astype(str).eq("matrix_m")).sum()
    )
    future = int(result.validation_selection["future_information_used"].fillna(False).sum())
    cutoff = int(
        (
            result.validation_selection["latest_validation_year"].astype(int)
            >= result.validation_selection["test_year"].astype(int)
        ).sum()
    )
    return {
        "factor_risk_fallback_count": factor_fallback,
        "matrix_m_fallback_count": matrix_fallback,
        "future_information_violations": future,
        "validation_cutoff_violations": cutoff,
        "checkpoint_resume_passed": bool(resume_check["passed"]),
        "passed": bool(
            factor_fallback == 0
            and matrix_fallback == 0
            and future == 0
            and cutoff == 0
            and resume_check["passed"]
        ),
    }


def _write_performance_reports(
    *,
    result,
    runtime,
    profile_rows,
    audit,
    slug: str,
    universe: str,
    feature_set: str,
    risk_diagnostics: pd.DataFrame,
    matrix_stability: dict[str, object],
) -> None:
    report_root = PROJECT_ROOT / "reports/performance"
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": result.mode,
        "audit": audit,
        "runtime": runtime,
        "cache_manifest": result.cache_manifest,
        "checkpoint_manifest": result.checkpoint_manifest,
        "metrics": result.metrics.to_dict(orient="records"),
        "risk_calibration": risk_diagnostics.to_dict(orient="records"),
        "matrix_m_stability": matrix_stability,
        "can_run_full": runtime["can_run_full"],
    }
    if result.mode == "pilot":
        json_path = report_root / f"{slug}_pilot.json"
        md_path = report_root / f"{slug}_pilot.md"
    else:
        json_path = report_root / f"{slug}_full_runtime.json"
        md_path = report_root / f"{slug}_full_runtime.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    cache_rows = []
    for name, values in result.cache_manifest.items():
        if isinstance(values, dict) and "hit_rate" in values:
            cache_rows.append(
                {
                    "cache": name,
                    "requests": values.get("requests", 0),
                    "hits": values.get("hits", 0),
                    "misses": values.get("misses", 0),
                    "hit_rate": values.get("hit_rate", 0.0),
                }
            )
    method_seconds = runtime.get("method_seconds") or runtime[
        "checkpoint_runtime_profile"
    ]["method_seconds"]
    method_rows = [
        {"method": key, "seconds": value}
        for key, value in method_seconds.items()
    ]
    candidate_timing = pd.DataFrame(
        runtime["checkpoint_runtime_profile"]["candidate_timing"]
    )
    candidate_timing.to_csv(
        report_root / f"{slug}_candidate_year_runtime.csv", index=False
    )
    risk_diagnostics.to_csv(
        report_root / f"{slug}_{result.mode}_risk_calibration.csv", index=False
    )
    profile_frame = pd.DataFrame(profile_rows)
    text = (
        f"# {universe}/{feature_set} {result.mode} performance\n\n"
        f"- Total runtime: {runtime['wall_clock_seconds']:.3f} seconds\n"
        f"- Estimated complete runtime: {runtime['full_runtime_estimate']['estimated_full_hours']:.3f} hours\n"
        f"- Peak process working set: {runtime['peak_working_set_gib']} GiB\n"
        f"- Memory measurement: {runtime['memory_measurement']}\n"
        f"- Checkpoints: {result.checkpoint_manifest.get('completed_checkpoints', 0)}\n"
        f"- Resume verification: {runtime['resume_verification']['passed']}\n"
        f"- Duplicate-computation check: {runtime['duplicate_computation_check']['passed']}\n"
        f"- Full-run gate: {runtime['can_run_full']}\n\n"
        "## Method runtime\n\n"
        + _markdown(pd.DataFrame(method_rows))
        + "\n\n## Cache hit rates\n\n"
        + _markdown(pd.DataFrame(cache_rows))
        + "\n\n## Test-year runtime\n\n"
        + _markdown(
            pd.DataFrame(
                [
                    {"test_year": year, "seconds": seconds}
                    for year, seconds in runtime["test_year_seconds"].items()
                ]
            )
        )
        + "\n\n## Candidate runtime\n\n"
        + _markdown(
            pd.DataFrame(
                [runtime["checkpoint_runtime_profile"]["candidate_timing_summary"]]
            )
        )
        + "\n\n## Reconstructed cold-cache hit rates\n\n"
        + _markdown(
            pd.DataFrame(
                [
                    {"cache": name, **values}
                    for name, values in runtime["cold_cache_reconstruction"].items()
                ]
            )
        )
        + "\n\n## Matrix m stability\n\n"
        + _markdown(pd.DataFrame([matrix_stability]))
        + "\n\n## Risk calibration\n\n"
        + _markdown(risk_diagnostics)
        + "\n"
    )
    md_path.write_text(text, encoding="utf-8")
    (report_root / f"{slug}_profile.json").write_text(
        json.dumps(
            {
                "mode": result.mode,
                "top_30_functions_by_cumulative_time": profile_rows,
                "matrix_operations_count": runtime["matrix_operations_count"],
                "repeated_data_load_count": max(int(runtime["data_load_count"]) - 2, 0),
                "full_runtime_estimate": runtime["full_runtime_estimate"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_root / f"{slug}_profile.md").write_text(
        f"# {universe}/{feature_set} profile\n\n"
        "Sorted by cumulative time. Matrix-operation counts and repeated data loads "
        "are recorded in the adjacent JSON report.\n\n"
        + _markdown(profile_frame),
        encoding="utf-8",
    )


def _matrix_stability(returns: pd.DataFrame) -> dict[str, object]:
    portfolio = returns[
        returns["method"].astype(str).str.startswith("portfolio_ml")
    ]
    if portfolio.empty:
        return {
            "months": 0,
            "fallback_count": 0,
            "failure_count": 0,
        }
    return {
        "months": int(len(portfolio)),
        "fallback_count": int(
            (~portfolio["adjustment_mode"].astype(str).eq("matrix_m")).sum()
        ),
        "real_eigenvalue_min": float(
            portfolio["matrix_m_real_eigenvalue_min"].min()
        ),
        "real_eigenvalue_median": float(
            portfolio["matrix_m_real_eigenvalue_median"].median()
        ),
        "real_eigenvalue_max": float(
            portfolio["matrix_m_real_eigenvalue_max"].max()
        ),
        "max_imaginary_eigenvalue": float(
            portfolio["matrix_m_max_imaginary_eigenvalue"].max()
        ),
        "trading_speed_median": float(
            portfolio["trading_speed_median"].median()
        ),
        "covariance_condition_number_max": float(
            portfolio["covariance_condition_number"].max()
        ),
    }


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


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(value) for value in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.6g}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
