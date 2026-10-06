from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "results/checkpoints/final_analysis"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports/final_analysis/preflight/final_analysis_checkpoint_integrity.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final-analysis checkpoints and caches")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--clean-corrupt-cache", action="store_true")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Check non-zero binary artifacts without loading every cache matrix.",
    )
    parser.add_argument(
        "--frontier-scenarios-only",
        action="store_true",
        help="Audit only the gamma_15 and gamma_20 checkpoint trees.",
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    corrupt: list[dict[str, object]] = []
    cleaned: list[str] = []
    checked = defaultdict(int)

    audit_roots = (
        [root / "frontier" / scenario for scenario in ("gamma_15", "gamma_20")]
        if args.frontier_scenarios_only
        else [root]
    )
    for path in (path for audit_root in audit_roots for path in audit_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        try:
            if path.stat().st_size == 0:
                raise ValueError("zero-byte file")
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                checked["json"] += 1
            elif path.suffix == ".parquet":
                if args.lightweight:
                    checked["parquet_nonzero"] += 1
                else:
                    pq.ParquetFile(path).metadata.num_rows
                    checked["parquet"] += 1
            elif path.suffix == ".npz":
                if args.lightweight:
                    checked["npz_nonzero"] += 1
                else:
                    with np.load(path, allow_pickle=False) as values:
                        for key in values.files:
                            values[key].shape
                    checked["npz"] += 1
            else:
                checked["other"] += 1
        except Exception as exc:  # audit must collect every damaged artifact
            is_cache_or_temp = "cache" in relative.parts or path.suffix in {
                ".tmp",
                ".partial",
            }
            row = {
                "path": str(relative),
                "size": path.stat().st_size if path.exists() else None,
                "error": f"{type(exc).__name__}: {exc}",
                "cache_or_temporary": is_cache_or_temp,
            }
            corrupt.append(row)
            if args.clean_corrupt_cache and is_cache_or_temp and path.exists():
                path.unlink()
                cleaned.append(str(relative))

    checkpoint_errors: list[dict[str, object]] = []
    for complete in (
        complete
        for audit_root in audit_roots
        for complete in audit_root.rglob("complete.json")
    ):
        directory = complete.parent
        relative = directory.relative_to(root)
        namespace = relative.parts[-3] if len(relative.parts) >= 3 else "unknown"
        required = ["returns.parquet", "state.parquet", "state_metadata.json"]
        if namespace == "test":
            required.extend(["predictions.parquet", "weights.parquet"])
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            checkpoint_errors.append(
                {"directory": str(relative), "missing": missing}
            )

    progress = {
        scenario: _scenario_progress(root / "frontier" / scenario)
        for scenario in ("gamma_15", "gamma_20")
    }
    historical_errors = sum(
        len(values["historical_non_blocking_error_records"])
        for values in progress.values()
    )
    blocking_errors = sum(
        len(values["blocking_error_records"]) for values in progress.values()
    )
    payload = {
        "root": str(root),
        "audit_scope": (
            "frontier_gamma_15_gamma_20"
            if args.frontier_scenarios_only
            else "all_final_analysis_checkpoints_and_caches"
        ),
        "audit_mode": "lightweight" if args.lightweight else "deep",
        "passed": not corrupt and not checkpoint_errors and blocking_errors == 0,
        "checked_file_counts": dict(checked),
        "corrupt_files": corrupt,
        "cleaned_cache_or_temporary_files": cleaned,
        "checkpoint_structure_errors": checkpoint_errors,
        "historical_non_blocking_error_count": historical_errors,
        "blocking_error_count": blocking_errors,
        "progress": progress,
    }
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 2


def _scenario_progress(root: Path) -> dict[str, object]:
    validation = root / "validation"
    test = root / "test"
    candidate_ids = sorted(
        path.name for path in validation.iterdir() if path.is_dir()
    ) if validation.exists() else []
    validation_counts: dict[int, int] = defaultdict(int)
    for complete in validation.glob("*/*/complete.json"):
        validation_counts[int(complete.parent.name)] += 1
    test_counts: dict[int, int] = defaultdict(int)
    for complete in test.glob("*/*/complete.json"):
        test_counts[int(complete.parent.name)] += 1
    expected = len(candidate_ids)
    complete_validation_years = sorted(
        year for year, count in validation_counts.items() if count == expected and expected > 0
    )
    partial_validation_years = {
        str(year): count
        for year, count in sorted(validation_counts.items())
        if count != expected
    }
    historical_errors: list[dict[str, object]] = []
    blocking_errors: list[dict[str, object]] = []
    for error_path in sorted(root.rglob("error.json")) if root.exists() else []:
        directory = error_path.parent
        complete_path = directory / "complete.json"
        required = [
            complete_path,
            directory / "returns.parquet",
            directory / "state.parquet",
            directory / "state_metadata.json",
        ]
        superseded = all(path.exists() and path.stat().st_size > 0 for path in required)
        error_payload = json.loads(error_path.read_text(encoding="utf-8"))
        row = {
            "path": str(error_path.relative_to(root)),
            "error_type": error_payload.get("error_type"),
            "failed_at_utc": error_payload.get("failed_at_utc"),
            "classification": (
                "historical_non_blocking_error" if superseded else "blocking_error"
            ),
            "reason": (
                "A later complete checkpoint with all required non-zero artifacts "
                "exists at the same candidate-year boundary."
                if superseded
                else "No complete replacement checkpoint exists at the same boundary."
            ),
        }
        if superseded:
            complete_payload = json.loads(complete_path.read_text(encoding="utf-8"))
            row["superseded_by_completed_at_utc"] = complete_payload.get(
                "completed_at_utc"
            )
            historical_errors.append(row)
        else:
            blocking_errors.append(row)
    return {
        "candidate_ids_observed": expected,
        "validation_candidate_years_complete": int(sum(validation_counts.values())),
        "fully_complete_validation_years": complete_validation_years,
        "latest_fully_complete_validation_year": (
            max(complete_validation_years) if complete_validation_years else None
        ),
        "partial_validation_years": partial_validation_years,
        "test_method_years_complete": int(sum(test_counts.values())),
        "test_counts_by_year": {str(k): v for k, v in sorted(test_counts.items())},
        "error_records": len(historical_errors) + len(blocking_errors),
        "historical_non_blocking_error_records": historical_errors,
        "blocking_error_records": blocking_errors,
    }


if __name__ == "__main__":
    raise SystemExit(main())
