from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.experiment import (  # noqa: E402
    ExperimentResult,
    run_model_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an implementable-frontier model experiment")
    parser.add_argument("--config", default="configs/model_top100_core10.yaml")
    parser.add_argument("--experiment-id")
    parser.add_argument("--methods", help="Comma-separated subset enabled by the config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()] if args.methods else None
    result = run_model_experiment(
        args.config,
        experiment_id=args.experiment_id,
        methods=methods,
        dry_run=args.dry_run,
    )
    if isinstance(result, ExperimentResult):
        print(f"experiment_id={result.experiment_id}")
        print(f"output_dir={result.output_dir}")
        print(f"report={result.report_path}")
        print(result.metrics.to_string(index=False))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
