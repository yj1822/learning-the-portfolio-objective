"""Public baseline entry point for the locked cumulative OOS experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked Top500/Core10 cumulative OOS experiment")
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", default="configs/model_top500_core10_cumulative.yaml")
    args, extra = parser.parse_known_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_cumulative_experiment.py"),
        "--config", args.config,
        "--mode", args.mode,
        *extra,
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
