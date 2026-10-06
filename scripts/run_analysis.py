"""Public entry point for the locked frontier and AUM scenario grids."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked final-analysis grid")
    parser.add_argument("--stage", choices=("frontier", "aum"), required=True)
    args, extra = parser.parse_known_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_final_analysis_grid.py"),
        "--stage", args.stage,
        *extra,
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
