"""Public data entry point; delegate to the existing research implementation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, validate, or sample the licensed input panel")
    parser.add_argument("--mode", choices=("build", "validate", "sample"), required=True)
    parser.add_argument("--config", default="configs/data_us_equity_ml.yaml")
    parser.add_argument("--download", action="store_true", help="Explicitly request licensed WRDS downloads")
    args, extra = parser.parse_known_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_us_equity_ml_data.py"),
        "--mode", args.mode,
        "--config", args.config,
        *(["--download"] if args.download else []),
        *extra,
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
