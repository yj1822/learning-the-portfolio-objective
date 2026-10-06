from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from implementable_frontier.data.config import DataConfigError, load_data_config  # noqa: E402
from implementable_frontier.data.hub import (  # noqa: E402
    TradingDataHubAdapter,
    TradingDataHubUnavailableError,
)
from implementable_frontier.data.preparation import (  # noqa: E402
    DataPreparationError,
    build_us_equity_ml_dataset,
    export_sample,
)
from implementable_frontier.data.validation import validate_us_equity_ml_dataset  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_data_config(args.config)
        if args.mode == "build":
            hub = TradingDataHubAdapter(
                project_root=args.data_hub_root,
                config_path=args.data_hub_config,
            )
            result = build_us_equity_ml_dataset(config, hub, download=args.download)
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            return 0 if result.validation_passed else 1
        if args.mode == "validate":
            report = validate_us_equity_ml_dataset(config=config)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["passed"] else 1
        _require_sample_arguments(parser, args)
        output = export_sample(
            config,
            universe=args.universe,
            feature_set=args.feature_set,
            start=args.start,
            end=args.end,
        )
        print(json.dumps({"sample": str(output)}, indent=2, ensure_ascii=False))
        return 0
    except (
        DataConfigError,
        TradingDataHubUnavailableError,
        DataPreparationError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, validate, or sample the thesis US-equity research dataset."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "data_us_equity_ml.yaml"),
    )
    parser.add_argument("--mode", choices=["build", "validate", "sample"], required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="In build mode, explicitly request licensed WRDS downloads through trading-data-hub.",
    )
    parser.add_argument("--data-hub-root", help="trading-data-hub checkout; or set TRADING_DATA_HUB_ROOT")
    parser.add_argument("--data-hub-config", help="sources.yaml path; or set DATAHUB_CONFIG")
    parser.add_argument("--universe", choices=["top100", "top300", "top500"])
    parser.add_argument("--feature-set", choices=["core10", "final15"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser


def _require_sample_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [name for name in ("universe", "feature_set", "start", "end") if getattr(args, name) is None]
    if missing:
        parser.error(f"--mode sample requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")


if __name__ == "__main__":
    raise SystemExit(main())
