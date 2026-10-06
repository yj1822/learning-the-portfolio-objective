from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.data.hub import TradingDataHubAdapter  # noqa: E402
from implementable_frontier.data.loaders import load_master_panel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild versioned factor-risk inputs from existing CRSP daily raw data"
    )
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--hub-root", required=True)
    parser.add_argument(
        "--raw-root",
        default=os.getenv("IEF_WRDS_RAW_ROOT"),
        required=os.getenv("IEF_WRDS_RAW_ROOT") is None,
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/us_equity_ml/risk_repaired",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--return-winsor", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    existing = list(output_dir.glob("*.parquet")) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Risk output already exists at {output_dir}; pass --overwrite explicitly"
        )
    panel = load_master_panel(
        start=args.start,
        end=args.end,
        universe="top500",
        feature_set="final15",
    )
    adapter = TradingDataHubAdapter(project_root=args.hub_root)
    builder = adapter._import_internal("datahub.us_equity.risk")
    outputs = builder.build_us_equity_risk_inputs(
        args.start,
        args.end,
        panel=panel,
        ewma_span=60,
        raw_root=Path(args.raw_root).expanduser().resolve(),
        output_root=output_dir,
        return_winsor_limits=(-args.return_winsor, args.return_winsor),
        style_winsor_quantiles=(0.01, 0.99),
        regression_ridge=args.ridge_alpha,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": args.start,
        "end": args.end,
        "raw_root": str(Path(args.raw_root).expanduser().resolve()),
        "output_dir": str(output_dir),
        "trading_data_hub": adapter.identity.__dict__,
        "return_winsor_limits": [-args.return_winsor, args.return_winsor],
        "style_winsor_quantiles": [0.01, 0.99],
        "regression": "ridge with unpenalized intercept",
        "ridge_alpha": args.ridge_alpha,
        "factor_covariance": "full EWMA daily covariance",
        "rows": {name: int(len(frame)) for name, frame in outputs.items()},
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
