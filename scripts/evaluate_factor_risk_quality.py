from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.data.loaders import (  # noqa: E402
    load_master_panel,
    load_risk_inputs,
)
from implementable_frontier.models.factor_risk_quality import (  # noqa: E402
    evaluate_factor_risk_quality,
    write_factor_risk_quality_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate repaired factor-risk inputs")
    parser.add_argument("--risk-dir", required=True)
    parser.add_argument("--start", default="2005-01-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--universe", default="top100")
    parser.add_argument("--config", default="configs/model_top100_core10.yaml")
    parser.add_argument("--output-dir", default="reports/factor_risk_quality/repaired")
    parser.add_argument("--raw-daily-glob")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel = load_master_panel(
        start=args.start,
        end=args.end,
        universe=args.universe,
        feature_set="core10",
    )
    risk_inputs = load_risk_inputs(
        start=args.start,
        end=args.end,
        universe=args.universe,
        root=(PROJECT_ROOT / args.risk_dir).resolve(),
    )
    quality = evaluate_factor_risk_quality(
        panel,
        risk_inputs,
        universe=args.universe,
        risk_settings=config["risk"],
    )
    if args.raw_daily_glob:
        import duckdb

        query = """
            select count(*) as count,
                   min(ret) as min,
                   quantile_cont(ret, 0.001) as p001,
                   quantile_cont(ret, 0.01) as p01,
                   quantile_cont(ret, 0.50) as p50,
                   quantile_cont(ret, 0.99) as p99,
                   quantile_cont(ret, 0.999) as p999,
                   max(ret) as max,
                   sum(case when abs(ret) > 1 then 1 else 0 end) as abs_gt_1
            from read_parquet(?) where ret is not null
        """
        row = duckdb.connect().execute(query, [args.raw_daily_glob]).fetchdf().iloc[0]
        quality["raw_daily_return"] = {
            key: int(value) if key in {"count", "abs_gt_1"} else float(value)
            for key, value in row.to_dict().items()
        }
        quality["raw_daily_return"]["unit_check"] = "decimal"
        quality["raw_daily_return"]["winsorization_used"] = [-0.5, 0.5]
    json_path, markdown_path = write_factor_risk_quality_report(
        quality, (PROJECT_ROOT / args.output_dir).resolve()
    )
    print(json.dumps({key: value for key, value in quality.items() if key != "monthly_diagnostics"}, indent=2, default=str))
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    return 0 if quality["quality_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
