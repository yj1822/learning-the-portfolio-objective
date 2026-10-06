from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data/diagnostics/final20_conservative_jkp"
ORIGINAL_PANEL = PROJECT_ROOT / "data/processed/us_equity_ml/master_panel.parquet"
OUTPUT_PANEL = (
    PROJECT_ROOT
    / "data/processed/us_equity_ml/master_panel_final20_conservative_full.parquet"
)
SELECTION_YAML = PROJECT_ROOT / "reports/feature_selection/final20_conservative.yaml"



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an isolated 1990-2007 Final20 Conservative pilot panel."
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    import yaml

    selection = yaml.safe_load(SELECTION_YAML.read_text(encoding="utf-8"))
    additions = list(selection["selected_additional_features"])
    details = {row["feature"]: row for row in selection["selected_feature_details"]}
    if len(additions) != 10 or set(additions) != set(details):
        raise RuntimeError("Final20 Conservative selection YAML is incomplete")
    fields = [str(details[feature]["jkp_field"]) for feature in additions]

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    first_source_year, last_source_year = 1990, 2024
    missing_years = [
        year
        for year in range(first_source_year, last_source_year + 1)
        if args.force_download
        or not (SOURCE_ROOT / f"year={year}" / "part.parquet").exists()
    ]
    if missing_years:
        fetch_source_partitions(missing_years, fields)

    source_files = sorted(SOURCE_ROOT.glob("year=*/part.parquet"))
    expected_partitions = last_source_year - first_source_year + 1
    if len(source_files) != expected_partitions:
        raise RuntimeError(
            f"Expected {expected_partitions} JKP source partitions, found {len(source_files)}"
        )
    source = pd.concat([pd.read_parquet(path) for path in source_files], ignore_index=True)
    source["eom"] = pd.to_datetime(source["eom"])
    source["permno"] = pd.to_numeric(source["permno"], errors="coerce")
    keyed_source = source[source["permno"].notna()]
    if keyed_source.duplicated(["eom", "permno"]).any():
        raise RuntimeError("JKP source has duplicate eom/permno rows")

    feature_frame = source[["eom", "permno"]].copy()
    missing_rates: dict[str, float] = {}
    for feature, field in zip(additions, fields, strict=True):
        raw = pd.to_numeric(source[field], errors="coerce")
        feature_frame[f"raw_{feature}"] = raw
        feature_frame[f"rank_{feature}"] = (
            source.assign(_value=raw)
            .groupby("eom", sort=False)["_value"]
            .rank(pct=True, method="average")
            .fillna(0.5)
        )
        missing_rates[feature] = float(raw.isna().mean())

    feature_frame = feature_frame[feature_frame["permno"].notna()].copy()
    feature_frame["permno"] = feature_frame["permno"].astype(int)

    original = pd.read_parquet(ORIGINAL_PANEL)
    original["eom"] = pd.to_datetime(original["eom"])
    extension_columns = [
        column
        for feature in additions
        for column in (f"raw_{feature}", f"rank_{feature}")
    ]
    overlap = sorted(set(extension_columns) & set(original.columns))
    if overlap:
        raise RuntimeError(f"Original panel already contains Final20 columns: {overlap}")
    extended = original.merge(
        feature_frame,
        on=["eom", "permno"],
        how="left",
        validate="one_to_one",
    )
    pilot = extended[
        extended["in_top500"].fillna(False)
        & extended["eom"].between("1995-01-31", "2024-12-31")
    ]
    if len(pilot) != 180_000 or pilot["eom"].nunique() != 360:
        raise RuntimeError("Final20 full panel does not contain 500 names for 360 months")
    for feature in additions:
        raw = pilot[f"raw_{feature}"]
        rank = pilot[f"rank_{feature}"]
        if rank.isna().any() or (~rank.between(0.0, 1.0)).any():
            raise RuntimeError(f"Invalid pilot ranks for {feature}")
        if not rank[raw.isna()].eq(0.5).all():
            raise RuntimeError(f"Missing values are not filled with 0.5 for {feature}")

    temporary = OUTPUT_PANEL.with_suffix(".tmp.parquet")
    extended.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT_PANEL)
    metadata = {
        "status": "full_panel_ready",
        "source": "contrib_global_factor.global_factor",
        "source_start": "1990-01-31",
        "source_end": "2024-12-31",
        "selection_yaml": str(SELECTION_YAML.relative_to(PROJECT_ROOT)),
        "original_panel": str(ORIGINAL_PANEL.relative_to(PROJECT_ROOT)),
        "output_panel": str(OUTPUT_PANEL.relative_to(PROJECT_ROOT)),
        "rows": int(len(extended)),
        "full_analysis_rows": int(len(pilot)),
        "full_analysis_months": int(pilot["eom"].nunique()),
        "additions": additions,
        "jkp_fields": fields,
        "eligible_cross_section_missing_rates": missing_rates,
        "original_columns_preserved": bool(
            original.equals(extended[original.columns])
        ),
    }
    OUTPUT_PANEL.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"output={OUTPUT_PANEL}")
    print(f"full_analysis_rows={len(pilot)}")
    print(f"full_analysis_months={pilot['eom'].nunique()}")
    print(f"original_columns_preserved={metadata['original_columns_preserved']}")
    return 0


def fetch_source_partitions(years: list[int], fields: list[str]) -> None:
    import wrds

    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    if not username or not password:
        raise RuntimeError("WRDS credentials are missing")
    selected = ", ".join(["eom", "permno", *fields])
    connection = wrds.Connection(
        wrds_username=str(username), wrds_password=str(password), verbose=False
    )
    try:
        for year in years:
            sql = f"""
                SELECT {selected}
                FROM contrib_global_factor.global_factor
                WHERE eom >= %(start_date)s
                  AND eom <= %(end_date)s
                  AND excntry = 'USA'
                  AND common = 1
                  AND exch_main = 1
                  AND primary_sec = 1
                  AND obs_main = 1
            """
            frame = connection.raw_sql(
                sql,
                params={
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                },
            )
            frame.columns = [str(column).lower() for column in frame.columns]
            frame["eom"] = pd.to_datetime(frame["eom"])
            output = SOURCE_ROOT / f"year={year}" / "part.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False)
            print(f"year={year} source_rows={len(frame)} status=written")
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
