from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_final20_features import CANDIDATE_DATA_ROOT, FEATURE_METADATA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch only preregistered 1995-2004 JKP fields for Final20 diagnostics."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import wrds

    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set WRDS_USERNAME and WRDS_PASSWORD in the environment")

    master = pd.read_parquet(
        PROJECT_ROOT / "data/processed/us_equity_ml/master_panel.parquet",
        columns=["eom", "permno", "in_top500"],
    )
    master["eom"] = pd.to_datetime(master["eom"])
    keys = master[
        master["in_top500"].fillna(False)
        & master["eom"].between("1995-01-31", "2004-12-31")
    ][["eom", "permno"]].copy()
    permnos = sorted(keys["permno"].astype(int).unique().tolist())
    fields = sorted({metadata["jkp_field"] for metadata in FEATURE_METADATA.values()})
    columns = ["eom", "permno", *fields]
    selected = ", ".join(columns)

    CANDIDATE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    connection = wrds.Connection(
        wrds_username=str(username),
        wrds_password=str(password),
        verbose=False,
    )
    try:
        for year in range(1995, 2005):
            output = CANDIDATE_DATA_ROOT / f"year={year}" / "part.parquet"
            if output.exists() and not args.force:
                print(f"year={year} status=skip_completed")
                continue
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
                  AND permno = ANY(%(permnos)s)
            """
            frame = connection.raw_sql(
                sql,
                params={
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "permnos": permnos,
                },
            )
            frame.columns = [str(column).lower() for column in frame.columns]
            frame["eom"] = pd.to_datetime(frame["eom"])
            frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype(int)
            frame = frame.merge(
                keys[keys["eom"].dt.year.eq(year)],
                on=["eom", "permno"],
                how="inner",
                validate="one_to_one",
            ).sort_values(["eom", "permno"])
            expected = int((keys["eom"].dt.year == year).sum())
            if len(frame) != expected:
                raise RuntimeError(
                    f"Top500 key coverage failed for {year}: {len(frame)}/{expected} rows"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False)
            print(f"year={year} rows={len(frame)} columns={len(frame.columns)} status=written")
    finally:
        connection.close()

    manifest = pd.DataFrame(
        [
            {
                "canonical_feature": feature,
                "jkp_field": metadata["jkp_field"],
                "theme": metadata["theme"],
            }
            for feature, metadata in FEATURE_METADATA.items()
        ]
    )
    manifest.to_csv(CANDIDATE_DATA_ROOT / "candidate_manifest.csv", index=False)
    print(f"candidate_fields={len(fields)}")
    print(f"output={CANDIDATE_DATA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
