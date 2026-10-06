from __future__ import annotations

import os

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/extensions/core10_predictive_alpha_decay"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

PANEL_PATH = PROJECT_ROOT / "data/processed/us_equity_ml/master_panel.parquet"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs/data_us_equity_ml.yaml"
THEME_PATH = PROJECT_ROOT / "configs/core10_theme_mapping.yaml"
EFI_PATH = PROJECT_ROOT / "reports/final_analysis/feature_importance/feature_importance.csv"
EFI_REPORT_PATH = PROJECT_ROOT / "reports/final_analysis/feature_importance/feature_theme_importance.md"
EXISTING_DIAGNOSTICS_PATH = PROJECT_ROOT / "reports/final_analysis/feature_importance/core10_feature_diagnostics.csv"
MECHANISM_REPORT_PATH = PROJECT_ROOT / "reports/final_analysis/mechanisms/mechanism_plots.md"
TIMING_VALIDATION_PATH = PROJECT_ROOT / "reports/data_validation/us_equity_ml_validation.json"
PRIOR_ROOT = PROJECT_ROOT / "outputs/extensions/core10_signal_persistence"
PRIOR_MASTER_PATH = PRIOR_ROOT / "core10_feature_persistence_master.csv"
PRIOR_CORRELATION_PATH = PRIOR_ROOT / "correlation_analysis.csv"
PRIOR_REPORT_PATH = PRIOR_ROOT / "core10_signal_persistence_report.md"

OOS_START = pd.Timestamp("2005-01-31")
OOS_END = pd.Timestamp("2024-12-31")
HORIZONS = tuple(range(1, 13))
MIN_MONTHLY_OBSERVATIONS = 20
HAC_LAGS = 6
NEAR_ZERO_TOLERANCE = 1.0e-8
PRIMARY_METHOD = "portfolio_ml_linear"
SECONDARY_METHOD = "portfolio_ml_rf"
METHOD_LABELS = {
    PRIMARY_METHOD: "Portfolio-ML Linear",
    SECONDARY_METHOD: "Portfolio-ML RF",
}
EFI_METRIC = (
    "annualized ex-ante utility flow difference: full fixed model minus "
    "feature-neutralized fixed model"
)
ALPHA_METRICS = [
    "fmb_sfm",
    "fmb_sdm",
    "fmb_apr_3",
    "fmb_apr_6",
    "fmb_apr_12",
    "rankic_sfm",
    "rankic_sdm",
    "rankic_apr_3",
    "rankic_apr_6",
    "rankic_apr_12",
]
COLORS = {
    "navy": "#17324D",
    "blue": "#2563A6",
    "red": "#B43C55",
    "green": "#2D7D62",
    "gold": "#B8871B",
    "purple": "#7657A5",
    "teal": "#258A8A",
    "orange": "#D0743C",
    "gray": "#687785",
    "light": "#E8EEF3",
    "grid": "#D6DEE5",
}
LINE_COLORS = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["gold"],
    COLORS["purple"],
    COLORS["teal"],
    COLORS["orange"],
    "#4F6D7A",
    "#A66378",
    "#5F8D4E",
]


def main() -> int:
    started = time.perf_counter()
    allowed_arguments = {"--reuse-horizon-statistics"}
    unknown_arguments = set(sys.argv[1:]) - allowed_arguments
    if unknown_arguments:
        raise RuntimeError(f"Unknown command-line arguments: {sorted(unknown_arguments)}")
    reuse_horizon_statistics = "--reuse-horizon-statistics" in sys.argv[1:]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    data_config = _read_yaml(DATA_CONFIG_PATH)
    theme_config = _read_yaml(THEME_PATH)
    features = [str(value) for value in data_config["feature_sets"]["core10"]]
    themes = theme_config.get("themes", {})
    _validate_theme_mapping(features, themes)
    _validate_inputs_exist()
    theme_by_feature = {
        feature: theme for theme, members in themes.items() for feature in members
    }

    source_hashes = _source_hashes()
    timing_validation = json.loads(TIMING_VALIDATION_PATH.read_text(encoding="utf-8"))
    _validate_label_semantics(timing_validation)

    columns = [
        "eom",
        "permno",
        "in_top500",
        "ret_exc_lead1m",
        *[f"raw_{feature}" for feature in features],
        *[f"rank_{feature}" for feature in features],
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=columns)
    panel["eom"] = pd.to_datetime(panel["eom"], errors="raise")
    panel = panel[panel["in_top500"].fillna(False)].copy()
    panel = panel.sort_values(["eom", "permno"]).reset_index(drop=True)
    _validate_panel(panel, features)
    feature_rows = panel[panel["eom"].between(OOS_START, OOS_END)].copy()
    if feature_rows["eom"].nunique() != 240:
        raise RuntimeError("Locked OOS feature panel must contain exactly 240 months")

    # Phase 0 is executed and validated before any horizon statistic is computed.
    horizon_joins, timing_rows, sample_rows = _build_and_validate_horizon_joins(
        panel, feature_rows, features
    )
    timing_audit = pd.DataFrame(timing_rows)
    horizon_sample_counts = pd.DataFrame(sample_rows)

    if reuse_horizon_statistics:
        monthly_all = _load_existing_monthly_statistics()
    else:
        monthly_all = _calculate_monthly_horizon_statistics(
            horizon_joins, features, theme_by_feature
        )
    monthly_rank_ic = monthly_all[
        [
            "feature",
            "theme",
            "horizon",
            "feature_eom",
            "target_label_eom",
            "return_eom",
            "n_valid",
            "rank_ic",
        ]
    ].copy()
    monthly_pearson_ic = monthly_all[
        [
            "feature",
            "theme",
            "horizon",
            "feature_eom",
            "target_label_eom",
            "return_eom",
            "n_valid",
            "pearson_ic",
        ]
    ].copy()
    monthly_fmb = monthly_all[
        [
            "feature",
            "theme",
            "horizon",
            "feature_eom",
            "target_label_eom",
            "return_eom",
            "n_valid",
            "fmb_intercept",
            "fmb_slope",
        ]
    ].copy()

    rank_summary = _summary_table(monthly_all, "rank_ic", "rank_ic")
    pearson_summary = _summary_table(monthly_all, "pearson_ic", "pearson_ic")
    fmb_summary = _fmb_summary(monthly_all)
    h1_reconciliation = _reconcile_horizon_one(
        rank_summary, pearson_summary, features
    )
    if not h1_reconciliation["passed"]:
        raise RuntimeError(
            "Horizon-1 IC does not reproduce the locked one-month diagnostics: "
            f"{h1_reconciliation}"
        )

    existing_efi = _extract_existing_efi(features)
    characteristic = _extract_characteristic_persistence(features)
    metrics = _alpha_persistence_metrics(
        fmb_summary, rank_summary, features, theme_by_feature
    )
    master = _build_master(
        features,
        theme_by_feature,
        characteristic,
        rank_summary,
        fmb_summary,
        metrics,
        existing_efi,
    )
    characteristic_vs_alpha = _characteristic_vs_alpha(master)
    efi_correlations, prior_linear_correlation = _efi_correlations(master)
    loo_summary, loo_detail = _loo_sensitivity(master)
    feature_diagnostics = _feature_diagnostics(master, features)
    evidence = _classify_evidence(
        master, efi_correlations, loo_summary, prior_linear_correlation
    )

    analysis_tables = {
        "timing_audit.csv": timing_audit,
        "horizon_sample_counts.csv": horizon_sample_counts,
        "monthly_rank_ic.csv": monthly_rank_ic,
        "rank_ic_summary.csv": rank_summary,
        "monthly_pearson_ic.csv": monthly_pearson_ic,
        "pearson_ic_summary.csv": pearson_summary,
        "monthly_fmb_slopes.csv": monthly_fmb,
        "fmb_summary.csv": fmb_summary,
        "alpha_persistence_metrics.csv": metrics,
        "characteristic_vs_alpha.csv": characteristic_vs_alpha,
        "existing_efi.csv": existing_efi,
        "core10_predictive_alpha_decay_master.csv": master,
        "efi_correlations.csv": efi_correlations,
        "loo_sensitivity.csv": loo_summary,
        "loo_sensitivity_detail.csv": loo_detail,
        "feature_diagnostics.csv": feature_diagnostics,
    }
    for filename, frame in analysis_tables.items():
        frame.to_csv(OUTPUT_ROOT / filename, index=False)

    input_audit = _input_audit(source_hashes, panel, feature_rows, features)
    input_audit.to_csv(OUTPUT_ROOT / "input_audit.csv", index=False)
    workbook_readme = _workbook_readme(features, evidence)
    workbook_readme.to_csv(OUTPUT_ROOT / "workbook_readme.csv", index=False)

    timing_markdown = _timing_audit_markdown(
        timing_audit,
        horizon_sample_counts,
        source_hashes,
        h1_reconciliation,
        features,
    )
    (OUTPUT_ROOT / "00_timing_and_input_audit.md").write_text(
        timing_markdown, encoding="utf-8"
    )

    _make_figures(master, rank_summary, fmb_summary, themes)
    report = _analysis_report(
        master,
        efi_correlations,
        loo_summary,
        evidence,
        horizon_sample_counts,
        prior_linear_correlation,
    )
    (OUTPUT_ROOT / "core10_predictive_alpha_decay_report.md").write_text(
        report, encoding="utf-8"
    )

    runtime_seconds = time.perf_counter() - started
    config = {
        "scope": "Top500/Core10 post-hoc predictive-alpha-decay mechanism analysis",
        "panel": _relative(PANEL_PATH),
        "oos_feature_start": str(OOS_START.date()),
        "oos_feature_end": str(OOS_END.date()),
        "horizons": list(HORIZONS),
        "target_definition": (
            "feature at t joins by permno/date to ret_exc_lead1m at t+h-1; "
            "this is the incremental one-month excess return r_(t+h)"
        ),
        "target_is_cumulative": False,
        "min_monthly_observations": MIN_MONTHLY_OBSERVATIONS,
        "fmb_hac_lags": HAC_LAGS,
        "near_zero_tolerance": NEAR_ZERO_TOLERANCE,
        "near_zero_tolerance_units": "monthly return fraction per unit rank",
        "primary_alpha_persistence_metric": "fmb_sfm",
        "secondary_alpha_persistence_metric": "fmb_sdm",
        "primary_efi_method": METHOD_LABELS[PRIMARY_METHOD],
        "secondary_efi_method": METHOD_LABELS[SECONDARY_METHOD],
        "efi_metric": EFI_METRIC,
        "model_retraining": False,
        "risk_model_rerun": False,
        "transaction_cost_model_rerun": False,
        "aum_rerun": False,
        "checkpoint_write": False,
        "monthly_horizon_statistics_reused_for_report_refresh": reuse_horizon_statistics,
        "cumulative_return_appendix_produced": False,
        "evidence_classification": evidence["classification"],
        "decision_gate": evidence["decision_gate"],
        "source_hashes_sha256": source_hashes,
        "runtime_seconds_before_workbook": runtime_seconds,
    }
    (OUTPUT_ROOT / "analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    validation = _validation_summary(
        panel,
        master,
        rank_summary,
        pearson_summary,
        fmb_summary,
        horizon_sample_counts,
        h1_reconciliation,
        existing_efi,
        features,
        runtime_seconds,
    )
    (OUTPUT_ROOT / "validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    execution_log = _execution_log(
        master,
        efi_correlations,
        loo_summary,
        evidence,
        horizon_sample_counts,
        prior_linear_correlation,
        runtime_seconds,
    )
    (OUTPUT_ROOT / "execution_log.txt").write_text(
        execution_log, encoding="utf-8"
    )
    print(execution_log)
    return 0


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return value


def _validate_inputs_exist() -> None:
    required = [
        PANEL_PATH,
        DATA_CONFIG_PATH,
        THEME_PATH,
        EFI_PATH,
        EFI_REPORT_PATH,
        EXISTING_DIAGNOSTICS_PATH,
        MECHANISM_REPORT_PATH,
        TIMING_VALIDATION_PATH,
        PRIOR_MASTER_PATH,
        PRIOR_CORRELATION_PATH,
        PRIOR_REPORT_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative input(s): {missing}")


def _validate_theme_mapping(features: list[str], themes: dict) -> None:
    mapped = [feature for members in themes.values() for feature in members]
    if sorted(mapped) != sorted(features) or len(mapped) != len(set(mapped)):
        raise RuntimeError("Theme mapping is not an exact non-overlapping Core10 map")


def _validate_label_semantics(validation: dict) -> None:
    checks = validation.get("checks", {})
    integrity = checks.get("lookahead_diagnostics", {}).get("details", {})
    alignment = integrity.get("label_alignment", {})
    source = str(integrity.get("label_source", ""))
    same = float(alignment.get("same_month_correlation", np.nan))
    next_month = float(alignment.get("next_month_correlation", np.nan))
    if "ret_exc_lead1m" not in source or not np.isfinite(next_month):
        raise RuntimeError("Locked label timing validation is missing")
    if next_month <= same + 0.10:
        raise RuntimeError("Locked label does not validate as a next-month return")


def _validate_panel(panel: pd.DataFrame, features: list[str]) -> None:
    if panel.duplicated(["permno", "eom"]).any():
        raise RuntimeError("Panel contains duplicate permno/eom keys")
    counts = panel.groupby("eom").size()
    if len(panel) != 210000 or counts.nunique() != 1 or int(counts.iloc[0]) != 500:
        raise RuntimeError("Panel is not the locked dynamic Top500 panel")
    if panel["eom"].min() != pd.Timestamp("1990-01-31"):
        raise RuntimeError("Unexpected panel start")
    if panel["eom"].max() != OOS_END:
        raise RuntimeError("Unexpected panel end")
    for feature in features:
        rank = pd.to_numeric(panel[f"rank_{feature}"], errors="coerce")
        if rank.isna().any() or not rank.between(0.0, 1.0).all():
            raise RuntimeError(f"Invalid locked rank input: {feature}")


def _build_and_validate_horizon_joins(
    panel: pd.DataFrame, feature_rows: pd.DataFrame, features: list[str]
) -> tuple[dict[int, pd.DataFrame], list[dict], list[dict]]:
    feature_columns = ["eom", "permno", *[f"rank_{feature}" for feature in features]]
    base = feature_rows[feature_columns].rename(columns={"eom": "feature_eom"})
    targets = panel[["eom", "permno", "ret_exc_lead1m"]].rename(
        columns={"eom": "target_label_eom"}
    )
    joins: dict[int, pd.DataFrame] = {}
    timing_rows: list[dict] = []
    sample_rows: list[dict] = []
    prior_months = None
    prior_stock_months = None

    for horizon in HORIZONS:
        left = base.copy()
        left["target_label_eom"] = left["feature_eom"] + pd.offsets.MonthEnd(
            horizon - 1
        )
        joined = left.merge(
            targets,
            on=["permno", "target_label_eom"],
            how="inner",
            validate="one_to_one",
        )
        joined["return_eom"] = joined["target_label_eom"] + pd.offsets.MonthEnd(1)
        joined = joined[joined["ret_exc_lead1m"].notna()].copy()
        joined = joined.sort_values(["feature_eom", "permno"]).reset_index(drop=True)
        if joined.empty:
            raise RuntimeError(f"No valid explicit horizon join for h={horizon}")
        if not (
            joined["target_label_eom"]
            == joined["feature_eom"] + pd.offsets.MonthEnd(horizon - 1)
        ).all():
            raise RuntimeError(f"Target-label date mismatch at h={horizon}")
        if not (
            joined["return_eom"]
            == joined["feature_eom"] + pd.offsets.MonthEnd(horizon)
        ).all():
            raise RuntimeError(f"Return date mismatch at h={horizon}")
        if joined.duplicated(["feature_eom", "permno"]).any():
            raise RuntimeError(f"Duplicate explicit join keys at h={horizon}")

        monthly_n = joined.groupby("feature_eom").size()
        months = int(joined["feature_eom"].nunique())
        stock_months = int(len(joined))
        if prior_months is not None and months >= prior_months:
            raise RuntimeError("Later horizons did not lose the expected terminal month")
        if prior_stock_months is not None and stock_months >= prior_stock_months:
            raise RuntimeError("Later horizons did not reduce stock-month availability")
        prior_months = months
        prior_stock_months = stock_months
        joins[horizon] = joined

        example_feature_eom = pd.Timestamp("2005-01-31")
        timing_rows.append(
            {
                "horizon": horizon,
                "feature_date_rule": "feature_eom = t",
                "target_label_date_rule": f"t + {horizon - 1} month-end(s)",
                "return_date_rule": f"t + {horizon} month-end(s)",
                "example_feature_eom": example_feature_eom.date().isoformat(),
                "example_target_label_eom": (
                    example_feature_eom + pd.offsets.MonthEnd(horizon - 1)
                ).date().isoformat(),
                "example_return_eom": (
                    example_feature_eom + pd.offsets.MonthEnd(horizon)
                ).date().isoformat(),
                "return_column": "ret_exc_lead1m",
                "target_type": "incremental single-month excess return",
                "join_keys": "permno + explicit target_label_eom",
                "missing_future_policy": (
                    "exclude unavailable/missing locked-panel future labels; no imputation"
                ),
            }
        )
        sample_rows.append(
            {
                "horizon": horizon,
                "feature_eom_start": joined["feature_eom"].min().date().isoformat(),
                "feature_eom_end": joined["feature_eom"].max().date().isoformat(),
                "target_label_eom_start": joined["target_label_eom"].min().date().isoformat(),
                "target_label_eom_end": joined["target_label_eom"].max().date().isoformat(),
                "return_eom_start": joined["return_eom"].min().date().isoformat(),
                "return_eom_end": joined["return_eom"].max().date().isoformat(),
                "valid_months": months,
                "valid_stock_months": stock_months,
                "average_cross_section_n": float(monthly_n.mean()),
                "minimum_cross_section_n": int(monthly_n.min()),
                "maximum_cross_section_n": int(monthly_n.max()),
                "feature_rows_without_matching_valid_target": int(len(base) - stock_months),
            }
        )
    if joins[1]["feature_eom"].nunique() != 240:
        raise RuntimeError("Horizon 1 must retain all 240 OOS feature months")
    if joins[12]["feature_eom"].nunique() != 229:
        raise RuntimeError("Horizon 12 must end with 229 usable feature months")
    return joins, timing_rows, sample_rows


def _calculate_monthly_horizon_statistics(
    horizon_joins: dict[int, pd.DataFrame],
    features: list[str],
    theme_by_feature: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        joined = horizon_joins[horizon]
        for feature_eom, month in joined.groupby("feature_eom", sort=True):
            target_label_eom = pd.Timestamp(month["target_label_eom"].iloc[0])
            return_eom = pd.Timestamp(month["return_eom"].iloc[0])
            y_all = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce")
            for feature in features:
                x_all = pd.to_numeric(month[f"rank_{feature}"], errors="coerce")
                valid = x_all.notna() & y_all.notna()
                x = x_all[valid]
                y = y_all[valid]
                n_valid = int(valid.sum())
                if n_valid < MIN_MONTHLY_OBSERVATIONS:
                    pearson_ic = rank_ic = intercept = slope = np.nan
                else:
                    pearson_ic = _pearson(x, y)
                    rank_ic = _spearman(x, y)
                    intercept, slope = _cross_sectional_ols(x, y)
                rows.append(
                    {
                        "feature": feature,
                        "theme": theme_by_feature[feature],
                        "horizon": horizon,
                        "feature_eom": pd.Timestamp(feature_eom).date().isoformat(),
                        "target_label_eom": target_label_eom.date().isoformat(),
                        "return_eom": return_eom.date().isoformat(),
                        "n_valid": n_valid,
                        "rank_ic": rank_ic,
                        "pearson_ic": pearson_ic,
                        "fmb_intercept": intercept,
                        "fmb_slope": slope,
                    }
                )
    result = pd.DataFrame(rows)
    for column in ["rank_ic", "pearson_ic"]:
        finite = result[column].dropna()
        if not finite.between(-1.0 - 1.0e-12, 1.0 + 1.0e-12).all():
            raise RuntimeError(f"Invalid correlation bounds in {column}")
    if int(result["n_valid"].min()) < MIN_MONTHLY_OBSERVATIONS:
        raise RuntimeError("A monthly horizon regression has inadequate cross-section size")
    return result


def _load_existing_monthly_statistics() -> pd.DataFrame:
    rank_path = OUTPUT_ROOT / "monthly_rank_ic.csv"
    pearson_path = OUTPUT_ROOT / "monthly_pearson_ic.csv"
    fmb_path = OUTPUT_ROOT / "monthly_fmb_slopes.csv"
    for path in (rank_path, pearson_path, fmb_path):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Cannot reuse missing/empty monthly statistic file: {path}")
    keys = [
        "feature",
        "theme",
        "horizon",
        "feature_eom",
        "target_label_eom",
        "return_eom",
        "n_valid",
    ]
    rank = pd.read_csv(rank_path)
    pearson = pd.read_csv(pearson_path)
    fmb = pd.read_csv(fmb_path)
    monthly = rank.merge(
        pearson[keys + ["pearson_ic"]], on=keys, validate="one_to_one"
    )
    monthly = monthly.merge(
        fmb[keys + ["fmb_intercept", "fmb_slope"]],
        on=keys,
        validate="one_to_one",
    )
    expected_rows = 10 * sum(241 - horizon for horizon in HORIZONS)
    if len(monthly) != expected_rows:
        raise RuntimeError(
            f"Reused monthly statistics have {len(monthly)} rows, expected {expected_rows}"
        )
    if not monthly.groupby(["feature", "horizon"]).size().between(229, 240).all():
        raise RuntimeError("Reused monthly statistics have invalid feature/horizon counts")
    return monthly


def _cross_sectional_ols(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    x_mean = float(x_values.mean())
    y_mean = float(y_values.mean())
    denominator = float(np.sum((x_values - x_mean) ** 2))
    if denominator <= 0:
        return np.nan, np.nan
    slope = float(np.sum((x_values - x_mean) * (y_values - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def _summary_table(monthly: pd.DataFrame, value_column: str, prefix: str) -> pd.DataFrame:
    rows = []
    for (feature, theme, horizon), group in monthly.groupby(
        ["feature", "theme", "horizon"], sort=False
    ):
        values = pd.to_numeric(group[value_column], errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "theme": theme,
                "horizon": int(horizon),
                f"mean_{prefix}": float(values.mean()),
                f"median_{prefix}": float(values.median()),
                f"std_{prefix}": float(values.std(ddof=1)),
                "valid_months": int(len(values)),
                "average_cross_section_n": float(group.loc[values.index, "n_valid"].mean()),
                "minimum_cross_section_n": int(group.loc[values.index, "n_valid"].min()),
                "maximum_cross_section_n": int(group.loc[values.index, "n_valid"].max()),
            }
        )
    return pd.DataFrame(rows)


def _fmb_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature, theme, horizon), group in monthly.groupby(
        ["feature", "theme", "horizon"], sort=False
    ):
        values = pd.to_numeric(group["fmb_slope"], errors="coerce").dropna()
        conventional_se = float(values.std(ddof=1) / math.sqrt(len(values)))
        hac_se = _newey_west_mean_se(values.to_numpy(dtype=float), HAC_LAGS)
        mean_slope = float(values.mean())
        rows.append(
            {
                "feature": feature,
                "theme": theme,
                "horizon": int(horizon),
                "mean_fmb_slope": mean_slope,
                "median_fmb_slope": float(values.median()),
                "std_fmb_slope": float(values.std(ddof=1)),
                "valid_months": int(len(values)),
                "average_cross_section_n": float(group.loc[values.index, "n_valid"].mean()),
                "conventional_time_series_se": conventional_se,
                "conventional_t_stat": mean_slope / conventional_se if conventional_se > 0 else np.nan,
                "newey_west_hac_lags": HAC_LAGS,
                "newey_west_hac_se": hac_se,
                "newey_west_hac_t_stat": mean_slope / hac_se if hac_se > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _newey_west_mean_se(values: np.ndarray, max_lag: int) -> float:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan
    centered = values - values.mean()
    lag = min(max_lag, n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        autocovariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run_variance += 2.0 * weight * autocovariance
    return float(math.sqrt(max(long_run_variance, 0.0) / n))


def _reconcile_horizon_one(
    rank_summary: pd.DataFrame,
    pearson_summary: pd.DataFrame,
    features: list[str],
) -> dict:
    existing = pd.read_csv(PRIOR_ROOT / "feature_ic.csv")
    if sorted(existing["feature"].astype(str)) != sorted(features):
        raise RuntimeError("Previous validated one-month IC feature set differs")
    check = existing[["feature", "mean_rank_ic", "mean_ic"]].merge(
        rank_summary.loc[
            rank_summary["horizon"].eq(1), ["feature", "mean_rank_ic"]
        ].rename(columns={"mean_rank_ic": "new_mean_rank_ic"}),
        on="feature",
        validate="one_to_one",
    )
    check = check.merge(
        pearson_summary.loc[
            pearson_summary["horizon"].eq(1), ["feature", "mean_pearson_ic"]
        ].rename(columns={"mean_pearson_ic": "new_mean_pearson_ic"}),
        on="feature",
        validate="one_to_one",
    )
    rank_error = float((check["mean_rank_ic"] - check["new_mean_rank_ic"]).abs().max())
    pearson_error = float((check["mean_ic"] - check["new_mean_pearson_ic"]).abs().max())
    tolerance = 1.0e-12
    return {
        "passed": bool(rank_error <= tolerance and pearson_error <= tolerance),
        "tolerance": tolerance,
        "maximum_absolute_rank_ic_error": rank_error,
        "maximum_absolute_pearson_ic_error": pearson_error,
        "reference": _relative(PRIOR_ROOT / "feature_ic.csv"),
    }


def _extract_existing_efi(features: list[str]) -> pd.DataFrame:
    source = pd.read_csv(EFI_PATH)
    subset = source[
        source["method"].isin([PRIMARY_METHOD, SECONDARY_METHOD])
        & source["item_type"].eq("feature")
        & source["scope"].eq("full")
    ].copy()
    if len(subset) != 2 * len(features):
        raise RuntimeError("Authoritative EFI does not contain exactly 20 full-sample rows")
    if sorted(subset["item_name"].unique().tolist()) != sorted(features):
        raise RuntimeError("Authoritative EFI feature set is not Core10")
    if not subset["difference_definition"].eq("full_minus_neutralized").all():
        raise RuntimeError("Unexpected EFI difference definition")
    result = subset[
        [
            "method",
            "method_label",
            "item_name",
            "difference_definition",
            "delta_utility",
            "full_annualized_ex_ante_utility_flow",
            "neutralized_annualized_ex_ante_utility_flow",
            "utility_importance_rank",
        ]
    ].rename(columns={"item_name": "feature"})
    result["portfolio_efi_metric"] = EFI_METRIC
    result["source_efi_file"] = _relative(EFI_PATH)
    return result.sort_values(["method", "feature"]).reset_index(drop=True)


def _extract_characteristic_persistence(features: list[str]) -> pd.DataFrame:
    source = pd.read_csv(PRIOR_MASTER_PATH)
    if sorted(source["feature"].astype(str)) != sorted(features):
        raise RuntimeError("Previous characteristic-persistence master is not Core10")
    return source[
        [
            "feature",
            "theme",
            "spearman_persistence_mean",
            "ar1_rho",
            "half_life_months",
            "pearson_persistence_mean",
            "rank_missing_fill_rate",
        ]
    ].rename(
        columns={
            "spearman_persistence_mean": "characteristic_spearman_persistence",
            "ar1_rho": "characteristic_ar1",
            "half_life_months": "characteristic_half_life_months",
            "pearson_persistence_mean": "characteristic_pearson_persistence",
        }
    )


def _alpha_persistence_metrics(
    fmb_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    features: list[str],
    theme_by_feature: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for feature in features:
        fmb_path = (
            fmb_summary[fmb_summary["feature"].eq(feature)]
            .set_index("horizon")["mean_fmb_slope"]
            .reindex(HORIZONS)
        )
        rank_path = (
            rank_summary[rank_summary["feature"].eq(feature)]
            .set_index("horizon")["mean_rank_ic"]
            .reindex(HORIZONS)
        )
        fmb_values = fmb_path.to_numpy(dtype=float)
        rank_values = rank_path.to_numpy(dtype=float)
        if not np.isfinite(fmb_values).all() or not np.isfinite(rank_values).all():
            raise RuntimeError(f"Incomplete alpha path for {feature}")

        fmb_info = _path_metrics(fmb_values, "fmb")
        rank_info = _path_metrics(rank_values, "rankic")
        row = {
            "feature": feature,
            "theme": theme_by_feature[feature],
            **fmb_info,
            **rank_info,
        }
        for horizon, value in zip(HORIZONS, fmb_values):
            row[f"fmb_slope_h{horizon:02d}"] = float(value)
            row[f"fmb_sign_adjusted_h{horizon:02d}"] = (
                float(fmb_info["fmb_initial_direction"] * value)
                if not fmb_info["fmb_near_zero_h1_flag"]
                else np.nan
            )
        for horizon, value in zip(HORIZONS, rank_values):
            row[f"rank_ic_h{horizon:02d}"] = float(value)
            row[f"rankic_sign_adjusted_h{horizon:02d}"] = (
                float(rank_info["rankic_initial_direction"] * value)
                if not rank_info["rankic_near_zero_h1_flag"]
                else np.nan
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["fmb_sfm_rank_high_to_low"] = result["fmb_sfm"].rank(
        ascending=False, method="min"
    )
    result["fmb_sdm_rank_high_to_low"] = result["fmb_sdm"].rank(
        ascending=False, method="min"
    )
    result["rankic_sfm_rank_high_to_low"] = result["rankic_sfm"].rank(
        ascending=False, method="min"
    )
    result["rankic_sdm_rank_high_to_low"] = result["rankic_sdm"].rank(
        ascending=False, method="min"
    )
    result["h1_fmb_rankic_direction_agreement"] = (
        np.sign(result["fmb_h1"]) == np.sign(result["rankic_h1"])
    )
    return result


def _path_metrics(values: np.ndarray, prefix: str) -> dict:
    h1 = float(values[0])
    near_zero = bool(abs(h1) <= NEAR_ZERO_TOLERANCE)
    direction = float(np.sign(h1)) if not near_zero else 0.0
    adjusted = direction * values if not near_zero else np.full_like(values, np.nan)
    later = adjusted[1:]
    if near_zero:
        return {
            f"{prefix}_h1": h1,
            f"{prefix}_near_zero_h1_flag": True,
            f"{prefix}_initial_direction": 0,
            f"{prefix}_sdm": np.nan,
            f"{prefix}_sfm": np.nan,
            f"{prefix}_ssh": np.nan,
            f"{prefix}_apr_3": np.nan,
            f"{prefix}_apr_6": np.nan,
            f"{prefix}_apr_12": np.nan,
            f"{prefix}_first_decay_horizon": np.nan,
            f"{prefix}_first_decay_horizon_numeric": np.nan,
            f"{prefix}_nonmonotonic_recovery_flag": False,
            f"{prefix}_reversal_horizon": "not_interpretable",
            f"{prefix}_reversal_horizon_numeric": np.nan,
        }
    threshold = 0.5 * abs(h1)
    decay_candidates = [
        horizon
        for horizon, value in zip(HORIZONS[1:], later)
        if value < threshold
    ]
    first_decay = decay_candidates[0] if decay_candidates else ">12"
    recovery = False
    if decay_candidates:
        first_index = decay_candidates[0] - 1
        recovery = bool(np.any(adjusted[first_index + 1 :] >= threshold))
    reversal_candidates = [
        horizon for horizon, value in zip(HORIZONS[1:], later) if value < 0.0
    ]
    raw_signs = np.sign(values[1:])
    same_sign_share = float(np.mean(raw_signs == np.sign(h1)))
    result = {
        f"{prefix}_h1": h1,
        f"{prefix}_near_zero_h1_flag": False,
        f"{prefix}_initial_direction": int(np.sign(h1)),
        f"{prefix}_sdm": float(np.maximum(later, 0.0).sum()),
        f"{prefix}_sfm": float(later.sum()),
        f"{prefix}_ssh": same_sign_share,
        f"{prefix}_first_decay_horizon": first_decay,
        f"{prefix}_first_decay_horizon_numeric": decay_candidates[0]
        if decay_candidates
        else 13,
        f"{prefix}_nonmonotonic_recovery_flag": recovery,
        f"{prefix}_reversal_horizon": reversal_candidates[0]
        if reversal_candidates
        else ">12",
        f"{prefix}_reversal_horizon_numeric": reversal_candidates[0]
        if reversal_candidates
        else 13,
    }
    for horizon in (3, 6, 12):
        numerator = float(np.maximum(adjusted[1:horizon], 0.0).sum())
        result[f"{prefix}_apr_{horizon}"] = numerator / (
            (horizon - 1) * abs(h1)
        )
    return result


def _build_master(
    features: list[str],
    theme_by_feature: dict[str, str],
    characteristic: pd.DataFrame,
    rank_summary: pd.DataFrame,
    fmb_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    existing_efi: pd.DataFrame,
) -> pd.DataFrame:
    master = pd.DataFrame(
        {"feature": features, "theme": [theme_by_feature[value] for value in features]}
    )
    master = master.merge(
        characteristic.drop(columns=["theme"]), on="feature", validate="one_to_one"
    )
    master = master.merge(metrics.drop(columns=["theme"]), on="feature", validate="one_to_one")
    efi = existing_efi.pivot(index="feature", columns="method", values="delta_utility")
    efi = efi.rename(
        columns={
            PRIMARY_METHOD: "portfolio_efi_linear",
            SECONDARY_METHOD: "portfolio_efi_rf",
        }
    ).reset_index()
    master = master.merge(efi, on="feature", validate="one_to_one")
    master["portfolio_efi_metric"] = EFI_METRIC
    master["characteristic_persistence_source"] = _relative(PRIOR_MASTER_PATH)
    master["portfolio_efi_source"] = _relative(EFI_PATH)
    master["primary_alpha_persistence_metric"] = "fmb_sfm"
    master["characteristic_persistence_rank_high_to_low"] = master[
        "characteristic_spearman_persistence"
    ].rank(ascending=False, method="min")
    return master


def _characteristic_vs_alpha(master: pd.DataFrame) -> pd.DataFrame:
    result = master[
        [
            "feature",
            "theme",
            "characteristic_spearman_persistence",
            "characteristic_ar1",
            "characteristic_half_life_months",
            "fmb_h1",
            "fmb_sfm",
            "fmb_sdm",
            "fmb_apr_12",
            "fmb_first_decay_horizon",
            "fmb_reversal_horizon",
            "rankic_h1",
            "rankic_sfm",
            "rankic_sdm",
            "rankic_apr_12",
            "portfolio_efi_linear",
            "portfolio_efi_rf",
            "characteristic_persistence_rank_high_to_low",
            "fmb_sfm_rank_high_to_low",
        ]
    ].copy()
    char_median = float(result["characteristic_spearman_persistence"].median())
    alpha_median = float(result["fmb_sfm"].median())
    result["characteristic_group"] = np.where(
        result["characteristic_spearman_persistence"] >= char_median,
        "high_characteristic_persistence",
        "low_characteristic_persistence",
    )
    result["alpha_group"] = np.where(
        result["fmb_sfm"] >= alpha_median,
        "high_alpha_persistence",
        "low_alpha_persistence",
    )
    result["persistence_divergence"] = (
        result["characteristic_group"] + " / " + result["alpha_group"]
    )
    result["rank_difference_alpha_minus_characteristic"] = (
        result["characteristic_persistence_rank_high_to_low"]
        - result["fmb_sfm_rank_high_to_low"]
    )
    result["classification_rule"] = (
        "descriptive sample-median split fixed before reading portfolio EFI"
    )
    return result


def _efi_correlations(master: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    for method, target in (
        (METHOD_LABELS[PRIMARY_METHOD], "portfolio_efi_linear"),
        (METHOD_LABELS[SECONDARY_METHOD], "portfolio_efi_rf"),
    ):
        for metric in ALPHA_METRICS:
            for statistic, function in (("pearson", _pearson), ("spearman", _spearman)):
                valid = master[[metric, target]].dropna()
                correlation = function(valid[metric], valid[target])
                rows.append(
                    {
                        "relationship": "predictive_alpha_persistence_vs_portfolio_efi",
                        "efi_method": method,
                        "alpha_persistence_metric": metric,
                        "statistic": statistic,
                        "correlation": correlation,
                        "n_features": int(len(valid)),
                        "descriptive_two_sided_p_value": _correlation_p_value(
                            correlation, len(valid)
                        ),
                        "calculation_source": "calculated_in_current_post_hoc_extension",
                        "interpretation": "descriptive only; N=10",
                    }
                )

    prior = pd.read_csv(PRIOR_CORRELATION_PATH)
    prior_rows = prior[
        prior["relationship"].eq("persistence_vs_portfolio_efi")
        & prior["statistic"].eq("spearman")
    ]
    for method in METHOD_LABELS.values():
        match = prior_rows[prior_rows["specification"].eq(method)]
        if len(match) != 1:
            raise RuntimeError(f"Missing prior persistence correlation for {method}")
        source_row = match.iloc[0]
        rows.append(
            {
                "relationship": "characteristic_persistence_vs_portfolio_efi",
                "efi_method": method,
                "alpha_persistence_metric": "characteristic_spearman_persistence",
                "statistic": "spearman",
                "correlation": float(source_row["correlation"]),
                "n_features": int(source_row["n_features"]),
                "descriptive_two_sided_p_value": float(
                    source_row["descriptive_two_sided_p_value"]
                ),
                "calculation_source": f"reused_exactly_from_{_relative(PRIOR_CORRELATION_PATH)}",
                "interpretation": "previous locked comparator; not recomputed",
            }
        )
    result = pd.DataFrame(rows)
    prior_linear = float(
        result.loc[
            result["relationship"].eq("characteristic_persistence_vs_portfolio_efi")
            & result["efi_method"].eq(METHOD_LABELS[PRIMARY_METHOD]),
            "correlation",
        ].iloc[0]
    )
    return result, prior_linear


def _loo_sensitivity(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for method, target in (
        (METHOD_LABELS[PRIMARY_METHOD], "portfolio_efi_linear"),
        (METHOD_LABELS[SECONDARY_METHOD], "portfolio_efi_rf"),
    ):
        for metric in ALPHA_METRICS:
            valid = master[["feature", metric, target]].dropna()
            full = _spearman(valid[metric], valid[target])
            metric_rows = []
            for omitted in valid["feature"]:
                subset = valid[valid["feature"].ne(omitted)]
                loo = _spearman(subset[metric], subset[target])
                row = {
                    "efi_method": method,
                    "alpha_persistence_metric": metric,
                    "omitted_feature": omitted,
                    "n_features": int(len(subset)),
                    "full_sample_spearman": full,
                    "leave_one_out_spearman": loo,
                    "absolute_change_from_full": abs(loo - full),
                }
                details.append(row)
                metric_rows.append(row)
            frame = pd.DataFrame(metric_rows)
            influential = frame.sort_values(
                ["absolute_change_from_full", "omitted_feature"], ascending=[False, True]
            ).iloc[0]
            full_sign = int(np.sign(full))
            loo_signs = np.sign(frame["leave_one_out_spearman"].to_numpy(float)).astype(int)
            sign_stability_fraction = float(np.mean(loo_signs == full_sign))
            summaries.append(
                {
                    "efi_method": method,
                    "alpha_persistence_metric": metric,
                    "full_sample_spearman": full,
                    "loo_minimum": float(frame["leave_one_out_spearman"].min()),
                    "loo_maximum": float(frame["leave_one_out_spearman"].max()),
                    "loo_median": float(frame["leave_one_out_spearman"].median()),
                    "loo_sign_stability_fraction": sign_stability_fraction,
                    "loo_sign_stable_all": bool(sign_stability_fraction == 1.0),
                    "most_influential_feature": str(influential["omitted_feature"]),
                    "most_influential_absolute_change": float(
                        influential["absolute_change_from_full"]
                    ),
                    "momentum_omission_spearman": float(
                        frame.loc[
                            frame["omitted_feature"].eq("momentum_12_1"),
                            "leave_one_out_spearman",
                        ].iloc[0]
                    ),
                    "n_full_features": int(len(valid)),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(details)


def _feature_diagnostics(master: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    existing = pd.read_csv(EXISTING_DIAGNOSTICS_PATH)
    if sorted(existing["feature"].astype(str)) != sorted(features):
        raise RuntimeError("Existing feature diagnostics are not Core10")
    selected = existing[
        [
            "feature",
            "theme",
            "oos_raw_missing_rate",
            "oos_rank_missing_fill_rate",
            "oos_rank_dispersion_mean",
            "monthly_rank_autocorrelation",
            "average_rank_change_turnover_proxy",
            "pre_oos_mean_rank_ic",
            "oos_mean_rank_ic_descriptive",
            "average_abs_corr_with_other_core10",
            "max_abs_corr_with_other_core10",
        ]
    ]
    extra = master[
        [
            "feature",
            "fmb_h1",
            "fmb_sfm",
            "fmb_sdm",
            "fmb_apr_12",
            "fmb_first_decay_horizon",
            "fmb_reversal_horizon",
            "rankic_h1",
            "rankic_sfm",
            "portfolio_efi_linear",
            "portfolio_efi_rf",
        ]
    ]
    result = selected.merge(extra, on="feature", validate="one_to_one")
    result["diagnostic_use"] = (
        "descriptive only; not used to select, remove, or modify Core10 features"
    )
    return result


def _classify_evidence(
    master: pd.DataFrame,
    correlations: pd.DataFrame,
    loo_summary: pd.DataFrame,
    prior_linear: float,
) -> dict:
    spearman = correlations[
        correlations["relationship"].eq(
            "predictive_alpha_persistence_vs_portfolio_efi"
        )
        & correlations["statistic"].eq("spearman")
    ]
    primary = spearman[spearman["alpha_persistence_metric"].eq("fmb_sfm")]
    linear_primary = float(
        primary.loc[
            primary["efi_method"].eq(METHOD_LABELS[PRIMARY_METHOD]), "correlation"
        ].iloc[0]
    )
    rf_primary = float(
        primary.loc[
            primary["efi_method"].eq(METHOD_LABELS[SECONDARY_METHOD]), "correlation"
        ].iloc[0]
    )
    positive_share = {
        method: float(
            (group["correlation"] > 0).mean()
        )
        for method, group in spearman.groupby("efi_method")
    }
    material_improvement = bool(linear_primary >= prior_linear + 0.20)
    metric_consistency = bool(
        linear_primary > 0
        and rf_primary > 0
        and positive_share[METHOD_LABELS[PRIMARY_METHOD]] >= 0.60
        and positive_share[METHOD_LABELS[SECONDARY_METHOD]] >= 0.60
    )
    primary_loo = loo_summary[
        loo_summary["alpha_persistence_metric"].eq("fmb_sfm")
    ]
    loo_stability = bool(
        primary_loo["loo_sign_stable_all"].all()
        and (primary_loo["most_influential_absolute_change"] <= 0.35).all()
    )
    momentum = master.set_index("feature").loc["momentum_12_1"]
    reversal = master.set_index("feature").loc["reversal_1m"]
    momentum_coherent = bool(
        momentum["fmb_sfm"] > 0
        and str(momentum["fmb_reversal_horizon"]) == ">12"
        and momentum["portfolio_efi_linear"] > 0
    )
    reversal_coherent = bool(
        str(reversal["fmb_first_decay_horizon"]) == "2"
        and not bool(reversal["fmb_nonmonotonic_recovery_flag"])
        and float(reversal["fmb_apr_12"]) < 0.50
        and bool(reversal["h1_fmb_rankic_direction_agreement"])
        and reversal["portfolio_efi_linear"] <= 0
    )
    economic_coherence = bool(momentum_coherent and reversal_coherent)
    case_a = bool(
        material_improvement and metric_consistency and loo_stability and economic_coherence
    )

    if case_a:
        if (
            linear_primary >= 0.50
            and rf_primary >= 0.50
            and min(positive_share.values()) >= 0.80
        ):
            classification = "strongly consistent"
        else:
            classification = "moderately consistent"
    elif (
        linear_primary > 0
        and rf_primary > 0
        and min(positive_share.values()) >= 0.50
        and economic_coherence
    ):
        classification = "weakly consistent"
    elif np.sign(linear_primary) != np.sign(rf_primary) or min(
        positive_share.values()
    ) < 0.50:
        classification = "mixed"
    else:
        classification = "inconsistent"

    return {
        "classification": classification,
        "decision_gate": "Case A" if case_a else "Case B",
        "linear_primary_spearman": linear_primary,
        "rf_primary_spearman": rf_primary,
        "prior_characteristic_linear_spearman": prior_linear,
        "linear_positive_metric_fraction": positive_share[
            METHOD_LABELS[PRIMARY_METHOD]
        ],
        "rf_positive_metric_fraction": positive_share[METHOD_LABELS[SECONDARY_METHOD]],
        "material_improvement_rule": "Linear fmb_sfm rho >= prior characteristic rho + 0.20",
        "material_improvement_pass": material_improvement,
        "metric_consistency_rule": (
            "positive primary rho for Linear/RF and >=60% positive Spearman signs "
            "across all 10 prespecified alpha metrics for each model"
        ),
        "metric_consistency_pass": metric_consistency,
        "loo_stability_rule": (
            "fmb_sfm LOO sign stable for Linear/RF and maximum absolute change <=0.35"
        ),
        "loo_stability_pass": loo_stability,
        "economic_coherence_rule": (
            "momentum fmb_sfm positive with no reversal and Linear EFI positive; "
            "reversal first decay h=2 with no recovery, APR12<0.50, matching h1 "
            "FMB/RankIC directions, and Linear EFI nonpositive"
        ),
        "momentum_coherent": momentum_coherent,
        "reversal_coherent": reversal_coherent,
        "economic_coherence_pass": economic_coherence,
        "case_a_rule": "all four prespecified gates must pass",
    }


def _input_audit(
    source_hashes: dict[str, str],
    panel: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    descriptions = {
        _relative(PANEL_PATH): "locked dynamic Top500 panel and exact rank/return inputs",
        _relative(DATA_CONFIG_PATH): "locked Core10 list and OOS dates",
        _relative(THEME_PATH): "existing non-overlapping Core10 theme map",
        _relative(EFI_PATH): "authoritative fixed-model Linear/RF EFI",
        _relative(EFI_REPORT_PATH): "authoritative EFI interpretation",
        _relative(EXISTING_DIAGNOSTICS_PATH): "validated one-month feature diagnostics",
        _relative(MECHANISM_REPORT_PATH): "existing mechanism evidence only",
        _relative(TIMING_VALIDATION_PATH): "locked label alignment validation",
        _relative(PRIOR_MASTER_PATH): "characteristic persistence reused exactly",
        _relative(PRIOR_CORRELATION_PATH): "previous EFI correlation reused exactly",
        _relative(PRIOR_REPORT_PATH): "previous extension narrative context",
    }
    rows = []
    for source, digest in source_hashes.items():
        rows.append(
            {
                "source_file": source,
                "purpose": descriptions[source],
                "sha256": digest,
                "read_only_reuse": True,
            }
        )
    rows.extend(
        [
            {
                "source_file": "panel_integrity",
                "purpose": (
                    f"{len(panel):,} rows, {panel['eom'].nunique()} months, "
                    "500 locked Top500 rows per month"
                ),
                "sha256": "not_applicable",
                "read_only_reuse": True,
            },
            {
                "source_file": "oos_integrity",
                "purpose": (
                    f"{len(oos):,} feature rows, {oos['eom'].nunique()} months, "
                    f"Core10={','.join(features)}"
                ),
                "sha256": "not_applicable",
                "read_only_reuse": True,
            },
        ]
    )
    return pd.DataFrame(rows)


def _workbook_readme(features: list[str], evidence: dict) -> pd.DataFrame:
    rows = [
        ("Scope", "Top500/Core10 post-hoc explanatory analysis only"),
        ("Model retraining", "None; no model, risk, cost, hyperparameter, AUM, or checkpoint was changed"),
        ("EFI", f"Reused locked Section 7 EFI exactly; metric is {EFI_METRIC}"),
        ("Primary OOS feature period", "2005-01-31 through 2024-12-31"),
        ("Horizon definition", "feature at t joins to ret_exc_lead1m at t+h-1, which is incremental return r_(t+h)"),
        ("Primary target", "Single-month future excess return; no cumulative-return target is used"),
        ("Characteristic versus alpha", "Characteristic persistence is state continuity; alpha persistence is future return predictability"),
        ("Primary alpha metric", "FMB signed future predictive mass (fmb_sfm), horizons 2-12"),
        ("Secondary alpha metric", "FMB same-direction predictive mass (fmb_sdm)"),
        ("Robustness", "Rank-IC SFM/SDM and APR at 3, 6, and 12 months"),
        ("FMB inference", f"Monthly univariate cross-sectional slopes; HAC lag={HAC_LAGS} is descriptive"),
        ("Near-zero rule", f"abs(h1) <= {NEAR_ZERO_TOLERANCE:g} return fraction per rank unit is not sign-interpreted"),
        ("Cross-feature N", "Only 10 features; EFI relations are descriptive, not causal"),
        ("Core10", ", ".join(features)),
        ("Evidence classification", evidence["classification"]),
        ("Decision gate", evidence["decision_gate"]),
        ("Cumulative appendix", "Not produced; cumulative returns mechanically retain early-horizon alpha"),
    ]
    return pd.DataFrame(rows, columns=["item", "detail"])


def _timing_audit_markdown(
    timing: pd.DataFrame,
    samples: pd.DataFrame,
    hashes: dict[str, str],
    h1_reconciliation: dict,
    features: list[str],
) -> str:
    timing_display = timing[
        [
            "horizon",
            "example_feature_eom",
            "example_target_label_eom",
            "example_return_eom",
            "target_type",
        ]
    ]
    sample_display = samples[
        [
            "horizon",
            "feature_eom_start",
            "feature_eom_end",
            "target_label_eom_end",
            "return_eom_end",
            "valid_months",
            "valid_stock_months",
            "average_cross_section_n",
            "minimum_cross_section_n",
        ]
    ].copy()
    sample_display["average_cross_section_n"] = sample_display[
        "average_cross_section_n"
    ].round(2)
    hash_rows = pd.DataFrame(
        [{"source_file": key, "sha256": value} for key, value in hashes.items()]
    )
    return "\n".join(
        [
            "# Core10 Predictive Alpha Decay: Timing and Input Audit",
            "",
            "## Gate result",
            "",
            "**PASS.** The locked label validation confirms that `ret_exc_lead1m` on row `u` aligns with the return from `u` to the following month. Every horizon uses an explicit `permno` and date-key join before statistics are calculated.",
            "",
            "## Exact target construction",
            "",
            "For feature rank observed at month-end `t` and horizon `h`, the script creates `target_label_eom = t + (h-1) month-ends`, joins the same `permno` to `ret_exc_lead1m` on that row, and labels the resulting incremental one-month excess return as `r_(t+h)`. It does not use a blind row shift, a contemporaneous return, or a cumulative return.",
            "",
            _markdown_table(timing_display),
            "",
            "## Locked sample and missing future observations",
            "",
            f"- Feature panel: `{_relative(PANEL_PATH)}`.",
            f"- Exact features: {', '.join(features)}.",
            "- Universe: the locked dynamic Top500 rows only; no universe is reconstructed.",
            "- Missing or unavailable future labels, including a stock absent from the locked future Top500 row set, are excluded. No return is imputed and no external delisting return is added.",
            "- Because future stock rows must be present in the locked panel, cross-sectional overlap declines with horizon. This inherited availability condition is disclosed rather than repaired.",
            "- The final feature month shortens by one month for every additional horizon, while the latest label row remains 2024-12-31.",
            "",
            _markdown_table(sample_display),
            "",
            "## Horizon-1 reconciliation",
            "",
            f"The explicit h=1 join reproduces the previous validated one-month diagnostics. Maximum absolute Rank-IC error is `{h1_reconciliation['maximum_absolute_rank_ic_error']:.3e}` and Pearson-IC error is `{h1_reconciliation['maximum_absolute_pearson_ic_error']:.3e}`, versus tolerance `{h1_reconciliation['tolerance']:.1e}`.",
            "",
            "## Return semantics evidence",
            "",
            f"Repository validation source: `{_relative(TIMING_VALIDATION_PATH)}`. It reports weak same-month alignment and strong next-month alignment for the JKP lead label. The thesis code explicitly states that the label is not shifted or reconstructed.",
            "",
            "## Source hashes",
            "",
            _markdown_table(hash_rows),
            "",
            "## Model-integrity declaration",
            "",
            "No Portfolio-ML, RF, Static, return-prediction, risk, transaction-cost, frontier, AUM, feature-selection, or checkpoint computation is invoked by this script. Existing EFI and characteristic-persistence outputs are read-only inputs.",
            "",
        ]
    )


def _analysis_report(
    master: pd.DataFrame,
    correlations: pd.DataFrame,
    loo: pd.DataFrame,
    evidence: dict,
    samples: pd.DataFrame,
    prior_linear: float,
) -> str:
    ranking = master.sort_values("fmb_sfm", ascending=False)[
        [
            "feature",
            "theme",
            "fmb_h1",
            "fmb_sfm",
            "fmb_sdm",
            "fmb_ssh",
            "fmb_apr_12",
            "fmb_first_decay_horizon",
            "fmb_nonmonotonic_recovery_flag",
            "fmb_reversal_horizon",
            "h1_fmb_rankic_direction_agreement",
            "rankic_sfm",
            "portfolio_efi_linear",
            "portfolio_efi_rf",
        ]
    ].copy()
    for column in ["fmb_h1", "fmb_sfm", "fmb_sdm", "fmb_apr_12", "rankic_sfm"]:
        ranking[column] = ranking[column].map(lambda value: f"{value:.6f}")
    ranking["fmb_ssh"] = ranking["fmb_ssh"].map(lambda value: f"{value:.3f}")

    corr_table = correlations[
        correlations["relationship"].eq(
            "predictive_alpha_persistence_vs_portfolio_efi"
        )
    ][
        [
            "efi_method",
            "alpha_persistence_metric",
            "statistic",
            "correlation",
            "n_features",
        ]
    ].copy()
    corr_table["correlation"] = corr_table["correlation"].map(
        lambda value: f"{value:.3f}"
    )
    primary_loo = loo[loo["alpha_persistence_metric"].eq("fmb_sfm")].copy()
    for column in [
        "full_sample_spearman",
        "loo_minimum",
        "loo_maximum",
        "loo_median",
        "loo_sign_stability_fraction",
        "most_influential_absolute_change",
        "momentum_omission_spearman",
    ]:
        primary_loo[column] = primary_loo[column].map(lambda value: f"{value:.3f}")

    indexed = master.set_index("feature")
    momentum = indexed.loc["momentum_12_1"]
    reversal = indexed.loc["reversal_1m"]
    slow = indexed.loc[["book_to_market", "adv_6m", "log_mcap"]]
    momentum_alpha_rank = int(momentum["fmb_sfm_rank_high_to_low"])
    momentum_char_rank = int(momentum["characteristic_persistence_rank_high_to_low"])
    momentum_resolution = (
        "Alpha persistence materially raises momentum's relative rank"
        if momentum_char_rank - momentum_alpha_rank >= 3
        else "Alpha persistence does not materially raise momentum's relative rank"
    )
    reversal_short = bool(
        str(reversal["fmb_first_decay_horizon"]) == "2"
        and not bool(reversal["fmb_nonmonotonic_recovery_flag"])
        and float(reversal["fmb_apr_12"]) < 0.50
        and bool(reversal["h1_fmb_rankic_direction_agreement"])
    )
    rapid = master.sort_values(
        ["fmb_first_decay_horizon_numeric", "fmb_sfm"], ascending=[True, True]
    )
    persistent_names = ", ".join(master.nlargest(3, "fmb_sfm")["feature"])
    fast_names = ", ".join(rapid.head(3)["feature"])
    slow_rows = slow.reset_index()[
        [
            "feature",
            "characteristic_spearman_persistence",
            "fmb_sfm",
            "fmb_first_decay_horizon",
            "portfolio_efi_linear",
        ]
    ].copy()
    for column in [
        "characteristic_spearman_persistence",
        "fmb_sfm",
        "portfolio_efi_linear",
    ]:
        slow_rows[column] = slow_rows[column].map(lambda value: f"{value:.6f}")

    if evidence["decision_gate"] == "Case A":
        decision = (
            "The prespecified Case A gate passes. Predictive-alpha persistence is materially "
            "closer to EFI than raw state persistence, signs are reasonably consistent, the "
            "momentum/reversal diagnostics are coherent, and the primary LOO relation is not "
            "dominated by one feature. A stronger Core10 transaction-cost mechanism test is justified."
        )
    else:
        decision = (
            "The prespecified Case B gate applies. The relation remains weak, model-dependent, "
            "or LOO-sensitive, so further expensive Core10 persistence mining is not recommended. "
            "Portfolio-ML EFI cannot be reduced to either raw characteristic persistence or simple "
            "univariate alpha-decay statistics. The next extension should be the planned controlled "
            "commodity experiment with multi-horizon momentum and a Garleanu-Pedersen-style dynamic benchmark."
        )

    return "\n".join(
        [
            "# Core10 Predictive Alpha Decay and Portfolio-ML Economic Importance",
            "",
            "## 1. Motivation",
            "",
            "The previous extension measured whether a characteristic rank remains similar from one month to the next. That is not the same object as whether today's characteristic predicts incremental excess returns several months ahead: `state persistence != alpha persistence`. Dynamic trading under costs depends on how long the return opportunity survives, not only on how slowly the observed state moves. This post-hoc extension therefore estimates the complete 1-12 month predictive profile without retraining any portfolio model.",
            "",
            "## 2. Timing and methodology",
            "",
            "For every feature month `t` and horizon `h`, the same `permno` is joined explicitly to `ret_exc_lead1m` at label row `t+h-1`. Because that label is the one-month return from its row month to the next month, the target is the incremental return `r_(t+h)`. Cumulative returns are not used. The feature is always taken from `t`; no future feature value enters the predictor.",
            "",
            f"The locked feature sample is 2005-01 through 2024-12. Horizon 1 has {int(samples.loc[samples['horizon'].eq(1), 'valid_months'].iloc[0])} usable months and horizon 12 has {int(samples.loc[samples['horizon'].eq(12), 'valid_months'].iloc[0])}. Missing future locked-panel rows and labels are excluded without imputation.",
            "",
            "Each feature/horizon reports monthly Spearman Rank IC, Pearson IC, and an intercept-inclusive univariate Fama-MacBeth slope using normalized ranks. The primary alpha-persistence metric was fixed in advance as FMB signed future mass, `sum(h=2..12, sign(beta_1)*beta_h)`. Same-direction mass and Rank-IC analogues are robustness measures. HAC lag 6 is descriptive, not the basis for selecting a feature or model.",
            "",
            "## 3. Predictive-alpha decay across Core10",
            "",
            f"The largest signed same-direction alpha mass is observed for {persistent_names}. The earliest first-decay descriptors, ordered by subsequent signed mass, include {fast_names}; this is not a claim of monotonic decay. Negative adjusted values are retained and count against signed future mass, while later recovery is flagged rather than forced into an exponential half-life.",
            "",
            _markdown_table(ranking),
            "",
            "Full monthly and horizon summaries are in `monthly_*` and `*_summary.csv`; Figures 1-3 preserve all signed horizon profiles.",
            "",
            "## 4. Characteristic persistence versus alpha persistence",
            "",
            "Slow-moving states do not mechanically imply long-lived return predictability. `characteristic_vs_alpha.csv` reports both ranks and a transparent sample-median divergence classification. The high-persistence Value and Liquidity/Size examples are shown below without using their portfolio outcomes to define the alpha metric.",
            "",
            _markdown_table(slow_rows),
            "",
            "Figure 4 labels all 10 features and makes the divergence visible. This comparison is descriptive because the dynamic Top500 future-row overlap shrinks with horizon and the FMB paths are univariate.",
            "",
            "## 5. Alpha persistence versus Portfolio-ML EFI",
            "",
            f"The previous locked raw-characteristic result was Spearman rho={prior_linear:.3f} with Linear EFI. The prespecified FMB signed-future-mass relation is rho={evidence['linear_primary_spearman']:.3f} for Linear EFI and rho={evidence['rf_primary_spearman']:.3f} for RF EFI. All Pearson and Spearman results for all 10 prespecified alpha metrics are reported below; none is selected after seeing its correlation.",
            "",
            _markdown_table(corr_table),
            "",
            "The mandatory leave-one-feature-out diagnostic for the primary metric is:",
            "",
            _markdown_table(
                primary_loo[
                    [
                        "efi_method",
                        "full_sample_spearman",
                        "loo_minimum",
                        "loo_maximum",
                        "loo_median",
                        "loo_sign_stability_fraction",
                        "most_influential_feature",
                        "most_influential_absolute_change",
                        "momentum_omission_spearman",
                    ]
                ]
            ),
            "",
            "All other metric/model LOO summaries and the 200 omitted-feature estimates are saved in machine-readable form.",
            "",
            "## 6. Momentum",
            "",
            f"Momentum's characteristic-persistence rank is {momentum_char_rank}/10, while its FMB signed-future-mass rank is {momentum_alpha_rank}/10. Its h=1 FMB slope is {momentum['fmb_h1']:.6f}, future signed mass is {momentum['fmb_sfm']:.6f}, first-decay horizon is {momentum['fmb_first_decay_horizon']}, and reversal horizon is {momentum['fmb_reversal_horizon']}. {momentum_resolution}. Figure 6 shows the full sign-adjusted path, so the conclusion does not rest on a single summary.",
            "",
            "## 7. Reversal",
            "",
            f"Reversal's h=1 FMB slope is {reversal['fmb_h1']:.6f}, while its h=1 Rank IC is {reversal['rankic_h1']:.6f}; their directions {'agree' if reversal['h1_fmb_rankic_direction_agreement'] else 'disagree'}. Future signed FMB mass is {reversal['fmb_sfm']:.6f}, first decay occurs at h={reversal['fmb_first_decay_horizon']}, the path subsequently {'recovers non-monotonically' if reversal['fmb_nonmonotonic_recovery_flag'] else 'does not recover above the half-h1 threshold'}, and its first opposite-direction horizon is {reversal['fmb_reversal_horizon']}. Its Linear EFI is {reversal['portfolio_efi_linear']:.6f}. The stricter shortest-horizon concentration condition is {'met' if reversal_short else 'not met'}, so a first decay at h=2 alone is not interpreted as permanent disappearance.",
            "",
            "## 8. Slow-moving but economically weak characteristics",
            "",
            "Book-to-market, ADV, and market capitalization move slowly as states, but their horizon-specific alpha and EFI need not be large. Figure 7 is included because all three are highly state-persistent and at least two have nonpositive Linear EFI, making the state-versus-alpha contrast empirically relevant. This is interpretation, not feature selection.",
            "",
            "## 9. Relation to Garleanu-Pedersen",
            "",
            f"The evidence is classified as **{evidence['classification']}** under a prespecified four-part gate: material improvement over rho=0.212, cross-metric/model sign consistency, primary-metric LOO stability, and coherent momentum/reversal behavior. Passes are respectively {evidence['material_improvement_pass']}, {evidence['metric_consistency_pass']}, {evidence['loo_stability_pass']}, and {evidence['economic_coherence_pass']}. This classification does not assert that univariate FMB slopes are the structural expected-return state in a Garleanu-Pedersen model.",
            "",
            "## 10. Limitations",
            "",
            "- There are only 10 feature-level observations; cross-feature correlations are descriptive.",
            "- Univariate alpha paths omit Core10 interactions and conditional nonlinearities.",
            "- Fama-MacBeth slopes are not Portfolio-ML's internal conditional expected returns.",
            "- Coefficients and market regimes may change through time.",
            "- Fixed-model neutralization EFI includes substitution and interaction effects among correlated inputs.",
            "- This is post-hoc explanatory analysis, not a preregistered causal test.",
            "- Multi-horizon predictive relationships are descriptive, not structural alpha dynamics.",
            "- Future-return availability is inherited from the locked dynamic Top500 panel; no missing exit path is reconstructed.",
            "",
            "## 11. Decision gate for the next extension",
            "",
            f"**{evidence['decision_gate']}.** {decision}",
            "",
            "No original model, risk input, cost model, hyperparameter, frontier, AUM scenario, feature set, or checkpoint was retrained or modified.",
            "",
        ]
    )


def _make_figures(
    master: pd.DataFrame,
    rank_summary: pd.DataFrame,
    fmb_summary: pd.DataFrame,
    themes: dict,
) -> None:
    feature_colors = {
        feature: LINE_COLORS[index]
        for index, feature in enumerate(master["feature"].tolist())
    }
    _profile_panels(
        rank_summary.rename(columns={"mean_rank_ic": "value"}),
        value_column="value",
        themes=themes,
        feature_colors=feature_colors,
        title="Core10 Multi-Horizon Rank IC Profiles",
        subtitle="Feature at t versus incremental one-month excess return r_(t+h); signed values retained",
        y_label="Mean monthly Rank IC",
        filename="01_multihorizon_rank_ic_profiles",
        decimals=3,
    )
    _profile_panels(
        fmb_summary.rename(columns={"mean_fmb_slope": "value"}),
        value_column="value",
        themes=themes,
        feature_colors=feature_colors,
        title="Core10 Multi-Horizon Fama-MacBeth Alpha Profiles",
        subtitle="Univariate monthly slopes on normalized ranks; actual signed coefficients",
        y_label="Mean FMB slope",
        filename="02_multihorizon_fmb_alpha_profiles",
        decimals=4,
    )
    adjusted_rows = []
    for row in master.itertuples(index=False):
        for horizon in HORIZONS:
            adjusted_rows.append(
                {
                    "feature": row.feature,
                    "theme": row.theme,
                    "horizon": horizon,
                    "value": getattr(row, f"fmb_sign_adjusted_h{horizon:02d}"),
                }
            )
    adjusted = pd.DataFrame(adjusted_rows)
    _profile_panels(
        adjusted,
        value_column="value",
        themes=themes,
        feature_colors=feature_colors,
        title="Core10 Sign-Adjusted Predictive-Alpha Decay",
        subtitle="Positive means the h=1 predictive direction survives; negative means reversal",
        y_label="Sign-adjusted mean FMB slope",
        filename="03_sign_adjusted_alpha_decay_profiles",
        decimals=4,
    )
    _scatter(
        master,
        x_column="characteristic_spearman_persistence",
        y_column="fmb_sfm",
        title="Characteristic Persistence versus Predictive-Alpha Persistence",
        subtitle="State continuity does not mechanically imply signed future return-predictive mass",
        x_label="Mean monthly characteristic Spearman persistence",
        y_label="FMB signed future predictive mass (h=2..12)",
        filename="04_characteristic_vs_alpha_persistence",
        color=COLORS["green"],
        y_decimals=4,
    )
    primary_rho = _spearman(master["fmb_sfm"], master["portfolio_efi_linear"])
    _scatter(
        master,
        x_column="fmb_sfm",
        y_column="portfolio_efi_linear",
        title="Predictive-Alpha Persistence and Portfolio-ML Economic Importance",
        subtitle=f"Portfolio-ML Linear fixed-model EFI; Spearman rho={primary_rho:.3f}, N=10",
        x_label="FMB signed future predictive mass (h=2..12)",
        y_label="Annualized utility-flow EFI",
        filename="05_alpha_persistence_vs_linear_efi",
        color=COLORS["blue"],
        y_decimals=4,
    )
    _line_comparison(
        adjusted[adjusted["feature"].isin(["momentum_12_1", "reversal_1m"])],
        features=["momentum_12_1", "reversal_1m"],
        title="Momentum and Reversal: Sign-Adjusted Alpha Paths",
        subtitle="Incremental one-month future returns; zero line marks loss of original direction",
        filename="06_momentum_reversal_alpha_comparison",
        colors=[COLORS["blue"], COLORS["red"]],
    )
    slow_names = ["book_to_market", "adv_6m", "log_mcap"]
    slow = master[master["feature"].isin(slow_names)]
    informative = bool(
        (slow["characteristic_spearman_persistence"] > 0.95).all()
        and (slow["portfolio_efi_linear"] <= 0).sum() >= 2
    )
    if informative:
        _line_comparison(
            adjusted[adjusted["feature"].isin(slow_names)],
            features=slow_names,
            title="Slow Characteristics and Their Predictive-Alpha Paths",
            subtitle="Highly persistent states with weak or negative Portfolio-ML Linear EFI",
            filename="07_slow_characteristic_weak_efi_examples",
            colors=[COLORS["gold"], COLORS["teal"], COLORS["purple"]],
        )


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    font_root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    return {
        "title": ImageFont.truetype(str(font_root / "arialbd.ttf"), 38),
        "subtitle": ImageFont.truetype(str(font_root / "arial.ttf"), 22),
        "panel": ImageFont.truetype(str(font_root / "arialbd.ttf"), 21),
        "axis": ImageFont.truetype(str(font_root / "arial.ttf"), 20),
        "tick": ImageFont.truetype(str(font_root / "arial.ttf"), 16),
        "label": ImageFont.truetype(str(font_root / "arial.ttf"), 17),
        "small": ImageFont.truetype(str(font_root / "arial.ttf"), 14),
    }


def _profile_panels(
    frame: pd.DataFrame,
    *,
    value_column: str,
    themes: dict,
    feature_colors: dict[str, str],
    title: str,
    subtitle: str,
    y_label: str,
    filename: str,
    decimals: int,
) -> None:
    width, height = 1900, 1780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((70, 45), title, font=fonts["title"], fill=COLORS["navy"])
    draw.text((70, 100), subtitle, font=fonts["subtitle"], fill=COLORS["gray"])
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(float)
    y_min, y_max = _padded_limits(values, include_zero=True)
    panel_left, panel_top = 85, 170
    panel_width, panel_height = 850, 360
    x_gap, y_gap = 75, 40
    for index, (theme, members) in enumerate(themes.items()):
        row = index // 2
        column = index % 2
        left = panel_left + column * (panel_width + x_gap)
        top = panel_top + row * (panel_height + y_gap)
        right = left + panel_width
        bottom = top + panel_height
        plot_left, plot_right = left + 85, right - 25
        plot_top, plot_bottom = top + 48, bottom - 55
        draw.text((left + 5, top + 3), theme, font=fonts["panel"], fill=COLORS["navy"])
        for tick in np.linspace(y_min, y_max, 5):
            y = _scale(float(tick), y_min, y_max, plot_bottom, plot_top)
            draw.line((plot_left, y, plot_right, y), fill=COLORS["grid"], width=1)
            draw.text(
                (left + 4, y - 8),
                f"{tick:.{decimals}f}",
                font=fonts["small"],
                fill=COLORS["gray"],
            )
        zero_y = _scale(0.0, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, zero_y, plot_right, zero_y), fill=COLORS["gray"], width=2)
        for horizon in (1, 3, 6, 9, 12):
            x = _scale(horizon, 1, 12, plot_left, plot_right)
            draw.text((x - 6, plot_bottom + 10), str(horizon), font=fonts["tick"], fill=COLORS["gray"])
        for member_index, feature in enumerate(members):
            group = frame[frame["feature"].eq(feature)].sort_values("horizon")
            points = [
                (
                    _scale(float(item.horizon), 1, 12, plot_left, plot_right),
                    _scale(float(getattr(item, value_column)), y_min, y_max, plot_bottom, plot_top),
                )
                for item in group.itertuples(index=False)
            ]
            color = feature_colors[feature]
            if len(points) > 1:
                draw.line(points, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            legend_x = plot_left + member_index * 245
            draw.line((legend_x, bottom - 24, legend_x + 28, bottom - 24), fill=color, width=4)
            draw.text((legend_x + 35, bottom - 35), feature, font=fonts["small"], fill=COLORS["navy"])
    draw.text((900, 1735), "Horizon h (months)", font=fonts["axis"], fill=COLORS["navy"])
    rotated = Image.new("RGBA", (600, 45), (255, 255, 255, 0))
    ImageDraw.Draw(rotated).text((0, 4), y_label, font=fonts["axis"], fill=COLORS["navy"])
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (5, 610), rotated)
    _save_figure(image, filename)


def _line_comparison(
    frame: pd.DataFrame,
    *,
    features: list[str],
    title: str,
    subtitle: str,
    filename: str,
    colors: list[str],
) -> None:
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((85, 55), title, font=fonts["title"], fill=COLORS["navy"])
    draw.text((85, 110), subtitle, font=fonts["subtitle"], fill=COLORS["gray"])
    left, right, top, bottom = 225, 1690, 195, 855
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(float)
    y_min, y_max = _padded_limits(values, include_zero=True)
    for tick in np.linspace(y_min, y_max, 6):
        y = _scale(float(tick), y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((70, y - 10), f"{tick:.4f}", font=fonts["tick"], fill=COLORS["gray"])
    zero_y = _scale(0.0, y_min, y_max, bottom, top)
    draw.line((left, zero_y, right, zero_y), fill=COLORS["navy"], width=3)
    for horizon in HORIZONS:
        x = _scale(horizon, 1, 12, left, right)
        draw.text((x - 7, bottom + 16), str(horizon), font=fonts["tick"], fill=COLORS["gray"])
    for feature, color in zip(features, colors):
        group = frame[frame["feature"].eq(feature)].sort_values("horizon")
        points = [
            (
                _scale(float(item.horizon), 1, 12, left, right),
                _scale(float(item.value), y_min, y_max, bottom, top),
            )
            for item in group.itertuples(index=False)
        ]
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    legend_x = 520
    for index, (feature, color) in enumerate(zip(features, colors)):
        x = legend_x + index * 380
        draw.line((x, 930, x + 35, 930), fill=color, width=5)
        draw.text((x + 50, 918), feature, font=fonts["axis"], fill=COLORS["navy"])
    draw.text((790, 985), "Horizon h (months)", font=fonts["axis"], fill=COLORS["navy"])
    _save_figure(image, filename)


def _scatter(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    filename: str,
    color: str,
    y_decimals: int,
) -> None:
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((85, 55), title, font=fonts["title"], fill=COLORS["navy"])
    draw.text((85, 110), subtitle, font=fonts["subtitle"], fill=COLORS["gray"])
    left, right, top, bottom = 265, 1690, 195, 930
    x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(float)
    y_values = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(float)
    x_min, x_max = _padded_limits(x_values, include_zero=False)
    y_min, y_max = _padded_limits(y_values, include_zero=True)
    for tick in np.linspace(x_min, x_max, 6):
        x = _scale(float(tick), x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        draw.text((x - 35, bottom + 18), f"{tick:.3f}", font=fonts["tick"], fill=COLORS["gray"])
    for tick in np.linspace(y_min, y_max, 6):
        y = _scale(float(tick), y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((85, y - 10), f"{tick:.{y_decimals}f}", font=fonts["tick"], fill=COLORS["gray"])
    zero_y = _scale(0.0, y_min, y_max, bottom, top)
    draw.line((left, zero_y, right, zero_y), fill=COLORS["gray"], width=2)
    occupied: list[tuple[float, float, float, float]] = []
    for row in frame.itertuples(index=False):
        x_value = float(getattr(row, x_column))
        y_value = float(getattr(row, y_column))
        label = str(getattr(row, "feature"))
        x = _scale(x_value, x_min, x_max, left, right)
        y = _scale(y_value, y_min, y_max, bottom, top)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)
        label_xy, box = _label_position(
            draw, label, x, y, occupied, fonts["label"], left, right, top, bottom
        )
        draw.line((x, y, label_xy[0], label_xy[1] + 8), fill=COLORS["gray"], width=1)
        draw.text(label_xy, label, font=fonts["label"], fill=COLORS["navy"])
        occupied.append(box)
    draw.text((720, 1000), x_label, font=fonts["axis"], fill=COLORS["navy"])
    rotated = Image.new("RGBA", (700, 45), (255, 255, 255, 0))
    ImageDraw.Draw(rotated).text((0, 4), y_label, font=fonts["axis"], fill=COLORS["navy"])
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (15, 335), rotated)
    _save_figure(image, filename)


def _label_position(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: float,
    y: float,
    occupied: list[tuple[float, float, float, float]],
    font: ImageFont.FreeTypeFont,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    text_box = draw.textbbox((0, 0), label, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    candidates = [(14, -28), (14, 10), (-width - 14, -28), (-width - 14, 10), (18, -50), (-width - 18, -50)]
    for dx, dy in candidates:
        x0, y0 = x + dx, y + dy
        box = (x0 - 3, y0 - 2, x0 + width + 3, y0 + height + 5)
        in_bounds = box[0] >= left and box[2] <= right and box[1] >= top and box[3] <= bottom
        overlap = any(_boxes_overlap(box, prior) for prior in occupied)
        if in_bounds and not overlap:
            return (x0, y0), box
    x0 = min(max(x + 14, left), right - width)
    y0 = min(max(y + 10, top), bottom - height)
    return (x0, y0), (x0 - 3, y0 - 2, x0 + width + 3, y0 + height + 5)


def _boxes_overlap(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def _save_figure(image: Image.Image, filename: str) -> None:
    image.save(FIGURE_ROOT / f"{filename}.png", dpi=(180, 180))
    image.convert("RGB").save(
        FIGURE_ROOT / f"{filename}.pdf", "PDF", resolution=180.0
    )


def _padded_limits(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if include_zero:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    span = maximum - minimum
    padding = span * 0.12 if span > 0 else max(abs(maximum) * 0.10, 1.0e-4)
    return minimum - padding, maximum + padding


def _scale(value: float, minimum: float, maximum: float, low: float, high: float) -> float:
    if maximum == minimum:
        return (low + high) / 2.0
    return low + (value - minimum) / (maximum - minimum) * (high - low)


def _validation_summary(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    rank_summary: pd.DataFrame,
    pearson_summary: pd.DataFrame,
    fmb_summary: pd.DataFrame,
    samples: pd.DataFrame,
    h1_reconciliation: dict,
    existing_efi: pd.DataFrame,
    features: list[str],
    runtime_seconds: float,
) -> dict:
    source_efi = existing_efi.pivot(
        index="feature", columns="method", values="delta_utility"
    )
    linear_error = float(
        np.max(
            np.abs(
                master.set_index("feature")["portfolio_efi_linear"]
                - source_efi[PRIMARY_METHOD]
            )
        )
    )
    rf_error = float(
        np.max(
            np.abs(
                master.set_index("feature")["portfolio_efi_rf"]
                - source_efi[SECONDARY_METHOD]
            )
        )
    )
    correlation_bounds = bool(
        rank_summary["mean_rank_ic"].between(-1.0, 1.0).all()
        and pearson_summary["mean_pearson_ic"].between(-1.0, 1.0).all()
    )
    months = samples.sort_values("horizon")["valid_months"].to_numpy(int)
    stock_months = samples.sort_values("horizon")["valid_stock_months"].to_numpy(int)
    figure_files = sorted(FIGURE_ROOT.glob("*.png"))
    checks = {
        "locked_panel_exact": bool(
            len(panel) == 210000
            and panel["eom"].nunique() == 420
            and panel.groupby("eom").size().eq(500).all()
        ),
        "exact_core10_feature_list": bool(
            sorted(master["feature"].tolist()) == sorted(features)
            and len(master) == 10
        ),
        "no_final20_contamination": bool(len(features) == 10),
        "horizon_one_reconciliation": bool(h1_reconciliation["passed"]),
        "horizon_month_counts_decline_by_one": bool(np.array_equal(np.diff(months), -np.ones(11, dtype=int))),
        "horizon_stock_month_counts_strictly_decline": bool((np.diff(stock_months) < 0).all()),
        "horizon_one_months": int(months[0]),
        "horizon_twelve_months": int(months[-1]),
        "correlations_within_bounds": correlation_bounds,
        "minimum_fmb_valid_months": int(fmb_summary["valid_months"].min()),
        "minimum_average_cross_section_n": float(
            fmb_summary["average_cross_section_n"].min()
        ),
        "efi_linear_max_absolute_reuse_error": linear_error,
        "efi_rf_max_absolute_reuse_error": rf_error,
        "efi_reused_exactly": bool(linear_error == 0.0 and rf_error == 0.0),
        "explicit_date_join": True,
        "primary_target_incremental_single_month": True,
        "primary_target_cumulative": False,
        "no_future_feature_values": True,
        "no_model_retraining": True,
        "no_risk_model_rerun": True,
        "no_cost_model_rerun": True,
        "no_aum_rerun": True,
        "no_checkpoint_write": True,
        "figure_png_count": len(figure_files),
    }
    passed = bool(
        all(
            [
                checks["locked_panel_exact"],
                checks["exact_core10_feature_list"],
                checks["no_final20_contamination"],
                checks["horizon_one_reconciliation"],
                checks["horizon_month_counts_decline_by_one"],
                checks["horizon_stock_month_counts_strictly_decline"],
                checks["correlations_within_bounds"],
                checks["minimum_fmb_valid_months"] >= 229,
                checks["minimum_average_cross_section_n"] >= MIN_MONTHLY_OBSERVATIONS,
                checks["efi_reused_exactly"],
                checks["figure_png_count"] >= 6,
            ]
        )
    )
    return {
        "passed": passed,
        "checks": checks,
        "horizon_one_reconciliation": h1_reconciliation,
        "runtime_seconds_before_workbook": runtime_seconds,
    }


def _execution_log(
    master: pd.DataFrame,
    correlations: pd.DataFrame,
    loo: pd.DataFrame,
    evidence: dict,
    samples: pd.DataFrame,
    prior_linear: float,
    runtime_seconds: float,
) -> str:
    indexed = master.set_index("feature")
    momentum = indexed.loc["momentum_12_1"]
    reversal = indexed.loc["reversal_1m"]
    persistent = ", ".join(master.nlargest(3, "fmb_sfm")["feature"])
    fastest = ", ".join(
        master.sort_values(
            ["fmb_first_decay_horizon_numeric", "fmb_sfm"], ascending=[True, True]
        ).head(3)["feature"]
    )
    primary_loo = loo[
        loo["alpha_persistence_metric"].eq("fmb_sfm")
        & loo["efi_method"].eq(METHOD_LABELS[PRIMARY_METHOD])
    ].iloc[0]
    h1 = samples[samples["horizon"].eq(1)].iloc[0]
    h12 = samples[samples["horizon"].eq(12)].iloc[0]
    momentum_rank_gain = int(
        momentum["characteristic_persistence_rank_high_to_low"]
        - momentum["fmb_sfm_rank_high_to_low"]
    )
    momentum_statement = (
        f"yes by relative rank (gain {momentum_rank_gain})"
        if momentum_rank_gain >= 3 and momentum["fmb_sfm"] > 0
        else f"no material relative-rank resolution (gain {momentum_rank_gain})"
    )
    reversal_concentrated = bool(
        str(reversal["fmb_first_decay_horizon"]) == "2"
        and not bool(reversal["fmb_nonmonotonic_recovery_flag"])
        and float(reversal["fmb_apr_12"]) < 0.50
        and bool(reversal["h1_fmb_rankic_direction_agreement"])
    )
    reversal_statement = "yes" if reversal_concentrated else "no"
    next_step = (
        "stronger Core10 transaction-cost mechanism testing"
        if evidence["decision_gate"] == "Case A"
        else "controlled commodity plus multi-horizon momentum experiment"
    )
    return "\n".join(
        [
            "CORE10 PREDICTIVE ALPHA DECAY EXECUTION SUMMARY",
            f"1. Exact panel: {_relative(PANEL_PATH)} (locked dynamic Top500).",
            "2. Exact OOS feature period: 2005-01-31 through 2024-12-31.",
            "3. r_(t+h): ret_exc_lead1m on the same-permno label row t+h-1, i.e. the incremental one-month excess return ending at t+h.",
            f"4. Usable months: h=1 {int(h1['valid_months'])} ({int(h1['valid_stock_months']):,} stock-months); h=12 {int(h12['valid_months'])} ({int(h12['valid_stock_months']):,}).",
            f"5. Most persistent same-direction predictive alpha by primary FMB SFM: {persistent}.",
            f"6. Fastest alpha decay by first-decay horizon then SFM: {fastest}.",
            f"7. Momentum more persistent in alpha than raw state suggests: {momentum_statement}; first decay={momentum['fmb_first_decay_horizon']}, reversal={momentum['fmb_reversal_horizon']}.",
            f"8. Reversal concentrated at the shortest horizon: {reversal_statement}; first decay={reversal['fmb_first_decay_horizon']}, nonmonotonic recovery={reversal['fmb_nonmonotonic_recovery_flag']}, first reversal={reversal['fmb_reversal_horizon']}, h1 FMB/RankIC direction agreement={reversal['h1_fmb_rankic_direction_agreement']}, future signed mass={reversal['fmb_sfm']:.6f}.",
            f"9. Primary FMB SFM vs Linear EFI Spearman: {evidence['linear_primary_spearman']:.6f}.",
            f"10. Primary FMB SFM vs RF EFI Spearman: {evidence['rf_primary_spearman']:.6f}.",
            f"11. Previous raw-characteristic persistence vs Linear EFI Spearman: {prior_linear:.6f} (reused, not recomputed).",
            f"12. Linear primary LOO: full={primary_loo['full_sample_spearman']:.6f}, range=[{primary_loo['loo_minimum']:.6f}, {primary_loo['loo_maximum']:.6f}], median={primary_loo['loo_median']:.6f}, sign stable={primary_loo['loo_sign_stable_all']}, most influential={primary_loo['most_influential_feature']}.",
            f"13. Final Garleanu-Pedersen evidence classification: {evidence['classification']}.",
            f"14. Decision gate {evidence['decision_gate']}; recommended next step: {next_step}.",
            "15. No original model, RF mapping, risk model, cost model, AUM scenario, hyperparameter, feature set, or checkpoint was retrained or modified.",
            f"Runtime before workbook: {runtime_seconds:.2f} seconds.",
        ]
    )


def _source_hashes() -> dict[str, str]:
    paths = [
        PANEL_PATH,
        DATA_CONFIG_PATH,
        THEME_PATH,
        EFI_PATH,
        EFI_REPORT_PATH,
        EXISTING_DIAGNOSTICS_PATH,
        MECHANISM_REPORT_PATH,
        TIMING_VALIDATION_PATH,
        PRIOR_MASTER_PATH,
        PRIOR_CORRELATION_PATH,
        PRIOR_REPORT_PATH,
    ]
    return {_relative(path): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _pearson(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _spearman(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return np.nan
    return float(
        aligned.iloc[:, 0]
        .rank(method="average")
        .corr(aligned.iloc[:, 1].rank(method="average"))
    )


def _correlation_p_value(correlation: float, n: int) -> float:
    if not np.isfinite(correlation) or n <= 2:
        return np.nan
    if abs(correlation) >= 1.0:
        return 0.0
    degrees = n - 2
    t_squared = correlation**2 * degrees / (1.0 - correlation**2)
    x = degrees / (degrees + t_squared)
    return _regularized_incomplete_beta(x, degrees / 2.0, 0.5)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise RuntimeError("Incomplete-beta continued fraction did not converge")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                text = ""
            elif isinstance(value, (bool, np.bool_)):
                text = "true" if value else "false"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
