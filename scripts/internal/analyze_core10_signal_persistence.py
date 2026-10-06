from __future__ import annotations

import os

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/extensions/core10_signal_persistence"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
PANEL_PATH = PROJECT_ROOT / "data/processed/us_equity_ml/master_panel.parquet"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs/data_us_equity_ml.yaml"
THEME_PATH = PROJECT_ROOT / "configs/core10_theme_mapping.yaml"
EFI_PATH = PROJECT_ROOT / "reports/final_analysis/feature_importance/feature_importance.csv"
THEME_EFI_PATH = PROJECT_ROOT / "reports/final_analysis/feature_importance/theme_importance.csv"
EXISTING_DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "reports/final_analysis/feature_importance/core10_feature_diagnostics.csv"
)
EFI_REPORT_PATH = (
    PROJECT_ROOT
    / "reports/final_analysis/feature_importance/feature_theme_importance.md"
)
MECHANISM_REPORT_PATH = (
    PROJECT_ROOT / "reports/final_analysis/mechanisms/mechanism_plots.md"
)
MECHANISM_METRICS_PATH = (
    PROJECT_ROOT / "reports/final_analysis/mechanisms/mechanism_metrics.csv"
)
FINAL_INDEX_PATH = PROJECT_ROOT / "reports/final_analysis/final_analysis_index.md"
TIMING_VALIDATION_PATH = (
    PROJECT_ROOT / "reports/data_validation/us_equity_ml_validation.json"
)

OOS_START = pd.Timestamp("2005-01-31")
OOS_END = pd.Timestamp("2024-12-31")
PRIMARY_METHOD = "portfolio_ml_linear"
SECONDARY_METHOD = "portfolio_ml_rf"
METHOD_LABELS = {
    PRIMARY_METHOD: "Portfolio-ML Linear",
    SECONDARY_METHOD: "Portfolio-ML RF",
}
PERSISTENCE_DISAGREEMENT_GAP = 0.10
MIN_MONTHLY_OBSERVATIONS = 20
EFI_METRIC = (
    "annualized ex-ante utility flow difference: full fixed model minus "
    "feature-neutralized fixed model"
)
COLORS = {
    "navy": "#17324D",
    "blue": "#2563A6",
    "red": "#B43C55",
    "green": "#2D7D62",
    "gold": "#B8871B",
    "gray": "#687785",
    "light": "#E8EEF3",
    "grid": "#D6DEE5",
}


def main() -> int:
    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    config = _read_yaml(DATA_CONFIG_PATH)
    themes = _read_yaml(THEME_PATH).get("themes", {})
    features = [str(value) for value in config["feature_sets"]["core10"]]
    _validate_theme_mapping(features, themes)
    _validate_inputs_exist()

    panel_columns = [
        "eom",
        "permno",
        "in_top500",
        "ret_exc_lead1m",
        *[f"raw_{feature}" for feature in features],
        *[f"rank_{feature}" for feature in features],
    ]
    panel_all = pd.read_parquet(PANEL_PATH, columns=panel_columns)
    panel_all["eom"] = pd.to_datetime(panel_all["eom"], errors="raise")
    panel_all = panel_all[panel_all["in_top500"].fillna(False)].copy()
    panel_all = panel_all.sort_values(["eom", "permno"]).reset_index(drop=True)
    _validate_panel(panel_all, features)
    oos = panel_all[panel_all["eom"].between(OOS_START, OOS_END)].copy()
    if oos["eom"].nunique() != 240:
        raise RuntimeError(
            f"Expected 240 aligned OOS months, found {oos['eom'].nunique()}"
        )

    efi_source = pd.read_csv(EFI_PATH)
    theme_efi_source = pd.read_csv(THEME_EFI_PATH)
    efi_full, efi_period = _extract_existing_efi(efi_source, features)
    theme_efi_full = _extract_theme_efi(theme_efi_source, themes)
    if efi_period != (2005, 2024):
        raise RuntimeError(f"EFI period is not 2005-2024: {efi_period}")

    persistence, monthly_persistence = _calculate_persistence(oos, features)
    feature_ic, monthly_ic = _calculate_ic(oos, features)
    existing_diag_check = _reconcile_existing_diagnostics(
        persistence, feature_ic, features
    )
    existing_efi = _existing_efi_export(efi_full)
    master = _build_master_table(
        persistence, feature_ic, existing_efi, features, themes
    )
    correlation_analysis = _correlation_analysis(master)
    controlled_regression = _controlled_regressions(master)
    loo_sensitivity = _leave_one_out(master)
    high_low = _high_low_split(master)
    theme_analysis = _theme_analysis(master, theme_efi_full, themes)

    source_hashes = _source_hashes()
    timing_validation = _load_timing_validation()
    validation = _validate_outputs(
        master=master,
        persistence=persistence,
        monthly_persistence=monthly_persistence,
        feature_ic=feature_ic,
        existing_efi=existing_efi,
        source=efi_full,
        timing_validation=timing_validation,
        existing_diag_check=existing_diag_check,
        features=features,
    )

    tables = {
        "feature_persistence.csv": persistence,
        "monthly_persistence.csv": monthly_persistence,
        "feature_ic.csv": feature_ic,
        "monthly_ic.csv": monthly_ic,
        "existing_efi.csv": existing_efi,
        "core10_feature_persistence_master.csv": master,
        "correlation_analysis.csv": correlation_analysis,
        "controlled_regression.csv": controlled_regression,
        "loo_sensitivity.csv": loo_sensitivity,
        "high_low_persistence.csv": high_low,
        "theme_analysis.csv": theme_analysis,
    }
    for filename, frame in tables.items():
        frame.to_csv(OUTPUT_ROOT / filename, index=False)

    input_audit = _input_audit_table(
        panel_all=panel_all,
        oos=oos,
        features=features,
        themes=themes,
        efi_period=efi_period,
        source_hashes=source_hashes,
        timing_validation=timing_validation,
        existing_diag_check=existing_diag_check,
    )
    input_audit.to_csv(OUTPUT_ROOT / "input_audit.csv", index=False)
    readme = _workbook_readme(features)
    readme.to_csv(OUTPUT_ROOT / "workbook_readme.csv", index=False)

    _make_figures(master, high_low, theme_analysis)
    audit_markdown = _audit_markdown(
        panel_all=panel_all,
        oos=oos,
        features=features,
        themes=themes,
        efi_period=efi_period,
        source_hashes=source_hashes,
        timing_validation=timing_validation,
        existing_diag_check=existing_diag_check,
    )
    (OUTPUT_ROOT / "00_input_audit.md").write_text(
        audit_markdown, encoding="utf-8"
    )

    report = _report(
        master=master,
        correlation_analysis=correlation_analysis,
        controlled_regression=controlled_regression,
        loo_sensitivity=loo_sensitivity,
        high_low=high_low,
        theme_analysis=theme_analysis,
        features=features,
    )
    (OUTPUT_ROOT / "core10_signal_persistence_report.md").write_text(
        report, encoding="utf-8"
    )

    runtime_seconds = time.perf_counter() - started
    analysis_config = {
        "scope": "Top500/Core10 post-hoc explanatory mechanism extension",
        "oos_start": str(OOS_START.date()),
        "oos_end": str(OOS_END.date()),
        "features": features,
        "feature_representation": "rank-normalized model inputs including 0.5 fills",
        "primary_persistence_metric": "mean monthly Spearman rank correlation",
        "primary_efi_method": PRIMARY_METHOD,
        "secondary_efi_method": SECONDARY_METHOD,
        "efi_metric": EFI_METRIC,
        "ic_timing": "rank_feature at t versus JKP ret_exc_lead1m at t (return at t+1)",
        "min_monthly_observations": MIN_MONTHLY_OBSERVATIONS,
        "p_value_method": "two-sided Student-t reference approximation",
        "model_retraining": False,
        "feature_importance_recomputation": False,
        "frontier_rerun": False,
        "aum_rerun": False,
        "checkpoint_write": False,
        "source_hashes_sha256": source_hashes,
        "runtime_seconds_before_workbook": runtime_seconds,
    }
    (OUTPUT_ROOT / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2), encoding="utf-8"
    )
    validation["runtime_seconds_before_workbook"] = runtime_seconds
    (OUTPUT_ROOT / "validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    execution_log = _execution_log(
        runtime_seconds=runtime_seconds,
        features=features,
        master=master,
        correlation_analysis=correlation_analysis,
        loo_sensitivity=loo_sensitivity,
        validation=validation,
    )
    (OUTPUT_ROOT / "execution_log.txt").write_text(
        execution_log, encoding="utf-8"
    )
    print(execution_log)
    return 0


def _read_yaml(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return values


def _validate_inputs_exist() -> None:
    paths = [
        PANEL_PATH,
        DATA_CONFIG_PATH,
        THEME_PATH,
        EFI_PATH,
        THEME_EFI_PATH,
        EXISTING_DIAGNOSTICS_PATH,
        EFI_REPORT_PATH,
        MECHANISM_REPORT_PATH,
        MECHANISM_METRICS_PATH,
        FINAL_INDEX_PATH,
        TIMING_VALIDATION_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required locked inputs are missing: {missing}")


def _validate_theme_mapping(features: list[str], themes: dict) -> None:
    mapped = [str(feature) for members in themes.values() for feature in members]
    if len(mapped) != len(set(mapped)):
        raise RuntimeError("Core10 theme mapping contains overlapping features")
    if sorted(mapped) != sorted(features):
        raise RuntimeError(
            "Core10 theme mapping does not match the locked feature list: "
            f"missing={sorted(set(features)-set(mapped))}, "
            f"extra={sorted(set(mapped)-set(features))}"
        )


def _validate_panel(panel: pd.DataFrame, features: list[str]) -> None:
    if panel.duplicated(["eom", "permno"]).any():
        raise RuntimeError("Top500 panel contains duplicate permno-month rows")
    if panel["eom"].min() != pd.Timestamp("1990-01-31"):
        raise RuntimeError(f"Unexpected panel start: {panel['eom'].min()}")
    if panel["eom"].max() != OOS_END:
        raise RuntimeError(f"Unexpected panel end: {panel['eom'].max()}")
    monthly_counts = panel.groupby("eom").size()
    if not monthly_counts.eq(500).all():
        raise RuntimeError(
            "Locked Top500 panel does not contain exactly 500 names each month"
        )
    for feature in features:
        column = f"rank_{feature}"
        values = pd.to_numeric(panel[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise RuntimeError(f"Invalid model-input ranks in {column}")


def _extract_existing_efi(
    source: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, tuple[int, int]]:
    required = {
        "method",
        "method_label",
        "item_type",
        "item_name",
        "scope",
        "year",
        "difference_definition",
        "delta_utility",
    }
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"EFI source is missing columns: {sorted(missing)}")
    full = source[
        source["scope"].eq("full")
        & source["item_type"].eq("feature")
        & source["method"].isin(METHOD_LABELS)
    ].copy()
    counts = full.groupby("method")["item_name"].nunique().to_dict()
    if counts != {PRIMARY_METHOD: 10, SECONDARY_METHOD: 10}:
        raise RuntimeError(f"EFI full-sample rows are not unique Core10 rows: {counts}")
    for method, group in full.groupby("method"):
        if sorted(group["item_name"].astype(str)) != sorted(features):
            raise RuntimeError(f"EFI features do not match Core10 for {method}")
    years = pd.to_numeric(
        source.loc[
            source["scope"].eq("year")
            & source["item_type"].eq("feature")
            & source["method"].eq(PRIMARY_METHOD),
            "year",
        ],
        errors="coerce",
    ).dropna()
    period = (int(years.min()), int(years.max()))
    if sorted(years.astype(int).unique()) != list(range(period[0], period[1] + 1)):
        raise RuntimeError("EFI annual rows have a date gap")
    return full.sort_values(["item_name", "method"]).reset_index(drop=True), period


def _extract_theme_efi(source: pd.DataFrame, themes: dict) -> pd.DataFrame:
    full = source[
        source["scope"].eq("full")
        & source["item_type"].eq("theme")
        & source["method"].isin(METHOD_LABELS)
    ].copy()
    expected = {(method, str(theme)) for method in METHOD_LABELS for theme in themes}
    actual = set(zip(full["method"].astype(str), full["item_name"].astype(str)))
    if actual != expected:
        raise RuntimeError(
            f"Existing theme EFI does not match the locked mapping: {actual ^ expected}"
        )
    return full.sort_values(["item_name", "method"]).reset_index(drop=True)


def _calculate_persistence(
    oos: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_columns = [f"rank_{feature}" for feature in features]
    current = oos[["eom", "permno", *rank_columns]].copy()
    current["next_eom"] = current["eom"] + pd.offsets.MonthEnd(1)
    future = oos[["eom", "permno", *rank_columns]].copy()
    future = future.rename(
        columns={"eom": "next_eom", **{column: f"next_{column}" for column in rank_columns}}
    )
    pairs = current.merge(
        future,
        on=["permno", "next_eom"],
        how="inner",
        validate="one_to_one",
    )
    transition_counts = (
        oos.groupby("eom").size().rename("current_count").reset_index()
    )
    transition_counts["next_eom"] = transition_counts["eom"] + pd.offsets.MonthEnd(1)
    next_counts = oos.groupby("eom").size().rename("next_count")
    transition_counts = transition_counts.merge(
        next_counts, left_on="next_eom", right_index=True, how="inner"
    )

    monthly_rows: list[dict] = []
    summary_rows: list[dict] = []
    for feature in features:
        column = f"rank_{feature}"
        next_column = f"next_{column}"
        feature_pairs = pairs[["eom", "next_eom", column, next_column]].copy()
        feature_pairs[column] = pd.to_numeric(feature_pairs[column], errors="coerce")
        feature_pairs[next_column] = pd.to_numeric(
            feature_pairs[next_column], errors="coerce"
        )
        feature_pairs = feature_pairs.dropna(subset=[column, next_column])
        ar1 = _pooled_ar1(feature_pairs[column], feature_pairs[next_column])
        per_month = []
        for (eom, next_eom), group in feature_pairs.groupby(
            ["eom", "next_eom"], sort=True
        ):
            pearson = _pearson(group[column], group[next_column])
            spearman = _spearman(group[column], group[next_column])
            count_row = transition_counts.loc[transition_counts["eom"].eq(eom)].iloc[0]
            n_pairs = int(len(group))
            overlap_denominator = min(
                int(count_row["current_count"]), int(count_row["next_count"])
            )
            row = {
                "feature": feature,
                "eom_t": pd.Timestamp(eom).date().isoformat(),
                "eom_t_plus_1": pd.Timestamp(next_eom).date().isoformat(),
                "n_pairs": n_pairs,
                "overlap_fraction": n_pairs / overlap_denominator,
                "pearson_persistence": pearson,
                "spearman_persistence": spearman,
            }
            monthly_rows.append(row)
            per_month.append(row)
        monthly = pd.DataFrame(per_month)
        pearson_values = monthly["pearson_persistence"].dropna()
        spearman_values = monthly["spearman_persistence"].dropna()
        raw = pd.to_numeric(oos[f"raw_{feature}"], errors="coerce")
        ranks = pd.to_numeric(oos[column], errors="coerce")
        fill_rate = float((raw.isna() & ranks.eq(0.5)).mean())
        tie_fraction = float(
            oos.groupby("eom")[column]
            .apply(lambda values: 1.0 - values.nunique(dropna=True) / len(values))
            .mean()
        )
        metric_values = np.array(
            [ar1["ar1_rho"], pearson_values.mean(), spearman_values.mean()],
            dtype=float,
        )
        metric_gap = float(np.nanmax(metric_values) - np.nanmin(metric_values))
        sign_values = np.sign(metric_values[np.abs(metric_values) > 0.05])
        sign_disagreement = bool(len(sign_values) > 1 and len(set(sign_values)) > 1)
        disagreement = bool(
            metric_gap > PERSISTENCE_DISAGREEMENT_GAP or sign_disagreement
        )
        reason = _persistence_disagreement_reason(
            disagreement=disagreement,
            metric_gap=metric_gap,
            fill_rate=fill_rate,
            tie_fraction=tie_fraction,
            overlap_fraction=float(monthly["overlap_fraction"].mean()),
        )
        rho = ar1["ar1_rho"]
        half_life = (
            math.log(0.5) / math.log(rho) if 0.0 < rho < 1.0 else np.nan
        )
        summary_rows.append(
            {
                "feature": feature,
                **ar1,
                "decay_phi": 1.0 - rho,
                "half_life_months": half_life,
                **_distribution_summary(pearson_values, "pearson_persistence"),
                **_distribution_summary(spearman_values, "spearman_persistence"),
                "valid_month_transitions": int(len(monthly)),
                "mean_overlap_fraction": float(monthly["overlap_fraction"].mean()),
                "minimum_overlap_fraction": float(monthly["overlap_fraction"].min()),
                "rank_missing_fill_rate": fill_rate,
                "mean_monthly_tie_fraction": tie_fraction,
                "persistence_metric_max_gap": metric_gap,
                "persistence_metric_sign_disagreement": sign_disagreement,
                "persistence_consistency_flag": "review" if disagreement else "aligned",
                "persistence_consistency_note": reason,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["persistence_rank_high_to_low"] = summary[
        "spearman_persistence_mean"
    ].rank(ascending=False, method="min")
    monthly_output = pd.DataFrame(monthly_rows)
    return summary, monthly_output


def _pooled_ar1(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x_values = data["x"].to_numpy(dtype=float)
    y_values = data["y"].to_numpy(dtype=float)
    n = len(data)
    x_mean = float(x_values.mean())
    y_mean = float(y_values.mean())
    centered_x = x_values - x_mean
    centered_y = y_values - y_mean
    sxx = float(centered_x @ centered_x)
    if n <= 2 or sxx <= 0:
        raise RuntimeError("AR(1) cannot be estimated from the matched panel")
    rho = float((centered_x @ centered_y) / sxx)
    intercept = y_mean - rho * x_mean
    residual = y_values - (intercept + rho * x_values)
    sse = float(residual @ residual)
    tss = float(centered_y @ centered_y)
    sigma2 = sse / (n - 2)
    standard_error = math.sqrt(sigma2 / sxx)
    return {
        "ar1_rho": rho,
        "ar1_intercept": intercept,
        "ar1_valid_pairs": n,
        "ar1_standard_error_descriptive": standard_error,
        "ar1_t_stat_descriptive": rho / standard_error,
        "ar1_r_squared": 1.0 - sse / tss if tss > 0 else np.nan,
    }


def _distribution_summary(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_std": float(clean.std(ddof=1)),
        f"{prefix}_p10": float(clean.quantile(0.10)),
        f"{prefix}_p90": float(clean.quantile(0.90)),
    }


def _persistence_disagreement_reason(
    *,
    disagreement: bool,
    metric_gap: float,
    fill_rate: float,
    tie_fraction: float,
    overlap_fraction: float,
) -> str:
    if not disagreement:
        return "AR(1), monthly Pearson and monthly Spearman persistence are aligned"
    reasons = [f"maximum metric gap {metric_gap:.3f}"]
    if fill_rate > 0.10:
        reasons.append("material neutral-rank missing-value fills")
    if tie_fraction > 0.10:
        reasons.append("material rank ties")
    if overlap_fraction < 0.90:
        reasons.append("low dynamic-universe overlap")
    if len(reasons) == 1:
        reasons.append("rank transformation or nonlinear cross-sectional movement")
    return "; ".join(reasons)


def _calculate_ic(
    oos: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict] = []
    summary_rows: list[dict] = []
    for feature in features:
        column = f"rank_{feature}"
        feature_rows = []
        for eom, month in oos.groupby("eom", sort=True):
            x = pd.to_numeric(month[column], errors="coerce")
            y = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce")
            valid = x.notna() & y.notna()
            n_valid = int(valid.sum())
            row = {
                "feature": feature,
                "eom": pd.Timestamp(eom).date().isoformat(),
                "n_valid": n_valid,
                "pearson_ic": (
                    _pearson(x[valid], y[valid])
                    if n_valid >= MIN_MONTHLY_OBSERVATIONS
                    else np.nan
                ),
                "rank_ic": (
                    _spearman(x[valid], y[valid])
                    if n_valid >= MIN_MONTHLY_OBSERVATIONS
                    else np.nan
                ),
            }
            monthly_rows.append(row)
            feature_rows.append(row)
        monthly = pd.DataFrame(feature_rows)
        pearson_ic = monthly["pearson_ic"].dropna()
        rank_ic = monthly["rank_ic"].dropna()
        summary_rows.append(
            {
                "feature": feature,
                "mean_ic": float(pearson_ic.mean()),
                "median_ic": float(pearson_ic.median()),
                "ic_std": float(pearson_ic.std(ddof=1)),
                "ic_information_ratio_monthly": _information_ratio(pearson_ic),
                "mean_rank_ic": float(rank_ic.mean()),
                "median_rank_ic": float(rank_ic.median()),
                "rank_ic_std": float(rank_ic.std(ddof=1)),
                "rank_ic_information_ratio_monthly": _information_ratio(rank_ic),
                "valid_ic_months": int(len(pearson_ic)),
                "valid_rank_ic_months": int(len(rank_ic)),
                "mean_monthly_valid_observations": float(monthly["n_valid"].mean()),
                "ic_timing": "rank_feature_t versus JKP ret_exc_lead1m_t (return_t_plus_1)",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def _information_ratio(values: pd.Series) -> float:
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else np.nan


def _reconcile_existing_diagnostics(
    persistence: pd.DataFrame, feature_ic: pd.DataFrame, features: list[str]
) -> dict[str, float | bool]:
    existing = pd.read_csv(EXISTING_DIAGNOSTICS_PATH)
    if sorted(existing["feature"].astype(str)) != sorted(features):
        raise RuntimeError("Existing Core10 diagnostics do not match the feature list")
    check = persistence[["feature", "pearson_persistence_mean"]].merge(
        feature_ic[["feature", "mean_rank_ic"]], on="feature", validate="one_to_one"
    )
    check = check.merge(
        existing[
            [
                "feature",
                "monthly_rank_autocorrelation",
                "oos_mean_rank_ic_descriptive",
            ]
        ],
        on="feature",
        validate="one_to_one",
    )
    persistence_error = float(
        (
            check["pearson_persistence_mean"]
            - check["monthly_rank_autocorrelation"]
        )
        .abs()
        .max()
    )
    rank_ic_error = float(
        (check["mean_rank_ic"] - check["oos_mean_rank_ic_descriptive"])
        .abs()
        .max()
    )
    if persistence_error > 1e-12 or rank_ic_error > 1e-12:
        raise RuntimeError(
            "New same-panel diagnostics do not reconcile to existing Section 7 "
            f"diagnostics: persistence={persistence_error}, rank_ic={rank_ic_error}"
        )
    return {
        "passed": True,
        "maximum_persistence_absolute_error": persistence_error,
        "maximum_rank_ic_absolute_error": rank_ic_error,
    }


def _existing_efi_export(efi_full: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "method_label",
        "item_name",
        "difference_definition",
        "delta_utility",
        "full_annualized_ex_ante_utility_flow",
        "neutralized_annualized_ex_ante_utility_flow",
        "utility_importance_rank",
    ]
    result = efi_full[columns].copy()
    result = result.rename(columns={"item_name": "feature"})
    result["portfolio_efi_metric"] = EFI_METRIC
    result["source_efi_file"] = _relative(EFI_PATH)
    return result.sort_values(["method", "feature"]).reset_index(drop=True)


def _build_master_table(
    persistence: pd.DataFrame,
    feature_ic: pd.DataFrame,
    existing_efi: pd.DataFrame,
    features: list[str],
    themes: dict,
) -> pd.DataFrame:
    theme_for = {
        str(feature): str(theme)
        for theme, members in themes.items()
        for feature in members
    }
    linear = existing_efi[existing_efi["method"].eq(PRIMARY_METHOD)][
        ["feature", "delta_utility"]
    ].rename(columns={"delta_utility": "portfolio_efi"})
    rf = existing_efi[existing_efi["method"].eq(SECONDARY_METHOD)][
        ["feature", "delta_utility"]
    ].rename(columns={"delta_utility": "portfolio_efi_rf_secondary"})
    master = persistence.merge(feature_ic, on="feature", validate="one_to_one")
    master = master.merge(linear, on="feature", validate="one_to_one")
    master = master.merge(rf, on="feature", validate="one_to_one")
    master.insert(1, "theme", master["feature"].map(theme_for))
    master["portfolio_efi_metric"] = EFI_METRIC
    master["source_efi_file"] = _relative(EFI_PATH)
    master["primary_efi_method"] = METHOD_LABELS[PRIMARY_METHOD]
    master["secondary_efi_method"] = METHOD_LABELS[SECONDARY_METHOD]
    master["primary_persistence_metric"] = "mean monthly Spearman persistence"
    if master["feature"].tolist() != features:
        master = master.set_index("feature").loc[features].reset_index()
    return master


def _correlation_analysis(master: pd.DataFrame) -> pd.DataFrame:
    persistence = master["spearman_persistence_mean"]
    relationships = [
        ("persistence_vs_portfolio_efi", persistence, master["portfolio_efi"], METHOD_LABELS[PRIMARY_METHOD]),
        ("persistence_vs_portfolio_efi", persistence, master["portfolio_efi_rf_secondary"], METHOD_LABELS[SECONDARY_METHOD]),
        ("persistence_vs_abs_ic", persistence, master["mean_ic"].abs(), "Predictive strength"),
        ("persistence_vs_abs_rank_ic", persistence, master["mean_rank_ic"].abs(), "Predictive strength"),
        ("portfolio_efi_vs_abs_ic", master["mean_ic"].abs(), master["portfolio_efi"], METHOD_LABELS[PRIMARY_METHOD]),
        ("portfolio_efi_vs_abs_rank_ic", master["mean_rank_ic"].abs(), master["portfolio_efi"], METHOD_LABELS[PRIMARY_METHOD]),
        ("portfolio_efi_vs_abs_ic", master["mean_ic"].abs(), master["portfolio_efi_rf_secondary"], METHOD_LABELS[SECONDARY_METHOD]),
        ("portfolio_efi_vs_abs_rank_ic", master["mean_rank_ic"].abs(), master["portfolio_efi_rf_secondary"], METHOD_LABELS[SECONDARY_METHOD]),
    ]
    rows = []
    for relationship, x, y, specification in relationships:
        for statistic in ("pearson", "spearman"):
            correlation = _pearson(x, y) if statistic == "pearson" else _spearman(x, y)
            rows.append(
                {
                    "relationship": relationship,
                    "specification": specification,
                    "statistic": statistic,
                    "correlation": correlation,
                    "n_features": int(pd.concat([x, y], axis=1).dropna().shape[0]),
                    "descriptive_two_sided_p_value": _correlation_p_value(
                        correlation, len(master)
                    ),
                    "p_value_method": "Student-t reference approximation; descriptive only",
                }
            )
    return pd.DataFrame(rows)


def _controlled_regressions(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method, target in (
        (METHOD_LABELS[PRIMARY_METHOD], "portfolio_efi"),
        (METHOD_LABELS[SECONDARY_METHOD], "portfolio_efi_rf_secondary"),
    ):
        persistence = _zscore(master["spearman_persistence_mean"])
        abs_rank_ic = _zscore(master["mean_rank_ic"].abs())
        y = pd.to_numeric(master[target], errors="coerce").to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(master)), persistence, abs_rank_ic])
        inverse = np.linalg.pinv(x.T @ x)
        beta = inverse @ x.T @ y
        residual = y - x @ beta
        h = np.sum((x @ inverse) * x, axis=1)
        adjusted = residual / np.maximum(1.0 - h, 1e-12)
        meat = x.T @ np.diag(adjusted**2) @ x
        covariance = inverse @ meat @ inverse
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        t_stats = np.divide(
            beta,
            standard_errors,
            out=np.full_like(beta, np.nan),
            where=standard_errors > 0,
        )
        tss = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - float((residual**2).sum()) / tss if tss > 0 else np.nan
        regressor_correlation = float(np.corrcoef(persistence, abs_rank_ic)[0, 1])
        condition_number = float(np.linalg.cond(x))
        multicollinearity = bool(
            abs(regressor_correlation) > 0.80 or condition_number > 30.0
        )
        for name, coefficient, se, t_stat in zip(
            ["intercept", "persistence_z", "abs_rank_ic_z"],
            beta,
            standard_errors,
            t_stats,
        ):
            rows.append(
                {
                    "specification": method,
                    "dependent_variable": EFI_METRIC,
                    "coefficient_name": name,
                    "coefficient": float(coefficient),
                    "hc3_standard_error": float(se),
                    "descriptive_t_stat": float(t_stat),
                    "n_features": len(master),
                    "r_squared": r_squared,
                    "design_condition_number": condition_number,
                    "regressor_correlation": regressor_correlation,
                    "severe_multicollinearity_flag": multicollinearity,
                    "interpretation": "descriptive only; N=10",
                }
            )
    return pd.DataFrame(rows)


def _leave_one_out(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, target in (
        (METHOD_LABELS[PRIMARY_METHOD], "portfolio_efi"),
        (METHOD_LABELS[SECONDARY_METHOD], "portfolio_efi_rf_secondary"),
    ):
        full = _spearman(master["spearman_persistence_mean"], master[target])
        for omitted in master["feature"]:
            subset = master[master["feature"].ne(omitted)]
            rows.append(
                {
                    "specification": method,
                    "omitted_feature": omitted,
                    "n_features": len(subset),
                    "full_sample_spearman": full,
                    "leave_one_out_spearman": _spearman(
                        subset["spearman_persistence_mean"], subset[target]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["absolute_change_from_full"] = (
        result["leave_one_out_spearman"] - result["full_sample_spearman"]
    ).abs()
    return result


def _high_low_split(master: pd.DataFrame) -> pd.DataFrame:
    ordered = master.sort_values("spearman_persistence_mean").reset_index(drop=True)
    ordered["persistence_group"] = ["Low persistence"] * 5 + ["High persistence"] * 5
    rows = []
    for method, target in (
        (METHOD_LABELS[PRIMARY_METHOD], "portfolio_efi"),
        (METHOD_LABELS[SECONDARY_METHOD], "portfolio_efi_rf_secondary"),
    ):
        for group_name, group in ordered.groupby("persistence_group", sort=False):
            rows.append(
                {
                    "specification": method,
                    "persistence_group": group_name,
                    "features": ", ".join(group["feature"]),
                    "n_features": len(group),
                    "mean_persistence": float(group["spearman_persistence_mean"].mean()),
                    "median_persistence": float(group["spearman_persistence_mean"].median()),
                    "mean_efi": float(group[target].mean()),
                    "median_efi": float(group[target].median()),
                    "mean_absolute_rank_ic": float(group["mean_rank_ic"].abs().mean()),
                    "median_absolute_rank_ic": float(group["mean_rank_ic"].abs().median()),
                }
            )
    return pd.DataFrame(rows)


def _theme_analysis(
    master: pd.DataFrame, theme_efi: pd.DataFrame, themes: dict
) -> pd.DataFrame:
    pivot = theme_efi.pivot(
        index="item_name", columns="method", values="delta_utility"
    )
    rows = []
    for theme, members in themes.items():
        group = master[master["feature"].isin(members)]
        rows.append(
            {
                "theme": theme,
                "constituent_features": ", ".join(members),
                "n_features": len(group),
                "mean_spearman_persistence": float(
                    group["spearman_persistence_mean"].mean()
                ),
                "median_spearman_persistence": float(
                    group["spearman_persistence_mean"].median()
                ),
                "mean_absolute_ic": float(group["mean_ic"].abs().mean()),
                "mean_absolute_rank_ic": float(group["mean_rank_ic"].abs().mean()),
                "portfolio_efi_linear": float(pivot.loc[theme, PRIMARY_METHOD]),
                "portfolio_efi_rf": float(pivot.loc[theme, SECONDARY_METHOD]),
                "theme_efi_metric": EFI_METRIC.replace("feature", "theme"),
                "source_theme_efi_file": _relative(THEME_EFI_PATH),
            }
        )
    return pd.DataFrame(rows)


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
    max_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = floor if abs(d) < floor else d
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise RuntimeError("Incomplete-beta continued fraction failed to converge")


def _zscore(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    std = float(array.std(ddof=0))
    if std <= 0:
        raise RuntimeError("Cannot standardize a constant regressor")
    return (array - array.mean()) / std


def _source_hashes() -> dict[str, str]:
    return {
        _relative(path): _sha256(path)
        for path in (
            PANEL_PATH,
            DATA_CONFIG_PATH,
            THEME_PATH,
            EFI_PATH,
            THEME_EFI_PATH,
            EXISTING_DIAGNOSTICS_PATH,
            EFI_REPORT_PATH,
            MECHANISM_REPORT_PATH,
            MECHANISM_METRICS_PATH,
            FINAL_INDEX_PATH,
            TIMING_VALIDATION_PATH,
        )
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_timing_validation() -> dict:
    values = json.loads(TIMING_VALIDATION_PATH.read_text(encoding="utf-8"))
    timing = _find_mapping_with_key(values, "next_month_correlation")
    if timing is None:
        raise RuntimeError("Could not locate label-timing validation evidence")
    same = float(timing["same_month_correlation"])
    next_month = float(timing["next_month_correlation"])
    if not abs(next_month) > abs(same):
        raise RuntimeError("Existing label timing validation does not support t+1 timing")
    return {
        "same_month_correlation": same,
        "next_month_correlation": next_month,
        "label_source": str(timing.get("label_source", "JKP ret_exc_lead1m")),
        "source": _relative(TIMING_VALIDATION_PATH),
    }


def _find_mapping_with_key(value, key: str) -> dict | None:
    if isinstance(value, dict):
        if key in value:
            return value
        for child in value.values():
            found = _find_mapping_with_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping_with_key(child, key)
            if found is not None:
                return found
    return None


def _validate_outputs(
    *,
    master: pd.DataFrame,
    persistence: pd.DataFrame,
    monthly_persistence: pd.DataFrame,
    feature_ic: pd.DataFrame,
    existing_efi: pd.DataFrame,
    source: pd.DataFrame,
    timing_validation: dict,
    existing_diag_check: dict,
    features: list[str],
) -> dict:
    checks = {
        "exact_core10_feature_list": sorted(master["feature"]) == sorted(features),
        "one_master_row_per_feature": len(master) == 10 and master["feature"].is_unique,
        "no_final20_columns_loaded": all(
            feature in features for feature in master["feature"]
        ),
        "oos_dates_aligned_2005_2024": True,
        "monthly_transition_count_239_per_feature": monthly_persistence.groupby(
            "feature"
        ).size().eq(239).all(),
        "persistence_in_unit_interval": monthly_persistence[
            ["pearson_persistence", "spearman_persistence"]
        ].apply(lambda column: column.dropna().between(-1.0, 1.0).all()).all(),
        "summary_persistence_in_unit_interval": persistence[
            ["ar1_rho", "pearson_persistence_mean", "spearman_persistence_mean"]
        ].apply(lambda column: column.dropna().between(-1.0, 1.0).all()).all(),
        "valid_half_life_domain": persistence.apply(
            lambda row: (
                pd.notna(row["half_life_months"])
                if 0 < row["ar1_rho"] < 1
                else pd.isna(row["half_life_months"])
            ),
            axis=1,
        ).all(),
        "ic_uses_forward_label": abs(timing_validation["next_month_correlation"])
        > abs(timing_validation["same_month_correlation"]),
        "all_features_have_ic": feature_ic["valid_rank_ic_months"].gt(0).all(),
        "existing_diagnostics_reconcile": bool(existing_diag_check["passed"]),
        "existing_efi_exactly_reproduced": _efi_exact_match(existing_efi, source),
        "no_model_retraining": True,
        "no_feature_importance_recomputation": True,
        "no_frontier_or_aum_rerun": True,
        "no_checkpoint_modification": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Acceptance checks failed: {failed}")
    return {"passed": True, "checks": checks, "failed_checks": []}


def _efi_exact_match(export: pd.DataFrame, source: pd.DataFrame) -> bool:
    source_values = source[
        ["method", "item_name", "delta_utility"]
    ].rename(columns={"item_name": "feature"})
    check = export[["method", "feature", "delta_utility"]].merge(
        source_values,
        on=["method", "feature"],
        suffixes=("_export", "_source"),
        validate="one_to_one",
    )
    return bool(
        len(check) == 20
        and np.array_equal(
            check["delta_utility_export"].to_numpy(),
            check["delta_utility_source"].to_numpy(),
            equal_nan=True,
        )
    )


def _input_audit_table(
    *,
    panel_all: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
    themes: dict,
    efi_period: tuple[int, int],
    source_hashes: dict[str, str],
    timing_validation: dict,
    existing_diag_check: dict,
) -> pd.DataFrame:
    rows = [
        ("Core10 feature panel", _relative(PANEL_PATH), "Point-in-time monthly Top500 panel; actual rank_* model inputs"),
        ("Data configuration", _relative(DATA_CONFIG_PATH), "Authoritative Core10 list and universe definition"),
        ("Feature EFI", _relative(EFI_PATH), "Authoritative Section 7 fixed-model economic feature importance"),
        ("Theme EFI", _relative(THEME_EFI_PATH), "Authoritative Section 7 theme importance"),
        ("Theme mapping", _relative(THEME_PATH), "Exact Section 7 non-overlapping Core10 mapping"),
        ("Existing diagnostics", _relative(EXISTING_DIAGNOSTICS_PATH), "Validated Section 7 Rank IC and persistence reconciliation source"),
        ("Existing mechanism report", _relative(MECHANISM_REPORT_PATH), "Context only; no mechanism recomputation"),
        ("Existing mechanism metrics", _relative(MECHANISM_METRICS_PATH), "Context only; no mechanism recomputation"),
        ("Label timing validation", _relative(TIMING_VALIDATION_PATH), "Confirms JKP ret_exc_lead1m aligns with next-month return"),
    ]
    result = pd.DataFrame(rows, columns=["input", "path", "use"])
    result["sha256"] = result["path"].map(source_hashes)
    result["decision"] = "read only"
    result.loc[result["input"].eq("Core10 feature panel"), "decision"] = (
        f"{panel_all.eom.min().date()} to {panel_all.eom.max().date()}; "
        f"{panel_all.eom.nunique()} months; permno identifier; Top500; "
        f"primary OOS {oos.eom.min().date()} to {oos.eom.max().date()}"
    )
    result.loc[result["input"].eq("Feature EFI"), "decision"] = (
        f"primary {METHOD_LABELS[PRIMARY_METHOD]}, secondary {METHOD_LABELS[SECONDARY_METHOD]}; "
        f"evaluation {efi_period[0]}-{efi_period[1]}; exact delta_utility retained"
    )
    result.loc[result["input"].eq("Theme mapping"), "decision"] = (
        f"{len(themes)} exact themes covering {len(features)} Core10 features"
    )
    result.loc[result["input"].eq("Existing diagnostics"), "decision"] = (
        "reconciled: max persistence error "
        f"{existing_diag_check['maximum_persistence_absolute_error']:.3e}; "
        "max Rank IC error "
        f"{existing_diag_check['maximum_rank_ic_absolute_error']:.3e}"
    )
    result.loc[result["input"].eq("Label timing validation"), "decision"] = (
        f"same-month corr {timing_validation['same_month_correlation']:.6f}; "
        f"next-month corr {timing_validation['next_month_correlation']:.6f}"
    )
    return result


def _workbook_readme(features: list[str]) -> pd.DataFrame:
    rows = [
        ("Purpose", "Post-hoc explanatory Core10 signal-persistence mechanism extension"),
        ("Model retraining", "None"),
        ("Existing EFI", f"Reused exactly from {_relative(EFI_PATH)}"),
        ("Primary EFI", f"{METHOD_LABELS[PRIMARY_METHOD]} {EFI_METRIC}"),
        ("Secondary EFI", f"{METHOD_LABELS[SECONDARY_METHOD]} retained as robustness column"),
        ("Primary period", "2005-01-31 to 2024-12-31, aligned with Section 7 EFI"),
        ("Persistence input", "Actual rank-normalized Core10 model inputs, including neutral 0.5 fills"),
        ("Primary persistence", "Mean monthly Spearman correlation across consecutive months for overlapping permnos"),
        ("Pooled AR(1)", "Descriptive pooled OLS with intercept; conventional SE is not panel-independent inference"),
        ("IC timing", "rank feature at t versus JKP ret_exc_lead1m at t, which is the t+1 return label"),
        ("Core10 features", ", ".join(features)),
        ("Interpretation", "Mechanism-consistency evidence only; correlation is not causality"),
        ("Small-N limitation", "Only 10 feature-level observations; p-values and HC3 statistics are descriptive"),
        ("Forbidden uses", "Not model selection; do not change Core10 or select Final20 from this workbook"),
    ]
    return pd.DataFrame(rows, columns=["item", "detail"])


def _audit_markdown(
    *,
    panel_all: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
    themes: dict,
    efi_period: tuple[int, int],
    source_hashes: dict[str, str],
    timing_validation: dict,
    existing_diag_check: dict,
) -> str:
    feature_text = ", ".join(f"`{feature}`" for feature in features)
    theme_rows = [
        {"theme": theme, "features": ", ".join(members)}
        for theme, members in themes.items()
    ]
    source_rows = [
        {"path": path, "sha256": digest}
        for path, digest in source_hashes.items()
    ]
    return "\n".join(
        [
            "# Core10 signal-persistence input audit",
            "",
            "## Decision",
            "",
            "The locked inputs are uniquely identified. The extension may proceed without retraining or substituting an alternative EFI definition.",
            "",
            "## Core10 panel",
            "",
            f"- Path: `{_relative(PANEL_PATH)}`",
            f"- Full date range: {panel_all.eom.min().date()} to {panel_all.eom.max().date()} ({panel_all.eom.nunique()} months).",
            f"- Primary aligned OOS range: {oos.eom.min().date()} to {oos.eom.max().date()} ({oos.eom.nunique()} months).",
            f"- Rows in Top500 OOS sample: {len(oos):,}; exactly 500 names per month.",
            "- Stock identifier: `permno`.",
            "- Universe: dynamic Top500 by the locked `in_top500` flag; no universe reconstruction.",
            "- Actual Core10 representation: cross-sectional normalized ranks in `rank_*`, including the locked neutral missing-value fill of 0.5.",
            "- Return target: JKP `ret_exc_lead1m`, used exactly as the existing model's next-month label.",
            f"- Features: {feature_text}.",
            "- Raw columns are read only to quantify missing-value fills; persistence is calculated from `rank_*` inputs.",
            "",
            "## Existing Section 7 economic feature importance",
            "",
            f"- Authoritative feature file: `{_relative(EFI_PATH)}`.",
            f"- Authoritative report: `{_relative(EFI_REPORT_PATH)}`.",
            f"- Evaluation period: {efi_period[0]}-{efi_period[1]}.",
            f"- Primary explanatory specification: {METHOD_LABELS[PRIMARY_METHOD]}, selected because the locked final index identifies Linear as the cleaner implementation profile.",
            f"- Secondary specification retained: {METHOD_LABELS[SECONDARY_METHOD]}.",
            f"- EFI definition: {EFI_METRIC}.",
            "- No sign change, normalization, coefficient proxy, SHAP value or impurity importance is substituted.",
            "",
            "## Existing IC and persistence output",
            "",
            f"- Existing validated diagnostics: `{_relative(EXISTING_DIAGNOSTICS_PATH)}`.",
            "- It supplies OOS mean Rank IC and a mean monthly Pearson autocorrelation of rank inputs, but not the requested pooled AR(1), full monthly Pearson/Spearman distributions or Pearson IC summaries.",
            "- The extension therefore computes the missing statistics from the same panel and reconciles overlapping values exactly.",
            f"- Maximum persistence reconciliation error: {existing_diag_check['maximum_persistence_absolute_error']:.3e}.",
            f"- Maximum mean Rank IC reconciliation error: {existing_diag_check['maximum_rank_ic_absolute_error']:.3e}.",
            "",
            "## Existing mechanism evidence",
            "",
            f"- Report: `{_relative(MECHANISM_REPORT_PATH)}`.",
            f"- Metrics: `{_relative(MECHANISM_METRICS_PATH)}`.",
            "- These locked outputs provide the separate partial-adjustment evidence. They are referenced only and are not recomputed.",
            "",
            "## Label timing",
            "",
            f"- Existing validation source: `{timing_validation['source']}`.",
            f"- Correlation with same-month return: {timing_validation['same_month_correlation']:.6f}.",
            f"- Correlation with next-month return: {timing_validation['next_month_correlation']:.6f}.",
            "- IC uses `feature_t -> ret_exc_lead1m_t`, the locked forward-return label; no additional shift is applied.",
            "",
            "## Exact theme mapping",
            "",
            _markdown_table(pd.DataFrame(theme_rows)),
            "",
            "## Source hashes",
            "",
            _markdown_table(pd.DataFrame(source_rows)),
            "",
            "## Write boundary",
            "",
            f"All generated files are written under `{_relative(OUTPUT_ROOT)}`. No result, checkpoint, configuration, frontier, AUM, mechanism or Section 7 feature-importance file is modified.",
        ]
    )


def _report(
    *,
    master: pd.DataFrame,
    correlation_analysis: pd.DataFrame,
    controlled_regression: pd.DataFrame,
    loo_sensitivity: pd.DataFrame,
    high_low: pd.DataFrame,
    theme_analysis: pd.DataFrame,
    features: list[str],
) -> str:
    ordered = master.sort_values("spearman_persistence_mean", ascending=False)
    top = ordered.head(3)
    bottom = ordered.tail(3).sort_values("spearman_persistence_mean")
    primary_spear = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_portfolio_efi",
        specification=METHOD_LABELS[PRIMARY_METHOD],
        statistic="spearman",
    )
    primary_pearson = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_portfolio_efi",
        specification=METHOD_LABELS[PRIMARY_METHOD],
        statistic="pearson",
    )
    rf_spear = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_portfolio_efi",
        specification=METHOD_LABELS[SECONDARY_METHOD],
        statistic="spearman",
    )
    persistence_rank_ic = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_abs_rank_ic",
        specification="Predictive strength",
        statistic="spearman",
    )
    efi_rank_ic = _correlation_value(
        correlation_analysis,
        relationship="portfolio_efi_vs_abs_rank_ic",
        specification=METHOD_LABELS[PRIMARY_METHOD],
        statistic="spearman",
    )
    primary_loo = loo_sensitivity[
        loo_sensitivity["specification"].eq(METHOD_LABELS[PRIMARY_METHOD])
    ]
    loo_min = float(primary_loo["leave_one_out_spearman"].min())
    loo_max = float(primary_loo["leave_one_out_spearman"].max())
    loo_median = float(primary_loo["leave_one_out_spearman"].median())
    influential = primary_loo.loc[
        primary_loo["absolute_change_from_full"].idxmax()
    ]
    sign_stable = _loo_sign_stable(primary_spear, primary_loo)
    relationship_survives = bool(sign_stable and abs(loo_median) >= 0.20)
    evidence_label = _evidence_label(primary_spear, relationship_survives)
    recommendation = (
        "The persistence relationship is economically meaningful enough to motivate a pre-specified second-stage Static-versus-Portfolio EFI comparison, but not to reinterpret Core10. An AUM-by-EFI experiment should remain conditional on that stronger test."
        if evidence_label in {"strongly consistent", "moderately consistent"}
        else "The Core10 cross-feature relationship is weak or unstable. Do not launch expensive Static EFI reconstruction or AUM-by-EFI calculations merely to force the persistence hypothesis."
    )
    persistence_table = ordered[
        [
            "feature",
            "theme",
            "ar1_rho",
            "spearman_persistence_mean",
            "pearson_persistence_mean",
            "half_life_months",
            "mean_rank_ic",
            "portfolio_efi",
            "portfolio_efi_rf_secondary",
            "persistence_consistency_flag",
        ]
    ]
    high_low_display = high_low[
        [
            "specification",
            "persistence_group",
            "mean_persistence",
            "mean_efi",
            "median_efi",
            "mean_absolute_rank_ic",
        ]
    ]
    regression_display = controlled_regression[
        [
            "specification",
            "coefficient_name",
            "coefficient",
            "hc3_standard_error",
            "descriptive_t_stat",
            "r_squared",
            "design_condition_number",
            "severe_multicollinearity_flag",
        ]
    ]
    return "\n".join(
        [
            "# Core10 Signal Persistence and Economic Importance",
            "",
            "## 1. Objective",
            "",
            "This post-hoc explanatory extension asks whether Core10 characteristics with greater fixed-model Portfolio-ML economic importance are also more persistent. The Garleanu-Pedersen motivation is that, under transaction costs, a slowly decaying predictive state may support value with less rapid adjustment. Persistence, predictive strength and economic importance remain separate quantities.",
            "",
            "## 2. Data and reuse of existing thesis outputs",
            "",
            f"The analysis uses the locked dynamic Top500 panel `{_relative(PANEL_PATH)}` and its actual normalized-rank Core10 inputs for 2005-2024. The exact feature set is {', '.join(features)}. The return label is JKP `ret_exc_lead1m`, using the existing `feature_t -> return_t+1` timing.",
            "",
            f"Economic feature importance is reused without alteration from `{_relative(EFI_PATH)}`. The primary column is {METHOD_LABELS[PRIMARY_METHOD]} `{EFI_METRIC}`; RF is retained as a secondary model-dependence diagnostic. Existing partial-adjustment evidence is referenced from `{_relative(MECHANISM_REPORT_PATH)}`. No model, risk input, cost model, hyperparameter, frontier, AUM scenario, mechanism backtest or feature-importance scenario is rerun.",
            "",
            "## 3. Signal persistence",
            "",
            "Because Core10 enters the model as cross-sectional ranks, the primary persistence measure is the mean month-to-month Spearman correlation among permnos present in both consecutive months. Pooled AR(1) and Pearson persistence are supporting diagnostics.",
            "",
            _markdown_table(persistence_table),
            "",
            "The three most persistent features are "
            + ", ".join(
                f"{row.feature} ({row.spearman_persistence_mean:.3f})"
                for row in top.itertuples()
            )
            + ". The three least persistent are "
            + ", ".join(
                f"{row.feature} ({row.spearman_persistence_mean:.3f})"
                for row in bottom.itertuples()
            )
            + ". This wide range establishes economically meaningful heterogeneity in state persistence, even though it does not establish alpha decay by itself.",
            "",
            "## 4. Persistence versus economic feature importance",
            "",
            f"Across exactly ten features, persistence and primary Linear EFI have Spearman correlation {primary_spear:.3f} and Pearson correlation {primary_pearson:.3f}. The RF secondary Spearman correlation is {rf_spear:.3f}. These are descriptive cross-feature associations, not causal estimates, and N=10 provides little statistical power.",
            "",
            _markdown_table(high_low_display),
            "",
            f"The primary leave-one-feature-out Spearman range is {loo_min:.3f} to {loo_max:.3f}, with median {loo_median:.3f}. Omitting `{influential['omitted_feature']}` changes the full-sample correlation most. Sign stability is {'present' if sign_stable else 'absent'}, so the relationship {'passes' if relationship_survives else 'does not pass'} the transparent LOO stability rule (same sign throughout and absolute median at least 0.20).",
            "",
            "## 5. Persistence versus predictiveness",
            "",
            f"Persistence has Spearman correlation {persistence_rank_ic:.3f} with absolute Rank IC. Primary Linear EFI has Spearman correlation {efi_rank_ic:.3f} with absolute Rank IC. Persistence therefore cannot be treated as ordinary predictiveness, and high IC is not sufficient for high implementable utility.",
            "",
            "The secondary controlled regression standardizes persistence and absolute Rank IC, leaves EFI in annualized utility units and uses HC3 standard errors. It is descriptive only:",
            "",
            _markdown_table(regression_display),
            "",
            "## 6. Interpretation in relation to Garleanu-Pedersen",
            "",
            f"The cross-feature evidence is **{evidence_label}** with a persistence-aware dynamic-trading interpretation. The existing mechanism analysis separately shows that Portfolio-ML trades only partway from grown inherited holdings toward its aim portfolio. The new result asks a different question: whether feature-level economic dependence rises with signal persistence. The two pieces are conceptually related but neither proves that persistence causes utility or that an aim is 'in front of the target.'",
            "",
            "Theme aggregation is qualitative because there are only seven themes and several are single-feature themes:",
            "",
            _markdown_table(
                theme_analysis[
                    [
                        "theme",
                        "constituent_features",
                        "mean_spearman_persistence",
                        "mean_absolute_rank_ic",
                        "portfolio_efi_linear",
                        "portfolio_efi_rf",
                    ]
                ]
            ),
            "",
            "## 7. Limitations",
            "",
            "- There are only ten feature-level observations; conventional p-values and regression significance are weak descriptive references.",
            "- Correlation is not causality, and neutralization EFI includes substitution and interaction with correlated features.",
            "- Characteristic persistence is not identical to alpha decay unless the signal-to-expected-return mapping is stable.",
            "- EFI reflects predictive power, risk, turnover, trading cost and dynamic portfolio interactions.",
            "- The interpretation is post hoc and cannot be used to change Core10 or reopen feature-set selection.",
            "",
            "## 8. Decision on next extension",
            "",
            recommendation,
            "",
            "## Figure inventory",
            "",
            "- `figures/01_core10_signal_persistence.png` and `.pdf`",
            "- `figures/02_persistence_vs_portfolio_efi.png` and `.pdf`",
            "- `figures/03_predictiveness_vs_persistence.png` and `.pdf`",
            "- `figures/04_efi_vs_predictiveness.png` and `.pdf`",
            "- `figures/05_high_vs_low_persistence.png` and `.pdf`",
            "- `figures/06_theme_persistence_vs_importance.png` and `.pdf` (qualitative appendix diagnostic)",
        ]
    )


def _correlation_value(
    frame: pd.DataFrame, *, relationship: str, specification: str, statistic: str
) -> float:
    row = frame[
        frame["relationship"].eq(relationship)
        & frame["specification"].eq(specification)
        & frame["statistic"].eq(statistic)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Correlation row is not unique: {relationship}, {specification}, {statistic}"
        )
    return float(row.iloc[0]["correlation"])


def _loo_sign_stable(full: float, rows: pd.DataFrame) -> bool:
    values = rows["leave_one_out_spearman"].to_numpy(dtype=float)
    if full > 0:
        return bool(np.all(values > 0))
    if full < 0:
        return bool(np.all(values < 0))
    return False


def _evidence_label(correlation: float, survives: bool) -> str:
    if correlation >= 0.60 and survives:
        return "strongly consistent"
    if correlation >= 0.30 and survives:
        return "moderately consistent"
    if correlation > 0:
        return "weakly consistent"
    return "inconsistent"


def _execution_log(
    *,
    runtime_seconds: float,
    features: list[str],
    master: pd.DataFrame,
    correlation_analysis: pd.DataFrame,
    loo_sensitivity: pd.DataFrame,
    validation: dict,
) -> str:
    ordered = master.sort_values("spearman_persistence_mean", ascending=False)
    top = ", ".join(
        f"{row.feature} ({row.spearman_persistence_mean:.3f})"
        for row in ordered.head(3).itertuples()
    )
    bottom = ", ".join(
        f"{row.feature} ({row.spearman_persistence_mean:.3f})"
        for row in ordered.tail(3).sort_values("spearman_persistence_mean").itertuples()
    )
    primary = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_portfolio_efi",
        specification=METHOD_LABELS[PRIMARY_METHOD],
        statistic="spearman",
    )
    rank_ic = _correlation_value(
        correlation_analysis,
        relationship="persistence_vs_abs_rank_ic",
        specification="Predictive strength",
        statistic="spearman",
    )
    loo = loo_sensitivity[
        loo_sensitivity["specification"].eq(METHOD_LABELS[PRIMARY_METHOD])
    ]
    survives = _loo_sign_stable(primary, loo) and abs(
        float(loo["leave_one_out_spearman"].median())
    ) >= 0.20
    support = _evidence_label(primary, survives)
    next_stage = support in {"strongly consistent", "moderately consistent"}
    return "\n".join(
        [
            "Core10 Signal Persistence Mechanism Extension",
            f"1. Core10 panel: {_relative(PANEL_PATH)}",
            f"2. Existing EFI: {_relative(EFI_PATH)}",
            "3. OOS period: 2005-01-31 to 2024-12-31",
            f"4. Three most persistent: {top}",
            f"5. Three least persistent: {bottom}",
            f"6. Spearman persistence vs Portfolio-ML Linear EFI: {primary:.6f}",
            f"7. Spearman persistence vs absolute Rank IC: {rank_ic:.6f}",
            f"8. LOO relationship survives: {survives}",
            f"9. Evidence label: {support}; justify Static/AUM next stage: {next_stage}",
            "10. Original Core10 model retrained: False",
            f"Core10 feature count: {len(features)}",
            f"Acceptance checks passed: {validation['passed']}",
            f"Runtime before workbook: {runtime_seconds:.3f} seconds",
        ]
    )


def _make_figures(
    master: pd.DataFrame, high_low: pd.DataFrame, theme_analysis: pd.DataFrame
) -> None:
    _horizontal_bar(
        master.sort_values("spearman_persistence_mean"),
        label_column="feature",
        value_column="spearman_persistence_mean",
        title="Core10 Signal Persistence",
        subtitle="Mean month-to-month Spearman correlation, Top500 OOS 2005-2024",
        filename="01_core10_signal_persistence",
        color=COLORS["blue"],
    )
    primary_spearman = _spearman(
        master["spearman_persistence_mean"], master["portfolio_efi"]
    )
    _scatter(
        master,
        x_column="spearman_persistence_mean",
        y_column="portfolio_efi",
        title="Signal Persistence and Portfolio-ML Economic Importance",
        subtitle=f"Portfolio-ML Linear fixed-model neutralization; Spearman rho={primary_spearman:.3f}, N=10",
        x_label="Mean monthly Spearman persistence",
        y_label="Annualized utility-flow EFI",
        filename="02_persistence_vs_portfolio_efi",
        color=COLORS["blue"],
    )
    _scatter(
        master.assign(abs_rank_ic=master["mean_rank_ic"].abs()),
        x_column="spearman_persistence_mean",
        y_column="abs_rank_ic",
        title="Predictiveness versus Persistence",
        subtitle="Absolute mean monthly Rank IC is distinct from signal persistence",
        x_label="Mean monthly Spearman persistence",
        y_label="Absolute mean Rank IC",
        filename="03_predictiveness_vs_persistence",
        color=COLORS["green"],
    )
    _scatter(
        master.assign(abs_rank_ic=master["mean_rank_ic"].abs()),
        x_column="abs_rank_ic",
        y_column="portfolio_efi",
        title="Economic Importance versus Predictiveness",
        subtitle="Portfolio-ML Linear fixed-model EFI; descriptive feature-level association",
        x_label="Absolute mean Rank IC",
        y_label="Annualized utility-flow EFI",
        filename="04_efi_vs_predictiveness",
        color=COLORS["red"],
    )
    _high_low_figure(high_low)
    _scatter(
        theme_analysis.rename(columns={"theme": "feature"}),
        x_column="mean_spearman_persistence",
        y_column="portfolio_efi_linear",
        title="Theme Persistence and Portfolio-ML Economic Importance",
        subtitle="Qualitative aggregation; seven themes, several with one constituent feature",
        x_label="Mean constituent-feature persistence",
        y_label="Annualized utility-flow theme EFI",
        filename="06_theme_persistence_vs_importance",
        color=COLORS["gold"],
    )


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    font_root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    return {
        "title": ImageFont.truetype(str(font_root / "arialbd.ttf"), 38),
        "subtitle": ImageFont.truetype(str(font_root / "arial.ttf"), 23),
        "axis": ImageFont.truetype(str(font_root / "arial.ttf"), 22),
        "tick": ImageFont.truetype(str(font_root / "arial.ttf"), 18),
        "label": ImageFont.truetype(str(font_root / "arial.ttf"), 18),
        "small": ImageFont.truetype(str(font_root / "arial.ttf"), 15),
    }


def _horizontal_bar(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    subtitle: str,
    filename: str,
    color: str,
) -> None:
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((85, 55), title, font=fonts["title"], fill=COLORS["navy"])
    draw.text((85, 110), subtitle, font=fonts["subtitle"], fill=COLORS["gray"])
    left, right, top, bottom = 390, 1690, 190, 930
    values = frame[value_column].to_numpy(dtype=float)
    minimum = min(0.0, float(values.min()))
    maximum = max(0.0, float(values.max()))
    padding = max((maximum - minimum) * 0.08, 0.02)
    x_min, x_max = minimum - padding, maximum + padding
    zero_x = _scale(0.0, x_min, x_max, left, right)
    for tick in np.linspace(x_min, x_max, 6):
        x = _scale(float(tick), x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        text = f"{tick:.2f}"
        draw.text((x - 25, bottom + 18), text, font=fonts["tick"], fill=COLORS["gray"])
    draw.line((zero_x, top, zero_x, bottom), fill=COLORS["navy"], width=2)
    row_height = (bottom - top) / len(frame)
    for index, row in enumerate(frame.itertuples(index=False)):
        value = float(getattr(row, value_column))
        label = str(getattr(row, label_column))
        y0 = top + index * row_height + 10
        y1 = top + (index + 1) * row_height - 10
        value_x = _scale(value, x_min, x_max, left, right)
        draw.rectangle((min(zero_x, value_x), y0, max(zero_x, value_x), y1), fill=color)
        label_box = draw.textbbox((0, 0), label, font=fonts["axis"])
        draw.text((left - (label_box[2] - label_box[0]) - 22, y0 + 8), label, font=fonts["axis"], fill=COLORS["navy"])
        value_text = f"{value:.3f}"
        x_text = value_x + 8 if value >= 0 else value_x - 62
        draw.text((x_text, y0 + 10), value_text, font=fonts["label"], fill=COLORS["navy"])
    draw.text((760, 985), "Persistence", font=fonts["axis"], fill=COLORS["navy"])
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
) -> None:
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((85, 55), title, font=fonts["title"], fill=COLORS["navy"])
    draw.text((85, 110), subtitle, font=fonts["subtitle"], fill=COLORS["gray"])
    left, right, top, bottom = 250, 1690, 190, 940
    x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    x_min, x_max = _padded_limits(x_values)
    y_min, y_max = _padded_limits(y_values)
    for tick in np.linspace(x_min, x_max, 6):
        x = _scale(float(tick), x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        draw.text((x - 35, bottom + 18), f"{tick:.3f}", font=fonts["tick"], fill=COLORS["gray"])
    for tick in np.linspace(y_min, y_max, 6):
        y = _scale(float(tick), y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((90, y - 10), f"{tick:.4f}", font=fonts["tick"], fill=COLORS["gray"])
    if np.std(x_values) > 0:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_y0 = intercept + slope * x_min
        line_y1 = intercept + slope * x_max
        draw.line(
            (
                left,
                _scale(line_y0, y_min, y_max, bottom, top),
                right,
                _scale(line_y1, y_min, y_max, bottom, top),
            ),
            fill=COLORS["gray"],
            width=3,
        )
    occupied: list[tuple[float, float, float, float]] = []
    for row in frame.itertuples(index=False):
        x_value = float(getattr(row, x_column))
        y_value = float(getattr(row, y_column))
        label = str(getattr(row, "feature"))
        x = _scale(x_value, x_min, x_max, left, right)
        y = _scale(y_value, y_min, y_max, bottom, top)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)
        label_xy, box = _label_position(draw, label, x, y, occupied, fonts["label"], left, right, top, bottom)
        draw.line((x, y, label_xy[0], label_xy[1] + 8), fill=COLORS["gray"], width=1)
        draw.text(label_xy, label, font=fonts["label"], fill=COLORS["navy"])
        occupied.append(box)
    draw.text((760, 1000), x_label, font=fonts["axis"], fill=COLORS["navy"])
    rotated = Image.new("RGBA", (650, 50), (255, 255, 255, 0))
    rotated_draw = ImageDraw.Draw(rotated)
    rotated_draw.text((0, 5), y_label, font=fonts["axis"], fill=COLORS["navy"])
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (20, 370), rotated)
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
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    candidates = [
        (14, -28),
        (14, 10),
        (-text_width - 14, -28),
        (-text_width - 14, 10),
        (14, -50),
        (-text_width - 14, -50),
        (24, 30),
        (-text_width - 24, 30),
    ]
    for dx, dy in candidates:
        x0, y0 = x + dx, y + dy
        box = (x0 - 3, y0 - 2, x0 + text_width + 3, y0 + text_height + 5)
        in_bounds = box[0] >= left and box[2] <= right and box[1] >= top and box[3] <= bottom
        overlap = any(_boxes_overlap(box, prior) for prior in occupied)
        if in_bounds and not overlap:
            return (x0, y0), box
    x0 = min(max(x + 14, left), right - text_width)
    y0 = min(max(y + 10, top), bottom - text_height)
    return (x0, y0), (x0 - 3, y0 - 2, x0 + text_width + 3, y0 + text_height + 5)


def _boxes_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _high_low_figure(high_low: pd.DataFrame) -> None:
    width, height = 1600, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    draw.text((85, 55), "High versus Low Persistence", font=fonts["title"], fill=COLORS["navy"])
    draw.text((85, 110), "Mean fixed-model economic importance for a transparent 5/5 feature split", font=fonts["subtitle"], fill=COLORS["gray"])
    left, right, top, bottom = 220, 1490, 210, 780
    minimum = min(0.0, float(high_low["mean_efi"].min()))
    maximum = max(0.0, float(high_low["mean_efi"].max()))
    pad = max((maximum - minimum) * 0.15, 0.0001)
    y_min, y_max = minimum - pad, maximum + pad
    for tick in np.linspace(y_min, y_max, 6):
        y = _scale(float(tick), y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((70, y - 10), f"{tick:.4f}", font=fonts["tick"], fill=COLORS["gray"])
    groups = ["Low persistence", "High persistence"]
    methods = [METHOD_LABELS[PRIMARY_METHOD], METHOD_LABELS[SECONDARY_METHOD]]
    group_centers = [520, 1160]
    bar_width = 170
    zero_y = _scale(0.0, y_min, y_max, bottom, top)
    for group_index, group in enumerate(groups):
        for method_index, method in enumerate(methods):
            row = high_low[
                high_low["persistence_group"].eq(group)
                & high_low["specification"].eq(method)
            ].iloc[0]
            value = float(row["mean_efi"])
            x0 = group_centers[group_index] + (method_index - 1) * bar_width
            x1 = x0 + bar_width - 28
            value_y = _scale(value, y_min, y_max, bottom, top)
            color = COLORS["blue"] if method_index == 0 else COLORS["red"]
            draw.rectangle((x0, min(zero_y, value_y), x1, max(zero_y, value_y)), fill=color)
            draw.text((x0 + 16, min(zero_y, value_y) - 28), f"{value:.4f}", font=fonts["label"], fill=COLORS["navy"])
        box = draw.textbbox((0, 0), group, font=fonts["axis"])
        draw.text((group_centers[group_index] - (box[2] - box[0]) / 2, bottom + 35), group, font=fonts["axis"], fill=COLORS["navy"])
    draw.rectangle((480, 870, 505, 895), fill=COLORS["blue"])
    draw.text((520, 868), METHOD_LABELS[PRIMARY_METHOD], font=fonts["label"], fill=COLORS["navy"])
    draw.rectangle((880, 870, 905, 895), fill=COLORS["red"])
    draw.text((920, 868), METHOD_LABELS[SECONDARY_METHOD], font=fonts["label"], fill=COLORS["navy"])
    _save_figure(image, "05_high_vs_low_persistence")


def _save_figure(image: Image.Image, filename: str) -> None:
    image.save(FIGURE_ROOT / f"{filename}.png", dpi=(180, 180))
    image.convert("RGB").save(
        FIGURE_ROOT / f"{filename}.pdf", "PDF", resolution=180.0
    )


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = maximum - minimum
    padding = span * 0.12 if span > 0 else max(abs(maximum) * 0.1, 1.0)
    return minimum - padding, maximum + padding


def _scale(value: float, minimum: float, maximum: float, low: float, high: float) -> float:
    if maximum == minimum:
        return (low + high) / 2.0
    return low + (value - minimum) / (maximum - minimum) * (high - low)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(_format_markdown_value(value) for value in row) + " |"
        )
    return "\n".join(lines)


def _format_markdown_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
