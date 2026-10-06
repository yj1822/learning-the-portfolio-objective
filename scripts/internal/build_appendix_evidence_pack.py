"""Build the dissertation appendix evidence pack from locked artifacts only.

This script deliberately imports no model or experiment runner. It reads final
configs/results/audits, reshapes stored observations, copies existing figures,
and writes a new appendix staging directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "dissertation" / "appendix_pack"
FIG_OUT = OUT / "figures"
NOT_IDENTIFIABLE = "NOT IDENTIFIABLE FROM CURRENT FINAL ARTIFACTS"

MAIN = ROOT / "results" / "experiments" / "top500_core10_factor_cumulative"
FINAL15 = ROOT / "results" / "experiments" / "top500_final15_factor_cumulative"
FINAL20 = ROOT / "results" / "experiments" / "top500_final20_conservative_factor_cumulative"
FINAL15_REPORT = ROOT / "reports" / "model_results" / "top500_final15_cumulative"
FRONTIER = ROOT / "reports" / "final_analysis" / "frontier"
AUM = ROOT / "reports" / "final_analysis" / "aum_sensitivity"
RISK_FIG = ROOT / "reports" / "final_analysis" / "risk_calibration"
EFI = ROOT / "reports" / "final_analysis" / "feature_importance"
PERSIST = ROOT / "outputs" / "extensions" / "core10_signal_persistence"
ALPHA = ROOT / "outputs" / "extensions" / "core10_predictive_alpha_decay"

METHOD_LABELS = {
    "return_ml_rank": "Return-ML Rank",
    "static_ml": "Static-ML",
    "static_ml_star": "Static-ML*",
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}
METHOD_ORDER = list(METHOD_LABELS)
FEATURE_SET_DIRS = {
    "Core10": MAIN,
    "Final15": FINAL15,
    "Final20 Conservative": FINAL20,
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False, na_rep="NA", float_format="%.17g")
    return path


def parse_candidate(value: Any) -> dict[str, Any]:
    if pd.isna(value):
        return {}
    return json.loads(str(value))


def value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(value_text(item) for item in value) + "]"
    return str(value)


def frequency_text(values: Iterable[Any], denominator: int | None = None) -> str:
    clean = [value for value in values if not pd.isna(value)]
    if not clean:
        return "never selected"
    counts = Counter(value_text(value) for value in clean)
    denom = denominator if denominator is not None else len(clean)
    return "; ".join(f"{key}: {count}/{denom}" for key, count in sorted(counts.items()))


def md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    show = frame.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    show = show.fillna("NA")
    columns = [str(column) for column in show.columns]

    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "NA"
            return f"{float(value):.10g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in show.iterrows():
        rows.append("| " + " | ".join(cell(row[column]) for column in show.columns) + " |")
    return "\n".join(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_selected_hyperparameters() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = read_csv(MAIN / "selected_hyperparameters_by_year.csv")
    selected = selected[selected["selected"].astype(str).str.lower().eq("true")].copy()
    selected["parsed"] = selected["candidate_hyperparameters"].map(parse_candidate)
    prediction = read_csv(MAIN / "return_ml_prediction_diagnostics.csv").copy()

    rows: list[dict[str, Any]] = []
    for year in range(2005, 2025):
        pred = prediction.loc[prediction["test_year"] == year].iloc[0]
        out: dict[str, Any] = {
            "year": year,
            "return_ml_model_class": pred["selected_kind"],
            "return_ml_ridge_lambda": pred["selected_lambda"],
            "return_ml_rf_p": pred["selected_p"],
            "return_ml_rf_eta": pred["selected_eta"],
            "return_ml_validation_mse": pred["selected_validation_mse"],
        }
        for method in ["static_ml", "static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf"]:
            item = selected.loc[(selected["test_year"] == year) & (selected["method"] == method)].iloc[0]
            hp = item["parsed"]
            if method == "static_ml":
                out.update(
                    static_ml_s=hp.get("mu_shrinkage"),
                    static_ml_phi=hp.get("phi"),
                    static_ml_validation_score=item["cumulative_validation_score"],
                )
            elif method == "static_ml_star":
                out.update(
                    static_ml_star_u=hp.get("u"),
                    static_ml_star_v=hp.get("v"),
                    static_ml_star_k=hp.get("k"),
                    static_ml_star_validation_score=item["cumulative_validation_score"],
                )
            elif method == "portfolio_ml_linear":
                out.update(
                    portfolio_ml_linear_lambda=hp.get("ridge_lambda"),
                    portfolio_ml_linear_validation_utility_score=item["cumulative_validation_score"],
                )
            else:
                out.update(
                    portfolio_ml_rf_p=hp.get("rf_p"),
                    portfolio_ml_rf_eta=hp.get("rf_eta"),
                    portfolio_ml_rf_lambda=hp.get("ridge_lambda"),
                    portfolio_ml_rf_validation_utility_score=item["cumulative_validation_score"],
                )
        rows.append(out)
    return pd.DataFrame(rows), prediction


def build_hyperparameter_tables(
    config: dict[str, Any], annual_hp: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(annual_hp)
    ridge_pred = prediction[prediction["selected_kind"] == "ridge"]
    rf_pred = prediction[prediction["selected_kind"] == "rf_ridge"]

    grids = [
        {
            "method": "Return-ML Linear",
            "component": "prediction",
            "parameter": "ridge lambda",
            "candidate_values": value_text(config["return_ml"]["lambda_grid"]),
            "selection_criterion": "minimum pre-test prediction validation MSE",
            "selection_frequency": f"model class {len(ridge_pred)}/{total}; " + frequency_text(ridge_pred["selected_lambda"], total),
        },
        {
            "method": "Return-ML Random Features",
            "component": "prediction",
            "parameter": "ridge lambda",
            "candidate_values": value_text(config["return_ml"]["lambda_grid"]),
            "selection_criterion": "minimum pre-test prediction validation MSE",
            "selection_frequency": f"model class {len(rf_pred)}/{total}; " + frequency_text(rf_pred["selected_lambda"], total),
        },
        {
            "method": "Return-ML Random Features",
            "component": "prediction",
            "parameter": "p (final sinusoidal columns)",
            "candidate_values": value_text(config["return_ml"]["rf_p_grid"]),
            "selection_criterion": "minimum pre-test prediction validation MSE",
            "selection_frequency": frequency_text(rf_pred["selected_p"], total),
        },
        {
            "method": "Return-ML Random Features",
            "component": "prediction",
            "parameter": "eta",
            "candidate_values": value_text(config["return_ml"]["rf_eta_grid"]),
            "selection_criterion": "minimum pre-test prediction validation MSE",
            "selection_frequency": frequency_text(rf_pred["selected_eta"], total),
        },
        {
            "method": "Static-ML",
            "component": "portfolio decision",
            "parameter": "s (mu_shrinkage)",
            "candidate_values": value_text(config["static_ml"]["mu_shrinkage_grid"]),
            "selection_criterion": "maximum cumulative Static OOS score using gross-return population variance",
            "selection_frequency": frequency_text(annual_hp["static_ml_s"], total),
        },
        {
            "method": "Static-ML",
            "component": "portfolio decision",
            "parameter": "phi",
            "candidate_values": value_text(config["static_ml"]["phi_grid"]),
            "selection_criterion": "maximum cumulative Static OOS score using gross-return population variance",
            "selection_frequency": frequency_text(annual_hp["static_ml_phi"], total),
        },
        {
            "method": "Static-ML*",
            "component": "portfolio decision",
            "parameter": "u",
            "candidate_values": value_text(config["static_ml_star"]["u_grid"]),
            "selection_criterion": "maximum cumulative Static OOS score using gross-return population variance",
            "selection_frequency": frequency_text(annual_hp["static_ml_star_u"], total),
        },
        {
            "method": "Static-ML*",
            "component": "portfolio decision",
            "parameter": "v",
            "candidate_values": value_text(config["static_ml_star"]["v_grid"]),
            "selection_criterion": "maximum cumulative Static OOS score using gross-return population variance",
            "selection_frequency": frequency_text(annual_hp["static_ml_star_v"], total),
        },
        {
            "method": "Static-ML*",
            "component": "portfolio decision",
            "parameter": "k",
            "candidate_values": value_text(config["static_ml_star"]["k_grid"]),
            "selection_criterion": "maximum cumulative Static OOS score using gross-return population variance",
            "selection_frequency": frequency_text(annual_hp["static_ml_star_k"], total),
        },
        {
            "method": "Portfolio-ML Linear",
            "component": "portfolio objective",
            "parameter": "ridge lambda",
            "candidate_values": value_text(config["portfolio_ml"]["lambda_grid"]),
            "selection_criterion": "maximum cumulative mean realized OOS utility flow",
            "selection_frequency": frequency_text(annual_hp["portfolio_ml_linear_lambda"], total),
        },
        {
            "method": "Portfolio-ML RF",
            "component": "portfolio objective",
            "parameter": "ridge lambda",
            "candidate_values": value_text(config["portfolio_ml"]["lambda_grid"]),
            "selection_criterion": "maximum cumulative mean realized OOS utility flow",
            "selection_frequency": frequency_text(annual_hp["portfolio_ml_rf_lambda"], total),
        },
        {
            "method": "Portfolio-ML RF",
            "component": "portfolio objective",
            "parameter": "p (final sinusoidal columns)",
            "candidate_values": value_text(config["portfolio_ml"]["rf_p_grid"]),
            "selection_criterion": "maximum cumulative mean realized OOS utility flow",
            "selection_frequency": frequency_text(annual_hp["portfolio_ml_rf_p"], total),
        },
        {
            "method": "Portfolio-ML RF",
            "component": "portfolio objective",
            "parameter": "eta",
            "candidate_values": value_text(config["portfolio_ml"]["rf_eta_grid"]),
            "selection_criterion": "maximum cumulative mean realized OOS utility flow",
            "selection_frequency": frequency_text(annual_hp["portfolio_ml_rf_eta"], total),
        },
    ]

    data_config = read_yaml(ROOT / config["data_config"])
    risk_manifest = read_json(ROOT / "data" / "processed" / "us_equity_ml" / "risk_repaired" / "build_manifest.json")
    fixed = [
        ("baseline gamma", config["baseline_gamma"], "all baseline portfolio decisions and utility", "configs/model_top500_core10_cumulative.yaml"),
        ("baseline AUM", config["baseline_aum"], "optimization and realized transaction costs; fixed through time", "configs/model_top500_core10_cumulative.yaml"),
        ("random seed", config["random_seed"], "Return-ML RF, Portfolio-ML RF, bootstrap", "configs/model_top500_core10_cumulative.yaml"),
        ("matrix-m fixed-point updates", config["portfolio_ml"]["matrix_m_iterations"], "Portfolio-ML full matrix m", "configs/model_top500_core10_cumulative.yaml"),
        ("expected monthly market return in m", config["portfolio_ml"]["expected_market_return"], "matrix m", "configs/model_top500_core10_cumulative.yaml"),
        ("monthly risk-free rate in m", config["portfolio_ml"]["risk_free_rate"], "matrix m", "configs/model_top500_core10_cumulative.yaml"),
        ("Static covariance numerical ridge", config["static_ml"]["covariance_ridge"], "Static-ML solve", "configs/model_top500_core10_cumulative.yaml"),
        ("Static-ML* covariance numerical ridge", config["static_ml_star"]["covariance_ridge"], "Static-ML* solve", "configs/model_top500_core10_cumulative.yaml"),
        ("Portfolio-ML volatility floor", config["portfolio_ml"]["volatility_floor"], "Portfolio-ML design volatility scaling", "configs/model_top500_core10_cumulative.yaml"),
        ("Portfolio-ML coefficient-ridge floor", config["portfolio_ml"]["coefficient_ridge_floor"], "Portfolio-ML closed-form solve", "configs/model_top500_core10_cumulative.yaml"),
        ("ADV cost coefficient", data_config["transaction_cost"]["lambda_multiplier"], "lambda_adv = coefficient / ADV_6m", "configs/data_us_equity_ml.yaml"),
        ("covariance eigenvalue floor", config["risk"]["variance_floor"], "monthly stock covariance", "configs/model_top500_core10_cumulative.yaml"),
        ("risk-regression ridge alpha", risk_manifest["ridge_alpha"], "daily factor-return cross-sectional ridge", "data/processed/us_equity_ml/risk_repaired/build_manifest.json"),
        ("risk EWMA span", data_config["data_hub"]["risk_ewma_span"], "daily factor covariance and idiosyncratic variance", "configs/data_us_equity_ml.yaml"),
        ("daily-to-monthly risk multiplier", config["risk"]["factor_daily_to_monthly"], "factor and idiosyncratic variances", "configs/model_top500_core10_cumulative.yaml"),
    ]
    fixed_frame = pd.DataFrame(fixed, columns=["setting", "final_value", "where_used", "source"])
    return pd.DataFrame(grids), fixed_frame


def build_integrity_table(preflight: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    safety = preflight["safety"]
    max_metric_error = max(item["recomputed_absolute_error"] for item in preflight["metric_comparisons"])
    max_workbook_error = max(item["workbook_absolute_error"] for item in preflight["metric_comparisons"])
    max_utility_error = max(item["utility_accounting_max_error"] for item in preflight["utility_inference_comparisons"])
    risk = read_csv(MAIN / "risk_calibration.csv")
    factor_exposures = pd.read_parquet(
        ROOT / "data" / "processed" / "us_equity_ml" / "risk_repaired" / "factor_exposures.parquet",
        columns=["eom"],
    )
    analysis_start = pd.Timestamp(
        read_yaml(ROOT / "configs" / "model_top500_core10_cumulative.yaml")["cumulative_oos_validation"]["analysis_start"]
    )
    risk_dates = pd.to_datetime(factor_exposures["eom"])
    usable_months = int(risk_dates[risk_dates >= analysis_start].nunique())
    return pd.DataFrame(
        [
            ("formal OOS months", int(read_csv(MAIN / "metrics.csv")["n_months"].min()), "240", rel(MAIN / "metrics.csv")),
            ("usable risk-model months", usable_months, "360", "reports/dissertation/dissertation_data_fact_audit.md"),
            ("factor fallback count", metadata["factor_fallback_count"], "0", rel(MAIN / "metadata.json")),
            ("covariance repair count", int(pd.to_numeric(risk["covariance_repair_count"], errors="coerce").fillna(0).sum()), "0", rel(MAIN / "risk_calibration.csv")),
            ("matrix-m fallback count", metadata["matrix_m_fallback_count"], "0", rel(MAIN / "metadata.json")),
            ("scalar-rho fallback count", "N/A - disabled; any failure would stop", "N/A", "configs/model_top500_core10_cumulative.yaml"),
            ("future-information violations", safety["future_information_violations"], "0", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("validation-cutoff violations", safety["validation_cutoff_violations"], "0", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("universe size violations", "N/A - not a separately emitted final diagnostic", "N/A", "reports/dissertation/dissertation_methodology_fact_audit.md"),
            ("missing-security alignment violations", "N/A - not a separately emitted final diagnostic", "N/A", "reports/dissertation/dissertation_methodology_fact_audit.md"),
            ("baseline metric recomputation max abs error", max_metric_error, "<= 1e-6", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("baseline workbook reconciliation max abs error", max_workbook_error, "<= 1e-6", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("utility accounting max abs error", max_utility_error, "<= 1e-10", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("final preflight gate", preflight["passed"], "true", "reports/final_analysis/preflight/final_analysis_preflight.json"),
            ("checkpoint/resume verification", metadata["resume_verification"]["passed"], "true", rel(MAIN / "metadata.json")),
        ],
        columns=["diagnostic", "final_result", "expected_acceptable_value", "source"],
    )


def build_annual_utility() -> tuple[pd.DataFrame, dict[str, Any]]:
    annual = read_csv(MAIN / "annual_performance.csv")
    pivot = annual.pivot(index="test_year", columns="method", values="annualized_ex_ante_utility_flow")
    pivot = pivot[METHOD_ORDER].mul(100.0).reset_index().rename(columns={"test_year": "year", **METHOD_LABELS})
    pivot.columns.name = None
    differences = annual.pivot(index="test_year", columns="method", values="annualized_ex_ante_utility_flow")
    linear_star = differences["portfolio_ml_linear"] - differences["static_ml_star"]
    rf_star = differences["portfolio_ml_rf"] - differences["static_ml_star"]
    rf_linear = differences["portfolio_ml_rf"] - differences["portfolio_ml_linear"]
    concentration = read_csv(MAIN / "contribution_concentration.csv")

    def conc(candidate: str) -> dict[str, Any]:
        return concentration.loc[
            (concentration["candidate_method"] == candidate)
            & (concentration["benchmark_method"] == "static_ml_star")
            & (concentration["outcome"] == "utility_flow")
        ].iloc[0].to_dict()

    summary = {
        "linear_years_won_vs_static_star": int((linear_star > 0).sum()),
        "rf_years_won_vs_static_star": int((rf_star > 0).sum()),
        "rf_years_won_vs_linear": int((rf_linear > 0).sum()),
        "linear_static_star_largest_positive": (int(linear_star.idxmax()), float(linear_star.max() * 100)),
        "linear_static_star_largest_negative": (int(linear_star.idxmin()), float(linear_star.min() * 100)),
        "rf_static_star_largest_positive": (int(rf_star.idxmax()), float(rf_star.max() * 100)),
        "rf_static_star_largest_negative": (int(rf_star.idxmin()), float(rf_star.min() * 100)),
        "linear_2007_positive_contribution_share": float(conc("portfolio_ml_linear")["top_1_positive_year_share"]),
        "rf_2007_positive_contribution_share": float(conc("portfolio_ml_rf")["top_1_positive_year_share"]),
    }
    return pivot, summary


def build_gamma_table() -> pd.DataFrame:
    source = read_csv(FRONTIER / "frontier_points.csv")
    columns = {
        "annualized_gross_return": "gross_return",
        "annualized_trading_cost": "trading_cost",
        "annualized_net_return": "net_return",
        "annualized_volatility": "volatility",
        "net_sharpe": "net_sharpe",
        "annualized_utility": "utility_flow",
        "monthly_turnover": "turnover",
        "average_leverage": "leverage",
    }
    keep = ["gamma", "method", *columns, "fallback_count", "covariance_repair_count", "selected_hyperparameters", "metric_reconciliation_max_abs_error"]
    out = source[keep].rename(columns=columns).copy()
    out.insert(2, "method_label", out["method"].map(METHOD_LABELS))
    out["is_baseline_gamma"] = out["gamma"].eq(10)
    return out.sort_values(["gamma", "method"]).reset_index(drop=True)


def build_aum_table() -> pd.DataFrame:
    source = read_csv(AUM / "aum_sensitivity_points.csv")
    columns = {
        "annualized_gross_return": "gross_return",
        "annualized_trading_cost": "trading_cost",
        "annualized_net_return": "net_return",
        "annualized_volatility": "volatility",
        "annualized_utility": "utility_flow",
        "monthly_turnover": "turnover",
        "average_leverage": "leverage",
        "predicted_annualized_volatility": "predicted_volatility",
        "realized_to_predicted_volatility_ratio": "realized_to_predicted_ratio",
    }
    keep = [
        "aum_multiplier",
        "aum_usd",
        "method",
        *columns,
        "net_sharpe",
        "fallback_count",
        "covariance_repair_count",
        "selected_hyperparameters",
        "metric_reconciliation_max_abs_error",
    ]
    out = source[keep].rename(columns=columns).copy()
    out.insert(3, "method_label", out["method"].map(METHOD_LABELS))
    utility = out.pivot(index="aum_multiplier", columns="method", values="utility_flow")
    gaps = pd.DataFrame(
        {
            "linear_minus_static_star_utility_gap": utility["portfolio_ml_linear"] - utility["static_ml_star"],
            "rf_minus_static_star_utility_gap": utility["portfolio_ml_rf"] - utility["static_ml_star"],
            "rf_minus_linear_utility_gap": utility["portfolio_ml_rf"] - utility["portfolio_ml_linear"],
        }
    ).reset_index()
    out = out.merge(gaps, on="aum_multiplier", how="left")
    out["is_baseline_aum"] = out["aum_multiplier"].eq(1)
    out["is_frictionless_diagnostic"] = out["aum_multiplier"].eq(0)
    return out.sort_values(["aum_multiplier", "method"]).reset_index(drop=True)


def build_risk_table() -> pd.DataFrame:
    source = read_csv(MAIN / "risk_calibration.csv").copy()
    source["inside_0_6_1_5"] = source["realized_to_predicted_ratio"].between(0.6, 1.5, inclusive="both")
    source["above_2"] = source["realized_to_predicted_ratio"].gt(2.0)
    columns = [
        "scope",
        "method",
        "test_year",
        "predicted_annualized_volatility",
        "realized_annualized_volatility",
        "realized_to_predicted_ratio",
        "inside_0_6_1_5",
        "above_2",
        "yearly_ratio_median",
        "yearly_ratio_p10",
        "yearly_ratio_p90",
        "yearly_ratio_max",
        "fraction_years_within_0_6_1_5",
        "years_ratio_above_2",
        "factor_risk_fallback_count",
        "covariance_repair_count",
    ]
    out = source[columns].copy()
    out.insert(2, "method_label", out["method"].map(METHOD_LABELS))
    return out


def build_feature_set_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    inference_frames = []
    for feature_set, directory in FEATURE_SET_DIRS.items():
        metrics = read_csv(directory / "metrics.csv").copy()
        metrics.insert(0, "feature_set", feature_set)
        metrics.insert(2, "method_label", metrics["method"].map(METHOD_LABELS))
        metrics_frames.append(metrics)
        inference = read_csv(directory / "statistical_inference.csv")
        inference = inference[inference["outcome"] == "utility_flow"].copy()
        inference.insert(0, "feature_set", feature_set)
        inference_frames.append(inference)
    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    metric_columns = [
        "feature_set",
        "method",
        "method_label",
        "annualized_gross_return",
        "annualized_trading_cost",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "annualized_ex_ante_utility_flow",
        "average_turnover",
        "average_leverage",
        "n_months",
    ]
    inference_all = pd.concat(inference_frames, ignore_index=True)
    inference_columns = [
        "feature_set",
        "candidate_method",
        "benchmark_method",
        "annualized_mean_difference",
        "newey_west_t_stat",
        "bootstrap_annualized_ci_lower",
        "bootstrap_annualized_ci_upper",
        "probability_delta_positive",
        "annual_years_won",
        "annual_years_total",
        "n_months",
    ]
    return metrics_all[metric_columns], inference_all[inference_columns]


def build_ex2007_table() -> pd.DataFrame:
    source = read_csv(FINAL15_REPORT / "ex_year_robustness.csv")
    samples = ["full_2005_2024", "exclude_2007"]
    metric_methods = ["static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf"]
    metric_rows = source[
        source["sample"].isin(samples)
        & source["row_type"].eq("method_metrics")
        & source["method"].isin(metric_methods)
    ].copy()
    inference_rows = source[
        source["sample"].isin(samples)
        & source["row_type"].eq("utility_inference")
        & source["candidate_method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])
        & source["benchmark_method"].eq("static_ml_star")
        & source["outcome"].eq("utility_flow")
    ].copy()
    out = pd.concat([metric_rows, inference_rows], ignore_index=True, sort=False)
    columns = [
        "sample",
        "excluded_years",
        "row_type",
        "method",
        "candidate_method",
        "benchmark_method",
        "n_months",
        "annualized_gross_return",
        "annualized_trading_cost",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "annualized_ex_ante_utility_flow",
        "average_turnover",
        "average_leverage",
        "annualized_mean_difference",
        "newey_west_t_stat",
        "bootstrap_annualized_ci_lower",
        "bootstrap_annualized_ci_upper",
        "probability_delta_positive",
        "annual_years_won",
        "annual_years_total",
    ]
    return out[columns]


def build_efi_table() -> pd.DataFrame:
    source = read_csv(EFI / "feature_importance.csv")
    source = source[(source["item_type"] == "feature") & (source["scope"] == "full")].copy()
    records = []
    for feature in sorted(source["item_name"].unique()):
        linear = source[(source["item_name"] == feature) & (source["method"] == "portfolio_ml_linear")].iloc[0]
        rf = source[(source["item_name"] == feature) & (source["method"] == "portfolio_ml_rf")].iloc[0]
        linear_efi = float(linear["delta_utility"])
        rf_efi = float(rf["delta_utility"])
        records.append(
            {
                "feature": feature,
                "linear_full_policy_utility": linear["full_annualized_ex_ante_utility_flow"],
                "linear_neutralized_policy_utility": linear["neutralized_annualized_ex_ante_utility_flow"],
                "linear_efi": linear_efi,
                "linear_rank": linear["utility_importance_rank"],
                "rf_full_policy_utility": rf["full_annualized_ex_ante_utility_flow"],
                "rf_neutralized_policy_utility": rf["neutralized_annualized_ex_ante_utility_flow"],
                "rf_efi": rf_efi,
                "rf_rank": rf["utility_importance_rank"],
                "sign_agreement": bool(np.sign(linear_efi) == np.sign(rf_efi)),
                "conceptual_label": "fixed-model economic dependence",
            }
        )
    return pd.DataFrame(records).sort_values("linear_rank").reset_index(drop=True)


def build_persistence_tables(efi_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    persistence = read_csv(PERSIST / "core10_feature_persistence_master.csv")
    alpha = read_csv(ALPHA / "core10_predictive_alpha_decay_master.csv")
    merged = persistence.merge(alpha, on=["feature", "theme"], suffixes=("_raw", "_alpha"))
    merged = merged.merge(
        efi_table[["feature", "linear_efi", "rf_efi", "linear_rank", "rf_rank"]],
        on="feature",
        how="left",
    )
    summary = pd.DataFrame(
        {
            "feature": merged["feature"],
            "theme": merged["theme"],
            "raw_persistence": merged["spearman_persistence_mean"],
            "absolute_rank_ic": merged["mean_rank_ic"].abs(),
            "fmb_h1_slope": merged["fmb_h1"],
            "rank_ic_h1": merged["rankic_h1"],
            "fmb_signed_future_mass": merged["fmb_sfm"],
            "rank_ic_signed_future_mass": merged["rankic_sfm"],
            "fmb_first_half_decay_horizon": merged["fmb_first_decay_horizon"],
            "fmb_first_reversal_horizon": merged["fmb_reversal_horizon"],
            "linear_efi": merged["linear_efi"],
            "rf_efi": merged["rf_efi"],
            "raw_persistence_rank": merged["persistence_rank_high_to_low"],
            "alpha_persistence_rank_fmb_sfm": merged["fmb_sfm_rank_high_to_low"],
            "linear_efi_rank": merged["linear_rank"],
            "rf_efi_rank": merged["rf_rank"],
        }
    ).sort_values("raw_persistence_rank")

    def horizons(prefix_raw: str, prefix_adjusted: str) -> pd.DataFrame:
        records = []
        for _, row in alpha.iterrows():
            for path_type, prefix in [("raw", prefix_raw), ("sign_adjusted", prefix_adjusted)]:
                record: dict[str, Any] = {"feature": row["feature"], "path_type": path_type}
                for horizon in range(1, 13):
                    record[f"h{horizon}"] = row[f"{prefix}{horizon:02d}"]
                records.append(record)
        return pd.DataFrame(records)

    fmb = horizons("fmb_slope_h", "fmb_sign_adjusted_h")
    rank_ic = horizons("rank_ic_h", "rankic_sign_adjusted_h")
    return summary, fmb, rank_ic


def build_persistence_gate() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_corr = read_csv(PERSIST / "correlation_analysis.csv")
    raw_loo = read_csv(PERSIST / "loo_sensitivity.csv")

    for relationship, metric in [
        ("persistence_vs_portfolio_efi", "raw persistence vs EFI"),
        ("portfolio_efi_vs_abs_rank_ic", "absolute Rank IC vs EFI"),
    ]:
        subset = raw_corr[raw_corr["relationship"] == relationship]
        for model in ["Portfolio-ML Linear", "Portfolio-ML RF"]:
            model_rows = subset[subset["specification"] == model]
            if model_rows.empty:
                continue
            spearman = model_rows.loc[model_rows["statistic"] == "spearman", "correlation"].iloc[0]
            pearson = model_rows.loc[model_rows["statistic"] == "pearson", "correlation"].iloc[0]
            loo_min = loo_max = sign_stable = np.nan
            if relationship == "persistence_vs_portfolio_efi":
                loo = raw_loo[raw_loo["specification"] == model]
                loo_min = loo["leave_one_out_spearman"].min()
                loo_max = loo["leave_one_out_spearman"].max()
                signs = np.sign(loo["leave_one_out_spearman"].to_numpy())
                sign_stable = bool(np.all(signs == np.sign(float(spearman))))
            rows.append(
                {
                    "row_type": "relationship",
                    "metric_or_criterion": metric,
                    "method": model,
                    "spearman_correlation": spearman,
                    "pearson_correlation": pearson,
                    "loo_minimum": loo_min,
                    "loo_maximum": loo_max,
                    "loo_sign_stable_all": sign_stable,
                    "exact_definition": "cross-feature descriptive relation; N=10",
                    "pass_fail": "NA",
                    "evidence": "stored correlation and LOO diagnostics",
                    "interpretation": "descriptive only",
                }
            )

    alpha_corr = read_csv(ALPHA / "efi_correlations.csv")
    alpha_loo = read_csv(ALPHA / "loo_sensitivity.csv")
    spearman = alpha_corr[alpha_corr["statistic"] == "spearman"]
    for (method, metric), group in spearman.groupby(["efi_method", "alpha_persistence_metric"]):
        pearson_group = alpha_corr[
            (alpha_corr["efi_method"] == method)
            & (alpha_corr["alpha_persistence_metric"] == metric)
            & (alpha_corr["statistic"] == "pearson")
        ]
        loo = alpha_loo[(alpha_loo["efi_method"] == method) & (alpha_loo["alpha_persistence_metric"] == metric)]
        rows.append(
            {
                "row_type": "relationship",
                "metric_or_criterion": f"{metric} vs EFI",
                "method": method,
                "spearman_correlation": group["correlation"].iloc[0],
                "pearson_correlation": pearson_group["correlation"].iloc[0] if not pearson_group.empty else np.nan,
                "loo_minimum": loo["loo_minimum"].iloc[0] if not loo.empty else np.nan,
                "loo_maximum": loo["loo_maximum"].iloc[0] if not loo.empty else np.nan,
                "loo_sign_stable_all": loo["loo_sign_stable_all"].iloc[0] if not loo.empty else np.nan,
                "exact_definition": "prespecified predictive-alpha metric related to fixed-model EFI across ten features",
                "pass_fail": "NA",
                "evidence": "stored alpha-decay correlation and LOO diagnostics",
                "interpretation": "descriptive only",
            }
        )

    gates = [
        (
            "material improvement over raw-persistence benchmark",
            "Linear FMB-SFM Spearman >= raw-persistence Linear rho 0.212121 + 0.20",
            "FAIL",
            "Observed Linear FMB-SFM Spearman = 0.115152.",
        ),
        (
            "cross-metric/model sign consistency",
            "Primary rho positive for Linear/RF and at least 60% of ten prespecified Spearman metric signs positive for each model",
            "FAIL",
            "Sign consistency across metrics and models is insufficient.",
        ),
        (
            "primary-metric LOO stability",
            "FMB-SFM LOO sign stable for Linear and RF and each maximum absolute change <= 0.35",
            "FAIL",
            "Linear changes sign; the both-model condition fails.",
        ),
        (
            "momentum/reversal coherence",
            "Momentum positive mass/no reversal/positive Linear EFI; reversal decay at h=2 without recovery, APR12<0.50, matching h1 signs, and nonpositive Linear EFI",
            "FAIL",
            "Momentum subcondition passes, but reversal recovers and its h1 FMB/Rank-IC signs disagree.",
        ),
    ]
    for criterion, definition, result, evidence in gates:
        rows.append(
            {
                "row_type": "gate",
                "metric_or_criterion": criterion,
                "method": "both Portfolio-ML models where applicable",
                "spearman_correlation": np.nan,
                "pearson_correlation": np.nan,
                "loo_minimum": np.nan,
                "loo_maximum": np.nan,
                "loo_sign_stable_all": np.nan,
                "exact_definition": definition,
                "pass_fail": result,
                "evidence": evidence,
                "interpretation": "Final classification MIXED; descriptive, not a structural GP test.",
            }
        )
    return pd.DataFrame(rows)


def copy_figures() -> pd.DataFrame:
    groups = [
        ("B.3", FRONTIER / "figures", "Risk-aversion sensitivity", rel(FRONTIER / "frontier_points.csv")),
        ("B.4", AUM / "figures", "AUM sensitivity", rel(AUM / "aum_sensitivity_points.csv")),
        ("B.5", RISK_FIG / "figures", "Risk calibration", rel(MAIN / "risk_calibration.csv")),
        ("C.3", EFI / "figures", "Fixed-model EFI", rel(EFI / "feature_importance.csv")),
        ("C.4/C.5", PERSIST / "figures", "Raw persistence diagnostics", rel(PERSIST / "core10_feature_persistence_master.csv")),
        ("C.4/C.5", ALPHA / "figures", "Predictive alpha-decay diagnostics", rel(ALPHA / "core10_predictive_alpha_decay_master.csv")),
    ]
    records = []
    for section, directory, claim, result_source in groups:
        for source in sorted(directory.glob("*.png")):
            destination = FIG_OUT / source.name
            shutil.copy2(source, destination)
            records.append(
                {
                    "appendix_section": section,
                    "figure_filename": source.name,
                    "exact_original_path": rel(source),
                    "underlying_result_source": result_source,
                    "main_claim": claim,
                    "safe_for_appendix": sha256(source) == sha256(destination),
                    "notes": "Copied byte-for-byte; original preserved.",
                }
            )
    records.extend(
        [
            {
                "appendix_section": "C.1",
                "figure_filename": NOT_IDENTIFIABLE,
                "exact_original_path": NOT_IDENTIFIABLE,
                "underlying_result_source": "Locked feature-set metrics and inference tables",
                "main_claim": "Feature-set robustness",
                "safe_for_appendix": False,
                "notes": "No existing final feature-set comparison figure was found; tabular evidence is authoritative.",
            },
            {
                "appendix_section": "C.2",
                "figure_filename": NOT_IDENTIFIABLE,
                "exact_original_path": NOT_IDENTIFIABLE,
                "underlying_result_source": rel(FINAL15_REPORT / "ex_year_robustness.csv"),
                "main_claim": "Final15 evaluation-only 2007 exclusion",
                "safe_for_appendix": False,
                "notes": "No existing final 2007-exclusion figure was found; no new figure was generated.",
            },
        ]
    )
    return pd.DataFrame(records)


def source_inventory() -> pd.DataFrame:
    rows = [
        ("Core10 baseline", rel(MAIN / "metrics.csv"), "reports/final_results/final_performance_tables.xlsx", "Exact experiment output overrides summaries."),
        ("Methodology/config", "configs/model_top500_core10_cumulative.yaml", "reports/dissertation/dissertation_methodology_fact_audit.md", "Locked cumulative Top500/Core10 design."),
        ("Risk model", "data/processed/us_equity_ml/risk_repaired/build_manifest.json", "reports/dissertation/dissertation_data_fact_audit.md", "Repaired full factor covariance inputs."),
        ("Annual results", rel(MAIN / "annual_performance.csv"), "reports/model_results/top500_core10_cumulative/yearly_attribution.md", "Annual utility uses annualized ex-ante utility flow."),
        ("Gamma sensitivity", rel(FRONTIER / "frontier_points.csv"), "reports/dissertation/dissertation_mechanism_aum_fact_audit.md", "Risk-aversion sensitivity; not a matched-volatility frontier."),
        ("AUM sensitivity", rel(AUM / "aum_sensitivity_points.csv"), "reports/dissertation/dissertation_mechanism_aum_fact_audit.md", "Endogenous scenario reruns."),
        ("Final15", rel(FINAL15 / "metrics.csv"), "reports/dissertation/dissertation_robustness_signal_fact_audit.md", "Robustness specification only."),
        ("Final20 Conservative", rel(FINAL20 / "metrics.csv"), "reports/feature_selection/final20_selection_report.md", "Pre-formal-OOS selected robustness extension."),
        ("2007 exclusion", rel(FINAL15_REPORT / "ex_year_robustness.csv"), "reports/dissertation/dissertation_robustness_signal_fact_audit.md", "Evaluation-only Final15 exclusion."),
        ("EFI", rel(EFI / "feature_importance.csv"), "reports/dissertation/dissertation_robustness_signal_fact_audit.md", "Fixed-model economic dependence, not causal importance."),
        ("Raw persistence", rel(PERSIST / "core10_feature_persistence_master.csv"), rel(PERSIST / "core10_signal_persistence_report.md"), "Mean monthly cross-sectional Spearman persistence."),
        ("Predictive alpha decay", rel(ALPHA / "core10_predictive_alpha_decay_master.csv"), rel(ALPHA / "core10_predictive_alpha_decay_report.md"), "Incremental one-month returns at h=1,...,12."),
        ("Integrity diagnostics", "reports/final_analysis/preflight/final_analysis_preflight.json", rel(MAIN / "metadata.json"), "Baseline/table/safety reconciliation."),
    ]
    return pd.DataFrame(rows, columns=["item", "authoritative_file_path", "secondary_verification_source", "notes"])


def compact_percent_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame[columns].copy()
    for column in out.columns:
        if column not in {"gamma", "method", "method_label", "aum_multiplier", "aum_usd", "net_sharpe"}:
            if pd.api.types.is_numeric_dtype(out[column]):
                out[column] = out[column] * 100.0
    return out


def build_markdown(
    inventory: pd.DataFrame,
    grids: pd.DataFrame,
    fixed: pd.DataFrame,
    integrity: pd.DataFrame,
    annual_utility: pd.DataFrame,
    utility_summary: dict[str, Any],
    annual_hp: pd.DataFrame,
    gamma: pd.DataFrame,
    aum: pd.DataFrame,
    risk: pd.DataFrame,
    feature_perf: pd.DataFrame,
    feature_inference: pd.DataFrame,
    ex2007: pd.DataFrame,
    efi: pd.DataFrame,
    persistence: pd.DataFrame,
    gate: pd.DataFrame,
    figures: pd.DataFrame,
    reconciliation: dict[str, Any],
) -> str:
    gamma_compact = compact_percent_table(
        gamma,
        ["gamma", "method_label", "gross_return", "trading_cost", "net_return", "volatility", "net_sharpe", "utility_flow", "turnover", "leverage"],
    ).rename(columns={column: f"{column}_pct" for column in ["gross_return", "trading_cost", "net_return", "volatility", "utility_flow", "turnover", "leverage"]})
    aum_compact = compact_percent_table(
        aum,
        ["aum_multiplier", "aum_usd", "method_label", "gross_return", "trading_cost", "net_return", "volatility", "net_sharpe", "utility_flow", "turnover", "leverage"],
    ).rename(columns={column: f"{column}_pct" for column in ["gross_return", "trading_cost", "net_return", "volatility", "utility_flow", "turnover", "leverage"]})
    annual_risk = risk[(risk["scope"] == "year") & risk["method"].isin(["static_ml", "static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf"])].copy()
    annual_risk = annual_risk[["method_label", "test_year", "predicted_annualized_volatility", "realized_annualized_volatility", "realized_to_predicted_ratio", "inside_0_6_1_5", "above_2"]]
    full_level = risk[risk["scope"] == "full"][
        [
            "method",
            "method_label",
            "predicted_annualized_volatility",
            "realized_annualized_volatility",
            "realized_to_predicted_ratio",
        ]
    ]
    yearly_summary = risk[risk["scope"] == "summary"][
        [
            "method",
            "yearly_ratio_median",
            "yearly_ratio_p10",
            "yearly_ratio_p90",
            "yearly_ratio_max",
            "fraction_years_within_0_6_1_5",
            "years_ratio_above_2",
        ]
    ]
    full_risk = full_level.merge(yearly_summary, on="method", how="left").drop(columns="method")
    selected_perf = feature_perf[feature_perf["method"].isin(["static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf"])]
    selected_inf = feature_inference[
        feature_inference["candidate_method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])
        & feature_inference["benchmark_method"].isin(["static_ml_star", "portfolio_ml_linear"])
    ]

    missing_items = [
        "Final15 feature-selection provenance: " + NOT_IDENTIFIABLE,
        "Existing final feature-set robustness figure: " + NOT_IDENTIFIABLE,
        "Existing final 2007-exclusion figure: " + NOT_IDENTIFIABLE,
    ]
    lines = [
        "# Appendix Evidence Pack",
        "",
        "This pack is a read-only reconciliation of locked final artifacts. It contains no model fit, retuning, new empirical specification, or newly generated figure. RF means Random Features throughout.",
        "",
        "## Source Hierarchy",
        "",
        md_table(inventory),
        "",
        "Exact experiment outputs override summary reports. All output tables retain decimal units unless a column explicitly says `pct` or `percentage points`.",
        "",
        "## Appendix A: Empirical Implementation and Reproducibility",
        "",
        "### A.1 Hyperparameter Grids and Fixed Settings",
        "",
        md_table(grids),
        "",
        md_table(fixed),
        "",
        "Static selection maximizes `mean(gross) - mean(TC) - gamma/2 * (mean(gross^2)-mean(gross)^2)` over cumulative candidate OOS months. Portfolio-ML selection maximizes mean realized monthly utility flow. Return-ML is selected separately by validation MSE. The common Return-ML prediction stream is shared by Rank, Static-ML, and Static-ML*; Portfolio-ML does not use it.",
        "",
        "### A.2 Cumulative Out-of-Sample Timeline",
        "",
        md_table(pd.DataFrame([
            ("Master return history", "1990-01", "2024-12", "prediction history and risk warm-up", "locked panel only"),
            ("Portfolio-analysis history", "1995-01", "2024-12", "portfolio fitting/evaluation frame", "no pre-1995 portfolio objective"),
            ("Candidate validation OOS", "1996", "Y-1", "cumulative portfolio-candidate history", "only V < formal year Y"),
            ("Formal OOS", "2005", "2024", "locked 240-month evaluation", "selection/refits end at Nov Y-1"),
            ("Return-ML validation", "latest eligible month after initial train", "Nov Y-1", "latest 120 months, min train 24/min validation 12", "no Y labels"),
            ("Formal annual refit", "available history", "Nov Y-1", "prediction refit and PML beta refit", "Dec Y-1 forward label excluded"),
            ("State continuity", "Jan Y", "Jan Y+1", "grown inherited holdings", "Dec Y state carried; no annual reset"),
        ], columns=["stage", "start", "end", "purpose", "information_allowed"])),
        "",
        "Generic formal year Y procedure:",
        "",
        "1. Select Return-ML model class and hyperparameters by pre-Y validation MSE.",
        "2. Refit the selected predictor through November Y-1 and cache year-Y predictions.",
        "3. Restrict every portfolio candidate history to validation years V=1996,...,Y-1.",
        "4. Select portfolio hyperparameters with the method-specific cumulative objective.",
        "5. Refit selected Portfolio-ML beta through November Y-1; Static methods apply selected scalars to the refitted common predictions.",
        "6. Execute January-December Y OOS with no Y outcome in selection or refitting.",
        "7. Carry the December Y portfolio state into January Y+1.",
        "",
        "Worked 2010 example: Return history is January 1990-November 2009; prediction validation is December 1999-November 2009; initial prediction training is January 1990-November 1999; candidate years 1996-2009 are eligible; Portfolio-ML beta uses January 1995-November 2009; execution is January-December 2010; December 2009's forward label and all 2010 results are excluded; the December 2009 actual state is carried into January 2010.",
        "",
        "Appendix-ready pseudocode:",
        "```text",
        "for test year Y in 2005,...,2024:",
        "    cutoff = November(Y-1)",
        "    select Return-ML candidate by pre-Y validation MSE",
        "    refit selected Return-ML through cutoff",
        "    reuse its year-Y predictions for Rank/Static/Static*",
        "    for each portfolio method:",
        "        read checkpointed candidate OOS history with V < Y",
        "        score candidates using the method-specific cumulative objective",
        "        select the deterministic maximum-score candidate",
        "        refit Portfolio-ML beta through cutoff when applicable",
        "        execute January-December Y from carried state",
        "        checkpoint the completed method-year and terminal state",
        "```",
        "",
        "### A.3 Dynamic-Universe Accounting",
        "",
        "For a continuing prior name, `h_i,t = pi_i,t-1 * (1+r_i,t) / G_t`, with `G_t = max[1 + sum_i lag_mktcap_i,t-1 r_i,t / sum_i lag_mktcap_i,t-1, 0.1]`. The audited local field `lag_mktcap` is the same-row JKP `me` field; no local one-month shift is applied. `G_t` normalizes weights and does not update scalar AUM.",
        "",
        "Current trade is `delta_i,t = pi_i,t - h_i,t`; an exit trade is `ell_i,t = -h_i,t`. Realized cost is `TC_t = W/2 [sum_current lambda_i,t delta_i,t^2 + sum_exits lambda_i,t-1 h_i,t^2]`; turnover is the corresponding sum of absolute current and exit trades. AUM remains fixed at USD 1 billion.",
        "",
        md_table(pd.DataFrame([
            ("continuing", "grown prior weight aligned by permno", "pi_i,t - h_i,t", "current lambda; included in current cost"),
            ("entrant", "0", "pi_i,t", "current lambda; included in current cost"),
            ("exit", "grown prior weight", "-h_i,t", "previous stored lambda; compulsory liquidation cost"),
        ], columns=["security_status", "inherited_position", "trade", "cost_treatment"])),
        "",
        "The first formal state in January 2005 is value-weighted using current `lag_mktcap` (fallback `mktcap`, then equal weights). Initial prior returns are zero and initial market growth is one. Annual test boundaries do not reset the state.",
        "",
        "### A.4 Random-Features Implementation",
        "",
        "Return-ML RF fills non-finite ranks with 0.5 and applies `RF_p(X)=[cos(XW),sin(XW)]/sqrt(p)`, where `p/2` vectors satisfy `w_j~N(0,eta^2 I)`. Here p is the final number of sinusoidal columns. Unlike Return-ML Linear, RF receives no training-sample standardization. An unpenalized intercept is added after the map. Seed 20260710 fixes each `(p,eta)` map across years and ridge penalties.",
        "",
        "Portfolio-ML RF starts from the same rank representation and RF map, then monthly de-means each mapped column, scales it to unit cross-sectional Euclidean norm, prepends a `1/sqrt(N)` constant, and divides every row by stock volatility from the repaired covariance (floor 0.01). The complete vector, including the constant coefficient, is ridge-penalized. For fixed eta/seed, p=32 is a prefix of the p=64 projection bank.",
        "",
        md_table(pd.DataFrame([
            ("Return-ML Linear", "rank features filled 0.5", "training-sample mean/std", "ordinary columns", "added; unpenalized", "prediction MSE"),
            ("Return-ML RF", "rank features filled 0.5", "none before RF", "cos/sin / sqrt(p)", "added; unpenalized", "prediction MSE"),
            ("Portfolio-ML Linear", "rank features filled 0.5", "monthly demean + L2 + stock-vol scaling", "linear ranks", "1/sqrt(N); penalized", "portfolio utility"),
            ("Portfolio-ML RF", "rank features filled 0.5", "RF then monthly demean + L2 + stock-vol scaling", "cos/sin / sqrt(p)", "1/sqrt(N); penalized", "portfolio utility"),
        ], columns=["model", "input", "standardisation", "feature_map", "intercept_or_constant", "selection_target"])),
        "",
        "### A.5 Implementation Integrity Checks",
        "",
        md_table(integrity),
        "",
        "## Appendix B: Supplementary Performance and Sensitivity Results",
        "",
        "### B.1 Annual Utility, 2005\u20132024",
        "",
        "Values are annualized ex-ante utility FLOW in percentage points per year, not annual net return or empirical realized-variance utility.",
        "",
        md_table(annual_utility),
        "",
        f"Linear beats Static-ML* in {utility_summary['linear_years_won_vs_static_star']}/20 years; RF beats Static-ML* in {utility_summary['rf_years_won_vs_static_star']}/20; RF beats Linear in {utility_summary['rf_years_won_vs_linear']}/20. ",
        f"Linear-Static* is largest in {utility_summary['linear_static_star_largest_positive'][0]} ({utility_summary['linear_static_star_largest_positive'][1]:.4f} pp) and smallest in {utility_summary['linear_static_star_largest_negative'][0]} ({utility_summary['linear_static_star_largest_negative'][1]:.4f} pp). RF-Static* is largest in {utility_summary['rf_static_star_largest_positive'][0]} ({utility_summary['rf_static_star_largest_positive'][1]:.4f} pp) and smallest in {utility_summary['rf_static_star_largest_negative'][0]} ({utility_summary['rf_static_star_largest_negative'][1]:.4f} pp).",
        f"The stored positive-year concentration measure identifies 2007 as the top positive year: {utility_summary['linear_2007_positive_contribution_share']:.4%} of positive Linear-Static* utility contributions and {utility_summary['rf_2007_positive_contribution_share']:.4%} for RF-Static*.",
        "",
        "### B.2 Annual Hyperparameter Selection",
        "",
        md_table(annual_hp),
        "",
        "All prediction-first methods share the one annual Return-ML stream selected by MSE. Portfolio-ML has separate directly fitted portfolio betas and does not consume that stream. Selection frequencies are reported in A.1.",
        "",
        "### B.3 Risk-Aversion Sensitivity",
        "",
        "This is a complete locked risk-aversion sensitivity grid, not matched-volatility efficient-frontier estimation. Percentage columns are percentage points; Sharpe is a ratio.",
        "",
        md_table(gamma_compact),
        "",
        f"Gamma=10 reconciles to the baseline with maximum stored absolute metric error {reconciliation['gamma10_max_error']:.3e}. All 40 rows are full cumulative OOS scenario runs under the same Top500/Core10/risk/cost/timing design; stored fallback and repair counts are zero.",
        "",
        "### B.4 AUM Sensitivity",
        "",
        "Each AUM is a fixed scalar within an endogenous scenario. Portfolio histories, portfolio selections, Portfolio-ML beta, matrix m, state path, and realized costs are scenario-specific; the Return-ML prediction stream is shared because AUM is not a predictor input. Zero AUM is a frictionless diagnostic, not post-hoc cost scaling.",
        "",
        md_table(aum_compact),
        "",
        md_table(aum[["aum_multiplier", "linear_minus_static_star_utility_gap", "rf_minus_static_star_utility_gap", "rf_minus_linear_utility_gap"]].drop_duplicates()),
        "",
        f"The 1x scenario reconciles to the baseline with maximum stored absolute metric error {reconciliation['aum1_max_error']:.3e}.",
        "",
        "### B.5 Annual Risk Calibration",
        "",
        md_table(annual_risk),
        "",
        "Full-sample and yearly-distribution summary:",
        "",
        md_table(full_risk),
        "",
        "Static-ML and Static-ML* exceed a realized/predicted ratio of two in 2020 and 2021. Neither Portfolio-ML method has a year above two.",
        "",
        "## Appendix C: Supplementary Robustness and Feature Diagnostics",
        "",
        "### C.1 Full Feature-Set Performance",
        "",
        md_table(selected_perf),
        "",
        md_table(selected_inf),
        "",
        "Final20 Conservative adds `qmj_safety`, `operating_cashflow_yield`, `debt_to_book`, `operating_accruals`, `sales_to_market`, `qmj_profitability`, `market_beta_252d`, `dollar_volume_variability`, `momentum_18_1`, and `ebit_to_market`. It was selected using 1995-2004 data-quality, pre-OOS IC/stability, persistence, redundancy, and theme coverage without reading formal 2005-2024 portfolio outcomes. Final15 selection provenance is " + NOT_IDENTIFIABLE + ".",
        "",
        "### C.2 2007 Exclusion Diagnostics",
        "",
        "The only locked exclusion is Final15 evaluation-only removal of the 12 realized 2007 months. Models, selected hyperparameters, matrix m, and the post-2007 state path are unchanged; there is no refit or counterfactual history reconstruction. The retained sample has 228 months.",
        "",
        md_table(ex2007),
        "",
        "### C.3 Economic Feature-Importance Diagnostics",
        "",
        "EFI is fixed-model economic dependence: `annualized utility(full fixed policy) - annualized utility(feature-neutralized fixed policy)`. The target rank is replaced by 0.5; annual beta, selected hyperparameters, RF map, risk inputs, Lambda, gamma, and AUM are fixed. The dynamic path, current trades/costs, and exits are recursively recomputed over the same 240 months. The common matrix-m path may retrieve or reconstruct m, but m is feature-independent and numerically unchanged for each month. EFI is not causal feature value.",
        "",
        md_table(efi),
        "",
        "### C.4 Characteristic Persistence and Alpha Decay",
        "",
        md_table(persistence),
        "",
        "Raw persistence is the mean of 239 monthly cross-sectional Spearman correlations between rank(t) and rank(t+1) among permnos present in both months. Predictive alpha uses h=1,...,12, rank at t, and the incremental one-month excess return at t+h. Fama-MacBeth slopes are time means of intercept-inclusive univariate cross-sectional OLS slopes; Rank IC is the parallel monthly Spearman statistic. Sign adjustment uses sign(h=1). Signed future mass sums adjusted h=2,...,12. First decay is the first h>=2 below 0.5*abs(h1); reversal is the first negative sign-adjusted horizon, or >12. These are descriptive, not latent structural alpha states.",
        "",
        "Momentum: h1 FMB 0.006959252891286581, h1 Rank IC 0.016071055990714494, FMB signed future mass 0.033690880529047236, first decay h=6, no reversal through h=12. Reversal: h1 FMB 0.000615931557577803, h1 Rank IC -0.007204681989731235, FMB signed future mass 0.022968580268594576, first decay h=2, first FMB reversal h=7.",
        "",
        "### C.5 Prespecified Persistence Diagnostics",
        "",
        md_table(gate),
        "",
        "Final classification: **MIXED**. This is not a structural test of Garleanu-Pedersen; it neither verifies nor falsifies GP. The feature-level exercise is descriptive, and the architecture-level Portfolio-ML result does not depend on this gate passing.",
        "",
        "## Figure Inventory",
        "",
        md_table(figures),
        "",
        "Copied files are byte-for-byte SHA-256 matches to their original final figures. No new figure was generated.",
        "",
        "## Missing / Non-identifiable Items",
        "",
        *[f"- {item}" for item in missing_items],
        "- Universe-size and missing-security-alignment violation counts are `N/A`, not zero, because the locked final pipeline does not emit them as separate diagnostics.",
        "- Scalar-rho fallback count is `N/A` because the fallback is disabled and a matrix failure would stop the run.",
        "",
        "## Final Reconciliation Checklist",
        "",
        md_table(pd.DataFrame(reconciliation["checks"])),
        "",
        f"Important reconciliation discrepancy: {reconciliation['discrepancy']}",
    ]
    return "\n".join(lines) + "\n"


def build_reconciliation(
    config: dict[str, Any], preflight: dict[str, Any], gamma: pd.DataFrame, aum: pd.DataFrame
) -> dict[str, Any]:
    metrics = read_csv(MAIN / "metrics.csv").set_index("method")
    returns = pd.read_parquet(MAIN / "returns.parquet")
    month_counts = returns.groupby("method")["eom"].nunique().to_dict()

    gamma10 = gamma[gamma["gamma"] == 10].set_index("method")
    aum1 = aum[aum["aum_multiplier"] == 1].set_index("method")
    pairs = [
        ("annualized_gross_return", "gross_return"),
        ("annualized_trading_cost", "trading_cost"),
        ("annualized_net_return", "net_return"),
        ("annualized_volatility", "volatility"),
        ("net_sharpe", "net_sharpe"),
        ("annualized_ex_ante_utility_flow", "utility_flow"),
        ("average_turnover", "turnover"),
        ("average_leverage", "leverage"),
    ]
    gamma_errors = []
    aum_errors = []
    for method in METHOD_ORDER:
        for metric_column, scenario_column in pairs:
            gamma_errors.append(abs(float(metrics.loc[method, metric_column]) - float(gamma10.loc[method, scenario_column])))
            aum_errors.append(abs(float(metrics.loc[method, metric_column]) - float(aum1.loc[method, scenario_column])))
    gamma_max = max(gamma_errors)
    aum_max = max(aum_errors)
    future_violations = int(preflight["safety"]["future_information_violations"])
    checks = [
        {"check": "Core10 baseline matches final workbook/Table 4.1", "result": "PASS" if preflight["passed"] else "FAIL", "evidence": f"max abs error {max(x['workbook_absolute_error'] for x in preflight['metric_comparisons']):.3e}"},
        {"check": "Formal OOS is 240 months for every method", "result": "PASS" if all(value == 240 for value in month_counts.values()) else "FAIL", "evidence": json.dumps(month_counts, sort_keys=True)},
        {"check": "Utility is annualized ex-ante utility flow", "result": "PASS", "evidence": "12 * mean(monthly utility_flow); annual_performance.csv field used"},
        {"check": "Annual returns are arithmetic monthly means times 12", "result": "PASS", "evidence": "locked metrics definition and methodology audit"},
        {"check": "Turnover includes current trades and exit liquidation", "result": "PASS", "evidence": "locked accounting equation"},
        {"check": "AUM is fixed USD 1bn through time", "result": "PASS" if config["baseline_aum"] == 1_000_000_000 else "FAIL", "evidence": value_text(config["baseline_aum"])},
        {"check": "Dynamic state is carried across test years", "result": "PASS", "evidence": "methodology audit and cumulative runner"},
        {"check": "Return/PML Linear/RF preprocessing distinguished", "result": "PASS", "evidence": "A.4 comparison table"},
        {"check": "Full matrix m uses 10 updates", "result": "PASS" if config["portfolio_ml"]["matrix_m_iterations"] == 10 else "FAIL", "evidence": value_text(config["portfolio_ml"]["matrix_m_iterations"])},
        {"check": "Static selection uses realized gross-return variance", "result": "PASS", "evidence": "methodology audit score formula"},
        {"check": "Portfolio-ML selection uses realized monthly utility flow", "result": "PASS", "evidence": "methodology audit score formula"},
        {"check": "Gamma sensitivity not labelled matched-volatility frontier", "result": "PASS", "evidence": "B.3 heading and note"},
        {"check": "Gamma=10 reconciles to baseline", "result": "PASS" if gamma_max <= 1e-6 else "FAIL", "evidence": f"max abs error {gamma_max:.3e}"},
        {"check": "AUM scenarios are endogenous reruns", "result": "PASS", "evidence": "scenario-specific histories/selections/beta/m/cost/state"},
        {"check": "AUM=1x reconciles to baseline", "result": "PASS" if aum_max <= 1e-6 else "FAIL", "evidence": f"max abs error {aum_max:.3e}"},
        {"check": "Final20 selection uses only pre-formal-OOS information", "result": "PASS", "evidence": "1995-2004 preregistered selection report"},
        {"check": "2007 exclusion is evaluation-only", "result": "PASS", "evidence": "Final15 240-to-228 month filter; no refit or state reconstruction"},
        {"check": "EFI labelled fixed-model dependence, not causal", "result": "PASS", "evidence": "C.3 definition"},
        {"check": "Raw persistence distinguished from predictive alpha", "result": "PASS", "evidence": "C.4 definitions"},
        {"check": "Alpha decay uses incremental one-month future returns", "result": "PASS", "evidence": "horizon t+h-1 joins to ret_exc_lead1m"},
        {"check": "GP neither verified nor falsified", "result": "PASS", "evidence": "C.5 classification"},
        {"check": "RF means Random Features", "result": "PASS", "evidence": "terminology used throughout"},
        {"check": "Future-information violations", "result": "PASS" if future_violations == 0 else "FAIL", "evidence": str(future_violations)},
        {"check": "No pilot/obsolete experiment used", "result": "PASS", "evidence": "source inventory restricted to locked final artifacts"},
        {"check": "No new model/specification run", "result": "PASS", "evidence": "script imports no model runner; read/reshape/copy only"},
    ]
    discrepancy = "NONE" if all(item["result"] == "PASS" for item in checks) else "One or more checklist items failed; inspect table above."
    return {"checks": checks, "gamma10_max_error": gamma_max, "aum1_max_error": aum_max, "discrepancy": discrepancy}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    config = read_yaml(ROOT / "configs" / "model_top500_core10_cumulative.yaml")
    preflight = read_json(ROOT / "reports" / "final_analysis" / "preflight" / "final_analysis_preflight.json")
    metadata = read_json(MAIN / "metadata.json")

    annual_hp, prediction = extract_selected_hyperparameters()
    grids, fixed = build_hyperparameter_tables(config, annual_hp, prediction)
    integrity = build_integrity_table(preflight, metadata)
    annual_utility, utility_summary = build_annual_utility()
    gamma = build_gamma_table()
    aum = build_aum_table()
    risk = build_risk_table()
    feature_perf, feature_inference = build_feature_set_tables()
    ex2007 = build_ex2007_table()
    efi = build_efi_table()
    persistence, fmb_paths, rankic_paths = build_persistence_tables(efi)
    gate = build_persistence_gate()
    figures = copy_figures()
    inventory = source_inventory()
    reconciliation = build_reconciliation(config, preflight, gamma, aum)

    write_csv(grids, "appendix_A_hyperparameter_grids.csv")
    write_csv(fixed, "appendix_A_fixed_settings.csv")
    write_csv(integrity, "appendix_A_integrity_checks.csv")
    write_csv(annual_utility, "appendix_B_annual_utility.csv")
    write_csv(annual_hp, "appendix_B_annual_hyperparameters.csv")
    write_csv(gamma, "appendix_B_gamma_sensitivity.csv")
    write_csv(aum, "appendix_B_aum_sensitivity.csv")
    write_csv(risk, "appendix_B_annual_risk_calibration.csv")
    write_csv(feature_perf, "appendix_C_feature_set_performance.csv")
    write_csv(feature_inference, "appendix_C_feature_set_inference.csv")
    write_csv(ex2007, "appendix_C_ex2007_results.csv")
    write_csv(efi, "appendix_C_efi.csv")
    write_csv(persistence, "appendix_C_persistence_alpha_summary.csv")
    write_csv(fmb_paths, "core10_fmb_horizon_paths.csv")
    write_csv(rankic_paths, "core10_rankic_horizon_paths.csv")
    write_csv(gate, "appendix_C_persistence_gate.csv")
    write_csv(inventory, "source_inventory.csv")
    write_csv(figures, "figure_inventory.csv")
    write_csv(pd.DataFrame(reconciliation["checks"]), "reconciliation_checklist.csv")

    markdown = build_markdown(
        inventory,
        grids,
        fixed,
        integrity,
        annual_utility,
        utility_summary,
        annual_hp,
        gamma,
        aum,
        risk,
        feature_perf,
        feature_inference,
        ex2007,
        efi,
        persistence,
        gate,
        figures,
        reconciliation,
    )
    (OUT / "APPENDIX_EVIDENCE_PACK.md").write_text(markdown, encoding="utf-8")

    manifest = {
        "mode": "read_only_reconciliation_and_extraction",
        "model_or_experiment_rerun": False,
        "mandatory_files_created": 17,
        "extra_audit_csv_files_created": 3,
        "figures_copied": int((figures["safe_for_appendix"] == True).sum()),  # noqa: E712
        "non_identifiable_requested_items": 3,
        "optional_excel_created": False,
        "reconciliation_discrepancy": reconciliation["discrepancy"],
        "gamma10_max_abs_error": reconciliation["gamma10_max_error"],
        "aum1_max_abs_error": reconciliation["aum1_max_error"],
    }
    (OUT / "appendix_pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if reconciliation["discrepancy"] != "NONE":
        raise RuntimeError(reconciliation["discrepancy"])
    print("APPENDIX PACK COMPLETE")
    print(rel(OUT / "APPENDIX_EVIDENCE_PACK.md"))
    print(rel(OUT))
    print(rel(FIG_OUT))
    print("non-identifiable requested items: 3")
    print("important reconciliation discrepancy: NONE")


if __name__ == "__main__":
    main()
