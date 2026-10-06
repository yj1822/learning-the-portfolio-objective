from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
RISK_ROOT = PROJECT_ROOT / "data/processed/us_equity_ml/risk_repaired"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/model_results/top500_core10_cumulative"
METHODS = ("static_ml", "static_ml_star")
DAILY_TO_MONTHLY = 21.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Static-ML risk calibration from completed outputs only."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--specification", default="Top500/Core10")
    args = parser.parse_args()
    results = args.results_dir.resolve()
    report_root = args.report_dir.resolve()

    annual = pd.read_csv(results / "annual_performance.csv")
    risk = pd.read_csv(results / "risk_calibration.csv")
    selected = pd.read_csv(results / "selected_hyperparameters_by_year.csv")
    returns = pd.read_parquet(results / "returns.parquet")
    weights = pd.read_parquet(results / "weights.parquet")

    diagnostic = build_yearly_diagnostic(
        annual=annual,
        risk=risk,
        selected=selected,
        returns=returns,
        weights=weights,
        exposures=pd.read_parquet(RISK_ROOT / "factor_exposures.parquet"),
        factor_covariance=pd.read_parquet(RISK_ROOT / "factor_cov_monthly.parquet"),
        idio_variance=pd.read_parquet(RISK_ROOT / "idio_var_monthly.parquet"),
    )
    _validate_output(diagnostic)
    report_root.mkdir(parents=True, exist_ok=True)
    output_csv = report_root / "static_risk_calibration_diagnostic.csv"
    output_md = report_root / "static_risk_calibration_diagnostic.md"
    diagnostic.to_csv(output_csv, index=False)
    output_md.write_text(
        build_report(diagnostic, specification=str(args.specification)),
        encoding="utf-8",
    )
    print(f"csv={output_csv}")
    print(f"report={output_md}")
    print(f"ratio_gt_2={int(diagnostic['ratio_gt_2'].sum())}")
    return 0


def build_yearly_diagnostic(
    *,
    annual: pd.DataFrame,
    risk: pd.DataFrame,
    selected: pd.DataFrame,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    idio_variance: pd.DataFrame,
) -> pd.DataFrame:
    risk_year = risk[
        risk["scope"].eq("year") & risk["method"].isin(METHODS)
    ].copy()
    risk_year["year"] = risk_year["test_year"].astype(int)
    annual_static = annual[annual["method"].isin(METHODS)].copy()
    annual_static["year"] = annual_static["test_year"].astype(int)
    chosen = selected[selected["method"].isin(METHODS)].copy()
    chosen["year"] = chosen["test_year"].astype(int)
    chosen["selected_hyperparameters"] = chosen["candidate_hyperparameters"].astype(str)
    parameter_rows = chosen["selected_hyperparameters"].map(_parameter_columns)
    parameters = pd.DataFrame(parameter_rows.tolist(), index=chosen.index)
    chosen = pd.concat([chosen, parameters], axis=1)

    factor = factor_risk_diagnostics(
        returns=returns,
        weights=weights,
        exposures=exposures,
        factor_covariance=factor_covariance,
        idio_variance=idio_variance,
    )
    result = risk_year.merge(
        annual_static,
        on=["method", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_annual"),
    ).merge(
        chosen[
            [
                "method",
                "year",
                "selected_hyperparameters",
                "phi",
                "mu_shrinkage",
                "u",
                "v",
                "k",
            ]
        ],
        on=["method", "year"],
        how="left",
        validate="one_to_one",
    ).merge(
        factor,
        on=["method", "year"],
        how="left",
        validate="one_to_one",
    )
    result = result.rename(
        columns={
            "predicted_annualized_volatility": "predicted_vol",
            "realized_annualized_volatility": "realized_vol",
            "realized_to_predicted_ratio": "realized_predicted_ratio",
            "average_turnover": "turnover",
            "average_leverage": "leverage",
            "annualized_trading_cost": "trading_cost",
            "annualized_net_return": "net_return",
            "annualized_ex_ante_utility_flow": "utility",
        }
    )
    result["ratio_gt_2"] = result["realized_predicted_ratio"].gt(2.0)
    result["variance_ratio_realized_to_predicted"] = (
        result["realized_predicted_ratio"] ** 2
    )
    result["volatility_gap"] = result["realized_vol"] - result["predicted_vol"]
    result["episode"] = result["year"].map(_episode)
    columns = [
        "year",
        "method",
        "predicted_vol",
        "realized_vol",
        "realized_predicted_ratio",
        "ratio_gt_2",
        "episode",
        "selected_hyperparameters",
        "phi",
        "mu_shrinkage",
        "u",
        "v",
        "k",
        "turnover",
        "leverage",
        "trading_cost",
        "net_return",
        "utility",
        "volatility_gap",
        "variance_ratio_realized_to_predicted",
        "predicted_factor_vol",
        "predicted_idio_vol",
        "annualized_factor_variance_contribution",
        "annualized_idio_variance_contribution",
        "factor_variance_share",
        "factor_exposure_l2_mean",
        "factor_exposure_l2_max",
        "style_exposure_l2_mean",
        "industry_exposure_l2_mean",
        "intercept_exposure_abs_mean",
        "dominant_factor",
        "dominant_factor_abs_exposure",
        "max_abs_monthly_net_return",
        "largest_month_variance_share",
        "risk_reconstruction_max_relative_error",
        "covariance_condition_number",
        "covariance_min_eigenvalue",
        "covariance_repair_count",
        "factor_risk_fallback_count",
        "risk_failure_source",
    ]
    result["risk_failure_source"] = "within_threshold"
    for method, group in result.groupby("method", sort=False):
        normal = group[~group["ratio_gt_2"]]
        leverage_reference = float(normal["leverage"].median())
        for index, row in group[group["ratio_gt_2"]].iterrows():
            if row["leverage"] > 1.5 * max(leverage_reference, 1.0e-12):
                source = "position_scaling"
            elif row["factor_variance_share"] >= 0.5:
                source = "factor_covariance_underestimation_or_regime_shock"
            else:
                source = "idiosyncratic_covariance_underestimation_or_regime_shock"
            result.loc[index, "risk_failure_source"] = source
    return result[columns].sort_values(["year", "method"]).reset_index(drop=True)


def factor_risk_diagnostics(
    *,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    idio_variance: pd.DataFrame,
) -> pd.DataFrame:
    static_weights = weights[weights["method"].isin(METHODS)].copy()
    static_returns = returns[returns["method"].isin(METHODS)].copy()
    for frame in (static_weights, static_returns, exposures, factor_covariance, idio_variance):
        frame["eom"] = pd.to_datetime(frame["eom"])
    exposures = exposures[exposures["universe"].eq("top500")].copy()
    factor_covariance = factor_covariance[
        factor_covariance["universe"].eq("top500")
    ].copy()
    idio_variance = idio_variance[idio_variance["universe"].eq("top500")].copy()
    factors = sorted(
        set(factor_covariance["factor_1"])
        & set(factor_covariance["factor_2"])
        & set(exposures.columns)
    )
    style_indices = [i for i, value in enumerate(factors) if value.startswith("rank_")]
    industry_indices = [
        i for i, value in enumerate(factors) if value.startswith("industry_")
    ]
    intercept_index = factors.index("intercept") if "intercept" in factors else None
    exposure_by_month = {
        eom: group.set_index("permno")
        for eom, group in exposures.groupby("eom", sort=False)
    }
    idio_by_month = {
        eom: group.set_index("permno")["idio_var"]
        for eom, group in idio_variance.groupby("eom", sort=False)
    }
    covariance_by_month = {
        eom: _factor_covariance_matrix(group, factors)
        for eom, group in factor_covariance.groupby("eom", sort=False)
    }
    stored_variance = static_returns.set_index(["eom", "method"])[
        "ex_ante_monthly_variance"
    ]
    net_returns = static_returns.set_index(["eom", "method"])["net_return"]
    monthly_rows: list[dict[str, object]] = []
    factor_exposure_rows: list[dict[str, object]] = []
    for (eom, method), group in static_weights.groupby(["eom", "method"], sort=True):
        month_exposures = exposure_by_month[pd.Timestamp(eom)]
        permnos = group["permno"].astype(int).to_numpy()
        loadings = month_exposures.reindex(permnos)[factors].fillna(0.0).to_numpy(float)
        portfolio_weights = group["weight"].to_numpy(float)
        factor_exposure = portfolio_weights @ loadings
        factor_cov = covariance_by_month[pd.Timestamp(eom)] * DAILY_TO_MONTHLY
        factor_variance = max(float(factor_exposure @ factor_cov @ factor_exposure), 0.0)
        idio = pd.to_numeric(
            idio_by_month[pd.Timestamp(eom)].reindex(permnos), errors="coerce"
        )
        valid = idio.notna() & np.isfinite(idio) & idio.gt(0.0)
        fallback = float(idio[valid].median())
        idio_values = idio.where(valid, fallback).to_numpy(float) * DAILY_TO_MONTHLY
        idio_component = max(float(np.sum(portfolio_weights**2 * idio_values)), 0.0)
        reconstructed = factor_variance + idio_component
        stored = float(stored_variance.loc[(pd.Timestamp(eom), method)])
        monthly_rows.append(
            {
                "eom": pd.Timestamp(eom),
                "year": int(group["test_year"].iloc[0]),
                "method": method,
                "factor_variance": factor_variance,
                "idio_variance": idio_component,
                "reconstructed_variance": reconstructed,
                "stored_variance": stored,
                "risk_reconstruction_relative_error": abs(reconstructed - stored)
                / max(abs(stored), 1.0e-16),
                "factor_exposure_l2": float(np.linalg.norm(factor_exposure)),
                "style_exposure_l2": float(np.linalg.norm(factor_exposure[style_indices])),
                "industry_exposure_l2": float(
                    np.linalg.norm(factor_exposure[industry_indices])
                ),
                "intercept_exposure_abs": (
                    abs(float(factor_exposure[intercept_index]))
                    if intercept_index is not None
                    else np.nan
                ),
                "net_return": float(net_returns.loc[(pd.Timestamp(eom), method)]),
            }
        )
        factor_exposure_rows.append(
            {
                "year": int(group["test_year"].iloc[0]),
                "method": method,
                **{
                    factor: abs(float(value))
                    for factor, value in zip(factors, factor_exposure, strict=True)
                },
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    factor_values = pd.DataFrame(factor_exposure_rows)
    rows: list[dict[str, object]] = []
    for (method, year), group in monthly.groupby(["method", "year"], sort=True):
        exposure_group = factor_values[
            factor_values["method"].eq(method) & factor_values["year"].eq(year)
        ]
        mean_abs_exposures = exposure_group[factors].mean()
        dominant = str(mean_abs_exposures.idxmax())
        centered = group["net_return"] - group["net_return"].mean()
        squared = centered**2
        rows.append(
            {
                "method": method,
                "year": int(year),
                "predicted_factor_vol": float(
                    np.sqrt(12.0 * group["factor_variance"].mean())
                ),
                "predicted_idio_vol": float(
                    np.sqrt(12.0 * group["idio_variance"].mean())
                ),
                "annualized_factor_variance_contribution": float(
                    12.0 * group["factor_variance"].mean()
                ),
                "annualized_idio_variance_contribution": float(
                    12.0 * group["idio_variance"].mean()
                ),
                "factor_variance_share": float(
                    group["factor_variance"].sum()
                    / max(group["reconstructed_variance"].sum(), 1.0e-16)
                ),
                "factor_exposure_l2_mean": float(group["factor_exposure_l2"].mean()),
                "factor_exposure_l2_max": float(group["factor_exposure_l2"].max()),
                "style_exposure_l2_mean": float(group["style_exposure_l2"].mean()),
                "industry_exposure_l2_mean": float(group["industry_exposure_l2"].mean()),
                "intercept_exposure_abs_mean": float(
                    group["intercept_exposure_abs"].mean()
                ),
                "dominant_factor": dominant,
                "dominant_factor_abs_exposure": float(mean_abs_exposures[dominant]),
                "max_abs_monthly_net_return": float(group["net_return"].abs().max()),
                "largest_month_variance_share": float(
                    squared.max() / max(squared.sum(), 1.0e-16)
                ),
                "risk_reconstruction_max_relative_error": float(
                    group["risk_reconstruction_relative_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(diagnostic: pd.DataFrame, *, specification: str) -> str:
    abnormal = diagnostic[diagnostic["ratio_gt_2"]].copy()
    abnormal_columns = [
        "year",
        "method",
        "predicted_vol",
        "realized_vol",
        "realized_predicted_ratio",
        "phi",
        "mu_shrinkage",
        "u",
        "v",
        "k",
        "turnover",
        "leverage",
        "factor_variance_share",
        "annualized_factor_variance_contribution",
        "annualized_idio_variance_contribution",
        "factor_exposure_l2_mean",
        "dominant_factor",
        "largest_month_variance_share",
        "risk_failure_source",
        "covariance_condition_number",
    ]
    episode = diagnostic[diagnostic["year"].isin([2008, 2020, 2021, 2022])][
        [
            "year",
            "method",
            "realized_predicted_ratio",
            "ratio_gt_2",
            "turnover",
            "leverage",
            "factor_variance_share",
        ]
    ]
    comparison_rows: list[dict[str, object]] = []
    for method, group in diagnostic.groupby("method", sort=True):
        for label, subset in (
            ("ratio_gt_2", group[group["ratio_gt_2"]]),
            ("other_years", group[~group["ratio_gt_2"]]),
        ):
            comparison_rows.append(
                {
                    "method": method,
                    "sample": label,
                    "years": int(len(subset)),
                    "mean_ratio": subset["realized_predicted_ratio"].mean(),
                    "mean_turnover": subset["turnover"].mean(),
                    "mean_leverage": subset["leverage"].mean(),
                    "mean_factor_variance_share": subset["factor_variance_share"].mean(),
                    "mean_factor_exposure_l2": subset["factor_exposure_l2_mean"].mean(),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    abnormal_years = sorted(abnormal["year"].unique().tolist())
    crisis_years = sorted(set(abnormal_years) & {2008, 2009, 2020, 2021, 2022})
    other_years = sorted(set(abnormal_years) - set(crisis_years))
    parameter_text = _parameter_diagnosis(diagnostic)
    exposure_text = _exposure_diagnosis(comparison)
    source_counts = abnormal["risk_failure_source"].value_counts().to_dict()
    crisis_text = (
        f"Crisis/regime years among the breaches are {crisis_years}; other breach years are "
        f"{other_years}."
    )
    return (
        f"# {specification} Static-ML risk calibration diagnostic\n\n"
        f"This report is computed only from the existing {specification} formal "
        "outputs and repaired factor-risk inputs. No model was refitted and no "
        "backtest was rerun. Predicted volatility is sqrt(12 times mean monthly "
        "ex-ante variance); realized volatility is sqrt(12) times the sample "
        "standard deviation of monthly net returns.\n\n"
        "## Findings\n\n"
        f"- There are {len(abnormal)} method-years with realized/predicted "
        f"volatility above 2: {abnormal_years}.\n"
        f"- {crisis_text}\n"
        f"- {parameter_text}\n"
        f"- {exposure_text}\n"
        f"- Failure-source classification counts are {source_counts}. This is a "
        "diagnostic attribution based on leverage and the factor/idiosyncratic "
        "share, not a causal model.\n"
        "- Factor-plus-idiosyncratic variance reconstruction matches the stored "
        "ex-ante variance to numerical tolerance. The gap is therefore not an "
        "accounting or covariance assembly bug; it is an ex-ante calibration "
        "miss during a volatility/regime shock.\n\n"
        "## Ratio above two\n\n"
        + _markdown(abnormal[abnormal_columns])
        + "\n\n## Crisis-year comparison\n\n"
        + _markdown(episode)
        + "\n\n## Abnormal versus other years\n\n"
        + _markdown(comparison)
        + "\n\n## Interpretation\n\n"
        "The classification above separates an unusually large position scale from "
        "a covariance calibration miss. Parameter recurrence in normal and breach "
        "years is reported explicitly, so a selected parameter is not treated as a "
        "cause merely because it appears in an abnormal year.\n\n"
        "The complete 40-row method-year table, including selected parameters and "
        "risk decomposition fields, is saved beside this report.\n"
    )


def _parameter_diagnosis(diagnostic: pd.DataFrame) -> str:
    static = diagnostic[diagnostic["method"].eq("static_ml")]
    star = diagnostic[diagnostic["method"].eq("static_ml_star")]
    static_bad = static[static["ratio_gt_2"]]
    star_bad = star[star["ratio_gt_2"]]
    static_params = sorted(
        {
            f"phi={row.phi:g}, mu_shrinkage={row.mu_shrinkage:g}"
            for row in static_bad.itertuples()
        }
    )
    star_params = sorted(
        {
            f"u={row.u:g}, v={row.v:g}, k={row.k:g}"
            for row in star_bad.itertuples()
        }
    )
    static_all = static.groupby(["phi", "mu_shrinkage"], dropna=False).size()
    star_all = star.groupby(["u", "v", "k"], dropna=False).size()
    static_bad_counts = static_bad.groupby(["phi", "mu_shrinkage"], dropna=False).size()
    star_bad_counts = star_bad.groupby(["u", "v", "k"], dropna=False).size()
    recurrence = []
    for key, count in static_bad_counts.items():
        recurrence.append(f"Static {key}: {int(count)}/{int(static_all.loc[key])} breaches")
    for key, count in star_bad_counts.items():
        recurrence.append(f"Static* {key}: {int(count)}/{int(star_all.loc[key])} breaches")
    return (
        "Static-ML breach settings are "
        + (", ".join(static_params) if static_params else "none")
        + "; Static-ML* breach settings are "
        + (", ".join(star_params) if star_params else "none")
        + ". Parameter recurrence: "
        + ("; ".join(recurrence) if recurrence else "no breaches")
    )


def _exposure_diagnosis(comparison: pd.DataFrame) -> str:
    pieces: list[str] = []
    for method in METHODS:
        rows = comparison[comparison["method"].eq(method)].set_index("sample")
        if "ratio_gt_2" not in rows.index:
            pieces.append(f"{method}: no ratio-above-two years")
            continue
        bad = rows.loc["ratio_gt_2"]
        normal = rows.loc["other_years"]
        pieces.append(
            f"{method}: breach-year leverage is {bad.mean_leverage:.3f} versus "
            f"{normal.mean_leverage:.3f}, turnover {bad.mean_turnover:.3f} versus "
            f"{normal.mean_turnover:.3f}, and factor-exposure L2 "
            f"{bad.mean_factor_exposure_l2:.3f} versus "
            f"{normal.mean_factor_exposure_l2:.3f}; factor-risk share is "
            f"{bad.mean_factor_variance_share:.3f} versus "
            f"{normal.mean_factor_variance_share:.3f}"
        )
    return "; ".join(pieces)


def _validate_output(diagnostic: pd.DataFrame) -> None:
    required = {
        "year",
        "method",
        "predicted_vol",
        "realized_vol",
        "realized_predicted_ratio",
        "selected_hyperparameters",
        "turnover",
        "leverage",
        "trading_cost",
        "net_return",
        "utility",
    }
    missing = required - set(diagnostic.columns)
    if missing:
        raise RuntimeError(f"Diagnostic output is missing columns: {sorted(missing)}")
    if len(diagnostic) != 40 or diagnostic[["year", "method"]].duplicated().any():
        raise RuntimeError("Expected one row for each of 20 years and two Static methods")
    if set(diagnostic["method"]) != set(METHODS):
        raise RuntimeError("Diagnostic output has unexpected methods")
    if not np.isfinite(
        diagnostic[["predicted_vol", "realized_vol", "realized_predicted_ratio"]]
    ).all().all():
        raise RuntimeError("Risk calibration output contains non-finite values")
    reconstruction_error = float(
        diagnostic["risk_reconstruction_max_relative_error"].max()
    )
    if reconstruction_error > 1.0e-10:
        raise RuntimeError(
            f"Factor-risk reconstruction error exceeds tolerance: {reconstruction_error}"
        )


def _parameter_columns(value: str) -> dict[str, float | None]:
    parameters = json.loads(value)
    return {
        name: float(parameters[name]) if parameters.get(name) is not None else np.nan
        for name in ("phi", "mu_shrinkage", "u", "v", "k")
    }


def _factor_covariance_matrix(group: pd.DataFrame, factors: list[str]) -> np.ndarray:
    index = {factor: position for position, factor in enumerate(factors)}
    matrix = np.zeros((len(factors), len(factors)), dtype=float)
    for row in group.itertuples(index=False):
        if row.factor_1 not in index or row.factor_2 not in index:
            continue
        i, j = index[row.factor_1], index[row.factor_2]
        matrix[i, j] = float(row.covariance)
        matrix[j, i] = float(row.covariance)
    return matrix


def _episode(year: int) -> str:
    if year in (2008, 2009):
        return "GFC"
    if year == 2020:
        return "COVID shock"
    if year == 2021:
        return "reopening/rotation"
    if year == 2022:
        return "inflation/tightening"
    return "other"


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    data = frame.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
