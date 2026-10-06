from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.data.config import load_data_config  # noqa: E402
from implementable_frontier.data.loaders import (  # noqa: E402
    load_master_panel,
    load_risk_inputs,
)
from implementable_frontier.models.config import load_model_config  # noqa: E402
from implementable_frontier.models.features import (  # noqa: E402
    monthly_cross_sectional_standardize,
    volatility_scale_design,
)
from implementable_frontier.models.formal_analysis import (  # noqa: E402
    monthly_utility_differences,
    risk_calibration_report,
    statistical_inference,
)
from implementable_frontier.models.metrics import performance_metrics  # noqa: E402
from implementable_frontier.models.risk import RiskModelProvider  # noqa: E402


FINAL_RESULTS = PROJECT_ROOT / "results/experiments/top500_final15_factor_cumulative"
CORE_RESULTS = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT_DIR = PROJECT_ROOT / "reports/model_results/top500_final15_cumulative"
CONFIG_PATH = PROJECT_ROOT / "configs/model_top500_final15_cumulative.yaml"
METHODS = ("static_ml", "static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf")
PORTFOLIO_COMPARISONS = {
    ("portfolio_ml_linear", "static_ml"),
    ("portfolio_ml_linear", "static_ml_star"),
    ("portfolio_ml_rf", "static_ml"),
    ("portfolio_ml_rf", "static_ml_star"),
}
SAMPLES = {
    "full_2005_2024": (),
    "exclude_2007": (2007,),
    "exclude_2020_2021": (2020, 2021),
    "exclude_2007_2020_2021": (2007, 2020, 2021),
    "exclude_2007_2020_2022": (2007, 2020, 2021, 2022),
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_model_config(CONFIG_PATH)
    values = config.values
    gamma = float(values["baseline_gamma"])
    robustness = values["robustness"]

    final_returns = _load_complete_returns(FINAL_RESULTS / "returns.parquet")
    core_returns = _load_complete_returns(CORE_RESULTS / "returns.parquet")
    selection = pd.read_csv(FINAL_RESULTS / "selected_hyperparameters_by_year.csv")

    ex_year, ex_context = build_ex_year_robustness(
        final_returns,
        core_returns,
        gamma=gamma,
        newey_west_lags=int(robustness["newey_west_lags"]),
        bootstrap_block_months=int(robustness["bootstrap_block_months"]),
        bootstrap_replications=int(robustness["bootstrap_replications"]),
        seed=int(values["random_seed"]),
    )
    ex_year.to_csv(REPORT_DIR / "ex_year_robustness.csv", index=False)
    (REPORT_DIR / "ex_year_robustness.md").write_text(
        build_ex_year_report(ex_year, ex_context), encoding="utf-8"
    )

    decomposition, decomposition_summary = build_yearly_decomposition()
    decomposition.to_csv(
        REPORT_DIR / "core10_vs_final15_yearly_decomposition.csv", index=False
    )
    (REPORT_DIR / "core10_vs_final15_yearly_decomposition.md").write_text(
        build_decomposition_report(decomposition, decomposition_summary),
        encoding="utf-8",
    )

    feature_output, feature_context = build_feature_diagnostics(config)
    feature_output.to_csv(REPORT_DIR / "final15_feature_diagnostics.csv", index=False)
    (REPORT_DIR / "final15_feature_diagnostics.md").write_text(
        build_feature_report(feature_output, feature_context), encoding="utf-8"
    )

    print(f"ex_year_rows={len(ex_year)}")
    print(f"decomposition_rows={len(decomposition)}")
    print(f"feature_rows={len(feature_output)}")
    print(f"report_dir={REPORT_DIR}")
    return 0


def _load_complete_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["eom"] = pd.to_datetime(frame["eom"])
    if frame.duplicated(["eom", "method"]).any():
        raise RuntimeError(f"Duplicate method-month rows in {path}")
    expected_years = set(range(2005, 2025))
    for method, group in frame.groupby("method", sort=True):
        years = set(group["test_year"].astype(int))
        if years != expected_years or len(group) != 240:
            raise RuntimeError(
                f"Incomplete result for {method} in {path}: {len(group)} rows, years={sorted(years)}"
            )
    return frame.sort_values(["eom", "method"]).reset_index(drop=True)


def build_ex_year_robustness(
    final_returns: pd.DataFrame,
    core_returns: pd.DataFrame,
    *,
    gamma: float,
    newey_west_lags: int,
    bootstrap_block_months: int,
    bootstrap_replications: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[pd.DataFrame] = []
    sample_inference: dict[str, pd.DataFrame] = {}
    sample_risk: dict[str, pd.DataFrame] = {}
    for sample_index, (sample, excluded) in enumerate(SAMPLES.items()):
        retained = final_returns[~final_returns["test_year"].isin(excluded)].copy()
        metrics = performance_metrics(retained, gamma=gamma)
        metrics.insert(0, "row_type", "method_metrics")
        metrics.insert(0, "excluded_years", ",".join(map(str, excluded)))
        metrics.insert(0, "sample", sample)
        rows.append(metrics)

        monthly = monthly_utility_differences(retained, gamma=gamma)
        inference = statistical_inference(
            monthly,
            newey_west_lags=newey_west_lags,
            bootstrap_block_months=bootstrap_block_months,
            bootstrap_replications=bootstrap_replications,
            seed=seed + sample_index,
        )
        inference = inference[
            inference["outcome"].eq("utility_flow")
            & inference.apply(
                lambda row: (row["candidate_method"], row["benchmark_method"])
                in PORTFOLIO_COMPARISONS,
                axis=1,
            )
        ].copy()
        sample_inference[sample] = inference
        inference.insert(0, "row_type", "utility_inference")
        inference.insert(0, "excluded_years", ",".join(map(str, excluded)))
        inference.insert(0, "sample", sample)
        rows.append(inference)

        risk = risk_calibration_report(retained)
        risk = risk[
            risk["scope"].eq("summary")
            & risk["method"].isin(("static_ml", "static_ml_star"))
        ].copy()
        sample_risk[sample] = risk
        risk.insert(0, "row_type", "static_risk_summary")
        risk.insert(0, "excluded_years", ",".join(map(str, excluded)))
        risk.insert(0, "sample", sample)
        rows.append(risk)

    final_concentration = _utility_concentration(final_returns, gamma)
    core_concentration = _utility_concentration(core_returns, gamma)
    context = {
        "inference": sample_inference,
        "risk": sample_risk,
        "final_concentration": final_concentration,
        "core_concentration": core_concentration,
    }
    return pd.concat(rows, ignore_index=True, sort=False), context


def _utility_concentration(returns: pd.DataFrame, gamma: float) -> pd.DataFrame:
    monthly = monthly_utility_differences(returns, gamma=gamma)
    annual = (
        monthly.groupby(
            ["candidate_method", "benchmark_method", "test_year"], sort=True
        )["delta_utility_flow"]
        .sum()
        .reset_index(name="annual_utility_difference")
    )
    rows: list[dict[str, object]] = []
    for keys, group in annual.groupby(
        ["candidate_method", "benchmark_method"], sort=True
    ):
        if keys not in PORTFOLIO_COMPARISONS:
            continue
        positive = group[group["annual_utility_difference"] > 0].sort_values(
            "annual_utility_difference", ascending=False
        )
        denominator = float(positive["annual_utility_difference"].sum())
        row: dict[str, object] = {
            "candidate_method": keys[0],
            "benchmark_method": keys[1],
            "years_won": int(len(positive)),
            "positive_utility_sum": denominator,
            "year_2007_difference": float(
                group.loc[group["test_year"].eq(2007), "annual_utility_difference"].iloc[0]
            ),
        }
        for count in (1, 3, 5):
            row[f"top_{count}_positive_year_share"] = (
                float(positive.head(count)["annual_utility_difference"].sum() / denominator)
                if denominator > 0
                else np.nan
            )
        row["year_2007_positive_share"] = max(row["year_2007_difference"], 0.0) / max(
            denominator, 1.0e-16
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_yearly_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(CORE_RESULTS / "annual_performance.csv")
    final = pd.read_csv(FINAL_RESULTS / "annual_performance.csv")
    core = core[core["method"].isin(METHODS)].copy()
    final = final[final["method"].isin(METHODS)].copy()
    values = {
        "net_return": "annualized_net_return",
        "utility": "annualized_ex_ante_utility_flow",
        "volatility": "annualized_volatility",
        "turnover": "average_turnover",
        "leverage": "average_leverage",
        "trading_cost": "annualized_trading_cost",
    }
    left = core[["method", "test_year", *values.values()]]
    right = final[["method", "test_year", *values.values()]]
    merged = left.merge(
        right,
        on=["method", "test_year"],
        how="inner",
        validate="one_to_one",
        suffixes=("_core10", "_final15"),
    ).rename(columns={"test_year": "year"})
    for short, column in values.items():
        merged[f"{short}_final15_minus_core10"] = (
            merged[f"{column}_final15"] - merged[f"{column}_core10"]
        )
    if len(merged) != 80:
        raise RuntimeError(f"Expected 80 method-year decomposition rows, found {len(merged)}")

    summary_rows: list[dict[str, object]] = []
    for method, group in merged.groupby("method", sort=True):
        positive = group[group["utility_final15_minus_core10"] > 0].sort_values(
            "utility_final15_minus_core10", ascending=False
        )
        positive_sum = float(positive["utility_final15_minus_core10"].sum())
        year_2007 = float(
            group.loc[group["year"].eq(2007), "utility_final15_minus_core10"].iloc[0]
        )
        summary_rows.append(
            {
                "method": method,
                **{
                    f"mean_{short}_delta": float(
                        group[f"{short}_final15_minus_core10"].mean()
                    )
                    for short in values
                },
                "utility_positive_years": int(len(positive)),
                "volatility_higher_years": int(
                    (group["volatility_final15_minus_core10"] > 0).sum()
                ),
                "year_2007_utility_delta": year_2007,
                "year_2007_positive_share": max(year_2007, 0.0)
                / max(positive_sum, 1.0e-16),
                "top_3_positive_year_share": float(
                    positive.head(3)["utility_final15_minus_core10"].sum()
                    / max(positive_sum, 1.0e-16)
                ),
                "best_utility_year": int(positive.iloc[0]["year"])
                if len(positive)
                else np.nan,
            }
        )
    return merged.sort_values(["year", "method"]), pd.DataFrame(summary_rows)


def build_feature_diagnostics(config) -> tuple[pd.DataFrame, dict[str, object]]:
    data_config = load_data_config(config.resolve_path(config.values["data_config"]))
    core = list(data_config.core10)
    added = [feature for feature in data_config.final15 if feature not in core]
    rank_columns = [f"rank_{feature}" for feature in data_config.final15]
    raw_columns = [f"raw_{feature}" for feature in data_config.final15]
    columns = [
        "eom",
        "permno",
        "ret_exc_lead1m",
        "vol_12m",
        *raw_columns,
        *rank_columns,
    ]
    panel = load_master_panel(
        start="1995-01-31",
        end="2024-12-31",
        universe="top500",
        feature_set="final15",
        columns=columns,
    )
    panel["year"] = panel["eom"].dt.year.astype(int)

    monthly_ic, monthly_std = _monthly_feature_statistics(panel, added)
    annual_ic = monthly_ic.groupby("year", sort=True)[added].mean()
    annual_std = monthly_std.groupby("year", sort=True)[added].mean()
    predictions = pd.read_parquet(FINAL_RESULTS / "predictions.parquet")
    prediction_relation = _prediction_relations(panel, predictions, added)
    beta_yearly, beta_quality = _recover_linear_beta_contributions(
        panel, predictions, config, rank_columns
    )

    summary_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for feature in added:
        raw = pd.to_numeric(panel[f"raw_{feature}"], errors="coerce")
        rank = pd.to_numeric(panel[f"rank_{feature}"], errors="coerce")
        missing = raw.isna()
        core_correlations = {
            core_feature: float(rank.corr(panel[f"rank_{core_feature}"]))
            for core_feature in core
        }
        annual_values = annual_ic[feature].dropna()
        absolute = annual_values.abs().sort_values(ascending=False)
        absolute_sum = float(absolute.sum())
        overall_ic = float(monthly_ic[feature].mean())
        relation_feature = prediction_relation[
            prediction_relation["feature"].eq(feature)
        ]
        beta_feature = beta_yearly[beta_yearly["feature"].eq(feature)]
        summary_rows.append(
            {
                "row_type": "feature_summary",
                "feature": feature,
                "raw_missing_rate": float(missing.mean()),
                "rank_missing_fill_rate": float((missing & rank.eq(0.5)).mean()),
                "missing_fill_correct_fraction": float(rank[missing].eq(0.5).mean())
                if missing.any()
                else 1.0,
                "rank_p1": float(rank.quantile(0.01)),
                "rank_p5": float(rank.quantile(0.05)),
                "rank_p50": float(rank.quantile(0.50)),
                "rank_p95": float(rank.quantile(0.95)),
                "rank_p99": float(rank.quantile(0.99)),
                "average_absolute_correlation_core10": float(
                    np.mean(np.abs(list(core_correlations.values())))
                ),
                "rank_ic_monthly_mean": overall_ic,
                "rank_ic_monthly_std": float(monthly_ic[feature].std(ddof=1)),
                "rank_ic_annual_positive_fraction": float(
                    (np.sign(annual_values) == np.sign(overall_ic)).mean()
                ),
                "rank_ic_top1_abs_year_share": float(absolute.head(1).sum())
                / max(absolute_sum, 1.0e-16),
                "rank_ic_top3_abs_year_share": float(absolute.head(3).sum())
                / max(absolute_sum, 1.0e-16),
                "rank_ic_concentrated_few_years": bool(
                    absolute.head(3).sum() / max(absolute_sum, 1.0e-16) > 0.5
                ),
                "static_prediction_corr_mean": float(
                    relation_feature["static_prediction_corr"].mean()
                ),
                "static_prediction_corr_abs_mean": float(
                    relation_feature["static_prediction_corr"].abs().mean()
                ),
                "feature_std_static_prediction_dispersion_corr": float(
                    relation_feature["feature_xs_std"].corr(
                        relation_feature["static_prediction_dispersion"]
                    )
                ),
                "linear_aim_corr_mean": float(
                    relation_feature["linear_aim_corr"].mean()
                ),
                "linear_aim_corr_abs_mean": float(
                    relation_feature["linear_aim_corr"].abs().mean()
                ),
                "linear_avg_abs_aim_contribution": float(
                    beta_feature["average_abs_aim_contribution"].mean()
                ),
                "linear_avg_contribution_share": float(
                    beta_feature["absolute_contribution_share"].mean()
                ),
                "linear_2007_contribution_share": _year_value(
                    beta_feature, 2007, "absolute_contribution_share"
                ),
                "linear_2020_2021_contribution_share": float(
                    beta_feature[beta_feature["year"].isin([2020, 2021])][
                        "absolute_contribution_share"
                    ].mean()
                ),
            }
        )
        for core_feature, correlation in core_correlations.items():
            correlation_rows.append(
                {
                    "row_type": "core10_correlation",
                    "feature": feature,
                    "core10_feature": core_feature,
                    "correlation": correlation,
                }
            )
        annual_relation = relation_feature.groupby("year", sort=True).agg(
            static_prediction_corr=("static_prediction_corr", "mean"),
            linear_aim_corr=("linear_aim_corr", "mean"),
            static_prediction_dispersion=("static_prediction_dispersion", "mean"),
        )
        for year in sorted(set(annual_ic.index) | set(beta_feature["year"])):
            beta_row = beta_feature[beta_feature["year"].eq(year)]
            yearly_rows.append(
                {
                    "row_type": "feature_year",
                    "feature": feature,
                    "year": int(year),
                    "cross_sectional_std": annual_std.at[year, feature]
                    if year in annual_std.index
                    else np.nan,
                    "annual_rank_ic": annual_ic.at[year, feature]
                    if year in annual_ic.index
                    else np.nan,
                    "static_prediction_corr": annual_relation.at[
                        year, "static_prediction_corr"
                    ]
                    if year in annual_relation.index
                    else np.nan,
                    "static_prediction_dispersion": annual_relation.at[
                        year, "static_prediction_dispersion"
                    ]
                    if year in annual_relation.index
                    else np.nan,
                    "linear_aim_corr": annual_relation.at[year, "linear_aim_corr"]
                    if year in annual_relation.index
                    else np.nan,
                    "linear_beta": float(beta_row["beta"].iloc[0])
                    if len(beta_row)
                    else np.nan,
                    "linear_average_abs_aim_contribution": float(
                        beta_row["average_abs_aim_contribution"].iloc[0]
                    )
                    if len(beta_row)
                    else np.nan,
                    "linear_absolute_contribution_share": float(
                        beta_row["absolute_contribution_share"].iloc[0]
                    )
                    if len(beta_row)
                    else np.nan,
                    "linear_aim_reconstruction_relative_rmse": float(
                        beta_row["aim_reconstruction_relative_rmse"].iloc[0]
                    )
                    if len(beta_row)
                    else np.nan,
                }
            )
    output = pd.concat(
        [
            pd.DataFrame(summary_rows),
            pd.DataFrame(correlation_rows),
            pd.DataFrame(yearly_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    context = {
        "added_features": added,
        "beta_quality": beta_quality,
        "panel_rows": int(len(panel)),
        "panel_months": int(panel["eom"].nunique()),
    }
    return output, context


def _monthly_feature_statistics(
    panel: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, object]] = []
    std_rows: list[dict[str, object]] = []
    for eom, group in panel.groupby("eom", sort=True):
        label = pd.to_numeric(group["ret_exc_lead1m"], errors="coerce")
        label_rank = label.rank(pct=True, method="average")
        ic_row: dict[str, object] = {"eom": eom, "year": int(eom.year)}
        std_row: dict[str, object] = {"eom": eom, "year": int(eom.year)}
        for feature in features:
            values = pd.to_numeric(group[f"rank_{feature}"], errors="coerce")
            ic_row[feature] = float(values.corr(label_rank))
            std_row[feature] = float(values.std(ddof=1))
        ic_rows.append(ic_row)
        std_rows.append(std_row)
    return pd.DataFrame(ic_rows), pd.DataFrame(std_rows)


def _prediction_relations(
    panel: pd.DataFrame, predictions: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    static = predictions[predictions["method"].eq("static_ml")][
        ["eom", "permno", "mu_hat"]
    ].copy()
    linear = predictions[predictions["method"].eq("portfolio_ml_linear")][
        ["eom", "permno", "aim_weight"]
    ].copy()
    for frame in (static, linear):
        frame["eom"] = pd.to_datetime(frame["eom"])
    needed = panel[panel["eom"].dt.year.between(2005, 2024)][
        ["eom", "permno", *[f"rank_{feature}" for feature in features]]
    ]
    merged = needed.merge(static, on=["eom", "permno"], validate="one_to_one").merge(
        linear, on=["eom", "permno"], validate="one_to_one"
    )
    rows: list[dict[str, object]] = []
    for eom, group in merged.groupby("eom", sort=True):
        for feature in features:
            values = pd.to_numeric(group[f"rank_{feature}"], errors="coerce")
            rows.append(
                {
                    "eom": eom,
                    "year": int(eom.year),
                    "feature": feature,
                    "feature_xs_std": float(values.std(ddof=1)),
                    "static_prediction_dispersion": float(group["mu_hat"].std(ddof=1)),
                    "static_prediction_corr": float(values.corr(group["mu_hat"])),
                    "linear_aim_corr": float(values.corr(group["aim_weight"])),
                }
            )
    return pd.DataFrame(rows)


def _recover_linear_beta_contributions(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    config,
    rank_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    values = config.values
    sample = panel[panel["eom"].dt.year.between(2005, 2024)].copy()
    risk_inputs = load_risk_inputs(
        start="2005-01-31",
        end="2024-12-31",
        universe="top500",
        root=config.resolve_path(values["risk"]["inputs_dir"]),
    )
    risk_settings = {
        **values["risk"],
        "persistent_cache_dir": str(
            config.resolve_path(values["cache"]["root"]) / "covariance"
        ),
    }
    provider = RiskModelProvider(
        sample,
        universe="top500",
        requested_model="factor",
        settings=risk_settings,
        risk_inputs=risk_inputs,
    )
    aims = predictions[predictions["method"].eq("portfolio_ml_linear")].copy()
    aims["eom"] = pd.to_datetime(aims["eom"])
    aim_months = {
        eom: group.set_index("permno")["aim_weight"]
        for eom, group in aims.groupby("eom", sort=True)
    }
    rows: list[dict[str, object]] = []
    relative_errors: list[float] = []
    for year, year_panel in sample.groupby(sample["eom"].dt.year, sort=True):
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        xtx = np.zeros((len(rank_columns) + 1, len(rank_columns) + 1))
        xty = np.zeros(len(rank_columns) + 1)
        for eom, month in year_panel.groupby("eom", sort=True):
            raw = month[rank_columns].to_numpy(float)
            raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)
            standardized = monthly_cross_sectional_standardize(raw, add_constant=True)
            volatility = provider.get_volatility(
                eom, month["permno"].astype(int).to_numpy()
            )
            design, _ = volatility_scale_design(
                standardized,
                volatility,
                floor=float(values["portfolio_ml"]["volatility_floor"]),
            )
            aim = (
                aim_months[pd.Timestamp(eom)]
                .reindex(month["permno"].astype(int))
                .to_numpy(float)
            )
            if not np.isfinite(aim).all():
                raise RuntimeError(f"Missing Linear aim weights for {eom}")
            xtx += design.T @ design
            xty += design.T @ aim
            blocks.append((design, aim))
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        squared_error = 0.0
        squared_aim = 0.0
        contribution_sums = np.zeros(len(rank_columns) + 1)
        observation_count = 0
        for design, aim in blocks:
            fitted = design @ beta
            squared_error += float(np.sum((fitted - aim) ** 2))
            squared_aim += float(np.sum(aim**2))
            contribution_sums += np.abs(design * beta).sum(axis=0)
            observation_count += len(aim)
        relative_rmse = float(np.sqrt(squared_error / max(squared_aim, 1.0e-30)))
        relative_errors.append(relative_rmse)
        feature_total = float(contribution_sums[1:].sum())
        for position, column in enumerate(rank_columns, start=1):
            rows.append(
                {
                    "year": int(year),
                    "feature": column.removeprefix("rank_"),
                    "beta": float(beta[position]),
                    "average_abs_aim_contribution": float(
                        contribution_sums[position] / observation_count
                    ),
                    "absolute_contribution_share": float(
                        contribution_sums[position] / max(feature_total, 1.0e-30)
                    ),
                    "aim_reconstruction_relative_rmse": relative_rmse,
                }
            )
    quality = {
        "method": "algebraic_recovery_from_persisted_aim_equals_B_beta",
        "years": len(relative_errors),
        "max_relative_rmse": max(relative_errors),
        "median_relative_rmse": float(np.median(relative_errors)),
        "risk_cache": provider.cache_info(),
    }
    return pd.DataFrame(rows), quality


def build_ex_year_report(
    output: pd.DataFrame, context: dict[str, object]
) -> str:
    metrics = output[output["row_type"].eq("method_metrics")][
        [
            "sample",
            "method",
            "annualized_net_return",
            "annualized_volatility",
            "net_sharpe",
            "average_turnover",
            "average_leverage",
            "annualized_trading_cost",
            "annualized_ex_ante_utility_flow",
        ]
    ]
    inference = output[output["row_type"].eq("utility_inference")][
        [
            "sample",
            "candidate_method",
            "benchmark_method",
            "annualized_mean_difference",
            "newey_west_t_stat",
            "bootstrap_annualized_ci_lower",
            "bootstrap_annualized_ci_upper",
            "annual_years_won",
            "annual_years_total",
        ]
    ]
    risk = output[output["row_type"].eq("static_risk_summary")][
        [
            "sample",
            "method",
            "yearly_ratio_median",
            "fraction_years_within_0_6_1_5",
            "years_ratio_above_2",
        ]
    ]
    final_concentration = context["final_concentration"]
    core_concentration = context["core_concentration"]
    full = inference[inference["sample"].eq("full_2005_2024")]
    ex_2007 = inference[inference["sample"].eq("exclude_2007")]
    positive_after_2007 = bool((ex_2007["annualized_mean_difference"] > 0).all())
    supported_after_2007 = bool(
        (ex_2007["bootstrap_annualized_ci_lower"] > 0).all()
    )
    final_top3 = float(final_concentration["top_3_positive_year_share"].mean())
    core_top3 = float(core_concentration["top_3_positive_year_share"].mean())
    pair_effect = full.merge(
        ex_2007,
        on=["candidate_method", "benchmark_method"],
        suffixes=("_full", "_exclude_2007"),
        validate="one_to_one",
    )[
        [
            "candidate_method",
            "benchmark_method",
            "annualized_mean_difference_full",
            "annualized_mean_difference_exclude_2007",
            "newey_west_t_stat_full",
            "newey_west_t_stat_exclude_2007",
        ]
    ]
    risk_ex_crisis = risk[risk["sample"].eq("exclude_2020_2021")]
    static_breaches = int(
        risk_ex_crisis.loc[risk_ex_crisis["method"].eq("static_ml"), "years_ratio_above_2"].iloc[0]
    )
    star_breaches = int(
        risk_ex_crisis.loc[
            risk_ex_crisis["method"].eq("static_ml_star"), "years_ratio_above_2"
        ].iloc[0]
    )
    return (
        "# Top500/Final15 ex-year robustness\n\n"
        "This diagnostic filters completed monthly OOS results only. It does not "
        "refit predictions, reselect hyperparameters, or rerun portfolios. HAC and "
        "the circular block bootstrap use the same project settings as the formal run.\n\n"
        "## Findings\n\n"
        f"- All four Portfolio-ML versus Static utility means remain positive after "
        f"excluding 2007: {positive_after_2007}. All four bootstrap lower bounds "
        f"remain above zero: {supported_after_2007}.\n"
        "- Excluding 2007 reduces every relative utility mean materially; Linear "
        "versus Static-ML* falls to approximately zero. Positive point estimates "
        "therefore do not constitute robust statistical support.\n"
        f"- After excluding 2020-2021, Static-ML retains {static_breaches} ratio-above-two "
        f"year and Static-ML* retains {star_breaches}. The pandemic exclusions do not "
        "fully repair aggregate Static calibration.\n"
        f"- Final15's mean top-three positive-year utility concentration is "
        f"{final_top3:.1%}, versus {core_top3:.1%} for Core10.\n"
        "- The Static risk table below shows directly whether removing 2020-2021 "
        "eliminates ratio-above-two years; retained-year ratios are not re-estimated.\n"
        "- Excluding years changes evaluation only. It is not a counterfactual "
        "hyperparameter-selection experiment.\n\n"
        "## Method metrics\n\n"
        + _markdown(metrics)
        + "\n\n## Utility inference\n\n"
        + _markdown(inference)
        + "\n\n## Effect of excluding 2007\n\n"
        + _markdown(pair_effect)
        + "\n\n## Static risk calibration by retained sample\n\n"
        + _markdown(risk)
        + "\n\n## Positive-year concentration\n\n### Final15\n\n"
        + _markdown(final_concentration)
        + "\n\n### Core10\n\n"
        + _markdown(core_concentration)
        + "\n"
    )


def build_decomposition_report(
    decomposition: pd.DataFrame, summary: pd.DataFrame
) -> str:
    static = summary[summary["method"].isin(["static_ml", "static_ml_star"])]
    portfolio = summary[
        summary["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])
    ]
    year_2007 = decomposition[decomposition["year"].eq(2007)].set_index("method")
    return (
        "# Top500 Core10 versus Final15 yearly decomposition\n\n"
        "All differences are Final15 minus Core10 and use the already completed "
        "specification-specific OOS portfolios. No model was refitted.\n\n"
        "## Findings\n\n"
        f"- Final15 raises annual volatility in a mean of "
        f"{static['volatility_higher_years'].mean():.1f}/20 Static method-years, "
        f"versus {portfolio['volatility_higher_years'].mean():.1f}/20 Portfolio-ML method-years.\n"
        f"- Static methods have a mean Final15 utility delta of "
        f"{static['mean_utility_delta'].mean():.6g}; Portfolio-ML methods have "
        f"{portfolio['mean_utility_delta'].mean():.6g}.\n"
        f"- In 2007, Final15-minus-Core10 utility is "
        f"{year_2007.at['static_ml', 'utility_final15_minus_core10']:.6g} for Static-ML, "
        f"{year_2007.at['static_ml_star', 'utility_final15_minus_core10']:.6g} for "
        f"Static-ML*, {year_2007.at['portfolio_ml_linear', 'utility_final15_minus_core10']:.6g} "
        f"for Linear, and {year_2007.at['portfolio_ml_rf', 'utility_final15_minus_core10']:.6g} "
        "for RF. The large 2007 relative Portfolio-ML advantage is thus primarily a "
        "Static deterioration under Final15, not a common Portfolio-ML gain.\n"
        f"- Portfolio-ML Final15 utility improves in only "
        f"{int(portfolio['utility_positive_years'].min())}-{int(portfolio['utility_positive_years'].max())} "
        "of 20 years, so the incremental feature-set effect is not uniformly positive.\n"
        "- The 2007 share and top-three-year share below quantify whether apparent "
        "incremental value is broad or concentrated.\n\n"
        "## Method summary\n\n"
        + _markdown(summary)
        + "\n\n## Yearly decomposition\n\n"
        + _markdown(
            decomposition[
                [
                    "year",
                    "method",
                    "net_return_final15_minus_core10",
                    "utility_final15_minus_core10",
                    "volatility_final15_minus_core10",
                    "turnover_final15_minus_core10",
                    "leverage_final15_minus_core10",
                    "trading_cost_final15_minus_core10",
                ]
            ]
        )
        + "\n"
    )


def build_feature_report(
    output: pd.DataFrame, context: dict[str, object]
) -> str:
    summary = output[output["row_type"].eq("feature_summary")].copy()
    correlations = output[output["row_type"].eq("core10_correlation")].copy()
    yearly = output[output["row_type"].eq("feature_year")].copy()
    beta_quality = context["beta_quality"]
    noise_like = summary[
        (summary["rank_ic_monthly_mean"].abs() < 0.01)
        & (summary["linear_avg_contribution_share"] < 0.05)
    ]["feature"].tolist()
    concentrated = summary[summary["rank_ic_concentrated_few_years"]]["feature"].tolist()
    informative = summary[summary["rank_ic_monthly_mean"].abs() >= 0.01]["feature"].tolist()
    weak_ic = summary[summary["rank_ic_monthly_mean"].abs() < 0.005]["feature"].tolist()
    total_contribution = float(summary["linear_avg_contribution_share"].sum())
    contribution_2007 = float(summary["linear_2007_contribution_share"].sum())
    contribution_2020_2021 = float(
        summary["linear_2020_2021_contribution_share"].sum()
    )
    report_columns = [
        "feature",
        "raw_missing_rate",
        "rank_missing_fill_rate",
        "rank_p1",
        "rank_p5",
        "rank_p50",
        "rank_p95",
        "rank_p99",
        "average_absolute_correlation_core10",
        "rank_ic_monthly_mean",
        "rank_ic_monthly_std",
        "rank_ic_annual_positive_fraction",
        "rank_ic_top3_abs_year_share",
        "static_prediction_corr_mean",
        "linear_aim_corr_mean",
        "linear_avg_abs_aim_contribution",
        "linear_avg_contribution_share",
        "linear_2007_contribution_share",
        "linear_2020_2021_contribution_share",
    ]
    return (
        "# Top500/Final15 added-feature diagnostics\n\n"
        f"The panel contains {context['panel_rows']:,} Top500 rows over "
        f"{context['panel_months']} months. Diagnostics use only persisted features, "
        "labels, predictions, and aim weights. No return model or portfolio was fitted.\n\n"
        "## Findings\n\n"
        f"- Features with absolute mean monthly rank IC below 1% and average Linear "
        f"contribution share below 5%: {noise_like}.\n"
        f"- Features with absolute mean monthly rank IC of at least 1% are "
        f"{informative}; features below 0.5% are {weak_ic}.\n"
        f"- Features whose top three absolute annual ICs exceed half of total annual "
        f"absolute IC: {concentrated}.\n"
        f"- The five added signals account for {total_contribution:.1%} of absolute "
        f"Linear aim contribution on average, {contribution_2007:.1%} in 2007, and "
        f"{contribution_2020_2021:.1%} in 2020-2021. They are used by the Linear "
        "portfolio, but they do not uniquely dominate the crisis windows.\n"
        f"- Linear beta recovery uses persisted aim = B beta. Maximum yearly relative "
        f"RMSE is {beta_quality['max_relative_rmse']:.3g}; median is "
        f"{beta_quality['median_relative_rmse']:.3g}. This is an algebraic recovery, "
        "not model retraining.\n"
        "- Contribution share is based on absolute B-column times recovered-beta "
        "contributions before cross-feature cancellation.\n\n"
        "## Feature summary\n\n"
        + _markdown(summary[report_columns])
        + "\n\n## Correlation with Core10\n\n"
        + _markdown(correlations[["feature", "core10_feature", "correlation"]])
        + "\n\n## Annual diagnostics\n\n"
        + _markdown(
            yearly[
                [
                    "year",
                    "feature",
                    "cross_sectional_std",
                    "annual_rank_ic",
                    "static_prediction_corr",
                    "linear_aim_corr",
                    "linear_beta",
                    "linear_average_abs_aim_contribution",
                    "linear_absolute_contribution_share",
                ]
            ]
        )
        + "\n"
    )


def _year_value(frame: pd.DataFrame, year: int, column: str) -> float:
    value = frame.loc[frame["year"].eq(year), column]
    return float(value.iloc[0]) if len(value) else np.nan


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
