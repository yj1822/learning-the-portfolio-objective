from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from implementable_frontier.models.metrics import annual_performance
from implementable_frontier.models.rolling_robustness import (
    _circular_block_bootstrap_means,
    _newey_west_mean_test,
)


COMPARISONS = (
    ("portfolio_ml_linear", "static_ml"),
    ("portfolio_ml_linear", "static_ml_star"),
    ("portfolio_ml_rf", "static_ml"),
    ("portfolio_ml_rf", "static_ml_star"),
    ("portfolio_ml_rf", "portfolio_ml_linear"),
)

DELTA_FIELDS = (
    "gross_return",
    "trading_cost",
    "net_return",
    "risk_penalty",
    "utility_flow",
    "turnover",
    "leverage",
)


@dataclass(frozen=True)
class FormalAnalysisResult:
    monthly_differences: pd.DataFrame
    statistical_inference: pd.DataFrame
    yearly_attribution: pd.DataFrame
    concentration: pd.DataFrame
    period_attribution: pd.DataFrame
    risk_calibration: pd.DataFrame
    validation_stability: pd.DataFrame
    risk_quality_passed: bool


def run_formal_analysis(
    *,
    output_dir: str | Path,
    report_dir: str | Path,
    returns: pd.DataFrame,
    annual: pd.DataFrame,
    validation_selection: pd.DataFrame,
    gamma: float,
    seed: int,
    newey_west_lags: int,
    bootstrap_block_months: int,
    bootstrap_replications: int,
) -> FormalAnalysisResult:
    output = Path(output_dir).resolve()
    reports = Path(report_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    monthly = monthly_utility_differences(returns, gamma=gamma)
    inference = statistical_inference(
        monthly,
        newey_west_lags=newey_west_lags,
        bootstrap_block_months=bootstrap_block_months,
        bootstrap_replications=bootstrap_replications,
        seed=seed,
    )
    yearly, concentration, periods = yearly_attribution(
        returns,
        annual,
        validation_selection,
        gamma=gamma,
    )
    risk = risk_calibration_report(returns)
    stability = validation_stability_report(validation_selection)
    quality_rows = risk[
        risk["scope"].eq("summary") & ~risk["method"].eq("markowitz_ml")
    ]
    risk_quality_passed = bool(
        not quality_rows.empty
        and quality_rows["quality_gate_passed"].astype("boolean").fillna(False).all()
    )
    metadata_path = output / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    universe = str(metadata.get("universe", "unknown"))
    feature_set = str(metadata.get("feature_set", "unknown"))
    test_years = sorted(pd.to_numeric(returns["test_year"], errors="coerce").dropna().astype(int).unique())

    monthly.to_parquet(output / "monthly_utility_differences.parquet", index=False)
    inference.to_csv(output / "statistical_inference.csv", index=False)
    yearly.to_csv(output / "yearly_attribution.csv", index=False)
    risk.to_csv(output / "risk_calibration.csv", index=False)
    concentration.to_csv(output / "contribution_concentration.csv", index=False)
    periods.to_csv(output / "period_attribution.csv", index=False)
    stability.to_csv(output / "validation_stability.csv", index=False)

    _write_reports(
        reports,
        returns=returns,
        annual=annual,
        inference=inference,
        yearly=yearly,
        concentration=concentration,
        periods=periods,
        risk=risk,
        stability=stability,
        risk_quality_passed=risk_quality_passed,
        gamma=gamma,
        universe=universe,
        feature_set=feature_set,
        first_test_year=min(test_years) if test_years else None,
        last_test_year=max(test_years) if test_years else None,
        markowitz_included=bool(returns["method"].eq("markowitz_ml").any()),
    )
    return FormalAnalysisResult(
        monthly,
        inference,
        yearly,
        concentration,
        periods,
        risk,
        stability,
        risk_quality_passed,
    )


def monthly_utility_differences(
    returns: pd.DataFrame, *, gamma: float
) -> pd.DataFrame:
    data = returns.copy()
    data["eom"] = pd.to_datetime(data["eom"])
    data["risk_penalty"] = (
        0.5
        * float(gamma)
        * pd.to_numeric(data["ex_ante_monthly_variance"], errors="coerce")
    )
    rows: list[pd.DataFrame] = []
    for candidate, benchmark in COMPARISONS:
        candidate_frame = data[data["method"].eq(candidate)][
            ["eom", "test_year", *DELTA_FIELDS]
        ].copy()
        benchmark_frame = data[data["method"].eq(benchmark)][
            ["eom", "test_year", *DELTA_FIELDS]
        ].copy()
        candidate_frame = candidate_frame.rename(
            columns={field: f"candidate_{field}" for field in DELTA_FIELDS}
        )
        benchmark_frame = benchmark_frame.rename(
            columns={
                "test_year": "benchmark_test_year",
                **{field: f"benchmark_{field}" for field in DELTA_FIELDS},
            }
        )
        merged = candidate_frame.merge(
            benchmark_frame, on="eom", how="inner", validate="one_to_one"
        )
        if merged.empty:
            continue
        if not np.array_equal(
            merged["test_year"].to_numpy(),
            merged["benchmark_test_year"].to_numpy(),
        ):
            raise ValueError("Method comparison has inconsistent test-year labels")
        merged["candidate_method"] = candidate
        merged["benchmark_method"] = benchmark
        for field in DELTA_FIELDS:
            merged[f"delta_{field}"] = (
                merged[f"candidate_{field}"] - merged[f"benchmark_{field}"]
            )
        expected_utility = merged["delta_net_return"] - merged["delta_risk_penalty"]
        error = float(
            np.nanmax(np.abs(expected_utility - merged["delta_utility_flow"]))
        )
        if error > 1.0e-10:
            raise ValueError(
                f"Utility-flow accounting mismatch for {candidate} vs {benchmark}: {error}"
            )
        keep = [
            "eom",
            "test_year",
            "candidate_method",
            "benchmark_method",
            *[f"delta_{field}" for field in DELTA_FIELDS],
        ]
        rows.append(merged[keep])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def statistical_inference(
    monthly: pd.DataFrame,
    *,
    newey_west_lags: int,
    bootstrap_block_months: int,
    bootstrap_replications: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (candidate, benchmark), group in monthly.groupby(
        ["candidate_method", "benchmark_method"], sort=False
    ):
        for field in DELTA_FIELDS:
            values = pd.to_numeric(group[f"delta_{field}"], errors="coerce").dropna()
            if values.empty:
                continue
            array = values.to_numpy(float)
            lags = min(int(newey_west_lags), max(len(array) - 1, 0))
            nw_se, nw_t = _newey_west_mean_test(array, lags)
            bootstrap = _circular_block_bootstrap_means(
                array,
                block_months=bootstrap_block_months,
                replications=bootstrap_replications,
                rng=rng,
            )
            annual_values = group.groupby("test_year", sort=True)[f"delta_{field}"].sum()
            rows.append(
                {
                    "candidate_method": candidate,
                    "benchmark_method": benchmark,
                    "outcome": field,
                    "n_months": int(len(array)),
                    "monthly_mean_difference": float(array.mean()),
                    "annualized_mean_difference": float(12.0 * array.mean()),
                    "newey_west_lags": lags,
                    "newey_west_monthly_standard_error": nw_se,
                    "newey_west_annualized_standard_error": 12.0 * nw_se,
                    "newey_west_t_stat": nw_t,
                    "bootstrap_block_months": int(bootstrap_block_months),
                    "bootstrap_replications": int(bootstrap_replications),
                    "bootstrap_annualized_ci_lower": float(12.0 * np.quantile(bootstrap, 0.025)),
                    "bootstrap_annualized_ci_upper": float(12.0 * np.quantile(bootstrap, 0.975)),
                    "probability_delta_positive": float(np.mean(bootstrap > 0.0)),
                    "median_annual_difference": float(annual_values.median()),
                    "annual_years_won": int((annual_values > 0.0).sum()),
                    "annual_years_total": int(len(annual_values)),
                    "full_sample_cumulative_difference": float(array.sum()),
                }
            )
    return pd.DataFrame(rows)


def yearly_attribution(
    returns: pd.DataFrame,
    annual: pd.DataFrame,
    validation_selection: pd.DataFrame,
    *,
    gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = returns.copy()
    data["risk_penalty"] = (
        0.5 * float(gamma) * pd.to_numeric(data["ex_ante_monthly_variance"], errors="coerce")
    )
    data["period"] = data["test_year"].astype(int).map(_period)
    selected = validation_selection[validation_selection["selected"]][
        ["test_year", "method", "candidate_hyperparameters"]
    ].copy()
    method_rows: list[dict[str, object]] = []
    annual_lookup = annual.set_index(["method", "test_year"])
    for (method, year), group in data.groupby(["method", "test_year"], sort=True):
        key = (method, year)
        metric = annual_lookup.loc[key] if key in annual_lookup.index else pd.Series(dtype=float)
        predicted = float(
            np.sqrt(12.0 * pd.to_numeric(group["ex_ante_monthly_variance"], errors="coerce").mean())
        )
        realized = float(np.sqrt(12.0) * group["net_return"].std(ddof=1))
        hp = selected[
            selected["test_year"].eq(year) & selected["method"].eq(method)
        ]["candidate_hyperparameters"]
        method_rows.append(
            {
                "method": method,
                "test_year": int(year),
                "period": _period(int(year)),
                "annualized_gross_return": metric.get("annualized_gross_return", np.nan),
                "annualized_net_return": metric.get("annualized_net_return", np.nan),
                "annualized_volatility": metric.get("annualized_volatility", np.nan),
                "annualized_utility": metric.get("annualized_ex_ante_utility_flow", np.nan),
                "empirical_utility": metric.get("empirical_utility", np.nan),
                "average_turnover": metric.get("average_turnover", np.nan),
                "annualized_trading_cost": metric.get("annualized_trading_cost", np.nan),
                "average_leverage": metric.get("average_leverage", np.nan),
                "predicted_annualized_volatility": predicted,
                "realized_annualized_volatility": realized,
                "realized_to_predicted_ratio": realized / predicted if predicted > 0 else np.nan,
                "selected_hyperparameters": hp.iloc[0] if len(hp) else "prediction_mse_only",
            }
        )
    method_year = pd.DataFrame(method_rows)

    monthly = monthly_utility_differences(returns, gamma=gamma)
    comparison_rows: list[dict[str, object]] = []
    for (candidate, benchmark, year), group in monthly.groupby(
        ["candidate_method", "benchmark_method", "test_year"], sort=True
    ):
        row: dict[str, object] = {
            "candidate_method": candidate,
            "benchmark_method": benchmark,
            "test_year": int(year),
            "period": _period(int(year)),
        }
        for field in DELTA_FIELDS:
            row[f"annual_{field}_difference"] = float(group[f"delta_{field}"].sum())
        comparison_rows.append(row)
    comparison_year = pd.DataFrame(comparison_rows)
    concentration = _concentration(comparison_year)
    periods = _period_attribution(comparison_year)
    return method_year, concentration, periods


def risk_calibration_report(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    for (method, year), group in data.groupby(["method", "test_year"], sort=True):
        annual_rows.append(_risk_row(group, method=method, scope="year", test_year=int(year)))
    annual = pd.DataFrame(annual_rows)
    rows.extend(annual_rows)
    for method, group in data.groupby("method", sort=True):
        rows.append(_risk_row(group, method=method, scope="full", test_year=np.nan))
        ratios = pd.to_numeric(
            annual[annual["method"].eq(method)]["realized_to_predicted_ratio"],
            errors="coerce",
        ).dropna()
        median = float(ratios.median()) if len(ratios) else np.nan
        within = float(ratios.between(0.6, 1.5).mean()) if len(ratios) else np.nan
        above_two = int((ratios > 2.0).sum())
        allowed_above_two = max(1, int(np.ceil(0.1 * len(ratios)))) if len(ratios) else 0
        fallback = _fallback_count(group)
        passed = bool(
            len(ratios)
            and 0.8 <= median <= 1.25
            and within >= 0.8
            and above_two <= allowed_above_two
            and fallback == 0
        )
        rows.append(
            {
                "scope": "summary",
                "method": method,
                "test_year": np.nan,
                "n_months": int(len(group)),
                "yearly_ratio_median": median,
                "yearly_ratio_p10": float(ratios.quantile(0.1)) if len(ratios) else np.nan,
                "yearly_ratio_p90": float(ratios.quantile(0.9)) if len(ratios) else np.nan,
                "yearly_ratio_max": float(ratios.max()) if len(ratios) else np.nan,
                "fraction_years_within_0_6_1_5": within,
                "years_ratio_above_2": above_two,
                "factor_risk_fallback_count": fallback,
                "covariance_repair_count": int(group["covariance_repair_count"].sum()),
                "covariance_min_eigenvalue": float(group["covariance_min_eigenvalue"].min()),
                "covariance_condition_number": float(group["covariance_condition_number"].max()),
                "quality_gate_passed": passed,
            }
        )
    return pd.DataFrame(rows)


def validation_stability_report(selection: pd.DataFrame) -> pd.DataFrame:
    chosen = selection[selection["selected"]].copy()
    rows: list[dict[str, object]] = []
    for method, group in chosen.sort_values("test_year").groupby("method", sort=True):
        parameters = group["candidate_hyperparameters"].astype(str)
        counts = parameters.value_counts()
        rows.append(
            {
                "method": method,
                "test_years": int(len(group)),
                "unique_parameter_sets": int(parameters.nunique()),
                "selection_changes": int((parameters != parameters.shift()).iloc[1:].sum()),
                "modal_parameter_share": float(counts.iloc[0] / len(group)),
                "modal_parameters": counts.index[0],
                "first_selected_parameters": parameters.iloc[0],
                "last_selected_parameters": parameters.iloc[-1],
                "future_information_violations": int(
                    group["future_information_used"].fillna(False).sum()
                ),
                "latest_validation_year_violation_count": int(
                    (group["latest_validation_year"].astype(int) >= group["test_year"].astype(int)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _risk_row(
    group: pd.DataFrame, *, method: str, scope: str, test_year: int | float
) -> dict[str, object]:
    predicted = float(
        np.sqrt(12.0 * pd.to_numeric(group["ex_ante_monthly_variance"], errors="coerce").mean())
    )
    realized = float(np.sqrt(12.0) * group["net_return"].std(ddof=1))
    return {
        "scope": scope,
        "method": method,
        "test_year": test_year,
        "n_months": int(len(group)),
        "predicted_annualized_volatility": predicted,
        "realized_annualized_volatility": realized,
        "realized_to_predicted_ratio": realized / predicted if predicted > 0 else np.nan,
        "factor_risk_fallback_count": _fallback_count(group),
        "covariance_repair_count": int(group["covariance_repair_count"].sum()),
        "covariance_min_eigenvalue": float(group["covariance_min_eigenvalue"].min()),
        "covariance_condition_number": float(group["covariance_condition_number"].max()),
    }


def _fallback_count(group: pd.DataFrame) -> int:
    wrong_model = ~group["risk_model_used"].astype(str).eq("factor")
    reasons = group.get("risk_fallback_reason", pd.Series(index=group.index, dtype=object))
    return int((wrong_model | reasons.notna()).sum())


def _concentration(comparison_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, benchmark), group in comparison_year.groupby(
        ["candidate_method", "benchmark_method"], sort=False
    ):
        for outcome in ("net_return", "utility_flow"):
            column = f"annual_{outcome}_difference"
            values = group.set_index("test_year")[column].sort_values(ascending=False)
            positive = values[values > 0.0]
            denominator = float(positive.sum())
            rows.append(
                {
                    "candidate_method": candidate,
                    "benchmark_method": benchmark,
                    "outcome": outcome,
                    "years_won": int((values > 0.0).sum()),
                    "years_total": int(len(values)),
                    "top_1_positive_year_share": _positive_share(positive, denominator, 1),
                    "top_3_positive_year_share": _positive_share(positive, denominator, 3),
                    "top_5_positive_year_share": _positive_share(positive, denominator, 5),
                    "best_year": int(values.index[0]),
                    "best_year_difference": float(values.iloc[0]),
                    "year_2008_2009_difference": float(values.reindex([2008, 2009]).fillna(0.0).sum()),
                    "year_2020_difference": float(values.get(2020, 0.0)),
                    "year_2022_difference": float(values.get(2022, 0.0)),
                    "full_sample_difference": float(values.sum()),
                }
            )
    return pd.DataFrame(rows)


def _period_attribution(comparison_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, benchmark, period), group in comparison_year.groupby(
        ["candidate_method", "benchmark_method", "period"], sort=False
    ):
        rows.append(
            {
                "candidate_method": candidate,
                "benchmark_method": benchmark,
                "period": period,
                "years": int(len(group)),
                "net_return_difference": float(group["annual_net_return_difference"].sum()),
                "utility_difference": float(group["annual_utility_flow_difference"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _positive_share(values: pd.Series, denominator: float, count: int) -> float:
    return float(values.head(count).sum() / denominator) if denominator > 0.0 else np.nan


def _period(year: int) -> str:
    if 2005 <= year <= 2009:
        return "2005-2009"
    if 2010 <= year <= 2014:
        return "2010-2014"
    if 2015 <= year <= 2019:
        return "2015-2019"
    return "2020-2024"


def _write_reports(
    report_dir: Path,
    *,
    returns: pd.DataFrame,
    annual: pd.DataFrame,
    inference: pd.DataFrame,
    yearly: pd.DataFrame,
    concentration: pd.DataFrame,
    periods: pd.DataFrame,
    risk: pd.DataFrame,
    stability: pd.DataFrame,
    risk_quality_passed: bool,
    gamma: float,
    universe: str,
    feature_set: str,
    first_test_year: int | None,
    last_test_year: int | None,
    markowitz_included: bool,
) -> None:
    metric_columns = [
        "method",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "average_turnover",
        "average_leverage",
        "annualized_trading_cost",
        "annualized_ex_ante_utility_flow",
    ]
    summary_metrics = annual_performance(
        returns.assign(test_year=1), gamma=gamma
    ).drop(columns="test_year", errors="ignore")
    summary_metrics = summary_metrics[
        [column for column in metric_columns if column in summary_metrics]
    ]
    utility = inference[inference["outcome"].eq("utility_flow")]
    main_methods = yearly[
        yearly["method"].isin(
            ["static_ml", "static_ml_star", "portfolio_ml_linear", "portfolio_ml_rf"]
        )
    ]

    (report_dir / "summary.md").write_text(
        f"# {universe.title()}/{feature_set.title()} cumulative OOS validation\n\n"
        "The formal result uses factor risk, full matrix m, gamma=10, the baseline "
        f"AUM, and test years {first_test_year}-{last_test_year}. "
        + (
            "Markowitz-ML remains diagnostic only.\n\n"
            if markowitz_included
            else "Markowitz-ML is excluded from this run.\n\n"
        )
        + "## Full-period metrics\n\n"
        + _markdown(summary_metrics)
        + "\n\n## Utility comparisons\n\n"
        + _markdown(
            utility[
                [
                    "candidate_method",
                    "benchmark_method",
                    "annualized_mean_difference",
                    "newey_west_t_stat",
                    "bootstrap_annualized_ci_lower",
                    "bootstrap_annualized_ci_upper",
                    "probability_delta_positive",
                ]
            ]
        )
        + f"\n\nPrimary-method risk-calibration quality gate passed: "
        + f"`{risk_quality_passed}`. "
        + (
            "Markowitz-ML is excluded from this aggregate gate and remains an "
            "unstable diagnostic.\n"
            if markowitz_included
            else "Markowitz-ML was not run.\n"
        ),
        encoding="utf-8",
    )
    (report_dir / "statistical_inference.md").write_text(
        "# Statistical inference\n\n"
        "Differences are candidate minus benchmark. Utility flow equals net return "
        "minus gamma/2 times ex-ante monthly variance, matching portfolio accounting.\n\n"
        + _markdown(inference),
        encoding="utf-8",
    )
    (report_dir / "yearly_attribution.md").write_text(
        "# Yearly attribution\n\n## Method-year results\n\n"
        + _markdown(main_methods)
        + "\n\n## Contribution concentration\n\n"
        + _markdown(concentration)
        + "\n\n## Five-year periods\n\n"
        + _markdown(periods),
        encoding="utf-8",
    )
    (report_dir / "risk_calibration.md").write_text(
        "# Risk calibration\n\n"
        "The quality gate requires median yearly realized/predicted volatility in "
        "[0.8, 1.25], at least 80% of years in [0.6, 1.5], limited ratios above 2, "
        "and zero factor-risk fallback.\n\n"
        + _markdown(risk),
        encoding="utf-8",
    )
    (report_dir / "validation_stability.md").write_text(
        "# Validation stability\n\n"
        "Every selected candidate uses cumulative validation years strictly before "
        "the test year. Return-ML prediction tuning is recorded separately in the "
        "experiment output.\n\n"
        + _markdown(stability),
        encoding="utf-8",
    )


def _markdown(frame: pd.DataFrame, *, max_rows: int = 200) -> str:
    if frame.empty:
        return "No rows."
    data = frame.head(max_rows).copy()
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
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(frame)} rows.")
    return "\n".join(lines)
