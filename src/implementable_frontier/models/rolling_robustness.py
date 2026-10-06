from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingRobustnessResult:
    monthly_differences: pd.DataFrame
    difference_statistics: pd.DataFrame
    advantage_by_year: pd.DataFrame
    advantage_concentration: pd.DataFrame
    crisis_period_analysis: pd.DataFrame
    risk_calibration_full: pd.DataFrame
    risk_calibration_by_year: pd.DataFrame


def rolling_robustness_analysis(
    returns: pd.DataFrame,
    *,
    gamma: float,
    seed: int,
    newey_west_lags: int = 6,
    bootstrap_block_months: int = 12,
    bootstrap_replications: int = 2_000,
) -> RollingRobustnessResult:
    monthly = monthly_method_differences(returns)
    statistics = method_difference_statistics(
        monthly,
        newey_west_lags=newey_west_lags,
        bootstrap_block_months=bootstrap_block_months,
        bootstrap_replications=bootstrap_replications,
        seed=seed,
    )
    annual = annual_advantage(monthly)
    concentration = advantage_concentration(annual)
    crisis = crisis_period_analysis(monthly)
    risk_full = risk_calibration(returns)
    risk_annual = risk_calibration(returns, by_year=True)
    return RollingRobustnessResult(
        monthly,
        statistics,
        annual,
        concentration,
        crisis,
        risk_full,
        risk_annual,
    )


def monthly_method_differences(returns: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("portfolio_ml_linear", "static_ml"),
        ("portfolio_ml_linear", "static_ml_star"),
        ("portfolio_ml_rf", "static_ml"),
        ("portfolio_ml_rf", "static_ml_star"),
    )
    columns = [
        "eom",
        "test_year",
        "candidate_method",
        "benchmark_method",
        "candidate_net_return",
        "benchmark_net_return",
        "net_return_difference",
        "candidate_utility_flow",
        "benchmark_utility_flow",
        "utility_flow_difference",
    ]
    if returns.empty:
        return pd.DataFrame(columns=columns)
    data = returns.copy()
    data["eom"] = pd.to_datetime(data["eom"])
    rows: list[pd.DataFrame] = []
    for candidate_method, benchmark_method in comparisons:
        candidate = data[data["method"].eq(candidate_method)][
            ["eom", "test_year", "net_return", "utility_flow"]
        ].rename(
            columns={
                "net_return": "candidate_net_return",
                "utility_flow": "candidate_utility_flow",
            }
        )
        benchmark = data[data["method"].eq(benchmark_method)][
            ["eom", "test_year", "net_return", "utility_flow"]
        ].rename(
            columns={
                "test_year": "benchmark_test_year",
                "net_return": "benchmark_net_return",
                "utility_flow": "benchmark_utility_flow",
            }
        )
        merged = candidate.merge(benchmark, on="eom", how="inner", validate="one_to_one")
        if merged.empty:
            continue
        if not merged["test_year"].equals(merged["benchmark_test_year"]):
            raise ValueError("Method comparison has inconsistent test-year labels")
        merged["candidate_method"] = candidate_method
        merged["benchmark_method"] = benchmark_method
        merged["net_return_difference"] = (
            merged["candidate_net_return"] - merged["benchmark_net_return"]
        )
        merged["utility_flow_difference"] = (
            merged["candidate_utility_flow"] - merged["benchmark_utility_flow"]
        )
        rows.append(merged[columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def method_difference_statistics(
    monthly_differences: pd.DataFrame,
    *,
    newey_west_lags: int,
    bootstrap_block_months: int,
    bootstrap_replications: int,
    seed: int,
) -> pd.DataFrame:
    columns = [
        "candidate_method",
        "benchmark_method",
        "outcome",
        "n_months",
        "monthly_mean_difference",
        "annualized_mean_difference",
        "newey_west_lags",
        "newey_west_standard_error",
        "newey_west_t_stat",
        "bootstrap_block_months",
        "bootstrap_replications",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_probability_mean_positive",
    ]
    if monthly_differences.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    grouped = monthly_differences.groupby(
        ["candidate_method", "benchmark_method"], sort=False
    )
    for (candidate, benchmark), group in grouped:
        for outcome, value_column in (
            ("net_return", "net_return_difference"),
            ("utility_flow", "utility_flow_difference"),
        ):
            values = pd.to_numeric(group[value_column], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            lags = min(int(newey_west_lags), max(len(values) - 1, 0))
            standard_error, t_stat = _newey_west_mean_test(values, lags)
            bootstrap_means = _circular_block_bootstrap_means(
                values,
                block_months=bootstrap_block_months,
                replications=bootstrap_replications,
                rng=rng,
            )
            ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975])
            mean = float(values.mean())
            rows.append(
                {
                    "candidate_method": candidate,
                    "benchmark_method": benchmark,
                    "outcome": outcome,
                    "n_months": int(len(values)),
                    "monthly_mean_difference": mean,
                    "annualized_mean_difference": 12.0 * mean,
                    "newey_west_lags": lags,
                    "newey_west_standard_error": standard_error,
                    "newey_west_t_stat": t_stat,
                    "bootstrap_block_months": int(bootstrap_block_months),
                    "bootstrap_replications": int(bootstrap_replications),
                    "bootstrap_ci_lower": float(ci_lower),
                    "bootstrap_ci_upper": float(ci_upper),
                    "bootstrap_probability_mean_positive": float(
                        np.mean(bootstrap_means > 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def annual_advantage(monthly_differences: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_method",
        "benchmark_method",
        "test_year",
        "n_months",
        "annualized_net_return_difference",
        "annualized_utility_flow_difference",
        "cumulative_net_return_difference",
        "cumulative_utility_flow_difference",
        "period_classification",
    ]
    if monthly_differences.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    grouped = monthly_differences.groupby(
        ["candidate_method", "benchmark_method", "test_year"], sort=False
    )
    for (candidate, benchmark, year), group in grouped:
        rows.append(
            {
                "candidate_method": candidate,
                "benchmark_method": benchmark,
                "test_year": int(year),
                "n_months": int(len(group)),
                "annualized_net_return_difference": 12.0
                * float(group["net_return_difference"].mean()),
                "annualized_utility_flow_difference": 12.0
                * float(group["utility_flow_difference"].mean()),
                "cumulative_net_return_difference": float(
                    group["net_return_difference"].sum()
                ),
                "cumulative_utility_flow_difference": float(
                    group["utility_flow_difference"].sum()
                ),
                "period_classification": _period_for_year(int(year)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def advantage_concentration(annual: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_method",
        "benchmark_method",
        "years",
        "positive_net_years",
        "positive_utility_years",
        "best_net_year",
        "best_net_year_difference",
        "worst_net_year",
        "worst_net_year_difference",
        "top3_positive_net_share",
        "crisis_net_difference_share",
    ]
    if annual.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (candidate, benchmark), group in annual.groupby(
        ["candidate_method", "benchmark_method"], sort=False
    ):
        net = group["cumulative_net_return_difference"]
        positive = net[net > 0.0].sort_values(ascending=False)
        positive_sum = float(positive.sum())
        crisis = group["period_classification"].ne("non_crisis")
        total = float(net.sum())
        best_index = net.idxmax()
        worst_index = net.idxmin()
        rows.append(
            {
                "candidate_method": candidate,
                "benchmark_method": benchmark,
                "years": int(len(group)),
                "positive_net_years": int((net > 0.0).sum()),
                "positive_utility_years": int(
                    (group["cumulative_utility_flow_difference"] > 0.0).sum()
                ),
                "best_net_year": int(group.loc[best_index, "test_year"]),
                "best_net_year_difference": float(net.loc[best_index]),
                "worst_net_year": int(group.loc[worst_index, "test_year"]),
                "worst_net_year_difference": float(net.loc[worst_index]),
                "top3_positive_net_share": (
                    float(positive.head(3).sum()) / positive_sum
                    if positive_sum > 0.0
                    else np.nan
                ),
                "crisis_net_difference_share": (
                    float(net[crisis].sum()) / total if abs(total) > 1.0e-15 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def crisis_period_analysis(monthly_differences: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_method",
        "benchmark_method",
        "period",
        "n_months",
        "annualized_net_return_difference",
        "annualized_utility_flow_difference",
        "net_positive_month_fraction",
    ]
    if monthly_differences.empty:
        return pd.DataFrame(columns=columns)
    data = monthly_differences.copy()
    data["period"] = data["test_year"].astype(int).map(_period_for_year)
    rows: list[dict[str, object]] = []
    grouped = data.groupby(
        ["candidate_method", "benchmark_method", "period"], sort=False
    )
    for (candidate, benchmark, period), group in grouped:
        rows.append(
            {
                "candidate_method": candidate,
                "benchmark_method": benchmark,
                "period": period,
                "n_months": int(len(group)),
                "annualized_net_return_difference": 12.0
                * float(group["net_return_difference"].mean()),
                "annualized_utility_flow_difference": 12.0
                * float(group["utility_flow_difference"].mean()),
                "net_positive_month_fraction": float(
                    (group["net_return_difference"] > 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def risk_calibration(returns: pd.DataFrame, *, by_year: bool = False) -> pd.DataFrame:
    group_columns = ["method", "test_year"] if by_year else ["method"]
    columns = [
        *group_columns,
        "n_months",
        "predicted_annualized_volatility",
        "realized_annualized_gross_volatility",
        "realized_annualized_net_volatility",
        "realized_to_predicted_gross_ratio",
        "gross_volatility_calibration_error",
    ]
    if returns.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in returns.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        variance = pd.to_numeric(
            group["ex_ante_monthly_variance"], errors="coerce"
        ).dropna()
        gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        predicted = float(np.sqrt(12.0 * variance.mean())) if len(variance) else np.nan
        realized_gross = (
            float(np.sqrt(12.0) * gross.std(ddof=1)) if len(gross) > 1 else np.nan
        )
        realized_net = (
            float(np.sqrt(12.0) * net.std(ddof=1)) if len(net) > 1 else np.nan
        )
        row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "n_months": int(len(group)),
                "predicted_annualized_volatility": predicted,
                "realized_annualized_gross_volatility": realized_gross,
                "realized_annualized_net_volatility": realized_net,
                "realized_to_predicted_gross_ratio": (
                    realized_gross / predicted if predicted > 0.0 else np.nan
                ),
                "gross_volatility_calibration_error": realized_gross - predicted,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _newey_west_mean_test(values: np.ndarray, lags: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    centered = values - values.mean()
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / n))
    t_stat = float(values.mean() / standard_error) if standard_error > 0.0 else np.nan
    return standard_error, t_stat


def _circular_block_bootstrap_means(
    values: np.ndarray,
    *,
    block_months: int,
    replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    block = min(max(int(block_months), 1), n)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    means = np.empty(int(replications), dtype=float)
    for replication in range(int(replications)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        means[replication] = float(values[indices].mean())
    return means


def _period_for_year(year: int) -> str:
    if year in {2008, 2009}:
        return "global_financial_crisis_2008_2009"
    if year == 2020:
        return "covid_2020"
    if year == 2022:
        return "inflation_tightening_2022"
    return "non_crisis"
