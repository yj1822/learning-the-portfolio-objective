from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.DataFrame, gamma: float) -> pd.DataFrame:
    return _performance_metrics(returns, gamma, ["method"])


def annual_performance(returns: pd.DataFrame, gamma: float) -> pd.DataFrame:
    if "test_year" not in returns or returns["test_year"].isna().all():
        return pd.DataFrame()
    return _performance_metrics(returns, gamma, ["method", "test_year"])


def _performance_metrics(
    returns: pd.DataFrame,
    gamma: float,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if returns.empty:
        return pd.DataFrame()
    for keys, group in returns.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        gross = pd.to_numeric(group["gross_return"], errors="coerce").dropna()
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        costs = pd.to_numeric(group["trading_cost"], errors="coerce").dropna()
        annual_gross = 12.0 * float(gross.mean())
        annual_net = 12.0 * float(net.mean())
        annual_cost = 12.0 * float(costs.mean())
        gross_vol = np.sqrt(12.0) * float(gross.std(ddof=1))
        net_vol = np.sqrt(12.0) * float(net.std(ddof=1))
        annual_variance = 12.0 * float(net.var(ddof=1))
        utility_flow = pd.to_numeric(
            group.get("utility_flow", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "n_months": int(len(group)),
                "annualized_gross_return": annual_gross,
                "annualized_trading_cost": annual_cost,
                "annualized_net_return": annual_net,
                "annualized_volatility": net_vol,
                "annualized_gross_volatility": gross_vol,
                "gross_sharpe": annual_gross / gross_vol if gross_vol > 0 else np.nan,
                "net_sharpe": annual_net / net_vol if net_vol > 0 else np.nan,
                "average_turnover": float(group["turnover"].mean()),
                "average_leverage": float(group["leverage"].mean()),
                "empirical_utility": annual_net - 0.5 * float(gamma) * annual_variance,
                "annualized_ex_ante_utility_flow": (
                    12.0 * float(utility_flow.mean()) if not utility_flow.empty else np.nan
                ),
                "max_drawdown": _max_drawdown(net),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validation_utility(returns: pd.DataFrame, gamma: float) -> float:
    if returns.empty:
        return float("-inf")
    net = pd.to_numeric(returns["net_return"], errors="coerce").dropna()
    if len(net) < 2:
        return float("-inf")
    return 12.0 * float(net.mean()) - 0.5 * float(gamma) * 12.0 * float(net.var(ddof=1))


def _max_drawdown(monthly_returns: pd.Series) -> float:
    if monthly_returns.empty:
        return float("nan")
    wealth = (1.0 + monthly_returns.clip(lower=-0.999999)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())
