from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InheritedPositions:
    current: pd.Series
    dropped: pd.Series
    wealth_growth: float
    missing_return_names: tuple[int, ...]


@dataclass(frozen=True)
class AccountingResult:
    inherited: pd.Series
    trade: pd.Series
    dropped_positions: pd.Series
    gross_return: float
    trading_cost: float
    net_return: float
    utility_flow: float
    turnover: float
    leverage: float
    long_exposure: float
    short_exposure: float
    n_names: int
    exit_cost: float
    wealth_growth: float
    weighted_lambda_exposure: float
    max_lambda_adv: float


class PortfolioAccounting:
    def __init__(
        self,
        *,
        aum: float,
        gamma: float,
        charge_exit_costs: bool = True,
        missing_exit_return: float = 0.0,
        wealth_growth_floor: float = 0.1,
    ) -> None:
        self.aum = float(aum)
        self.gamma = float(gamma)
        self.charge_exit_costs = bool(charge_exit_costs)
        self.missing_exit_return = float(missing_exit_return)
        self.wealth_growth_floor = float(wealth_growth_floor)

    def compute_inherited_weights(
        self,
        prev_weights: pd.Series,
        current_permnos: Sequence[int],
        realized_returns: pd.Series,
        wealth_growth: float | None = None,
    ) -> InheritedPositions:
        previous = _series_by_permno(prev_weights)
        returns = _series_by_permno(realized_returns)
        missing = previous.index.difference(returns.index)
        aligned_returns = returns.reindex(previous.index).fillna(self.missing_exit_return)
        portfolio_return = float((previous * aligned_returns).sum())
        growth = float(wealth_growth if wealth_growth is not None else 1.0 + portfolio_return)
        growth = max(growth, self.wealth_growth_floor)
        grown = previous * (1.0 + aligned_returns) / growth
        current_index = pd.Index([int(value) for value in current_permnos], dtype="int64")
        inherited = grown.reindex(current_index).fillna(0.0)
        dropped = grown.loc[~grown.index.isin(current_index)]
        return InheritedPositions(
            current=inherited,
            dropped=dropped,
            wealth_growth=growth,
            missing_return_names=tuple(int(value) for value in missing),
        )

    def evaluate(
        self,
        current_weights: pd.Series,
        current_lambda: pd.Series,
        forward_returns: pd.Series,
        inherited: InheritedPositions,
        *,
        previous_lambda: pd.Series | None = None,
        ex_ante_variance: float = 0.0,
    ) -> AccountingResult:
        weights = _series_by_permno(current_weights)
        lambdas = _series_by_permno(current_lambda).reindex(weights.index)
        if lambdas.isna().any() or (lambdas <= 0).any():
            raise ValueError("Current lambda_adv must be positive for every portfolio asset")
        inherited_current = inherited.current.reindex(weights.index).fillna(0.0)
        trade = weights - inherited_current
        current_cost = self.aum / 2.0 * float((lambdas * trade**2).sum())
        exit_cost = 0.0
        if self.charge_exit_costs and not inherited.dropped.empty:
            prior_lambda = _series_by_permno(
                previous_lambda if previous_lambda is not None else pd.Series(dtype=float)
            )
            fallback = float(lambdas.median())
            dropped_lambda = prior_lambda.reindex(inherited.dropped.index).fillna(fallback)
            exit_cost = self.aum / 2.0 * float(
                (dropped_lambda * inherited.dropped**2).sum()
            )
        returns = _series_by_permno(forward_returns).reindex(weights.index).fillna(0.0)
        gross = float((weights * returns).sum())
        total_cost = current_cost + exit_cost
        net = gross - total_cost
        leverage = float(weights.abs().sum())
        long_exposure = float(weights.clip(lower=0).sum())
        short_exposure = float((-weights.clip(upper=0)).sum())
        turnover = float(trade.abs().sum() + inherited.dropped.abs().sum())
        weighted_lambda = (
            float((weights.abs() * lambdas).sum() / leverage) if leverage > 0 else 0.0
        )
        return AccountingResult(
            inherited=inherited_current,
            trade=trade,
            dropped_positions=inherited.dropped,
            gross_return=gross,
            trading_cost=total_cost,
            net_return=net,
            utility_flow=net - 0.5 * self.gamma * float(ex_ante_variance),
            turnover=turnover,
            leverage=leverage,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            n_names=int((weights != 0).sum()),
            exit_cost=exit_cost,
            wealth_growth=inherited.wealth_growth,
            weighted_lambda_exposure=weighted_lambda,
            max_lambda_adv=float(lambdas.max()),
        )


def _series_by_permno(values: pd.Series) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    result = pd.Series(values, dtype=float).copy()
    result.index = pd.Index([int(value) for value in result.index], dtype="int64")
    return result.groupby(level=0).last().sort_index()
