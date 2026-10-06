from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from implementable_frontier.models.accounting import PortfolioAccounting
from implementable_frontier.models.features import (
    RandomFourierFeatures,
    monthly_cross_sectional_standardize,
)
from implementable_frontier.models.portfolios import (
    compute_adjustment_matrix,
    static_ml_star_weights,
    static_ml_weights,
)
from implementable_frontier.models.rolling_robustness import (
    monthly_method_differences,
    rolling_robustness_analysis,
)
from implementable_frontier.models.return_ml import ReturnModelSpec, RidgeReturnModel
from implementable_frontier.models.risk import RiskModelProvider
from implementable_frontier.models.backtest import run_backtest
from implementable_frontier.models.splits import fast_split, rolling_yearly_splits
from implementable_frontier.models.metrics import annual_performance


def test_random_fourier_features_are_reproducible_and_have_expected_shape() -> None:
    values = np.arange(30, dtype=float).reshape(10, 3) / 10.0
    first = RandomFourierFeatures(3, 32, 0.5, 17).transform(values)
    second = RandomFourierFeatures(3, 32, 0.5, 17).transform(values)
    assert first.shape == (10, 32)
    np.testing.assert_allclose(first, second)


def test_monthly_standardization_demeans_and_uses_unit_norm() -> None:
    values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])
    result = monthly_cross_sectional_standardize(values, add_constant=False)
    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.sqrt((result**2).sum(axis=0)), 1.0)


def test_accounting_aligns_permnos_and_charges_quadratic_cost() -> None:
    accounting = PortfolioAccounting(aum=100.0, gamma=10.0)
    inherited = accounting.compute_inherited_weights(
        pd.Series([0.4, 0.6], index=[2, 1]),
        [1, 2],
        pd.Series([0.0, 0.0], index=[1, 2]),
    )
    weights = pd.Series([0.5, 0.5], index=[1, 2])
    result = accounting.evaluate(
        weights,
        pd.Series([0.02, 0.01], index=[2, 1]),
        pd.Series([0.1, -0.1], index=[1, 2]),
        inherited,
    )
    expected_cost = 100.0 / 2.0 * (0.01 * (-0.1) ** 2 + 0.02 * 0.1**2)
    assert result.trading_cost == pytest.approx(expected_cost)
    assert result.turnover == pytest.approx(0.2)
    assert result.leverage == 1.0


def test_accounting_charges_exit_cost_for_dropped_names() -> None:
    accounting = PortfolioAccounting(aum=100.0, gamma=10.0)
    inherited = accounting.compute_inherited_weights(
        pd.Series([0.6, 0.4], index=[1, 2]),
        [1],
        pd.Series([0.0, 0.0], index=[1, 2]),
    )
    result = accounting.evaluate(
        pd.Series([0.6], index=[1]),
        pd.Series([0.01], index=[1]),
        pd.Series([0.0], index=[1]),
        inherited,
        previous_lambda=pd.Series([0.01, 0.02], index=[1, 2]),
    )
    assert result.exit_cost == pytest.approx(100.0 / 2.0 * 0.02 * 0.4**2)
    assert list(result.dropped_positions.index) == [2]


def test_risk_provider_returns_positive_diagonal_covariance() -> None:
    panel = pd.DataFrame(
        {
            "eom": pd.to_datetime(["2020-01-31"] * 3),
            "permno": [1, 2, 3],
            "vol_12m": [0.1, np.nan, 0.3],
        }
    )
    provider = RiskModelProvider(
        panel,
        universe="top100",
        requested_model="diagonal",
        settings={"diagonal_source": "vol_12m", "variance_floor": 1.0e-6},
    )
    result = provider.get_covariance("2020-01-31", [3, 1, 2])
    assert result.covariance.shape == (3, 3)
    assert (np.diag(result.covariance) > 0).all()
    assert result.missing_assets == (2,)


def test_factor_risk_assembles_and_imputes_small_idio_gap() -> None:
    eom = pd.Timestamp("2020-01-31")
    panel = pd.DataFrame(
        {"eom": [eom] * 3, "permno": [1, 2, 3], "vol_12m": [0.1, 0.1, 0.1]}
    )
    risk_inputs = {
        "factor_exposures": pd.DataFrame(
            {
                "eom": [eom] * 3,
                "permno": [1, 2, 3],
                "universe": ["top100"] * 3,
                "factor_x": [1.0, 0.0, -1.0],
            }
        ),
        "factor_cov_monthly": pd.DataFrame(
            {
                "eom": [eom],
                "universe": ["top100"],
                "factor_1": ["factor_x"],
                "factor_2": ["factor_x"],
                "covariance": [1.0e-4],
                "diagonal_fallback": [True],
            }
        ),
        "idio_var_monthly": pd.DataFrame(
            {
                "eom": [eom, eom],
                "universe": ["top100", "top100"],
                "permno": [1, 2],
                "idio_var": [1.0e-4, 2.0e-4],
            }
        ),
        "factor_returns_daily": pd.DataFrame(),
    }
    provider = RiskModelProvider(
        panel,
        universe="top100",
        requested_model="factor",
        settings={
            "diagonal_source": "vol_12m",
            "variance_floor": 1.0e-6,
            "factor_daily_to_monthly": 21.0,
            "allow_factor_fallback": False,
            "factor_max_idio_missing_fraction": 0.5,
            "factor_variance_ratio_max": 25.0,
            "factor_max_annualized_asset_volatility": 2.0,
        },
        risk_inputs=risk_inputs,
    )
    result = provider.get_covariance(eom, [1, 2, 3])
    assert result.used_model == "factor"
    assert result.idio_imputed_assets == (3,)
    assert result.factor_covariance_diagonal_fallback
    assert result.covariance[0, 2] < 0


def test_risk_scaled_backtest_hits_annualized_ex_ante_target() -> None:
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    panel = pd.DataFrame(
        {
            "eom": np.repeat(dates, 2),
            "permno": [1, 2, 1, 2],
            "vol_12m": [0.1] * 4,
            "lambda_adv": [1.0e-9] * 4,
            "ret_exc_lead1m": [0.01, -0.01, 0.01, -0.01],
        }
    )
    provider = RiskModelProvider(
        panel,
        universe="top100",
        requested_model="diagonal",
        settings={"diagonal_source": "vol_12m", "variance_floor": 1.0e-6},
    )
    result = run_backtest(
        panel,
        method="markowitz_ml",
        output_method="markowitz_ml_scaled_10pct_vol",
        risk_provider=provider,
        gamma=10.0,
        aum=1.0e8,
        accounting_settings={},
        method_settings={"covariance_ridge": 0.0},
        predictions=np.array([0.01, -0.01, 0.01, -0.01]),
        target_annual_volatility=0.10,
    )
    np.testing.assert_allclose(result.returns["ex_ante_annualized_volatility"], 0.10)


def test_static_ml_handles_singular_covariance_with_regularization() -> None:
    result = static_ml_weights(
        np.array([0.01, -0.01]),
        np.ones((2, 2)),
        np.zeros(2),
        np.array([1.0e-9, 2.0e-9]),
        gamma=10.0,
        aum=1.0e8,
        phi=3.0,
        mu_shrinkage=0.25,
        ridge=1.0e-8,
    )
    assert result.weights.shape == (2,)
    assert np.isfinite(result.weights).all()


def test_static_ml_star_matches_thesis_adjusted_quadratic_rule() -> None:
    expected_returns = np.array([0.02, -0.01])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    inherited = np.array([0.3, -0.2])
    lambdas = np.array([1.0e-9, 2.0e-9])
    gamma, aum, u, v, k = 10.0, 1.0e8, 0.5, 0.25, 2.0
    result = static_ml_star_weights(
        expected_returns,
        covariance,
        inherited,
        lambdas,
        gamma=gamma,
        aum=aum,
        u=u,
        v=v,
        k=k,
        ridge=0.0,
    )
    adjusted = covariance + v * np.diag(np.diag(covariance))
    tc_precision = aum * k * np.diag(lambdas)
    expected = np.linalg.solve(
        gamma * adjusted + tc_precision,
        expected_returns * u + tc_precision @ inherited,
    )
    np.testing.assert_allclose(result.weights, expected)


def test_rolling_robustness_outputs_monthly_inference_and_risk_calibration() -> None:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rows: list[dict[str, object]] = []
    method_levels = {
        "static_ml": 0.001,
        "static_ml_star": 0.002,
        "portfolio_ml_linear": 0.004,
        "portfolio_ml_rf": 0.003,
    }
    for method, level in method_levels.items():
        for index, eom in enumerate(dates):
            net = level + 0.002 * np.sin(index)
            rows.append(
                {
                    "eom": eom,
                    "test_year": eom.year,
                    "method": method,
                    "gross_return": net + 0.0001,
                    "net_return": net,
                    "utility_flow": net - 0.0005,
                    "ex_ante_monthly_variance": 0.01**2,
                }
            )
    returns = pd.DataFrame(rows)
    monthly = monthly_method_differences(returns)
    assert len(monthly) == 24 * 4
    analysis = rolling_robustness_analysis(
        returns,
        gamma=10.0,
        seed=7,
        newey_west_lags=3,
        bootstrap_block_months=6,
        bootstrap_replications=100,
    )
    assert len(analysis.difference_statistics) == 8
    assert set(analysis.risk_calibration_full["method"]) == set(method_levels)
    assert (
        analysis.difference_statistics["bootstrap_ci_lower"]
        <= analysis.difference_statistics["bootstrap_ci_upper"]
    ).all()


def test_dynamic_adjustment_is_stable_for_diagonal_risk() -> None:
    result = compute_adjustment_matrix(
        np.diag([0.02, 0.03]),
        np.array([1.0e-9, 2.0e-9]),
        gamma=10.0,
        aum=1.0e9,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
        allow_scalar_fallback=False,
        scalar_rho=0.5,
    )
    assert result.mode == "matrix_m"
    assert result.matrix.shape == (2, 2)
    assert 0.0 < result.spectral_radius < 1.0


def test_dynamic_adjustment_preserves_dense_covariance_result() -> None:
    """Synthetic regression case recorded before the equation-based rewrite."""
    covariance = np.array(
        [
            [0.00037991888866204275, -0.00010761541538319085],
            [-0.00010761541538319085, 0.0002861061016375227],
        ]
    )
    costs = np.array([4.4727538442493773e-10, 2.5534621844491144e-09])
    result = compute_adjustment_matrix(
        covariance,
        costs,
        gamma=10.0,
        aum=1.0e9,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
        allow_scalar_fallback=False,
        scalar_rho=0.5,
    )
    expected = np.array(
        [
            [0.9122044530072988, 0.018157824156236683],
            [0.003180602332496783, 0.9680117144127287],
        ]
    )
    np.testing.assert_allclose(result.matrix, expected, atol=1.0e-13, rtol=0.0)


def test_zero_aum_matrix_m_is_exact_frictionless_limit() -> None:
    result = compute_adjustment_matrix(
        np.array([[0.02, 0.005], [0.005, 0.03]]),
        np.array([1.0e-9, 2.0e-9]),
        gamma=10.0,
        aum=0.0,
        expected_market_return=0.007,
        risk_free_rate=0.0,
        iterations=10,
        allow_scalar_fallback=False,
        scalar_rho=0.5,
    )
    assert result.mode == "matrix_m"
    np.testing.assert_array_equal(result.matrix, np.zeros((2, 2)))
    assert result.spectral_radius == 0.0


def test_return_ml_records_training_end_and_does_not_fit_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "eom": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "rank_x": [0.1, 0.2, 0.3],
            "ret_exc_lead1m": [0.01, 0.02, 99.0],
        }
    )
    model = RidgeReturnModel(ReturnModelSpec("ridge", 0.1), ["rank_x"]).fit(frame.iloc[:2])
    assert model.train_end_ == pd.Timestamp("2020-02-29")


def test_split_protocols_have_no_overlap() -> None:
    config = {
        "fast_split": {
            "train_start": "1995-01-31",
            "train_end": "2014-12-31",
            "validation_start": "2015-01-31",
            "validation_end": "2017-12-31",
            "test_start": "2018-01-31",
            "test_end": "2024-12-31",
        },
        "rolling_yearly": {
            "first_test_year": 2005,
            "last_test_year": 2006,
            "analysis_start": "1995-01-31",
            "validation_years": 3,
        },
    }
    dates = pd.Series(pd.date_range("1995-01-31", "2024-12-31", freq="ME"))
    for split in [fast_split(config), *rolling_yearly_splits(config)]:
        train, validation, test = split.masks(dates)
        assert not (train & validation).any()
        assert not (train & test).any()
        assert not (validation & test).any()


def test_annual_performance_groups_each_rolling_test_year() -> None:
    returns = pd.DataFrame(
        {
            "method": ["static_ml"] * 4,
            "test_year": [2020, 2020, 2021, 2021],
            "gross_return": [0.01, 0.02, 0.03, 0.04],
            "trading_cost": [0.001] * 4,
            "net_return": [0.009, 0.019, 0.029, 0.039],
            "turnover": [0.1] * 4,
            "leverage": [1.0] * 4,
        }
    )
    metrics = annual_performance(returns, gamma=10.0)
    assert metrics["test_year"].tolist() == [2020, 2021]
    assert metrics["n_months"].tolist() == [2, 2]
