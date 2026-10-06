from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pytest

from implementable_frontier.models.backtest import BacktestResult, BacktestState
from implementable_frontier.models.cache import (
    DesignMatrixCache,
    PersistentAdjustmentCache,
    StaticPrecisionCache,
    TransactionCostCache,
)
from implementable_frontier.models.checkpoint import CheckpointStore
from implementable_frontier.models.cumulative_experiment import (
    CumulativeExperimentRunner,
    TimeBudgetPause,
    cumulative_validation_years,
)
from implementable_frontier.models.features import RandomFourierFeatures
from implementable_frontier.models.formal_analysis import (
    monthly_utility_differences,
    risk_calibration_report,
    statistical_inference,
    validation_stability_report,
)
from implementable_frontier.models.method_audit import run_small_sample_parity
from implementable_frontier.models.portfolios import AdjustmentMatrixResult
from implementable_frontier.models.portfolios import (
    static_ml_precision_matrix,
    static_ml_star_precision_matrix,
    static_ml_star_weights,
    static_ml_weights,
)
from implementable_frontier.models.config import load_model_config
from implementable_frontier.models.risk import RiskModelProvider


def test_cumulative_validation_uses_only_years_before_test() -> None:
    years = cumulative_validation_years(1996, 2005)
    assert list(years) == list(range(1996, 2005))
    assert max(years) < 2005
    with pytest.raises(ValueError):
        cumulative_validation_years(2005, 2005)


def test_time_budget_pause_is_recorded_only_at_checkpoint_boundary(
    tmp_path: Path,
) -> None:
    runner = CumulativeExperimentRunner.__new__(CumulativeExperimentRunner)
    runner.time_budget_seconds = 0.001
    runner.run_started = time.perf_counter() - 1.0
    runner.checkpoints = CheckpointStore(tmp_path / "checkpoints", resume=True)
    with pytest.raises(TimeBudgetPause) as paused:
        runner._maybe_pause_time_budget(
            namespace="validation", item_id="candidate", year=2001
        )
    status = json.loads(
        (runner.checkpoints.root / "run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "paused_time_budget"
    assert paused.value.payload["last_checkpoint_boundary"] == {
        "namespace": "validation",
        "item_id": "candidate",
        "year": 2001,
    }
    assert runner.checkpoints.errors == 0


def test_segment_restores_previous_test_year_state(tmp_path: Path) -> None:
    runner = CumulativeExperimentRunner.__new__(CumulativeExperimentRunner)
    runner.values = {
        "methods": ["static_ml", "portfolio_ml_linear"],
        "cumulative_oos_validation": {"first_test_year": 2005},
    }
    runner.checkpoints = CheckpointStore(tmp_path / "checkpoints")
    runner.final_states = {}
    state = BacktestState(
        weights=pd.Series([0.25], index=[10001]),
        forward_returns=pd.Series([0.01], index=[10001]),
        lambda_adv=pd.Series([1.0e-9], index=[10001]),
        market_growth=1.01,
    )
    empty = pd.DataFrame()
    result = BacktestResult(empty, empty, empty, [], {"factor"}, {"matrix_m"}, state)
    for method in runner.values["methods"]:
        runner.checkpoints.save("test", method, 2009, result)

    runner._restore_test_states_before(2010)

    assert set(runner.final_states) == {"static_ml", "portfolio_ml_linear"}
    assert runner.final_states["static_ml"].weights.loc[10001] == pytest.approx(0.25)


def test_segment_restore_fails_when_previous_year_is_missing(tmp_path: Path) -> None:
    runner = CumulativeExperimentRunner.__new__(CumulativeExperimentRunner)
    runner.values = {
        "methods": ["static_ml"],
        "cumulative_oos_validation": {"first_test_year": 2005},
    }
    runner.checkpoints = CheckpointStore(tmp_path / "checkpoints")
    runner.final_states = {}
    with pytest.raises(RuntimeError, match="requires test states from 2009"):
        runner._restore_test_states_before(2010)


def test_validation_stability_detects_future_information() -> None:
    selection = pd.DataFrame(
        {
            "test_year": [2005, 2006],
            "method": ["static_ml", "static_ml"],
            "candidate_hyperparameters": ["a", "b"],
            "selected": [True, True],
            "future_information_used": [False, True],
            "latest_validation_year": [2004, 2006],
        }
    )
    result = validation_stability_report(selection).iloc[0]
    assert result["future_information_violations"] == 1
    assert result["latest_validation_year_violation_count"] == 1


def test_rf_bank_is_fixed_and_nested_across_p() -> None:
    values = np.arange(30, dtype=float).reshape(10, 3) / 10.0
    small = RandomFourierFeatures(3, 32, 0.5, 17, max_features=64)
    large = RandomFourierFeatures(3, 64, 0.5, 17, max_features=64)
    np.testing.assert_allclose(small.weights, large.weights)
    small_values = small.transform(values)
    large_values = large.transform(values)
    scale = np.sqrt(64.0 / 32.0)
    np.testing.assert_allclose(small_values[:, :16], large_values[:, :16] * scale)
    np.testing.assert_allclose(small_values[:, 16:], large_values[:, 32:48] * scale)


def test_rf_cache_does_not_recompute_for_lambda_only_change(tmp_path: Path) -> None:
    cache = DesignMatrixCache(tmp_path / "rf")
    counter = {"calls": 0}

    def compute() -> np.ndarray:
        counter["calls"] += 1
        return np.eye(2)

    key = ("core10", "top100", pd.Timestamp("2005-01-31"), 32, 0.5, 7, "v2")
    first = cache.get_or_compute(key, np.array([1, 2]), compute)
    second = cache.get_or_compute(key, np.array([1, 2]), compute)
    np.testing.assert_allclose(first, second)
    assert counter["calls"] == 1
    assert cache.info()["hits"] == 1
    assert cache.info()["misses"] == 1
    subset = cache.get_or_compute(key, np.array([2, 1]), compute)
    np.testing.assert_allclose(subset, first)
    assert counter["calls"] == 2


def test_covariance_cache_reuses_month_and_permno_order() -> None:
    eom = pd.Timestamp("2005-01-31")
    panel = pd.DataFrame(
        {"eom": [eom, eom], "permno": [1, 2], "vol_12m": [0.2, 0.3]}
    )
    provider = RiskModelProvider(
        panel,
        universe="top100",
        requested_model="diagonal",
        settings={"diagonal_source": "vol_12m", "variance_floor": 1.0e-8},
    )
    first = provider.get_covariance(eom, [1, 2])
    second = provider.get_covariance(eom, [1, 2])
    np.testing.assert_allclose(first.covariance, second.covariance)
    assert provider.cache_info()["hits"] == 1
    reversed_order = provider.get_covariance(eom, [2, 1])
    np.testing.assert_allclose(
        reversed_order.covariance, first.covariance[np.ix_([1, 0], [1, 0])]
    )
    assert provider.cache_info()["misses"] == 2


def test_matrix_m_cache_key_separates_gamma_and_aum(tmp_path: Path) -> None:
    cache = PersistentAdjustmentCache(tmp_path / "m")
    expected_eigenvalues = np.array([0.5, 0.5])
    result = AdjustmentMatrixResult(
        np.eye(2) * 0.5,
        "matrix_m",
        0.5,
        None,
        expected_eigenvalues,
    )
    base = ("top100", pd.Timestamp("2005-01-31"), "factor", "v2", 1.0e9, 10.0, "cost-v1")
    cache[base] = result
    reloaded = PersistentAdjustmentCache(tmp_path / "m")[base]
    assert reloaded.mode == "matrix_m"
    np.testing.assert_allclose(reloaded.eigenvalues, expected_eigenvalues)
    with pytest.raises(KeyError):
        _ = cache[base[:-2] + (20.0, "cost-v1")]
    with pytest.raises(KeyError):
        _ = cache[base[:4] + (2.0e9, 10.0, "cost-v1")]


def test_transaction_cost_cache_validates_dynamic_permno_order(tmp_path: Path) -> None:
    cache = TransactionCostCache(tmp_path / "tc")
    values = cache.get(
        universe="top100",
        eom=pd.Timestamp("2005-01-31"),
        aum=1.0e9,
        cost_model_version="v1",
        permnos=np.array([1, 2]),
        values=np.array([1.0e-9, 2.0e-9]),
    )
    assert list(values.index) == [1, 2]
    with pytest.raises(ValueError):
        cache.get(
            universe="top100",
            eom=pd.Timestamp("2005-01-31"),
            aum=1.0e9,
            cost_model_version="v1",
            permnos=np.array([2, 1]),
            values=np.array([2.0e-9, 1.0e-9]),
        )


def test_static_precision_cache_matches_direct_solve() -> None:
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(8, 8))
    covariance = raw @ raw.T / 100.0 + np.eye(8) * 0.01
    lambdas = np.linspace(1.0e-9, 2.0e-9, 8)
    expected = rng.normal(scale=0.01, size=8)
    inherited = rng.normal(scale=0.1, size=8)
    cache = StaticPrecisionCache()

    static_matrix = static_ml_precision_matrix(
        covariance, lambdas, gamma=10, aum=1.0e9, phi=3, ridge=1.0e-8
    )
    static_inverse = cache.get_or_compute(("static", 3), lambda: static_matrix)
    direct_static = static_ml_weights(
        expected,
        covariance,
        inherited,
        lambdas,
        gamma=10,
        aum=1.0e9,
        phi=3,
        mu_shrinkage=0.25,
        ridge=1.0e-8,
    )
    cached_static = static_ml_weights(
        expected,
        covariance,
        inherited,
        lambdas,
        gamma=10,
        aum=1.0e9,
        phi=3,
        mu_shrinkage=0.25,
        ridge=1.0e-8,
        precision_inverse=static_inverse.matrix,
    )
    np.testing.assert_allclose(cached_static.weights, direct_static.weights, rtol=1e-11)

    star_matrix = static_ml_star_precision_matrix(
        covariance, lambdas, gamma=10, aum=1.0e9, v=0.5, k=2, ridge=1.0e-8
    )
    star_inverse = cache.get_or_compute(("star", 0.5, 2), lambda: star_matrix)
    direct_star = static_ml_star_weights(
        expected,
        covariance,
        inherited,
        lambdas,
        gamma=10,
        aum=1.0e9,
        u=0.25,
        v=0.5,
        k=2,
        ridge=1.0e-8,
    )
    cached_star = static_ml_star_weights(
        expected,
        covariance,
        inherited,
        lambdas,
        gamma=10,
        aum=1.0e9,
        u=0.25,
        v=0.5,
        k=2,
        ridge=1.0e-8,
        precision_inverse=star_inverse.matrix,
    )
    np.testing.assert_allclose(cached_star.weights, direct_star.weights, rtol=1e-11)
    cache.get_or_compute(("star", 0.5, 2), lambda: star_matrix)
    assert cache.info()["hits"] == 1
    assert cache.info()["misses"] == 2


def test_top500_config_is_isolated_and_keeps_small_grids() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "configs/model_top500_core10_cumulative.yaml")
    values = config.values
    assert values["universe"] == "top500"
    assert values["feature_set"] == "core10"
    assert values["protocol"] == "cumulative_oos_validation"
    assert values["cumulative_oos_validation"]["pilot_test_years"] == [2005, 2006, 2007]
    assert "markowitz_ml" not in values["methods"]
    assert values["portfolio_ml"]["rf_p_grid"] == [32, 64]
    assert "top500_core10_cumulative" in values["checkpoint"]["root"]
    assert values["profiling"]["full_run_limit_hours"] == 18


def test_top500_final15_config_preserves_core_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    core = load_model_config(root / "configs/model_top500_core10_cumulative.yaml")
    final = load_model_config(root / "configs/model_top500_final15_cumulative.yaml")
    assert final.values["universe"] == "top500"
    assert final.values["feature_set"] == "final15"
    assert final.values["baseline_aum"] == core.values["baseline_aum"]
    assert final.values["baseline_gamma"] == core.values["baseline_gamma"]
    assert final.values["methods"] == core.values["methods"]
    assert final.values["portfolio_ml"]["rf_p_grid"] == core.values["portfolio_ml"]["rf_p_grid"]
    assert final.values["portfolio_ml"]["rf_eta_grid"] == core.values["portfolio_ml"]["rf_eta_grid"]
    assert "top500_final15_cumulative" in final.values["checkpoint"]["root"]
    assert final.values["cache"]["root"] == core.values["cache"]["root"]
    assert final.values["profiling"]["full_run_limit_hours"] == 24


def test_checkpoint_resume_preserves_state_and_skips_completed(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints", resume=True)
    result = BacktestResult(
        predictions=pd.DataFrame(),
        weights=pd.DataFrame(),
        returns=pd.DataFrame(
            {"eom": [pd.Timestamp("2005-01-31")], "method": ["static_ml"]}
        ),
        risk_models_used={"factor"},
        adjustment_modes=set(),
        final_state=BacktestState(
            weights=pd.Series([0.3], index=[1]),
            forward_returns=pd.Series([0.01], index=[1]),
            lambda_adv=pd.Series([1.0e-9], index=[1]),
            market_growth=1.01,
        ),
    )
    store.save("validation", "candidate", 2005, result)
    assert store.has("validation", "candidate", 2005)
    loaded = store.load("validation", "candidate", 2005)
    assert loaded.final_state.market_growth == pytest.approx(1.01)
    assert loaded.final_state.weights.loc[1] == pytest.approx(0.3)
    assert store.writes == 1


def test_small_sample_parity_covers_foc_hessian_and_solver() -> None:
    result = run_small_sample_parity(seed=60710)
    checks = {row["name"]: row for row in result["checks"]}
    assert result["passed"]
    assert checks["gradient_norm"]["passed"]
    assert checks["hessian_psd"]["passed"]
    assert checks["closed_form_vs_iterative_solver"]["passed"]


def test_formal_differences_hac_bootstrap_and_risk_calibration() -> None:
    dates = pd.date_range("2005-01-31", periods=24, freq="ME")
    levels = {
        "static_ml": 0.001,
        "static_ml_star": 0.0015,
        "portfolio_ml_linear": 0.002,
        "portfolio_ml_rf": 0.0025,
    }
    rows: list[dict[str, object]] = []
    variance = 0.01**2
    for method, level in levels.items():
        for index, eom in enumerate(dates):
            net = level + 0.001 * np.sin(index)
            rows.append(
                {
                    "eom": eom,
                    "test_year": eom.year,
                    "method": method,
                    "gross_return": net + 0.0001,
                    "trading_cost": 0.0001,
                    "net_return": net,
                    "ex_ante_monthly_variance": variance,
                    "utility_flow": net - 5.0 * variance,
                    "turnover": 0.2,
                    "leverage": 1.0,
                    "risk_model_used": "factor",
                    "risk_fallback_reason": None,
                    "covariance_repair_count": 0,
                    "covariance_min_eigenvalue": 1.0e-5,
                    "covariance_condition_number": 100.0,
                }
            )
    returns = pd.DataFrame(rows)
    monthly = monthly_utility_differences(returns, gamma=10.0)
    assert len(monthly) == 24 * 5
    np.testing.assert_allclose(
        monthly["delta_utility_flow"],
        monthly["delta_net_return"] - monthly["delta_risk_penalty"],
    )
    inference = statistical_inference(
        monthly,
        newey_west_lags=3,
        bootstrap_block_months=6,
        bootstrap_replications=100,
        seed=7,
    )
    assert len(inference) == 5 * 7
    assert inference["newey_west_t_stat"].notna().any()
    assert (
        inference["bootstrap_annualized_ci_lower"]
        <= inference["bootstrap_annualized_ci_upper"]
    ).all()
    risk = risk_calibration_report(returns)
    assert set(risk[risk["scope"].eq("summary")]["method"]) == set(levels)
    assert risk["factor_risk_fallback_count"].fillna(0).eq(0).all()
