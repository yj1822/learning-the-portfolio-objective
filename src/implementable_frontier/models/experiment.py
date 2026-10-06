from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from implementable_frontier.data.loaders import load_master_panel, load_risk_inputs
from implementable_frontier.models.backtest import (
    BacktestResult,
    BacktestState,
    run_backtest,
)
from implementable_frontier.models.config import ModelConfig, load_model_config
from implementable_frontier.models.metrics import (
    annual_performance,
    performance_metrics,
    validation_utility,
)
from implementable_frontier.models.portfolio_ml import (
    PortfolioMLModel,
    PortfolioMLSpec,
    AdjustmentCache,
    portfolio_ml_specs,
)
from implementable_frontier.models.return_ml import (
    ReturnModelSpec,
    RidgeReturnModel,
    return_model_specs,
)
from implementable_frontier.models.risk import RiskModelProvider
from implementable_frontier.models.rolling_robustness import (
    RollingRobustnessResult,
    rolling_robustness_analysis,
)
from implementable_frontier.models.splits import ExperimentSplit, experiment_splits
from implementable_frontier.models.unit_audit import (
    add_backtest_reconciliation,
    audit_model_units,
    unit_audit_markdown,
)


@dataclass
class ExperimentResult:
    experiment_id: str
    output_dir: Path
    report_path: Path
    metrics: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class _SplitResult:
    backtests: list[BacktestResult]
    validation_rows: list[dict[str, object]]
    states: dict[str, BacktestState]
    selected: dict[str, dict[str, object]]


def run_model_experiment(
    config_path: str | Path,
    *,
    experiment_id: str | None = None,
    methods: Sequence[str] | None = None,
    dry_run: bool = False,
) -> ExperimentResult | dict[str, object]:
    config = load_model_config(config_path)
    selected_methods = _selected_methods(config, methods)
    splits = experiment_splits(config.values)
    start = min(split.train_start for split in splits)
    end = max(split.test_end for split in splits)
    panel = load_master_panel(
        start=str(start.date()),
        end=str(end.date()),
        universe=str(config.values["universe"]),
        feature_set=str(config.values["feature_set"]),
    )
    dry_summary = {
        "config": str(config.path),
        "methods": selected_methods,
        "protocol": config.values["protocol"],
        "date_range": [str(panel["eom"].min().date()), str(panel["eom"].max().date())],
        "rows": int(len(panel)),
        "months": int(panel["eom"].nunique()),
        "universe": config.values["universe"],
        "feature_set": config.values["feature_set"],
        "risk_model_requested": config.values["risk_model"],
    }
    if dry_run:
        return dry_summary

    experiment_id = experiment_id or _default_experiment_id(config)
    output_dir = config.experiments_dir / experiment_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment output already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_inputs = (
        load_risk_inputs(
            start=str(start.date()),
            end=str(end.date()),
            universe=str(config.values["universe"]),
            root=(
                config.resolve_path(config.values["risk"]["inputs_dir"])
                if config.values["risk"].get("inputs_dir")
                else None
            ),
        )
        if config.values["risk_model"] == "factor"
        else None
    )
    unit_audit = audit_model_units(
        panel,
        aum=float(config.values["baseline_aum"]),
        gamma=float(config.values["baseline_gamma"]),
        risk_inputs=risk_inputs,
    )
    risk_provider = RiskModelProvider(
        panel,
        universe=str(config.values["universe"]),
        requested_model=str(config.values["risk_model"]),
        settings=config.values["risk"],
        risk_inputs=risk_inputs,
    )
    adjustment_cache: AdjustmentCache = {}
    all_backtests: list[BacktestResult] = []
    validation_rows: list[dict[str, object]] = []
    states: dict[str, BacktestState] = {}
    selections: dict[str, dict[str, object]] = {}
    for split in splits:
        split_result = _run_split(
            panel,
            split,
            config,
            selected_methods,
            risk_provider,
            states,
            adjustment_cache,
        )
        all_backtests.extend(split_result.backtests)
        validation_rows.extend(split_result.validation_rows)
        states = split_result.states
        selections[split.name] = split_result.selected

    predictions = _concat_frames([result.predictions for result in all_backtests])
    weights = _concat_frames([result.weights for result in all_backtests])
    returns = _concat_frames([result.returns for result in all_backtests])
    gamma = float(config.values["baseline_gamma"])
    unit_audit = add_backtest_reconciliation(
        unit_audit,
        panel,
        weights,
        returns,
        aum=float(config.values["baseline_aum"]),
        gamma=gamma,
    )
    metrics = performance_metrics(returns, gamma)
    scaled_mask = returns.get(
        "is_risk_scaled_diagnostic", pd.Series(False, index=returns.index)
    ).fillna(False).astype(bool)
    metrics_raw = performance_metrics(returns.loc[~scaled_mask], gamma)
    metrics_scaled = performance_metrics(returns.loc[scaled_mask], gamma)
    annual_metrics = annual_performance(returns.loc[~scaled_mask], gamma)
    robustness = (
        rolling_robustness_analysis(
            returns.loc[~scaled_mask],
            gamma=gamma,
            seed=int(config.values["random_seed"]),
            newey_west_lags=int(
                config.values.get("robustness", {}).get("newey_west_lags", 6)
            ),
            bootstrap_block_months=int(
                config.values.get("robustness", {}).get("bootstrap_block_months", 12)
            ),
            bootstrap_replications=int(
                config.values.get("robustness", {}).get("bootstrap_replications", 2_000)
            ),
        )
        if config.values["protocol"] == "rolling_yearly"
        else None
    )
    diagnostic_summary = _diagnostic_summary(returns)
    diagnostic_findings = _diagnostic_findings(
        predictions,
        metrics_raw,
        metrics_scaled,
        diagnostic_summary,
        selections,
        config,
    )
    validation = pd.DataFrame(validation_rows)
    selected_by_year = (
        validation.loc[validation["selected"].fillna(False)].reset_index(drop=True)
        if not validation.empty
        else pd.DataFrame()
    )
    warnings = list(
        dict.fromkeys(warning for result in all_backtests for warning in result.warnings)
    )
    risk_models_used = sorted(
        {value for result in all_backtests for value in result.risk_models_used}
    )
    adjustment_modes = sorted(
        {value for result in all_backtests for value in result.adjustment_modes}
    )
    metadata = _metadata(
        config,
        experiment_id,
        selected_methods,
        panel,
        risk_models_used,
        adjustment_modes,
        selections,
        returns,
        unit_audit,
        diagnostic_findings,
    )
    _write_outputs(
        config,
        output_dir,
        metadata,
        predictions,
        weights,
        returns,
        metrics,
        metrics_raw,
        metrics_scaled,
        annual_metrics,
        selected_by_year,
        validation,
        warnings,
        unit_audit,
        diagnostic_summary,
        diagnostic_findings,
        robustness,
    )
    report_path = _write_report(
        config,
        experiment_id,
        metadata,
        metrics_raw,
        metrics_scaled,
        warnings,
        unit_audit,
        diagnostic_summary,
        diagnostic_findings,
    )
    if config.values["protocol"] == "rolling_yearly":
        report_path = _write_rolling_summary(
            config,
            experiment_id,
            output_dir,
            metrics_raw,
            annual_metrics,
            selected_by_year,
            robustness,
        )
    return ExperimentResult(experiment_id, output_dir, report_path, metrics, metadata)


def _run_split(
    panel: pd.DataFrame,
    split: ExperimentSplit,
    config: ModelConfig,
    methods: list[str],
    risk_provider: RiskModelProvider,
    states: dict[str, BacktestState],
    adjustment_cache: AdjustmentCache,
) -> _SplitResult:
    train_mask, validation_mask, test_mask = split.masks(panel["eom"])
    train = panel.loc[train_mask].sort_values(["eom", "permno"]).reset_index(drop=True)
    validation = panel.loc[validation_mask].sort_values(["eom", "permno"]).reset_index(drop=True)
    test = panel.loc[test_mask].sort_values(["eom", "permno"]).reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{split.name} has an empty train, validation, or test partition")
    gamma = float(config.values["baseline_gamma"])
    aum = float(config.values["baseline_aum"])
    seed = int(config.values["random_seed"])
    accounting = config.values["accounting"]
    diagnostics = config.values.get("diagnostics", {})
    backtests: list[BacktestResult] = []
    selection_rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    updated_states = states.copy()

    return_methods = {
        "return_ml_rank",
        "markowitz_ml",
        "static_ml",
        "static_ml_star",
    }
    selected_return_spec: ReturnModelSpec | None = None
    selection_return_model: RidgeReturnModel | None = None
    test_return_model: RidgeReturnModel | None = None
    validation_predictions: np.ndarray | None = None
    test_predictions: np.ndarray | None = None
    if return_methods.intersection(methods):
        candidates: list[tuple[float, ReturnModelSpec, RidgeReturnModel]] = []
        return_groups: dict[tuple[object, ...], list[ReturnModelSpec]] = {}
        for spec in return_model_specs(config.values["return_ml"], seed):
            key = (spec.kind, spec.rf_p, spec.rf_eta, spec.seed)
            return_groups.setdefault(key, []).append(spec)
        candidate_index = 0
        for group_specs in return_groups.values():
            model = RidgeReturnModel(group_specs[0], config.feature_columns).fit(train)
            for spec in group_specs:
                model.solve_ridge(spec.ridge_lambda)
                candidate_predictions = model.predict(validation)
                result = run_backtest(
                    validation,
                    method="return_ml_rank",
                    risk_provider=risk_provider,
                    gamma=gamma,
                    aum=aum,
                    accounting_settings=accounting,
                    method_settings=_settings(config.values["return_ml"], diagnostics),
                    predictions=candidate_predictions,
                )
                utility = validation_utility(result.returns, gamma)
                candidates.append((utility, spec, model))
                selection_rows.append(
                    _selection_row(
                        split,
                        "return_ml",
                        candidate_index,
                        spec.metadata(),
                        utility,
                    )
                )
                candidate_index += 1
        best_index = int(np.argmax([item[0] for item in candidates]))
        _, selected_return_spec, selection_return_model = candidates[best_index]
        selection_return_model.solve_ridge(selected_return_spec.ridge_lambda)
        _mark_selected(selection_rows, split, "return_ml", best_index)
        validation_predictions = selection_return_model.predict(validation)
        selected["return_ml"] = selected_return_spec.metadata()
        refit = pd.concat([train, validation], ignore_index=True)
        test_return_model = RidgeReturnModel(selected_return_spec, config.feature_columns).fit(refit)
        test_predictions = test_return_model.predict(test)

    if "equal_weight" in methods:
        result = run_backtest(
            test,
            method="equal_weight",
            risk_provider=risk_provider,
            gamma=gamma,
            aum=aum,
            accounting_settings=accounting,
            method_settings=_settings({}, diagnostics),
            test_year=split.test_year,
            initial_state=states.get("equal_weight"),
        )
        backtests.append(result)
        updated_states["equal_weight"] = result.final_state

    for method in ("return_ml_rank", "markowitz_ml"):
        if method not in methods:
            continue
        settings = config.values["return_ml"] if method == "return_ml_rank" else config.values["markowitz_ml"]
        beta_norm = (
            float(np.linalg.norm(test_return_model.coef_))
            if test_return_model is not None
            else None
        )
        result = run_backtest(
            test,
            method=method,
            risk_provider=risk_provider,
            gamma=gamma,
            aum=aum,
            accounting_settings=accounting,
            method_settings=_settings(settings, diagnostics),
            predictions=test_predictions,
            test_year=split.test_year,
            initial_state=states.get(method),
            beta_norm=beta_norm,
        )
        backtests.append(result)
        updated_states[method] = result.final_state
        target_volatility = _risk_scaling_target(config, method)
        if target_volatility is not None:
            scaled_method = _scaled_method_name(method, target_volatility)
            scaled_result = run_backtest(
                test,
                method=method,
                output_method=scaled_method,
                risk_provider=risk_provider,
                gamma=gamma,
                aum=aum,
                accounting_settings=accounting,
                method_settings=_settings(settings, diagnostics),
                predictions=test_predictions,
                test_year=split.test_year,
                initial_state=states.get(scaled_method),
                target_annual_volatility=target_volatility,
                beta_norm=beta_norm,
            )
            backtests.append(scaled_result)
            updated_states[scaled_method] = scaled_result.final_state

    if "static_ml_star" in methods:
        if validation_predictions is None or test_predictions is None:
            raise RuntimeError("Static-ML* requires selected Return-ML predictions")
        star_candidates: list[tuple[float, float, float, float]] = []
        candidate_index = 0
        star_settings = config.values["static_ml_star"]
        for u in star_settings["u_grid"]:
            for v in star_settings["v_grid"]:
                for k in star_settings["k_grid"]:
                    candidate_settings = _settings(
                        {
                            **star_settings,
                            "u": float(u),
                            "v": float(v),
                            "k": float(k),
                        },
                        diagnostics,
                    )
                    result = run_backtest(
                        validation,
                        method="static_ml_star",
                        risk_provider=risk_provider,
                        gamma=gamma,
                        aum=aum,
                        accounting_settings=accounting,
                        method_settings=candidate_settings,
                        predictions=validation_predictions,
                    )
                    utility = validation_utility(result.returns, gamma)
                    star_candidates.append(
                        (utility, float(u), float(v), float(k))
                    )
                    selection_rows.append(
                        _selection_row(
                            split,
                            "static_ml_star",
                            candidate_index,
                            {
                                "u": float(u),
                                "v": float(v),
                                "official_g": float(v),
                                "k": float(k),
                                "covariance_adjustment": "cov_add",
                            },
                            utility,
                        )
                    )
                    candidate_index += 1
        best_index = int(np.argmax([item[0] for item in star_candidates]))
        _, u, v, k = star_candidates[best_index]
        _mark_selected(selection_rows, split, "static_ml_star", best_index)
        selected["static_ml_star"] = {
            "u": u,
            "v": v,
            "official_g": v,
            "k": k,
            "covariance_adjustment": "cov_add",
            "selection_information": "past_validation_only",
        }
        final_settings = _settings(
            {**star_settings, "u": u, "v": v, "k": k}, diagnostics
        )
        result = run_backtest(
            test,
            method="static_ml_star",
            risk_provider=risk_provider,
            gamma=gamma,
            aum=aum,
            accounting_settings=accounting,
            method_settings=final_settings,
            predictions=test_predictions,
            test_year=split.test_year,
            initial_state=states.get("static_ml_star"),
            beta_norm=(
                float(np.linalg.norm(test_return_model.coef_))
                if test_return_model is not None
                else None
            ),
        )
        backtests.append(result)
        updated_states["static_ml_star"] = result.final_state
        target_volatility = _risk_scaling_target(config, "static_ml_star")
        if target_volatility is not None:
            scaled_method = _scaled_method_name("static_ml_star", target_volatility)
            scaled_result = run_backtest(
                test,
                method="static_ml_star",
                output_method=scaled_method,
                risk_provider=risk_provider,
                gamma=gamma,
                aum=aum,
                accounting_settings=accounting,
                method_settings=final_settings,
                predictions=test_predictions,
                test_year=split.test_year,
                initial_state=states.get(scaled_method),
                target_annual_volatility=target_volatility,
                beta_norm=(
                    float(np.linalg.norm(test_return_model.coef_))
                    if test_return_model is not None
                    else None
                ),
            )
            backtests.append(scaled_result)
            updated_states[scaled_method] = scaled_result.final_state

    if "static_ml" in methods:
        if validation_predictions is None or test_predictions is None:
            raise RuntimeError("Static-ML requires selected Return-ML predictions")
        static_candidates: list[tuple[float, float, float]] = []
        candidate_index = 0
        for phi in config.values["static_ml"]["phi_grid"]:
            for shrinkage in config.values["static_ml"]["mu_shrinkage_grid"]:
                candidate_settings = _settings(
                    {
                        **config.values["static_ml"],
                        "phi": float(phi),
                        "mu_shrinkage": float(shrinkage),
                    },
                    diagnostics,
                )
                result = run_backtest(
                    validation,
                    method="static_ml",
                    risk_provider=risk_provider,
                    gamma=gamma,
                    aum=aum,
                    accounting_settings=accounting,
                    method_settings=candidate_settings,
                    predictions=validation_predictions,
                )
                utility = validation_utility(result.returns, gamma)
                static_candidates.append((utility, float(phi), float(shrinkage)))
                selection_rows.append(
                    _selection_row(
                        split,
                        "static_ml",
                        candidate_index,
                        {"phi": float(phi), "mu_shrinkage": float(shrinkage)},
                        utility,
                    )
                )
                candidate_index += 1
        best_index = int(np.argmax([item[0] for item in static_candidates]))
        _, phi, shrinkage = static_candidates[best_index]
        _mark_selected(selection_rows, split, "static_ml", best_index)
        selected["static_ml"] = {"phi": phi, "mu_shrinkage": shrinkage}
        result = run_backtest(
            test,
            method="static_ml",
            risk_provider=risk_provider,
            gamma=gamma,
            aum=aum,
            accounting_settings=accounting,
            method_settings=_settings(
                {
                    **config.values["static_ml"],
                    "phi": phi,
                    "mu_shrinkage": shrinkage,
                },
                diagnostics,
            ),
            predictions=test_predictions,
            test_year=split.test_year,
            initial_state=states.get("static_ml"),
            beta_norm=(
                float(np.linalg.norm(test_return_model.coef_))
                if test_return_model is not None
                else None
            ),
        )
        backtests.append(result)
        updated_states["static_ml"] = result.final_state
        target_volatility = _risk_scaling_target(config, "static_ml")
        if target_volatility is not None:
            scaled_method = _scaled_method_name("static_ml", target_volatility)
            scaled_result = run_backtest(
                test,
                method="static_ml",
                output_method=scaled_method,
                risk_provider=risk_provider,
                gamma=gamma,
                aum=aum,
                accounting_settings=accounting,
                method_settings=_settings(
                    {
                        **config.values["static_ml"],
                        "phi": phi,
                        "mu_shrinkage": shrinkage,
                    },
                    diagnostics,
                ),
                predictions=test_predictions,
                test_year=split.test_year,
                initial_state=states.get(scaled_method),
                target_annual_volatility=target_volatility,
                beta_norm=(
                    float(np.linalg.norm(test_return_model.coef_))
                    if test_return_model is not None
                    else None
                ),
            )
            backtests.append(scaled_result)
            updated_states[scaled_method] = scaled_result.final_state

    for method, kind in (
        ("portfolio_ml_linear", "linear"),
        ("portfolio_ml_rf", "rf"),
    ):
        if method not in methods:
            continue
        model, candidate_rows, selected_spec = _select_portfolio_ml(
            train,
            validation,
            split,
            method,
            kind,
            config,
            risk_provider,
            gamma,
            aum,
            adjustment_cache,
        )
        selection_rows.extend(candidate_rows)
        selected[method] = selected_spec.metadata()
        refit = pd.concat([train, validation], ignore_index=True)
        final_model = PortfolioMLModel(
            selected_spec,
            config.feature_columns,
            risk_provider,
            gamma=gamma,
            aum=aum,
            settings=config.values["portfolio_ml"],
            adjustment_cache=adjustment_cache,
        ).fit(refit)
        result = run_backtest(
            test,
            method=method,
            risk_provider=risk_provider,
            gamma=gamma,
            aum=aum,
            accounting_settings=accounting,
            method_settings=_settings(config.values["portfolio_ml"], diagnostics),
            portfolio_model=final_model,
            test_year=split.test_year,
            initial_state=states.get(method),
            beta_norm=float(np.linalg.norm(final_model.beta_)),
        )
        result.warnings.extend(final_model.diagnostics)
        backtests.append(result)
        updated_states[method] = result.final_state
        target_volatility = _risk_scaling_target(config, method)
        if target_volatility is not None:
            scaled_method = _scaled_method_name(method, target_volatility)
            scaled_result = run_backtest(
                test,
                method=method,
                output_method=scaled_method,
                risk_provider=risk_provider,
                gamma=gamma,
                aum=aum,
                accounting_settings=accounting,
                method_settings=_settings(config.values["portfolio_ml"], diagnostics),
                portfolio_model=final_model,
                test_year=split.test_year,
                initial_state=states.get(scaled_method),
                target_annual_volatility=target_volatility,
                beta_norm=float(np.linalg.norm(final_model.beta_)),
            )
            scaled_result.warnings.extend(final_model.diagnostics)
            backtests.append(scaled_result)
            updated_states[scaled_method] = scaled_result.final_state
    return _SplitResult(backtests, selection_rows, updated_states, selected)


def _select_portfolio_ml(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    split: ExperimentSplit,
    method: str,
    kind: str,
    config: ModelConfig,
    risk_provider: RiskModelProvider,
    gamma: float,
    aum: float,
    adjustment_cache: AdjustmentCache,
) -> tuple[PortfolioMLModel, list[dict[str, object]], PortfolioMLSpec]:
    settings = config.values["portfolio_ml"]
    seed = int(config.values["random_seed"])
    specs = portfolio_ml_specs(settings, kind, seed)
    map_groups: dict[tuple[object, ...], list[PortfolioMLSpec]] = {}
    for spec in specs:
        key = (spec.kind, spec.rf_p, spec.rf_eta, spec.seed)
        map_groups.setdefault(key, []).append(spec)
    candidates: list[tuple[float, PortfolioMLSpec, PortfolioMLModel]] = []
    rows: list[dict[str, object]] = []
    candidate_index = 0
    for group_specs in map_groups.values():
        base_model = PortfolioMLModel(
            group_specs[0],
            config.feature_columns,
            risk_provider,
            gamma=gamma,
            aum=aum,
            settings=settings,
            adjustment_cache=adjustment_cache,
        ).fit(train)
        for spec in group_specs:
            base_model.solve_beta(spec.ridge_lambda)
            result = run_backtest(
                validation,
                method=method,
                risk_provider=risk_provider,
                gamma=gamma,
                aum=aum,
                accounting_settings=config.values["accounting"],
                method_settings=_settings(settings, config.values.get("diagnostics", {})),
                portfolio_model=base_model,
            )
            utility = validation_utility(result.returns, gamma)
            candidates.append((utility, spec, base_model))
            rows.append(
                _selection_row(split, method, candidate_index, spec.metadata(), utility)
            )
            candidate_index += 1
    best_index = int(np.argmax([item[0] for item in candidates]))
    _, selected_spec, selected_model = candidates[best_index]
    selected_model.solve_beta(selected_spec.ridge_lambda)
    _mark_selected(rows, split, method, best_index)
    return selected_model, rows, selected_spec


def _selection_row(
    split: ExperimentSplit,
    method: str,
    candidate_index: int,
    candidate: dict[str, object],
    utility: float,
) -> dict[str, object]:
    return {
        "year": split.test_year,
        "split": split.name,
        "method": method,
        "candidate_index": candidate_index,
        "candidate_hyperparameters": json.dumps(candidate, sort_keys=True),
        "validation_utility": float(utility),
        "selected": False,
    }


def _mark_selected(
    rows: list[dict[str, object]],
    split: ExperimentSplit,
    method: str,
    candidate_index: int,
) -> None:
    for row in reversed(rows):
        if (
            row["split"] == split.name
            and row["method"] == method
            and row["candidate_index"] == candidate_index
        ):
            row["selected"] = True
            return
    raise RuntimeError("Unable to mark selected validation candidate")


def _settings(values: dict[str, object], diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        **values,
        "max_abs_weight_warning": diagnostics.get("max_abs_weight_warning", 5.0),
        "max_leverage_warning": diagnostics.get("max_leverage_warning", 20.0),
        "covariance_condition_warning": diagnostics.get(
            "covariance_condition_warning", 1.0e10
        ),
    }


def _selected_methods(config: ModelConfig, methods: Sequence[str] | None) -> list[str]:
    selected = list(config.methods) if methods is None else [str(value) for value in methods]
    unknown = set(selected) - set(config.methods)
    if unknown:
        raise ValueError(f"Methods are not enabled by the config: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one method must be selected")
    return selected


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    # Keep the common backtest schema even when an optional diagnostic is all-null
    # for a method-only run (for example aim leverage outside Portfolio-ML).
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _risk_scaling_target(config: ModelConfig, method: str) -> float | None:
    settings = config.values.get("diagnostic_risk_scaling", {})
    if not bool(settings.get("enabled", False)):
        return None
    if method not in settings.get("methods", []):
        return None
    target = float(settings.get("target_annual_volatility", 0.10))
    if target <= 0:
        raise ValueError("diagnostic risk-scaling target must be positive")
    return target


def _scaled_method_name(method: str, target: float) -> str:
    target_percent = target * 100.0
    label = f"{target_percent:g}".replace(".", "p")
    return f"{method}_scaled_{label}pct_vol"


def _diagnostic_summary(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for method, group in returns.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "months": int(len(group)),
                "leverage_mean": float(group["leverage"].mean()),
                "leverage_max": float(group["leverage"].max()),
                "long_exposure_mean": float(group["long_exposure"].mean()),
                "short_exposure_mean": float(group["short_exposure"].mean()),
                "max_abs_weight_max": float(group["max_abs_single_name_weight"].max()),
                "top10_concentration_mean": float(
                    group["top10_abs_weight_concentration"].mean()
                ),
                "turnover_mean": float(group["turnover"].mean()),
                "trading_cost_annualized": 12.0 * float(group["trading_cost"].mean()),
                "beta_norm": float(group["beta_norm"].dropna().median())
                if group["beta_norm"].notna().any()
                else np.nan,
                "aim_leverage_mean": float(group["aim_portfolio_leverage"].mean()),
                "aim_leverage_max": float(group["aim_portfolio_leverage"].max()),
                "inherited_leverage_mean": float(
                    group["inherited_position_leverage"].mean()
                ),
                "covariance_diagonal_min": float(group["covariance_diagonal_min"].min()),
                "covariance_diagonal_median": float(
                    group["covariance_diagonal_median"].median()
                ),
                "covariance_diagonal_max": float(group["covariance_diagonal_max"].max()),
                "covariance_condition_median": float(
                    group["covariance_condition_number"].median()
                ),
                "matrix_m_eigenvalue_median": _optional_median(
                    group["matrix_m_eigenvalue_median"]
                ),
                "trading_speed_median": _optional_median(
                    group["trading_speed_median"]
                ),
                "pre_scale_annualized_volatility_mean": float(
                    group["pre_scale_annualized_volatility"].mean()
                ),
                "risk_scale_factor_mean": float(group["risk_scale_factor"].mean()),
                "risk_model_used": ",".join(sorted(group["risk_model_used"].unique())),
            }
        )
    return pd.DataFrame(rows)


def _optional_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(finite.median()) if not finite.empty else np.nan


def _diagnostic_findings(
    predictions: pd.DataFrame,
    metrics_raw: pd.DataFrame,
    metrics_scaled: pd.DataFrame,
    summary: pd.DataFrame,
    selections: dict[str, dict[str, object]],
    config: ModelConfig,
) -> list[str]:
    findings: list[str] = []
    markowitz_predictions = predictions[
        predictions["method"].eq("markowitz_ml")
    ]["mu_hat"].dropna()
    if not markowitz_predictions.empty:
        level = float(markowitz_predictions.mean())
        dispersion = float(markowitz_predictions.std(ddof=1))
        markowitz_summary = summary[summary["method"].eq("markowitz_ml")]
        leverage = (
            float(markowitz_summary.iloc[0]["leverage_mean"])
            if not markowitz_summary.empty
            else np.nan
        )
        findings.append(
            "Markowitz expected returns have a common monthly level of "
            f"{level:.4%} and cross-sectional standard deviation of only "
            f"{dispersion:.4%}; inverse-covariance allocation therefore remains "
            f"highly sensitive to the risk model (mean leverage {leverage:.3f})."
        )
    selected = next(iter(selections.values()), {})
    for method in ("portfolio_ml_linear", "portfolio_ml_rf"):
        if method in selected:
            ridge = selected[method].get("ridge_lambda")
            row = summary[summary["method"].eq(method)]
            if not row.empty:
                findings.append(
                    f"{method} selected ridge lambda={ridge:g}; raw mean/max leverage "
                    f"is {row.iloc[0]['leverage_mean']:.3f}/{row.iloc[0]['leverage_max']:.3f}."
                )
    if not metrics_scaled.empty:
        min_vol = float(metrics_scaled["annualized_gross_volatility"].min())
        max_vol = float(metrics_scaled["annualized_gross_volatility"].max())
        target = float(
            config.values.get("diagnostic_risk_scaling", {}).get(
                "target_annual_volatility", 0.10
            )
        )
        calibration = (
            "the requested covariance is reasonably calibrated"
            if max_vol <= 1.5 * target
            else "the requested covariance materially understates portfolio risk"
        )
        findings.append(
            f"Risk-scaled portfolios hit the {target:.1%} ex-ante target each month, "
            f"while realized annualized gross volatility is {min_vol:.1%} to {max_vol:.1%}; "
            f"{calibration}."
        )
    return findings


def _metadata(
    config: ModelConfig,
    experiment_id: str,
    methods: list[str],
    panel: pd.DataFrame,
    risk_models_used: list[str],
    adjustment_modes: list[str],
    selections: dict[str, dict[str, object]],
    returns: pd.DataFrame,
    unit_audit: dict[str, Any],
    diagnostic_findings: list[str],
) -> dict[str, Any]:
    processed_root = config.project_root / "data" / "processed" / "us_equity_ml"
    data_files = [processed_root / "master_panel.parquet", processed_root / "cost_inputs.parquet"]
    fallback_reasons = sorted(
        {
            str(value)
            for value in returns.get("risk_fallback_reason", pd.Series(dtype=object)).dropna()
            if str(value)
        }
    )
    model_months = (
        returns.groupby("risk_model_used")["eom"].nunique().astype(int).to_dict()
        if not returns.empty
        else {}
    )
    return {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _git_commit(config.project_root),
        "official_code": {
            "repository": "https://github.com/theisij/ml-and-the-implementable-efficient-frontier",
            "commit": "13d836b5187cbfa69730dd256d8eb767e65ca21f",
            "reuse": "Thesis-equation implementation of random features, dynamic adjustment, and the closed-form Portfolio-ML objective",
        },
        "data_files": [
            {
                "path": str(path),
                "sha256": _sha256(path) if path.exists() else None,
            }
            for path in data_files
        ],
        "rows_loaded": int(len(panel)),
        "data_start": str(pd.Timestamp(panel["eom"].min()).date()),
        "data_end": str(pd.Timestamp(panel["eom"].max()).date()),
        "universe": config.values["universe"],
        "feature_set": config.values["feature_set"],
        "methods": methods,
        "risk_model_requested": config.values["risk_model"],
        "risk_models_used": risk_models_used,
        "risk_model_month_counts": model_months,
        "factor_risk_attempted": config.values["risk_model"] == "factor",
        "factor_risk_successful": "factor" in risk_models_used,
        "factor_risk_fallback_reason_count": len(fallback_reasons),
        "factor_risk_fallback_reason_examples": fallback_reasons[:5],
        "factor_covariance_input_diagonal_fallback": (
            unit_audit.get("factor_risk_inputs", {}).get(
                "factor_covariance_diagonal_fallback_share"
            )
        ),
        "matrix_m_requested": bool(config.values["portfolio_ml"]["use_matrix_m"]),
        "adjustment_modes_used": adjustment_modes,
        "scalar_rho_fallback_used": any(
            value.startswith("scalar_rho") for value in adjustment_modes
        ),
        "random_seed": int(config.values["random_seed"]),
        "protocol": config.values["protocol"],
        "baseline_gamma": float(config.values["baseline_gamma"]),
        "baseline_aum": float(config.values["baseline_aum"]),
        "portfolio_ml_linearization": (
            "B recursion uses the cross-sectional mean realized return as an "
            "exogenous wealth normalizer; realized backtests use each strategy's "
            "own portfolio wealth growth"
        ),
        "selected_hyperparameters": selections,
        "unit_audit": unit_audit,
        "diagnostic_findings": diagnostic_findings,
    }


def _write_outputs(
    config: ModelConfig,
    output_dir: Path,
    metadata: dict[str, Any],
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_raw: pd.DataFrame,
    metrics_scaled: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    selected_by_year: pd.DataFrame,
    validation: pd.DataFrame,
    warnings: list[str],
    unit_audit: dict[str, Any],
    diagnostic_summary: pd.DataFrame,
    diagnostic_findings: list[str],
    robustness: RollingRobustnessResult | None,
) -> None:
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config.values, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    weights.to_parquet(output_dir / "weights.parquet", index=False)
    returns.to_parquet(output_dir / "returns.parquet", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    metrics_raw.to_csv(output_dir / "metrics_raw.csv", index=False)
    metrics_scaled.to_csv(output_dir / "metrics_risk_scaled.csv", index=False)
    annual_metrics.to_csv(output_dir / "annual_performance.csv", index=False)
    selected_by_year.to_csv(
        output_dir / "selected_hyperparameters_by_year.csv", index=False
    )
    validation.to_parquet(output_dir / "validation_selection.parquet", index=False)
    diagnostic_summary.to_csv(output_dir / "diagnostics_summary.csv", index=False)
    if robustness is not None:
        robustness.monthly_differences.to_parquet(
            output_dir / "monthly_method_differences.parquet", index=False
        )
        robustness.difference_statistics.to_csv(
            output_dir / "method_difference_statistics.csv", index=False
        )
        robustness.advantage_by_year.to_csv(
            output_dir / "portfolio_ml_advantage_by_year.csv", index=False
        )
        robustness.advantage_concentration.to_csv(
            output_dir / "advantage_concentration.csv", index=False
        )
        robustness.crisis_period_analysis.to_csv(
            output_dir / "crisis_period_analysis.csv", index=False
        )
        robustness.risk_calibration_full.to_csv(
            output_dir / "risk_calibration_full.csv", index=False
        )
        robustness.risk_calibration_by_year.to_csv(
            output_dir / "risk_calibration_by_year.csv", index=False
        )
    (output_dir / "unit_audit.json").write_text(
        json.dumps(unit_audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (output_dir / "unit_audit.md").write_text(
        unit_audit_markdown(unit_audit), encoding="utf-8"
    )
    fallback_reasons = sorted(
        {
            str(value)
            for value in returns.get("risk_fallback_reason", pd.Series(dtype=object)).dropna()
            if str(value)
        }
    )
    diagnostics = [
        "# Experiment Diagnostics",
        "",
        "## Monthly Diagnostic Summary",
        "",
        _markdown_table(diagnostic_summary),
        "",
        "## Interpretation",
        "",
        *[f"- {value}" for value in diagnostic_findings],
        "",
        "## Risk Fallback Reasons",
        "",
        *([f"- {value}" for value in fallback_reasons] if fallback_reasons else ["- None"]),
        "",
        "## Warnings",
        "",
        f"Warnings recorded: {len(warnings)}",
        "",
        *([f"- {value}" for value in warnings] if warnings else ["- None"]),
        "",
    ]
    (output_dir / "diagnostics.md").write_text("\n".join(diagnostics), encoding="utf-8")


def _write_report(
    config: ModelConfig,
    experiment_id: str,
    metadata: dict[str, Any],
    metrics_raw: pd.DataFrame,
    metrics_scaled: pd.DataFrame,
    warnings: list[str],
    unit_audit: dict[str, Any],
    diagnostic_summary: pd.DataFrame,
    diagnostic_findings: list[str],
) -> Path:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    path = config.reports_dir / f"{experiment_id}_summary.md"
    lines = [
        f"# Model Experiment: {experiment_id}",
        "",
        f"- Data: {metadata['data_start']} to {metadata['data_end']}",
        f"- Universe / features: {metadata['universe']} / {metadata['feature_set']}",
        f"- Risk requested / used: {metadata['risk_model_requested']} / {', '.join(metadata['risk_models_used'])}",
        f"- Adjustment mode: {', '.join(metadata['adjustment_modes_used']) or 'not applicable'}",
        f"- Official code reference: commit `{metadata['official_code']['commit']}`",
        "",
        f"- Factor risk successful: {metadata['factor_risk_successful']}",
        "",
        "## Raw Metrics",
        "",
        _markdown_table(metrics_raw),
        "",
        "## Risk-Scaled Diagnostic Metrics",
        "",
        _markdown_table(metrics_scaled),
        "",
        "## Unit Audit",
        "",
        f"- Returns: {unit_audit['returns']['check']}",
        f"- Volatility/covariance: {unit_audit['volatility_and_covariance']['check']}",
        f"- ADV/transaction cost: {unit_audit['adv_and_transaction_cost']['check']}",
        f"- Utility: {unit_audit['utility']['check']}",
        f"- Factor risk inputs: {unit_audit.get('factor_risk_inputs', {}).get('check', 'not requested')}",
        "",
        "## Diagnostic Summary",
        "",
        _markdown_table(diagnostic_summary),
        "",
        "## Interpretation",
        "",
        *[f"- {value}" for value in diagnostic_findings],
        "",
        "## Top Warnings",
        "",
        *([f"- {value}" for value in warnings[:20]] if warnings else ["- None"]),
        "",
        "## Next Step",
        "",
        "Remain on top100/core10 until the upstream factor-return scale issue is repaired and the stabilized models pass rolling-year robustness checks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_rolling_summary(
    config: ModelConfig,
    experiment_id: str,
    output_dir: Path,
    full_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    selected_by_year: pd.DataFrame,
    robustness: RollingRobustnessResult | None,
) -> Path:
    unstable = set(config.values.get("unstable_diagnostic_methods", []))
    main_metrics = full_metrics[~full_metrics["method"].isin(unstable)].reset_index(drop=True)
    unstable_metrics = full_metrics[full_metrics["method"].isin(unstable)].reset_index(drop=True)
    comparisons = _rolling_method_comparison(annual_metrics)
    difference_statistics = (
        robustness.difference_statistics if robustness is not None else pd.DataFrame()
    )
    concentration = (
        robustness.advantage_concentration if robustness is not None else pd.DataFrame()
    )
    crisis = robustness.crisis_period_analysis if robustness is not None else pd.DataFrame()
    risk_full = robustness.risk_calibration_full if robustness is not None else pd.DataFrame()
    risk_annual = (
        robustness.risk_calibration_by_year if robustness is not None else pd.DataFrame()
    )
    lines = [
        "# Top100/Core10 Rolling-Year Summary",
        "",
        f"- Experiment: `{experiment_id}`",
        "- Protocol: train through Y-4, validate Y-3:Y-1, refit through Y-1, test in Y",
        f"- Test years: {int(annual_metrics['test_year'].min())}-{int(annual_metrics['test_year'].max())}",
        "- Markowitz-ML is reported only as an unstable diagnostic.",
        "",
        "## Full-Period Main Metrics",
        "",
        _markdown_table(main_metrics),
        "",
        "## Portfolio-ML Versus Static Benchmarks",
        "",
        _markdown_table(comparisons),
        "",
        "## Monthly Difference Inference",
        "",
        "Utility-flow differences use monthly net return minus gamma/2 times ex-ante monthly variance.",
        "",
        _markdown_table(difference_statistics),
        "",
        "## Advantage Concentration",
        "",
        _markdown_table(concentration),
        "",
        "## Crisis-Period Analysis",
        "",
        _markdown_table(crisis),
        "",
        "## Full-Period Risk Calibration",
        "",
        _markdown_table(risk_full),
        "",
        "## Annual Risk Calibration",
        "",
        _markdown_table(risk_annual),
        "",
        "## Annual Performance",
        "",
        _markdown_table(annual_metrics[~annual_metrics["method"].isin(unstable)]),
        "",
        "## Selected Hyperparameters By Year",
        "",
        _markdown_table(selected_by_year),
        "",
        "## Unstable Diagnostic",
        "",
        _markdown_table(unstable_metrics),
        "",
    ]
    text = "\n".join(lines)
    output_path = output_dir / "rolling_yearly_summary.md"
    output_path.write_text(text, encoding="utf-8")
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = config.reports_dir / f"{experiment_id}_rolling_yearly_summary.md"
    report_copy.write_text(text, encoding="utf-8")
    return output_path


def _rolling_method_comparison(annual_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for benchmark_method in ("static_ml", "static_ml_star"):
        benchmark = annual_metrics[
            annual_metrics["method"].eq(benchmark_method)
        ].set_index("test_year")
        for method in ("portfolio_ml_linear", "portfolio_ml_rf"):
            candidate = annual_metrics[annual_metrics["method"].eq(method)].set_index(
                "test_year"
            )
            common = candidate.index.intersection(benchmark.index)
            if common.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "benchmark": benchmark_method,
                    "years_compared": int(len(common)),
                    "net_return_win_years": int(
                        (
                            candidate.loc[common, "annualized_net_return"]
                            > benchmark.loc[common, "annualized_net_return"]
                        ).sum()
                    ),
                    "net_sharpe_win_years": int(
                        (
                            candidate.loc[common, "net_sharpe"]
                            > benchmark.loc[common, "net_sharpe"]
                        ).sum()
                    ),
                    "utility_win_years": int(
                        (
                            candidate.loc[common, "empirical_utility"]
                            > benchmark.loc[common, "empirical_utility"]
                        ).sum()
                    ),
                    "mean_net_return_difference": float(
                        (
                            candidate.loc[common, "annualized_net_return"]
                            - benchmark.loc[common, "annualized_net_return"]
                        ).mean()
                    ),
                    "mean_net_sharpe_difference": float(
                        (
                            candidate.loc[common, "net_sharpe"]
                            - benchmark.loc[common, "net_sharpe"]
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No metrics were produced."
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6g}" if pd.notna(value) else "")
    headers = [str(value) for value in display.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend([[str(value) for value in row] for row in display.itertuples(index=False, name=None)])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_experiment_id(config: ModelConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.values['experiment_name']}_{stamp}"
