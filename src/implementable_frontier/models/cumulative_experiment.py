from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from implementable_frontier.data.loaders import load_master_panel, load_risk_inputs
from implementable_frontier.models.backtest import BacktestResult, BacktestState, run_backtest
from implementable_frontier.models.cache import (
    DesignMatrixCache,
    PersistentAdjustmentCache,
    StaticPrecisionCache,
    TransactionCostCache,
)
from implementable_frontier.models.checkpoint import CheckpointStore
from implementable_frontier.models.config import ModelConfig, load_model_config
from implementable_frontier.models.metrics import annual_performance, performance_metrics
from implementable_frontier.models.portfolio_ml import (
    PortfolioMLModel,
    PortfolioMLSpec,
    portfolio_ml_specs,
)
from implementable_frontier.models.return_ml import (
    ReturnModelSpec,
    RidgeReturnModel,
    return_model_specs,
)
from implementable_frontier.models.risk import RiskModelProvider


@dataclass(frozen=True)
class Candidate:
    method: str
    parameters: dict[str, object]
    candidate_id: str


class TimeBudgetPause(RuntimeError):
    """Cooperative pause raised only after an atomic checkpoint boundary."""

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__("paused_time_budget")
        self.payload = payload


@dataclass
class RuntimeTracker:
    method_seconds: dict[str, float]
    candidate_seconds: list[dict[str, object]]
    test_year_seconds: dict[int, float]
    portfolio_fit_count: int = 0
    return_fit_count: int = 0
    linear_solve_count: int = 0

    @classmethod
    def create(cls) -> "RuntimeTracker":
        return cls({}, [], {})

    def add_method(self, method: str, seconds: float) -> None:
        self.method_seconds[method] = self.method_seconds.get(method, 0.0) + seconds


@dataclass
class CumulativeRunResult:
    mode: str
    output_dir: Path
    returns: pd.DataFrame
    weights: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    annual_performance: pd.DataFrame
    validation_selection: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    prediction_diagnostics: pd.DataFrame
    runtime: dict[str, object]
    cache_manifest: dict[str, object]
    checkpoint_manifest: dict[str, object]


class ReturnPredictionManager:
    def __init__(
        self,
        panel: pd.DataFrame,
        config: ModelConfig,
        design_cache: DesignMatrixCache,
        root: Path,
        runtime: RuntimeTracker,
    ) -> None:
        self.panel = panel.sort_values(["eom", "permno"]).reset_index(drop=True)
        self.config = config
        self.design_cache = design_cache
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.memory: dict[int, tuple[pd.DataFrame, ReturnModelSpec, pd.DataFrame]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, year: int) -> tuple[pd.DataFrame, ReturnModelSpec, pd.DataFrame]:
        if year in self.memory:
            self.hits += 1
            return self.memory[year]
        directory = self.root / str(int(year))
        if (directory / "complete.json").exists():
            predictions = pd.read_parquet(directory / "predictions.parquet")
            selection = pd.read_csv(directory / "selection.csv")
            metadata = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            spec = ReturnModelSpec(**metadata["selected_spec"])
            result = (predictions, spec, selection)
            self.memory[year] = result
            self.hits += 1
            return result
        self.misses += 1
        started = time.perf_counter()
        predictions, spec, selection = self._fit_year(year)
        directory.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(directory / "predictions.parquet", index=False)
        selection.to_csv(directory / "selection.csv", index=False)
        (directory / "complete.json").write_text(
            json.dumps(
                {
                    "year": int(year),
                    "selected_spec": asdict(spec),
                    "selection_target": "prediction_mse",
                    "lookahead_cutoff": str(pd.Timestamp(f"{year - 1}-11-30").date()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.writes += 1
        self.runtime.add_method("return_ml_prediction", time.perf_counter() - started)
        result = (predictions, spec, selection)
        self.memory[year] = result
        return result

    def info(self) -> dict[str, float | int]:
        requests = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "requests": requests,
            "hit_rate": self.hits / requests if requests else 0.0,
        }

    def _fit_year(
        self, year: int
    ) -> tuple[pd.DataFrame, ReturnModelSpec, pd.DataFrame]:
        settings = self.config.values["cumulative_oos_validation"]
        cutoff = pd.Timestamp(f"{year - 1}-11-30")
        history = self.panel[pd.to_datetime(self.panel["eom"]) <= cutoff].copy()
        months = sorted(pd.to_datetime(history["eom"]).unique())
        minimum_train = int(settings["return_min_train_months"])
        minimum_validation = int(settings["return_min_validation_months"])
        desired_validation = int(settings["return_validation_years"]) * 12
        available_validation = len(months) - minimum_train
        validation_months = min(desired_validation, available_validation)
        if validation_months < minimum_validation:
            raise ValueError(
                f"Return-ML year {year} has only {validation_months} validation months"
            )
        validation_start = pd.Timestamp(months[-validation_months])
        train = history[pd.to_datetime(history["eom"]) < validation_start].copy()
        validation = history[pd.to_datetime(history["eom"]) >= validation_start].copy()
        test = self.panel[pd.to_datetime(self.panel["eom"]).dt.year.eq(year)].copy()
        candidates: list[tuple[float, ReturnModelSpec]] = []
        rows: list[dict[str, object]] = []
        groups: dict[tuple[object, ...], list[ReturnModelSpec]] = {}
        for spec in return_model_specs(
            self.config.values["return_ml"], int(self.config.values["random_seed"])
        ):
            key = (spec.kind, spec.rf_p, spec.rf_eta, spec.seed, spec.rf_max_p)
            groups.setdefault(key, []).append(spec)
        candidate_index = 0
        for group_specs in groups.values():
            model = RidgeReturnModel(
                group_specs[0],
                self.config.feature_columns,
                design_cache=self.design_cache,
                cache_context={
                    "feature_set": self.config.values["feature_set"],
                    "universe": self.config.values["universe"],
                    "rf_version": self.config.values["cache"]["rf_version"],
                },
            ).fit(train)
            self.runtime.return_fit_count += 1
            for spec in group_specs:
                model.solve_ridge(spec.ridge_lambda)
                self.runtime.linear_solve_count += 1
                predicted = model.predict(validation)
                realized = pd.to_numeric(
                    validation["ret_exc_lead1m"], errors="coerce"
                ).to_numpy(float)
                finite = np.isfinite(realized) & np.isfinite(predicted)
                mse = float(np.mean((predicted[finite] - realized[finite]) ** 2))
                candidates.append((mse, spec))
                rows.append(
                    {
                        "prediction_year": int(year),
                        "candidate_index": candidate_index,
                        "candidate_hyperparameters": json.dumps(
                            asdict(spec), sort_keys=True
                        ),
                        "validation_mse": mse,
                        "validation_start": validation_start,
                        "validation_end": validation["eom"].max(),
                        "train_end": train["eom"].max(),
                        "selected": False,
                    }
                )
                candidate_index += 1
        best_index = int(np.argmin([value[0] for value in candidates]))
        best_mse, best_spec = candidates[best_index]
        rows[best_index]["selected"] = True
        final_model = RidgeReturnModel(
            best_spec,
            self.config.feature_columns,
            design_cache=self.design_cache,
            cache_context={
                "feature_set": self.config.values["feature_set"],
                "universe": self.config.values["universe"],
                "rf_version": self.config.values["cache"]["rf_version"],
            },
        ).fit(history)
        self.runtime.return_fit_count += 1
        predicted = final_model.predict(test)
        predictions = test[["eom", "permno", "ret_exc_lead1m"]].copy()
        predictions["mu_hat"] = predicted
        predictions["prediction_year"] = int(year)
        predictions["selected_validation_mse"] = best_mse
        return predictions, best_spec, pd.DataFrame(rows)


class CumulativeExperimentRunner:
    def __init__(
        self,
        config_path: str | Path,
        *,
        resume: bool | None = None,
        time_budget_hours: float | None = None,
    ) -> None:
        self.config = load_model_config(config_path)
        if self.config.values["protocol"] != "cumulative_oos_validation":
            raise ValueError("Cumulative runner requires cumulative_oos_validation protocol")
        self.values = self.config.values
        cumulative = self.values["cumulative_oos_validation"]
        self.panel = load_master_panel(
            start=str(cumulative["data_start"]),
            end=f"{int(cumulative['last_test_year'])}-12-31",
            universe=str(self.values["universe"]),
            feature_set=str(self.values["feature_set"]),
        ).sort_values(["eom", "permno"]).reset_index(drop=True)
        analysis_start = pd.Timestamp(cumulative["analysis_start"])
        self.risk_panel = self.panel[pd.to_datetime(self.panel["eom"]) >= analysis_start].copy()
        checkpoint_root = self.config.resolve_path(self.values["checkpoint"]["root"])
        cache_root = self.config.resolve_path(self.values["cache"]["root"])
        resume_enabled = (
            bool(self.values["checkpoint"]["resume"])
            if resume is None
            else bool(resume)
        )
        self.checkpoints = CheckpointStore(checkpoint_root, resume=resume_enabled)
        if time_budget_hours is not None and float(time_budget_hours) <= 0:
            raise ValueError("time_budget_hours must be positive when provided")
        self.time_budget_seconds = (
            float(time_budget_hours) * 3600.0
            if time_budget_hours is not None
            else None
        )
        self.run_started: float | None = None
        self.runtime = RuntimeTracker.create()
        cache_settings = self.values["cache"]
        self.design_cache = DesignMatrixCache(
            self._configured_cache_path(cache_settings, "rf_design_root", cache_root / "rf_design")
        )
        self.adjustment_cache = PersistentAdjustmentCache(
            self._configured_cache_path(cache_settings, "matrix_m_root", cache_root / "matrix_m")
        )
        self.transaction_cache = TransactionCostCache(
            self._configured_cache_path(
                cache_settings, "transaction_cost_root", cache_root / "transaction_cost"
            )
        )
        self.static_precision_cache = StaticPrecisionCache()
        risk_inputs = load_risk_inputs(
            start=str(analysis_start.date()),
            end=f"{int(cumulative['last_test_year'])}-12-31",
            universe=str(self.values["universe"]),
            root=self.config.resolve_path(self.values["risk"]["inputs_dir"]),
        )
        risk_settings = {
            **self.values["risk"],
            "persistent_cache_dir": str(
                self._configured_cache_path(
                    cache_settings, "covariance_root", cache_root / "covariance"
                )
            ),
        }
        self.risk_provider = RiskModelProvider(
            self.risk_panel,
            universe=str(self.values["universe"]),
            requested_model="factor",
            settings=risk_settings,
            risk_inputs=risk_inputs,
        )
        self.prediction_manager = ReturnPredictionManager(
            self.panel,
            self.config,
            self.design_cache,
            self._configured_cache_path(
                cache_settings,
                "return_prediction_root",
                checkpoint_root / "return_predictions",
            ),
            self.runtime,
        )
        self.candidates = self._candidates()
        self.candidate_states: dict[str, BacktestState] = {}
        self.final_states: dict[str, BacktestState] = {}
        self.validation_timings: list[dict[str, object]] = []
        self.data_load_count = 2
        self.validation_target_last_year: int | None = None

    def _configured_cache_path(
        self,
        settings: dict[str, object],
        key: str,
        default: Path,
    ) -> Path:
        value = settings.get(key)
        return self.config.resolve_path(value) if value is not None else default

    def run(
        self,
        *,
        mode: str,
        test_years_override: list[int] | None = None,
        output_dir_override: str | Path | None = None,
    ) -> CumulativeRunResult:
        if mode not in {"pilot", "full"}:
            raise ValueError("mode must be pilot or full")
        cumulative = self.values["cumulative_oos_validation"]
        if test_years_override is not None:
            test_years = sorted({int(value) for value in test_years_override})
            expected = list(range(test_years[0], test_years[-1] + 1))
            first = int(cumulative["first_test_year"])
            last = int(cumulative["last_test_year"])
            if test_years != expected or test_years[0] < first or test_years[-1] > last:
                raise ValueError("test_years_override must be a contiguous in-range segment")
        else:
            test_years = (
                [int(value) for value in cumulative["pilot_test_years"]]
                if mode == "pilot"
                else list(
                    range(
                        int(cumulative["first_test_year"]),
                        int(cumulative["last_test_year"]) + 1,
                    )
                )
            )
        max_feature_data_year = self.values.get("max_feature_data_year")
        if max_feature_data_year is not None and max(test_years) > int(max_feature_data_year):
            raise RuntimeError(
                f"Feature data are available only through {int(max_feature_data_year)}; "
                f"refusing to run test year {max(test_years)}"
            )
        started = time.perf_counter()
        self.run_started = started
        self._write_run_status(
            "running",
            mode=mode,
            test_years=test_years,
            time_budget_hours=(
                self.time_budget_seconds / 3600.0
                if self.time_budget_seconds is not None
                else None
            ),
        )
        self._ensure_validation_history(max(test_years) - 1)
        selections, selected_rows = self._select_for_test_years(test_years)
        self._restore_test_states_before(min(test_years))
        final_results = self._run_test_years(test_years, selections)
        elapsed = time.perf_counter() - started
        returns = _concat([result.returns for result in final_results])
        weights = _concat([result.weights for result in final_results])
        predictions = _concat([result.predictions for result in final_results])
        gamma = float(self.values["baseline_gamma"])
        metrics = performance_metrics(returns, gamma)
        annual = annual_performance(returns, gamma)
        prediction_diagnostics = self._prediction_diagnostics(test_years)
        cache_manifest = self.cache_manifest()
        checkpoint_manifest = self.checkpoints.info()
        runtime = self.runtime_manifest(mode, elapsed, test_years)
        output_dir = self._write_core_outputs(
            mode,
            returns,
            weights,
            predictions,
            metrics,
            annual,
            selected_rows,
            prediction_diagnostics,
            runtime,
            cache_manifest,
            checkpoint_manifest,
            output_dir_override=output_dir_override,
        )
        self.checkpoints.write_manifest(
            {"mode": mode, "cache_manifest": cache_manifest, "runtime": runtime}
        )
        self._write_run_status(
            "complete",
            mode=mode,
            test_years=test_years,
            elapsed_seconds=elapsed,
        )
        return CumulativeRunResult(
            mode,
            output_dir,
            returns,
            weights,
            predictions,
            metrics,
            annual,
            selected_rows,
            selected_rows[selected_rows["selected"]].reset_index(drop=True),
            prediction_diagnostics,
            runtime,
            cache_manifest,
            checkpoint_manifest,
        )

    def _restore_test_states_before(self, first_year: int) -> None:
        configured_first = int(
            self.values["cumulative_oos_validation"]["first_test_year"]
        )
        if int(first_year) <= configured_first:
            return
        previous_year = int(first_year) - 1
        supported = {
            "return_ml_rank",
            "markowitz_ml",
            "static_ml",
            "static_ml_star",
            "portfolio_ml_linear",
            "portfolio_ml_rf",
        }
        methods = [value for value in self.values["methods"] if value in supported]
        missing = [
            method
            for method in methods
            if not self.checkpoints.exists("test", method, previous_year)
        ]
        if missing:
            raise RuntimeError(
                f"Segment starting {first_year} requires test states from "
                f"{previous_year}; missing {missing}"
            )
        for method in methods:
            self.final_states[method] = self.checkpoints.load_state(
                "test", method, previous_year
            )

    def verify_resume(self, last_validation_year: int) -> dict[str, object]:
        writes_before = self.checkpoints.writes
        hits_before = self.checkpoints.hits
        prediction_hits_before = self.prediction_manager.hits
        first_year = int(
            self.values["cumulative_oos_validation"]["validation_oos_start_year"]
        )
        for year in range(first_year, last_validation_year + 1):
            self.prediction_manager.get(year)
            for candidates in self.candidates.values():
                for candidate in candidates:
                    if not self.checkpoints.has(
                        "validation", candidate.candidate_id, year
                    ):
                        raise RuntimeError(
                            f"Resume verification missing {candidate.candidate_id}/{year}"
                        )
        return {
            "passed": self.checkpoints.writes == writes_before
            and self.checkpoints.hits > hits_before
            and self.prediction_manager.hits > prediction_hits_before,
            "new_checkpoint_writes": self.checkpoints.writes - writes_before,
            "checkpoint_hits": self.checkpoints.hits - hits_before,
            "prediction_cache_hits": self.prediction_manager.hits
            - prediction_hits_before,
        }

    def cache_manifest(self) -> dict[str, object]:
        return {
            "covariance": self.risk_provider.cache_info(),
            "matrix_m": self.adjustment_cache.info(),
            "random_features": self.design_cache.info(),
            "transaction_cost": self.transaction_cache.info(),
            "static_precision": self.static_precision_cache.info(),
            "return_predictions": self.prediction_manager.info(),
            "validation_history": self.checkpoints.info(),
            "keys": {
                "covariance": [
                    "universe",
                    "eom",
                    "risk_model",
                    "factor_model_version",
                    "permno_order_fingerprint",
                ],
                "transaction_cost": [
                    "universe",
                    "eom",
                    "AUM",
                    "cost_model_version",
                ],
                "matrix_m": [
                    "universe",
                    "eom",
                    "risk_model",
                    "factor_model_version",
                    "AUM",
                    "gamma",
                    "cost_model_version",
                ],
                "random_features": [
                    "feature_set",
                    "universe",
                    "eom",
                    "p",
                    "eta",
                    "seed",
                    "RF_version",
                    "design_namespace",
                    "permno_order_fingerprint",
                ],
                "static_precision": [
                    "method",
                    "eom",
                    "gamma",
                    "AUM",
                    "phi_or_v_k",
                    "ridge",
                    "permno_order",
                ],
            },
        }

    def runtime_manifest(
        self, mode: str, elapsed: float, test_years: list[int]
    ) -> dict[str, object]:
        candidate_frame = pd.DataFrame(self.runtime.candidate_seconds)
        return {
            "mode": mode,
            "total_seconds": elapsed,
            "method_seconds": self.runtime.method_seconds,
            "test_year_seconds": self.runtime.test_year_seconds,
            "candidate_timing": {
                "count": int(len(candidate_frame)),
                "mean_seconds": float(candidate_frame["seconds"].mean())
                if not candidate_frame.empty
                else 0.0,
                "p50_seconds": float(candidate_frame["seconds"].median())
                if not candidate_frame.empty
                else 0.0,
                "p95_seconds": float(candidate_frame["seconds"].quantile(0.95))
                if not candidate_frame.empty
                else 0.0,
                "max_seconds": float(candidate_frame["seconds"].max())
                if not candidate_frame.empty
                else 0.0,
            },
            "candidate_year_seconds": self.runtime.candidate_seconds,
            "portfolio_fit_count": self.runtime.portfolio_fit_count,
            "return_fit_count": self.runtime.return_fit_count,
            "linear_solve_count": self.runtime.linear_solve_count,
            "data_load_count": self.data_load_count,
            "matrix_operations_count": {
                "covariance_assemblies": self.risk_provider.cache_stats.misses,
                "matrix_m_computations": self.adjustment_cache.stats.misses,
                "rf_design_computations": self.design_cache.stats.misses,
                "static_precision_inversions": self.static_precision_cache.stats.misses,
                "portfolio_model_fits": self.runtime.portfolio_fit_count,
                "linear_solves": self.runtime.linear_solve_count,
            },
            "test_years": test_years,
        }

    def _ensure_validation_history(self, last_year: int) -> None:
        self.validation_target_last_year = int(last_year)
        first_year = int(
            self.values["cumulative_oos_validation"]["validation_oos_start_year"]
        )
        for year in range(first_year, last_year + 1):
            self.static_precision_cache.clear()
            predictions, _, _ = self.prediction_manager.get(year)
            test = self._year_frame(year)
            aligned_predictions = _align_predictions(test, predictions)
            train = self._portfolio_training_frame(year)
            self._run_static_candidate_year(
                year, test, aligned_predictions, "static_ml"
            )
            self._run_static_candidate_year(
                year, test, aligned_predictions, "static_ml_star"
            )
            self._run_portfolio_candidate_year(year, train, test, "portfolio_ml_linear")
            self._run_portfolio_candidate_year(year, train, test, "portfolio_ml_rf")

    def _run_static_candidate_year(
        self,
        year: int,
        test: pd.DataFrame,
        predictions: np.ndarray,
        method: str,
    ) -> None:
        for candidate in self.candidates[method]:
            if self.checkpoints.has("validation", candidate.candidate_id, year):
                if self._needs_restored_state(candidate.candidate_id, year):
                    self.candidate_states[candidate.candidate_id] = (
                        self.checkpoints.load_state(
                            "validation", candidate.candidate_id, year
                        )
                    )
                continue
            started = time.perf_counter()
            try:
                result = run_backtest(
                    test,
                    method=method,
                    risk_provider=self.risk_provider,
                    gamma=float(self.values["baseline_gamma"]),
                    aum=float(self.values["baseline_aum"]),
                    accounting_settings=self.values["accounting"],
                    method_settings=self._method_settings(method, candidate.parameters),
                    predictions=predictions,
                    test_year=year,
                    initial_state=self.candidate_states.get(candidate.candidate_id),
                    transaction_cache=self.transaction_cache,
                    static_precision_cache=self.static_precision_cache,
                )
                self._enforce_quality(result, method)
                seconds = time.perf_counter() - started
                self.checkpoints.save(
                    "validation",
                    candidate.candidate_id,
                    year,
                    result,
                    metadata={
                        "parameters": candidate.parameters,
                        "seconds": seconds,
                    },
                )
                self.candidate_states[candidate.candidate_id] = result.final_state
                self.runtime.add_method(method, seconds)
                self.runtime.candidate_seconds.append(
                    {
                        "method": method,
                        "candidate_id": candidate.candidate_id,
                        "year": year,
                        "seconds": seconds,
                    }
                )
                self._maybe_pause_time_budget(
                    namespace="validation",
                    item_id=candidate.candidate_id,
                    year=year,
                )
            except TimeBudgetPause:
                raise
            except Exception as exc:
                self.checkpoints.record_error(
                    "validation", candidate.candidate_id, year, exc
                )
                raise

    def _run_portfolio_candidate_year(
        self,
        year: int,
        train: pd.DataFrame,
        test: pd.DataFrame,
        method: str,
    ) -> None:
        kind = "linear" if method == "portfolio_ml_linear" else "rf"
        groups: dict[tuple[object, ...], list[Candidate]] = {}
        for candidate in self.candidates[method]:
            parameters = candidate.parameters
            key = (
                parameters.get("kind"),
                parameters.get("rf_p"),
                parameters.get("rf_eta"),
                parameters.get("seed"),
            )
            groups.setdefault(key, []).append(candidate)
        for group_candidates in groups.values():
            missing = [
                candidate
                for candidate in group_candidates
                if not self.checkpoints.has("validation", candidate.candidate_id, year)
            ]
            for candidate in group_candidates:
                if candidate not in missing:
                    if self._needs_restored_state(candidate.candidate_id, year):
                        self.candidate_states[candidate.candidate_id] = (
                            self.checkpoints.load_state(
                                "validation", candidate.candidate_id, year
                            )
                        )
            if not missing:
                continue
            base_parameters = group_candidates[0].parameters
            base_spec = PortfolioMLSpec(
                kind=kind,
                ridge_lambda=float(base_parameters["ridge_lambda"]),
                rf_p=(
                    int(base_parameters["rf_p"])
                    if base_parameters.get("rf_p") is not None
                    else None
                ),
                rf_eta=(
                    float(base_parameters["rf_eta"])
                    if base_parameters.get("rf_eta") is not None
                    else None
                ),
                seed=int(base_parameters["seed"]),
            )
            fit_started = time.perf_counter()
            model = self._fit_portfolio_model(base_spec, train)
            fit_seconds = time.perf_counter() - fit_started
            self.runtime.add_method(method, fit_seconds)
            for candidate in missing:
                started = time.perf_counter()
                try:
                    model.solve_beta(float(candidate.parameters["ridge_lambda"]))
                    self.runtime.linear_solve_count += 1
                    result = run_backtest(
                        test,
                        method=method,
                        risk_provider=self.risk_provider,
                        gamma=float(self.values["baseline_gamma"]),
                        aum=float(self.values["baseline_aum"]),
                        accounting_settings=self.values["accounting"],
                        method_settings=self._method_settings(
                            method, candidate.parameters
                        ),
                        portfolio_model=model,
                        test_year=year,
                        initial_state=self.candidate_states.get(candidate.candidate_id),
                        beta_norm=float(np.linalg.norm(model.beta_)),
                        transaction_cache=self.transaction_cache,
                        static_precision_cache=self.static_precision_cache,
                    )
                    self._enforce_quality(result, method)
                    seconds = time.perf_counter() - started
                    self.checkpoints.save(
                        "validation",
                        candidate.candidate_id,
                        year,
                        result,
                        metadata={
                            "parameters": candidate.parameters,
                            "seconds": seconds,
                            "fit_group_seconds": fit_seconds,
                            "gradient_norm": float(np.linalg.norm(model.gradient())),
                            "hessian_min_eigenvalue": float(
                                np.linalg.eigvalsh(model.quadratic_hessian()).min()
                            ),
                        },
                    )
                    self.candidate_states[candidate.candidate_id] = result.final_state
                    self.runtime.add_method(method, seconds)
                    self.runtime.candidate_seconds.append(
                        {
                            "method": method,
                            "candidate_id": candidate.candidate_id,
                            "year": year,
                            "seconds": seconds,
                        }
                    )
                    self._maybe_pause_time_budget(
                        namespace="validation",
                        item_id=candidate.candidate_id,
                        year=year,
                    )
                except TimeBudgetPause:
                    raise
                except Exception as exc:
                    self.checkpoints.record_error(
                        "validation", candidate.candidate_id, year, exc
                    )
                    raise

    def _select_for_test_years(
        self, test_years: list[int]
    ) -> tuple[dict[int, dict[str, Candidate]], pd.DataFrame]:
        selections: dict[int, dict[str, Candidate]] = {}
        rows: list[dict[str, object]] = []
        first_year = int(
            self.values["cumulative_oos_validation"]["validation_oos_start_year"]
        )
        last_needed_year = max(test_years) - 1
        histories: dict[str, pd.DataFrame] = {}
        for candidates in self.candidates.values():
            for candidate in candidates:
                histories[candidate.candidate_id] = _concat(
                    [
                        self.checkpoints.load_returns(
                            "validation", candidate.candidate_id, validation_year
                        )
                        for validation_year in range(first_year, last_needed_year + 1)
                    ]
                )
        for year in test_years:
            selections[year] = {}
            for method, candidates in self.candidates.items():
                candidate_scores: list[tuple[float, Candidate, int, int]] = []
                for candidate in candidates:
                    history_years = cumulative_validation_years(
                        first_year,
                        year,
                    )
                    history = histories[candidate.candidate_id]
                    history = history[
                        pd.to_numeric(history["test_year"], errors="coerce").lt(year)
                    ]
                    score = self._selection_score(method, history)
                    candidate_scores.append(
                        (
                            score,
                            candidate,
                            int(history["eom"].nunique()),
                            len(history_years),
                        )
                    )
                best_index = int(np.argmax([value[0] for value in candidate_scores]))
                selections[year][method] = candidate_scores[best_index][1]
                for index, (score, candidate, months, years) in enumerate(candidate_scores):
                    rows.append(
                        {
                            "test_year": year,
                            "method": method,
                            "candidate_id": candidate.candidate_id,
                            "candidate_hyperparameters": json.dumps(
                                candidate.parameters, sort_keys=True
                            ),
                            "cumulative_validation_score": score,
                            "validation_months": months,
                            "validation_years": years,
                            "latest_validation_year": year - 1,
                            "selected": index == best_index,
                            "future_information_used": False,
                        }
                    )
        return selections, pd.DataFrame(rows)

    def _run_test_years(
        self,
        test_years: list[int],
        selections: dict[int, dict[str, Candidate]],
    ) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        for year in test_years:
            self.static_precision_cache.clear()
            year_started = time.perf_counter()
            predictions_frame, return_spec, _ = self.prediction_manager.get(year)
            test = self._year_frame(year)
            predictions = _align_predictions(test, predictions_frame)
            train = self._portfolio_training_frame(year)
            return_model = RidgeReturnModel(
                return_spec, self.config.feature_columns
            ).fit(self._return_training_frame(year))
            beta_norm = float(np.linalg.norm(return_model.coef_))
            supported_methods = (
                "return_ml_rank",
                "markowitz_ml",
                "static_ml",
                "static_ml_star",
                "portfolio_ml_linear",
                "portfolio_ml_rf",
            )
            for method in [
                value for value in self.values["methods"] if value in supported_methods
            ]:
                item_id = method
                if self.checkpoints.has("test", item_id, year):
                    result = self.checkpoints.load(
                        "test", item_id, year, include_artifacts=True
                    )
                    self.final_states[method] = result.final_state
                    results.append(result)
                    continue
                started = time.perf_counter()
                portfolio_model = None
                method_predictions = None
                parameters: dict[str, object]
                if method in {"return_ml_rank", "markowitz_ml"}:
                    parameters = (
                        self.values["return_ml"]
                        if method == "return_ml_rank"
                        else self.values["markowitz_ml"]
                    )
                    method_predictions = predictions
                elif method in {"static_ml", "static_ml_star"}:
                    selected = selections[year][method]
                    parameters = selected.parameters
                    method_predictions = predictions
                else:
                    selected = selections[year][method]
                    parameters = selected.parameters
                    portfolio_model = self._fit_portfolio_model(
                        PortfolioMLSpec(
                            kind=str(parameters["kind"]),
                            ridge_lambda=float(parameters["ridge_lambda"]),
                            rf_p=(
                                int(parameters["rf_p"])
                                if parameters.get("rf_p") is not None
                                else None
                            ),
                            rf_eta=(
                                float(parameters["rf_eta"])
                                if parameters.get("rf_eta") is not None
                                else None
                            ),
                            seed=int(parameters["seed"]),
                        ),
                        train,
                    )
                try:
                    result = run_backtest(
                        test,
                        method=method,
                        risk_provider=self.risk_provider,
                        gamma=float(self.values["baseline_gamma"]),
                        aum=float(self.values["baseline_aum"]),
                        accounting_settings=self.values["accounting"],
                        method_settings=self._method_settings(method, parameters),
                        predictions=method_predictions,
                        portfolio_model=portfolio_model,
                        test_year=year,
                        initial_state=self.final_states.get(method),
                        beta_norm=(
                            float(np.linalg.norm(portfolio_model.beta_))
                            if portfolio_model is not None
                            else beta_norm
                        ),
                        transaction_cache=self.transaction_cache,
                        static_precision_cache=self.static_precision_cache,
                    )
                    self._enforce_quality(result, method)
                    seconds = time.perf_counter() - started
                    self.checkpoints.save(
                        "test",
                        item_id,
                        year,
                        result,
                        metadata={
                            "parameters": parameters,
                            "seconds": seconds,
                            "return_spec": asdict(return_spec),
                        },
                        include_artifacts=True,
                    )
                    self.final_states[method] = result.final_state
                    self.runtime.add_method(method, seconds)
                    results.append(result)
                    self._maybe_pause_time_budget(
                        namespace="test",
                        item_id=item_id,
                        year=year,
                    )
                except TimeBudgetPause:
                    raise
                except Exception as exc:
                    self.checkpoints.record_error("test", item_id, year, exc)
                    raise
            self.runtime.test_year_seconds[year] = time.perf_counter() - year_started
        return results

    def _prediction_diagnostics(self, years: list[int]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for year in years:
            predictions, spec, _ = self.prediction_manager.get(year)
            monthly_ic = []
            monthly_dispersion = []
            for _, group in predictions.groupby("eom", sort=True):
                predicted_rank = group["mu_hat"].rank()
                realized_rank = group["ret_exc_lead1m"].rank()
                monthly_ic.append(float(predicted_rank.corr(realized_rank)))
                monthly_dispersion.append(float(group["mu_hat"].std(ddof=1)))
            finite = predictions[["mu_hat", "ret_exc_lead1m"]].dropna()
            rows.append(
                {
                    "test_year": year,
                    "prediction_mean": float(predictions["mu_hat"].mean()),
                    "mean_cross_sectional_prediction_std": float(
                        np.nanmean(monthly_dispersion)
                    ),
                    "mean_rank_ic": float(np.nanmean(monthly_ic)),
                    "test_mse_diagnostic": float(
                        np.mean(
                            (
                                finite["mu_hat"] - finite["ret_exc_lead1m"]
                            )
                            ** 2
                        )
                    ),
                    "selected_validation_mse": float(
                        predictions["selected_validation_mse"].iloc[0]
                    ),
                    "selected_kind": spec.kind,
                    "selected_lambda": spec.ridge_lambda,
                    "selected_p": spec.rf_p,
                    "selected_eta": spec.rf_eta,
                    "seed": spec.seed,
                }
            )
        return pd.DataFrame(rows)

    def _selection_score(self, method: str, history: pd.DataFrame) -> float:
        if history.empty:
            return float("-inf")
        if method in {"static_ml", "static_ml_star"}:
            gross = pd.to_numeric(history["gross_return"], errors="coerce").dropna()
            costs = pd.to_numeric(history["trading_cost"], errors="coerce").reindex(
                gross.index
            )
            variance = float((gross**2).mean() - gross.mean() ** 2)
            return float(
                gross.mean()
                - costs.mean()
                - 0.5 * float(self.values["baseline_gamma"]) * variance
            )
        return float(pd.to_numeric(history["utility_flow"], errors="coerce").mean())

    def _needs_restored_state(self, candidate_id: str, year: int) -> bool:
        last_year = int(self.validation_target_last_year or year)
        return year >= last_year or not self.checkpoints.exists(
            "validation", candidate_id, year + 1
        )

    def _fit_portfolio_model(
        self, spec: PortfolioMLSpec, train: pd.DataFrame
    ) -> PortfolioMLModel:
        model = PortfolioMLModel(
            spec,
            self.config.feature_columns,
            self.risk_provider,
            gamma=float(self.values["baseline_gamma"]),
            aum=float(self.values["baseline_aum"]),
            settings=self.values["portfolio_ml"],
            adjustment_cache=self.adjustment_cache,
            design_cache=self.design_cache,
            transaction_cache=self.transaction_cache,
        ).fit(train)
        self.runtime.portfolio_fit_count += 1
        return model

    def _method_settings(
        self, method: str, parameters: dict[str, object]
    ) -> dict[str, object]:
        base = {
            **self.values.get(method, {}),
            **parameters,
            **self.values["method_execution"],
            "covariance_condition_warning": 1.0e10,
            "max_abs_weight_warning": 5.0,
            "max_leverage_warning": 20.0,
        }
        return base

    def _enforce_quality(self, result: BacktestResult, method: str) -> None:
        if result.risk_models_used != {"factor"}:
            raise RuntimeError(
                f"{method} used risk models {sorted(result.risk_models_used)}; factor required"
            )
        if method.startswith("portfolio_ml"):
            if result.adjustment_modes != {"matrix_m"}:
                raise RuntimeError(
                    f"{method} adjustment modes {sorted(result.adjustment_modes)}; matrix_m required"
                )
            returns = result.returns
            tolerance = float(
                self.values["method_execution"]["matrix_eigenvalue_tolerance"]
            )
            real_min = returns["matrix_m_real_eigenvalue_min"].min()
            real_max = returns["matrix_m_real_eigenvalue_max"].max()
            imaginary = returns["matrix_m_max_imaginary_eigenvalue"].max()
            if real_min < -tolerance or real_max > 1.0 + tolerance or imaginary > tolerance:
                raise RuntimeError(
                    f"{method} matrix_m eigenvalue quality gate failed: "
                    f"min={real_min}, max={real_max}, imaginary={imaginary}"
                )

    def _maybe_pause_time_budget(
        self,
        *,
        namespace: str,
        item_id: str,
        year: int,
    ) -> None:
        if self.time_budget_seconds is None or self.run_started is None:
            return
        elapsed = time.perf_counter() - self.run_started
        if elapsed < self.time_budget_seconds:
            return
        details = {
            "paused_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "time_budget_seconds": self.time_budget_seconds,
            "last_checkpoint_boundary": {
                "namespace": namespace,
                "item_id": item_id,
                "year": int(year),
            },
            "checkpoint_manifest": self.checkpoints.info(),
        }
        payload = {"status": "paused_time_budget", **details}
        self._write_run_status("paused_time_budget", **details)
        raise TimeBudgetPause(payload)

    def _write_run_status(self, status: str, **values: object) -> None:
        payload = {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        path = self.checkpoints.root / "run_status.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        temporary.replace(path)

    def _year_frame(self, year: int) -> pd.DataFrame:
        return self.risk_panel[
            pd.to_datetime(self.risk_panel["eom"]).dt.year.eq(year)
        ].sort_values(["eom", "permno"]).reset_index(drop=True)

    def _portfolio_training_frame(self, year: int) -> pd.DataFrame:
        cutoff = pd.Timestamp(f"{year - 1}-11-30")
        return self.risk_panel[
            pd.to_datetime(self.risk_panel["eom"]) <= cutoff
        ].sort_values(["eom", "permno"]).reset_index(drop=True)

    def _return_training_frame(self, year: int) -> pd.DataFrame:
        cutoff = pd.Timestamp(f"{year - 1}-11-30")
        return self.panel[pd.to_datetime(self.panel["eom"]) <= cutoff].copy()

    def _candidates(self) -> dict[str, list[Candidate]]:
        output: dict[str, list[Candidate]] = {}
        static = []
        for phi in self.values["static_ml"]["phi_grid"]:
            for shrinkage in self.values["static_ml"]["mu_shrinkage_grid"]:
                parameters = {
                    **self.values["static_ml"],
                    "phi": float(phi),
                    "mu_shrinkage": float(shrinkage),
                }
                static.append(
                    Candidate(
                        "static_ml",
                        parameters,
                        self.checkpoints.candidate_id("static_ml", parameters),
                    )
                )
        output["static_ml"] = static
        star = []
        for u in self.values["static_ml_star"]["u_grid"]:
            for v in self.values["static_ml_star"]["v_grid"]:
                for k in self.values["static_ml_star"]["k_grid"]:
                    parameters = {
                        **self.values["static_ml_star"],
                        "u": float(u),
                        "v": float(v),
                        "official_g": float(v),
                        "k": float(k),
                    }
                    star.append(
                        Candidate(
                            "static_ml_star",
                            parameters,
                            self.checkpoints.candidate_id(
                                "static_ml_star", parameters
                            ),
                        )
                    )
        output["static_ml_star"] = star
        for method, kind in (
            ("portfolio_ml_linear", "linear"),
            ("portfolio_ml_rf", "rf"),
        ):
            candidates = []
            for spec in portfolio_ml_specs(
                self.values["portfolio_ml"],
                kind,
                int(self.values["random_seed"]),
            ):
                parameters = spec.metadata()
                candidates.append(
                    Candidate(
                        method,
                        parameters,
                        self.checkpoints.candidate_id(method, parameters),
                    )
                )
            output[method] = candidates
        return output

    def _write_core_outputs(
        self,
        mode: str,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        annual: pd.DataFrame,
        validation: pd.DataFrame,
        prediction_diagnostics: pd.DataFrame,
        runtime: dict[str, object],
        cache_manifest: dict[str, object],
        checkpoint_manifest: dict[str, object],
        *,
        output_dir_override: str | Path | None = None,
    ) -> Path:
        if output_dir_override is not None:
            output = Path(output_dir_override).resolve()
        elif mode == "full":
            output = self.config.resolve_path(self.values["outputs"]["formal_dir"])
        else:
            output = self.config.resolve_path(
                self.values["outputs"].get(
                    "pilot_dir",
                    f"results/experiments/{self.values['experiment_name']}_pilot",
                )
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "config_used.yaml").write_text(
            yaml.safe_dump(self.values, sort_keys=False), encoding="utf-8"
        )
        returns.to_parquet(output / "returns.parquet", index=False)
        weights.to_parquet(output / "weights.parquet", index=False)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics.to_csv(output / "metrics.csv", index=False)
        annual.to_csv(output / "annual_performance.csv", index=False)
        validation.to_parquet(output / "validation_selection.parquet", index=False)
        validation[validation["selected"]].to_csv(
            output / "selected_hyperparameters_by_year.csv", index=False
        )
        prediction_diagnostics.to_csv(
            output / "return_ml_prediction_diagnostics.csv", index=False
        )
        (output / "runtime_profile.json").write_text(
            json.dumps(runtime, indent=2, default=str), encoding="utf-8"
        )
        (output / "cache_manifest.json").write_text(
            json.dumps(cache_manifest, indent=2, default=str), encoding="utf-8"
        )
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(checkpoint_manifest, indent=2, default=str), encoding="utf-8"
        )
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "protocol": "cumulative_oos_validation",
                    "universe": str(self.values["universe"]),
                    "feature_set": str(self.values["feature_set"]),
                    "risk_model": str(self.values["risk_model"]),
                    "factor_fallback_count": int(
                        returns.get("risk_fallback_reason", pd.Series(dtype=object))
                        .notna()
                        .sum()
                    ),
                    "matrix_m_fallback_count": int(
                        returns["adjustment_mode"]
                        .astype(str)
                        .str.contains("fallback")
                        .sum()
                    ),
                    "test_years": [
                        int(value)
                        for value in sorted(returns["test_year"].astype(int).unique())
                    ],
                    "return_tuning": "MSE, separate from portfolio utility",
                    "lookahead_policy": "one-month labels available only through November before each annual refit",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output


def _align_predictions(test: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    left = test[["eom", "permno"]].copy()
    right = predictions[["eom", "permno", "mu_hat"]].copy()
    merged = left.merge(right, on=["eom", "permno"], how="left", validate="one_to_one")
    if merged["mu_hat"].isna().any():
        raise ValueError("Return prediction cache is missing test rows")
    return merged["mu_hat"].to_numpy(float)


def cumulative_validation_years(
    validation_oos_start_year: int, test_year: int
) -> range:
    if int(validation_oos_start_year) >= int(test_year):
        raise ValueError("Cumulative validation must start before the test year")
    return range(int(validation_oos_start_year), int(test_year))


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
