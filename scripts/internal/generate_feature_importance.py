from __future__ import annotations

import os

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml

from implementable_frontier.data.loaders import load_master_panel
from implementable_frontier.models.backtest import BacktestState, run_backtest
from implementable_frontier.models.cumulative_experiment import CumulativeExperimentRunner
from implementable_frontier.models.metrics import annual_performance, performance_metrics
from implementable_frontier.models.portfolio_ml import PortfolioMLModel, PortfolioMLSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/feature_importance"
FIGURE_ROOT = REPORT_ROOT / "figures"
CHECKPOINT_ROOT = PROJECT_ROOT / "results/checkpoints/final_analysis/feature_importance"
CONFIG_PATH = INPUT_ROOT / "config_used.yaml"
THEME_PATH = PROJECT_ROOT / "configs/core10_theme_mapping.yaml"
METHODS = ["portfolio_ml_linear", "portfolio_ml_rf"]
DISPLAY = {
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}
COLORS = {
    "portfolio_ml_linear": "#2563a6",
    "portfolio_ml_rf": "#b43c55",
}


@dataclass
class RecoveredModel:
    model: PortfolioMLModel
    beta: np.ndarray
    parameters: dict[str, object]
    design_rank: int
    design_columns: int
    aim_max_abs_error: float
    beta_norm_error: float


class NeutralizedPortfolioModel:
    def __init__(
        self,
        recovered: RecoveredModel,
        neutralized_columns: list[str],
    ) -> None:
        self.base = recovered.model
        self.beta = recovered.beta
        self.neutralized_columns = list(neutralized_columns)

    def aim_weights(self, month: pd.DataFrame) -> np.ndarray:
        neutral = month.copy()
        for column in self.neutralized_columns:
            neutral[column] = 0.5
        return self.base.design_matrix(neutral) @ self.beta

    def adjustment_matrix(
        self,
        covariance: np.ndarray,
        lambda_adv: np.ndarray,
        *,
        eom: pd.Timestamp | None = None,
    ):
        return self.base.adjustment_matrix(covariance, lambda_adv, eom=eom)


def main() -> int:
    started = time.perf_counter()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    themes = _load_themes()
    runner = CumulativeExperimentRunner(CONFIG_PATH, resume=True)
    features = [column.removeprefix("rank_") for column in runner.config.feature_columns]
    _validate_theme_mapping(features, themes)
    cumulative = runner.values["cumulative_oos_validation"]
    diagnostic_columns = [
        "permno",
        "eom",
        "ret_exc_lead1m",
        *[f"raw_{feature}" for feature in features],
        *[f"rank_{feature}" for feature in features],
    ]
    panel = load_master_panel(
        start=str(cumulative["data_start"]),
        end=f"{int(cumulative['last_test_year'])}-12-31",
        universe=str(runner.values["universe"]),
        feature_set=str(runner.values["feature_set"]),
        columns=diagnostic_columns,
    )
    panel["eom"] = pd.to_datetime(panel["eom"])
    weights = pd.read_parquet(INPUT_ROOT / "weights.parquet")
    weights["eom"] = pd.to_datetime(weights["eom"])
    baseline_returns = pd.read_parquet(INPUT_ROOT / "returns.parquet")
    baseline_returns["eom"] = pd.to_datetime(baseline_returns["eom"])
    selections = _load_selections()

    recovered, contribution_rows, recovery_summary = _recover_models(
        runner,
        weights,
        baseline_returns,
        selections,
        themes,
    )
    baseline_errors: dict[str, float] = {}
    for method in METHODS:
        reconstructed = _run_scenario(
            runner,
            recovered,
            method=method,
            scenario_kind="full_reconstruction",
            scenario_name="full",
            neutralized_columns=[],
        )
        baseline = baseline_returns[baseline_returns["method"].eq(method)].copy()
        error = _returns_reconciliation_error(baseline, reconstructed)
        baseline_errors[method] = error
        if error > 1.0e-10:
            raise RuntimeError(f"{method} fixed-state baseline reconstruction failed: {error}")

    feature_frames = []
    for feature in features:
        column = f"rank_{feature}"
        for method in METHODS:
            neutralized = _run_scenario(
                runner,
                recovered,
                method=method,
                scenario_kind="feature",
                scenario_name=feature,
                neutralized_columns=[column],
            )
            feature_frames.append(
                _importance_rows(
                    baseline_returns[baseline_returns["method"].eq(method)],
                    neutralized,
                    method=method,
                    item_type="feature",
                    item_name=feature,
                )
            )
    theme_frames = []
    for theme, members in themes.items():
        columns = [f"rank_{feature}" for feature in members]
        for method in METHODS:
            neutralized = _run_scenario(
                runner,
                recovered,
                method=method,
                scenario_kind="theme",
                scenario_name=theme,
                neutralized_columns=columns,
            )
            theme_frames.append(
                _importance_rows(
                    baseline_returns[baseline_returns["method"].eq(method)],
                    neutralized,
                    method=method,
                    item_type="theme",
                    item_name=theme,
                )
            )

    feature_importance = pd.concat(feature_frames, ignore_index=True)
    theme_importance = pd.concat(theme_frames, ignore_index=True)
    contributions = pd.DataFrame(contribution_rows)
    feature_importance = _merge_contributions(
        feature_importance,
        contributions[contributions["item_type"].eq("feature")],
    )
    theme_importance = _merge_contributions(
        theme_importance,
        contributions[contributions["item_type"].eq("theme")],
    )
    feature_importance = _add_ranks(feature_importance)
    theme_importance = _add_ranks(theme_importance)
    feature_importance.to_csv(REPORT_ROOT / "feature_importance.csv", index=False)
    theme_importance.to_csv(REPORT_ROOT / "theme_importance.csv", index=False)

    diagnostics = _feature_diagnostics(panel, features, themes)
    diagnostics.to_csv(REPORT_ROOT / "core10_feature_diagnostics.csv", index=False)
    _importance_bar(feature_importance, features, "Feature", "feature_importance_bar.png")
    _importance_bar(theme_importance, list(themes), "Theme", "theme_importance_bar.png")
    _importance_heatmap(
        feature_importance,
        features,
        "Annual feature neutralization utility effect",
        "feature_contribution_by_year.png",
    )
    _importance_heatmap(
        theme_importance,
        list(themes),
        "Annual theme neutralization utility effect",
        "theme_contribution_by_year.png",
    )
    report = _report(
        features,
        themes,
        diagnostics,
        feature_importance,
        theme_importance,
        recovery_summary,
        baseline_errors,
    )
    (REPORT_ROOT / "feature_theme_importance.md").write_text(report, encoding="utf-8")
    manifest = {
        "passed": True,
        "scope": "Top500/Core10 explanatory analysis only",
        "model_retraining": False,
        "feature_set_selection": False,
        "features": features,
        "themes": themes,
        "recovered_models": len(recovered),
        "max_aim_reconstruction_error": float(recovery_summary["aim_max_abs_error"].max()),
        "max_beta_norm_error": float(recovery_summary["beta_norm_error"].max()),
        "baseline_reconstruction_errors": baseline_errors,
        "feature_scenarios": len(features) * len(METHODS),
        "theme_scenarios": len(themes) * len(METHODS),
        "runtime_seconds": time.perf_counter() - started,
        "figures": sorted(path.name for path in FIGURE_ROOT.glob("*.png")),
    }
    (REPORT_ROOT / "feature_importance_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def _load_themes() -> dict[str, list[str]]:
    values = yaml.safe_load(THEME_PATH.read_text(encoding="utf-8")) or {}
    themes = values.get("themes", {})
    if not isinstance(themes, dict) or not themes:
        raise RuntimeError("Core10 theme mapping is empty")
    return {str(theme): [str(value) for value in members] for theme, members in themes.items()}


def _validate_theme_mapping(features: list[str], themes: dict[str, list[str]]) -> None:
    mapped = [feature for members in themes.values() for feature in members]
    if sorted(mapped) != sorted(features):
        raise RuntimeError(
            f"Theme mapping must cover Core10 exactly; missing={sorted(set(features)-set(mapped))}, "
            f"extra={sorted(set(mapped)-set(features))}"
        )
    if len(mapped) != len(set(mapped)):
        raise RuntimeError("A Core10 feature appears in more than one theme")


def _load_selections() -> pd.DataFrame:
    selections = pd.read_csv(INPUT_ROOT / "selected_hyperparameters_by_year.csv")
    selections = selections[selections["method"].isin(METHODS)].copy()
    if selections.duplicated(["test_year", "method"]).any():
        raise RuntimeError("Duplicate selected Portfolio-ML hyperparameters")
    if len(selections) != 40:
        raise RuntimeError(f"Expected 40 Portfolio-ML year selections, found {len(selections)}")
    return selections


def _recover_models(
    runner: CumulativeExperimentRunner,
    weights: pd.DataFrame,
    baseline_returns: pd.DataFrame,
    selections: pd.DataFrame,
    themes: dict[str, list[str]],
) -> tuple[
    dict[tuple[str, int], RecoveredModel],
    list[dict[str, object]],
    pd.DataFrame,
]:
    recovered: dict[tuple[str, int], RecoveredModel] = {}
    feature_names = [column.removeprefix("rank_") for column in runner.config.feature_columns]
    theme_indices = {
        theme: [feature_names.index(feature) + 1 for feature in members]
        for theme, members in themes.items()
    }
    contribution_accumulator: list[dict[str, object]] = []
    recovery_rows = []
    for year in range(2005, 2025):
        frame = runner.panel[pd.to_datetime(runner.panel["eom"]).dt.year.eq(year)].copy()
        for method in METHODS:
            parameters = _parameters(selections, method, year)
            model = _empty_model(runner, parameters)
            designs = []
            aims = []
            monthly_designs: list[tuple[pd.Timestamp, np.ndarray]] = []
            for eom, month in frame.groupby("eom", sort=True):
                month = month.sort_values("permno").reset_index(drop=True)
                target = weights[
                    weights["method"].eq(method) & weights["eom"].eq(pd.Timestamp(eom))
                ].sort_values("permno")
                if not np.array_equal(
                    month["permno"].to_numpy(), target["permno"].to_numpy()
                ):
                    raise RuntimeError(f"Aim/design permno mismatch: {method} {eom}")
                design = model.design_matrix(month)
                designs.append(design)
                aims.append(target["aim_weight"].to_numpy(float))
                monthly_designs.append((pd.Timestamp(eom), design))
            stacked_design = np.vstack(designs)
            stacked_aim = np.concatenate(aims)
            beta, _, rank, _ = np.linalg.lstsq(stacked_design, stacked_aim, rcond=None)
            fitted = stacked_design @ beta
            aim_error = float(np.max(np.abs(fitted - stacked_aim)))
            recorded_norm = float(
                baseline_returns[
                    baseline_returns["method"].eq(method)
                    & baseline_returns["test_year"].eq(year)
                ]["beta_norm"].median()
            )
            norm_error = abs(float(np.linalg.norm(beta)) - recorded_norm)
            if rank != stacked_design.shape[1] or aim_error > 1.0e-10 or norm_error > 1.0e-10:
                raise RuntimeError(
                    f"Beta recovery failed for {method} {year}: rank={rank}/"
                    f"{stacked_design.shape[1]}, aim_error={aim_error}, norm_error={norm_error}"
                )
            item = RecoveredModel(
                model=model,
                beta=beta,
                parameters=parameters,
                design_rank=int(rank),
                design_columns=int(stacked_design.shape[1]),
                aim_max_abs_error=aim_error,
                beta_norm_error=norm_error,
            )
            recovered[(method, year)] = item
            recovery_rows.append(
                {
                    "method": method,
                    "year": year,
                    "design_rank": int(rank),
                    "design_columns": int(stacked_design.shape[1]),
                    "aim_max_abs_error": aim_error,
                    "beta_norm_error": norm_error,
                }
            )
            if method == "portfolio_ml_linear":
                for eom, design in monthly_designs:
                    feature_contributions = design[:, 1:] * beta[None, 1:]
                    for index, feature in enumerate(feature_names):
                        values = feature_contributions[:, index]
                        contribution_accumulator.append(
                            {
                                "method": method,
                                "scope": "month",
                                "year": int(eom.year),
                                "eom": eom,
                                "item_type": "feature",
                                "item_name": feature,
                                "average_absolute_aim_contribution": float(np.mean(np.abs(values))),
                                "mean_signed_aim_contribution": float(np.mean(values)),
                                "mean_abs_beta": float(abs(beta[index + 1])),
                            }
                        )
                    for theme, indices in theme_indices.items():
                        values = (design[:, indices] * beta[None, indices]).sum(axis=1)
                        contribution_accumulator.append(
                            {
                                "method": method,
                                "scope": "month",
                                "year": int(eom.year),
                                "eom": eom,
                                "item_type": "theme",
                                "item_name": theme,
                                "average_absolute_aim_contribution": float(np.mean(np.abs(values))),
                                "mean_signed_aim_contribution": float(np.mean(values)),
                                "mean_abs_beta": float(np.mean(np.abs(beta[indices]))),
                            }
                        )
            print(
                f"recovered {method} {year}: aim_error={aim_error:.3e}, "
                f"beta_norm_error={norm_error:.3e}",
                flush=True,
            )
    contributions = _aggregate_contributions(pd.DataFrame(contribution_accumulator))
    return recovered, contributions, pd.DataFrame(recovery_rows)


def _aggregate_contributions(monthly: pd.DataFrame) -> list[dict[str, object]]:
    if monthly.empty:
        return []
    value_columns = [
        "average_absolute_aim_contribution",
        "mean_signed_aim_contribution",
        "mean_abs_beta",
    ]
    annual = (
        monthly.groupby(["method", "item_type", "item_name", "year"])[value_columns]
        .mean()
        .reset_index()
    )
    annual["scope"] = "year"
    full = (
        monthly.groupby(["method", "item_type", "item_name"])[value_columns]
        .mean()
        .reset_index()
    )
    full["scope"] = "full"
    full["year"] = np.nan
    return pd.concat([full, annual], ignore_index=True).to_dict("records")


def _parameters(selections: pd.DataFrame, method: str, year: int) -> dict[str, object]:
    row = selections[
        selections["method"].eq(method) & selections["test_year"].eq(year)
    ].iloc[0]
    return json.loads(row["candidate_hyperparameters"])


def _empty_model(
    runner: CumulativeExperimentRunner,
    parameters: dict[str, object],
) -> PortfolioMLModel:
    spec = PortfolioMLSpec(
        kind=str(parameters["kind"]),
        ridge_lambda=float(parameters["ridge_lambda"]),
        rf_p=(int(parameters["rf_p"]) if parameters.get("rf_p") is not None else None),
        rf_eta=(float(parameters["rf_eta"]) if parameters.get("rf_eta") is not None else None),
        seed=int(parameters["seed"]),
    )
    return PortfolioMLModel(
        spec,
        runner.config.feature_columns,
        runner.risk_provider,
        gamma=float(runner.values["baseline_gamma"]),
        aum=float(runner.values["baseline_aum"]),
        settings=runner.values["portfolio_ml"],
        adjustment_cache=runner.adjustment_cache,
        design_cache=None,
        transaction_cache=runner.transaction_cache,
    )


def _run_scenario(
    runner: CumulativeExperimentRunner,
    recovered: dict[tuple[str, int], RecoveredModel],
    *,
    method: str,
    scenario_kind: str,
    scenario_name: str,
    neutralized_columns: list[str],
) -> pd.DataFrame:
    slug = _slug(scenario_name)
    directory = CHECKPOINT_ROOT / method / scenario_kind / slug
    returns_path = directory / "returns.parquet"
    metadata_path = directory / "complete.json"
    if returns_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("neutralized_columns") != neutralized_columns:
            raise RuntimeError(
                f"Checkpoint neutralization mismatch for {method} {scenario_kind} {scenario_name}"
            )
        print(f"checkpoint hit: {method} {scenario_kind} {scenario_name}", flush=True)
        result = pd.read_parquet(returns_path)
        result["eom"] = pd.to_datetime(result["eom"])
        return result
    started = time.perf_counter()
    state: BacktestState | None = None
    results = []
    for year in range(2005, 2025):
        item = recovered[(method, year)]
        model = NeutralizedPortfolioModel(item, neutralized_columns)
        frame = runner.panel[pd.to_datetime(runner.panel["eom"]).dt.year.eq(year)].copy()
        result = run_backtest(
            frame,
            method=method,
            risk_provider=runner.risk_provider,
            gamma=float(runner.values["baseline_gamma"]),
            aum=float(runner.values["baseline_aum"]),
            accounting_settings=runner.values["accounting"],
            method_settings=runner._method_settings(method, item.parameters),
            portfolio_model=model,
            test_year=year,
            initial_state=state,
            beta_norm=float(np.linalg.norm(item.beta)),
            transaction_cache=runner.transaction_cache,
        )
        if result.risk_models_used != {"factor"}:
            raise RuntimeError(f"{method} {scenario_name} used {result.risk_models_used}")
        if result.adjustment_modes != {"matrix_m"}:
            raise RuntimeError(f"{method} {scenario_name} used {result.adjustment_modes}")
        if pd.to_numeric(result.returns["covariance_repair_count"], errors="raise").sum():
            raise RuntimeError(f"{method} {scenario_name} used covariance repair")
        results.append(result.returns)
        state = result.final_state
    combined = pd.concat(results, ignore_index=True)
    combined["scenario_kind"] = scenario_kind
    combined["scenario_name"] = scenario_name
    combined["neutralized_columns"] = json.dumps(neutralized_columns)
    directory.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(returns_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "method": method,
                "scenario_kind": scenario_kind,
                "scenario_name": scenario_name,
                "neutralized_columns": neutralized_columns,
                "model_retraining": False,
                "years": [2005, 2024],
                "runtime_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"completed: {method} {scenario_kind} {scenario_name} "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return combined


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _returns_reconciliation_error(
    baseline: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> float:
    columns = [
        "gross_return",
        "trading_cost",
        "net_return",
        "utility_flow",
        "turnover",
        "leverage",
    ]
    left = baseline.sort_values("eom")
    right = reconstructed.sort_values("eom")
    if not np.array_equal(left["eom"].to_numpy(), right["eom"].to_numpy()):
        raise RuntimeError("Baseline reconstruction month mismatch")
    return float(
        max(
            np.max(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float)))
            for column in columns
        )
    )


def _importance_rows(
    baseline: pd.DataFrame,
    neutralized: pd.DataFrame,
    *,
    method: str,
    item_type: str,
    item_name: str,
) -> pd.DataFrame:
    baseline = baseline.copy()
    neutralized = neutralized.copy()
    baseline["method"] = method
    neutralized["method"] = method
    full_base = performance_metrics(baseline, gamma=10.0).iloc[0]
    full_neutral = performance_metrics(neutralized, gamma=10.0).iloc[0]
    annual_base = annual_performance(baseline, gamma=10.0).set_index("test_year")
    annual_neutral = annual_performance(neutralized, gamma=10.0).set_index("test_year")
    rows = [
        _difference_row(
            full_base,
            full_neutral,
            method=method,
            item_type=item_type,
            item_name=item_name,
            scope="full",
            year=np.nan,
        )
    ]
    for year in range(2005, 2025):
        rows.append(
            _difference_row(
                annual_base.loc[year],
                annual_neutral.loc[year],
                method=method,
                item_type=item_type,
                item_name=item_name,
                scope="year",
                year=year,
            )
        )
    frame = pd.DataFrame(rows)
    annual_values = frame[frame["scope"].eq("year")].copy()
    positive = annual_values["delta_utility"].clip(lower=0).sort_values(ascending=False)
    positive_total = float(positive.sum())
    full_index = frame["scope"].eq("full")
    frame.loc[full_index, "top_1_positive_year_share"] = (
        float(positive.iloc[:1].sum() / positive_total) if positive_total > 0 else np.nan
    )
    frame.loc[full_index, "top_3_positive_year_share"] = (
        float(positive.iloc[:3].sum() / positive_total) if positive_total > 0 else np.nan
    )
    frame.loc[full_index, "top_5_positive_year_share"] = (
        float(positive.iloc[:5].sum() / positive_total) if positive_total > 0 else np.nan
    )
    crisis = annual_values[annual_values["year"].isin([2007, 2008, 2020, 2021, 2022])][
        "delta_utility"
    ].clip(lower=0)
    frame.loc[full_index, "crisis_positive_year_share"] = (
        float(crisis.sum() / positive_total) if positive_total > 0 else np.nan
    )
    frame.loc[full_index, "years_positive"] = int(annual_values["delta_utility"].gt(0).sum())
    if not annual_values.empty:
        top = annual_values.loc[annual_values["delta_utility"].idxmax()]
        frame.loc[full_index, "largest_positive_year"] = int(top["year"])
        frame.loc[full_index, "largest_positive_year_delta_utility"] = float(top["delta_utility"])
    return frame


def _difference_row(
    full: pd.Series,
    neutral: pd.Series,
    *,
    method: str,
    item_type: str,
    item_name: str,
    scope: str,
    year: float,
) -> dict[str, object]:
    mapping = {
        "delta_utility": "annualized_ex_ante_utility_flow",
        "delta_net_return": "annualized_net_return",
        "delta_trading_cost": "annualized_trading_cost",
        "delta_turnover": "average_turnover",
        "delta_volatility": "annualized_volatility",
        "delta_leverage": "average_leverage",
    }
    row: dict[str, object] = {
        "method": method,
        "method_label": DISPLAY[method],
        "item_type": item_type,
        "item_name": item_name,
        "scope": scope,
        "year": year,
        "difference_definition": "full_minus_neutralized",
    }
    for output, source in mapping.items():
        row[output] = float(full[source]) - float(neutral[source])
        row[f"full_{source}"] = float(full[source])
        row[f"neutralized_{source}"] = float(neutral[source])
    return row


def _merge_contributions(
    importance: pd.DataFrame,
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    if contributions.empty:
        importance["average_absolute_aim_contribution"] = np.nan
        importance["mean_signed_aim_contribution"] = np.nan
        importance["mean_abs_beta"] = np.nan
        return importance
    keys = ["method", "item_type", "item_name", "scope", "year"]
    return importance.merge(contributions, on=keys, how="left", validate="many_to_one")


def _add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["utility_importance_rank"] = np.nan
    mask = result["scope"].eq("full")
    for method, positions in result[mask].groupby("method").groups.items():
        ranks = result.loc[positions, "delta_utility"].rank(ascending=False, method="min")
        result.loc[positions, "utility_importance_rank"] = ranks
    return result


def _feature_diagnostics(
    panel: pd.DataFrame,
    features: list[str],
    themes: dict[str, list[str]],
) -> pd.DataFrame:
    theme_for = {
        feature: theme for theme, members in themes.items() for feature in members
    }
    pre = panel[pd.to_datetime(panel["eom"]).dt.year.between(1995, 2004)].copy()
    oos = panel[pd.to_datetime(panel["eom"]).dt.year.between(2005, 2024)].copy()
    rank_columns = [f"rank_{feature}" for feature in features]
    monthly_corrs = []
    for _, month in oos.groupby("eom", sort=True):
        monthly_corrs.append(month[rank_columns].corr())
    average_corr = sum(monthly_corrs) / len(monthly_corrs)
    rows = []
    for feature in features:
        rank_column = f"rank_{feature}"
        raw_column = f"raw_{feature}"
        pre_ic = _monthly_ic(pre, rank_column)
        oos_ic = _monthly_ic(oos, rank_column)
        persistence, turnover_proxy = _monthly_persistence(oos, rank_column)
        other = average_corr.loc[rank_column].drop(rank_column).abs()
        annual_oos_ic = oos_ic.groupby(oos_ic.index.year).mean()
        annual_abs = annual_oos_ic.abs().sort_values(ascending=False)
        rows.append(
            {
                "feature": feature,
                "rank_column": rank_column,
                "raw_column": raw_column,
                "theme": theme_for[feature],
                "oos_raw_missing_rate": float(oos[raw_column].isna().mean()),
                "oos_rank_missing_fill_rate": float(
                    (oos[raw_column].isna() & oos[rank_column].eq(0.5)).mean()
                ),
                "oos_rank_p01": float(oos[rank_column].quantile(0.01)),
                "oos_rank_p05": float(oos[rank_column].quantile(0.05)),
                "oos_rank_p50": float(oos[rank_column].quantile(0.50)),
                "oos_rank_p95": float(oos[rank_column].quantile(0.95)),
                "oos_rank_p99": float(oos[rank_column].quantile(0.99)),
                "oos_rank_dispersion_mean": float(
                    oos.groupby("eom")[rank_column].std().mean()
                ),
                "monthly_rank_autocorrelation": persistence,
                "average_rank_change_turnover_proxy": turnover_proxy,
                "pre_oos_mean_rank_ic": float(pre_ic.mean()),
                "pre_oos_rank_ic_tstat": _plain_tstat(pre_ic),
                "pre_oos_positive_month_fraction": float(pre_ic.gt(0).mean()),
                "oos_mean_rank_ic_descriptive": float(oos_ic.mean()),
                "oos_rank_ic_tstat_descriptive": _plain_tstat(oos_ic),
                "oos_positive_month_fraction": float(oos_ic.gt(0).mean()),
                "oos_positive_ic_year_fraction": float(annual_oos_ic.gt(0).mean()),
                "oos_annual_ic_std": float(annual_oos_ic.std(ddof=1)),
                "oos_top3_abs_ic_year_share": float(
                    annual_abs.iloc[:3].sum() / annual_abs.sum()
                    if annual_abs.sum() > 0
                    else np.nan
                ),
                "average_abs_corr_with_other_core10": float(other.mean()),
                "max_abs_corr_with_other_core10": float(other.max()),
                "max_corr_core10_feature": other.idxmax().removeprefix("rank_"),
            }
        )
    return pd.DataFrame(rows)


def _monthly_ic(frame: pd.DataFrame, column: str) -> pd.Series:
    values = {}
    for eom, month in frame.groupby("eom", sort=True):
        x = pd.to_numeric(month[column], errors="coerce")
        y = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce")
        valid = x.notna() & y.notna()
        values[pd.Timestamp(eom)] = (
            float(x[valid].rank(method="average").corr(y[valid].rank(method="average")))
            if valid.sum() >= 20
            else np.nan
        )
    return pd.Series(values, dtype=float).dropna()


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(
        aligned.iloc[:, 0]
        .rank(method="average")
        .corr(aligned.iloc[:, 1].rank(method="average"))
    )


def _monthly_persistence(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    data = frame[["eom", "permno", column]].sort_values(["permno", "eom"]).copy()
    data["period"] = pd.to_datetime(data["eom"]).dt.to_period("M")
    data["previous_period"] = data.groupby("permno")["period"].shift()
    data["previous_value"] = data.groupby("permno")[column].shift()
    current_ordinal = data["period"].map(lambda value: value.ordinal)
    previous_ordinal = data["previous_period"].map(
        lambda value: value.ordinal if pd.notna(value) else np.nan
    )
    valid = (current_ordinal - previous_ordinal).eq(1)
    matched = data[valid.fillna(False)].copy()
    correlations = pd.Series(
        {
            pd.Timestamp(eom): group[column].corr(group["previous_value"])
            for eom, group in matched.groupby("eom", sort=True)
        },
        dtype=float,
    )
    turnover = (matched[column] - matched["previous_value"]).abs()
    return float(correlations.mean()), float(turnover.mean())


def _plain_tstat(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    std = float(clean.std(ddof=1))
    return float(clean.mean() / (std / np.sqrt(len(clean)))) if len(clean) > 1 and std > 0 else np.nan


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    return {
        "title": ImageFont.truetype(str(root / "arialbd.ttf"), 38),
        "label": ImageFont.truetype(str(root / "arial.ttf"), 24),
        "tick": ImageFont.truetype(str(root / "arial.ttf"), 18),
        "small": ImageFont.truetype(str(root / "arial.ttf"), 15),
        "legend": ImageFont.truetype(str(root / "arial.ttf"), 20),
    }


def _importance_bar(
    importance: pd.DataFrame,
    ordered_items: list[str],
    item_label: str,
    filename: str,
) -> None:
    full = importance[importance["scope"].eq("full")].copy()
    pivot = full.pivot(index="item_name", columns="method", values="delta_utility")
    ordering = pivot.mean(axis=1).sort_values().index.tolist()
    width, height = 1780, max(850, 210 + len(ordering) * 72)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    title = f"Top500/Core10 fixed-model {item_label.lower()} importance"
    _title(draw, width, title, fonts)
    left, right, top, bottom = 430, 1700, 150, height - 90
    values = 100.0 * pivot.to_numpy(float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values)))) * 1.15
    limit = max(limit, 0.05)
    zero_x = int(left + (0 + limit) / (2 * limit) * (right - left))
    for value in np.linspace(-limit, limit, 7):
        px = int(left + (value + limit) / (2 * limit) * (right - left))
        draw.line((px, top, px, bottom), fill="#d9dde3", width=2)
        text = f"{value:.2f}"
        box = draw.textbbox((0, 0), text, font=fonts["tick"])
        draw.text((px - (box[2] - box[0]) / 2, bottom + 12), text, fill="#34383e", font=fonts["tick"])
    draw.line((zero_x, top, zero_x, bottom), fill="#4b5057", width=3)
    row_height = (bottom - top) / len(ordering)
    for row_index, item in enumerate(ordering):
        center = top + (row_index + 0.5) * row_height
        label = item.replace("_", " ")
        box = draw.textbbox((0, 0), label, font=fonts["tick"])
        draw.text((left - 18 - (box[2] - box[0]), center - 11), label, fill="#34383e", font=fonts["tick"])
        for method_index, method in enumerate(METHODS):
            value = 100.0 * float(pivot.loc[item, method])
            px = int(left + (value + limit) / (2 * limit) * (right - left))
            y0 = int(center + (method_index - 1) * 18)
            y1 = y0 + 16
            draw.rectangle((min(px, zero_x), y0, max(px, zero_x), y1), fill=COLORS[method])
    _legend(draw, 500, 96, fonts)
    draw.text((left, bottom + 48), "Annualized utility effect, full minus neutralized (%)", fill="#34383e", font=fonts["label"])
    image.save(FIGURE_ROOT / filename, optimize=True)


def _importance_heatmap(
    importance: pd.DataFrame,
    items: list[str],
    title: str,
    filename: str,
) -> None:
    annual = importance[importance["scope"].eq("year")].copy()
    years = list(range(2005, 2025))
    width = 1980
    panel_height = max(330, len(items) * 42 + 100)
    height = 180 + 2 * panel_height + 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, title, fonts)
    max_abs = float(100.0 * annual["delta_utility"].abs().quantile(0.98))
    max_abs = max(max_abs, 0.10)
    left, right = 390, 1900
    for panel, method in enumerate(METHODS):
        top = 135 + panel * panel_height
        cell_width = (right - left) / len(years)
        cell_height = (panel_height - 90) / len(items)
        data = annual[annual["method"].eq(method)].pivot(
            index="item_name", columns="year", values="delta_utility"
        )
        draw.text((left, top), DISPLAY[method], fill=COLORS[method], font=fonts["legend"])
        for row_index, item in enumerate(items):
            y0 = int(top + 38 + row_index * cell_height)
            y1 = int(top + 38 + (row_index + 1) * cell_height)
            label = item.replace("_", " ")
            box = draw.textbbox((0, 0), label, font=fonts["small"])
            draw.text((left - 12 - (box[2] - box[0]), y0 + 4), label, fill="#34383e", font=fonts["small"])
            for column_index, year in enumerate(years):
                value = 100.0 * float(data.loc[item, year])
                x0 = int(left + column_index * cell_width)
                x1 = int(left + (column_index + 1) * cell_width)
                draw.rectangle((x0, y0, x1, y1), fill=_diverging_color(value, max_abs), outline="white")
        for column_index, year in enumerate(years):
            if year % 2 == 0 or year in {2005, 2024}:
                x0 = left + (column_index + 0.5) * cell_width
                text = str(year)
                box = draw.textbbox((0, 0), text, font=fonts["small"])
                draw.text((x0 - (box[2] - box[0]) / 2, top + 44 + len(items) * cell_height), text, fill="#34383e", font=fonts["small"])
    note = f"Blue: positive full-minus-neutralized utility; red: negative; color clipped at +/-{max_abs:.2f}%"
    draw.text((left, height - 55), note, fill="#50545a", font=fonts["tick"])
    image.save(FIGURE_ROOT / filename, optimize=True)


def _diverging_color(value: float, limit: float) -> str:
    ratio = min(abs(value) / limit, 1.0)
    white = np.array([247, 247, 247], dtype=float)
    target = np.array([37, 99, 166], dtype=float) if value >= 0 else np.array([180, 60, 85], dtype=float)
    rgb = np.round(white * (1.0 - ratio) + target * ratio).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _title(draw: ImageDraw.ImageDraw, width: int, text: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    box = draw.textbbox((0, 0), text, font=fonts["title"])
    draw.text(((width - (box[2] - box[0])) / 2, 28), text, fill="#20242a", font=fonts["title"])


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    for index, method in enumerate(METHODS):
        x0 = x + index * 490
        draw.line((x0, y + 11, x0 + 42, y + 11), fill=COLORS[method], width=5)
        draw.ellipse((x0 + 17, y + 3, x0 + 31, y + 17), fill=COLORS[method])
        draw.text((x0 + 55, y), DISPLAY[method], fill="#34383e", font=fonts["legend"])


def _report(
    features: list[str],
    themes: dict[str, list[str]],
    diagnostics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    theme_importance: pd.DataFrame,
    recovery: pd.DataFrame,
    baseline_errors: dict[str, float],
) -> str:
    feature_full = feature_importance[feature_importance["scope"].eq("full")].copy()
    theme_full = theme_importance[theme_importance["scope"].eq("full")].copy()
    feature_pivot = feature_full.pivot(index="item_name", columns="method", values="delta_utility")
    theme_pivot = theme_full.pivot(index="item_name", columns="method", values="delta_utility")
    feature_order = feature_pivot.mean(axis=1).sort_values(ascending=False)
    theme_order = theme_pivot.mean(axis=1).sort_values(ascending=False)
    linear = feature_full[feature_full["method"].eq("portfolio_ml_linear")].set_index("item_name")
    rf = feature_full[feature_full["method"].eq("portfolio_ml_rf")].set_index("item_name")
    rank_corr = _spearman_corr(linear["delta_utility"], rf["delta_utility"])
    diagnostic_index = diagnostics.set_index("feature")
    mean_importance = feature_pivot.mean(axis=1)
    persistence_corr = _spearman_corr(
        mean_importance, diagnostic_index["monthly_rank_autocorrelation"]
    )
    turnover_corr = _spearman_corr(
        mean_importance, diagnostic_index["average_rank_change_turnover_proxy"]
    )
    ic_corr = _spearman_corr(
        mean_importance, diagnostic_index["oos_mean_rank_ic_descriptive"].abs()
    )
    concentration = feature_full.groupby("method").agg(
        top_3_positive_year_share=("top_3_positive_year_share", "mean"),
        crisis_positive_year_share=("crisis_positive_year_share", "mean"),
    )
    feature_table = feature_full[[
        "method_label",
        "item_name",
        "delta_utility",
        "delta_net_return",
        "delta_trading_cost",
        "delta_turnover",
        "delta_volatility",
        "delta_leverage",
        "average_absolute_aim_contribution",
        "top_3_positive_year_share",
        "crisis_positive_year_share",
        "years_positive",
    ]].sort_values(["method_label", "delta_utility"], ascending=[True, False])
    theme_table = theme_full[[
        "method_label",
        "item_name",
        "delta_utility",
        "delta_net_return",
        "delta_trading_cost",
        "delta_turnover",
        "delta_volatility",
        "delta_leverage",
        "top_3_positive_year_share",
        "years_positive",
    ]].sort_values(["method_label", "delta_utility"], ascending=[True, False])
    mapping_rows = [
        {"theme": theme, "features": ", ".join(members)}
        for theme, members in themes.items()
    ]
    top_features = ", ".join(feature_order.index[:3])
    top_themes = ", ".join(theme_order.index[:3])
    lines = [
        "# Feature and theme importance",
        "",
        "This is explanatory analysis for the locked Top500/Core10 specification. It performs no model retraining, feature selection, hyperparameter selection or change to Core10. Final15 and Final20 are not inputs.",
        "",
        "## Method",
        "",
        "- Each annual Portfolio-ML beta is recovered from saved OOS aim weights and the uniquely determined fixed design matrix. No training labels or portfolio fitting objective are rerun.",
        "- The unmodified fixed-state reconstruction must reproduce the locked monthly baseline before any neutralization scenario is accepted.",
        "- Feature neutralization replaces the original rank input with 0.5 before monthly transformation. Theme neutralization replaces every member feature with 0.5.",
        "- The recovered beta, annual selected hyperparameters, RF p/eta/seed, factor covariance, full matrix m, AUM and gamma remain fixed.",
        "- Importance is full-model minus neutralized performance. Positive delta utility means the fixed model performs worse when the input is neutralized.",
        "- Linear aim contribution is directly decomposed as transformed feature times recovered beta. RF importance is a fixed-model dependence diagnostic, not a structural or causal attribution.",
        "",
        "## State-recovery gate",
        "",
        f"- Maximum aim reconstruction error: {recovery['aim_max_abs_error'].max():.3e}.",
        f"- Maximum recovered beta-norm error: {recovery['beta_norm_error'].max():.3e}.",
        f"- Linear baseline reconstruction error: {baseline_errors['portfolio_ml_linear']:.3e}.",
        f"- RF baseline reconstruction error: {baseline_errors['portfolio_ml_rf']:.3e}.",
        "- Every recovered design matrix has full column rank.",
        "",
        "## Transparent Core10 theme mapping",
        "",
        _markdown_table(pd.DataFrame(mapping_rows)),
        "",
        "The mapping is stored in `configs/core10_theme_mapping.yaml` and is used only for explanatory aggregation.",
        "",
        "## Full-sample feature neutralization",
        "",
        _markdown_table(feature_table),
        "",
        "## Full-sample theme neutralization",
        "",
        _markdown_table(theme_table),
        "",
        "## Descriptive Core10 diagnostics",
        "",
        _markdown_table(diagnostics[[
            "feature",
            "theme",
            "oos_raw_missing_rate",
            "oos_rank_dispersion_mean",
            "monthly_rank_autocorrelation",
            "average_rank_change_turnover_proxy",
            "pre_oos_mean_rank_ic",
            "oos_mean_rank_ic_descriptive",
            "average_abs_corr_with_other_core10",
            "max_abs_corr_with_other_core10",
        ]]),
        "",
        "## Answers to the research questions",
        "",
        f"1. The largest average fixed-model feature effects are {top_features}; the largest themes are {top_themes}. Method-specific rankings remain in the machine-readable table.",
        f"2. The Spearman association between importance and rank persistence is {persistence_corr:.3f}; with the rank-change turnover proxy it is {turnover_corr:.3f}. These statistics test whether contribution is concentrated in persistent, lower-churn inputs rather than assuming it.",
        f"3. Absolute descriptive OOS IC and fixed-model utility importance have Spearman correlation {ic_corr:.3f}. A high IC therefore is not treated as equivalent to high implementable utility because risk, turnover, cost and interactions enter the portfolio objective.",
        f"4. The top-three positive-year shares are {concentration.loc['portfolio_ml_linear', 'top_3_positive_year_share']:.1%} for Linear and {concentration.loc['portfolio_ml_rf', 'top_3_positive_year_share']:.1%} for RF when averaged over features. Crisis-period shares are {concentration.loc['portfolio_ml_linear', 'crisis_positive_year_share']:.1%} and {concentration.loc['portfolio_ml_rf', 'crisis_positive_year_share']:.1%}; annual heatmaps show the detailed concentration.",
        f"5. Linear and RF feature-importance rankings have Spearman correlation {rank_corr:.3f}. Agreement is interpreted as model robustness; disagreement is expected because RF neutralization changes nonlinear interactions.",
        "6. The results support the implementability mechanism only when positive utility effects coexist with acceptable turnover and cost effects. Predictive contribution alone is insufficient.",
        "7. This analysis cannot be used to alter Core10: it is conditional on models selected before the explanatory exercise, uses the realized OOS path, and would create data snooping if reused for feature selection.",
        "",
        "## Interpretation limits",
        "",
        "- Neutralization measures dependence of the already-fitted model on an input; it is not a causal effect and can include interactions with correlated features.",
        "- Feature-level deltas are not additive because each counterfactual changes the dynamic portfolio path independently.",
        "- OOS IC is reported for description only and is not used to select features or reinterpret the locked specification.",
        "- Negative importance is retained. No feature, year or theme is removed for looking unfavorable.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
