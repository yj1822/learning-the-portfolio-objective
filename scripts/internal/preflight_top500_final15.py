from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.data.config import load_data_config  # noqa: E402
from implementable_frontier.data.loaders import (  # noqa: E402
    load_master_panel,
    load_risk_inputs,
)
from implementable_frontier.models.backtest import run_backtest  # noqa: E402
from implementable_frontier.models.cache import (  # noqa: E402
    DesignMatrixCache,
    PersistentAdjustmentCache,
    TransactionCostCache,
)
from implementable_frontier.models.config import load_model_config  # noqa: E402
from implementable_frontier.models.features import (  # noqa: E402
    RandomFourierFeatures,
    monthly_cross_sectional_standardize,
    volatility_scale_design,
)
from implementable_frontier.models.portfolio_ml import (  # noqa: E402
    PortfolioMLModel,
    PortfolioMLSpec,
)
from implementable_frontier.models.risk import RiskModelProvider  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/model_top500_final15_cumulative.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results/experiments/top500_final15_factor_cumulative"
REPORT_DIR = PROJECT_ROOT / "reports/model_results/top500_final15_cumulative"
JSON_PATH = OUTPUT_DIR / "preflight_diagnostics.json"
MD_PATH = REPORT_DIR / "preflight_report.md"
REPRESENTATIVE_MONTHS = (
    pd.Timestamp("2005-01-31"),
    pd.Timestamp("2008-09-30"),
    pd.Timestamp("2020-03-31"),
    pd.Timestamp("2022-06-30"),
    pd.Timestamp("2024-12-31"),
)


def main() -> int:
    global CONFIG_PATH, OUTPUT_DIR, REPORT_DIR, JSON_PATH, MD_PATH, REPRESENTATIVE_MONTHS
    parser = argparse.ArgumentParser(description="Run feature-set preflight diagnostics")
    parser.add_argument("--config", default=str(CONFIG_PATH.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--report-dir", default=str(REPORT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--start", default="1995-01-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--representative-months", nargs="*")
    args = parser.parse_args()
    CONFIG_PATH = (PROJECT_ROOT / args.config).resolve()
    OUTPUT_DIR = (PROJECT_ROOT / args.output_dir).resolve()
    REPORT_DIR = (PROJECT_ROOT / args.report_dir).resolve()
    JSON_PATH = OUTPUT_DIR / "preflight_diagnostics.json"
    MD_PATH = REPORT_DIR / "preflight_report.md"
    if args.representative_months:
        REPRESENTATIVE_MONTHS = tuple(pd.Timestamp(value) for value in args.representative_months)

    started = time.perf_counter()
    config = load_model_config(CONFIG_PATH)
    values = config.values
    if values.get("data_master_panel_path"):
        os.environ["IEF_MASTER_PANEL_PATH"] = str(
            config.resolve_path(values["data_master_panel_path"])
        )
    data_config = load_data_config(config.resolve_path(values["data_config"]))
    checks: list[dict[str, object]] = []

    feature_set = str(values["feature_set"])
    feature_diagnostics = _feature_diagnostics(
        data_config, feature_set, args.start, args.end, checks
    )
    panel = load_master_panel(
        start=args.start,
        end=args.end,
        universe="top500",
        feature_set=feature_set,
    )
    risk_inputs = load_risk_inputs(
        start=args.start,
        end=args.end,
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
        panel,
        universe="top500",
        requested_model="factor",
        settings=risk_settings,
        risk_inputs=risk_inputs,
    )
    cache_root = config.resolve_path(values["cache"]["root"])
    adjustment_cache = PersistentAdjustmentCache(cache_root / "matrix_m")
    transaction_cache = TransactionCostCache(cache_root / "transaction_cost")
    design_cache = DesignMatrixCache(cache_root / "rf_design")

    rf_diagnostics = _rf_diagnostics(panel, provider, values, checks)
    risk_diagnostics = _risk_cost_m_diagnostics(
        panel,
        provider,
        values,
        adjustment_cache,
        design_cache,
        transaction_cache,
        checks,
    )
    closed_form = _closed_form_diagnostics(
        panel,
        provider,
        values,
        adjustment_cache,
        design_cache,
        transaction_cache,
        checks,
    )
    passed = all(bool(check["passed"]) for check in checks)
    payload = {
        "passed": passed,
        "config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_set": feature_set,
        "sample_start": args.start,
        "sample_end": args.end,
        "feature_diagnostics": feature_diagnostics,
        "rf_diagnostics": rf_diagnostics,
        "risk_cost_m_diagnostics": risk_diagnostics,
        "closed_form_diagnostics": closed_form,
        "checks": checks,
        "failed_checks": [check for check in checks if not bool(check["passed"])],
        "cache_info": {
            "covariance": provider.cache_info(),
            "matrix_m": adjustment_cache.info(),
            "random_features": design_cache.info(),
            "transaction_cost": transaction_cache.info(),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    MD_PATH.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"passed={passed}")
    print(f"json={JSON_PATH}")
    print(f"report={MD_PATH}")
    return 0 if passed else 1


def _feature_diagnostics(
    data_config,
    feature_set: str,
    start: str,
    end: str,
    checks: list[dict[str, object]],
) -> dict[str, object]:
    path = Path(os.environ.get("IEF_MASTER_PANEL_PATH", data_config.master_panel_path))
    features = list(data_config.feature_sets[feature_set])
    columns = [
        "eom",
        "permno",
        "in_top500",
        *[f"raw_{name}" for name in features],
        *[f"rank_{name}" for name in features],
    ]
    available = set(pq.read_schema(path).names)
    missing_columns = sorted(set(columns) - available)
    _check(checks, f"{feature_set}_columns_present", not missing_columns, missing_columns)
    if missing_columns:
        return {"missing_columns": missing_columns}
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["in_top500"].fillna(False)].copy()
    frame["eom"] = pd.to_datetime(frame["eom"])
    frame = frame[frame["eom"].between(start, end)]
    raw_missing: dict[str, float] = {}
    rank_ranges: dict[str, dict[str, float | int]] = {}
    fill_failures: dict[str, int] = {}
    for name in features:
        raw = pd.to_numeric(frame[f"raw_{name}"], errors="coerce")
        rank = pd.to_numeric(frame[f"rank_{name}"], errors="coerce")
        missing = raw.isna()
        raw_missing[name] = float(missing.mean())
        fill_failures[name] = int((rank[missing] - 0.5).abs().gt(1.0e-12).sum())
        rank_ranges[name] = {
            "min": float(rank.min()),
            "max": float(rank.max()),
            "p01": float(rank.quantile(0.01)),
            "p05": float(rank.quantile(0.05)),
            "p50": float(rank.quantile(0.50)),
            "p95": float(rank.quantile(0.95)),
            "p99": float(rank.quantile(0.99)),
            "cross_sectional_std_median": float(
                frame.assign(_rank=rank).groupby("eom")["_rank"].std().median()
            ),
            "missing": int(rank.isna().sum()),
            "out_of_range": int((~rank.between(0.0, 1.0)).sum()),
        }
    subset = set(data_config.core10).issubset(features)
    _check(checks, f"core10_subset_{feature_set}", subset, sorted(set(data_config.core10) - set(features)))
    _check(
        checks,
        "raw_feature_missing_rate_below_50pct",
        max(raw_missing.values(), default=0.0) <= 0.50,
        raw_missing,
    )
    _check(
        checks,
        "rank_features_in_zero_one",
        all(value["missing"] == 0 and value["out_of_range"] == 0 for value in rank_ranges.values()),
        rank_ranges,
    )
    _check(
        checks,
        "raw_missing_filled_with_0_5",
        sum(fill_failures.values()) == 0,
        fill_failures,
    )
    return {
        "rows": int(len(frame)),
        "months": int(pd.to_datetime(frame["eom"]).nunique()),
        "raw_feature_missing_rate": raw_missing,
        "rank_feature_ranges": rank_ranges,
        "missing_fill_failures": fill_failures,
        "core10": list(data_config.core10),
        "feature_set": feature_set,
        "features": features,
        "core10_is_subset": subset,
    }


def _rf_diagnostics(
    panel: pd.DataFrame,
    provider: RiskModelProvider,
    values: dict[str, object],
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    first_pilot_year = int(values["cumulative_oos_validation"]["pilot_test_years"][0])
    month = panel[pd.to_datetime(panel["eom"]).eq(pd.Timestamp(f"{first_pilot_year}-01-31"))].sort_values("permno")
    data_config = load_data_config(CONFIG_PATH.parent / "data_us_equity_ml.yaml")
    features = [f"rank_{name}" for name in data_config.feature_sets[str(values["feature_set"])]]
    raw = month[features].to_numpy(float)
    volatility = provider.get_volatility(month["eom"].iloc[0], month["permno"].to_numpy(int))
    diagnostics: list[dict[str, object]] = []
    for p in values["portfolio_ml"]["rf_p_grid"]:
        for eta in values["portfolio_ml"]["rf_eta_grid"]:
            rff = RandomFourierFeatures(
                len(features),
                int(p),
                float(eta),
                int(values["random_seed"]),
                max_features=max(int(item) for item in values["portfolio_ml"]["rf_p_grid"]),
            )
            transformed = rff.transform(raw)
            standardized = monthly_cross_sectional_standardize(transformed, add_constant=True)
            scaled, missing_volatility = volatility_scale_design(
                standardized,
                volatility,
                floor=float(values["portfolio_ml"]["volatility_floor"]),
            )
            singular = np.linalg.svd(scaled, compute_uv=False)
            tolerance = max(float(singular.max()) * 1.0e-10, 1.0e-12)
            effective_rank = int((singular > tolerance).sum())
            condition = float(singular.max() / max(singular.min(), 1.0e-16))
            norms = np.linalg.norm(standardized, axis=0)
            means = standardized.mean(axis=0)
            normalized = standardized / np.maximum(norms, 1.0e-16)
            gram = np.abs(normalized.T @ normalized)
            duplicates = int(np.triu(gram > 1.0 - 1.0e-12, k=1).sum())
            zero_columns = int((norms < 1.0e-10).sum())
            passed = bool(
                np.isfinite(scaled).all()
                and effective_rank >= int(np.ceil(0.75 * scaled.shape[1]))
                and condition <= 1.0e12
                and zero_columns == 0
                and duplicates == 0
            )
            row = {
                "p": int(p),
                "eta": float(eta),
                "seed": int(values["random_seed"]),
                "rows": int(scaled.shape[0]),
                "columns": int(scaled.shape[1]),
                "condition_number": condition,
                "effective_rank": effective_rank,
                "column_mean_abs_max_before_vol_scaling": float(np.abs(means[1:]).max()),
                "column_norm_min_before_vol_scaling": float(norms.min()),
                "column_norm_median_before_vol_scaling": float(np.median(norms)),
                "column_norm_max_before_vol_scaling": float(norms.max()),
                "near_zero_columns": zero_columns,
                "duplicate_columns": duplicates,
                "missing_volatility": int(missing_volatility),
                "rf_matrix_hash": hashlib.sha256(np.ascontiguousarray(scaled).tobytes()).hexdigest(),
                "passed": passed,
            }
            diagnostics.append(row)
            _check(checks, f"rf_p{p}_eta{eta}", passed, row)
    return diagnostics


def _risk_cost_m_diagnostics(
    panel: pd.DataFrame,
    provider: RiskModelProvider,
    values: dict[str, object],
    adjustment_cache: PersistentAdjustmentCache,
    design_cache: DesignMatrixCache,
    transaction_cache: TransactionCostCache,
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    model = _portfolio_model(
        PortfolioMLSpec("linear", 1000.0, seed=int(values["random_seed"])),
        provider,
        values,
        adjustment_cache,
        design_cache,
        transaction_cache,
    )
    rows: list[dict[str, object]] = []
    for eom in REPRESENTATIVE_MONTHS:
        month = panel[pd.to_datetime(panel["eom"]).eq(eom)].sort_values("permno")
        permnos = month["permno"].to_numpy(int)
        risk = provider.get_covariance(eom, permnos)
        lambdas = pd.to_numeric(month["lambda_adv"], errors="coerce").to_numpy(float)
        adjustment = model.adjustment_matrix(risk.covariance, lambdas, eom=eom)
        eigenvalues = adjustment.eigenvalues
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvals(adjustment.matrix)
        real = np.real(eigenvalues)
        imaginary = np.abs(np.imag(eigenvalues))
        row = {
            "eom": str(eom.date()),
            "n_names": int(len(month)),
            "risk_model_used": risk.used_model,
            "factor_fallback": bool(risk.used_model != "factor" or risk.fallback_reason),
            "covariance_repair_count": int(risk.covariance_repair_count),
            "covariance_min_eigenvalue": float(risk.min_eigenvalue),
            "covariance_condition_number": float(risk.condition_number),
            "lambda_min": float(lambdas.min()),
            "lambda_median": float(np.median(lambdas)),
            "lambda_max": float(lambdas.max()),
            "lambda_all_positive": bool(np.isfinite(lambdas).all() and (lambdas > 0).all()),
            "matrix_m_mode": adjustment.mode,
            "matrix_m_fallback": adjustment.mode != "matrix_m",
            "matrix_m_real_eigenvalue_min": float(real.min()),
            "matrix_m_real_eigenvalue_median": float(np.median(real)),
            "matrix_m_real_eigenvalue_max": float(real.max()),
            "matrix_m_max_imaginary": float(imaginary.max()),
        }
        passed = bool(
            not row["factor_fallback"]
            and row["covariance_repair_count"] == 0
            and row["lambda_all_positive"]
            and not row["matrix_m_fallback"]
            and real.min() >= -1.0e-7
            and real.max() <= 1.0 + 1.0e-7
            and imaginary.max() <= 1.0e-7
        )
        row["passed"] = passed
        rows.append(row)
        _check(checks, f"risk_cost_m_{eom.strftime('%Y_%m')}", passed, row)
    return rows


def _closed_form_diagnostics(
    panel: pd.DataFrame,
    provider: RiskModelProvider,
    values: dict[str, object],
    adjustment_cache: PersistentAdjustmentCache,
    design_cache: DesignMatrixCache,
    transaction_cache: TransactionCostCache,
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    dates = pd.to_datetime(panel["eom"])
    train = panel[dates.between("2003-01-31", "2004-12-31")]
    test = panel[dates.between("2005-01-31", "2005-03-31")]
    specs = (
        PortfolioMLSpec("linear", 1000.0, seed=int(values["random_seed"])),
        PortfolioMLSpec("rf", 1000.0, rf_p=32, rf_eta=0.5, seed=int(values["random_seed"])),
    )
    rows: list[dict[str, object]] = []
    for spec in specs:
        model = _portfolio_model(
            spec,
            provider,
            values,
            adjustment_cache,
            design_cache,
            transaction_cache,
        ).fit(train)
        method = "portfolio_ml_linear" if spec.kind == "linear" else "portfolio_ml_rf"
        result = run_backtest(
            test,
            method=method,
            risk_provider=provider,
            gamma=float(values["baseline_gamma"]),
            aum=float(values["baseline_aum"]),
            accounting_settings=values["accounting"],
            method_settings={**values["portfolio_ml"], **values["method_execution"]},
            portfolio_model=model,
            test_year=2005,
            beta_norm=float(np.linalg.norm(model.beta_)),
            transaction_cache=transaction_cache,
        )
        gradient_norm = float(np.linalg.norm(model.gradient()))
        gradient_tolerance = 1.0e-7 * (1.0 + float(np.linalg.norm(model.mean_numerator_)))
        hessian_min = float(np.linalg.eigvalsh(model.quadratic_hessian()).min())
        objective_hat = float(model.objective())
        objective_zero = float(model.objective(np.zeros_like(model.beta_)))
        leverage_mean = float(result.returns["leverage"].mean())
        leverage_max = float(result.returns["leverage"].max())
        turnover_mean = float(result.returns["turnover"].mean())
        turnover_max = float(result.returns["turnover"].max())
        trading_cost_mean = float(result.returns["trading_cost"].mean())
        trading_cost_max = float(result.returns["trading_cost"].max())
        passed = bool(
            gradient_norm <= gradient_tolerance
            and hessian_min >= -1.0e-8
            and np.isfinite(model.beta_).all()
            and objective_hat >= objective_zero - 1.0e-10
            and result.risk_models_used == {"factor"}
            and result.adjustment_modes == {"matrix_m"}
            and leverage_mean <= float(values["safety"]["max_portfolio_mean_leverage"])
            and leverage_max <= float(values["safety"]["max_portfolio_monthly_leverage"])
            and turnover_mean <= float(values["safety"]["max_portfolio_mean_turnover"])
            and turnover_max <= float(values["safety"]["max_portfolio_monthly_turnover"])
            and np.isfinite(trading_cost_mean)
            and trading_cost_mean >= 0.0
        )
        row = {
            "kind": spec.kind,
            "ridge_lambda": spec.ridge_lambda,
            "rf_p": spec.rf_p,
            "rf_eta": spec.rf_eta,
            "train_months": int(model.training_months_),
            "test_months": int(result.returns["eom"].nunique()),
            "gradient_norm": gradient_norm,
            "gradient_tolerance": gradient_tolerance,
            "hessian_min_eigenvalue": hessian_min,
            "beta_norm": float(np.linalg.norm(model.beta_)),
            "objective_at_beta_hat": objective_hat,
            "objective_at_beta_zero": objective_zero,
            "objective_improvement": objective_hat - objective_zero,
            "mean_leverage": leverage_mean,
            "max_monthly_leverage": leverage_max,
            "mean_turnover": turnover_mean,
            "max_monthly_turnover": turnover_max,
            "mean_monthly_trading_cost": trading_cost_mean,
            "max_monthly_trading_cost": trading_cost_max,
            "factor_fallback_count": int((result.returns["risk_model_used"] != "factor").sum()),
            "matrix_m_fallback_count": int((result.returns["adjustment_mode"] != "matrix_m").sum()),
            "passed": passed,
        }
        rows.append(row)
        _check(checks, f"portfolio_closed_form_{spec.kind}", passed, row)
    return rows


def _portfolio_model(
    spec: PortfolioMLSpec,
    provider: RiskModelProvider,
    values: dict[str, object],
    adjustment_cache: PersistentAdjustmentCache,
    design_cache: DesignMatrixCache,
    transaction_cache: TransactionCostCache,
) -> PortfolioMLModel:
    data_config = load_data_config(CONFIG_PATH.parent / "data_us_equity_ml.yaml")
    return PortfolioMLModel(
        spec,
        [f"rank_{name}" for name in data_config.feature_sets[str(values["feature_set"])]],
        provider,
        gamma=float(values["baseline_gamma"]),
        aum=float(values["baseline_aum"]),
        settings=values["portfolio_ml"],
        adjustment_cache=adjustment_cache,
        design_cache=design_cache,
        transaction_cache=transaction_cache,
    )


def _check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: object,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _markdown_report(payload: dict[str, object]) -> str:
    checks = pd.DataFrame(payload["checks"])[["name", "passed"]]
    features = pd.DataFrame(
        [
            {"feature": feature, "raw_missing_rate": rate}
            for feature, rate in payload["feature_diagnostics"]["raw_feature_missing_rate"].items()
        ]
    )
    return (
        f"# Top500/{payload['feature_set']} preflight report\n\n"
        f"Overall gate: **{'PASS' if payload['passed'] else 'FAIL'}**. No full "
        "backtest is started by this script. The guarded runner may continue only "
        "when the adjacent JSON contains `passed: true`.\n\n"
        "## Checks\n\n"
        + _markdown(checks)
        + "\n\n## Raw feature missing rates\n\n"
        + _markdown(features)
        + "\n\n## RF diagnostics\n\n"
        + _markdown(pd.DataFrame(payload["rf_diagnostics"]))
        + "\n\n## Risk, cost and matrix m\n\n"
        + _markdown(pd.DataFrame(payload["risk_cost_m_diagnostics"]))
        + "\n\n## Portfolio-ML closed form\n\n"
        + _markdown(pd.DataFrame(payload["closed_form_diagnostics"]))
        + "\n"
    )


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
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
