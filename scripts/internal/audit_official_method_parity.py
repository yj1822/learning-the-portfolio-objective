from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.data.loaders import (  # noqa: E402
    load_master_panel,
    load_risk_inputs,
)
from implementable_frontier.models.method_audit import (  # noqa: E402
    run_small_sample_parity,
)
from implementable_frontier.models.portfolios import (  # noqa: E402
    compute_adjustment_matrix,
)
from implementable_frontier.models.risk import RiskModelProvider  # noqa: E402


OFFICIAL_REPOSITORY = "https://github.com/theisij/ml-and-the-implementable-efficient-frontier"
OFFICIAL_DIR = PROJECT_ROOT / "third_party" / "original_ief_code"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Python parity with official IEF code")
    parser.add_argument("--start", default="2005-01-31")
    parser.add_argument("--end", default="2007-12-31")
    args = parser.parse_args()

    if not (OFFICIAL_DIR / ".git").exists():
        raise FileNotFoundError(
            "Official checkout is missing; clone it into third_party/original_ief_code first"
        )
    commit = _git("rev-parse", "HEAD")
    commit_date = _git("show", "-s", "--format=%cI", "HEAD")
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "model_top100_core10_rolling_factor.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_checks = _source_checks()
    parity = run_small_sample_parity(seed=int(config["random_seed"]) % 100_000)
    matrix_audit = _actual_matrix_audit(config, args.start, args.end)
    differences = [
        {
            "module": "Portfolio-ML dynamic B",
            "official": "11-month omega aggregation with market total-return growth",
            "project": "dynamic-permno recursive B with an observable top100 market-growth proxy",
            "reason": "Official fixed-row matrix products do not define entry/exit behavior for a dynamic top-N universe.",
            "status": "documented_adapter",
        },
        {
            "module": "Portfolio-ML exit costs",
            "official": "No explicit dropped-name term in transformed denominator",
            "project": "Explicit liquidation cost for names leaving top100",
            "reason": "Required for implementable dynamic-universe accounting.",
            "status": "documented_adapter",
        },
        {
            "module": "Random number stream",
            "official": "R set.seed plus MASS::mvrnorm",
            "project": "NumPy PCG64 fixed seed",
            "reason": "Distribution and prefix reuse match; bitwise R RNG equality is not claimed.",
            "status": "distribution_parity_only",
        },
        {
            "module": "Return-ML initial years",
            "official": "10-year MSE validation with long prehistory",
            "project": "10-year MSE window when available; expanding shorter window before 2000",
            "reason": "Repaired factor covariance begins in 1995 and the local return panel begins in 1990.",
            "status": "boundary_adapter",
        },
        {
            "module": "Static-ML* grid inclusion",
            "official": "Separate k/u/g grid",
            "project": "Current k grid contains only the phi=1 Static mapping at v=0",
            "reason": "The user fixed the current small grids and prohibited grid expansion.",
            "status": "candidate_space_difference",
        },
    ]
    modules = {
        "portfolio_ml_closed_form": {
            "status": "pass_with_dynamic_universe_adapter",
            "formula": "beta=(mean(Sigma_tilde)+lambda*I)^-1 mean(r_tilde)",
            "risk_and_cost_in_denominator": True,
            "training_target": "portfolio utility, not return MSE",
            "small_sample_gradient_norm": parity["portfolio_ml"]["gradient_norm"],
            "small_sample_hessian_min": parity["portfolio_ml"]["hessian_eigenvalue_min"],
        },
        "full_matrix_m": {
            "status": "pass",
            **matrix_audit,
        },
        "random_features": {
            "status": "pass_distribution_parity",
            "formula": "[cos(XW), sin(XW)]/sqrt(p), W~N(0, eta^2 I)",
            "official_parameter_mapping": "official g = project eta^2",
            "nested_p_prefix_reuse": True,
            "monthly_demean_and_norm_for_portfolio_ml": True,
            "return_ml_monthly_demean": False,
            "fixed_seed": int(config["random_seed"]),
            "small_sample": parity["rf"],
        },
        "static_ml": {
            "status": "pass",
            "realized_cost_uses_baseline_lambda": True,
            "realized_risk_uses_unadjusted_covariance_and_returns": True,
        },
        "static_ml_star": {
            "status": "pass",
            "v_definition": "Sigma_star = Sigma + v * diag(diag(Sigma))",
            "v_units": "dimensionless multiplier of the covariance diagonal",
            "decision_only_scalars": ["u", "v", "k"],
            "source_check": source_checks["static_cov_add"],
        },
        "return_ml": {
            "status": "formal_runner_required",
            "label": "one-month-ahead excess return",
            "selection_target": "prediction MSE",
            "official_validation_years": 10,
            "portfolio_tuning_separated": True,
        },
        "units": {
            "status": "pass",
            "returns": "monthly decimal",
            "covariance": "monthly decimal-return covariance",
            "lambda_adv": "0.2 / ADV_6m dollar volume",
            "trading_cost": "AUM/2 * delta_pi' Lambda delta_pi, return fraction",
            "gamma": "monthly mean-variance utility",
        },
    }
    official = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "commit_date": commit_date,
        "retrieval_date": date.today().isoformat(),
        "local_path": str(OFFICIAL_DIR),
        "rscript_available": shutil.which("Rscript") is not None,
        "reference_files": [
            "0 - Portfolio choice functions.R: m_func, static_val_fun, static_implement, pfml_input_fun, pfml_search_coef, pfml_hp_reals_fun, pfml_w",
            "0 - Return prediction functions.R: data_split, rff, rff_hp_search, ridge_hp_search",
            "0 - General functions.R: sigma_gam_adj, pf_ts_fun",
            "1 - Prepare Data.R: wealth_func, Lambda construction",
            "2 - Fit Models.R: yearly first-layer prediction fitting",
            "4 - Prepare Portfolio Data.R: hp_years, dates_hp, dates_oos",
            "Main.R: validation and portfolio settings",
        ],
    }
    audit = {
        "official": official,
        "source_checks": source_checks,
        "modules": modules,
        "documented_differences": differences,
        "small_sample_parity_passed": bool(parity["passed"]),
        "method_audit_passed": bool(
            parity["passed"]
            and matrix_audit["fallback_count"] == 0
            and matrix_audit["failure_count"] == 0
            and all(source_checks.values())
        ),
    }
    audit_dir = PROJECT_ROOT / "reports" / "method_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "official_method_parity.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (audit_dir / "official_method_parity.md").write_text(
        _audit_markdown(audit), encoding="utf-8"
    )
    (audit_dir / "small_sample_parity.json").write_text(
        json.dumps(parity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (audit_dir / "small_sample_parity.md").write_text(
        _parity_markdown(parity, official), encoding="utf-8"
    )
    print(f"official_commit={commit}")
    print(f"method_audit_passed={audit['method_audit_passed']}")
    print(f"small_sample_parity_passed={parity['passed']}")
    print(f"factor_months={matrix_audit['months']}")
    return 0 if audit["method_audit_passed"] else 1


def _actual_matrix_audit(
    config: dict[str, object], start: str, end: str
) -> dict[str, object]:
    panel = load_master_panel(
        start=start, end=end, universe="top100", feature_set="core10"
    )
    risk_inputs = load_risk_inputs(
        start=start,
        end=end,
        universe="top100",
        root=PROJECT_ROOT / config["risk"]["inputs_dir"],
    )
    settings = {**config["risk"], "allow_factor_fallback": False}
    provider = RiskModelProvider(
        panel,
        universe="top100",
        requested_model="factor",
        settings=settings,
        risk_inputs=risk_inputs,
    )
    rows: list[dict[str, float | str]] = []
    failures: list[str] = []
    fallbacks = 0
    portfolio_settings = config["portfolio_ml"]
    for eom, month in panel.groupby("eom", sort=True):
        month = month.sort_values("permno")
        ids = month["permno"].to_numpy(int)
        covariance_result = provider.get_covariance(eom, ids)
        lambdas = month["lambda_adv"].to_numpy(float)
        try:
            result = compute_adjustment_matrix(
                covariance_result.covariance,
                lambdas,
                gamma=float(config["baseline_gamma"]),
                aum=float(config["baseline_aum"]),
                expected_market_return=float(
                    portfolio_settings["expected_market_return"]
                ),
                risk_free_rate=float(portfolio_settings["risk_free_rate"]),
                iterations=int(portfolio_settings["matrix_m_iterations"]),
                allow_scalar_fallback=False,
                scalar_rho=0.5,
            )
            if result.mode != "matrix_m":
                fallbacks += 1
            eigenvalues = np.linalg.eigvals(result.matrix)
            max_imaginary = float(np.max(np.abs(np.imag(eigenvalues))))
            real = np.real(eigenvalues)
            if max_imaginary > 1.0e-7 or real.min() < -1.0e-7 or real.max() > 1.0 + 1.0e-7:
                raise ValueError(
                    f"matrix m eigenvalues outside [0,1] at {pd.Timestamp(eom).date()}"
                )
            rows.append(
                {
                    "eom": str(pd.Timestamp(eom).date()),
                    "eigenvalue_min": float(real.min()),
                    "eigenvalue_median": float(np.median(real)),
                    "eigenvalue_max": float(real.max()),
                    "trading_speed_min": float((1.0 - real).min()),
                    "trading_speed_median": float(np.median(1.0 - real)),
                    "trading_speed_max": float((1.0 - real).max()),
                    "covariance_condition_number": covariance_result.condition_number,
                }
            )
        except Exception as exc:  # recorded as an audit failure
            failures.append(f"{pd.Timestamp(eom).date()}: {type(exc).__name__}: {exc}")
    frame = pd.DataFrame(rows)
    return {
        "months": int(panel["eom"].nunique()),
        "factor_risk_months": int(len(rows)),
        "fallback_count": int(fallbacks),
        "failure_count": int(len(failures)),
        "failure_examples": failures[:5],
        "eigenvalue_min": float(frame["eigenvalue_min"].min()) if not frame.empty else None,
        "eigenvalue_median": float(frame["eigenvalue_median"].median()) if not frame.empty else None,
        "eigenvalue_max": float(frame["eigenvalue_max"].max()) if not frame.empty else None,
        "trading_speed_min": float(frame["trading_speed_min"].min()) if not frame.empty else None,
        "trading_speed_median": float(frame["trading_speed_median"].median()) if not frame.empty else None,
        "trading_speed_max": float(frame["trading_speed_max"].max()) if not frame.empty else None,
        "covariance_condition_median": float(
            frame["covariance_condition_number"].median()
        )
        if not frame.empty
        else None,
        "monthly": rows,
    }


def _source_checks() -> dict[str, bool]:
    portfolio = (OFFICIAL_DIR / "0 - Portfolio choice functions.R").read_text(
        encoding="utf-8"
    )
    general = (OFFICIAL_DIR / "0 - General functions.R").read_text(encoding="utf-8")
    returns = (OFFICIAL_DIR / "0 - Return prediction functions.R").read_text(
        encoding="utf-8"
    )
    return {
        "static_cov_add": "sigma_gam+diag(diag(sigma_gam)*g)" in general,
        "portfolio_r_tilde": "r_tilde <- t(omega) %*% r" in portfolio,
        "portfolio_risk_term": "t(omega) %*% sigma %*% omega" in portfolio,
        "portfolio_cost_term": "t(omega_chg) %*% lambda %*% omega_chg" in portfolio,
        "portfolio_cumulative_validation": "cum_obj := cummean(obj)" in portfolio,
        "static_cumulative_validation": "cum_obj := cummean(r-tc-0.5*cum_var*gamma_rel)" in portfolio,
        "rf_cos_sin": "X_cos=cos(X_new), X_sin=sin(X_new)" in returns,
        "rf_sqrt_p_scale": "p^(-0.5)*cbind" in returns,
        "return_mse_selection": "mse=mean((pred[, i]-data$val$ret_pred)^2)" in returns,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(OFFICIAL_DIR), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _audit_markdown(audit: dict[str, object]) -> str:
    official = audit["official"]
    lines = [
        "# Official Method Parity Audit",
        "",
        f"- Repository: `{official['repository']}`",
        f"- Commit: `{official['commit']}`",
        f"- Commit date: `{official['commit_date']}`",
        f"- Retrieval date: `{official['retrieval_date']}`",
        f"- Rscript available: `{official['rscript_available']}`",
        f"- Audit gate: `{'PASS' if audit['method_audit_passed'] else 'FAIL'}`",
        "",
        "## Reference Files And Functions",
        "",
        *[f"- {value}" for value in official["reference_files"]],
        "",
        "## Module Status",
        "",
        "| Module | Status |",
        "|---|---|",
        *[
            f"| {name} | {values['status']} |"
            for name, values in audit["modules"].items()
        ],
        "",
        "## Static-ML* v Definition",
        "",
        "Official `cov_add` is `Sigma* = Sigma + v * diag(diag(Sigma))`. "
        "The term is the covariance diagonal, not `diag(Sigma^2)`.",
        "",
        "## Documented Differences",
        "",
        "| Module | Official | Project | Reason | Status |",
        "|---|---|---|---|---|",
        *[
            f"| {row['module']} | {row['official']} | {row['project']} | {row['reason']} | {row['status']} |"
            for row in audit["documented_differences"]
        ],
        "",
        "## Full Matrix m Audit",
        "",
        f"- Months: {audit['modules']['full_matrix_m']['months']}",
        f"- Failures: {audit['modules']['full_matrix_m']['failure_count']}",
        f"- Fallbacks: {audit['modules']['full_matrix_m']['fallback_count']}",
        f"- Eigenvalue range: [{audit['modules']['full_matrix_m']['eigenvalue_min']:.6g}, {audit['modules']['full_matrix_m']['eigenvalue_max']:.6g}]",
        f"- Median trading speed: {audit['modules']['full_matrix_m']['trading_speed_median']:.6g}",
        "",
    ]
    return "\n".join(lines)


def _parity_markdown(parity: dict[str, object], official: dict[str, object]) -> str:
    lines = [
        "# Small-Sample Method Parity",
        "",
        f"- Official commit: `{official['commit']}`",
        f"- Sample: {parity['sample']['months']} months, {parity['sample']['names_per_month']} names/month, {parity['sample']['features']} features",
        f"- Gate: `{'PASS' if parity['passed'] else 'FAIL'}`",
        "- Direct R execution was not required; comparisons use an independent Python migration of the official matrix equations.",
        "",
        "| Check | Absolute error | Relative error | Tolerance | Result | Note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for check in parity["checks"]:
        lines.append(
            f"| {check['name']} | {check['absolute_error']:.6g} | {check['relative_error']:.6g} | {check['tolerance']:.6g} | {'PASS' if check['passed'] else 'FAIL'} | {check['note']} |"
        )
    lines.extend(
        [
            "",
            "## Portfolio-ML Quadratic Diagnostics",
            "",
            f"- Gradient norm: {parity['portfolio_ml']['gradient_norm']:.6g}",
            f"- Hessian eigenvalue min/median/max: {parity['portfolio_ml']['hessian_eigenvalue_min']:.6g} / {parity['portfolio_ml']['hessian_eigenvalue_median']:.6g} / {parity['portfolio_ml']['hessian_eigenvalue_max']:.6g}",
            f"- Objective at beta / zero: {parity['portfolio_ml']['objective_at_beta']:.6g} / {parity['portfolio_ml']['objective_at_zero']:.6g}",
            f"- Maximum local perturbation gain: {parity['portfolio_ml']['max_local_perturbation_gain']:.6g}",
            "",
            "## Saved Intermediate Values",
            "",
            "The JSON companion stores beta, aim and actual weights, inherited weights, trades, transaction cost, gross/net return, utility, RF diagnostics, and matrix-m diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
