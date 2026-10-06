from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from implementable_frontier.models.risk import RiskModelProvider


def evaluate_factor_risk_quality(
    panel: pd.DataFrame,
    risk_inputs: dict[str, pd.DataFrame],
    *,
    universe: str,
    risk_settings: dict[str, object],
) -> dict[str, Any]:
    provider = RiskModelProvider(
        panel,
        universe=universe,
        requested_model="factor",
        settings={**risk_settings, "allow_factor_fallback": True},
        risk_inputs=risk_inputs,
    )
    monthly_rows: list[dict[str, object]] = []
    equal_returns: list[float] = []
    for eom, month in panel.groupby("eom", sort=True):
        month = month.sort_values("permno")
        result = provider.get_covariance(eom, month["permno"].astype(int).tolist())
        n_assets = len(month)
        weights = np.full(n_assets, 1.0 / n_assets)
        predicted_volatility = float(
            np.sqrt(max(12.0 * weights @ result.covariance @ weights, 0.0))
        )
        realized = pd.to_numeric(month["ret_exc_lead1m"], errors="coerce")
        equal_returns.append(float(realized.mean()))
        monthly_rows.append(
            {
                "eom": pd.Timestamp(eom),
                "used_model": result.used_model,
                "fallback_reason": result.fallback_reason,
                "predicted_equal_weight_annual_volatility": predicted_volatility,
                "median_asset_annual_volatility": float(
                    np.sqrt(12.0 * result.diagonal_median)
                ),
                "condition_number": result.condition_number,
                "idio_imputed_assets": len(result.idio_imputed_assets),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    factor_returns = _finite(risk_inputs["factor_returns_daily"]["factor_return"])
    factor_covariance = risk_inputs["factor_cov_monthly"]
    factor_diagonal = _finite(
        factor_covariance.loc[
            factor_covariance["factor_1"].eq(factor_covariance["factor_2"]),
            "covariance",
        ]
    )
    idio_variance = _finite(risk_inputs["idio_var_monthly"]["idio_var"])
    design_condition = _finite(
        risk_inputs["factor_returns_daily"].get(
            "design_condition_number", pd.Series(dtype=float)
        )
    )
    exposures = risk_inputs["factor_exposures"]
    style_columns = [column for column in exposures if column.startswith("rank_")]
    style_means = (
        exposures.groupby("eom")[style_columns].mean().abs().to_numpy().ravel()
        if style_columns
        else np.array([])
    )
    style_stds = (
        exposures.groupby("eom")[style_columns].std(ddof=0).to_numpy().ravel()
        if style_columns
        else np.array([])
    )
    usable_fraction = float(monthly["used_model"].eq("factor").mean())
    realized_annual_volatility = float(
        np.sqrt(12.0) * pd.Series(equal_returns).std(ddof=1)
    )
    predicted_average = float(
        monthly["predicted_equal_weight_annual_volatility"].mean()
    )
    factor_abs = factor_returns.abs()
    quality_passed = bool(
        usable_fraction >= 0.95
        and factor_abs.quantile(0.999) <= float(
            risk_settings.get("factor_return_abs_p999_max", 0.25)
        )
        and monthly["median_asset_annual_volatility"].median()
        <= float(risk_settings.get("factor_max_annualized_asset_volatility", 2.0))
    )
    return {
        "quality_passed": quality_passed,
        "months": int(len(monthly)),
        "factor_usable_month_fraction": usable_fraction,
        "factor_fallback_months": int(monthly["used_model"].ne("factor").sum()),
        "factor_fallback_reason_examples": monthly["fallback_reason"].dropna().unique()[:5].tolist(),
        "daily_factor_return": _distribution(factor_returns),
        "daily_absolute_factor_return": _distribution(factor_abs),
        "daily_factor_covariance_diagonal": _distribution(factor_diagonal),
        "daily_idiosyncratic_variance": _distribution(idio_variance),
        "regression_design_condition": _distribution(design_condition),
        "factor_covariance_diagonal_fallback_share": float(
            factor_covariance.get("diagonal_fallback", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .mean()
        ),
        "exposure_audit": {
            "intercept_present": "intercept" in exposures,
            "style_columns": style_columns,
            "max_absolute_monthly_style_mean": float(np.nanmax(style_means))
            if style_means.size
            else None,
            "median_monthly_style_std": float(np.nanmedian(style_stds))
            if style_stds.size
            else None,
        },
        "ex_ante_vs_realized_equal_weight": {
            "average_predicted_annual_volatility": predicted_average,
            "realized_annual_volatility": realized_annual_volatility,
            "realized_to_predicted_ratio": (
                realized_annual_volatility / predicted_average
                if predicted_average > 0
                else None
            ),
        },
        "monthly_diagnostics": monthly.to_dict(orient="records"),
    }


def write_factor_risk_quality_report(
    quality: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "factor_risk_quality.json"
    markdown_path = root / "factor_risk_quality.md"
    json_path.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    summary = {key: value for key, value in quality.items() if key != "monthly_diagnostics"}
    lines = ["# Factor Risk Quality Report", ""]
    for section, values in summary.items():
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        if isinstance(values, dict):
            lines.extend(f"- `{key}`: {value}" for key, value in values.items())
        else:
            lines.append(str(values))
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    if values.empty:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "p999": None, "max": None}
    return {
        "count": int(len(values)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "p999": float(values.quantile(0.999)),
        "max": float(values.max()),
    }


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
