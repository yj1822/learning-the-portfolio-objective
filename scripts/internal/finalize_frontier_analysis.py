from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from implementable_frontier.models.formal_analysis import (  # noqa: E402
    risk_calibration_report,
)
from implementable_frontier.models.metrics import performance_metrics  # noqa: E402


GAMMAS = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
METHODS = [
    "return_ml_rank",
    "static_ml",
    "static_ml_star",
    "portfolio_ml_linear",
    "portfolio_ml_rf",
]
DISPLAY = {
    "return_ml_rank": "Return-ML Rank",
    "static_ml": "Static-ML",
    "static_ml_star": "Static-ML*",
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}
COLORS = {
    "return_ml_rank": "#50545a",
    "static_ml": "#d97706",
    "static_ml_star": "#16875b",
    "portfolio_ml_linear": "#2563a6",
    "portfolio_ml_rf": "#b43c55",
}
BASELINE_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/frontier"
FIGURE_ROOT = REPORT_ROOT / "figures"


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_metrics = pd.read_csv(BASELINE_ROOT / "metrics.csv").set_index("method")
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for gamma in GAMMAS:
        root = _result_root(gamma)
        returns = pd.read_parquet(root / "returns.parquet")
        validation = pd.read_parquet(root / "validation_selection.parquet")
        reported = pd.read_csv(root / "metrics.csv").set_index("method")
        recomputed = performance_metrics(returns, gamma=gamma).set_index("method")
        max_metric_error = _max_metric_error(reported, recomputed)
        if max_metric_error > 1.0e-6:
            raise RuntimeError(
                f"gamma={gamma:g} metric reconciliation failed: {max_metric_error}"
            )
        quality = _quality(returns, validation)
        quality_rows.append({"gamma": gamma, **quality})
        if not quality["passed"]:
            raise RuntimeError(f"gamma={gamma:g} quality gate failed: {quality}")
        calibration = risk_calibration_report(returns)
        calibration = calibration[calibration["scope"].eq("full")].set_index("method")
        selections = _selection_summary(root)
        for method in METHODS:
            values = recomputed.loc[method]
            monthly = returns[returns["method"].eq(method)]
            rows.append(
                {
                    "gamma": gamma,
                    "method": method,
                    "method_label": DISPLAY[method],
                    "source": "locked_baseline_reused" if gamma == 10 else "controlled_recomputation",
                    "annualized_gross_return": values["annualized_gross_return"],
                    "annualized_net_return": values["annualized_net_return"],
                    "annualized_volatility": values["annualized_volatility"],
                    "net_sharpe": values["net_sharpe"],
                    "monthly_turnover": values["average_turnover"],
                    "average_leverage": values["average_leverage"],
                    "annualized_trading_cost": values["annualized_trading_cost"],
                    "annualized_utility": values["annualized_ex_ante_utility_flow"],
                    "empirical_utility": values["empirical_utility"],
                    "realized_to_predicted_volatility_ratio": calibration.loc[
                        method, "realized_to_predicted_ratio"
                    ],
                    "predicted_annualized_volatility": calibration.loc[
                        method, "predicted_annualized_volatility"
                    ],
                    "max_leverage": pd.to_numeric(
                        monthly["leverage"], errors="coerce"
                    ).max(),
                    "max_monthly_turnover": pd.to_numeric(
                        monthly["turnover"], errors="coerce"
                    ).max(),
                    "fallback_count": quality["factor_fallback_count"]
                    + quality["matrix_m_fallback_count"],
                    "covariance_repair_count": quality["covariance_repair_count"],
                    "selected_hyperparameters": selections.get(method, {}).get(
                        "modal_parameters", "not_applicable"
                    ),
                    "selected_hyperparameter_distinct_count": selections.get(
                        method, {}
                    ).get("distinct_count", 0),
                    "metric_reconciliation_max_abs_error": max_metric_error,
                }
            )
    points = pd.DataFrame(rows).sort_values(["method", "gamma"]).reset_index(drop=True)
    _validate_baseline(points, baseline_metrics)
    points.to_csv(REPORT_ROOT / "frontier_points.csv", index=False)
    points.to_parquet(REPORT_ROOT / "frontier_points.parquet", index=False)
    _plot(points, "annualized_net_return", "Annualized net return", "frontier_net_return_vs_volatility.png", percent_y=True)
    _plot(points, "annualized_utility", "Annualized utility flow", "frontier_utility_vs_volatility.png", percent_y=True)
    _plot(points, "annualized_trading_cost", "Annualized trading cost", "frontier_cost_vs_volatility.png", percent_y=True)
    _plot(points, "monthly_turnover", "Average monthly turnover", "frontier_turnover_vs_volatility.png")
    _plot(points, "average_leverage", "Average gross leverage", "frontier_leverage_vs_volatility.png")
    report = _report(points, pd.DataFrame(quality_rows))
    (REPORT_ROOT / "implementable_frontier.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "points": len(points),
                "gammas": GAMMAS,
                "methods": METHODS,
                "fallback_count": int(points["fallback_count"].sum()),
                "covariance_repair_count": int(points["covariance_repair_count"].sum()),
                "baseline_max_abs_error": float(
                    points.loc[points["gamma"].eq(10), "metric_reconciliation_max_abs_error"].max()
                ),
                "output": str(REPORT_ROOT),
            },
            indent=2,
        )
    )
    return 0


def _result_root(gamma: float) -> Path:
    if gamma == 10:
        return BASELINE_ROOT
    slug = f"{gamma:g}".replace(".", "p")
    return PROJECT_ROOT / f"results/final_analysis/frontier/gamma_{slug}"


def _max_metric_error(reported: pd.DataFrame, recomputed: pd.DataFrame) -> float:
    columns = [
        "annualized_gross_return",
        "annualized_net_return",
        "annualized_volatility",
        "net_sharpe",
        "average_turnover",
        "average_leverage",
        "annualized_trading_cost",
        "empirical_utility",
        "annualized_ex_ante_utility_flow",
    ]
    left = reported.loc[METHODS, columns].astype(float)
    right = recomputed.loc[METHODS, columns].astype(float)
    return float((left - right).abs().to_numpy().max())


def _quality(returns: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    portfolio = returns[returns["method"].astype(str).str.startswith("portfolio_ml")]
    factor = int(
        (
            ~returns["risk_model_used"].astype(str).eq("factor")
            | returns["risk_fallback_reason"].notna()
        ).sum()
    )
    matrix = int((~portfolio["adjustment_mode"].astype(str).eq("matrix_m")).sum())
    repair = int(
        pd.to_numeric(returns["covariance_repair_count"], errors="coerce")
        .fillna(0)
        .sum()
    )
    future = int(validation["future_information_used"].fillna(False).sum())
    cutoff = int(
        (
            pd.to_numeric(validation["latest_validation_year"], errors="coerce")
            >= pd.to_numeric(validation["test_year"], errors="coerce")
        ).sum()
    )
    return {
        "passed": factor == matrix == repair == future == cutoff == 0,
        "factor_fallback_count": factor,
        "matrix_m_fallback_count": matrix,
        "covariance_repair_count": repair,
        "future_information_violations": future,
        "validation_cutoff_violations": cutoff,
    }


def _selection_summary(root: Path) -> dict[str, dict[str, object]]:
    path = root / "selected_hyperparameters_by_year.csv"
    if not path.exists():
        return {}
    selections = pd.read_csv(path)
    output: dict[str, dict[str, object]] = {}
    for method, group in selections.groupby("method"):
        parameters = group["candidate_hyperparameters"].astype(str)
        output[str(method)] = {
            "modal_parameters": parameters.value_counts().index[0],
            "distinct_count": int(parameters.nunique()),
        }
    return output


def _validate_baseline(points: pd.DataFrame, baseline: pd.DataFrame) -> None:
    mapping = {
        "annualized_net_return": "annualized_net_return",
        "annualized_volatility": "annualized_volatility",
        "net_sharpe": "net_sharpe",
        "monthly_turnover": "average_turnover",
        "average_leverage": "average_leverage",
        "annualized_trading_cost": "annualized_trading_cost",
        "annualized_utility": "annualized_ex_ante_utility_flow",
    }
    baseline_points = points[points["gamma"].eq(10)].set_index("method")
    errors = []
    for point_column, baseline_column in mapping.items():
        errors.append(
            (
                baseline_points.loc[METHODS, point_column].astype(float)
                - baseline.loc[METHODS, baseline_column].astype(float)
            ).abs().max()
        )
    if max(errors) > 1.0e-6:
        raise RuntimeError(f"gamma=10 does not reproduce baseline: {max(errors)}")


def _plot(
    points: pd.DataFrame,
    y_column: str,
    y_label: str,
    filename: str,
    *,
    percent_y: bool = False,
) -> None:
    width, height = 1900, 1240
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    title_font = ImageFont.truetype(str(font_root / "arialbd.ttf"), 42)
    label_font = ImageFont.truetype(str(font_root / "arial.ttf"), 28)
    tick_font = ImageFont.truetype(str(font_root / "arial.ttf"), 23)
    legend_font = ImageFont.truetype(str(font_root / "arial.ttf"), 23)
    note_font = ImageFont.truetype(str(font_root / "arial.ttf"), 20)
    left, right, top, bottom = 175, 1840, 120, 1080
    x_values = 100.0 * points["annualized_volatility"].to_numpy(float)
    y_values = points[y_column].to_numpy(float) * (100.0 if percent_y else 1.0)
    x_min, x_max = _padded_limits(x_values)
    y_min, y_max = _padded_limits(y_values)

    def xy(x: float, y: float) -> tuple[int, int]:
        px = left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return int(px), int(py)

    grid_color = "#d9dde3"
    axis_color = "#34383e"
    for value in np.linspace(x_min, x_max, 6):
        px, _ = xy(float(value), y_min)
        draw.line((px, top, px, bottom), fill=grid_color, width=2)
        text = f"{value:.1f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((px - (box[2] - box[0]) / 2, bottom + 18), text, fill=axis_color, font=tick_font)
    for value in np.linspace(y_min, y_max, 6):
        _, py = xy(x_min, float(value))
        draw.line((left, py, right, py), fill=grid_color, width=2)
        text = f"{value:.2f}" if percent_y else f"{value:.3f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((left - 20 - (box[2] - box[0]), py - 13), text, fill=axis_color, font=tick_font)
    draw.line((left, top, left, bottom), fill=axis_color, width=3)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=3)

    for method in METHODS:
        group = points[points["method"].eq(method)].sort_values("annualized_volatility")
        x = 100.0 * group["annualized_volatility"].to_numpy(float)
        y = group[y_column].to_numpy(float)
        if percent_y:
            y = 100.0 * y
        coordinates = [xy(float(a), float(b)) for a, b in zip(x, y)]
        draw.line(coordinates, fill=COLORS[method], width=5, joint="curve")
        for px, py in coordinates:
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=COLORS[method])
        baseline = group[group["gamma"].eq(10)]
        bx = 100.0 * float(baseline["annualized_volatility"].iloc[0])
        by = float(baseline[y_column].iloc[0]) * (100.0 if percent_y else 1.0)
        px, py = xy(bx, by)
        draw.polygon(_star_points(px, py, 20, 9), fill=COLORS[method], outline="white")

    title = f"Top500/Core10 implementable frontier: {y_label.lower()}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 35), title, fill="#20242a", font=title_font)
    x_label = "Annualized realized volatility (%)"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right - (x_box[2] - x_box[0])) / 2, 1140), x_label, fill=axis_color, font=label_font)
    y_text = y_label + (" (%)" if percent_y else "")
    y_layer = Image.new("RGBA", (700, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_layer)
    y_draw.text((0, 8), y_text, fill=axis_color, font=label_font)
    y_layer = y_layer.rotate(90, expand=True)
    image.paste(y_layer, (35, int((top + bottom - y_layer.height) / 2)), y_layer)

    legend_x, legend_y = 250, 90
    for index, method in enumerate(METHODS):
        column, row = index % 3, index // 3
        x0 = legend_x + column * 510
        y0 = legend_y + row * 42
        draw.line((x0, y0 + 12, x0 + 45, y0 + 12), fill=COLORS[method], width=5)
        draw.ellipse((x0 + 17, y0 + 4, x0 + 33, y0 + 20), fill=COLORS[method])
        draw.text((x0 + 58, y0), DISPLAY[method], fill=axis_color, font=legend_font)
    note = "Stars mark the locked gamma=10 baseline"
    note_box = draw.textbbox((0, 0), note, font=note_font)
    draw.text((right - (note_box[2] - note_box[0]), 1195), note, fill="#50545a", font=note_font)
    image.save(FIGURE_ROOT / filename, format="PNG", optimize=True)


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = high - low
    padding = 0.08 * span if span > 0 else max(abs(low) * 0.08, 0.1)
    return low - padding, high + padding


def _star_points(
    center_x: int, center_y: int, outer: float, inner: float
) -> list[tuple[int, int]]:
    points = []
    for index in range(10):
        angle = -np.pi / 2 + index * np.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append(
            (
                int(center_x + radius * np.cos(angle)),
                int(center_y + radius * np.sin(angle)),
            )
        )
    return points


def _report(points: pd.DataFrame, quality: pd.DataFrame) -> str:
    summary = (
        points.groupby("method", as_index=False)
        .agg(
            net_return_min=("annualized_net_return", "min"),
            net_return_max=("annualized_net_return", "max"),
            volatility_min=("annualized_volatility", "min"),
            volatility_max=("annualized_volatility", "max"),
            utility_min=("annualized_utility", "min"),
            utility_max=("annualized_utility", "max"),
            max_leverage=("max_leverage", "max"),
            max_turnover=("max_monthly_turnover", "max"),
        )
    )
    summary["method"] = summary["method"].map(DISPLAY)
    same_gamma = points.pivot(index="gamma", columns="method")
    star_utility_gain = float(
        (
            same_gamma["annualized_utility"]["static_ml_star"]
            - same_gamma["annualized_utility"]["static_ml"]
        ).mean()
    )
    linear_static_gain = float(
        (
            same_gamma["annualized_utility"]["portfolio_ml_linear"]
            - same_gamma["annualized_utility"]["static_ml"]
        ).mean()
    )
    rf_static_gain = float(
        (
            same_gamma["annualized_utility"]["portfolio_ml_rf"]
            - same_gamma["annualized_utility"]["static_ml"]
        ).mean()
    )
    rank = points[points["method"].eq("return_ml_rank")].iloc[0]
    rank_cost_share = float(
        rank["annualized_trading_cost"]
        / max(abs(rank["annualized_gross_return"]), 1.0e-12)
    )
    stability = []
    for method in METHODS:
        group = points[points["method"].eq(method)].sort_values("gamma")
        volatility = group["annualized_volatility"].to_numpy(float)
        stability.append(
            {
                "method": DISPLAY[method],
                "volatility_monotonicity_violations": int(np.sum(np.diff(volatility) > 1.0e-8)),
                "risk_ratio_min": float(group["realized_to_predicted_volatility_ratio"].min()),
                "risk_ratio_max": float(group["realized_to_predicted_volatility_ratio"].max()),
                "max_leverage": float(group["max_leverage"].max()),
            }
        )
    baseline = points[points["gamma"].eq(10)][
        [
            "method_label",
            "annualized_net_return",
            "annualized_volatility",
            "annualized_utility",
            "monthly_turnover",
            "average_leverage",
            "annualized_trading_cost",
        ]
    ]
    return (
        "# Implementable efficient frontier\n\n"
        "This is a controlled gamma-grid mechanism analysis for the locked Top500/Core10 specification. Gamma is not selected from these results. The fixed grid is 2.5, 5, 7.5, 10, 15, 20, 30 and 50; gamma 10 reuses the locked baseline.\n\n"
        "## Quality gate\n\n"
        f"All {len(GAMMAS)} gamma points passed. Factor fallback, matrix-m fallback, covariance repair, future-information violations and validation-cutoff violations are all zero. The gamma=10 metric reconciliation error is at most {points.loc[points['gamma'].eq(10), 'metric_reconciliation_max_abs_error'].max():.3g}.\n\n"
        "## Baseline gamma 10\n\n"
        + _markdown(baseline)
        + "\n\n## Frontier ranges\n\n"
        + _markdown(summary)
        + "\n\n## Stability\n\n"
        + _markdown(pd.DataFrame(stability))
        + "\n\n## Interpretation\n\n"
        f"- Portfolio-ML Linear exceeds Static-ML by an average {linear_static_gain:.4%} annualized utility flow at the same gamma; Portfolio-ML RF exceeds Static-ML by {rf_static_gain:.4%}. The plotted risk coordinates allow direct comparison at nearby realized-volatility levels.\n"
        "- The leverage and turnover panels separate scale from implementation. A Portfolio-ML advantage accompanied by lower turnover/cost and comparable leverage is not mechanically a leverage result.\n"
        f"- Static-ML* changes average annualized utility by {star_utility_gain:.4%} relative to Static-ML across the fixed grid.\n"
        f"- Return-ML Rank's annualized trading cost is {rank_cost_share:.2f} times the absolute annualized gross return, documenting poor net implementability.\n"
        "- Linear versus RF stability should be read from monotonicity, risk-ratio range, leverage and turnover jointly; no gamma point is deleted for looking unfavorable.\n"
        "- Gamma 10 is retained because it is the pre-locked baseline, not because this frontier reselected it.\n\n"
        "## Quality details\n\n"
        + _markdown(quality)
        + "\n"
    )


def _markdown(frame: pd.DataFrame) -> str:
    data = frame.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = [str(value) for value in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
