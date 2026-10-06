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


MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
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
BASELINE_AUM = 1.0e9
BASELINE_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/aum_sensitivity"
FIGURE_ROOT = REPORT_ROOT / "figures"


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    baseline_metrics = pd.read_csv(BASELINE_ROOT / "metrics.csv").set_index("method")
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for multiplier in MULTIPLIERS:
        root = _result_root(multiplier)
        returns = pd.read_parquet(root / "returns.parquet")
        validation = pd.read_parquet(root / "validation_selection.parquet")
        reported = pd.read_csv(root / "metrics.csv").set_index("method")
        recomputed = performance_metrics(returns, gamma=10.0).set_index("method")
        max_metric_error = _max_metric_error(reported, recomputed)
        if max_metric_error > 1.0e-6:
            raise RuntimeError(
                f"AUM multiplier={multiplier:g} metric reconciliation failed: "
                f"{max_metric_error}"
            )
        quality = _quality(returns, validation)
        quality_rows.append({"aum_multiplier": multiplier, **quality})
        if not quality["passed"]:
            raise RuntimeError(
                f"AUM multiplier={multiplier:g} quality gate failed: {quality}"
            )
        calibration = risk_calibration_report(returns)
        calibration = calibration[calibration["scope"].eq("full")].set_index("method")
        selections = _selection_summary(root)
        for method in METHODS:
            values = recomputed.loc[method]
            monthly = returns[returns["method"].eq(method)]
            rows.append(
                {
                    "aum_multiplier": multiplier,
                    "aum_usd": BASELINE_AUM * multiplier,
                    "method": method,
                    "method_label": DISPLAY[method],
                    "source": (
                        "locked_baseline_reused"
                        if multiplier == 1
                        else "endogenous_recomputation"
                    ),
                    "annualized_gross_return": values["annualized_gross_return"],
                    "annualized_net_return": values["annualized_net_return"],
                    "annualized_volatility": values["annualized_volatility"],
                    "net_sharpe": values["net_sharpe"],
                    "monthly_turnover": values["average_turnover"],
                    "average_leverage": values["average_leverage"],
                    "annualized_trading_cost": values["annualized_trading_cost"],
                    "annualized_utility": values[
                        "annualized_ex_ante_utility_flow"
                    ],
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
    points = pd.DataFrame(rows).sort_values(
        ["method", "aum_multiplier"]
    ).reset_index(drop=True)
    zero_utility = points[points["aum_multiplier"].eq(0)].set_index("method")[
        "annualized_utility"
    ]
    points["utility_degradation_from_zero_cost"] = points.apply(
        lambda row: zero_utility.loc[row["method"]] - row["annualized_utility"],
        axis=1,
    )
    _validate_baseline(points, baseline_metrics)
    points.to_csv(REPORT_ROOT / "aum_sensitivity_points.csv", index=False)
    points.to_parquet(REPORT_ROOT / "aum_sensitivity_points.parquet", index=False)
    _plot(
        points,
        "annualized_net_return",
        "Annualized net return",
        "aum_net_return.png",
        percent_y=True,
    )
    _plot(
        points,
        "annualized_utility",
        "Annualized utility flow",
        "aum_utility.png",
        percent_y=True,
    )
    _plot(
        points,
        "annualized_trading_cost",
        "Annualized trading cost",
        "aum_trading_cost.png",
        percent_y=True,
    )
    _plot(points, "monthly_turnover", "Average monthly turnover", "aum_turnover.png")
    _plot(points, "average_leverage", "Average gross leverage", "aum_leverage.png")
    _plot(
        points,
        "utility_degradation_from_zero_cost",
        "Utility degradation from zero-cost case",
        "aum_utility_degradation.png",
        percent_y=True,
    )
    report = _report(points, pd.DataFrame(quality_rows))
    (REPORT_ROOT / "aum_sensitivity.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "points": len(points),
                "multipliers": MULTIPLIERS,
                "methods": METHODS,
                "fallback_count": int(points["fallback_count"].sum()),
                "covariance_repair_count": int(
                    points["covariance_repair_count"].sum()
                ),
                "baseline_max_abs_error": float(
                    points.loc[
                        points["aum_multiplier"].eq(1),
                        "metric_reconciliation_max_abs_error",
                    ].max()
                ),
                "output": str(REPORT_ROOT),
            },
            indent=2,
        )
    )
    return 0


def _result_root(multiplier: float) -> Path:
    if multiplier == 1:
        return BASELINE_ROOT
    slug = f"{multiplier:g}".replace(".", "p")
    return PROJECT_ROOT / f"results/final_analysis/aum/multiplier_{slug}"


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
    baseline_points = points[points["aum_multiplier"].eq(1)].set_index("method")
    errors = []
    for point_column, baseline_column in mapping.items():
        errors.append(
            (
                baseline_points.loc[METHODS, point_column].astype(float)
                - baseline.loc[METHODS, baseline_column].astype(float)
            ).abs().max()
        )
    if max(errors) > 1.0e-6:
        raise RuntimeError(f"AUM multiplier=1 does not reproduce baseline: {max(errors)}")


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
    y_values = points[y_column].to_numpy(float) * (100.0 if percent_y else 1.0)
    y_min, y_max = _padded_limits(y_values)
    x_positions = {value: index for index, value in enumerate(MULTIPLIERS)}

    def xy(multiplier: float, y: float) -> tuple[int, int]:
        px = left + x_positions[multiplier] / (len(MULTIPLIERS) - 1) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return int(px), int(py)

    grid_color = "#d9dde3"
    axis_color = "#34383e"
    for value in MULTIPLIERS:
        px, _ = xy(value, y_min)
        draw.line((px, top, px, bottom), fill=grid_color, width=2)
        text = f"{value:g}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text(
            (px - (box[2] - box[0]) / 2, bottom + 18),
            text,
            fill=axis_color,
            font=tick_font,
        )
    for value in np.linspace(y_min, y_max, 6):
        _, py = xy(0.0, float(value))
        draw.line((left, py, right, py), fill=grid_color, width=2)
        text = f"{value:.2f}" if percent_y else f"{value:.3f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text(
            (left - 20 - (box[2] - box[0]), py - 13),
            text,
            fill=axis_color,
            font=tick_font,
        )
    draw.line((left, top, left, bottom), fill=axis_color, width=3)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=3)

    for method in METHODS:
        group = points[points["method"].eq(method)].sort_values("aum_multiplier")
        x = group["aum_multiplier"].to_numpy(float)
        y = group[y_column].to_numpy(float)
        if percent_y:
            y = 100.0 * y
        coordinates = [xy(float(a), float(b)) for a, b in zip(x, y)]
        draw.line(coordinates, fill=COLORS[method], width=5, joint="curve")
        for px, py in coordinates:
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=COLORS[method])
        baseline = group[group["aum_multiplier"].eq(1)]
        by = float(baseline[y_column].iloc[0]) * (100.0 if percent_y else 1.0)
        px, py = xy(1.0, by)
        draw.polygon(_star_points(px, py, 20, 9), fill=COLORS[method], outline="white")

    title = f"Top500/Core10 AUM sensitivity: {y_label.lower()}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) / 2, 35),
        title,
        fill="#20242a",
        font=title_font,
    )
    x_label = "AUM multiplier relative to locked baseline"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(
        ((left + right - (x_box[2] - x_box[0])) / 2, 1140),
        x_label,
        fill=axis_color,
        font=label_font,
    )
    y_text = y_label + (" (%)" if percent_y else "")
    y_layer = Image.new("RGBA", (760, 60), (255, 255, 255, 0))
    ImageDraw.Draw(y_layer).text((0, 8), y_text, fill=axis_color, font=label_font)
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
    note = "Stars mark the locked AUM multiplier=1 baseline"
    note_box = draw.textbbox((0, 0), note, font=note_font)
    draw.text(
        (right - (note_box[2] - note_box[0]), 1195),
        note,
        fill="#50545a",
        font=note_font,
    )
    image.save(FIGURE_ROOT / filename, format="PNG", optimize=True)


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = high - low
    padding = 0.12 * span if span > 0 else max(abs(low) * 0.12, 0.1)
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
    pivot = points.pivot(index="aum_multiplier", columns="method")
    changes = []
    for method in METHODS:
        zero = points[
            points["method"].eq(method) & points["aum_multiplier"].eq(0)
        ].iloc[0]
        high = points[
            points["method"].eq(method) & points["aum_multiplier"].eq(10)
        ].iloc[0]
        changes.append(
            {
                "method": DISPLAY[method],
                "utility_change_0_to_10": high["annualized_utility"]
                - zero["annualized_utility"],
                "net_return_change_0_to_10": high["annualized_net_return"]
                - zero["annualized_net_return"],
                "turnover_change_0_to_10": high["monthly_turnover"]
                - zero["monthly_turnover"],
                "leverage_change_0_to_10": high["average_leverage"]
                - zero["average_leverage"],
                "cost_at_10": high["annualized_trading_cost"],
            }
        )
    changes_frame = pd.DataFrame(changes)
    most_sensitive = changes_frame.sort_values("utility_change_0_to_10").iloc[0][
        "method"
    ]
    rank = changes_frame[changes_frame["method"].eq(DISPLAY["return_ml_rank"])].iloc[0]
    linear = changes_frame[
        changes_frame["method"].eq(DISPLAY["portfolio_ml_linear"])
    ].iloc[0]
    rf = changes_frame[changes_frame["method"].eq(DISPLAY["portfolio_ml_rf"])].iloc[0]
    baseline = points[points["aum_multiplier"].eq(1)][
        [
            "method_label",
            "annualized_net_return",
            "annualized_utility",
            "monthly_turnover",
            "average_leverage",
            "annualized_trading_cost",
        ]
    ]
    star_advantage = (
        pivot["annualized_utility"]["static_ml_star"]
        - pivot["annualized_utility"]["static_ml"]
    )
    return (
        "# AUM sensitivity\n\n"
        "This is an endogenous AUM sensitivity analysis for the locked "
        "Top500/Core10 specification. AUM is not selected from these results. "
        "The fixed multipliers are 0, 0.25, 0.5, 1, 2, 5 and 10; multiplier 0 "
        "is a frictionless diagnostic and multiplier 1 reuses the locked baseline.\n\n"
        "## Quality gate\n\n"
        f"All {len(MULTIPLIERS)} AUM points passed. Factor fallback, matrix-m "
        "fallback, covariance repair, future-information violations and "
        "validation-cutoff violations are all zero. The multiplier=1 metric "
        "reconciliation error is at most "
        f"{points.loc[points['aum_multiplier'].eq(1), 'metric_reconciliation_max_abs_error'].max():.3g}.\n\n"
        "## Locked baseline\n\n"
        + _markdown(baseline)
        + "\n\n## Change from zero cost to 10x AUM\n\n"
        + _markdown(changes_frame)
        + "\n\n## Interpretation\n\n"
        f"- {most_sensitive} has the largest utility decline from the zero-cost "
        "diagnostic to 10x baseline AUM.\n"
        f"- Return-ML Rank utility changes by {rank['utility_change_0_to_10']:.4%}; "
        "its fixed rank construction cannot endogenously trade less as AUM rises.\n"
        f"- Portfolio-ML Linear utility changes by {linear['utility_change_0_to_10']:.4%} "
        f"and Portfolio-ML RF by {rf['utility_change_0_to_10']:.4%}. Their turnover "
        "and leverage paths show whether the transaction-cost-aware objective "
        "adapts portfolio intensity as AUM increases.\n"
        f"- Static-ML* exceeds Static-ML utility at {int((star_advantage > 0).sum())} "
        f"of {len(MULTIPLIERS)} fixed AUM points; no point is removed or retuned "
        "after observing the sensitivity results.\n"
        "- Linear versus RF robustness is assessed jointly from utility degradation, "
        "turnover, leverage, trading cost and risk calibration, not from one endpoint.\n"
        "- Multiplier 1 remains the baseline because it was locked before this "
        "sensitivity analysis.\n\n"
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
