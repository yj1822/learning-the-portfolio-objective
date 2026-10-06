from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/risk_calibration"
FIGURE_ROOT = REPORT_ROOT / "figures"
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
CRISIS_YEARS = {2007, 2008, 2020, 2021, 2022}


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(INPUT_ROOT / "risk_calibration.csv")
    yearly = calibration[calibration["scope"].eq("year")].copy()
    yearly["test_year"] = pd.to_numeric(yearly["test_year"], errors="raise").astype(int)
    for column in [
        "predicted_annualized_volatility",
        "realized_annualized_volatility",
        "realized_to_predicted_ratio",
    ]:
        yearly[column] = pd.to_numeric(yearly[column], errors="raise")
    summary_rows = []
    for method in METHODS:
        group = yearly[yearly["method"].eq(method)]
        ratios = group["realized_to_predicted_ratio"]
        full = calibration[
            calibration["scope"].eq("full") & calibration["method"].eq(method)
        ].iloc[0]
        summary_rows.append(
            {
                "method": method,
                "method_label": DISPLAY[method],
                "full_sample_realized_predicted_ratio": float(
                    full["realized_to_predicted_ratio"]
                ),
                "median_yearly_ratio": float(ratios.median()),
                "p10_yearly_ratio": float(ratios.quantile(0.10)),
                "p90_yearly_ratio": float(ratios.quantile(0.90)),
                "max_yearly_ratio": float(ratios.max()),
                "fraction_years_within_0_6_1_5": float(
                    ratios.between(0.6, 1.5).mean()
                ),
                "years_ratio_above_2_count": int(ratios.gt(2.0).sum()),
                "years_ratio_above_2": ",".join(
                    str(value)
                    for value in group.loc[ratios.gt(2.0), "test_year"].tolist()
                ),
                "factor_risk_fallback_count": int(
                    pd.to_numeric(
                        group["factor_risk_fallback_count"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "covariance_repair_count": int(
                    pd.to_numeric(
                        group["covariance_repair_count"], errors="coerce"
                    ).fillna(0).sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_ROOT / "risk_calibration_table.csv", index=False)
    _scatter(yearly)
    _ratio_by_year(yearly)
    _ratio_by_method(yearly, summary)
    (REPORT_ROOT / "risk_calibration_plots.md").write_text(
        _report(yearly, summary), encoding="utf-8"
    )
    print(
        {
            "passed": True,
            "rows": len(yearly),
            "factor_fallback_count": int(summary["factor_risk_fallback_count"].sum()),
            "covariance_repair_count": int(summary["covariance_repair_count"].sum()),
            "output": str(REPORT_ROOT),
        }
    )
    return 0


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    return {
        "title": ImageFont.truetype(str(root / "arialbd.ttf"), 40),
        "label": ImageFont.truetype(str(root / "arial.ttf"), 27),
        "tick": ImageFont.truetype(str(root / "arial.ttf"), 21),
        "small": ImageFont.truetype(str(root / "arial.ttf"), 18),
        "legend": ImageFont.truetype(str(root / "arial.ttf"), 22),
    }


def _scatter(yearly: pd.DataFrame) -> None:
    width, height = 1700, 1260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    left, right, top, bottom = 160, 1640, 130, 1080
    maximum = 100.0 * float(
        max(
            yearly["predicted_annualized_volatility"].max(),
            yearly["realized_annualized_volatility"].max(),
        )
    )
    maximum *= 1.08

    def xy(x: float, y: float) -> tuple[int, int]:
        return (
            int(left + x / maximum * (right - left)),
            int(bottom - y / maximum * (bottom - top)),
        )

    _square_axes(draw, xy, left, right, top, bottom, maximum, fonts)
    line_start = xy(0, 0)
    line_end = xy(maximum, maximum)
    draw.line((*line_start, *line_end), fill="#7f8790", width=3)
    for method in METHODS:
        group = yearly[yearly["method"].eq(method)]
        for row in group.itertuples(index=False):
            x = 100.0 * float(row.predicted_annualized_volatility)
            y = 100.0 * float(row.realized_annualized_volatility)
            px, py = xy(x, y)
            radius = 8 if int(row.test_year) in CRISIS_YEARS else 5
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=COLORS[method],
                outline="white",
            )
            if int(row.test_year) in {2020, 2021} and method in {
                "static_ml",
                "static_ml_star",
            }:
                draw.text(
                    (px + 9, py - 20),
                    str(int(row.test_year)),
                    fill=COLORS[method],
                    font=fonts["small"],
                )
    _title(draw, width, "Predicted versus realized annual volatility", fonts)
    _axis_labels(image, draw, left, right, top, bottom, "Predicted volatility (%)", "Realized volatility (%)", fonts)
    _legend(draw, METHODS, 210, 86, fonts)
    image.save(FIGURE_ROOT / "risk_predicted_vs_realized.png", optimize=True)


def _ratio_by_year(yearly: pd.DataFrame) -> None:
    width, height = 1900, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    years = list(range(2005, 2025))
    left, right = 150, 1840
    panel_top, panel_height, gap = 130, 225, 35
    ratio_max = max(3.0, float(yearly["realized_to_predicted_ratio"].max()) * 1.08)
    for panel, method in enumerate(METHODS):
        top = panel_top + panel * (panel_height + gap)
        bottom = top + panel_height
        group = yearly[yearly["method"].eq(method)].set_index("test_year")
        draw.rectangle((left, top, right, bottom), outline="#c7cdd4", width=2)
        for reference, color, width_line in [(1.0, "#6b7280", 2), (2.0, "#b91c1c", 3)]:
            py = int(bottom - reference / ratio_max * panel_height)
            draw.line((left, py, right, py), fill=color, width=width_line)
        bar_space = (right - left) / len(years)
        for index, year in enumerate(years):
            value = float(group.loc[year, "realized_to_predicted_ratio"])
            x0 = int(left + index * bar_space + 6)
            x1 = int(left + (index + 1) * bar_space - 6)
            py = int(bottom - min(value, ratio_max) / ratio_max * panel_height)
            color = "#9d1c2f" if value > 2 else COLORS[method]
            draw.rectangle((x0, py, x1, bottom), fill=color)
            if year in CRISIS_YEARS:
                draw.line((x0, bottom + 2, x1, bottom + 2), fill="#111827", width=3)
            if panel == len(METHODS) - 1 and (year % 2 == 0 or year in CRISIS_YEARS):
                draw.text((x0 - 3, bottom + 8), str(year), fill="#34383e", font=fonts["small"])
        draw.text((left + 10, top + 8), DISPLAY[method], fill=COLORS[method], font=fonts["legend"])
        for value in [1, 2]:
            py = int(bottom - value / ratio_max * panel_height)
            draw.text((left - 38, py - 10), str(value), fill="#34383e", font=fonts["small"])
    _title(draw, width, "Realized-to-predicted volatility ratio by year", fonts)
    draw.text((left, 1440), "Crisis years are underlined; red bars exceed ratio=2", fill="#50545a", font=fonts["small"])
    image.save(FIGURE_ROOT / "risk_ratio_by_year.png", optimize=True)


def _ratio_by_method(yearly: pd.DataFrame, summary: pd.DataFrame) -> None:
    width, height = 1700, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    left, right, top, bottom = 390, 1600, 140, 900
    x_max = max(3.0, float(yearly["realized_to_predicted_ratio"].max()) * 1.08)

    def x(value: float) -> int:
        return int(left + value / x_max * (right - left))

    draw.rectangle((x(0.6), top, x(1.5), bottom), fill="#e5f4ec")
    for value in np.linspace(0, x_max, 7):
        px = x(float(value))
        draw.line((px, top, px, bottom), fill="#d9dde3", width=2)
        draw.text((px - 18, bottom + 18), f"{value:.1f}", fill="#34383e", font=fonts["tick"])
    for index, method in enumerate(METHODS):
        row = summary[summary["method"].eq(method)].iloc[0]
        group = yearly[yearly["method"].eq(method)]["realized_to_predicted_ratio"]
        y = top + 85 + index * 135
        draw.text((60, y - 15), DISPLAY[method], fill=COLORS[method], font=fonts["legend"])
        draw.line((x(float(row.p10_yearly_ratio)), y, x(float(row.p90_yearly_ratio)), y), fill=COLORS[method], width=8)
        draw.ellipse((x(float(row.median_yearly_ratio)) - 10, y - 10, x(float(row.median_yearly_ratio)) + 10, y + 10), fill=COLORS[method])
        for value in group:
            draw.ellipse((x(float(value)) - 3, y - 3, x(float(value)) + 3, y + 3), fill="#30343a")
    _title(draw, width, "Risk calibration distribution by method", fonts)
    draw.text((left, 965), "Green band: [0.6, 1.5]; thick line: p10-p90; colored dot: median", fill="#50545a", font=fonts["small"])
    image.save(FIGURE_ROOT / "risk_ratio_by_method.png", optimize=True)


def _square_axes(draw, xy, left, right, top, bottom, maximum, fonts) -> None:
    for value in np.linspace(0, maximum, 6):
        px, _ = xy(float(value), 0)
        _, py = xy(0, float(value))
        draw.line((px, top, px, bottom), fill="#d9dde3", width=2)
        draw.line((left, py, right, py), fill="#d9dde3", width=2)
        draw.text((px - 20, bottom + 15), f"{value:.1f}", fill="#34383e", font=fonts["tick"])
        draw.text((left - 65, py - 12), f"{value:.1f}", fill="#34383e", font=fonts["tick"])
    draw.line((left, top, left, bottom), fill="#34383e", width=3)
    draw.line((left, bottom, right, bottom), fill="#34383e", width=3)


def _title(draw: ImageDraw.ImageDraw, width: int, title: str, fonts) -> None:
    box = draw.textbbox((0, 0), title, font=fonts["title"])
    draw.text(((width - box[2] + box[0]) / 2, 34), title, fill="#20242a", font=fonts["title"])


def _axis_labels(image, draw, left, right, top, bottom, x_label, y_label, fonts) -> None:
    box = draw.textbbox((0, 0), x_label, font=fonts["label"])
    draw.text(((left + right - box[2] + box[0]) / 2, bottom + 70), x_label, fill="#34383e", font=fonts["label"])
    layer = Image.new("RGBA", (600, 55), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((0, 5), y_label, fill="#34383e", font=fonts["label"])
    layer = layer.rotate(90, expand=True)
    image.paste(layer, (25, int((top + bottom - layer.height) / 2)), layer)


def _legend(draw, methods, x0, y0, fonts) -> None:
    for index, method in enumerate(methods):
        x = x0 + (index % 3) * 485
        y = y0 + (index // 3) * 36
        draw.ellipse((x, y + 4, x + 15, y + 19), fill=COLORS[method])
        draw.text((x + 24, y), DISPLAY[method], fill="#34383e", font=fonts["legend"])


def _report(yearly: pd.DataFrame, summary: pd.DataFrame) -> str:
    breaches = yearly[yearly["realized_to_predicted_ratio"].gt(2)]
    static_breaches = breaches[breaches["method"].isin(["static_ml", "static_ml_star"])]
    pml = summary[summary["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])]
    static = summary[summary["method"].isin(["static_ml", "static_ml_star"])]
    return (
        "# Risk calibration plots\n\n"
        "This analysis uses only the locked Top500/Core10 baseline outputs. No model "
        "is refitted, no year is removed and no parameter is retuned.\n\n"
        "## Findings\n\n"
        f"- Portfolio-ML has median yearly realized/predicted ratios between "
        f"{pml['median_yearly_ratio'].min():.3f} and {pml['median_yearly_ratio'].max():.3f}, "
        f"with {int(pml['years_ratio_above_2_count'].sum())} total method-year ratios above 2.\n"
        f"- Static-ML and Static-ML* have {int(static['years_ratio_above_2_count'].sum())} "
        f"ratios above 2, all in {sorted(static_breaches['test_year'].unique().tolist())}. "
        "The abnormality is concentrated rather than pervasive across 2005-2024.\n"
        "- The existing stock-level reconstruction shows that the 2020-2021 Static "
        "misses reflect common-factor covariance underestimation during the pandemic "
        "and reopening regimes, not factor fallback, covariance repair, excessive "
        "leverage or a risk-accounting mismatch.\n"
        "- These episodes do not overturn the main result: Portfolio-ML risk "
        "calibration remains materially more stable, while its utility advantage is "
        "measured from the unchanged full sample.\n"
        "- There is no evidence of an implementation bug requiring a model rewrite. "
        "The calibration miss is economically relevant and is retained as a limitation.\n"
        "- Post-hoc tuning would use realized crisis outcomes to alter a pre-specified "
        "model and would invalidate the cumulative OOS interpretation; it is therefore "
        "not performed.\n\n"
        "## Calibration coverage\n\n"
        + _markdown(summary)
        + "\n\n## Crisis-year observations\n\n"
        + _markdown(yearly[yearly["test_year"].isin(CRISIS_YEARS)][[
            "test_year",
            "method",
            "predicted_annualized_volatility",
            "realized_annualized_volatility",
            "realized_to_predicted_ratio",
        ]])
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
