from __future__ import annotations

import os

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"
DATA_ROOT = PROJECT_ROOT / "data/processed/us_equity_ml"
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis/mechanisms"
FIGURE_ROOT = REPORT_ROOT / "figures"
BASELINE_AUM = 1_000_000_000.0
METHODS = [
    "return_ml_rank",
    "static_ml",
    "static_ml_star",
    "portfolio_ml_linear",
    "portfolio_ml_rf",
]
PML_METHODS = ["portfolio_ml_linear", "portfolio_ml_rf"]
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


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    returns = _load_returns()
    weights = _load_stock_data()
    stock_monthly, reconciliation = _stock_mechanisms(weights)
    monthly = _monthly_metrics(returns, stock_monthly)
    annual = _annual_metrics(monthly)
    output = _combined_output(monthly, annual)
    output.to_csv(REPORT_ROOT / "mechanism_metrics.csv", index=False)

    _plot_cumulative(monthly)
    _grouped_year_chart(annual, "annualized_net_return", "Annual net return", "annual_net_return.png", True)
    _grouped_year_chart(
        annual,
        "annualized_ex_ante_utility_flow",
        "Annual utility flow",
        "annual_utility.png",
        True,
    )
    _line_chart(monthly, "turnover", "Monthly turnover", "monthly_turnover.png")
    _grouped_year_chart(annual, "average_turnover", "Average monthly turnover", "annual_turnover.png")
    _trading_cost_time_series(monthly)
    _annual_trading_cost_chart(annual)
    _line_chart(monthly, "leverage", "Gross leverage", "monthly_leverage.png")
    _grouped_year_chart(annual, "average_leverage", "Average gross leverage", "annual_leverage.png")
    _line_chart(monthly, "net_drawdown", "Net-return drawdown", "net_drawdown.png", True)
    _grouped_year_chart(annual, "robust_cost_to_gross_ratio", "Cost / absolute gross return", "cost_to_gross_return.png", True, cap=5.0)
    _cost_concentration_chart(monthly)
    _liquidity_chart(monthly)
    _aim_adjustment_chart(annual)

    report = _report(monthly, annual, reconciliation)
    (REPORT_ROOT / "mechanism_plots.md").write_text(report, encoding="utf-8")
    manifest = {
        "passed": True,
        "specification": "Top500/Core10 locked baseline",
        "model_refit": False,
        "monthly_rows": int(len(monthly)),
        "annual_rows": int(len(annual)),
        "stock_rows": int(len(weights)),
        "max_stock_cost_reconciliation_error": reconciliation["max_abs_error"],
        "factor_fallback_count": int(returns["factor_fallback"].sum()),
        "matrix_m_fallback_count": int(returns["matrix_m_fallback"].sum()),
        "covariance_repair_count": int(returns["covariance_repair_count"].sum()),
        "figures": sorted(path.name for path in FIGURE_ROOT.glob("*.png")),
    }
    (REPORT_ROOT / "mechanism_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _load_returns() -> pd.DataFrame:
    returns = pd.read_parquet(INPUT_ROOT / "returns.parquet")
    returns["eom"] = pd.to_datetime(returns["eom"])
    returns = returns[returns["method"].isin(METHODS)].copy()
    counts = returns.groupby("method")["eom"].nunique()
    if counts.reindex(METHODS).isna().any() or not counts.eq(240).all():
        raise RuntimeError(f"Expected 240 months per method, found {counts.to_dict()}")
    if returns.duplicated(["eom", "method"]).any():
        raise RuntimeError("Duplicate method-month rows in baseline returns")
    if not returns["risk_model_used"].eq("factor").all():
        raise RuntimeError("Non-factor risk row found in locked baseline")
    returns["factor_fallback"] = returns["risk_model_used"].ne("factor").astype(int)
    returns["matrix_m_fallback"] = returns["adjustment_mode"].eq("scalar_rho").astype(int)
    if returns["factor_fallback"].sum() or returns["matrix_m_fallback"].sum():
        raise RuntimeError("Fallback found in locked baseline")
    if pd.to_numeric(returns["covariance_repair_count"], errors="raise").sum():
        raise RuntimeError("Covariance repair found in locked baseline")
    return returns.sort_values(["method", "eom"]).reset_index(drop=True)


def _load_stock_data() -> pd.DataFrame:
    weights = pd.read_parquet(INPUT_ROOT / "weights.parquet")
    costs = pd.read_parquet(
        DATA_ROOT / "cost_inputs.parquet",
        columns=["eom", "permno", "adv_6m", "lambda_adv", "in_top500"],
    )
    weights["eom"] = pd.to_datetime(weights["eom"])
    costs["eom"] = pd.to_datetime(costs["eom"])
    costs = costs[costs["in_top500"].fillna(False)].copy()
    costs["liquidity_decile"] = costs.groupby("eom")["adv_6m"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 10, labels=False, duplicates="drop"
        )
        + 1
    )
    weights = weights[weights["method"].isin(METHODS)].merge(
        costs.drop(columns="in_top500"),
        on=["eom", "permno"],
        how="left",
        validate="many_to_one",
    )
    required = ["weight", "inherited_weight", "trade", "adv_6m", "lambda_adv", "liquidity_decile"]
    if weights[required].isna().any().any():
        missing = weights[required].isna().sum()
        raise RuntimeError(f"Missing stock mechanism inputs: {missing[missing.gt(0)].to_dict()}")
    identity_error = (weights["trade"] - (weights["weight"] - weights["inherited_weight"])).abs().max()
    if float(identity_error) > 1.0e-12:
        raise RuntimeError(f"Trade identity failed: {identity_error}")
    weights["abs_weight"] = weights["weight"].abs()
    weights["abs_trade"] = weights["trade"].abs()
    weights["stock_cost"] = 0.5 * BASELINE_AUM * weights["lambda_adv"] * weights["trade"].pow(2)
    weights["aim_delta"] = weights["aim_weight"] - weights["inherited_weight"]
    weights["aim_cost"] = 0.5 * BASELINE_AUM * weights["lambda_adv"] * weights["aim_delta"].pow(2)
    return weights


def _stock_mechanisms(weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for (eom, method), group in weights.groupby(["eom", "method"], sort=True):
        gross = float(group["abs_weight"].sum())
        within_cost = float(group["stock_cost"].sum())
        n_top = max(1, int(math.ceil(0.10 * len(group))))
        top_cost = float(group.nlargest(n_top, "stock_cost")["stock_cost"].sum())
        top_trade_cost = float(group.nlargest(n_top, "abs_trade")["stock_cost"].sum())
        row: dict[str, float | str | pd.Timestamp] = {
            "eom": eom,
            "method": method,
            "reconstructed_within_universe_cost": within_cost,
            "gross_weight_for_liquidity": gross,
            "portfolio_weighted_adv": _weighted_average(group["adv_6m"], group["abs_weight"]),
            "portfolio_weighted_lambda_adv": _weighted_average(group["lambda_adv"], group["abs_weight"]),
            "low_adv_decile_exposure": _weight_share(group, 1),
            "high_adv_decile_exposure": _weight_share(group, 10),
            "top_10pct_names_cost_share": top_cost / within_cost if within_cost > 0 else np.nan,
            "top_10pct_abs_trades_cost_share": top_trade_cost / within_cost if within_cost > 0 else np.nan,
            "actual_trade_l1": float(group["abs_trade"].sum()),
            "actual_trade_l2": float(np.sqrt(np.square(group["trade"]).sum())),
            "lambda_weighted_actual_distance": float(np.sqrt((group["lambda_adv"] * group["trade"].pow(2)).sum())),
            "aim_trade_l1": np.nan,
            "lambda_weighted_aim_distance": np.nan,
            "actual_to_aim_l1_ratio": np.nan,
            "actual_to_aim_lambda_distance_ratio": np.nan,
            "immediate_aim_cost": np.nan,
        }
        if method in PML_METHODS:
            if group["aim_weight"].isna().any():
                raise RuntimeError(f"Missing Portfolio-ML aim weights for {method} {eom}")
            aim_l1 = float(group["aim_delta"].abs().sum())
            aim_lambda = float(np.sqrt((group["lambda_adv"] * group["aim_delta"].pow(2)).sum()))
            row.update(
                {
                    "aim_trade_l1": aim_l1,
                    "lambda_weighted_aim_distance": aim_lambda,
                    "actual_to_aim_l1_ratio": row["actual_trade_l1"] / aim_l1 if aim_l1 > 0 else np.nan,
                    "actual_to_aim_lambda_distance_ratio": row["lambda_weighted_actual_distance"] / aim_lambda if aim_lambda > 0 else np.nan,
                    "immediate_aim_cost": float(group["aim_cost"].sum()),
                }
            )
        rows.append(row)
    stock_monthly = pd.DataFrame(rows)
    reported = pd.read_parquet(
        INPUT_ROOT / "returns.parquet",
        columns=["eom", "method", "trading_cost", "exit_cost"],
    )
    reported["eom"] = pd.to_datetime(reported["eom"])
    check = stock_monthly.merge(reported, on=["eom", "method"], validate="one_to_one")
    check["target"] = check["trading_cost"] - check["exit_cost"]
    check["error"] = check["reconstructed_within_universe_cost"] - check["target"]
    max_error = float(check["error"].abs().max())
    if max_error > 1.0e-12:
        raise RuntimeError(f"Stock-level transaction-cost reconciliation failed: {max_error}")
    return stock_monthly, {
        "max_abs_error": max_error,
        "mean_abs_error": float(check["error"].abs().mean()),
    }


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    return float(np.average(values, weights=weights)) if total > 0 else np.nan


def _weight_share(group: pd.DataFrame, decile: int) -> float:
    total = float(group["abs_weight"].sum())
    if total <= 0:
        return np.nan
    return float(group.loc[group["liquidity_decile"].eq(decile), "abs_weight"].sum() / total)


def _monthly_metrics(returns: pd.DataFrame, stock_monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = returns.merge(stock_monthly, on=["eom", "method"], validate="one_to_one")
    monthly["exit_cost_share"] = np.where(
        monthly["trading_cost"].gt(0), monthly["exit_cost"] / monthly["trading_cost"], np.nan
    )
    monthly["stock_cost_coverage"] = np.where(
        monthly["trading_cost"].gt(0),
        monthly["reconstructed_within_universe_cost"] / monthly["trading_cost"],
        np.nan,
    )
    monthly["year"] = monthly["eom"].dt.year
    monthly["near_zero_monthly_gross_return"] = monthly["gross_return"].abs().lt(1.0e-4)
    monthly["robust_monthly_cost_to_gross_ratio"] = monthly["trading_cost"] / monthly["gross_return"].abs().clip(lower=1.0e-4)
    pieces = []
    for method, group in monthly.groupby("method", sort=False):
        group = group.sort_values("eom").copy()
        group["cumulative_gross_return"] = (1.0 + group["gross_return"]).cumprod() - 1.0
        group["cumulative_net_return"] = (1.0 + group["net_return"]).cumprod() - 1.0
        net_wealth = 1.0 + group["cumulative_net_return"]
        group["net_drawdown"] = net_wealth / net_wealth.cummax() - 1.0
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["method", "eom"])


def _annual_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    annual_reported = pd.read_csv(INPUT_ROOT / "annual_performance.csv").rename(columns={"test_year": "year"})
    annual_reported = annual_reported[annual_reported["method"].isin(METHODS)].copy()
    extra_columns = [
        "portfolio_weighted_adv",
        "portfolio_weighted_lambda_adv",
        "low_adv_decile_exposure",
        "high_adv_decile_exposure",
        "top_10pct_names_cost_share",
        "top_10pct_abs_trades_cost_share",
        "exit_cost_share",
        "actual_trade_l1",
        "aim_trade_l1",
        "actual_to_aim_l1_ratio",
        "lambda_weighted_actual_distance",
        "lambda_weighted_aim_distance",
        "actual_to_aim_lambda_distance_ratio",
        "immediate_aim_cost",
    ]
    extras = monthly.groupby(["method", "year"])[extra_columns].mean().reset_index()
    annual = annual_reported.merge(extras, on=["method", "year"], validate="one_to_one")
    annual["near_zero_annual_gross_return"] = annual["annualized_gross_return"].abs().lt(0.005)
    annual["robust_cost_to_gross_ratio"] = annual["annualized_trading_cost"] / annual["annualized_gross_return"].abs().clip(lower=0.005)
    return annual.sort_values(["method", "year"]).reset_index(drop=True)


def _combined_output(monthly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    monthly_out = monthly.copy()
    monthly_out.insert(0, "scope", "month")
    annual_out = annual.copy()
    annual_out.insert(0, "scope", "year")
    annual_out["eom"] = pd.NaT
    columns = list(dict.fromkeys(list(monthly_out.columns) + list(annual_out.columns)))
    return pd.concat(
        [monthly_out.reindex(columns=columns), annual_out.reindex(columns=columns)],
        ignore_index=True,
    )


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    root = Path(os.environ.get("IEF_FONT_ROOT", "."))
    return {
        "title": ImageFont.truetype(str(root / "arialbd.ttf"), 38),
        "label": ImageFont.truetype(str(root / "arial.ttf"), 27),
        "tick": ImageFont.truetype(str(root / "arial.ttf"), 20),
        "small": ImageFont.truetype(str(root / "arial.ttf"), 17),
        "legend": ImageFont.truetype(str(root / "arial.ttf"), 21),
    }


def _plot_cumulative(monthly: pd.DataFrame) -> None:
    width, height = 1900, 1420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, "Top500/Core10 cumulative gross and net returns", fonts)
    for panel, (column, label) in enumerate(
        [("cumulative_gross_return", "Gross cumulative return"), ("cumulative_net_return", "Net cumulative return")]
    ):
        top = 175 + panel * 585
        bottom = top + 450
        _draw_line_panel(image, draw, monthly, column, label, 170, 1840, top, bottom, fonts, True)
    _legend(draw, METHODS, 230, 92, fonts)
    image.save(FIGURE_ROOT / "cumulative_gross_net_returns.png", optimize=True)


def _line_chart(
    data: pd.DataFrame,
    column: str,
    label: str,
    filename: str,
    percent_y: bool = False,
) -> None:
    width, height = 1900, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, f"Top500/Core10: {label.lower()}", fonts)
    _draw_line_panel(image, draw, data, column, label, 170, 1840, 180, 930, fonts, percent_y)
    _legend(draw, METHODS, 230, 90, fonts)
    image.save(FIGURE_ROOT / filename, optimize=True)


def _draw_line_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    column: str,
    label: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    percent_y: bool,
    methods: list[str] | None = None,
) -> None:
    methods = METHODS if methods is None else methods
    data = data[data["method"].isin(methods)]
    dates = sorted(pd.to_datetime(data["eom"].dropna().unique()))
    values = pd.to_numeric(data[column], errors="coerce").dropna().to_numpy(float)
    if percent_y:
        values = 100.0 * values
    y_min, y_max = _padded_limits(values, include_zero=True)

    def xy(date: pd.Timestamp, value: float) -> tuple[int, int]:
        index = dates.index(pd.Timestamp(date))
        px = left + index / max(1, len(dates) - 1) * (right - left)
        py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(px), int(py)

    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, percent_y)
    tick_dates = [date for date in dates if date.month == 12 and date.year in {2005, 2010, 2015, 2020, 2024}]
    for date in tick_dates:
        px, _ = xy(date, y_min)
        draw.line((px, bottom, px, bottom + 7), fill="#34383e", width=2)
        text = str(date.year)
        box = draw.textbbox((0, 0), text, font=fonts["tick"])
        draw.text((px - (box[2] - box[0]) / 2, bottom + 12), text, fill="#34383e", font=fonts["tick"])
    for method in methods:
        group = data[data["method"].eq(method)].sort_values("eom")
        coordinates = []
        for row in group.itertuples(index=False):
            value = float(getattr(row, column)) * (100.0 if percent_y else 1.0)
            coordinates.append(xy(pd.Timestamp(row.eom), value))
        draw.line(coordinates, fill=COLORS[method], width=4, joint="curve")
    _axis_labels(image, draw, left, right, top, bottom, "Month", label + (" (%)" if percent_y else ""), fonts)


def _trading_cost_time_series(monthly: pd.DataFrame) -> None:
    width, height = 1900, 1420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, "Top500/Core10 monthly trading cost", fonts)
    _draw_line_panel(
        image,
        draw,
        monthly,
        "trading_cost",
        "All methods: monthly trading cost",
        170,
        1840,
        175,
        620,
        fonts,
        True,
        METHODS,
    )
    non_rank = [method for method in METHODS if method != "return_ml_rank"]
    _draw_line_panel(
        image,
        draw,
        monthly,
        "trading_cost",
        "Static and Portfolio-ML: monthly trading cost",
        170,
        1840,
        785,
        1230,
        fonts,
        True,
        non_rank,
    )
    _legend(draw, METHODS, 230, 90, fonts)
    image.save(FIGURE_ROOT / "monthly_trading_cost.png", optimize=True)


def _annual_trading_cost_chart(annual: pd.DataFrame) -> None:
    width, height = 1960, 1420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, "Top500/Core10 annualized trading cost by year", fonts)
    _draw_grouped_year_panel(
        draw,
        annual,
        "annualized_trading_cost",
        "All methods",
        170,
        1900,
        175,
        620,
        fonts,
        METHODS,
        True,
    )
    non_rank = [method for method in METHODS if method != "return_ml_rank"]
    _draw_grouped_year_panel(
        draw,
        annual,
        "annualized_trading_cost",
        "Static and Portfolio-ML",
        170,
        1900,
        785,
        1230,
        fonts,
        non_rank,
        True,
    )
    _legend(draw, METHODS, 250, 92, fonts)
    image.save(FIGURE_ROOT / "annual_trading_cost.png", optimize=True)


def _draw_grouped_year_panel(
    draw: ImageDraw.ImageDraw,
    annual: pd.DataFrame,
    column: str,
    panel_label: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    methods: list[str],
    percent_y: bool,
) -> None:
    data = annual[annual["method"].isin(methods)]
    years = sorted(data["year"].unique())
    values = data[column].to_numpy(float) * (100.0 if percent_y else 1.0)
    y_min, y_max = _padded_limits(values, include_zero=True)
    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, percent_y)
    group_width = (right - left) / len(years)
    bar_width = max(4, int(group_width * 0.72 / len(methods)))
    for year_index, year in enumerate(years):
        center = left + (year_index + 0.5) * group_width
        for method_index, method in enumerate(methods):
            row = data[data["method"].eq(method) & data["year"].eq(year)].iloc[0]
            value = float(row[column]) * (100.0 if percent_y else 1.0)
            py = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
            x0 = int(center + (method_index - len(methods) / 2) * bar_width)
            draw.rectangle((x0, py, x0 + bar_width - 1, bottom), fill=COLORS[method])
        if year % 2 == 0 or year in {2005, 2024}:
            text = str(year)
            box = draw.textbbox((0, 0), text, font=fonts["small"])
            draw.text((center - (box[2] - box[0]) / 2, bottom + 10), text, fill="#34383e", font=fonts["small"])
    draw.text((left + 10, top + 8), panel_label, fill="#34383e", font=fonts["legend"])


def _grouped_year_chart(
    annual: pd.DataFrame,
    column: str,
    label: str,
    filename: str,
    percent_y: bool = False,
    cap: float | None = None,
) -> None:
    width, height = 1960, 1160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    left, right, top, bottom = 170, 1900, 180, 970
    years = sorted(annual["year"].unique())
    raw = annual[column].to_numpy(float)
    plot_values = np.clip(raw, None, cap) if cap is not None else raw
    if percent_y:
        plot_values = 100.0 * plot_values
    y_min, y_max = _padded_limits(plot_values, include_zero=True)
    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, percent_y)
    group_width = (right - left) / len(years)
    bar_width = max(4, int(group_width * 0.14))
    zero_y = int(bottom - (0.0 - y_min) / (y_max - y_min) * (bottom - top))
    for year_index, year in enumerate(years):
        center = left + (year_index + 0.5) * group_width
        for method_index, method in enumerate(METHODS):
            row = annual[annual["method"].eq(method) & annual["year"].eq(year)].iloc[0]
            value = float(row[column])
            clipped = cap is not None and value > cap
            if cap is not None:
                value = min(value, cap)
            value *= 100.0 if percent_y else 1.0
            py = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
            x0 = int(center + (method_index - 2.5) * bar_width)
            x1 = x0 + bar_width - 1
            draw.rectangle((x0, min(py, zero_y), x1, max(py, zero_y)), fill=COLORS[method])
            if clipped:
                draw.line((x0, py + 5, x1, py - 5), fill="white", width=2)
        if year % 2 == 0 or year in {2005, 2024}:
            text = str(year)
            box = draw.textbbox((0, 0), text, font=fonts["small"])
            draw.text((center - (box[2] - box[0]) / 2, bottom + 13), text, fill="#34383e", font=fonts["small"])
    title = f"Top500/Core10: {label.lower()} by year"
    _title(draw, width, title, fonts)
    _legend(draw, METHODS, 250, 92, fonts)
    note = "Values above 500% are clipped" if cap is not None else ""
    _axis_labels(image, draw, left, right, top, bottom, "Test year", label + (" (%)" if percent_y else ""), fonts)
    if note:
        draw.text((right - 330, bottom + 65), note, fill="#50545a", font=fonts["small"])
    image.save(FIGURE_ROOT / filename, optimize=True)


def _cost_concentration_chart(monthly: pd.DataFrame) -> None:
    summary = monthly.groupby("method")[["top_10pct_names_cost_share", "top_10pct_abs_trades_cost_share", "exit_cost_share"]].mean()
    labels = ["Top 10% names by cost", "Top 10% trades by size", "Exit costs"]
    columns = list(summary.columns)
    _category_bar_chart(summary, columns, labels, "Transaction-cost concentration", "trade_cost_concentration.png", True)


def _liquidity_chart(monthly: pd.DataFrame) -> None:
    summary = monthly.groupby("method").agg(
        weighted_adv_millions=("portfolio_weighted_adv", lambda x: x.median() / 1.0e6),
        low_adv_decile_exposure=("low_adv_decile_exposure", "mean"),
        high_adv_decile_exposure=("high_adv_decile_exposure", "mean"),
    )
    width, height = 1850, 1240
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, "Top500/Core10 liquidity exposure", fonts)
    _draw_simple_method_bars(draw, summary["weighted_adv_millions"], "Median portfolio-weighted ADV (USD millions)", 170, 1780, 150, 560, fonts, False)
    exposure = summary[["low_adv_decile_exposure", "high_adv_decile_exposure"]]
    _draw_multi_method_bars(draw, exposure, ["Lowest ADV decile", "Highest ADV decile"], "Absolute-weight exposure (%)", 170, 1780, 720, 1100, fonts, True)
    image.save(FIGURE_ROOT / "liquidity_exposure.png", optimize=True)


def _aim_adjustment_chart(annual: pd.DataFrame) -> None:
    data = annual[annual["method"].isin(PML_METHODS)].copy()
    width, height = 1900, 1370
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, "Portfolio-ML inherited-position and aim adjustment", fonts)
    for panel, (column, label) in enumerate(
        [
            ("actual_to_aim_l1_ratio", "Actual / aim L1 adjustment ratio"),
            ("actual_to_aim_lambda_distance_ratio", "Actual / aim Lambda-distance ratio"),
        ]
    ):
        top = 155 + panel * 570
        bottom = top + 455
        _draw_annual_line_panel(draw, data, column, label, 170, 1840, top, bottom, fonts, PML_METHODS)
    _legend(draw, PML_METHODS, 560, 95, fonts)
    image.save(FIGURE_ROOT / "inherited_aim_adjustment.png", optimize=True)


def _category_bar_chart(
    summary: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    filename: str,
    percent_y: bool,
) -> None:
    width, height = 1850, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    _title(draw, width, f"Top500/Core10: {title.lower()}", fonts)
    _draw_multi_method_bars(draw, summary[columns], labels, title + (" (%)" if percent_y else ""), 170, 1780, 190, 880, fonts, percent_y)
    image.save(FIGURE_ROOT / filename, optimize=True)


def _draw_multi_method_bars(
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    labels: list[str],
    y_label: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    percent_y: bool,
) -> None:
    values = data.to_numpy(float) * (100.0 if percent_y else 1.0)
    y_min, y_max = _padded_limits(values.ravel(), include_zero=True)
    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, percent_y)
    category_width = (right - left) / len(data.columns)
    bar_width = max(15, int(category_width / (len(METHODS) + 2)))
    for category_index, column in enumerate(data.columns):
        center = left + (category_index + 0.5) * category_width
        for method_index, method in enumerate(METHODS):
            value = float(data.loc[method, column]) * (100.0 if percent_y else 1.0)
            py = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
            x0 = int(center + (method_index - 2.5) * bar_width)
            draw.rectangle((x0, py, x0 + bar_width - 3, bottom), fill=COLORS[method])
        label = labels[category_index]
        box = draw.textbbox((0, 0), label, font=fonts["tick"])
        draw.text((center - (box[2] - box[0]) / 2, bottom + 15), label, fill="#34383e", font=fonts["tick"])
    _legend(draw, METHODS, 230, top - 60, fonts)
    _rotated_label(draw._image, y_label, 35, int((top + bottom) / 2), fonts["label"])


def _draw_simple_method_bars(
    draw: ImageDraw.ImageDraw,
    values: pd.Series,
    label: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    percent_y: bool,
) -> None:
    scaled = values.to_numpy(float) * (100.0 if percent_y else 1.0)
    y_min, y_max = _padded_limits(scaled, include_zero=True)
    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, percent_y)
    width = (right - left) / len(METHODS)
    for index, method in enumerate(METHODS):
        value = float(values.loc[method]) * (100.0 if percent_y else 1.0)
        py = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
        x0 = int(left + index * width + width * 0.20)
        x1 = int(left + (index + 1) * width - width * 0.20)
        draw.rectangle((x0, py, x1, bottom), fill=COLORS[method])
        text = DISPLAY[method].replace("Portfolio-ML ", "PML ").replace("Return-ML ", "Return ")
        box = draw.textbbox((0, 0), text, font=fonts["small"])
        draw.text(((x0 + x1 - (box[2] - box[0])) / 2, bottom + 12), text, fill="#34383e", font=fonts["small"])
    _rotated_label(draw._image, label, 35, int((top + bottom) / 2), fonts["label"])


def _draw_annual_line_panel(
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    column: str,
    label: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
    methods: list[str],
) -> None:
    years = sorted(data["year"].unique())
    values = data[column].dropna().to_numpy(float)
    y_min, y_max = _padded_limits(values, include_zero=True)
    _draw_axes(draw, left, right, top, bottom, y_min, y_max, fonts, False)

    def xy(year: int, value: float) -> tuple[int, int]:
        px = left + years.index(year) / (len(years) - 1) * (right - left)
        py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return int(px), int(py)

    for method in methods:
        group = data[data["method"].eq(method)].sort_values("year")
        coordinates = [xy(int(row.year), float(getattr(row, column))) for row in group.itertuples(index=False)]
        draw.line(coordinates, fill=COLORS[method], width=5, joint="curve")
        for px, py in coordinates:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLORS[method])
    for year in years:
        if year % 2 == 0 or year in {2005, 2024}:
            px, _ = xy(year, y_min)
            text = str(year)
            box = draw.textbbox((0, 0), text, font=fonts["small"])
            draw.text((px - (box[2] - box[0]) / 2, bottom + 10), text, fill="#34383e", font=fonts["small"])
    draw.text((left + 12, top + 10), label, fill="#34383e", font=fonts["legend"])


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    top: int,
    bottom: int,
    y_min: float,
    y_max: float,
    fonts: dict[str, ImageFont.FreeTypeFont],
    percent: bool,
) -> None:
    for value in np.linspace(y_min, y_max, 6):
        py = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, py, right, py), fill="#d9dde3", width=2)
        if percent:
            decimals = 2 if max(abs(y_min), abs(y_max)) < 1.0 else 1
            text = f"{value:.{decimals}f}"
        else:
            text = f"{value:.3f}"
        box = draw.textbbox((0, 0), text, font=fonts["tick"])
        draw.text((left - 18 - (box[2] - box[0]), py - 11), text, fill="#34383e", font=fonts["tick"])
    if y_min < 0 < y_max:
        py = int(bottom - (0 - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, py, right, py), fill="#737980", width=3)
    draw.line((left, top, left, bottom), fill="#34383e", width=3)
    draw.line((left, bottom, right, bottom), fill="#34383e", width=3)


def _padded_limits(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    observed_low, observed_high = float(finite.min()), float(finite.max())
    low, high = observed_low, observed_high
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    padding = 0.10 * span if span > 0 else max(abs(low) * 0.10, 0.1)
    if include_zero and observed_low >= 0:
        return 0.0, high + padding
    if include_zero and observed_high <= 0:
        return low - padding, 0.0
    return low - padding, high + padding


def _title(draw: ImageDraw.ImageDraw, width: int, text: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    box = draw.textbbox((0, 0), text, font=fonts["title"])
    draw.text(((width - (box[2] - box[0])) / 2, 30), text, fill="#20242a", font=fonts["title"])


def _legend(
    draw: ImageDraw.ImageDraw,
    methods: list[str],
    start_x: int,
    start_y: int,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    for index, method in enumerate(methods):
        column, row = index % 3, index // 3
        x0 = start_x + column * 520
        y0 = start_y + row * 40
        draw.line((x0, y0 + 11, x0 + 42, y0 + 11), fill=COLORS[method], width=5)
        draw.ellipse((x0 + 17, y0 + 3, x0 + 31, y0 + 17), fill=COLORS[method])
        draw.text((x0 + 55, y0), DISPLAY[method], fill="#34383e", font=fonts["legend"])


def _axis_labels(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    left: int,
    right: int,
    top: int,
    bottom: int,
    x_label: str,
    y_label: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> None:
    box = draw.textbbox((0, 0), x_label, font=fonts["label"])
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 58), x_label, fill="#34383e", font=fonts["label"])
    _rotated_label(image, y_label, 35, int((top + bottom) / 2), fonts["label"])


def _rotated_label(image: Image.Image, text: str, x: int, center_y: int, font: ImageFont.FreeTypeFont) -> None:
    layer = Image.new("RGBA", (900, 60), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((0, 8), text, fill="#34383e", font=font)
    layer = layer.crop(layer.getbbox()).rotate(90, expand=True)
    image.paste(layer, (x, int(center_y - layer.height / 2)), layer)


def _report(monthly: pd.DataFrame, annual: pd.DataFrame, reconciliation: dict[str, float]) -> str:
    metrics = pd.read_csv(INPUT_ROOT / "metrics.csv").set_index("method").loc[METHODS]
    mechanism = monthly.groupby("method").agg(
        portfolio_weighted_adv=("portfolio_weighted_adv", "median"),
        portfolio_weighted_lambda_adv=("portfolio_weighted_lambda_adv", "mean"),
        low_adv_decile_exposure=("low_adv_decile_exposure", "mean"),
        high_adv_decile_exposure=("high_adv_decile_exposure", "mean"),
        top_10pct_names_cost_share=("top_10pct_names_cost_share", "mean"),
        top_10pct_abs_trades_cost_share=("top_10pct_abs_trades_cost_share", "mean"),
        exit_cost_share=("exit_cost_share", "mean"),
    )
    table = metrics[[
        "annualized_gross_return",
        "annualized_trading_cost",
        "annualized_net_return",
        "annualized_volatility",
        "average_turnover",
        "average_leverage",
        "annualized_ex_ante_utility_flow",
    ]].join(mechanism)
    table.insert(0, "method_label", [DISPLAY[m] for m in table.index])

    comparisons = []
    for pml in PML_METHODS:
        for benchmark in ["static_ml", "static_ml_star"]:
            gross_difference = metrics.loc[pml, "annualized_gross_return"] - metrics.loc[benchmark, "annualized_gross_return"]
            cost_saving = metrics.loc[benchmark, "annualized_trading_cost"] - metrics.loc[pml, "annualized_trading_cost"]
            net_difference = metrics.loc[pml, "annualized_net_return"] - metrics.loc[benchmark, "annualized_net_return"]
            comparisons.append(
                {
                    "comparison": f"{DISPLAY[pml]} vs {DISPLAY[benchmark]}",
                    "gross_return_difference": gross_difference,
                    "cost_saving": cost_saving,
                    "net_return_difference": net_difference,
                    "cost_saving_share_of_net_difference": cost_saving / net_difference if net_difference else np.nan,
                }
            )
    comparison_table = pd.DataFrame(comparisons)
    pml_aim = monthly[monthly["method"].isin(PML_METHODS)].groupby("method").agg(
        actual_trade_l1=("actual_trade_l1", "mean"),
        aim_trade_l1=("aim_trade_l1", "mean"),
        actual_to_aim_l1_ratio=("actual_to_aim_l1_ratio", "mean"),
        actual_to_aim_lambda_distance_ratio=("actual_to_aim_lambda_distance_ratio", "mean"),
        immediate_aim_cost=("immediate_aim_cost", "mean"),
        actual_within_universe_cost=("reconstructed_within_universe_cost", "mean"),
    )
    pml_aim.insert(0, "method_label", [DISPLAY[m] for m in pml_aim.index])
    rank = metrics.loc["return_ml_rank"]
    rank_cost_ratio = rank["annualized_trading_cost"] / abs(rank["annualized_gross_return"])
    static = metrics.loc["static_ml"]
    static_star = metrics.loc["static_ml_star"]
    near_zero_years = annual.groupby("method")["near_zero_annual_gross_return"].sum().astype(int)

    lines = [
        "# Mechanism plots",
        "",
        "This analysis uses only the locked Top500/Core10 baseline at gamma=10 and baseline AUM. It reads existing OOS returns, weights, inherited positions, Portfolio-ML aim weights, ADV and Lambda inputs; it performs no model refit, hyperparameter selection or feature-set search.",
        "",
        "## Accounting and data gate",
        "",
        f"- The analysis covers 240 OOS months, 5 methods and {len(monthly):,} method-month observations.",
        f"- Stock-level in-universe transaction costs reconcile to reported cost net of exit cost with maximum absolute error {reconciliation['max_abs_error']:.3e}.",
        "- Factor fallback, matrix-m fallback and covariance repair counts are zero.",
        "- Stock-level cost concentration excludes exit-liquidation costs because exited names are not present in the current-month 500-name weight file; exit costs are reported separately.",
        "",
        "## Baseline mechanism metrics",
        "",
        _markdown_table(table.reset_index(drop=True)),
        "",
        "## Portfolio-ML return and cost decomposition",
        "",
        _markdown_table(comparison_table),
        "",
        "Cost saving is benchmark cost minus Portfolio-ML cost. The remaining net-return difference is attributable to the gross-return difference.",
        "",
        "## Inherited-position and aim adjustment",
        "",
        _markdown_table(pml_aim.reset_index(drop=True)),
        "",
        "The aim comparison is available only for Portfolio-ML because Static and Rank outputs do not store a distinct aim portfolio. Lambda distance is sqrt(delta' Lambda delta); the ratio compares implemented trading with an immediate move from the grown inherited position to the aim portfolio.",
        "",
        "## Interpretation",
        "",
        f"- Portfolio-ML Linear combines {metrics.loc['portfolio_ml_linear', 'annualized_net_return']:.2%} net return with {metrics.loc['portfolio_ml_linear', 'average_turnover']:.2%} monthly turnover and {metrics.loc['portfolio_ml_linear', 'annualized_trading_cost']:.2%} annual cost. Its advantage is mostly gross-return generation, with lower cost adding to implementability.",
        f"- Portfolio-ML RF has the higher point net return ({metrics.loc['portfolio_ml_rf', 'annualized_net_return']:.2%}) but also higher volatility, turnover, leverage and cost than Linear. This explains why Linear is the more stable implementation despite RF's larger point estimate.",
        f"- Static-ML* lowers monthly turnover from {static['average_turnover']:.2%} to {static_star['average_turnover']:.2%}, leverage from {static['average_leverage']:.3f} to {static_star['average_leverage']:.3f}, and annual trading cost from {static['annualized_trading_cost']:.2%} to {static_star['annualized_trading_cost']:.2%}. It improves implementation but does not close the utility gap to Portfolio-ML.",
        f"- Return-ML Rank earns only {rank['annualized_gross_return']:.2%} gross while paying {rank['annualized_trading_cost']:.2%} annual cost. Cost is {rank_cost_ratio:.2f} times absolute gross return, so its net-of-cost failure is mechanically transparent.",
        "- Portfolio-ML implementability is visible in lower L1 trading and lower Lambda-weighted movement away from inherited positions. Linear does not always hold the highest-ADV names, so its low realized cost is primarily a trading-intensity result rather than a mechanical liquidity tilt.",
        "- Predictive ranking alone is not sufficient for implementability: the Rank portfolio demonstrates that unconstrained trading intensity can erase gross predictability.",
        "",
        "## Cost-to-gross-return diagnostic",
        "",
        "Annual cost-to-gross ratios use annualized cost divided by max(abs(annualized gross return), 0.5%). The plot caps ratios at 500% and marks the cap; this prevents near-zero gross-return years from dominating the scale.",
        "",
        _markdown_table(pd.DataFrame({"method": [DISPLAY[m] for m in METHODS], "near_zero_gross_return_years": near_zero_years.reindex(METHODS).to_numpy()})),
        "",
        "## Figure inventory",
        "",
    ]
    captions = {
        "cumulative_gross_net_returns.png": "Cumulative gross and net returns separate signal performance from implementation cost.",
        "annual_net_return.png": "Annual net returns show whether full-sample differences are concentrated in a few years.",
        "annual_utility.png": "Annual utility flow applies the objective-consistent risk penalty.",
        "monthly_turnover.png": "Monthly turnover compares trading intensity through time.",
        "annual_turnover.png": "Annual average turnover summarizes persistent method differences.",
        "monthly_trading_cost.png": "Monthly transaction costs show cost spikes and crisis-period implementation pressure.",
        "annual_trading_cost.png": "Annualized transaction cost aggregates the monthly implementation burden.",
        "monthly_leverage.png": "Gross leverage distinguishes scale from portfolio-construction efficiency.",
        "annual_leverage.png": "Annual average leverage identifies persistent scaling differences.",
        "net_drawdown.png": "Net-return drawdowns compare downside paths after transaction costs.",
        "cost_to_gross_return.png": "The robust annual cost-to-gross ratio highlights methods whose signal is consumed by trading.",
        "trade_cost_concentration.png": "Cost concentration measures dependence on the largest names or trades, with exit cost separate.",
        "liquidity_exposure.png": "Portfolio-weighted ADV and liquidity-decile exposure describe cost-bearing holdings.",
        "inherited_aim_adjustment.png": "Portfolio-ML adjustment ratios show how full matrix m tempers immediate movement to aim weights.",
    }
    for filename, caption in captions.items():
        lines.append(f"- `figures/{filename}`: {caption}")
    lines.extend(
        [
            "",
            "## Scope limitation",
            "",
            "This is explanatory mechanism analysis, not model selection. No result here changes Core10, gamma, AUM, the RF grid or the locked method ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = list(formatted.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
