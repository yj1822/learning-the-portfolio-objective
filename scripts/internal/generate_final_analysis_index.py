from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports/final_analysis"
RESULT_ROOT = PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative"

METHOD_LABELS = {
    "return_ml_rank": "Return-ML Rank",
    "static_ml": "Static-ML",
    "static_ml_star": "Static-ML*",
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}


FIGURES = {
    "frontier_net_return_vs_volatility": (
        "Implementable Efficient Frontier",
        "Net return and realized volatility across the fixed gamma grid for the locked Top500/Core10 methods.",
        "main text",
        "frontier_points.csv; controlled gamma recomputation; gamma=10 highlighted",
    ),
    "frontier_utility_vs_volatility": (
        "Implementable Efficient Frontier",
        "Objective-consistent utility and realized volatility show the implementable frontier across risk aversion.",
        "main text",
        "frontier_points.csv; controlled gamma recomputation; gamma=10 highlighted",
    ),
    "frontier_cost_vs_volatility": (
        "Implementable Efficient Frontier",
        "Trading cost against realized volatility separates implementation burden from portfolio risk.",
        "appendix",
        "frontier_points.csv; controlled gamma recomputation",
    ),
    "frontier_turnover_vs_volatility": (
        "Implementable Efficient Frontier",
        "Monthly turnover against realized volatility compares trading intensity along each method's frontier.",
        "appendix",
        "frontier_points.csv; controlled gamma recomputation",
    ),
    "frontier_leverage_vs_volatility": (
        "Implementable Efficient Frontier",
        "Gross leverage against realized volatility identifies whether frontier outcomes are driven by portfolio scale.",
        "appendix",
        "frontier_points.csv; controlled gamma recomputation; low-gamma points retained",
    ),
    "aum_net_return": (
        "AUM Sensitivity",
        "Annualized net returns show how endogenous transaction-cost-aware portfolios respond to increasing AUM.",
        "appendix",
        "aum_sensitivity_points.csv; endogenous AUM recomputation; multiplier=1 baseline",
    ),
    "aum_utility": (
        "AUM Sensitivity",
        "Implementable utility across AUM shows that Portfolio-ML remains comparatively resilient to capacity pressure.",
        "main text",
        "aum_sensitivity_points.csv; endogenous AUM recomputation; multiplier=1 baseline",
    ),
    "aum_trading_cost": (
        "AUM Sensitivity",
        "Annualized trading cost rises with AUM but differs sharply across portfolio-construction methods.",
        "appendix",
        "aum_sensitivity_points.csv; endogenous AUM recomputation",
    ),
    "aum_turnover": (
        "AUM Sensitivity",
        "Turnover responses demonstrate which methods endogenously reduce trading as AUM increases.",
        "appendix",
        "aum_sensitivity_points.csv; endogenous AUM recomputation",
    ),
    "aum_leverage": (
        "AUM Sensitivity",
        "Average gross leverage across AUM distinguishes capacity adaptation from mechanical cost scaling.",
        "appendix",
        "aum_sensitivity_points.csv; endogenous AUM recomputation",
    ),
    "aum_utility_degradation": (
        "AUM Sensitivity",
        "Utility degradation relative to the frictionless diagnostic quantifies each method's sensitivity to trading costs.",
        "main text",
        "aum_sensitivity_points.csv; zero-cost point is diagnostic only",
    ),
    "cumulative_gross_net_returns": (
        "Main Results and Mechanisms",
        "Cumulative gross and net returns show how transaction costs separate predictive performance from implementability.",
        "main text",
        "locked Top500/Core10 baseline monthly returns; gamma=10; baseline AUM",
    ),
    "annual_net_return": (
        "Main Results and Mechanisms",
        "Annual net returns reveal the timing and persistence of method-level performance differences.",
        "appendix",
        "locked Top500/Core10 baseline monthly returns",
    ),
    "annual_utility": (
        "Main Results and Mechanisms",
        "Annual objective-consistent utility compares methods after return, risk and transaction costs.",
        "main text",
        "locked Top500/Core10 utility-flow definition",
    ),
    "monthly_turnover": (
        "Implementation Mechanisms",
        "Monthly turnover documents the high trading intensity of Return-ML Rank and the lower intensity of Portfolio-ML.",
        "main text",
        "locked Top500/Core10 baseline weights and inherited positions",
    ),
    "annual_turnover": (
        "Implementation Mechanisms",
        "Annual average turnover summarizes persistent implementation differences across methods.",
        "appendix",
        "locked Top500/Core10 baseline weights",
    ),
    "monthly_trading_cost": (
        "Implementation Mechanisms",
        "Monthly transaction costs expose crisis-period cost spikes and persistent differences in implementation burden.",
        "appendix",
        "ADV quadratic cost accounting; exit cost included in reported totals",
    ),
    "annual_trading_cost": (
        "Implementation Mechanisms",
        "Annualized transaction costs show how trading intensity translates into realized implementation drag.",
        "appendix",
        "ADV quadratic cost accounting",
    ),
    "monthly_leverage": (
        "Implementation Mechanisms",
        "Monthly gross leverage distinguishes portfolio scale from net-return and utility improvements.",
        "appendix",
        "locked Top500/Core10 baseline weights",
    ),
    "annual_leverage": (
        "Implementation Mechanisms",
        "Annual average leverage identifies persistent scaling differences across methods.",
        "appendix",
        "locked Top500/Core10 baseline weights",
    ),
    "net_drawdown": (
        "Main Results and Mechanisms",
        "Net-return drawdowns compare downside paths after all modeled transaction costs.",
        "appendix",
        "locked Top500/Core10 net returns",
    ),
    "cost_to_gross_return": (
        "Implementation Mechanisms",
        "The robust cost-to-gross-return ratio shows when implementation costs consume the underlying gross signal.",
        "main text",
        "annual cost divided by max(abs gross return, 0.5 percent); plot capped at 500 percent",
    ),
    "trade_cost_concentration": (
        "Implementation Mechanisms",
        "Stock-level cost concentration measures dependence on the largest names and trades, with exit cost reported separately.",
        "appendix",
        "stock-level in-universe trades; exit liquidations unavailable by name",
    ),
    "liquidity_exposure": (
        "Implementation Mechanisms",
        "Portfolio-weighted ADV and liquidity-decile exposures test whether low cost is driven by a mechanical liquidity tilt.",
        "appendix",
        "stock-level weights, ADV and lambda_adv inputs",
    ),
    "inherited_aim_adjustment": (
        "Implementation Mechanisms",
        "Actual-to-aim trade distances show how full matrix m tempers immediate movement away from inherited positions.",
        "main text",
        "Portfolio-ML saved aim weights, grown inherited positions and Lambda distances",
    ),
    "risk_predicted_vs_realized": (
        "Risk Calibration",
        "Predicted and realized annual volatility demonstrate more stable calibration for Portfolio-ML than for Static portfolios.",
        "main text",
        "locked Top500/Core10 factor-risk forecasts and realized net-return volatility",
    ),
    "risk_ratio_by_year": (
        "Risk Calibration",
        "Yearly realized-to-predicted volatility ratios isolate the Static risk misses in 2020 and 2021.",
        "main text",
        "locked Top500/Core10 annual risk calibration; no years removed",
    ),
    "risk_ratio_by_method": (
        "Risk Calibration",
        "Method-level calibration distributions summarize coverage and tail risk across the full OOS period.",
        "appendix",
        "locked Top500/Core10 annual risk calibration",
    ),
    "feature_importance_bar": (
        "Feature and Theme Importance",
        "Fixed-model feature neutralization identifies Momentum and volatility-related inputs as the largest positive utility dependencies.",
        "main text",
        "Top500/Core10 only; recovered annual beta; no retraining; full minus neutralized utility",
    ),
    "theme_importance_bar": (
        "Feature and Theme Importance",
        "Theme neutralization shows that Momentum and Risk/Low-volatility carry the largest positive utility effects.",
        "main text",
        "transparent Core10 theme mapping; fixed-model neutralization; no feature selection",
    ),
    "feature_contribution_by_year": (
        "Feature and Theme Importance",
        "Annual feature-neutralization effects show that importance varies through time and is not confined to one crisis year.",
        "appendix",
        "Top500/Core10 fixed-model neutralization; colors clipped symmetrically",
    ),
    "theme_contribution_by_year": (
        "Feature and Theme Importance",
        "Annual theme-neutralization effects display time variation and nonlinear differences between Linear and RF models.",
        "appendix",
        "Top500/Core10 fixed-model neutralization; RF interpretation is diagnostic",
    ),
}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _main_metrics_table(metrics: pd.DataFrame) -> str:
    rows = [
        "| Method | Net return | Volatility | Sharpe | Turnover | Leverage | Cost | Utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_LABELS:
        row = metrics.loc[metrics["method"].eq(method)].iloc[0]
        rows.append(
            "| {label} | {net} | {vol} | {sharpe:.3f} | {turnover} | {lev:.3f} | {cost} | {utility} |".format(
                label=METHOD_LABELS[method],
                net=_pct(row["annualized_net_return"]),
                vol=_pct(row["annualized_volatility"]),
                sharpe=row["net_sharpe"],
                turnover=_pct(row["average_turnover"]),
                lev=row["average_leverage"],
                cost=_pct(row["annualized_trading_cost"]),
                utility=_pct(row["annualized_ex_ante_utility_flow"]),
            )
        )
    return "\n".join(rows)


def _inference_table(inference: pd.DataFrame) -> str:
    targets = [
        ("portfolio_ml_linear", "static_ml"),
        ("portfolio_ml_linear", "static_ml_star"),
        ("portfolio_ml_rf", "static_ml"),
        ("portfolio_ml_rf", "static_ml_star"),
        ("portfolio_ml_rf", "portfolio_ml_linear"),
    ]
    utility = inference[inference["outcome"].eq("utility_flow")]
    rows = [
        "| Comparison | Annual utility difference | HAC t-stat | Bootstrap 95% CI | P(delta > 0) | Years won |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, benchmark in targets:
        row = utility[
            utility["candidate_method"].eq(candidate)
            & utility["benchmark_method"].eq(benchmark)
        ].iloc[0]
        rows.append(
            f"| {METHOD_LABELS[candidate]} vs {METHOD_LABELS[benchmark]} | "
            f"{_pct(row['annualized_mean_difference'])} | {row['newey_west_t_stat']:.3f} | "
            f"[{_pct(row['bootstrap_annualized_ci_lower'])}, {_pct(row['bootstrap_annualized_ci_upper'])}] | "
            f"{_pct(row['probability_delta_positive'])} | {int(row['annual_years_won'])}/{int(row['annual_years_total'])} |"
        )
    return "\n".join(rows)


def _figure_manifest() -> pd.DataFrame:
    images = sorted(REPORT_ROOT.rglob("*.png"))
    rows = []
    for path in images:
        stem = path.stem
        if stem not in FIGURES:
            raise RuntimeError(f"Figure is missing manifest metadata: {path}")
        section, caption, placement, notes = FIGURES[stem]
        rows.append(
            {
                "figure_file_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "suggested_dissertation_section": section,
                "one_sentence_caption": caption,
                "placement": placement,
                "dependencies_notes": notes,
            }
        )
    missing = sorted(set(FIGURES) - {path.stem for path in images})
    if missing:
        raise RuntimeError(f"Manifest metadata has no corresponding figure: {missing}")
    manifest = pd.DataFrame(rows)
    if manifest["figure_file_path"].duplicated().any():
        raise RuntimeError("Duplicate figure path in dissertation manifest")
    return manifest


def main() -> int:
    required = [
        RESULT_ROOT / "metrics.csv",
        RESULT_ROOT / "statistical_inference.csv",
        REPORT_ROOT / "frontier/frontier_points.csv",
        REPORT_ROOT / "aum_sensitivity/aum_sensitivity_points.csv",
        REPORT_ROOT / "mechanisms/mechanism_metrics.csv",
        REPORT_ROOT / "risk_calibration/risk_calibration_table.csv",
        REPORT_ROOT / "feature_importance/feature_importance.csv",
        REPORT_ROOT / "feature_importance/theme_importance.csv",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Final-analysis inputs are missing: {missing}")

    metrics = pd.read_csv(required[0])
    inference = pd.read_csv(required[1])
    frontier = pd.read_csv(required[2])
    aum = pd.read_csv(required[3])
    risk = pd.read_csv(required[5])
    feature = pd.read_csv(required[6])
    theme = pd.read_csv(required[7])

    if len(frontier) != 40 or set(frontier["gamma"]) != {2.5, 5, 7.5, 10, 15, 20, 30, 50}:
        raise RuntimeError("Frontier is not the complete fixed 8-gamma grid")
    if len(aum) != 35 or set(aum["aum_multiplier"]) != {0, 0.25, 0.5, 1, 2, 5, 10}:
        raise RuntimeError("AUM sensitivity is not the complete fixed 7-point grid")
    if int(frontier["fallback_count"].sum()) != 0 or int(aum["fallback_count"].sum()) != 0:
        raise RuntimeError("A final-analysis scenario contains a fallback")
    if int(frontier["covariance_repair_count"].sum()) != 0 or int(aum["covariance_repair_count"].sum()) != 0:
        raise RuntimeError("A final-analysis scenario contains a covariance repair")

    feature_full = feature[feature["scope"].eq("full")]
    theme_full = theme[theme["scope"].eq("full")]
    top_linear = feature_full[feature_full["method"].eq("portfolio_ml_linear")].nlargest(3, "delta_utility")
    top_rf = feature_full[feature_full["method"].eq("portfolio_ml_rf")].nlargest(3, "delta_utility")
    top_themes_linear = theme_full[theme_full["method"].eq("portfolio_ml_linear")].nlargest(3, "delta_utility")
    top_themes_rf = theme_full[theme_full["method"].eq("portfolio_ml_rf")].nlargest(3, "delta_utility")

    manifest = _figure_manifest()
    manifest_path = REPORT_ROOT / "dissertation_figures_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    pml_risk = risk[risk["method"].isin(["portfolio_ml_linear", "portfolio_ml_rf"])]
    static_risk = risk[risk["method"].isin(["static_ml", "static_ml_star"])]
    main_figures = manifest[manifest["placement"].eq("main text")]
    appendix_figures = manifest[manifest["placement"].eq("appendix")]

    lines = [
        "# Final analysis index",
        "",
        "## Locked specification",
        "",
        "- Main specification: Top500/Core10, repaired factor risk, cumulative OOS validation, full matrix m, ADV quadratic transaction cost, gamma=10 and baseline AUM.",
        "- Final15 and Final20 Conservative remain robustness specifications only.",
        "- Feature-set search is closed. No result in this analysis changes Core10, gamma, AUM, the RF grid or the method specification.",
        "- Frontier, AUM sensitivity and feature importance are mechanism, sensitivity and explanatory analyses, not model selection.",
        "",
        "## Main results recap",
        "",
        _main_metrics_table(metrics),
        "",
        _inference_table(inference),
        "",
        "Portfolio-ML Linear and RF both improve objective-consistent utility relative to Static-ML and Static-ML*. Linear has the cleaner implementation profile; RF has the higher point estimate, but RF versus Linear is not statistically resolved. Return-ML Rank fails net of costs because its annual cost exceeds its gross return.",
        "",
        "## Implementable frontier",
        "",
        "All eight pre-specified gamma points completed with zero factor fallback, matrix-m fallback and covariance repair. Portfolio-ML Linear and RF exceed Static-ML by average annualized utility flow of 1.964% and 2.346% across the fixed grid. Realized volatility is monotone for every method, while low-gamma points expose materially higher leverage, especially for Static portfolios. Gamma=10 exactly reconciles to the locked baseline and is highlighted because it was pre-specified, not reselected.",
        "",
        "## AUM sensitivity",
        "",
        "All seven pre-specified AUM points completed endogenously. From the zero-cost diagnostic to 10x baseline AUM, Return-ML Rank utility falls 12.787 percentage points, Portfolio-ML Linear falls 0.273 points and RF falls 0.858 points. Portfolio-ML reduces turnover and leverage as cost pressure increases; Static-ML* exceeds Static-ML at all seven fixed AUM points.",
        "",
        "## Portfolio mechanisms",
        "",
        "At the locked baseline, Portfolio-ML's utility advantage is primarily gross-return generation, while lower turnover and cost preserve implementability. Linear trades 4.74% per month at 0.01% annual cost versus 13.76% and 0.10% for Static-ML. The implemented L1 trade is only 28.5% of the immediate aim move for Linear and 26.4% for RF, showing how full matrix m retains inherited positions. Return-ML Rank pays 1.28% annual cost against only 0.16% gross return.",
        "",
        "## Risk calibration",
        "",
        f"Portfolio-ML yearly realized/predicted volatility ratios have medians from {pml_risk['median_yearly_ratio'].min():.3f} to {pml_risk['median_yearly_ratio'].max():.3f}, with zero years above 2. Static-ML and Static-ML* have {int(static_risk['years_ratio_above_2_count'].sum())} method-years above 2, all in 2020-2021, consistent with common-factor covariance underestimation in those regimes. These observations are retained without deleting years or retuning the model.",
        "",
        "## Feature and theme importance",
        "",
        f"Linear's largest positive fixed-model feature dependencies are {', '.join(top_linear['item_name'])}; RF's are {', '.join(top_rf['item_name'])}. Linear's leading themes are {', '.join(top_themes_linear['item_name'])}; RF's are {', '.join(top_themes_rf['item_name'])}. Momentum and Risk/Low-volatility are the two leading themes for both methods.",
        "",
        "Feature-importance rankings correlate 0.782 between Linear and RF. Importance is not strongly associated with rank persistence or the rank-change turnover proxy, and absolute descriptive OOS IC has only 0.188 rank correlation with utility importance. This supports the paper's distinction between predictability and implementable utility: risk, costs, trading dynamics and nonlinear interactions matter. Neutralization is model dependence, not a causal effect, and is not used to alter Core10.",
        "",
        "## Data and reproducibility limitations",
        "",
        "The implementation relies on the timing conventions embedded in the JKP characteristic data. I do not independently reconstruct accounting reporting lags from raw Compustat filing dates. This is a data limitation rather than a model-selection issue and should be disclosed in the dissertation data and limitations sections.",
        "",
        "The separate results package contains final reports, tables, figures, experiment outputs and final-analysis outputs. It is not a complete reproducibility archive: licensed raw WRDS/CRSP/JKP data, processed data, source code, third-party code, checkpoints and caches are excluded. Full reproduction requires the licensed data, project source code, environment/configuration and run scripts from the main workspace or repository.",
        "",
        "## Dissertation figure inventory",
        "",
        f"The machine-readable manifest contains {len(manifest)} figures: {len(main_figures)} recommended for the main text and {len(appendix_figures)} for the appendix.",
        "",
        "| Figure | Suggested section | Placement |",
        "| --- | --- | --- |",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(
            f"| `{row.figure_file_path}` | {row.suggested_dissertation_section} | {row.placement} |"
        )
    lines.extend(
        [
            "",
            "## Writing recommendations",
            "",
            "### Main results",
            "",
            "Use the net-return/volatility and utility/volatility frontier figures, AUM utility and utility-degradation figures, cumulative gross/net returns, annual utility, monthly turnover, cost-to-gross-return, inherited/aim adjustment, both principal risk-calibration figures, and the two full-sample importance bars. Together they establish the economic result, implementation mechanism, capacity response, risk quality and explanatory feature evidence.",
            "",
            "### Robustness section",
            "",
            "Refer to Final15 and Final20 Conservative only through the locked final-results tables and narrative. They are not inputs to any figure in this index. Use the AUM and gamma-grid analyses as sensitivity evidence for the Core10 main specification, without treating gamma or AUM as newly selected settings.",
            "",
            "### Appendix",
            "",
            "Place detailed cost, turnover and leverage frontier panels; AUM mechanism panels; annual and monthly mechanism diagnostics; drawdowns, liquidity and trade-cost concentration; the method-level calibration distribution; and annual feature/theme heatmaps in the appendix. The heatmaps are especially useful for documenting time concentration and RF interaction sensitivity without overloading the main empirical narrative.",
            "",
            "## Completion and quality status",
            "",
            "- Baseline reconciliation passed before final analysis.",
            "- Frontier: complete, 8/8 gamma points passed.",
            "- AUM sensitivity: complete, 7/7 points passed.",
            "- Mechanism plots: complete, 14 figures.",
            "- Risk calibration plots: complete, 3 figures.",
            "- Feature/theme importance: complete, 34 fixed-model neutralizations plus two baseline reconstructions; no retraining.",
            "- Factor fallback, matrix-m fallback, covariance repair and future-information violations: zero in controlled final-analysis scenarios.",
            "- The project can proceed to results-chapter writing without reopening feature-set or model-specification search.",
        ]
    )
    index_path = REPORT_ROOT / "final_analysis_index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        {
            "index": str(index_path),
            "manifest": str(manifest_path),
            "figures": len(manifest),
            "main_text": len(main_figures),
            "appendix": len(appendix_figures),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
