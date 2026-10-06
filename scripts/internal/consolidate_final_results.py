from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "reports/final_results"
WORKBOOK_DATA = PROJECT_ROOT / ".artifact_tool_final_results/workbook_data.json"

SPECIFICATIONS = {
    "Core10": PROJECT_ROOT / "results/experiments/top500_core10_factor_cumulative",
    "Final15": PROJECT_ROOT / "results/experiments/top500_final15_factor_cumulative",
    "Final20 Conservative": PROJECT_ROOT
    / "results/experiments/top500_final20_conservative_factor_cumulative",
}

METHOD_LABELS = {
    "return_ml_rank": "Return-ML Rank",
    "static_ml": "Static-ML",
    "static_ml_star": "Static-ML*",
    "portfolio_ml_linear": "Portfolio-ML Linear",
    "portfolio_ml_rf": "Portfolio-ML RF",
}
METHOD_ORDER = list(METHOD_LABELS)
INFERENCE_ORDER = [
    ("portfolio_ml_linear", "static_ml"),
    ("portfolio_ml_linear", "static_ml_star"),
    ("portfolio_ml_rf", "static_ml"),
    ("portfolio_ml_rf", "static_ml_star"),
    ("portfolio_ml_rf", "portfolio_ml_linear"),
]
METRIC_COLUMNS = [
    "annualized_net_return",
    "annualized_volatility",
    "net_sharpe",
    "average_turnover",
    "average_leverage",
    "annualized_trading_cost",
    "annualized_ex_ante_utility_flow",
]
METRIC_LABELS = {
    "annualized_net_return": "Net return",
    "annualized_volatility": "Volatility",
    "net_sharpe": "Sharpe",
    "average_turnover": "Turnover",
    "average_leverage": "Leverage",
    "annualized_trading_cost": "Trading cost",
    "annualized_ex_ante_utility_flow": "Utility",
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKBOOK_DATA.parent.mkdir(parents=True, exist_ok=True)

    metrics = {name: _load_metrics(root) for name, root in SPECIFICATIONS.items()}
    inference = {
        name: _load_inference(root) for name, root in SPECIFICATIONS.items()
    }
    risk = {name: _load_risk(root) for name, root in SPECIFICATIONS.items()}

    main_metrics = metrics["Core10"].copy()
    main_inference = inference["Core10"].copy()
    robustness_metrics = pd.concat(metrics.values(), ignore_index=True)
    pairwise = _pairwise_deltas(metrics)
    risk_table = pd.concat(risk.values(), ignore_index=True)

    (OUTPUT_DIR / "final_specification_lock_in.md").write_text(
        _lock_in_memo(metrics, inference, risk), encoding="utf-8"
    )
    (OUTPUT_DIR / "final_performance_tables.md").write_text(
        _performance_tables_markdown(
            main_metrics, main_inference, robustness_metrics, pairwise, risk_table
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "robustness_narrative.md").write_text(
        _robustness_narrative(metrics, inference, risk), encoding="utf-8"
    )
    (OUTPUT_DIR / "next_stage_plan.md").write_text(
        _next_stage_plan(), encoding="utf-8"
    )

    payload = _workbook_payload(
        main_metrics, main_inference, robustness_metrics, pairwise, risk_table
    )
    WORKBOOK_DATA.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"output_dir={OUTPUT_DIR}")
    print(f"workbook_data={WORKBOOK_DATA}")
    return 0


def _load_metrics(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "metrics.csv")
    frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
    frame["method_order"] = frame["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    frame = frame.sort_values("method_order")
    frame.insert(0, "specification", _spec_name(root))
    frame["method"] = frame["method"].map(METHOD_LABELS)
    return frame[["specification", "method", *METRIC_COLUMNS]].reset_index(drop=True)


def _load_inference(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "statistical_inference.csv")
    frame = frame[frame["outcome"].eq("utility_flow")].copy()
    order = {pair: index for index, pair in enumerate(INFERENCE_ORDER)}
    frame["comparison_order"] = [
        order.get((candidate, benchmark), 99)
        for candidate, benchmark in zip(
            frame["candidate_method"], frame["benchmark_method"], strict=False
        )
    ]
    frame = frame[frame["comparison_order"].lt(99)].sort_values("comparison_order")
    frame.insert(0, "specification", _spec_name(root))
    frame["comparison"] = (
        frame["candidate_method"].map(METHOD_LABELS)
        + " vs "
        + frame["benchmark_method"].map(METHOD_LABELS)
    )
    frame["years_won"] = (
        frame["annual_years_won"].astype(int).astype(str)
        + "/"
        + frame["annual_years_total"].astype(int).astype(str)
    )
    return frame[
        [
            "specification",
            "comparison",
            "annualized_mean_difference",
            "newey_west_t_stat",
            "bootstrap_annualized_ci_lower",
            "bootstrap_annualized_ci_upper",
            "probability_delta_positive",
            "annual_years_won",
            "annual_years_total",
            "years_won",
        ]
    ].reset_index(drop=True)


def _load_risk(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "risk_calibration.csv")
    frame = frame[frame["scope"].eq("summary")].copy()
    frame = frame[frame["method"].isin(METHOD_ORDER)]
    frame["method_order"] = frame["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    frame = frame.sort_values("method_order")
    frame.insert(0, "specification", _spec_name(root))
    frame["method"] = frame["method"].map(METHOD_LABELS)
    return frame[
        [
            "specification",
            "method",
            "yearly_ratio_median",
            "yearly_ratio_max",
            "fraction_years_within_0_6_1_5",
            "years_ratio_above_2",
            "quality_gate_passed",
        ]
    ].reset_index(drop=True)


def _spec_name(root: Path) -> str:
    for name, candidate in SPECIFICATIONS.items():
        if root == candidate:
            return name
    raise KeyError(root)


def _pairwise_deltas(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = [
        ("Core10", "Final15"),
        ("Core10", "Final20 Conservative"),
        ("Final15", "Final20 Conservative"),
    ]
    rows: list[dict[str, object]] = []
    for left_name, right_name in pairs:
        left = metrics[left_name].set_index("method")
        right = metrics[right_name].set_index("method")
        for method in [METHOD_LABELS[item] for item in METHOD_ORDER]:
            if method not in left.index or method not in right.index:
                continue
            row: dict[str, object] = {
                "comparison": f"{right_name} minus {left_name}",
                "left_specification": left_name,
                "right_specification": right_name,
                "method": method,
            }
            for column in METRIC_COLUMNS:
                row[column] = float(right.loc[method, column] - left.loc[method, column])
            rows.append(row)
    return pd.DataFrame(rows)


def _lock_in_memo(
    metrics: dict[str, pd.DataFrame],
    inference: dict[str, pd.DataFrame],
    risk: dict[str, pd.DataFrame],
) -> str:
    core_inf = inference["Core10"].set_index("comparison")
    final15_metrics = metrics["Final15"].set_index("method")
    core_metrics = metrics["Core10"].set_index("method")
    final20_metrics = metrics["Final20 Conservative"].set_index("method")
    static_risk = risk["Core10"].set_index("method")
    return f"""# Final specification lock-in memo

## Locked specifications

**Main specification:** Top500/Core10, repaired factor risk, cumulative OOS validation, full matrix m, ADV quadratic transaction cost, gamma 10, current baseline AUM, Return-ML Rank, Static-ML, Static-ML*, Portfolio-ML Linear and Portfolio-ML RF. Markowitz remains diagnostic only.

**Robustness specifications:** Top500/Final15 and Top500/Final20 Conservative. Final20 Conservative is a pre-specified robustness extension and does not replace Core10.

## 1. Why Top500/Core10 is the main specification

Core10 is the most disciplined specification: it uses a compact feature set, limits model-selection degrees of freedom, has the longest established validation trail, and produces the cleanest joint evidence on implementable utility and Portfolio-ML risk calibration. Portfolio-ML Linear and RF earn annual utility-flow improvements of {_pct(core_inf.loc['Portfolio-ML Linear vs Static-ML', 'annualized_mean_difference'])} and {_pct(core_inf.loc['Portfolio-ML RF vs Static-ML', 'annualized_mean_difference'])} over Static-ML. Against Static-ML*, the improvements are {_pct(core_inf.loc['Portfolio-ML Linear vs Static-ML*', 'annualized_mean_difference'])} and {_pct(core_inf.loc['Portfolio-ML RF vs Static-ML*', 'annualized_mean_difference'])}, with HAC t-statistics of {core_inf.loc['Portfolio-ML Linear vs Static-ML*', 'newey_west_t_stat']:.3f} and {core_inf.loc['Portfolio-ML RF vs Static-ML*', 'newey_west_t_stat']:.3f}. Portfolio-ML Linear/RF have risk-calibration medians close to one and no years with realized/predicted volatility above two.

## 2. Why Final15 is not the main specification

Final15 does not provide a uniformly better implementable result. Portfolio-ML point estimates are similar to Core10, while Static-ML and Static-ML* become materially more levered and volatile. For example, Static-ML volatility rises from {_pct(core_metrics.loc['Static-ML', 'annualized_volatility'])} to {_pct(final15_metrics.loc['Static-ML', 'annualized_volatility'])}, and leverage rises from {core_metrics.loc['Static-ML', 'average_leverage']:.3f} to {final15_metrics.loc['Static-ML', 'average_leverage']:.3f}. The added characteristics therefore increase specification complexity without delivering cleaner main evidence.

## 3. Why Final20 Conservative is a robustness extension

Final20 Conservative was selected before its full portfolio backtest using data quality, pre-OOS IC, persistence, missingness, redundancy and theme-coverage rules. It preserves the Portfolio-ML result: Linear and RF annual utility are {_pct(final20_metrics.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'])} and {_pct(final20_metrics.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'])}. Relative to Core10, these utilities change by only {_pct(final20_metrics.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'] - core_metrics.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'])} and {_pct(final20_metrics.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'] - core_metrics.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'])}. This is useful robustness evidence, but not a material improvement over the simpler Core10 specification.

## 4. Why no Balanced or further feature-set search

The specification search is closed. Running Balanced, Final25, Final30 or further variants would increase researcher degrees of freedom after observing multiple full OOS portfolio results. That would weaken the interpretation of subsequent inference and create a feature-set selection channel based on realized portfolio outcomes. No additional feature set will be used to replace or optimize the locked Core10 main specification.

## 5. Treatment of Static risk-calibration exceptions

Static risk-calibration exceptions are reported, not post-hoc tuned away. Under Core10, Static-ML has a median realized/predicted ratio of {static_risk.loc['Static-ML', 'yearly_ratio_median']:.3f}, a maximum of {static_risk.loc['Static-ML', 'yearly_ratio_max']:.3f}, and {int(static_risk.loc['Static-ML', 'years_ratio_above_2'])} years above two. Static-ML* has a median of {static_risk.loc['Static-ML*', 'yearly_ratio_median']:.3f}, a maximum of {static_risk.loc['Static-ML*', 'yearly_ratio_max']:.3f}, and {int(static_risk.loc['Static-ML*', 'years_ratio_above_2'])} years above two. These methods remain economically relevant benchmarks, but their calibration limitations must accompany performance comparisons. Portfolio-ML conclusions are not inferred from Static risk calibration alone.

## 6. Scope of subsequent analysis

All frontier, AUM-sensitivity and feature-importance work must use Top500/Core10. Final15 and Final20 Conservative may appear only as robustness or appendix evidence. This memo authorizes no additional model training or feature-set search.

## 7. JKP accounting-data timing limitation

The implementation relies on the timing conventions embedded in the JKP characteristic data. I do not independently reconstruct accounting reporting lags from raw Compustat filing dates. This applies to accounting characteristics such as book-to-market, earnings-to-price, gross profitability, operating profitability and asset growth. It is treated as a data limitation to disclose in the dissertation data and limitations sections, not as a blocking error or a model-selection issue.
"""


def _performance_tables_markdown(
    main_metrics: pd.DataFrame,
    main_inference: pd.DataFrame,
    robustness_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    risk_table: pd.DataFrame,
) -> str:
    main_display = _display_metrics(main_metrics.drop(columns="specification"))
    inference_display = main_inference[
        [
            "comparison",
            "annualized_mean_difference",
            "newey_west_t_stat",
            "bootstrap_annualized_ci_lower",
            "bootstrap_annualized_ci_upper",
            "probability_delta_positive",
            "years_won",
        ]
    ].rename(
        columns={
            "comparison": "Comparison",
            "annualized_mean_difference": "Annual utility difference",
            "newey_west_t_stat": "HAC t-stat",
            "bootstrap_annualized_ci_lower": "Bootstrap CI lower",
            "bootstrap_annualized_ci_upper": "Bootstrap CI upper",
            "probability_delta_positive": "P(delta > 0)",
            "years_won": "Years won",
        }
    )
    for column in (
        "Annual utility difference",
        "Bootstrap CI lower",
        "Bootstrap CI upper",
        "P(delta > 0)",
    ):
        inference_display[column] = inference_display[column].map(_pct)
    inference_display["HAC t-stat"] = inference_display["HAC t-stat"].map(
        lambda value: f"{value:.3f}"
    )

    robustness_display = robustness_metrics.copy()
    robustness_display = _display_metrics(robustness_display)

    pairwise_display = pairwise[
        ["comparison", "method", *METRIC_COLUMNS]
    ].rename(columns={"comparison": "Comparison", "method": "Method", **METRIC_LABELS})
    for column in ("Net return", "Volatility", "Turnover", "Trading cost", "Utility"):
        pairwise_display[column] = pairwise_display[column].map(_pct)
    for column in ("Sharpe", "Leverage"):
        pairwise_display[column] = pairwise_display[column].map(
            lambda value: f"{value:.3f}"
        )

    risk_display = risk_table.rename(
        columns={
            "specification": "Specification",
            "method": "Method",
            "yearly_ratio_median": "Median ratio",
            "yearly_ratio_max": "Max ratio",
            "fraction_years_within_0_6_1_5": "Years within [0.6, 1.5]",
            "years_ratio_above_2": "Years ratio > 2",
            "quality_gate_passed": "Method gate",
        }
    )
    risk_display["Median ratio"] = risk_display["Median ratio"].map(
        lambda value: f"{value:.3f}"
    )
    risk_display["Max ratio"] = risk_display["Max ratio"].map(
        lambda value: f"{value:.3f}"
    )
    risk_display["Years within [0.6, 1.5]"] = risk_display[
        "Years within [0.6, 1.5]"
    ].map(_pct)
    risk_display["Years ratio > 2"] = risk_display["Years ratio > 2"].astype(int)

    return (
        "# Final performance tables\n\n"
        "All returns, volatility, transaction costs and utility-flow values are annualized. Turnover is the average monthly turnover. Utility is the empirical objective-consistent annualized ex-ante utility flow.\n\n"
        "## 1. Main Top500/Core10 metrics\n\n"
        + _markdown(main_display)
        + "\n\n## 2. Main utility inference\n\n"
        + _markdown(inference_display)
        + "\n\n## 3. Robustness metrics by specification\n\n"
        + _markdown(robustness_display)
        + "\n\n## 4. Pairwise robustness differences\n\n"
        "Differences are the right-hand specification minus the left-hand specification.\n\n"
        + _markdown(pairwise_display)
        + "\n\n## 5. Risk calibration\n\n"
        + _markdown(risk_display)
        + "\n"
    )


def _robustness_narrative(
    metrics: dict[str, pd.DataFrame],
    inference: dict[str, pd.DataFrame],
    risk: dict[str, pd.DataFrame],
) -> str:
    core = metrics["Core10"].set_index("method")
    final15 = metrics["Final15"].set_index("method")
    final20 = metrics["Final20 Conservative"].set_index("method")
    core_inf = inference["Core10"].set_index("comparison")
    final20_inf = inference["Final20 Conservative"].set_index("comparison")
    return f"""# Robustness narrative

## Top500/Core10 gives the cleanest main evidence

Core10 combines the smallest locked feature set with the clearest utility inference. Portfolio-ML Linear and RF deliver annualized utility of {_pct(core.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'])} and {_pct(core.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'])}. Linear versus Static-ML* has a HAC t-statistic of {core_inf.loc['Portfolio-ML Linear vs Static-ML*', 'newey_west_t_stat']:.3f} and a block-bootstrap interval of [{_pct(core_inf.loc['Portfolio-ML Linear vs Static-ML*', 'bootstrap_annualized_ci_lower'])}, {_pct(core_inf.loc['Portfolio-ML Linear vs Static-ML*', 'bootstrap_annualized_ci_upper'])}]. RF versus Static-ML* has a HAC t-statistic of {core_inf.loc['Portfolio-ML RF vs Static-ML*', 'newey_west_t_stat']:.3f}. Both Portfolio-ML variants also pass risk calibration without any year above a realized/predicted ratio of two.

## Final15: more characteristics do not guarantee more implementable utility

Final15 slightly raises Portfolio-ML point estimates, but it also materially increases Static portfolio risk and leverage. Static-ML volatility rises by {_pct(final15.loc['Static-ML', 'annualized_volatility'] - core.loc['Static-ML', 'annualized_volatility'])}, and Static-ML* volatility rises by {_pct(final15.loc['Static-ML*', 'annualized_volatility'] - core.loc['Static-ML*', 'annualized_volatility'])}. Static-ML* leverage rises from {core.loc['Static-ML*', 'average_leverage']:.3f} to {final15.loc['Static-ML*', 'average_leverage']:.3f}. The extra characteristics therefore do not mechanically improve implementable utility or calibration.

## Final20 Conservative: pre-specified expansion preserves the result

Final20 Conservative uses persistent, low-missing and relatively low-redundancy additions selected without 2005-2024 portfolio utility. Portfolio-ML Linear utility differs from Core10 by {_pct(final20.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'] - core.loc['Portfolio-ML Linear', 'annualized_ex_ante_utility_flow'])}; RF differs by {_pct(final20.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'] - core.loc['Portfolio-ML RF', 'annualized_ex_ante_utility_flow'])}. The Portfolio-ML advantage remains positive, but Final20 does not materially strengthen inference against Static-ML*: its HAC t-statistics are {final20_inf.loc['Portfolio-ML Linear vs Static-ML*', 'newey_west_t_stat']:.3f} for Linear and {final20_inf.loc['Portfolio-ML RF vs Static-ML*', 'newey_west_t_stat']:.3f} for RF.

## Final interpretation

The three specifications support the same qualitative mechanism: portfolio-level machine learning improves implementable utility relative to static allocation rules, while RF versus Linear is not statistically resolved. Core10 remains the main specification because it achieves this with the least feature-search discretion and the cleanest joint utility/risk evidence. Final15 and Final20 Conservative are robustness checks only. No further feature-set search is warranted.

## Common data-timing limitation

All three specifications rely on the timing conventions embedded in the JKP characteristic data. The implementation does not independently reconstruct accounting reporting lags from raw Compustat filing dates. This limitation is common to the main and robustness specifications and is not used to select among them.
"""


def _next_stage_plan() -> str:
    return """# Next-stage plan

This document defines the next analytical stage but does not execute it. Every primary analysis below must use the locked Top500/Core10 specification, repaired factor risk, cumulative OOS validation, full matrix m, ADV quadratic transaction costs, gamma 10 and the current baseline AUM.

## 1. Implementable efficient frontier

- Re-estimate the Top500/Core10 frontier over a pre-declared risk-aversion or target-risk grid.
- Plot annualized net return or utility against realized volatility, with separate curves for Static-ML, Static-ML* and Portfolio-ML Linear/RF.
- Report transaction-cost-adjusted and gross counterparts, confidence bands where feasible, and calibration diagnostics at each frontier point.
- Keep Markowitz diagnostic separate from the main frontier.

## 2. AUM sensitivity

- Use a pre-declared AUM grid centered on the current baseline.
- Hold features, RF grid, validation protocol, gamma and cost model fixed.
- Report net return, utility, turnover, trading cost, leverage and the location/shape of the implementable frontier as AUM changes.
- Test whether Portfolio-ML advantages survive rising market-impact penalties.

## 3. Turnover, trading cost and leverage mechanisms

- Monthly time-series plots of turnover, transaction cost and gross leverage.
- Distributions and annual summaries by method.
- Aim versus actual Portfolio-ML leverage and inherited-position leverage.
- Crisis-period annotations for 2008-2009, 2020 and 2022 without retuning the model around those events.

## 4. Risk-calibration plots

- Predicted versus realized annualized volatility scatter plots by method and year.
- Time series of realized/predicted volatility ratios with the [0.6, 1.5] calibration band and ratio-two warning line.
- Separate disclosure of Static-ML and Static-ML* exception years.
- Factor versus idiosyncratic risk contribution plots where already available from the locked risk model.

## 5. Feature and feature-theme importance

- Base all main importance analysis on Core10.
- Use methods compatible with portfolio utility and dynamic positions; do not infer economic importance solely from return-prediction MSE.
- Pre-specify permutation, ablation or theme-group definitions before calculation.
- Final20 Conservative may be shown only in an optional appendix as a theme-level robustness check; Final15 must not become a new selection pool.

## Execution guardrails

- No Balanced, Final25, Final30 or additional feature-set search.
- No change to baseline AUM, gamma, RF grid or validation protocol except inside the explicitly defined AUM/frontier grids.
- No new model run begins from this plan without a separate execution instruction and preflight.
"""


def _display_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.rename(
        columns={"specification": "Specification", "method": "Method", **METRIC_LABELS}
    ).copy()
    for column in ("Net return", "Volatility", "Turnover", "Trading cost", "Utility"):
        if column in display:
            display[column] = display[column].map(_pct)
    for column in ("Sharpe", "Leverage"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    return display


def _workbook_payload(
    main_metrics: pd.DataFrame,
    main_inference: pd.DataFrame,
    robustness_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    risk_table: pd.DataFrame,
) -> dict[str, object]:
    return {
        "title": "Final Results Consolidation: Top500/Core10 Main Specification",
        "subtitle": "Locked main and robustness specifications; no additional feature-set search",
        "main_metrics": _records(main_metrics.drop(columns="specification")),
        "main_inference": _records(main_inference),
        "robustness_metrics": _records(robustness_metrics),
        "pairwise_deltas": _records(pairwise),
        "risk_calibration": _records(risk_table),
        "specification_lock": [
            {
                "Role": "Main",
                "Specification": "Top500/Core10",
                "Decision": "Locked",
                "Use": "All primary results; future frontier, AUM and feature-importance analysis",
            },
            {
                "Role": "Robustness",
                "Specification": "Top500/Final15",
                "Decision": "Locked robustness only",
                "Use": "Shows that adding characteristics does not mechanically improve implementable utility",
            },
            {
                "Role": "Robustness",
                "Specification": "Top500/Final20 Conservative",
                "Decision": "Locked pre-specified robustness",
                "Use": "Persistent, low-missing and low-redundancy feature expansion",
            },
            {
                "Role": "Excluded",
                "Specification": "Balanced / Final25 / Final30 / further feature sets",
                "Decision": "Do not run",
                "Use": "Specification search is closed",
            },
        ],
        "sources": [
            {
                "Specification": name,
                "Metrics source": str(root / "metrics.csv"),
                "Inference source": str(root / "statistical_inference.csv"),
                "Risk source": str(root / "risk_calibration.csv"),
            }
            for name, root in SPECIFICATIONS.items()
        ],
        "definitions": {
            "Return/volatility/cost/utility": "Annualized fractions",
            "Turnover": "Average monthly turnover fraction",
            "Utility": "Annualized ex-ante utility flow consistent with the empirical objective",
            "Risk calibration band": "Realized/predicted annualized volatility ratio in [0.6, 1.5]",
            "Pairwise deltas": "Right-hand specification minus left-hand specification",
        },
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _pct(value: float) -> str:
    return f"{float(value):.2%}"


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    values = frame.astype(str)
    columns = [str(column) for column in values.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in values.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
