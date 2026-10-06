from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PANEL = PROJECT_ROOT / "data/processed/us_equity_ml/master_panel.parquet"
DATA_CONFIG = PROJECT_ROOT / "configs/data_us_equity_ml.yaml"
REPORT_DIR = PROJECT_ROOT / "reports/feature_selection"
CANDIDATE_DATA_ROOT = PROJECT_ROOT / "data/diagnostics/final20_jkp_pre_oos"
PRE_OOS_START = pd.Timestamp("1995-01-31")
PRE_OOS_END = pd.Timestamp("2004-12-31")

FEATURE_METADATA = {
    # Value
    "assets_to_market": {"jkp_field": "at_me", "theme": "Value", "meaning": "Total assets scaled by market equity."},
    "sales_to_market": {"jkp_field": "sale_me", "theme": "Value", "meaning": "Sales scaled by market equity."},
    "dividend_yield": {"jkp_field": "div_me", "theme": "Value", "meaning": "Cash dividends scaled by market equity."},
    "ebit_to_market": {"jkp_field": "ebit_me", "theme": "Value", "meaning": "EBIT scaled by market equity."},
    # Quality
    "piotroski_f_score": {"jkp_field": "f_score", "theme": "Quality", "meaning": "Piotroski composite accounting-strength score."},
    "altman_z_score": {"jkp_field": "z_score", "theme": "Quality", "meaning": "Altman financial-distress score."},
    "qmj_safety": {"jkp_field": "qmj_safety", "theme": "Quality", "meaning": "Safety component of quality-minus-junk."},
    "qmj_composite": {"jkp_field": "qmj", "theme": "Quality", "meaning": "Composite quality-minus-junk characteristic."},
    # Profitability
    "cash_operating_profitability": {"jkp_field": "cop_at", "theme": "Profitability", "meaning": "Cash-based operating profits scaled by assets."},
    "operating_cashflow_to_assets": {"jkp_field": "ocf_at", "theme": "Profitability", "meaning": "Operating cash flow scaled by assets."},
    "net_income_to_assets": {"jkp_field": "ni_at", "theme": "Profitability", "meaning": "Net income scaled by assets."},
    "qmj_profitability": {"jkp_field": "qmj_prof", "theme": "Profitability", "meaning": "Profitability component of quality-minus-junk."},
    # Investment / asset growth
    "net_operating_assets": {"jkp_field": "noa_at", "theme": "Investment / asset growth", "meaning": "Net operating assets scaled by total assets."},
    "ppe_inventory_growth": {"jkp_field": "ppeinv_gr1a", "theme": "Investment / asset growth", "meaning": "Annual growth in property, plant, equipment, and inventory scaled by assets."},
    "capex_growth": {"jkp_field": "capx_gr1", "theme": "Investment / asset growth", "meaning": "One-year capital-expenditure growth."},
    "inventory_growth": {"jkp_field": "inv_gr1", "theme": "Investment / asset growth", "meaning": "One-year inventory growth."},
    # Low risk / volatility
    "betting_against_beta": {"jkp_field": "betabab_1260d", "theme": "Low risk / volatility", "meaning": "Long-horizon beta used by betting-against-beta strategies."},
    "market_beta_252d": {"jkp_field": "beta_252d", "theme": "Low risk / volatility", "meaning": "Market beta estimated over 252 trading days."},
    "high_low_volatility_21d": {"jkp_field": "rvolhl_21d", "theme": "Low risk / volatility", "meaning": "Short-window high-low realized volatility."},
    "maximum_return_21d": {"jkp_field": "rmax5_21d", "theme": "Low risk / volatility", "meaning": "Average of the five largest daily returns over 21 days; a lottery-risk proxy."},
    # Liquidity
    "turnover_126d": {"jkp_field": "turnover_126d", "theme": "Liquidity", "meaning": "Share turnover over 126 trading days."},
    "zero_trades_252d": {"jkp_field": "zero_trades_252d", "theme": "Liquidity", "meaning": "Zero-trading-day illiquidity over 252 trading days."},
    "bidask_spread_21d": {"jkp_field": "bidaskhl_21d", "theme": "Liquidity", "meaning": "High-low effective bid-ask spread over 21 days."},
    "dollar_volume_variability": {"jkp_field": "dolvol_var_126d", "theme": "Liquidity", "meaning": "Variability of dollar trading volume over 126 days."},
    # Medium-term momentum
    "momentum_9_1": {"jkp_field": "ret_9_1", "theme": "Medium-term momentum", "meaning": "Return from months t-9 through t-1."},
    "intermediate_momentum_12_7": {"jkp_field": "ret_12_7", "theme": "Medium-term momentum", "meaning": "Return from months t-12 through t-7."},
    "momentum_18_1": {"jkp_field": "ret_18_1", "theme": "Medium-term momentum", "meaning": "Return from months t-18 through t-1."},
    "residual_momentum_12_1": {"jkp_field": "resff3_12_1", "theme": "Medium-term momentum", "meaning": "Fama-French residual momentum over months t-12 through t-1."},
    # Earnings / cashflow yield
    "operating_cashflow_yield": {"jkp_field": "ocf_me", "theme": "Earnings / cashflow yield", "meaning": "Operating cash flow scaled by market equity."},
    "free_cashflow_yield": {"jkp_field": "fcf_me", "theme": "Earnings / cashflow yield", "meaning": "Free cash flow scaled by market equity."},
    "cash_operating_profit_yield": {"jkp_field": "cop_me", "theme": "Earnings / cashflow yield", "meaning": "Cash operating profits scaled by market equity."},
    "ebitda_yield": {"jkp_field": "ebitda_me", "theme": "Earnings / cashflow yield", "meaning": "EBITDA scaled by market equity."},
    # Leverage / balance-sheet strength
    "debt_to_assets": {"jkp_field": "debt_at", "theme": "Leverage / balance-sheet strength", "meaning": "Total debt scaled by assets."},
    "net_debt_to_market": {"jkp_field": "netdebt_me", "theme": "Leverage / balance-sheet strength", "meaning": "Net debt scaled by market equity."},
    "debt_to_book": {"jkp_field": "debt_be", "theme": "Leverage / balance-sheet strength", "meaning": "Total debt scaled by book equity."},
    "cash_to_liabilities": {"jkp_field": "cash_lt", "theme": "Leverage / balance-sheet strength", "meaning": "Cash scaled by total liabilities; balance-sheet liquidity."},
    # Conservative growth / accruals
    "operating_accruals": {"jkp_field": "oaccruals_at", "theme": "Conservative growth / accruals", "meaning": "Operating accruals scaled by assets."},
    "total_accruals": {"jkp_field": "taccruals_at", "theme": "Conservative growth / accruals", "meaning": "Total accruals scaled by assets."},
    "working_capital_growth": {"jkp_field": "cowc_gr1a", "theme": "Conservative growth / accruals", "meaning": "Annual change in operating working capital scaled by assets."},
    "long_term_noa_growth": {"jkp_field": "lnoa_gr1a", "theme": "Conservative growth / accruals", "meaning": "Annual growth in long-term net operating assets."},
}

THEME_BONUS = {
    "Value": 0.8,
    "Quality": 1.0,
    "Profitability": 0.8,
    "Investment / asset growth": 0.8,
    "Low risk / volatility": 0.6,
    "Liquidity": 0.6,
    "Medium-term momentum": 0.5,
    "Earnings / cashflow yield": 1.0,
    "Leverage / balance-sheet strength": 1.0,
    "Conservative growth / accruals": 1.0,
}


def main() -> int:
    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    core10 = list(config["feature_sets"]["core10"])
    schema = set(pq.read_schema(MASTER_PANEL).names)
    master_features = sorted(
        column.removeprefix("raw_")
        for column in schema
        if column.startswith("raw_")
        and f"rank_{column.removeprefix('raw_')}" in schema
    )
    candidates = list(FEATURE_METADATA)

    selected_columns = [
        "eom",
        "permno",
        "in_top500",
        "ret_exc_lead1m",
        *[f"rank_{feature}" for feature in core10],
    ]
    panel = pd.read_parquet(MASTER_PANEL, columns=selected_columns)
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel = panel[
        panel["in_top500"].fillna(False)
        & panel["eom"].between(PRE_OOS_START, PRE_OOS_END)
    ].copy()
    panel = panel.sort_values(["permno", "eom"]).reset_index(drop=True)
    if panel["eom"].nunique() != 120 or len(panel) != 60_000:
        raise RuntimeError(
            f"Expected 60,000 Top500 pre-OOS rows over 120 months, found "
            f"{len(panel)} rows over {panel['eom'].nunique()} months"
        )

    candidate_files = sorted(CANDIDATE_DATA_ROOT.glob("year=*/part.parquet"))
    if len(candidate_files) != 10:
        raise RuntimeError(
            f"Expected ten cached JKP pre-OOS partitions, found {len(candidate_files)}"
        )
    jkp = pd.concat([pd.read_parquet(path) for path in candidate_files], ignore_index=True)
    jkp["eom"] = pd.to_datetime(jkp["eom"])
    if jkp.duplicated(["eom", "permno"]).any() or len(jkp) != 60_000:
        raise RuntimeError("JKP diagnostic cache has duplicate or incomplete Top500 keys")
    panel = panel.merge(jkp, on=["eom", "permno"], how="left", validate="one_to_one")
    for feature, metadata in FEATURE_METADATA.items():
        field = metadata["jkp_field"]
        panel[f"raw_{feature}"] = pd.to_numeric(panel[field], errors="coerce")
        panel[f"rank_{feature}"] = (
            panel.groupby("eom", sort=False)[f"raw_{feature}"]
            .rank(pct=True, method="average")
            .fillna(0.5)
        )

    scores, annual_ic = score_candidates(panel, candidates, core10)
    scores = scores.sort_values("balanced_score", ascending=False).reset_index(drop=True)
    scores.insert(0, "balanced_rank", np.arange(1, len(scores) + 1))
    conservative_order = scores["conservative_score"].rank(
        method="first", ascending=False
    ).astype(int)
    scores.insert(1, "conservative_rank", conservative_order)

    balanced = select_diversified(
        scores,
        score_column="balanced_score",
        eligibility_column="balanced_eligible",
    )
    conservative = select_diversified(
        scores,
        score_column="conservative_score",
        eligibility_column="conservative_eligible",
    )
    scores["balanced_decision"] = np.where(
        scores["feature_name"].isin(balanced), "include", "exclude"
    )
    scores["conservative_decision"] = np.where(
        scores["feature_name"].isin(conservative), "include", "exclude"
    )
    scores["proposed_include_exclude"] = scores.apply(
        lambda row: (
            "include_both"
            if row["balanced_decision"] == row["conservative_decision"] == "include"
            else "include_balanced"
            if row["balanced_decision"] == "include"
            else "include_conservative"
            if row["conservative_decision"] == "include"
            else "exclude"
        ),
        axis=1,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(REPORT_DIR / "final20_candidate_scores.csv", index=False)
    write_selection_yaml(
        REPORT_DIR / "final20_balanced.yaml",
        version_name="Final20 Balanced",
        core10=core10,
        selected=balanced,
        scores=scores,
        ranking_column="balanced_score",
    )
    write_selection_yaml(
        REPORT_DIR / "final20_conservative.yaml",
        version_name="Final20 Conservative",
        core10=core10,
        selected=conservative,
        scores=scores,
        ranking_column="conservative_score",
    )
    report = build_report(
        scores=scores,
        annual_ic=annual_ic,
        core10=core10,
        balanced=balanced,
        conservative=conservative,
        master_features=master_features,
        panel=panel,
    )
    (REPORT_DIR / "final20_selection_report.md").write_text(report, encoding="utf-8")
    print(f"status=selection_complete")
    print(f"available_additional_features={len(candidates)}")
    print(f"required_additional_features=10")
    print(f"balanced={balanced}")
    print(f"conservative={conservative}")
    print(f"report={REPORT_DIR / 'final20_selection_report.md'}")
    return 0


def score_candidates(
    panel: pd.DataFrame, candidates: list[str], core10: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_ic_rows: list[dict[str, object]] = []
    monthly_std_rows: list[dict[str, object]] = []
    monthly_corr_rows: list[dict[str, object]] = []
    label = panel["ret_exc_lead1m"]
    for eom, group in panel.groupby("eom", sort=True):
        label_rank = pd.to_numeric(group["ret_exc_lead1m"], errors="coerce").rank(
            pct=True, method="average"
        )
        ic_row: dict[str, object] = {"eom": eom, "year": int(eom.year)}
        std_row: dict[str, object] = {"eom": eom}
        for candidate in candidates:
            values = pd.to_numeric(group[f"rank_{candidate}"], errors="coerce")
            ic_row[candidate] = float(values.corr(label_rank))
            std_row[candidate] = float(values.std(ddof=1))
            for core_feature in core10:
                monthly_corr_rows.append(
                    {
                        "eom": eom,
                        "feature": candidate,
                        "core_feature": core_feature,
                        "correlation": float(
                            values.corr(group[f"rank_{core_feature}"])
                        ),
                    }
                )
        monthly_ic_rows.append(ic_row)
        monthly_std_rows.append(std_row)
    monthly_ic = pd.DataFrame(monthly_ic_rows)
    annual_ic = monthly_ic.groupby("year", sort=True)[candidates].mean()
    monthly_std = pd.DataFrame(monthly_std_rows)
    monthly_corr = pd.DataFrame(monthly_corr_rows)
    persistence = monthly_rank_autocorrelation(panel, candidates)

    rows: list[dict[str, object]] = []
    for feature in candidates:
        raw = pd.to_numeric(panel[f"raw_{feature}"], errors="coerce")
        rank = pd.to_numeric(panel[f"rank_{feature}"], errors="coerce")
        missing = raw.isna()
        dispersion = pd.to_numeric(monthly_std[feature], errors="coerce")
        ic = pd.to_numeric(monthly_ic[feature], errors="coerce").dropna()
        annual = pd.to_numeric(annual_ic[feature], errors="coerce").dropna()
        mean_ic = float(ic.mean())
        nw_se, nw_t = newey_west_mean(ic.to_numpy(float), lags=6)
        sign = 1.0 if mean_ic >= 0 else -1.0
        same_sign_fraction = float((sign * annual > 0).mean())
        positive_fraction = float((annual > 0).mean())
        annual_abs = annual.abs().sort_values(ascending=False)
        top3_share = float(
            annual_abs.head(3).sum() / max(float(annual_abs.sum()), 1.0e-16)
        )
        feature_corr = monthly_corr[monthly_corr["feature"].eq(feature)]
        pair_corr = feature_corr.groupby("core_feature")["correlation"].mean()
        avg_abs_corr = float(pair_corr.abs().mean())
        max_abs_corr = float(pair_corr.abs().max())
        most_correlated = str(pair_corr.abs().idxmax())
        monthly_persistence = float(persistence[feature].mean())
        trading_sensitive = FEATURE_METADATA[feature]["jkp_field"] in {
            "rvolhl_21d",
            "rmax5_21d",
            "bidaskhl_21d",
        }
        trading_sensitivity_penalty = (
            max(0.70 - monthly_persistence, 0.0) / 0.70 if trading_sensitive else 0.0
        )

        missing_score = float(np.clip(1.0 - missing.mean() / 0.20, 0.0, 1.0))
        dispersion_score = float(np.clip(dispersion.median() / 0.20, 0.0, 1.0))
        data_quality_score = 0.7 * missing_score + 0.3 * dispersion_score
        pre_oos_ic_score = float(np.clip(abs(mean_ic) / 0.03, 0.0, 1.0))
        concentration_score = float(np.clip(1.0 - top3_share / 0.60, 0.0, 1.0))
        ic_stability_score = 0.6 * same_sign_fraction + 0.4 * concentration_score
        persistence_score = float(np.clip((monthly_persistence - 0.20) / 0.75, 0.0, 1.0))
        redundancy_penalty = 0.6 * avg_abs_corr + 0.4 * max_abs_corr
        theme_bonus = THEME_BONUS[FEATURE_METADATA[feature]["theme"]]
        balanced_score = (
            0.25 * data_quality_score
            + 0.25 * pre_oos_ic_score
            + 0.15 * ic_stability_score
            + 0.20 * persistence_score
            + 0.10 * (1.0 - redundancy_penalty)
            + 0.05 * theme_bonus
            - 0.05 * trading_sensitivity_penalty
        )
        conservative_score = (
            0.35 * data_quality_score
            + 0.05 * pre_oos_ic_score
            + 0.10 * ic_stability_score
            + 0.30 * persistence_score
            + 0.20 * (1.0 - redundancy_penalty)
            - 0.05 * trading_sensitivity_penalty
        )
        hard_failure = (
            missing.mean() > 0.20
            or dispersion.median() < 0.02
            or monthly_persistence < 0.20
            or max_abs_corr > 0.90
            or (trading_sensitive and monthly_persistence < 0.70)
        )
        conservative_eligible = bool(
            not hard_failure
            and missing.mean() <= 0.10
            and dispersion.median() >= 0.05
            and monthly_persistence >= 0.60
            and max_abs_corr <= 0.85
            and avg_abs_corr <= 0.35
        )
        reason = candidate_reason(
            feature,
            missing_rate=float(missing.mean()),
            mean_ic=mean_ic,
            persistence=monthly_persistence,
            max_abs_corr=max_abs_corr,
            hard_failure=hard_failure,
        )
        rows.append(
            {
                "feature_name": feature,
                "jkp_field_name": FEATURE_METADATA[feature]["jkp_field"],
                "theme": FEATURE_METADATA[feature]["theme"],
                "economic_meaning": FEATURE_METADATA[feature]["meaning"],
                "raw_missing_rate": float(missing.mean()),
                "rank_missing_fill_rate": float((missing & rank.eq(0.5)).mean()),
                "missing_fill_correct_fraction": float(rank[missing].eq(0.5).mean())
                if missing.any()
                else 1.0,
                "rank_dispersion_median_xs_std": float(dispersion.median()),
                "rank_dispersion_p10_xs_std": float(dispersion.quantile(0.10)),
                "near_constant_month_fraction": float((dispersion < 0.01).mean()),
                "pre_oos_mean_rank_ic": mean_ic,
                "pre_oos_ic_newey_west_se": nw_se,
                "pre_oos_ic_t_stat": nw_t,
                "positive_ic_year_fraction": positive_fraction,
                "same_sign_ic_year_fraction": same_sign_fraction,
                "top3_abs_ic_year_concentration": top3_share,
                "monthly_rank_autocorrelation": monthly_persistence,
                "average_abs_corr_with_core10": avg_abs_corr,
                "max_abs_corr_with_core10": max_abs_corr,
                "most_correlated_core10_feature": most_correlated,
                "data_quality_score": data_quality_score,
                "pre_oos_ic_score": pre_oos_ic_score,
                "ic_stability_score": ic_stability_score,
                "persistence_score": persistence_score,
                "redundancy_penalty": redundancy_penalty,
                "theme_diversification_bonus": theme_bonus,
                "trading_sensitive_21d": trading_sensitive,
                "trading_sensitivity_penalty": trading_sensitivity_penalty,
                "balanced_score": balanced_score,
                "conservative_score": conservative_score,
                "balanced_eligible": not hard_failure,
                "conservative_eligible": conservative_eligible,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows), annual_ic.reset_index()


def select_diversified(
    scores: pd.DataFrame,
    *,
    score_column: str,
    eligibility_column: str,
) -> list[str]:
    eligible = scores[scores[eligibility_column]].sort_values(
        score_column, ascending=False
    )
    selected: list[str] = []
    theme_counts: dict[str, int] = {}
    theme_order = [
        "Quality",
        "Earnings / cashflow yield",
        "Leverage / balance-sheet strength",
        "Conservative growth / accruals",
        "Value",
        "Profitability",
        "Investment / asset growth",
        "Low risk / volatility",
        "Liquidity",
        "Medium-term momentum",
    ]
    for theme in theme_order:
        group = eligible[eligible["theme"].eq(theme)]
        if len(group):
            feature = str(group.iloc[0]["feature_name"])
            selected.append(feature)
            theme_counts[theme] = 1
        if len(selected) == 10:
            return selected
    for row in eligible.itertuples(index=False):
        feature = str(row.feature_name)
        theme = str(row.theme)
        if feature in selected or theme_counts.get(theme, 0) >= 2:
            continue
        selected.append(feature)
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        if len(selected) == 10:
            return selected
    raise RuntimeError(
        f"Only {len(selected)} features satisfy {eligibility_column} with the "
        "preregistered two-per-theme cap"
    )


def monthly_rank_autocorrelation(
    panel: pd.DataFrame, candidates: list[str]
) -> pd.DataFrame:
    ordered = panel.sort_values(["permno", "eom"]).copy()
    previous_eom = ordered.groupby("permno")["eom"].shift()
    current_month_number = ordered["eom"].dt.year * 12 + ordered["eom"].dt.month
    previous_month_number = previous_eom.dt.year * 12 + previous_eom.dt.month
    consecutive = (current_month_number - previous_month_number).eq(1)
    rows: list[dict[str, object]] = []
    for feature in candidates:
        current = ordered[f"rank_{feature}"]
        lagged = ordered.groupby("permno")[f"rank_{feature}"].shift()
        temporary = pd.DataFrame(
            {
                "eom": ordered["eom"],
                "current": current.where(consecutive),
                "lagged": lagged.where(consecutive),
            }
        )
        for eom, group in temporary.groupby("eom", sort=True):
            rows.append(
                {
                    "eom": eom,
                    "feature": feature,
                    "autocorrelation": float(group["current"].corr(group["lagged"])),
                }
            )
    return pd.DataFrame(rows).pivot(index="eom", columns="feature", values="autocorrelation")


def newey_west_mean(values: np.ndarray, lags: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    centered = array - array.mean()
    n = len(array)
    variance = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    se = float(np.sqrt(max(variance, 0.0) / n))
    return se, float(array.mean() / se) if se > 0 else np.nan


def candidate_reason(
    feature: str,
    *,
    missing_rate: float,
    mean_ic: float,
    persistence: float,
    max_abs_corr: float,
    hard_failure: bool,
) -> str:
    pieces = [
        f"pre-OOS IC={mean_ic:.4f}",
        f"persistence={persistence:.3f}",
        f"max Core10 correlation={max_abs_corr:.3f}",
        f"missing={missing_rate:.1%}",
    ]
    if hard_failure:
        pieces.append("fails at least one preregistered hard screen")
    elif FEATURE_METADATA[feature]["theme"] == "Medium-term momentum" and persistence < 0.5:
        pieces.append("momentum signal receives a low-persistence trading-cost caution")
    else:
        pieces.append("passes available-data hard screens")
    return "; ".join(pieces)


def write_selection_yaml(
    path: Path,
    *,
    version_name: str,
    core10: list[str],
    selected: list[str],
    scores: pd.DataFrame,
    ranking_column: str,
) -> None:
    if len(selected) != 10:
        raise RuntimeError(f"{version_name} must contain exactly ten additions")
    lookup = scores.set_index("feature_name")
    payload = {
        "version": 1,
        "name": version_name,
        "status": "preregistered_feature_selection_complete",
        "do_not_run": True,
        "pilot_recommended": True,
        "selection_period": {"start": "1995-01-31", "end": "2004-12-31"},
        "portfolio_results_used": False,
        "source": "contrib_global_factor.global_factor",
        "candidate_pool_size": len(scores),
        "selection_rule": {
            "score": ranking_column,
            "theme_cap": 2,
            "required_additions": 10,
            "balanced_hard_screen": (
                "missing<=0.20, rank_std>=0.02, persistence>=0.20, "
                "max_core_corr<=0.90; 21d trading-sensitive fields require persistence>=0.70"
            ),
            "conservative_hard_screen": (
                "balanced screen plus missing<=0.10, rank_std>=0.05, "
                "persistence>=0.60, max_core_corr<=0.85, avg_core_corr<=0.35"
            ),
        },
        "core10": core10,
        "selected_additional_features": selected,
        "final20": [*core10, *selected],
        "selected_feature_details": [
            {
                "feature": feature,
                "jkp_field": str(lookup.at[feature, "jkp_field_name"]),
                "theme": str(lookup.at[feature, "theme"]),
                "score": float(lookup.at[feature, ranking_column]),
                "pre_oos_mean_rank_ic": float(
                    lookup.at[feature, "pre_oos_mean_rank_ic"]
                ),
                "monthly_rank_autocorrelation": float(
                    lookup.at[feature, "monthly_rank_autocorrelation"]
                ),
                "max_abs_corr_with_core10": float(
                    lookup.at[feature, "max_abs_corr_with_core10"]
                ),
            }
            for feature in selected
        ],
        "execution_note": (
            "This YAML records feature selection only. It is intentionally marked "
            "do_not_run for the current task; no model or backtest has been executed."
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_report(
    *,
    scores: pd.DataFrame,
    annual_ic: pd.DataFrame,
    core10: list[str],
    balanced: list[str],
    conservative: list[str],
    master_features: list[str],
    panel: pd.DataFrame,
) -> str:
    display = scores[
        [
            "balanced_rank",
            "conservative_rank",
            "feature_name",
            "jkp_field_name",
            "theme",
            "raw_missing_rate",
            "rank_dispersion_median_xs_std",
            "pre_oos_mean_rank_ic",
            "pre_oos_ic_t_stat",
            "positive_ic_year_fraction",
            "top3_abs_ic_year_concentration",
            "monthly_rank_autocorrelation",
            "average_abs_corr_with_core10",
            "max_abs_corr_with_core10",
            "balanced_score",
            "conservative_score",
            "balanced_eligible",
            "conservative_eligible",
            "proposed_include_exclude",
            "reason",
        ]
    ]
    excluded = scores[scores["proposed_include_exclude"].eq("exclude")]
    high_ic_cutoff = float(scores["pre_oos_mean_rank_ic"].abs().quantile(0.75))
    high_ic_excluded = excluded[
        excluded["pre_oos_mean_rank_ic"].abs() >= high_ic_cutoff
    ][
        [
            "feature_name",
            "theme",
            "pre_oos_mean_rank_ic",
            "monthly_rank_autocorrelation",
            "max_abs_corr_with_core10",
            "raw_missing_rate",
            "reason",
        ]
    ]
    selection = scores[scores["feature_name"].isin(set(balanced) | set(conservative))][
        [
            "feature_name",
            "jkp_field_name",
            "theme",
            "economic_meaning",
            "pre_oos_mean_rank_ic",
            "pre_oos_ic_t_stat",
            "positive_ic_year_fraction",
            "monthly_rank_autocorrelation",
            "average_abs_corr_with_core10",
            "max_abs_corr_with_core10",
            "balanced_decision",
            "conservative_decision",
        ]
    ].sort_values(["theme", "feature_name"])
    balanced_themes = scores.set_index("feature_name").loc[balanced, "theme"].value_counts()
    conservative_themes = scores.set_index("feature_name").loc[
        conservative, "theme"
    ].value_counts()
    return (
        "# Preregistered Final20 feature-selection report\n\n"
        "## Decision\n\n"
        "**Status: selection complete. Core10 remains the main specification; Final20 "
        "is an optional extension only. No model or backtest was run.**\n\n"
        f"The existing master panel contains {len(master_features)} configured features. "
        f"A wider, preregistered pool of {len(scores)} JKP fields was queried only for "
        "1995-2004. Balanced and Conservative each select ten additions under a "
        "two-per-theme cap.\n\n"
        "## Scope and anti-snooping controls\n\n"
        f"- Diagnostic sample: Top500, 1995-01 through 2004-12, {len(panel):,} stock-months.\n"
        "- Labels: JKP next-month excess returns already present in the master panel.\n"
        "- No 2005-2024 returns, portfolio utility, predictions, weights, selected "
        "hyperparameters, or experiment result files were read.\n"
        "- All score components and hard screens were fixed before inspecting candidate scores.\n"
        "- Balanced score weights: data quality 25%, absolute pre-OOS IC 25%, IC "
        "stability 15%, persistence 20%, non-redundancy 10%, theme bonus 5%.\n"
        "- Conservative score weights: data quality 35%, absolute pre-OOS IC 5%, IC "
        "stability 10%, persistence 30%, non-redundancy 20%.\n"
        "- Hard exclusion: missing rate above 20%, median cross-sectional rank std below "
        "0.02, monthly persistence below 0.20, maximum Core10 correlation above 0.90, "
        "or a 21-day trading-sensitive signal with persistence below 0.70.\n\n"
        "- Conservative eligibility additionally requires missing rate at most 10%, "
        "rank std at least 0.05, persistence at least 0.60, maximum Core10 correlation "
        "at most 0.85, and average absolute Core10 correlation at most 0.35.\n\n"
        "## Selected additions\n\n"
        f"### Balanced\n\n`{balanced}`\n\nTheme counts: `{balanced_themes.to_dict()}`.\n\n"
        f"### Conservative\n\n`{conservative}`\n\nTheme counts: `{conservative_themes.to_dict()}`.\n\n"
        + markdown(selection)
        + "\n\n## Full candidate diagnostics\n\n"
        + markdown(display)
        + "\n\n## Annual pre-OOS rank IC\n\n"
        + markdown(annual_ic)
        + "\n\n## Requested questions\n\n"
        "1. **Why Final20 is an extension:** Core10 is the locked main specification. "
        "A larger set increases estimation dimension and data-mining risk, so it can "
        "only be a preregistered robustness extension.\n"
        "2. **Final15 problem:** its five additions cover only two momentum horizons, "
        "liquidity, earnings yield, and profitability. It lacks broad leverage, "
        "balance-sheet strength, accruals, conservative investment, and additional "
        "low-risk coverage.\n"
        "3. **Data-snooping protection:** the report uses only 1995-2004 characteristic "
        "and next-month-return data and never reads portfolio outputs.\n"
        f"4. **Balanced additions:** {balanced}.\n"
        f"5. **Conservative additions:** {conservative}.\n"
        "6. **Economic meanings:** supplied in the selected-additions table and candidate CSV.\n"
        "7. **Core10 correlation:** average and maximum absolute monthly correlations, "
        "plus the most-correlated Core10 feature, are reported in the CSV.\n"
        "8. **Pre-OOS IC and persistence:** mean rank IC, HAC t-stat, annual sign "
        "fractions, concentration, and monthly autocorrelation are reported above.\n"
        "9. **High-IC exclusions:** listed below. A high IC cannot override low "
        "persistence, excessive redundancy, missingness, or the theme cap.\n\n"
        + markdown(high_ic_excluded)
        + "\n\n10. **Pilot recommendation:** yes, a small Top500/Final20 pilot is justified "
        "as a later robustness exercise because the selection is fully pre-OOS and "
        "theme-diversified. It must not replace Core10, and no pilot is started in "
        "this task.\n"
    )


def markdown(frame: pd.DataFrame) -> str:
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
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
