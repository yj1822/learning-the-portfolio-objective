# Licensed input contract

The empirical panel is intentionally absent. CRSP/WRDS-linked security-level data, JKP characteristics, factor-risk inputs, predictions, and weights must be obtained or built under the user's own data rights. Do not commit raw or processed copies.

## Provenance and periods

`configs/data_us_equity_ml.yaml` defines the data build. The adapter reads an authorised `trading-data-hub` U.S. equity panel and risk inputs. The research history begins in January 1990, portfolio analysis begins in January 1995, and the formal OOS period is January 2005–December 2024. The `ret_exc_lead1m` label is supplied by the upstream JKP panel; this repository does not silently reconstruct or shift it. Verify the configured characteristic aliases against the licensed release.

## Processed files expected by the loaders

Default root: `data/processed/us_equity_ml/`. `IEF_DATA_ROOT` changes the read root, `IEF_MASTER_PANEL_PATH` selects an existing panel, and `IEF_DATA_CONFIG` selects a data config. The principal files are:

| File | Contract |
| --- | --- |
| `master_panel.parquet` | One row per security/month, ordered by month and universe rank; identifiers, monthly returns and lead excess-return label, market/liquidity/cost inputs, universe flags, and configured features. |
| `cost_inputs.parquet` | `permno`, `eom`, `adv_6m`, `lambda_adv`, `in_top100`, `in_top300`, `in_top500`. |
| `risk/factor_exposures.parquet` | Factor exposures consumed by the configured factor-risk provider. |
| `risk/factor_returns_daily.parquet` | Daily factor returns for risk estimation. |
| `risk/factor_cov_monthly.parquet` | Monthly factor covariance. |
| `risk/idio_var_monthly.parquet` | Monthly idiosyncratic variances. |

The panel must contain `permno`, `eom`, `ret_1m`, `ret_exc_lead1m`, `price`, `mktcap`, `lag_mktcap`, `adv_6m`, `lambda_adv`, `vol_12m`, `universe_rank_mcap`, and the three `in_top*` flags. The build contract also requires `raw_` and `rank_` columns for each configured Final15 characteristic. `lambda_adv` is checked against the configured multiplier divided by `adv_6m`; do not substitute a different cost definition. See `src/implementable_frontier/data/preparation.py`, `loaders.py`, and the validation code for executable field and coverage checks.

## Preparing authorised inputs

Install this package with `python -m pip install -e ".[dev]"`. Separately configure the authorised `trading-data-hub` checkout and licensed data source. From the repository root:

```bash
python scripts/prepare_data.py --mode build
python scripts/prepare_data.py --mode validate
```

`--download` explicitly requests a licensed WRDS fetch. It requires credentials supplied through the environment, never committed files. The build writes local metadata, a risk-status manifest, and validation reports. A missing risk file or failed validation is an actionable failure; do not treat a partial panel as a reproduced baseline. The synthetic test suite is usable without these licensed inputs.
