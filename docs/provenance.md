# Code provenance review for the public repository

## Scope and evidence

This review uses the supplied thesis, code/config/test attachments, prior cleaned candidate, results archive, and the [Jensen et al. reference repository at commit `13d836b5187cbfa69730dd256d8eb767e65ca21f`](https://github.com/theisij/ml-and-the-implementable-efficient-frontier/tree/13d836b5187cbfa69730dd256d8eb767e65ca21f), supplied by the author for this review. The pinned checkout has no `LICENSE`, `COPYING`, or `NOTICE` file in its tracked tree; its README requests academic citation but does not state redistribution terms. GitHub's [licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) says a public repository without a licence does not grant general reproduction or redistribution rights. These classifications describe code provenance, not a legal opinion. A = independently expressed from equations or routine programming; B = conceptual/mathematical adaptation; C = structural adaptation of external implementation; D = close source translation.

The earlier candidate's `models/portfolios.py::_official_matrix_m` explicitly called itself a “Port of official R m_func at upstream commit 13d836b.” Direct comparison with [the pinned R `m_func`](https://github.com/theisij/ml-and-the-implementable-efficient-frontier/blob/13d836b5187cbfa69730dd256d8eb767e65ca21f/0%20-%20Portfolio%20choice%20functions.R) confirms a D-level concern: the same named intermediate matrices and calculation sequence appeared in Python. Its diagonal helper and matrix-square-root helper were part of that pathway. The candidate removes them and provides `models/dynamic_adjustment.py`, derived from Jiang (2026), §3.7.1's fixed-point equations with scalar diagonal evaluation and NumPy broadcast scaling. This is an independent *expression* of the required mathematics, not a methodological change. Jensen et al. remain credited.

The first candidate's `models/method_audit.py::reference_matrix_m` was also too structurally close to R `m_func`: its variables and ordered matrix construction followed the R sequence. It has now been rewritten as an independent spectral evaluation of the thesis's scalar quadratic root followed by linear solves. This retains an audit path distinct from the production square-root routine.

## Function-level assessment

| Local file and function | Identifiable basis | Class | Reason and action |
| --- | --- | :---: | --- |
| `models/portfolios.py::equal_weight` | Elementary equal weighting | A | Standard vector normalization; retain. |
| `models/portfolios.py::rank_long_short` | Thesis ranking design; R `rank_ml_implement` | B | The centered-rank formula is shared, but the short NumPy implementation has no distinctive R control flow or data-table expression; retain. |
| `models/portfolios.py::top_bottom_portfolio` | Thesis ranking benchmark | B | Ordinary quantile long-short construction; retain. |
| `models/portfolios.py::markowitz_weights` | Mean-variance first-order condition | A | Linear solve with ridge and singular fallback; retain. |
| `models/portfolios.py::static_ml_weights` | Thesis Static-ML quadratic objective; R `m_static` and `static_val_fun` as methodological references | B | Direct quadratic solve with inherited holdings, plus Python-specific cache/fallback handling. The shared matrix expression is the thesis equation, not a copied R block; retain. |
| `models/portfolios.py::static_ml_star_weights` | Thesis Static-ML* specification; R `static_val_fun` | B | Same closed-form objective, but class-independent Python solver with cache/fallback logic rather than the R monthly-loop structure; retain with attribution. |
| `models/portfolios.py::static_ml_precision_matrix` | Static objective Hessian | A | One mathematical matrix expression; retain. |
| `models/portfolios.py::static_ml_star_precision_matrix` | Adjusted static objective Hessian | B | Equation-level diagonal adjustment and trading penalty; retain with attribution. |
| `models/portfolios.py::compute_adjustment_matrix` | Thesis §3.7.1 plus local stability/fallback policy | B | Calls independently expressed fixed point, then checks spectral radius; retain. |
| `models/portfolios.py::_regularized_covariance` | Standard symmetric ridge covariance | A | Routine helper; retain. |
| `models/portfolios.py::_limit_leverage` | Gross-leverage constraint | A | Routine scaling; retain. |
| `models/portfolios.py::_solution` | Local result packaging | A | Routine diagnostics; retain. |
| `models/dynamic_adjustment.py::adjustment_matrix_from_risk` | Jiang thesis §3.7.1, Jensen et al. dynamic objective | A | Newly expressed from mathematical equations; broadcast risk scaling and fixed-point update; retain with attribution. |
| `models/dynamic_adjustment.py::_diagonal_adjustment` | Diagonal specialization of same equations | A | New assetwise recurrence; retain. |
| `models/dynamic_adjustment.py::_initial_dense_policy` | Positive semidefinite matrix root in thesis initialization | A | New eigendecomposition helper; retain. |
| `models/method_audit.py::SyntheticRiskProvider.__init__` | Local synthetic test fixture | A | Stores fixture panel; retain. |
| `models/method_audit.py::SyntheticRiskProvider.get_covariance` | Local synthetic covariance fixture | A | Generates deterministic test covariance; retain. |
| `models/method_audit.py::SyntheticRiskProvider.get_volatility` | Local synthetic risk fixture | A | Supplies synthetic volatility; retain. |
| `models/method_audit.py::synthetic_parity_panel` | Local parity-test design | A | Synthetic panel, no external source evidence; retain. |
| `models/method_audit.py::run_small_sample_parity` | Local cross-check orchestration | B | Checks Jensen-inspired policies against independent formulas; retain as integrity logic. |
| `models/method_audit.py::reference_portfolio_sufficient_statistics` | Thesis portfolio objective and accounting equations | B | Equation-level independent benchmark; retain. |
| `models/method_audit.py::reference_matrix_m` | Thesis §3.7.1 matrix equation; checked against R `m_func` | A | Rewritten from earlier C/D-like structure using rationalized scalar eigenvalue mapping and linear solves; retains mathematical parity without the R expression sequence. |
| `models/method_audit.py::reference_static_ml_star` | Thesis Static-ML* first-order condition | B | Compact direct solve used for a separate numerical check; retain. |
| `models/method_audit.py::direct_quadratic_solver` | Generic gradient iteration | A | Local independent solver for test parity; retain. |
| `models/method_audit.py::_check` | Local comparison diagnostic | A | Computes absolute and relative error; retain. |

The data classes `PortfolioSolution`, `AdjustmentMatrixResult`, and `ParityCheck` hold local diagnostics and have no identified external source expression. A targeted comparison also covered `features.py` against R `rff`, `return_ml.py` against R return-prediction functions, `portfolio_ml.py` against R `pfml_input_fun`/hyperparameter routines, and `accounting.py` against R `w_fun`/`pf_ts_fun`. Their shared elements are standard transforms or the thesis's economic equations; the Python modules use distinct object structure, explicit dynamic-universe alignment, and separate accounting. No additional close translation was identified in those paths. `scripts/internal/audit_official_method_parity.py` cites the upstream R names and requires an external checkout for comparison; it does not embed R source in this candidate.

## Numerical check and remaining rights work

Before the production replacement, 18 deterministic matrix cases (2, 3, and 5 assets; dense/diagonal covariance; AUM 0, 0.1, and 1 billion) were snapshotted. The production replacement's maximum elementwise absolute difference was **1.5543122344752192e-15**. Before rewriting the audit reference, its 12 positive-AUM outputs were snapshotted; after ten fixed-point iterations the rewritten reference had **0.0 maximum difference** across those cases. All 11 small-sample parity checks and the full pytest suite still pass. No licensed-data rerun or new empirical claim was made.

**No identified close translation remains in the reviewed public research core after the two matrix-path rewrites.** This is a targeted source comparison, not an exhaustive copyright certification of every historical script. The reference repository has no tracked licence file at the pinned commit; public visibility and a request to cite the paper do not settle redistribution rights. This repository contains no reference R files and no `LICENSE`. A further rights review remains appropriate before any open-source licence decision.
