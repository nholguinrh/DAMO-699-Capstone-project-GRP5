# Validation Record — `e44a055` (PR #119, Round-3 remediation)

**Reviewer role:** Lead Data Scientist / Principal Analytics Engineer
**Commit validated:** `e44a055` — *"fix(xgboost): restore objective ranking monotonicity, truthful seed disclosure, and 4-input manifest"*
**Method:** every item re-derived by executing the shipped code against the shipped artifacts. No claim below is taken from the commit message.

---

## Verdict: ✅ **VALIDATED — approve and merge**

**Methodological Risk: LOW.** No leakage, no metric corruption, no row duplication, no
non-determinism. Full suite **177 passed** (73 → 175 → 177 across the three remediation rounds).
The two must-fix items from Round 3 (R3-1, R3-3) are both resolved, and four of the five
should-fix / housekeeping items with them.

---

## 1. R3-1 — Objective ranking monotonicity: ✅ **Fixed**

| Check | Result |
|---|---|
| Non-monotonic adjacent rank pairs | **0** (was 8 of 27) |
| Outside-band ordering monotonic in `mean_relative_rmse` | **True** |
| Band members | 1 |
| Band leader == global objective minimum | **True** |
| Canonical baseline | **Rank 3**, 1.050857 — matches §4.3 line 155 |
| Former tier-0 winner | **Rank 6**, 1.058223 — matches §4.3 line 164 |

```
 rank  depth    lr  n_est  subsample  mean_relative_rmse  std     in_band  seeds  canonical
    1      2  0.01    600        1.0            1.038096  0.000000   True      1
    2      2  0.01    600        0.8            1.045429  0.006101  False      5
    3      3  0.03    150        0.8            1.050857  0.002665  False      5      ✔
    4      3  0.01    600        1.0            1.052395  0.000000  False      1
    5      4  0.01    600        1.0            1.056970  0.000000  False      1
    6      2  0.01    600        0.7            1.058223  0.003635  False      5
```

Parsimony is now correctly scoped to the indifference band (`in_band` sorted by
`n_estimators`/`max_depth`, everything else by the objective), and
`test_rank_is_monotonic_in_objective_outside_the_indifference_band` guards both the ordering and
the band-leader-is-global-minimum invariant.

**Also fixed: an error of mine.** The Round-2 caveat text I supplied wrote the leader-to-runner-up
gap as $7.3\times10^{-4}$; the true gap is $7.3\times10^{-3}$. §4.3 line 162 now reads
$7.3\times10^{-3}$ (runner-up) and $1.28\times10^{-2}$ (canonical), both of which I confirmed
against the CSV. Writing it as `1e-4` would have understated the separation by an order of
magnitude and made the configurations look far more tied than they are.

**Concern withdrawn.** I flagged `compare_best_to_canonical` using `results_df.iloc[0]` as a latent
risk if the band held ≥ 2 members. With parsimony-within-band now the intended selection rule,
`iloc[0]` *is* the selected configuration — the one frozen into production — so it is the correct
row for both `best_config` and the `relative_gain_pct` comparison. No change needed.

---

## 2. R3-2 — Seed disclosure: ✅ **Fixed**

`n_selection_seeds` and `selection_seeds` now report what was executed, and a `dispersion_basis`
column distinguishes structural from empirical zero dispersion:

| rank | subsample / colsample | `n_selection_seeds` | `selection_seeds` | `dispersion_basis` |
|---|---|---|---|---|
| 1 | 1.0 / 1.0 | **1** | `42` | `deterministic` |
| 2 | 0.8 / 0.8 | 5 | `42,0,1,7,2024` | `multi_seed` |
| 3 | 0.8 / 0.8 | 5 | `42,0,1,7,2024` | `multi_seed` |
| 4 | 1.0 / 1.0 | **1** | `42` | `deterministic` |

§4.3 item 4 discloses it in prose and states explicitly that the frozen rank-1 specification is of
the deterministic class. The audit artifact no longer overstates the robustness evidence for the
configuration that shipped.

---

## 3. R3-3 — $R^2_{OOS}$ correction: ✅ **Fixed**

§4.3 line 176 now reads *"$R^2_{OOS} = -9.27\%$ (realized) and $-4.02\%$ after the Clark-West
adjustment"*. Cross-checked against `outputs/clark_west_test_results.csv`:

| model | h | N | `r2_oos` | `r2_oos_adj` |
|---|---|---|---|---|
| XGBoost (Tuned) | 1 | 745 | −9.2681 % | −4.0235 % |

Exact match. The stale −9.03 % (pre-#119, N=741) is gone.

---

## 4. R3-4 — Data manifest: ✅ **Fixed**

- **All four inputs recorded** with sha256, size, row count and date bounds: `gold_features.csv`
  (4 268 rows, 2010-02-19 → 2026-06-30), `bank_of_canada_data.csv` (4 552),
  `fred_rates.csv` (4 563), `statcan_cpi.csv` (210).
- **Emitted by both pipelines** — `xgboost_baseline.run_pipeline` and now `model_comparison`.
- **Bound to the invariant.** `test_common_sample_size_is_attributable_to_the_recorded_vintage`
  asserts the recorded gold sha256 *and* `n_forecasts == 745` together, so `745` is now checkable
  against a digest rather than folklore. This was the substance of R2-4 and it is closed.

Minor residual: `statcan_cpi.csv` carries no `start_date`/`end_date` (no `date` column — the CPI
file uses a different date field name). Not a defect, but the manifest attests to less for that
input than for the other three.

---

## 5. Regression checks (previously fixed items still holding)

| Invariant | Result |
|---|---|
| Search CSV rows reconcile (per-horizon mean vs `mean_relative_rmse`) | **0 of 28 fail** at atol 1e-4 |
| C-1 leakage: scored origins inside selection window | **0 of 3 728** |
| Frozen constants match rank 1 (`depth 2, lr .01, 600, 1.0, 1.0, 1.0, 0.5`) | **match** |
| Downstream artifacts consistent (correctly not regenerated) | ✔ |
| Rank-1 row reproduces from a clean re-run | **exact** on `mean_relative_rmse`, all three `rel_rmse_h*`, `n_selection_seeds`, `dispersion_basis` |
| `naive_rmse_h*_sd` dead columns (R3-5) | **removed** |
| §4.3 selection-power caveat (R3-6) | **refreshed**, figures verified |
| `charts.py` legacy-label fallback comment (R3-9) | **added** |

---

## 6. Outstanding (non-blocking)

| # | Item | Assessment |
|---|---|---|
| R3-7 | `src/xgboost_baseline.py:34-39` still uses a `try: from src.… except ImportError:` shim, so `xgboost_baseline` and `src.xgboost_baseline` remain two importable module objects with independent `_CACHED_GOLD_DF` / `HAS_XGBOOST`. | Acceptable. Harmless in practice; the clean fix (a package `__init__.py` and one import style) is a repo-wide change that does not belong in this PR. |
| R3-8 | `src/project_paths.py` still has no trailing newline (file ends `t`). | Cosmetic; POSIX/`git diff` nit only. |

Neither blocks merge.

---

## 7. Summary of the remediation arc

| Round | Risk | Blocking findings | Outcome |
|---|---|---|---|
| 1 (`8843c72`) | **CRITICAL** | C-1 selection leakage over 76.1 % of scored origins; C-2 unversioned vintage confound; C-3 §4.3 partition dates wrong on every segment | Blocked |
| 2 (`f8f3297`) | **MEDIUM** | Leakage eliminated and independently verified; new: audit CSV internally inconsistent in 19/28 rows, OOS null deleted rather than reported | Changes requested |
| 3 (`e9790be`) | **LOW** | Null result reported with retraction; ranking regression from my own Round-2 diff | Changes requested |
| 4 (`e44a055`) | **LOW** | — | ✅ **Approved** |

The substantive econometric result stands and is well evidenced: **leakage-safe hyperparameter
tuning yields no out-of-sample gain over untuned defaults** (+0.052 pp on the 3-horizon mean, worse
at $h \in \{1,20\}$, better at $h=5$), and the apparent gain in the original submission was
selection bias that vanished once the selection window was moved off the scored origins. §4.3 item 5
reports this, quantifies it, ties it to the EMH framing in §4.2, and retracts the earlier claim with
the reason — which is the standard this kind of benchmark work should be held to.

**Recommendation: merge.**
