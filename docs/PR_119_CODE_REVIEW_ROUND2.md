# Code Review Round 2 — `feat/119-xgboost-hyperparameter-tuning` (Issue #119)

**Reviewer role:** Lead Data Scientist / Principal Analytics Engineer
**Commit under review:** `f8f3297` — *"fix(xgboost): eliminate selection leakage via burn-in holdout + multi-seed evaluation (PR #119 review)"*
**Baseline for this round:** `8843c72` (state reviewed in `docs/PR_119_CODE_REVIEW.md`)
**Verification:** all findings below were re-derived by executing the shipped code against the shipped data, not read from the diff.

---

## 1. Methodological Risk Assessment: **MEDIUM** (was **CRITICAL**)

**The blocking finding is genuinely resolved.** C-1 was not patched cosmetically — the selection
window is now index-bounded at `MIN_TRAIN`, and I independently confirmed **zero** overlap between
the tuning partitions and the scored forecast origins at every horizon. Full suite: **73 passed**.

| Round-1 finding | Status | Verified how |
|---|---|---|
| **C-1** selection leakage (76.1 % of origins) | ✅ **Fixed** | Partition manifest + recomputed: 0 of 3,728 scored origins touched at h=1/5/20 |
| **C-2** unversioned vintage confound | ⚠️ **Partially** — 1 of 3 parts | Strict `== 745` restored; no vintage manifest; tuning delta still never isolated |
| **C-3** wrong partition dates in §4.3 | ✅ **Fixed** | Geometry now symbolic + `r3_xgboost_search_partitions.csv`; 4.2×/17.7× corrected |
| **H-1** tautological embargo test | ✅ **Fixed** | Test now calls `evaluate_inner_split_config` and asserts on returned indices |
| **H-2** single-seed argmin | ✅ **Fixed** | 5 seeds, `std_relative_rmse` published, "none beat Naïve" disclosed |
| **M-1** silent leaky fallbacks | ✅ **Fixed** | `ValueError` + dedicated regression test |
| **M-2** stale duplicated defaults | ✅ **Fixed** | Single `resolved` dict against baseline constants |
| **M-3** conformal miscoverage | ⚠️ **Partially** | Disclosed, but the disclosed numbers are already wrong (R2-3) |
| **L-1** split "XGBoost" / "XGBoost (Tuned)" labels | ❌ **Not fixed** | `app.py:121,528`, `charts.py:11,37` still `"XGBoost"` |
| **L-2** `MIN_FIT_ROWS` in loop | ✅ **Fixed** | Hoisted to `xgboost_baseline.py:53` |
| **L-3** dual module identity | ❌ **Not fixed** | `xgboost_baseline.py:32` still bare `from gold_feature_pipeline import …` |
| **L-4** retrospective SHAP | ✅ **Fixed** | Docstring caveat added |
| **L-5** `rel_gain` returning `0.0` | ✅ **Fixed** | Returns `None` |

**Why the risk is Medium rather than Low.** Removing the leakage removed the result. On the same
data vintage the tuned specification is **no better than the untuned defaults out-of-sample**
(R2-1, measured below) — and §4.3 responded by *deleting* the before/after comparison rather than
reporting the null. Separately, the audit CSV that documents the selection is now internally
inconsistent in 19 of its 28 rows (R2-2). Nothing here is leakage; all of it is reporting integrity.

---

## 2. C-1 Remediation — Verified Correct

Recomputed from the shipped code and data:

```
first scored origin: 2012-02-17 (index 500)
  h= 1: selection touches idx 0..498 (through 2012-02-15); overlap with scored origins = 0
  h= 5: selection touches idx 0..494 (through 2012-02-09); overlap with scored origins = 0
  h=20: selection touches idx 0..479 (through 2012-01-19); overlap with scored origins = 0
scored origins inside selection pool: 0
```

The `outputs/r3_xgboost_search_partitions.csv` manifest is exactly the provenance artifact
requested in C-3, with both indices and calendar dates. `evaluate_inner_split_config` now returns
`(record, partitions)` and `run_grid_search` returns `(results_df, partitions_df)` — a clean
breaking change, correctly propagated to every caller and test. This is a good fix.

---

## 3. New Findings

### 🟠 R2-1 — Leakage-free tuning delivers **no** out-of-sample improvement, and §4.3 removed the comparison instead of reporting it

**Files:** `docs/CLARK_WEST_METHODOLOGY.md:160-161` (§4.3 item 5); `outputs/r3_xgboost_vs_naive.csv`;
`src/xgboost_baseline.py:57-64`

Round 1's C-2 asked for the tuning delta to be re-derived on a **fixed** data vintage. That was not
done, so I ran it: the pre-#119 canonical configuration (`depth 3, lr 0.03, n_est 150, subsample
0.8, colsample 0.8, mcw 1.0, λ 1.0`) through `run_rolling_cv` on the **current** vintage,
`N = 3728` origins — directly comparable to the shipped tuned numbers.

| $h$ | Untuned canonical RMSE | **Tuned (shipped)** RMSE | Untuned vs Naïve | **Tuned vs Naïve** | Δ from tuning |
|---|---|---|---|---|---|
| 1 | 0.029672 | **0.029690** | −2.940 % | **−2.997 %** | **−0.057 pp (worse)** |
| 5 | 0.065164 | **0.064930** | −3.323 % | **−2.953 %** | +0.370 pp (better) |
| 20 | 0.129038 | **0.129240** | −2.307 % | **−2.464 %** | **−0.157 pp (worse)** |
| **mean** | | | **−2.857 %** | **−2.805 %** | **+0.052 pp — a wash** |

Tuning is worse at two of three horizons and improves the 3-horizon mean by 0.05 pp. **This is the
cleanest possible confirmation that C-1 was real**: the apparent "+1.13 % / 4.5 % MSPE reduction"
reported at `8843c72` was the selection bias, and it disappeared the moment the leakage did.

That is a legitimate and publishable finding — it strengthens the paper's EMH argument. What is not
legitimate is how §4.3 handled it. Item 5 previously read:

> *"Tuning reduced $h=1$ MSPE from $0.000921$ to $0.000880$ (a 4.5 % error reduction …)"*

and now reads:

> *"**Realized OOS Performance**: On the canonical $N=745$ common sample, `XGBoost (Tuned)` captures short-term momentum at $h=5$ …"*

The before/after comparison was **deleted**, not corrected. On the shipped `N=745` sample the
tuned h=1 MSPE went **0.000880 → 0.000921** and $R^2_{OOS}$ **−4.43 % → −9.27 %**; at h=20,
**0.016439 → 0.016676** and **−4.69 % → −6.20 %**. A reader of §4.3 alone cannot tell that the
tuning exercise produced no OOS gain — which is the single most important result of Issue #119.

```diff
-5. **Realized OOS Performance**: On the canonical $N=745$ common sample, `XGBoost (Tuned)` captures short-term momentum at $h=5$ ($CW = +1.425, p_{\text{raw}} = 0.0771, q_{\text{horizon}} = 0.1928, R^2_{OOS,\text{adj}} = +3.50\%$) and moderate multi-step structure at $h=20$ ($CW = +0.875, p_{\text{raw}} = 0.1908, R^2_{OOS,\text{adj}} = +2.90\%$).
+5. **Realized OOS Performance — Tuning Yields No Out-of-Sample Gain**: Holding the data vintage
+   fixed ($N = 3{,}728$ rolling origins), the leakage-safe tuned specification is statistically
+   indistinguishable from the untuned Issue #101 defaults:
+
+   | $h$ | RMSE (untuned defaults) | RMSE (tuned) | vs. Naïve (untuned) | vs. Naïve (tuned) |
+   |---|---|---|---|---|
+   | 1 | 0.029672 | 0.029690 | $-2.94\%$ | $-3.00\%$ |
+   | 5 | 0.065164 | 0.064930 | $-3.32\%$ | $-2.95\%$ |
+   | 20 | 0.129038 | 0.129240 | $-2.31\%$ | $-2.46\%$ |
+
+   Tuning is marginally worse at $h \in \{1, 20\}$ and marginally better at $h=5$; the 3-horizon
+   mean improvement changes by $+0.05$ pp. **This is the substantive result of Issue #119**, and it
+   is consistent with §4.2: if $\Delta s_t$ is a martingale difference sequence, no reweighting of a
+   gradient-boosted ensemble over the same causal information set can extract forecastable
+   structure, and hyperparameter search cannot manufacture it.
+
+   For transparency: an earlier revision of this section reported a $4.5\%$ $h=1$ MSPE reduction
+   from tuning. That figure was produced by a selection window that overlapped $76\%$ of the scored
+   forecast origins; it was selection bias, not signal, and it vanished once the window was
+   re-scoped to the pre-evaluation burn-in block. On the canonical $N=745$ sample the tuned
+   $R^2_{OOS}$ at $h=1$ is $-9.27\%$, essentially where the untuned benchmark stood.
```

If the tuned configuration cannot be shown to beat the defaults out-of-sample, the honest options
are (a) ship it anyway and report the null as above — my recommendation, it is the stronger
scientific result — or (b) revert `src/xgboost_baseline.py:57-64` to the canonical defaults and
present Issue #119 as a negative result. What should not ship is a §4.3 that reads as a success.

---

### 🟠 R2-2 — The search audit CSV is internally inconsistent: per-horizon columns are seed-42 only, summary columns are 5-seed means

**File:** `src/xgboost_grid_search.py:368` (`out = dict(seed_records[0])`);
`outputs/r3_xgboost_hyperparameter_search.csv`

`evaluate_config_across_seeds` copies the **first seed's** record wholesale, then overwrites only
`mean_relative_rmse`, `std_relative_rmse`, `mean_val_rmse`, `std_val_rmse` and
`val_improvement_pct`. Every per-horizon column — `val_rmse_h{1,5,20}`, `naive_rmse_h{…}`,
`rel_rmse_h{…}` — and the `seed` column survive from seed 42 alone.

Consequence: within a single row the parts do not sum to the whole.

| rank | `rel_rmse_h1` | `rel_rmse_h5` | `rel_rmse_h20` | mean of the three | reported `mean_relative_rmse` | Δ |
|---|---|---|---|---|---|---|
| 1 | 1.0116 | 1.0789 | 1.0238 | 1.038100 | 1.038096 | 0.000004 |
| 2 | 1.0189 | 1.1055 | 1.0090 | 1.044467 | 1.045429 | 0.000962 |
| **3** (canonical) | 1.0200 | 1.1350 | 1.0104 | **1.055133** | **1.050857** | **0.004276** |
| **7** | 1.0303 | 1.1323 | 1.0065 | **1.056367** | **1.062791** | **0.006424** |

**19 of 28 rows fail reconciliation** at a 1e-4 tolerance; the maximum discrepancy is **0.025114** —
larger than the entire rank-1-to-rank-8 spread (0.0267). Rows with `std_relative_rmse = 0.0`
reconcile exactly (deterministic configs, `subsample = colsample = 1.0`); every stochastic row does
not. The `seed,42` column on all 28 rows also directly contradicts `n_selection_seeds,5` in the
same row.

This does **not** change the winner — ranking uses `mean_relative_rmse`, which is correctly the
5-seed mean of per-seed means. But `outputs/r3_xgboost_hyperparameter_search.csv` is the
hyperparameter provenance artifact cited by §4.3, and an examiner who averages the three published
per-horizon ratios gets a **different ranking** than the published one (rank 3 and rank 7 swap).

```diff
     out = dict(seed_records[0])
-    mean_rel_vals = [r["mean_relative_rmse"] for r in seed_records]
-    mean_val_vals = [r["mean_val_rmse"] for r in seed_records]
+
+    # Average EVERY numeric metric across seeds, not just the three summary columns.
+    # Copying seed_records[0] wholesale and overwriting only the aggregates leaves the
+    # per-horizon columns at seed 42 while the summary is a 5-seed mean, so the row's
+    # parts stop reconciling with its whole -- and a reader re-averaging the published
+    # per-horizon ratios recovers a different ranking than the one shipped.
+    per_h_keys = [k for k in seed_records[0]
+                  if k.startswith(("val_rmse_h", "naive_rmse_h", "rel_rmse_h"))]
+    for k in per_h_keys:
+        vals = [r[k] for r in seed_records]
+        out[k] = round(float(np.mean(vals)), 6)
+        out[f"{k}_sd"] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else 0.0
+
+    # `seed` describes a single fit and is meaningless on an aggregated row.
+    out.pop("seed", None)
+    out["selection_seeds"] = ",".join(str(s) for s in seeds)
+
+    mean_rel_vals = [r["mean_relative_rmse"] for r in seed_records]
+    mean_val_vals = [r["mean_val_rmse"] for r in seed_records]
```

Add the reconciliation as a test so the artifact cannot drift out of self-consistency again:

```python
# tests/test_xgboost_grid_search.py
def test_search_csv_rows_reconcile_internally():
    """Every published row's per-horizon ratios must average to its published
    mean_relative_rmse. A mismatch means the row mixes single-seed detail with
    multi-seed aggregates, and the published ranking cannot be re-derived."""
    d = pd.read_csv(OUTPUTS_DIR / "r3_xgboost_hyperparameter_search.csv")
    per_h = d[[f"rel_rmse_h{h}" for h in (1, 5, 20)]].mean(axis=1)
    assert np.allclose(per_h, d["mean_relative_rmse"], atol=1e-4)
```

---

## 4. Medium-Severity Findings

### 🟡 R2-3 — The M-3 disclosure hardcodes coverage figures that were already stale in the same commit

**File:** `src/dashboard/app.py:126-134`

The new tooltip states *"realized coverage is 86.8/84.9/84.6 % at h=1/5/20"*. Those are the
**`8843c72`** numbers — the ones I quoted in Round 1. The same commit regenerated
`outputs/r3_xgboost_prediction_intervals.csv`:

| $h$ | tooltip says | artifact says | drift |
|---|---|---|---|
| 1 | 86.8 % | 86.78 % | ok |
| 5 | 84.9 % | **85.03 %** | wrong |
| 20 | 84.6 % | **84.01 %** | wrong |

My Round-1 diff supplied those literals as illustrative text; embedding them verbatim reproduces
exactly the doc-drift class that C-3 was raised for. Read them from the artifact:

```diff
-    help=(
-        "Split-conformal empirical prediction intervals for XGBoost. Finite-sample validity "
-        "requires exchangeable residuals, which overlapping h-step targets violate; realized "
-        "coverage is 86.8/84.9/84.6% at h=1/5/20 against a 90% nominal target "
-        "(outputs/r3_xgboost_prediction_intervals.csv). Treat as indicative, not guaranteed."
-    ),
+    help=_conformal_disclosure(),   # rendered from the artifact, never hardcoded
```

```python
# src/dashboard/data_loader.py
@st.cache_data(show_spinner=False)
def _conformal_disclosure() -> str:
    """Render the interval caveat from the shipped artifact so the coverage figures
    quoted to users can never drift from the numbers the pipeline produced."""
    iv = pd.read_csv(OUTPUTS_DIR / "r3_xgboost_prediction_intervals.csv").sort_values("horizon")
    cov = "/".join(f"{c:.1f}" for c in iv["empirical_coverage_90"])
    hs = "/".join(str(int(h)) for h in iv["horizon"])
    return (
        "Split-conformal empirical prediction intervals for XGBoost. Finite-sample validity "
        "requires exchangeable residuals, which overlapping h-step targets violate; realized "
        f"coverage is {cov}% at h={hs} against a 90% nominal target "
        "(outputs/r3_xgboost_prediction_intervals.csv). Treat as indicative, not guaranteed."
    )
```

Note also that h=20 coverage fell **84.55 % → 84.01 %** in this commit, leaving only 1.01 pp of
headroom above the new `>= 83.0` regression floor.

---

### 🟡 R2-4 — C-2 parts 1 and 2 remain outstanding; the restored invariant is pinned to an unrecorded vintage

**Files:** `tests/test_model_comparison.py:421`; `.gitignore:18-21`; no `outputs/data_manifest.json`

Restoring `assert (primary_df["n_forecasts"] == 745).all()` is the right move and closes part 3.
Parts 1 and 2 were not done: the vintage refresh was never split into its own commit, and there is
still no manifest recording which inputs produced the committed outputs. `data/processed/*` remains
gitignored.

The invariant is therefore now *stricter* but no more *attributable*: `745` is pinned to a dataset
that exists only on one machine. The next `python -m src.data_pipeline` run that shifts the sample
by one row fails the suite with no way to distinguish "expected refresh" from "pipeline regression".
The `write_data_manifest` / `outputs/data_manifest.json` proposal from Round 1 §C-2 part 2 is what
closes this; with it in place, `745` becomes checkable against a recorded sha256 rather than folklore.

---

### 🟡 R2-5 — Hardcoded FDR literals have now been rewritten twice in two commits

**File:** `tests/test_model_comparison.py:551`

```
8843c72:  expected_q = 0.9127 → 0.9177 / 0.2150 → 0.6755 / 0.2278 → 0.2788
f8f3297:  expected_q = 0.9177 → 0.9446 / 0.6755 → 0.1928 / 0.2788 → 0.2788
```

This test reads the committed CSV and asserts three magic numbers, so it re-breaks on every
regeneration and is edited to match — which means it can never detect the thing it exists to detect.
Assert the FDR *contract* instead, which is what the m=15 family claim actually rests on:

```diff
-        # Verify that XGBoost's displayed q-value originates from the unified m=15 family contract
-        xgb_row = cw[cw["model_key"] == "xgboost"].iloc[0]
-        expected_q = 0.9446 if h == 1 else (0.1928 if h == 5 else 0.2788)
-        assert np.isclose(xgb_row["cw_p_adj_horizon"], expected_q, atol=1e-3)
+        # Verify XGBoost's q-value obeys the unified m=15 BH contract, rather than
+        # pinning a literal that has to be rewritten on every regeneration (and so
+        # can never fail for the reason it exists).
+        xgb_row = cw[cw["model_key"] == "xgboost"].iloc[0]
+        p, q = xgb_row["cw_p_value"], xgb_row["cw_p_adj_horizon"]
+        assert q >= p - 1e-12, "BH adjustment must be non-decreasing in p"
+        assert q <= 1.0 + 1e-12
+        # Horizon-stratified BH over m=5 models: q_i >= p_i * m / rank_i
+        ranks = cw["cw_p_value"].rank(method="min")
+        r = float(ranks[cw["model_key"] == "xgboost"].iloc[0])
+        assert q >= p * len(cw) / r - 1e-6
```

---

### 🟡 R2-6 — Selection now rests on 77–84 validation rows from a single calm regime, and the winner is ~1.2 sd from runner-up

**Files:** `outputs/r3_xgboost_search_partitions.csv`; `src/xgboost_grid_search.py:443-447`

This is the accepted cost of the C-1 fix — Round 1 offered exactly this trade-off and the team took
the burn-in option. It is the *correct* choice on leakage grounds and I am not asking to reverse it.
But its consequences should be stated in §4.3 rather than left implicit:

- `inner_val` is **84 / 83 / 77** rows at h = 1 / 5 / 20, all inside **2011-05 → 2011-11**: one
  post-GFC, pre-taper-tantrum, near-zero-policy-rate regime. The frozen configuration then governs
  2012–2026 including COVID and the 2022 tightening.
- 28 candidates are ranked on ~84 validation points. `mean_relative_rmse` degraded from 1.011 to
  **1.038** (rank 1) and h=5 relative RMSE is now **1.079–1.148** — the selection objective is far
  from the evaluation behaviour.
- Rank 1 (1.038096, sd 0.000000) vs rank 2 (1.045429, sd 0.006101): gap 0.00733 ≈ **1.2 sd**,
  ≈ 2.7 standard errors of the 5-seed mean. Weakly identified.
- Rank 1, 4, 5 all have `std_relative_rmse` **exactly 0** because `subsample = colsample = 1.0`
  makes them deterministic. The multi-seed check therefore provides *no* robustness information for
  the configuration that won, while penalising nothing in the stochastic ones — the comparison is
  asymmetric by construction.
- The tie-break Round 1 asked for (ties **within 1 sd** → parsimony) was implemented as an
  exact-float tie-break on `["mean_relative_rmse", "max_depth", "n_estimators"]`, which with
  6-decimal rounding effectively never fires. Under a 1-sd rule, rank 1 and rank 2 tie and the more
  parsimonious wins on `n_estimators`.

```diff
-    results_df = results_df.sort_values(
-        ["mean_relative_rmse", "max_depth", "n_estimators"],
-        ascending=[True, True, True],
-    ).reset_index(drop=True)
+    # Rank on the seed-mean, then collapse configurations whose objective is within
+    # 1 sd of the leader into a single indifference band and prefer the most
+    # parsimonious member. A 7e-3 gap against a 6e-3 seed sd on ~84 validation rows
+    # does not identify a winner; breaking exact float ties alone never fires.
+    results_df = results_df.sort_values("mean_relative_rmse").reset_index(drop=True)
+    band = float(results_df["std_relative_rmse"].replace(0.0, np.nan).median())
+    leader = float(results_df["mean_relative_rmse"].iloc[0])
+    results_df["within_1sd_of_leader"] = (
+        results_df["mean_relative_rmse"] <= leader + (band if np.isfinite(band) else 0.0)
+    )
+    results_df = results_df.sort_values(
+        ["within_1sd_of_leader", "n_estimators", "max_depth", "mean_relative_rmse"],
+        ascending=[False, True, True, True],
+    ).reset_index(drop=True)
```

Add to §4.3:

```diff
+   *Selection-power caveat*: the leakage-safe burn-in block yields only 77–84 validation
+   observations (2011-05 → 2011-11), a single low-volatility regime. The rank-1 configuration
+   leads rank 2 by $7.3\times10^{-4}$ against a per-seed $\sigma$ of $6.1\times10^{-3}$, so the
+   ordering inside the leading band is not identified; the frozen specification should be read as
+   "a member of the indifference set", not as an optimum. This is the unavoidable cost of
+   guaranteeing zero overlap with the scored origins, and is the reason §4.3 item 5 reports no
+   out-of-sample gain.
```

---

### 🟡 R2-7 — The regularization conclusion reversed 180°, and §4.3 keeps only the prose that still fits

**File:** `docs/CLARK_WEST_METHODOLOGY.md:154-159`

| | `8843c72` winner | `f8f3297` winner |
|---|---|---|
| `subsample` / `colsample_bytree` | 0.7 / 0.7 | **1.0 / 1.0** |
| `min_child_weight` / `reg_lambda` | 5.0 / 5.0 | **1.0 / 0.5** |
| Round-1 seed study said | tier-0 beats tier-2 by ~0.010 at **every** seed (~15σ) | tier-2 now wins; tier-0 falls to **rank 6** |

The winner moved from the most-regularized tier to the least-regularized one. §4.3 retained the two
bullets that survive (`max_depth = 2` "shallow trees suppress variance", `learning_rate = 0.01`
"conservative gradient shrinkage") and silently dropped the regularization rationale, leaving
`subsample = 1.0 & colsample_bytree = 1.0` and `min_child_weight = 1.0 & reg_lambda = 0.5` bare.

Keeping the narrative that fits and deleting the one that does not is the same selective-reporting
pattern as R2-1. The reversal is itself informative and should be reported:

```diff
-   - `subsample = 1.0` & `colsample_bytree = 1.0`
-   - `min_child_weight = 1.0` & `reg_lambda = 0.5`
+   - `subsample = 1.0` & `colsample_bytree = 1.0` (no bagging)
+   - `min_child_weight = 1.0` & `reg_lambda = 0.5` (weakest L2 tier in the space)
+
+   The regularization tier that wins is **sensitive to the selection window**: on a (leaky)
+   pre-2023 pool the most-regularized tier (`subsample = 0.7`, `min_child_weight = 5`,
+   `reg_lambda = 5`) dominated at every seed, whereas on the leakage-safe 2010–2012 burn-in block
+   the least-regularized tier wins and the former falls to rank 6. With ~84 validation
+   observations the search cannot distinguish regularization regimes, which is consistent with
+   item 5: no configuration in the space carries out-of-sample signal.
```

---

## 5. Low-Severity

| # | File:line | Issue |
|---|---|---|
| R2-8 | `tests/test_xgboost_grid_search.py:186` | `assert saved_df["is_canonical"].sum() == 1` was **deleted** rather than adapted. I ran it against the current code — it still returns `1` and would still pass. Coverage of the canonical-flagging logic (which `compare_best_to_canonical` depends on) was given up for nothing. Restore it. |
| R2-9 | `src/xgboost_grid_search.py:349-352` | `evaluate_config_across_seeds` runs all 5 seeds even for deterministic configs (`subsample = colsample = 1.0`), 5× wasted compute for a guaranteed `sd = 0`. Short-circuit when `subsample == 1.0 and colsample_bytree == 1.0`, or state in the record that dispersion is structurally zero. |
| R2-10 | `app.py:121,528`, `charts.py:11,37` vs `data_loader.py:305`, `model_comparison.py:792,961` | **L-1 not fixed** despite the commit message ("Standardize labels"). The sidebar, ribbon and spec table still say `"XGBoost"`; metrics and Clark-West tables say `"XGBoost (Tuned)"`. Only the spec-table *parameter string* was updated. |
| R2-11 | `src/xgboost_baseline.py:32` | **L-3 not fixed** despite the commit message ("harmonize imports"). `xgboost_baseline` still does bare `from gold_feature_pipeline import …` while `xgboost_grid_search:49` does `from src.xgboost_baseline import …`; `tests/test_xgboost_grid_search.py:23` imports the bare form. Two module objects, two `_CACHED_GOLD_DF` caches, two `HAS_XGBOOST` flags remain reachable. |
| R2-12 | `docs/CLARK_WEST_METHODOLOGY.md:143` | §4.2 now says LSTM and XGBoost "**both** capture short-term nonlinear momentum" on the strength of $p_{\text{raw}} = 0.0771$, $q_{\text{horizon}} = 0.1928$. Neither is significant after FDR control, and §5 says so two sections later. Soften to "show the largest, though non-significant, adjusted MSPE reductions". |

---

## 6. What Improved

- **C-1 is a real fix, not a cosmetic one.** Index-bounded selection, a machine-readable partition
  manifest, and independently verified zero overlap. This was the blocking issue and it is closed.
- **M-1 is exemplary.** Every silent leaky fallback replaced by a `ValueError` carrying the actual
  row counts, plus `test_undersized_pool_raises_value_error_without_leaky_fallback` asserting on the
  message. Failing loudly is the right default for a partitioning routine.
- **H-1 fixed properly.** The embargo test now calls the shipped function and asserts on the
  returned partition indices; the `>= h` relaxation is correct given `inner_train_end_idx` is an
  exclusive bound (true gap is `h + 1` rows).
- **H-2 fixed with intellectual honesty.** §4.3 item 4 now states outright that *no* candidate beat
  the Naïve benchmark on the selection window. That sentence is the most valuable addition in the
  commit, and it is exactly what the EMH framing in §4.2 predicts.
- **M-2 clean.** One `resolved` dict, sourced from `xgboost_baseline` constants, used for both the
  fit and the logged record.
- The `(record, partitions)` / `(results_df, partitions_df)` signature change was propagated to
  every call site and test without a miss — 73/73 pass.

---

## 7. Merge Recommendation

**Approve with required changes — no longer blocking on leakage.**

Must fix before merge (reporting integrity, not correctness of the pipeline):
1. **R2-1** — restore the tuned-vs-untuned comparison to §4.3 item 5 and report the null result.
   Numbers supplied above; no re-run needed.
2. **R2-2** — average all per-horizon metrics across seeds so published rows reconcile, drop the
   misleading `seed` column, add the reconciliation test.
3. **R2-3** — render the coverage disclosure from the artifact instead of hardcoding stale literals.

Should fix in this PR:
4. **R2-4** (data manifest), **R2-5** (assert the BH contract, not literals), **R2-6** (1-sd
   indifference band + selection-power caveat), **R2-7** (report the regularization reversal).

Housekeeping: **R2-8** restore the deleted assertion; **R2-10 / R2-11** finish L-1 and L-3, which
the commit message claims are done but are not.

The methodology is now sound. What remains is making the write-up say what the numbers say.
