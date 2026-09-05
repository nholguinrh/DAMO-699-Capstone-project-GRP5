# Code Review Round 3 — `feat/119-xgboost-hyperparameter-tuning` (Issue #119)

**Reviewer role:** Lead Data Scientist / Principal Analytics Engineer
**Commit under review:** `e9790be` — *"fix(xgboost): reconcile search metrics, report null OOS gain, and add data manifest (PR #119 review round 2)"*
**Baseline for this round:** `f8f3297` (state reviewed in `docs/PR_119_CODE_REVIEW_ROUND2.md`)
**Verification:** every claim below was re-derived by executing the shipped code against the shipped artifacts. Full suite: **175 passed**.

---

## 1. Methodological Risk Assessment: **LOW** (was **MEDIUM**, originally **CRITICAL**)

The methodology is sound and the reporting is now honest. §4.3 item 5 does something rare and
correct: it states the null result, quantifies it, **and explicitly retracts the earlier inflated
claim** with the reason. That paragraph is the strongest part of this PR.

One regression: **the `rank` column no longer orders configurations by the selection objective**
(R3-1) — and that is my fault. The 1-sd indifference-band diff I supplied in Round 2 §R2-6 applies
the parsimony ordering *globally* instead of *within the band*; it was applied verbatim and it
scrambled ranks 2–28. The band contains exactly one configuration here, so the change delivered no
benefit and pure damage. Corrected code below. It has propagated three wrong rank references into
§4.3, but it does **not** affect the winner, the frozen constants, or any downstream artifact.

| Round-2 finding | Status | Verified how |
|---|---|---|
| **R2-1** OOS null deleted from §4.3 | ✅ **Fixed, exemplary** | Item 5 restored with full table + retraction of the earlier 4.5 % claim |
| **R2-2** search rows not reconciling | ✅ **Fixed** | **0 of 28** rows fail at atol 1e-4 (was 19/28); max Δ **1.9e-5**; `seed` dropped, `selection_seeds` added |
| **R2-3** hardcoded stale coverage | ✅ **Fixed** | `get_conformal_disclosure()` renders from the artifact, with a graceful missing-file path |
| **R2-4** data manifest | ⚠️ **Partial** | Manifest exists but covers 1 of 4 inputs, is emitted by one pipeline only, and nothing asserts it (R3-4) |
| **R2-5** brittle q-value literals | ✅ **Fixed, improved** | BH contract asserted — and my assertion direction was wrong; they corrected it (see §5) |
| **R2-6** 1-sd indifference band | ❌ **Regression** | Ranking non-monotonic in the objective (R3-1) |
| **R2-7** regularization reversal | ✅ **Fixed** | Documented in §4.3, though the rank it cites is now stale (R3-1) |
| **R2-8** deleted `is_canonical` assertion | ✅ **Restored** | `tests/test_xgboost_grid_search.py:207` |
| **R2-9** deterministic short-circuit | ⚠️ **Partial** | Works, but `n_selection_seeds` still reports 5 (R3-2) |
| **R2-10** split labels | ✅ **Fixed** | `XGB_DISPLAY_LABEL` routed through `charts.py:5`, `app.py:42,123,128,525` |
| **R2-11** dual module identity | ⚠️ **Partial** | `try: from src.… except ImportError:` shim; both module objects still reachable |
| **R2-12** overstated momentum claim | ✅ **Fixed** | Now "the largest, though non-significant, adjusted MSPE reductions" |

**Downstream artifacts are consistent — no re-run needed.** I verified that rank 1 is unchanged
(`depth 2, lr 0.01, n_est 600, subsample 1.0, colsample 1.0, mcw 1.0, λ 0.5`) and matches
`src/xgboost_baseline.py:63-70` exactly. That is why `r3_xgboost_forecasts.csv`,
`clark_west_test_results.csv`, the SHAP tables and the interval artifact were correctly **not**
regenerated in this commit. Fixing R3-1 changes only the `rank` column of one CSV and three
sentences of prose.

---

## 2. Critical / Blocking Findings

**None.** No leakage, no metric corruption, no non-determinism, no row duplication.

---

## 3. High-Severity Finding

### 🟠 R3-1 — The `rank` column is no longer monotonic in the selection objective *(regression caused by my Round-2 diff)*

**File:** `src/xgboost_grid_search.py:474-484`; `outputs/r3_xgboost_hyperparameter_search.csv`;
`docs/CLARK_WEST_METHODOLOGY.md:155, 162, 164`

The shipped ranking:

| rank | depth | lr | n_est | subsample | `mean_relative_rmse` | in band | canonical |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 0.01 | 600 | 1.0 | **1.038096** | True | |
| 2 | 2 | 0.10 | 150 | 1.0 | 1.083673 | False | |
| 3 | 2 | 0.10 | 150 | 0.8 | 1.113384 | False | |
| 4 | 2 | 0.10 | 150 | 0.7 | 1.130194 | False | |
| **5** | 3 | 0.03 | 150 | 0.8 | **1.050857** | False | **✔** |

`rank` 5 has a **lower** objective than ranks 2, 3 and 4. Measured: the ordering is
**non-monotonic at 8 of 27 adjacent pairs**.

**Root cause — my error.** The Round-2 diff read:

```python
results_df.sort_values(["within_1sd_of_leader", "n_estimators", "max_depth", "mean_relative_rmse"],
                       ascending=[False, True, True, True])
```

`within_1sd_of_leader` is only the *first* key, so every configuration **outside** the band is also
ordered by `n_estimators` then `max_depth`, with the objective demoted to a fourth-level tie-break.
Parsimony ordering is only meaningful *inside* an indifference band; outside it, ordering must be by
the objective. Compounding this, the band here contains exactly **one** configuration (leader
1.038096, band width = median non-zero seed sd ≈ 0.0066, next objective 1.050857), so the tie-break
never fires and the change is pure damage.

**Analytical risk.** `rank` is the column §4.3 and any examiner reads as "how the search ordered the
candidates". Three claims in §4.3 are now wrong against the artifact:

| §4.3 claim | Line | Artifact says |
|---|---|---|
| canonical baseline at "**Rank 3**, 1.0509" | 155 | **Rank 5** (objective-rank would be 3) |
| "rank-1 leads **rank 2** by $7.3\times10^{-4}$ against $\sigma = 6.1\times10^{-3}$" | 162 | rank 2 is now 1.083673 (σ = 0); the true runner-up is canonical at 1.050857, gap **1.28×10⁻²** |
| former tier-0 winner "falls to **rank 6**" | 164 | it sits at **rank 22**; its objective-rank is 3 (1.058223) |

There is also a latent correctness issue: `compare_best_to_canonical` takes `results_df.iloc[0]` as
`best_config`. Today that is still the objective minimum by coincidence — the sole band member is
also the global minimum. With two or more band members, `iloc[0]` becomes the most parsimonious band
member while the field is still documented as "top-ranked winning configuration", and
`relative_gain_pct` would then be computed against a non-minimal "best".

**Fix — apply parsimony only within the band:**

```diff
     # Rank on seed-mean, then collapse configurations within 1 sd of leader into indifference band (R2-6)
     results_df = results_df.sort_values("mean_relative_rmse").reset_index(drop=True)
     band = float(results_df["std_relative_rmse"].replace(0.0, np.nan).median())
     leader = float(results_df["mean_relative_rmse"].iloc[0])
     results_df["within_1sd_of_leader"] = (
         results_df["mean_relative_rmse"] <= leader + (band if np.isfinite(band) else 0.0)
     )
-    results_df = results_df.sort_values(
-        ["within_1sd_of_leader", "n_estimators", "max_depth", "mean_relative_rmse"],
-        ascending=[False, True, True, True],
-    ).reset_index(drop=True)
+    # Parsimony is a tie-break WITHIN the indifference band only. Outside it the
+    # objective is the ordering, full stop -- sorting the whole frame by
+    # n_estimators/max_depth demotes the objective to a fourth-level key and makes
+    # `rank` non-monotonic in the very quantity it claims to rank.
+    in_band = (
+        results_df[results_df["within_1sd_of_leader"]]
+        .sort_values(["n_estimators", "max_depth", "mean_relative_rmse"])
+    )
+    out_band = (
+        results_df[~results_df["within_1sd_of_leader"]]
+        .sort_values("mean_relative_rmse")
+    )
+    results_df = pd.concat([in_band, out_band], ignore_index=True)
     results_df["rank"] = range(1, len(results_df) + 1)
```

Guard it so the invariant cannot silently break again:

```python
# tests/test_xgboost_grid_search.py
def test_rank_is_monotonic_in_objective_outside_the_indifference_band():
    """Parsimony re-ordering applies only inside the 1-sd band. Every configuration
    outside it must be ranked by the selection objective, or `rank` stops meaning
    what §4.3 and the CSV header claim it means."""
    d = pd.read_csv(OUTPUTS_DIR / "r3_xgboost_hyperparameter_search.csv")
    outside = d[~d["within_1sd_of_leader"]].sort_values("rank")
    assert outside["mean_relative_rmse"].is_monotonic_increasing
    # The band leader must still be the global objective minimum.
    assert np.isclose(d["mean_relative_rmse"].min(),
                      d.loc[d["within_1sd_of_leader"], "mean_relative_rmse"].min())
```

After re-running, update §4.3 lines 155 / 162 / 164 to the corrected ranks. The prose claims
themselves stay true — canonical really is the objective runner-up, and tier-0 really did fall —
only the rank integers change.

---

## 4. Medium-Severity Findings

### 🟡 R3-2 — `n_selection_seeds` reports 5 for rows evaluated on 1 seed

**File:** `src/xgboost_grid_search.py:409` (`out["n_selection_seeds"] = len(seeds)`); lines 354-359

The R2-9 short-circuit is correct in substance — with `subsample = colsample = 1.0` the fit is
deterministic and 5 seeds is 5× wasted compute for a guaranteed `sd = 0`. But the record is written
from `len(seeds)`, not from what was executed. The shipped CSV therefore claims
`n_selection_seeds = 5` and `selection_seeds = "42,0,1,7,2024"` for the **rank-1 winning
configuration**, which was fitted **once**, on seed 42 alone.

That is the same defect class R2-2 just fixed: an audit column that does not describe what ran — and
it now overstates the robustness evidence for precisely the configuration that was frozen into
production. The saving is real; only the disclosure is wrong.

```diff
-    out["n_selection_seeds"] = len(seeds)
+    # Report what was actually executed. Deterministic configurations are fitted once
+    # (dispersion is structurally zero, not empirically estimated); claiming 5 seeds
+    # for a single fit overstates the robustness evidence for the frozen winner.
+    out["n_selection_seeds"] = len(eval_seeds)
+    out["selection_seeds"] = ",".join(str(s) for s in eval_seeds)
+    out["dispersion_basis"] = "deterministic" if is_deterministic else "multi_seed"
```

and in §4.3 item 4:

```diff
-   Evaluated across five independent seeds `(42, 0, 1, 7, 2024)` to guard against seed noise.
+   Evaluated across five independent seeds `(42, 0, 1, 7, 2024)` to guard against seed noise.
+   Configurations with `subsample = colsample_bytree = 1.0` are deterministic by construction and
+   are fitted once; their reported dispersion is structurally zero rather than empirically
+   estimated, and `dispersion_basis` in the search CSV records which basis applies to each row.
+   The frozen rank-1 specification is of this deterministic class.
```

---

### 🟡 R3-3 — The transparency paragraph misstates the tuned $R^2_{OOS}$

**File:** `docs/CLARK_WEST_METHODOLOGY.md:176`

> *"On the canonical $N=745$ sample, the tuned $R^2_{OOS}$ at $h=1$ is $-9.03\%$ (unadjusted) / $-9.27\%$ (adjusted)…"*

From `outputs/clark_west_test_results.csv`:

| model | h | N | `r2_oos` | `r2_oos_adj` |
|---|---|---|---|---|
| XGBoost (Tuned) | 1 | 745 | **−9.2681 %** | **−4.0235 %** |

Both figures are wrong: −9.27 % is the *realized/unadjusted* value (the doc labels it "adjusted"),
the adjusted value is −4.02 %, and −9.03 % is a stale number from the pre-#119 `main` table at
$N = 741$ — the very vintage-confound this paragraph exists to disclose.

```diff
-   On the canonical $N=745$ sample, the tuned $R^2_{OOS}$ at $h=1$ is $-9.03\%$ (unadjusted) / $-9.27\%$ (adjusted), essentially where the untuned benchmark stood.
+   On the canonical $N=745$ sample, the tuned $h=1$ figures are $R^2_{OOS} = -9.27\%$ (realized) and
+   $-4.02\%$ after the Clark-West adjustment, essentially where the untuned benchmark stood.
```

Given this paragraph's purpose is to correct a previously misstated number, it is the one place in
the document where a stale figure does the most damage.

---

### 🟡 R3-4 — The data manifest is written but never verified, and covers one of four inputs

**Files:** `src/project_paths.py:57-96`; `src/xgboost_baseline.py:680-682`; `outputs/data_manifest.json`

`write_data_manifest` works and the emitted JSON is well-formed (sha256, size, 4 268 rows,
2010-02-19 → 2026-06-30). Four gaps against R2-4's purpose — making `N = 745` *attributable*:

1. **One input recorded.** Only `gold_features.csv`. The three upstream files
   (`bank_of_canada_data.csv`, `fred_rates.csv`, `statcan_cpi.csv`) are absent, so a raw-data
   refresh that happens to leave the gold layer byte-identical is invisible, and the gold hash
   attests to nothing about its own provenance.
2. **One pipeline emits it.** Only `xgboost_baseline.run_pipeline`. `model_comparison` — which
   produces `clark_west_test_results.csv`, the file carrying the `== 745` invariant — writes none.
3. **Nothing asserts it.** No test ties the manifest to any artifact, so `745` is still pinned to
   folklore rather than to a recorded digest. This was the substance of R2-4.
4. **`except Exception: pass`** (`project_paths.py:86-87`) silently drops `row_count` and the date
   bounds on any CSV read failure, producing a manifest that looks valid but attests to less than it
   appears to.

```diff
-        gold_csv = PROCESSED_DIR / "gold_features.csv"
-        if gold_csv.exists():
-            write_data_manifest([gold_csv], out_dir / "data_manifest.json")
+        # Record the whole input chain, not just its last link: a gold-layer hash
+        # cannot attest to the raw vintage it was derived from.
+        write_data_manifest(
+            [
+                PROCESSED_DIR / "gold_features.csv",
+                PROCESSED_DIR / "bank_of_canada_data.csv",
+                PROCESSED_DIR / "fred_rates.csv",
+                PROCESSED_DIR / "statcan_cpi.csv",
+            ],
+            out_dir / "data_manifest.json",
+        )
```

```diff
-            except Exception:
-                pass
+            except Exception:  # noqa: BLE001 - manifest must be loud, not decorative
+                logger.warning("Could not read %s for manifest row/date bounds", p, exc_info=True)
+                entry["row_count"] = None
```

and the test that makes `745` mean something:

```python
# tests/test_model_comparison.py
EXPECTED_GOLD_SHA256 = "91979adb6c3b71a06bde7f58db6ddadc83b4c2e3ebe8783363e6e8709ce84e82"

def test_common_sample_size_is_attributable_to_the_recorded_vintage():
    """`n_forecasts == 745` is only meaningful if we know which inputs produced it.
    If this fails on the sha but not the count (or vice versa), the vintage moved and
    the committed outputs must be regenerated in the same commit."""
    manifest = json.loads((OUTPUTS_DIR / "data_manifest.json").read_text())
    assert manifest["gold_features.csv"]["sha256"] == EXPECTED_GOLD_SHA256
    cw = pd.read_csv(OUTPUTS_DIR / "clark_west_test_results.csv")
    assert (cw["n_forecasts"] == 745).all()
```

Also call `write_data_manifest` from `model_comparison`'s pipeline entry point.

---

## 5. Low-Severity

| # | File:line | Issue |
|---|---|---|
| R3-5 | `outputs/r3_xgboost_hyperparameter_search.csv` | `naive_rmse_h1_sd`, `naive_rmse_h5_sd`, `naive_rmse_h20_sd` are structurally `0.0` on every row — the naïve RMSE is a property of the validation targets and cannot vary by seed. Exclude `naive_rmse_h*` from the `_sd` emission in `evaluate_config_across_seeds`; three dead columns in a 36-column audit artifact. |
| R3-6 | `docs/CLARK_WEST_METHODOLOGY.md:162` | The selection-power caveat quotes `f8f3297`'s figures (gap $7.3\times10^{-4}$, $\sigma = 6.1\times10^{-3}$). Under the current CSV the leader-to-runner-up gap is $1.28\times10^{-2}$ against the leader's structural $\sigma = 0$. Refresh together with the R3-1 rank fix. |
| R3-7 | `src/xgboost_baseline.py:31-36` | R2-11 is a shim, not a fix: `try: from src.… except ImportError: from …` still leaves `xgboost_baseline` and `src.xgboost_baseline` as two importable module objects with independent `_CACHED_GOLD_DF` and `HAS_XGBOOST` (`tests/test_xgboost_grid_search.py:23` still uses the bare form while `xgboost_grid_search.py:49` uses `src.`). Harmless today; the real fix is a package `__init__.py` and one import style. |
| R3-8 | `src/project_paths.py:96` | No trailing newline (`\ No newline at end of file`). |
| R3-9 | `src/dashboard/charts.py:39,77` | `XGB_DISPLAY_LABEL` is routed correctly, but both sites retain a literal `"XGBoost"` fallback. Defensible for stale Streamlit session state — worth a one-line comment saying that is why, otherwise it reads as an incomplete rename. |

---

## 6. What Improved

- **§4.3 item 5 is the best thing in this PR.** It reports the null result with a full
  same-vintage table, names it "the substantive econometric result of Issue #119", ties it back to
  the EMH argument in §4.2, **and retracts the earlier 4.5 % claim with the reason it was wrong**.
  Publishing a negative result and explaining a withdrawn one is exactly the standard this kind of
  benchmark work should be held to.
- **R2-2 fully closed.** 0 of 28 rows fail reconciliation (was 19/28), max residual 1.9e-5 —
  pure rounding. `seed` dropped, `selection_seeds` added, per-horizon `_sd` columns published.
- **R2-5 fixed better than I specified.** My Round-2 assertion `q >= p * m / r` had the inequality
  **backwards** — BH step-up gives $q_i = \min_{j \ge i} (m/j)p_j \le (m/i)p_i$. The implementation
  corrected the direction *and* added an independent `statsmodels.multipletests` recomputation as
  the primary check. That is a stronger test than the one I proposed.
- **R2-3 done properly**, including a graceful fallback string when the artifact is absent, so the
  dashboard cannot crash on a fresh clone.
- **R2-10 clean.** A single `XGB_DISPLAY_LABEL` constant in `charts.py` routed through all four
  call sites, rather than four independent edits.
- **175/175 tests pass**, up from 73 — including the new reconciliation and restored
  `is_canonical` guards.

---

## 7. Merge Recommendation

**Approve after R3-1.** The methodology is sound and the write-up is honest; what remains is one
presentation regression plus disclosure accuracy.

Must fix before merge:
1. **R3-1** — restore objective-monotonic ranking outside the indifference band (corrected diff
   above, and the bug is mine), re-export the search CSV, and refresh the three rank references in
   §4.3 lines 155 / 162 / 164. No model re-runs required: the winner and the frozen constants do
   not change, and I verified the downstream artifacts stay consistent.
2. **R3-3** — correct the $R^2_{OOS}$ figures in the transparency paragraph to −9.27 % realized /
   −4.02 % adjusted.

Should fix in this PR:
3. **R3-2** — report `n_selection_seeds` as executed, add `dispersion_basis`.
4. **R3-4** — record all four inputs, emit from `model_comparison` too, and add the test that binds
   `745` to the recorded sha256. Without that test the manifest is decorative.

Housekeeping: R3-5 through R3-9.

With R3-1 and R3-3 applied, this PR is ready to merge and the Issue #119 result — that leakage-safe
hyperparameter tuning yields **no** out-of-sample gain over untuned defaults — is a defensible,
well-evidenced contribution to the capstone.
