# Code Review — `feat/119-xgboost-hyperparameter-tuning` (Issue #119)

**Reviewer role:** Lead Data Scientist / Principal Analytics Engineer
**Commits reviewed:** `ce24a26`, `36fcdd7`, `8843c72` (vs `main` @ `ce85758`)
**Scope:** `src/xgboost_grid_search.py` (new, 438 LOC), `src/xgboost_baseline.py`, `src/model_comparison.py`, `src/dashboard/*`, `tests/*`, `docs/CLARK_WEST_METHODOLOGY.md`, 8 regenerated `outputs/*.csv`

---

## 1. Methodological Risk Assessment: **CRITICAL**

The engineering is clean, the search is bit-for-bit deterministic, and the split-conformal
denominator fix (`ce24a26`) is correct and well-tested. **However, the central claim of the PR —
that hyperparameters were selected on a hold-out disjoint from the evaluation set — is false as
implemented.** The selection window (pre-2023) covers **76.1 % of the rolling-CV test origins**
that the headline Clark-West / RMSE / regime tables score as out-of-sample, and **100 % of the
inner-validation rows used to pick the winning configuration are themselves scored test origins.**

Compounding this, the branch bundles an **un-versioned refresh of `data/processed/`** with the
tuning change, so the reported "tuning improved h=1 MSPE from 0.000921 to 0.000880" is not
identified — the naïve benchmark MSPE also moved (0.000845 → 0.000843), which is impossible under
a pure hyperparameter change.

The numbers in the branch are **internally consistent and reproducible on this machine**, but they
**do not measure what the documentation says they measure**. The `XGBoost (Tuned)` row of Table 1
is an in-sample-selected result presented as out-of-sample. **Do not merge as-is.**

| Dimension | Verdict |
|---|---|
| Methodological integrity / statistical rigor | ❌ **Critical** — selection leakage over 76 % of the evaluation set |
| Reproducibility | ❌ **Critical** — outputs derive from gitignored, re-pulled inputs; effect confounded |
| Determinism | ✅ **Pass** — `n_jobs=1`, seed threaded, re-run reproduces the CSV exactly |
| Test coverage | ⚠️ **Medium** — the leakage guarantee is asserted by a test that re-implements it |
| Interval calibration | ⚠️ **Medium** — coverage degraded at every horizon and both nominal levels |

---

## 2. Critical / Blocking Findings

### 🔴 C-1 — Selection leakage: the "pre-2023 hold-out" *is* the out-of-sample evaluation set

**Files:** `src/xgboost_grid_search.py:9-11, 168, 186-191`; `src/xgboost_baseline.py:51-52, 227-250`;
`docs/CLARK_WEST_METHODOLOGY.md:148-151`

The module docstring and §4.3 assert the selection window ends "before the out-of-sample regime
evaluation period", "guaranteeing zero contamination of the 2023–2026 test evaluation window".
The evaluation window is **not** 2023–2026. `make_rolling_folds(n_samples=4228, min_train=500,
n_folds=5)` places the first test origin at index 500:

```
Rolling-CV test origins (outputs/r3_xgboost_forecasts.csv), per horizon:
  fold 0: 2012-02-17 .. 2014-12-25   (745)
  fold 1: 2014-12-26 .. 2017-11-02   (745)
  fold 2: 2017-11-03 .. 2020-09-10   (745)
  fold 3: 2020-09-11 .. 2023-07-20   (745)
  fold 4: 2023-07-21 .. 2026-06-02   (748)
```

Measured overlap with the tuning partitions actually produced by
`evaluate_inner_split_config` on the real data (`n_pool = 3336` pre-2023 rows):

| Partition | Actual dates | Rows | Also a scored test origin |
|---|---|---|---|
| `inner_train` (h=1) | 2010-03-19 .. 2018-11-26 | 2267 | 1767 / 2267 |
| **`inner_val` (h=1)** | **2018-11-28 .. 2021-01-27** | **566** | **566 / 566 (100 %)** |
| `inner_val` (h=5) | 2018-11-20 .. 2021-01-18 | 565 | 565 / 565 |
| `inner_val` (h=20) | 2018-10-19 .. 2020-12-09 | 559 | 559 / 559 |
| conformal calibration | 2021-01-29 .. 2022-12-29 | 500 | 500 / 500 |

- **2836 / 3728 (76.1 %)** of rolling-CV test origins lie inside the selection pool.
- **567 / 745 (76.1 %)** of the reconciled Clark-West common-sample origins lie inside it.
- **113 / 745 (15.2 %)** of the Clark-West origins lie inside the h=1 `inner_val` block — the exact
  rows whose RMSE chose `max_depth=2, lr=0.01, n_est=600, subsample=0.7, mcw=5, λ=5`.

**Analytical risk.** Every `XGBoost (Tuned)` figure in Table 1, `r3_xgboost_vs_naive.csv`,
`r3_regime_segmented_metrics.csv` and the dashboard carries an optimistic selection bias of unknown
magnitude. The Clark-West statistic assumes the larger model's specification was fixed *ex ante* of
the forecast origins; here it was chosen to minimise error on those origins. The CW/BH inference is
therefore anti-conservative for the XGBoost arm only, breaking the like-for-like comparison against
ARIMA/VAR/VECM and against `LSTM (Tuned)` (whose Issue #90 protocol needs the same audit).
Note the h=5 CW statistic moved from +1.366 to +0.240 and h=20 from +1.334 to +1.067 — the paper's
"XGBoost captures short-term nonlinear momentum" claim (line 143) is no longer supported by its own
numbers even before the bias correction.

**Required fix — the selection window must end before the first scored origin.** The only
partition boundary that satisfies this is `min_train`:

```diff
--- a/src/xgboost_grid_search.py
+++ b/src/xgboost_grid_search.py
@@
+# The rolling-CV evaluation scores every origin from index MIN_TRAIN onward
+# (2012-02-17 .. 2026-06-02). Any row at or beyond that index is an evaluation
+# origin, so hyperparameter selection must be confined strictly below it --
+# a calendar cut-off such as 2023-01-01 leaks 76% of the evaluation set.
+SELECTION_END_INDEX: int = MIN_TRAIN
+
 def evaluate_inner_split_config(
     data: pd.DataFrame,
     feature_cols: list[str],
     target_cols: dict[int, str],
     config: dict[str, Any],
     horizons: list[int] | None = None,
-    selection_end_date: str = "2023-01-01",
+    selection_end_index: int = SELECTION_END_INDEX,
     cal_ratio: float = 0.15,
     val_ratio: float = 0.20,
     seed: int = SEED,
 ) -> dict[str, Any]:
@@
-    # Isolate the pre-evaluation selection window
-    if "date" in data.columns and (data["date"] < selection_end_date).any():
-        train_pool = data[data["date"] < selection_end_date].copy()
-    else:
-        # Fallback for synthetic/unit-test fixtures smaller than 2023
-        pool_size = min(len(data), MIN_TRAIN)
-        train_pool = data.iloc[:pool_size].copy()
+    # Isolate the pre-evaluation selection window: strictly the rows that are
+    # never scored as forecast origins by make_rolling_folds().
+    pool_size = min(len(data), selection_end_index)
+    if pool_size < 60:
+        raise ValueError(
+            f"Selection pool of {pool_size} rows is too small for a leakage-safe "
+            f"embargoed inner split; refusing to fall back to a leaky partition."
+        )
+    train_pool = data.iloc[:pool_size].copy()
```

With `MIN_TRAIN = 500` the pool is 500 rows (2010-03-19 .. 2012-02-16), which yields
`inner_train ≈ 330 / inner_val ≈ 84` rows at h=1 — thin, but *honest*. If that is judged too small
to select 7 hyperparameters, the alternative (and the option I recommend) is to **raise `MIN_TRAIN`
so that the evaluation window starts after the selection window**, and re-run the whole battery for
all five models on the shortened common sample so the comparison stays paired:

```diff
--- a/src/xgboost_baseline.py
+++ b/src/xgboost_baseline.py
-MIN_TRAIN = 500
+# First scored forecast origin. Set so that the pre-origin block is large
+# enough to host the Issue #119 hyperparameter selection without any selected
+# row ever being scored out-of-sample.
+MIN_TRAIN = 3336   # 2010-03-19 .. 2022-12-30; first origin becomes 2023-01-03
```

Whichever is chosen, §4.3 item 1 must be rewritten to state the *index* boundary and the number of
scored origins that precede it (currently: zero must). Do not ship the current wording.

---

### 🔴 C-2 — Un-versioned data refresh bundled with the tuning change: the reported effect is not identified

**Files:** `.gitignore:18-21`; `outputs/r3_xgboost_vs_naive.csv`; `outputs/clark_west_test_results.csv`;
`docs/CLARK_WEST_METHODOLOGY.md:102-118, 160`; `tests/test_model_comparison.py:415-417`

`data/processed/*` is gitignored; the local inputs were regenerated at `2026-09-05 07:45`, one
minute before the commit at `07:46:46`. The evaluation sample changed as a result:

| Quantity | `main` | branch | Sensitive to XGB hyperparameters? |
|---|---|---|---|
| First forecast origin | 2012-03-19 | 2012-02-17 | **No** |
| Origins per fold | 741 / 741 / 741 / 741 / 743 | 745 / 745 / 745 / 745 / 748 | **No** |
| `n_forecasts` (`r3_xgboost_vs_naive`) | 3707 | 3728 | **No** |
| Clark-West common sample `N` | 741 | 745 | **No** |
| `mspe_naive`, h=1 | 0.000845 | 0.000843 | **No** |
| `rmse_naive`, h=20 | 0.12639 | 0.12613 | **No** |

**A change in the naïve random-walk MSPE is proof that the data, not just the model, changed.**

§4.3 item 5 (line 160) nonetheless attributes the entire h=1 movement to tuning:

> *"Tuning reduced $h=1$ MSPE from $0.000921$ to $0.000880$ (a 4.5% error reduction …)"*

That comparison spans two different samples and is not attributable to the hyperparameters. The
same applies to every "before/after" cell in Table 1 — 15 rows moved, including the four models
this PR did not touch.

**Required fix (three parts):**

1. Regenerate the *baseline* outputs on the refreshed sample first, on a separate commit, so the
   tuning diff isolates the hyperparameter effect:
   ```bash
   git checkout main -- src/xgboost_baseline.py
   python -m src.xgboost_baseline && python -m src.model_comparison   # commit: "chore: refresh outputs on 2026-09-05 data vintage"
   git checkout feat/119-xgboost-hyperparameter-tuning -- src/xgboost_baseline.py
   python -m src.xgboost_baseline && python -m src.model_comparison   # commit: "feat(119): tuned hyperparameters"
   ```
   Then restate §4.3 item 5 against the same-vintage baseline.
2. Pin the data vintage. Since `data/processed/` cannot be committed, record a manifest so the
   outputs are attributable:
   ```python
   # src/project_paths.py  (new)
   def write_data_manifest(paths: list[Path], out: Path) -> None:
       """Record sha256 + row count + date range of every input feeding an output run."""
   ```
   and emit `outputs/data_manifest.json` from `xgboost_baseline.main()` /
   `model_comparison.main()`.
3. Restore the pinned invariant that was weakened to hide this:
   ```diff
   --- a/tests/test_model_comparison.py
   +++ b/tests/test_model_comparison.py
   -    # Reconciled common sample: exactly 745 (or 741) origin dates across all arms
   -    assert (primary_df["n_forecasts"].isin([741, 745])).all()
   +    # Reconciled common sample: one pinned N across all arms. A change here means
   +    # the data vintage moved -- regenerate outputs and update this constant in the
   +    # same commit, never widen the accepted set.
   +    assert (primary_df["n_forecasts"] == EXPECTED_COMMON_SAMPLE_N).all()
   ```
   As written, `isin([741, 745])` is also **internally inconsistent** with
   `tests/test_model_comparison.py:549`, which asserts exact q-values
   (`0.9177 / 0.6755 / 0.2788`, `atol=1e-3`) that only hold for `N = 745`. A 741-row sample passes
   line 417 and fails line 549 — the relaxation buys nothing and costs the invariant.

---

### 🔴 C-3 — §4.3's stated partition boundaries do not match any partition the code produces

**File:** `docs/CLARK_WEST_METHODOLOGY.md:149-153`

Line 150 documents:
`inner_train (2012–2020) → inner_val (2020–2022) → conformal calibration (pre-test 10%) → test origins (2023–2026)`

Measured from `evaluate_inner_split_config` on the shipped data — **every segment is wrong**:

| Segment | Documented | Actual (h=1) |
|---|---|---|
| `inner_train` | 2012–2020 | **2010-03-19 .. 2018-11-26** |
| `inner_val` | 2020–2022 | **2018-11-28 .. 2021-01-27** |
| calibration | "pre-test **10 %**" | **2021-01-29 .. 2022-12-29** (`cal_ratio = 0.15`, i.e. **15 %**) |
| test origins | 2023–2026 | **2012-02-17 .. 2026-06-02** |

Line 152's "$h=20$ error variance is $\approx 4.3\times$ larger than $h=1$" is also mis-stated:
from the shipped search CSV the *RMSE* ratio is `0.098466 / 0.023388 = 4.21×`; the **variance**
ratio is `17.7×`. Fix the noun or the number.

```diff
-   $$\text{inner\_train (2012–2020)} \xrightarrow{\text{embargo } h} \text{inner\_val (2020–2022)} \xrightarrow{\text{embargo } h} \text{conformal calibration (pre-test 10\%)} \xrightarrow{\text{embargo } h} \text{test origins (2023–2026)}$$
+   $$\text{inner\_train}_{[0,\,k)} \xrightarrow{\text{embargo } h} \text{inner\_val}_{[k+h,\,f)} \xrightarrow{\text{embargo } h} \text{calibration}_{[f+h,\,n)} \;\Big|\; \text{first scored origin at index } \texttt{MIN\_TRAIN}$$
+   with $n = $ `SELECTION_END_INDEX`, $f = n - \lceil 0.15n \rceil - h$, $k = f - \lceil 0.20f \rceil - h$.
+   Concrete dates for the shipped vintage are emitted to `outputs/r3_xgboost_search_partitions.csv`.
-3. **Multi-Horizon Objective Function**: Because $h=20$ error variance is naturally $\approx 4.3\times$ larger than $h=1$,
+3. **Multi-Horizon Objective Function**: Because the $h=20$ naïve RMSE scale is $\approx 4.2\times$ that of $h=1$,
```

Emit the partition boundaries as a real artifact rather than prose, so the doc cannot drift again:

```python
# src/xgboost_grid_search.py — inside evaluate_inner_split_config, after the split is computed
partition_log.append({
    "horizon": h,
    "inner_train_start": inner_train["date"].iloc[0],  "inner_train_end": inner_train["date"].iloc[-1],
    "inner_val_start":   inner_val["date"].iloc[0],    "inner_val_end":   inner_val["date"].iloc[-1],
    "cal_start": train_data["date"].iloc[fit_end + h], "cal_end": train_data["date"].iloc[-1],
    "first_scored_origin": data["date"].iloc[MIN_TRAIN],
})
```

---

## 3. High-Severity Findings

### 🟠 H-1 — The leakage guarantee is asserted by a test that re-implements it, not by one that calls it

**File:** `tests/test_xgboost_grid_search.py:91-132`

`test_inner_validation_disjoint_from_calibration_and_test` never calls
`evaluate_inner_split_config`. Lines 101-115 re-derive `embargo_end`, `cal_size`, `fit_end`,
`val_size`, `inner_train_end` and the `+ h` embargo **inside the test body**, then assert that this
private copy is self-consistent. Deleting `+ h` from
`src/xgboost_grid_search.py:229` — the single line the whole leakage claim rests on — leaves this
test green. The docstring at lines 7-8 ("inner validation split is strictly disjoint …") is
therefore unverified.

```diff
+import src.xgboost_grid_search as gs
+
 class TestInnerSplitTemporalEmbargo:
-    def test_inner_validation_disjoint_from_calibration_and_test(self, synthetic_gold_df):
-        data, feature_cols, target_cols = engineer_tabular_features(...)
-        h = 5
-        ...  # 25 lines re-deriving the production arithmetic
+    def test_production_split_preserves_embargo(self, synthetic_gold_df, monkeypatch):
+        """Capture the partitions the SHIPPED function actually builds and assert the
+        embargo on those -- not on a copy of the arithmetic maintained in the test."""
+        data, feature_cols, target_cols = engineer_tabular_features(
+            synthetic_gold_df, lags=[1, 2], horizons=[1, 5],
+        )
+
+        class _Spy:
+            def fit(self, X, y): return self
+            def predict(self, X): return np.zeros(len(X))
+
+        monkeypatch.setattr(gs, "create_model", lambda **kw: _Spy())
+        monkeypatch.setattr(gs, "_PARTITION_LOG", [], raising=False)
+        gs.evaluate_inner_split_config(
+            data, feature_cols, target_cols,
+            {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 5},
+            horizons=[5], selection_end_index=len(data),
+        )
+        p = gs._PARTITION_LOG[0]           # emitted by the C-3 partition_log change
+        h = 5
+        assert p["inner_val_start_idx"] - p["inner_train_end_idx"] > h, "embargo collapsed"
+        assert p["cal_start_idx"] > p["inner_val_end_idx"], "val/calibration overlap"
+        assert p["first_scored_origin_idx"] >= p["cal_end_idx"], "selection window reaches into scored origins"
```

The same critique applies to `test_search_is_deterministic` (lines 138-146): calling
`run_grid_search` twice in one process with `n_jobs=1` cannot detect the failure modes that matter
(thread-order accumulation, BLAS/OpenMP variation, library-version drift). It is not wrong, just
non-load-bearing; pin `xgboost.__version__` in `requirements.txt` and assert the shipped
`outputs/r3_xgboost_hyperparameter_search.csv` reproduces instead. *(I re-ran rank 1 and the
canonical config on this machine — `1.010947` and `1.022498`, matching the committed CSV to all six
decimals under xgboost 3.4.1. Determinism itself is fine.)*

---

### 🟠 H-2 — Selection–evaluation protocol mismatch, and the winning depth/lr is inside seed noise

**Files:** `src/xgboost_grid_search.py:162-285`; `outputs/r3_xgboost_hyperparameter_search.csv`

Selection uses **one fixed inner split**; evaluation uses a **5-fold expanding window**. A
configuration is therefore chosen for a single 2018–2021 regime and frozen across 2012–2026,
spanning ZIRP, the COVID shock, the 2022 tightening and the 2024–25 easing cycle. The regime table
already shows the consequence: the tuned model is worse than `main` in Regime 2 (h=20 RMSE
improvement −6.34 % → **−8.98 %**) and Regime 3 (−8.16 % → **−9.77 %**), while better in Regime 1.

I ran the top-4 configurations plus the canonical baseline across five seeds:

| Config | seed 0 | 1 | 7 | 42 | 2024 | mean | sd |
|---|---|---|---|---|---|---|---|
| **R1** d2 / .01 / 600 / tier0 | 1.01101 | 1.01119 | 1.01122 | 1.01095 | 1.01042 | **1.01096** | 0.00032 |
| R2 d3 / .01 / 600 / tier0 | 1.01210 | 1.01166 | 1.01164 | 1.01211 | 1.01043 | 1.01159 | 0.00068 |
| R3 d4 / .01 / 600 / tier0 | 1.01231 | 1.01245 | 1.01482 | 1.01271 | 1.01154 | 1.01277 | 0.00123 |
| R4 d2 / .03 / 300 / tier0 | 1.01537 | 1.01461 | **1.01108** | 1.01389 | 1.01127 | 1.01324 | 0.00196 |
| R7 d2 / .01 / 600 / tier1 | 1.02244 | 1.02118 | 1.02209 | 1.02162 | 1.01833 | 1.02113 | 0.00164 |
| R8 canonical (d3/.03/150/tier1) | 1.02273 | 1.02361 | 1.02291 | 1.02250 | 1.02242 | 1.02283 | 0.00048 |

Read honestly: the **regularization tier is a robust finding** (tier0 beats tier1/canonical by
~0.010 at every seed, ≈ 15σ). The **depth/learning-rate choice is not** — R1 vs R2 is 0.00063,
below R2's own seed sd of 0.00068, and the argmin flips to R4 at seed 7. Freezing
`max_depth = 2, lr = 0.01, n_estimators = 600` as a single point estimate over-claims; the paper's
narrative rationale ("shallow trees suppress variance on noisy differenced financial series",
line 155) is post-hoc for a difference the search cannot resolve.

```diff
--- a/src/xgboost_grid_search.py
+++ b/src/xgboost_grid_search.py
+SELECTION_SEEDS: tuple[int, ...] = (42, 0, 1, 7, 2024)
+
+def evaluate_config_across_seeds(data, feature_cols, target_cols, config, **kw) -> dict[str, Any]:
+    """Average the selection objective over SELECTION_SEEDS and report its dispersion.
+
+    A single-seed argmin over 28 candidates separated by ~6e-4 is not identified when the
+    per-seed sd of the objective is ~3e-4 to ~2e-3: the winner would change with the seed.
+    Ranking on the seed-mean, and publishing the sd, makes the selection auditable.
+    """
+    recs = [evaluate_inner_split_config(data, feature_cols, target_cols, config, seed=s, **kw)
+            for s in SELECTION_SEEDS]
+    out = dict(recs[0])
+    vals = [r["mean_relative_rmse"] for r in recs]
+    out["mean_relative_rmse"] = float(np.mean(vals))
+    out["mean_relative_rmse_sd"] = float(np.std(vals, ddof=1))
+    out["n_selection_seeds"] = len(SELECTION_SEEDS)
+    return out
```

and rank on `mean_relative_rmse`, breaking ties within `1 sd` toward the more parsimonious model.

**Also required in §4.3 item 4.** All 28 candidates score `mean_relative_rmse > 1.0` — **no
configuration beats the naïve random walk on the selection window**, and `val_improvement_pct` is
negative for all 28 (rank 1 = **−1.095 %**). The doc's "+1.13 % improvement" is measured against the
*canonical XGBoost defaults*, not against the benchmark. Say so explicitly:

```diff
-4. **Winning Configuration & Regularization Insights**: The top-ranked configuration achieved an overall relative RMSE of $1.0109$ (+1.13% improvement over baseline defaults at Rank 8):
+4. **Winning Configuration & Regularization Insights**: No candidate in the 28-configuration space
+   attained $\mathcal{L}_{\text{search}} < 1$ — i.e. **none beat the naïve random walk on the
+   selection window**, consistent with §4.2. The top-ranked configuration reached
+   $\mathcal{L}_{\text{search}} = 1.0109$ ($-1.10\%$ vs. Naïve), a $1.13\%$ reduction relative to
+   the untuned XGBoost defaults (Rank 8, $1.0225$). The regularization tier is robust across seeds
+   ($\approx 15\sigma$); the depth / learning-rate choice within that tier is **not** identified by
+   this search ($\Delta = 6\times10^{-4}$ vs. per-seed $\sigma \approx 7\times10^{-4}$):
```

---

## 4. Medium-Severity Findings

### 🟡 M-1 — Silent fallback to a leaky, embargo-free split

**File:** `src/xgboost_grid_search.py:186-191, 211-215, 222-225`

Three fallback branches drop the embargo entirely and log nothing:

```python
186  if "date" in data.columns and (data["date"] < selection_end_date).any():   # True for a 30-row pool
211  if fit_end < 20:
212      half = max(10, n_tr // 2)
213      inner_train = train_data.iloc[:half]
214      inner_val = train_data.iloc[half:]        # <-- no `+ h`: inner_train targets overlap inner_val
222  if inner_train_end < 10:
223      half = max(5, n_fit // 2)                  # <-- same, and inner_val now includes the calibration block
```

A pool of *any* size ≥ 1 satisfies line 186, so a mis-specified `selection_end_date`, a truncated
vintage, or a shorter `MIN_TRAIN` silently converts a "leakage-safe" search into a leaky one with
no exception, no warning, and identical CSV schema. Separately, when `n_tr < 10`,
`half = max(10, n_tr // 2)` exceeds `len(train_data)`, `inner_val` is empty, and
`np.sqrt(np.mean([]))` yields `nan` — `sort_values` then places the config silently last and
`rank` is still assigned, so a config that never evaluated is reported as merely poor.

```diff
-        if fit_end < 20:
-            # Minimal fixture fallback
-            half = max(10, n_tr // 2)
-            inner_train = train_data.iloc[:half]
-            inner_val = train_data.iloc[half:]
-        else:
-            fit_data = train_data.iloc[:fit_end]
-            n_fit = len(fit_data)
-            val_size = max(int(n_fit * val_ratio), 20)
-            inner_train_end = n_fit - val_size - h
-
-            if inner_train_end < 10:
-                half = max(5, n_fit // 2)
-                inner_train = fit_data.iloc[:half]
-                inner_val = fit_data.iloc[half:]
-            else:
-                # Causal h-step embargo between inner_train and inner_val
-                inner_train = fit_data.iloc[:inner_train_end]
-                inner_val = fit_data.iloc[inner_train_end + h:]
+        # No embargo-free fallback: a split that cannot honour the h-step gap is not a
+        # leakage-safe split, and silently substituting one would invalidate every
+        # downstream selection claim. Fail loudly instead.
+        if fit_end < 20:
+            raise ValueError(
+                f"h={h}: fit block of {fit_end} rows cannot host an embargoed inner split "
+                f"(pool={n_pool}, cal_size={cal_size}). Increase the selection window."
+            )
+        fit_data = train_data.iloc[:fit_end]
+        n_fit = len(fit_data)
+        val_size = max(int(n_fit * val_ratio), 20)
+        inner_train_end = n_fit - val_size - h
+        if inner_train_end < 10:
+            raise ValueError(
+                f"h={h}: only {inner_train_end} inner-train rows remain after the "
+                f"{val_size}-row validation block and {h}-step embargo."
+            )
+        # Causal h-step embargo between inner_train and inner_val
+        inner_train = fit_data.iloc[:inner_train_end]
+        inner_val = fit_data.iloc[inner_train_end + h:]
+        if len(inner_val) == 0:
+            raise ValueError(f"h={h}: empty inner validation block after embargo.")
```

Unit fixtures should be sized to satisfy the real protocol, not have the protocol relaxed for them.

---

### 🟡 M-2 — Search-time defaults are stale duplicates; the exported audit row can disagree with the fitted model

**Files:** `src/xgboost_grid_search.py:241-249, 267-275`; `src/xgboost_baseline.py:56-64`

`evaluate_inner_split_config` hardcodes its own fallbacks:

```python
241  max_depth=config.get("max_depth", 3),                  # xgboost_baseline.MAX_DEPTH is now 2
242  learning_rate=config.get("learning_rate", 0.03),       # LEARNING_RATE is now 0.01
243  n_estimators=config.get("n_estimators", 150),          # N_ESTIMATORS is now 600
246  colsample_bytree=config.get("colsample_bytree", 0.8),  # COLSAMPLE_BYTREE is now 0.7
247  min_child_weight=config.get("min_child_weight", 1.0),  # MIN_CHILD_WEIGHT is now 5.0
248  reg_lambda=config.get("reg_lambda", 1.0),              # REG_LAMBDA is now 5.0
```

These are the **pre-tuning** values, so a partially-specified config is silently evaluated under a
different specification than the same dict would produce through `run_rolling_cv`. Worse, the
exported record at lines 267-275 logs `config.get("colsample_bytree")` → `None` while the model
was fitted with `0.8`. `outputs/r3_xgboost_hyperparameter_search.csv` is the hyperparameter
provenance artifact for the thesis; it must record what was fitted. (The shipped CSV is unaffected
because the XGBoost branch of `generate_search_space` populates all seven keys — this is a latent
trap, reachable today only via the `sklearn` fallback and `tests/…:139-141`.)

```diff
-        model = create_model(
-            loss="squared_error",
-            max_depth=config.get("max_depth", 3),
-            learning_rate=config.get("learning_rate", 0.03),
-            n_estimators=config.get("n_estimators", 150),
-            subsample=config.get("subsample", 0.8),
-            colsample_bytree=config.get("colsample_bytree", 0.8),
-            min_child_weight=config.get("min_child_weight", 1.0),
-            reg_lambda=config.get("reg_lambda", 1.0),
-            reg_alpha=config.get("reg_alpha", 0.0),
-            random_state=seed,
-        )
+        # Resolve every parameter ONCE against the single source of truth in
+        # xgboost_baseline, then log exactly what was fitted -- a search CSV that
+        # records None for a parameter the model actually used is not an audit trail.
+        resolved = {
+            "max_depth": config.get("max_depth", MAX_DEPTH),
+            "learning_rate": config.get("learning_rate", LEARNING_RATE),
+            "n_estimators": config.get("n_estimators", N_ESTIMATORS),
+            "subsample": config.get("subsample", SUBSAMPLE),
+            "colsample_bytree": config.get("colsample_bytree", COLSAMPLE_BYTREE),
+            "min_child_weight": config.get("min_child_weight", MIN_CHILD_WEIGHT),
+            "reg_lambda": config.get("reg_lambda", REG_LAMBDA),
+            "reg_alpha": config.get("reg_alpha", REG_ALPHA),
+        }
+        model = create_model(loss="squared_error", random_state=seed, **resolved)
@@
-    record: dict[str, Any] = {
-        "max_depth": config.get("max_depth"),
-        "learning_rate": config.get("learning_rate"),
-        "n_estimators": config.get("n_estimators"),
-        "subsample": config.get("subsample"),
-        "colsample_bytree": config.get("colsample_bytree"),
-        "min_child_weight": config.get("min_child_weight"),
-        "reg_lambda": config.get("reg_lambda"),
-    }
+    record: dict[str, Any] = dict(resolved)   # exactly the fitted specification
+    record["seed"] = seed
```

(Import `MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, SUBSAMPLE, COLSAMPLE_BYTREE, MIN_CHILD_WEIGHT,
REG_LAMBDA, REG_ALPHA` at `src/xgboost_grid_search.py:46-55`. Note `resolved` must be hoisted above
the horizon loop so it is in scope at the record site.)

---

### 🟡 M-3 — Conformal interval coverage degraded at every horizon and both nominal levels

**Files:** `outputs/r3_xgboost_prediction_intervals.csv`; `src/dashboard/app.py:127-131`

| h | 90 % cov. (main → branch) | width | 95 % cov. (main → branch) | width |
|---|---|---|---|---|
| 1 | 87.81 → **86.78** | 0.0901 → 0.0863 | 91.99 → **91.66** | 0.1103 → 0.1069 |
| 5 | 87.19 → **84.92** | 0.2030 → 0.1942 | 92.91 → **91.20** | 0.2480 → 0.2362 |
| 20 | 85.41 → **84.55** | 0.3861 → 0.3727 | 93.36 → **92.09** | 0.4935 → 0.4707 |

Intervals got **narrower and less covering** — and this is *after* `ce24a26` raised the quantile
level (dividing by `n_cal + 1` instead of `n_cal`), which should have widened them. The heavier
regularization shrinks calibration residuals relative to test-period residuals, so the miscoverage
is a genuine tuning side-effect, not the denominator fix. h=5 at nominal 90 % is now 5.1 pp short.

Root cause is structural: split conformal requires exchangeability, which overlapping $h$-step
targets on a serially dependent series violate; the guarantee does not hold for $h \in \{5, 20\}$.
The dashboard nonetheless advertises "90 % or 95 % **calibrated** empirical prediction intervals"
(`src/dashboard/app.py:131`). At minimum, retitle and disclose:

```diff
-    help="Displays 90% or 95% calibrated empirical prediction intervals for XGBoost.",
+    help=(
+        "Split-conformal empirical prediction intervals for XGBoost. Finite-sample validity "
+        "requires exchangeable residuals, which overlapping h-step targets violate; realized "
+        "coverage is 86.8/84.9/84.6% at h=1/5/20 against a 90% nominal target "
+        "(outputs/r3_xgboost_prediction_intervals.csv). Treat as indicative, not guaranteed."
+    ),
```

and add a coverage regression guard so a future tuning round cannot quietly worsen it further:

```python
# tests/test_xgboost_baseline.py
def test_shipped_intervals_within_disclosed_miscoverage_budget():
    """Nominal-90% coverage must not fall below 83% on the shipped artifact.
    Conformal validity is already void under h-step overlap; this bounds the damage."""
    iv = pd.read_csv(OUTPUTS_DIR / "r3_xgboost_prediction_intervals.csv")
    assert (iv["empirical_coverage_90"] >= 83.0).all()
    assert (iv["empirical_coverage_95"] >= 90.0).all()
    assert (iv["empirical_coverage_95"] > iv["empirical_coverage_90"]).all()
```

---

## 5. Low-Severity / Polish

| # | File:line | Issue | Fix |
|---|---|---|---|
| L-1 | `src/dashboard/charts.py:11`, `app.py:121`, `app.py:523` vs `data_loader.py:305`, `model_comparison.py:792,961` | The sidebar selector, the ribbon legend and the spec table say **"XGBoost"**; the performance table and Clark-West table say **"XGBoost (Tuned)"**. Same screen, two names. No data loss (`charts.py` keys on columns), but it reads as two models. | Pick one label and route it through a single `XGB_DISPLAY_LABEL = "XGBoost (Tuned)"` constant imported by all five sites. |
| L-2 | `src/xgboost_baseline.py:428` | `MIN_FIT_ROWS = 10` is a module-level policy constant declared **inside** the per-horizon `for` loop, re-bound on every iteration. The threshold also silently dropped 30 → 10, which permits "conformal" intervals from a 10-row fit. | Hoist next to `MIN_TRAIN` at line 51 and document the 10-row floor as test-fixture-only. |
| L-3 | `src/xgboost_grid_search.py:40-45` vs `src/xgboost_baseline.py:26-33` | Grid search imports `src.xgboost_baseline`; the baseline imports bare `gold_feature_pipeline`. Both `src/` and the project root are on `sys.path`, so `xgboost_baseline` and `src.xgboost_baseline` can load as **two distinct module objects** with two independent `_CACHED_GOLD_DF` caches and two `HAS_XGBOOST` flags (`tests/test_xgboost_grid_search.py:21-36` already imports both ways). | Make `src/` a package and use `from .xgboost_baseline import …` consistently. |
| L-4 | `src/xgboost_baseline.py:561-612` | `compute_tree_shap_interpretability` fits on the **full sample including 2023-2026**; the regenerated `r3_xgboost_shap_*.csv` therefore describe an in-sample fit. Pre-existing, but the attributions changed materially in this PR. | Add a one-line caveat where SHAP is presented, or fit on the pre-`MIN_TRAIN` block. |
| L-5 | `src/xgboost_grid_search.py:394` | `rel_gain` is computed but `0.0` is returned when no canonical row matched, which is indistinguishable from "tied with canonical". | Return `None`. |

---

## 6. What Is Good

- `_conformal_quantile_level` (`src/xgboost_baseline.py:243-255`) is the correct Vovk et al. (2005)
  construction, and `TestConformalCalibrationFix` (`tests/test_xgboost_baseline.py:317-395`) is
  exemplary: it calls the **production helper**, encodes the regression it guards against, and
  documents *why* the 20-row `cal_size` floor is exactly the right margin. This is the standard the
  rest of the test suite should be held to (cf. H-1).
- The search is genuinely deterministic — `n_jobs=1` (`src/xgboost_baseline.py:283,296`), seed
  threaded end-to-end. I reproduced rank 1 (`1.010947`) and canonical (`1.022498`) to six decimals.
- Naïve-standardized multi-horizon objective (`src/xgboost_grid_search.py:236-265`) is the right
  choice and correctly implemented — the $\Delta$-space naïve forecast is exactly zero, and
  level-space and difference-space RMSE coincide here.
- The `sklearn` fallback branch (`generate_search_space:145-157`) correctly prunes unsupported
  parameters rather than letting `GradientBoostingRegressor` raise.
- Explicitly injecting the canonical configuration into the grid so the tuning gain is auditable
  (`generate_search_space:132-144`) is good practice.

---

## 7. Merge Recommendation

**Request changes.** Blocking: **C-1**, **C-2**, **C-3**. Should land in the same PR: **H-1**, **H-2**.

Minimal path to approval:
1. Re-scope the selection window to end at the first scored forecast origin (C-1), re-run the search,
   re-freeze, regenerate all `outputs/*`, and restate Table 1.
2. Split the data-vintage refresh into its own commit and re-derive the tuning delta on a fixed
   sample; restore the pinned `n_forecasts` invariant (C-2).
3. Rewrite §4.3 against emitted partition boundaries rather than prose (C-3).
4. Make the embargo test call the production function (H-1) and rank on a seed-averaged objective
   with dispersion published (H-2).

Until C-1 is resolved, the `XGBoost (Tuned)` row of Table 1 should not be cited as an out-of-sample
result in the capstone deliverable.
