# M2_CHECKLIST.md — Milestone 2: Preliminary Analysis (Fri Aug 07)

## Parallel-track model

M2 is deliberately **not** run as one owner per functional area (that was the proposal's model).
For the three technical rounds — data collection, EDA, and baseline models — two people build
independent solutions in parallel, using genuinely different approaches, and the other two
review. The team then picks the better one (or keeps both) before moving on. The point isn't
redundancy for its own sake — it's that everyone leaves M2 having actually understood the data
and the pipeline first-hand, not just reviewed someone else's PR, and every technical decision has
been pressure-tested by someone who built an alternative rather than rubber-stamped.

Only after all three rounds converge do the two remaining tasks — feature engineering and the
outcome/early-results plan — get a single owner each; by that point "understanding the data" is
no longer the open question, so there's nothing left to parallelize.

| Round | Task | Path A | Path B | A's reviewer | B's reviewer |
|-------|------|--------|--------|---------------|---------------|
| 1 | Data collection & prep | **Giti** — sequential/synchronous pull, one source at a time, manual back-off | **Lerneir** — per-source client classes, concurrent fetch, config-driven series list | Mitchel | Nelson |
| 2 | Data exploration (EDA) | **Nelson** — Python-native statistical EDA (`01_eda`) | **Mitchel** — Power BI visual/interactive exploration | Lerneir | Giti |
| 3 | Baseline models | **Giti** — VAR, lag order via AIC | **Mitchel** — VAR, lag order via BIC | Nelson | Lerneir |
| — | Feature engineering (single) | **Lerneir** (owner) | — | Giti (reviewer) | — |
| — | Outcome / early-results plan (single) | **Nelson** (owner) | — | Mitchel (reviewer) | — |

**Load check:** every member is a doer/owner exactly twice and a reviewer exactly twice across
the five rows above (Giti: 1, 3 / 2, feature-eng-review; Lerneir: 1, feature-eng / 2, 3-review;
Nelson: 2, outcome-plan / 1-review, 3-review; Mitchel: 2, 3 / 1-review, outcome-plan-review). This
is the deliberate rebalance from the proposal phase, where Giti and Mitchel's sections (title
page, references, outcomes, ethics) carried less technical weight than Lerneir's and Nelson's —
here all four get equal hands-on build time on the actual analysis.

No one reviews their own round's other path (reviewers are always drawn from people not building
either path that round), and reviewer pairs rotate round to round rather than the same two people
always checking each other.

### Why these specific technical forks

- **Round 1** — both paths must still follow section 5.1's Bronze-layer contract (HTTP pulls
  against BoC Valet / FRED / StatCan, exponential back-off, timestamped local cache); they differ
  in orchestration (sequential script vs. concurrent client classes), not in which APIs get
  called. Compare on: does it survive a rate-limit response, is it readable by someone who didn't
  write it, how long does a full pull take.
- **Round 2** — Approach A and B aren't competing on correctness, they're complementary by design:
  a reproducible notebook record vs. a stakeholder-facing dashboard (Power BI is already in the
  stack per `README.md`'s Tools & Stack). It's fine — expected, even — for the team to keep both
  rather than eliminate one.
- **Round 3** — AIC vs. BIC lag-order selection is explicitly named as the tradeoff in
  `proposal/sections/05_analytical_approach.md` §5.3 ("selected using AIC and BIC to balance
  model fit against overfitting risk"), so this fork tests an alternative already inside the
  approved methodology — neither path touches ARIMA, so this round doesn't depend on
  `proposal/build/FLAGS.txt` Issue 1 being resolved first.

## Convergence decisions (fill in as each round concludes — Week 4–5 team syncs)

**Round 1 — Data collection:** chosen path — **Giti's sequential/synchronous pull (Path A)**.
Rationale — decided at Aug 09 team sync. Lerneir's concurrent-client path (Path B) shipped and
works, but its follow-up fix (`#37`, FRED key leaking into retry logs + dead StatCan URL) is
unresolved; rather than block on it, the team is proceeding on Path A and moving `#37` to the
backlog since nothing is currently building on top of Path B.

**Round 2 — EDA:** chosen approach — **keeping both**, Python (`01_eda`, Nelson, `#26`) as
reproducible record / Power BI (Mitchel, `#27`) as dashboard. Rationale — decided at Aug 09 team
sync: `#27` reviewed and approved; the two approaches are complementary by design per this
document's own §"Why these specific technical forks", so no elimination was needed.

**Round 3 — Baseline models:** chosen lag-order criterion — **keeping both as a documented
sensitivity check**, not picking a winner. Rationale — reviewed both Aug 16: AIC (`#28`, lag=10,
5-var feature set) and BIC (`#29`, lag=0, 6-var feature set incl. `usdcad`) agree on the finding
that actually matters — **VAR does not beat the naive random-walk benchmark at any of the 1-/5-/
20-day horizons, under either lag-order criterion.** AIC's own Diebold-Mariano test finds naive
*significantly* better at h=1 on MAE (p=0.0005); no horizon or loss function ever favors VAR.
BIC's RMSE/MAE are numerically close to naive throughout (BIC selected lag 0, effectively a
constant-drift model), though no DM significance test was run on that path to confirm the gap
isn't noise.

Not treating this as a clean AIC-vs-BIC horse race, because the two runs aren't a controlled
comparison as executed: AIC's script (`#28`/PR #42) used a 5-variable feature set with no
`usdcad`; BIC's notebook (`#29`/PR #45) labels its 6-variable set (incl. `usdcad`) as "the common
Round 3 feature set" but it isn't the one AIC actually used, and the two runs land on different
modeling-sample sizes as a result. So "BIC selected a shorter lag" and "BIC's numbers are closer
to naive" are confounded with "BIC's run also had an extra input variable" — can't currently
attribute the difference to the lag-order criterion alone.

Given both paths already converge on the substantive answer (VAR ≤ naive baseline here), re-running
either to remove the confound isn't worth blocking Round 3 on — flagging it instead as a caveat for
whoever cites this comparison later, and as a strong reason the naive floor and the upcoming LSTM
(`#49`) — not VAR/VECM — carry the real forecasting-improvement question forward. Two open items
before `#28`/`#29` can close: PR #45 (BIC) is still unmerged and unreviewed by Lerneir per the
reviewer pairing, and it has no Diebold-Mariano significance test — add one before treating its
RMSE/MAE gaps as anything other than descriptive. Carries into Week 6 per `TIMELINE.md` §6.

**Aug 19 correction — `usdcad` restored as a predictor across `#28`/`#29`/`#49`.** Supersedes the
"5-var feature set" framing above. While building `#49`'s LSTM baseline, review against
`proposal/sections/03_analytical_objective.md` §3.1 found that the research question explicitly
names USD/CAD as a required "macroeconomic transmission variable" — it is a committed predictor,
not an optional one, and §5.3 lists it as part of the VAR/VECM system. The 5-variable convergence
described above was an undocumented omission in `#28`'s original script, not a reviewed
methodological choice: `#29` originally had 6 variables (including `usdcad`), and the Aug 16 "fix"
matched it *down* to `#28` to control the AIC-vs-BIC comparison, rather than adding `usdcad` to
`#28` to match the proposal. EDA supports no exclusion either way — `usdcad` is I(1)/stationary
after one difference, same as every other series (`notebooks/01_eda/eda.ipynb`, ADF table).

Decision: restore `usdcad` as the 6th predictor everywhere. Applied Aug 19:
- `#28` (`src/EDA_VAR_AIC _lag _order.py`): `FEATURE_SET` now includes `usdcad`; AIC-selected lag
  order unchanged at 10; VAR still does not beat naive at any horizon.
- `#29` (PR #45, merged Aug 19 on Lerneir's existing approval, then updated on
  `artifact/m2-lstm-shap`): `FEATURES` now includes `usdcad`; BIC still selects lag 0.
  RMSE/MAE/DM results are numerically unchanged from the 5-variable run, because a lag-0 VAR
  reduces to a constant-drift model on the target's own history and never reads any regressor —
  the fix corrects proposal alignment and #28/#29 comparability, not this particular result.
- `#49` (`src/lstm_baseline.py`): `FEATURES` now includes `d_usdcad` from its first commit on
  `artifact/m2-lstm-shap`, so the LSTM was proposal-correct from the start; SHAP puts `d_usdcad`
  as the 2nd-most-important feature by mean |SHAP|, behind `d_us_treasury_10y`.

All three models now share the same proposal-correct 6-variable feature set, which is what makes
`#50`'s eventual pairwise Diebold-Mariano comparison across Naive/ARIMA/VAR/VECM/LSTM valid.

**Aug 20 correction — off-by-one in `#28`'s (`src/EDA_VAR_AIC _lag _order.py`) forecast origin
anchor, found while building `#50`'s cross-model alignment.** `evaluate()` anchored `last_level`
(and therefore both the naive and VAR forecast base) on `level_idx_for_diff_row[origin]`, which
resolves to the differenced row just *past* the training cutoff — `train = diffed.iloc[:origin]`
excludes that row, so the anchor was reading one business day of otherwise-unseen level data into
every origin, and #28's origin grid never lined up date-for-date with #29/ARIMA/VECM's (0 dates in
common, verified). Fixed to `level_idx_for_diff_row[origin - 1]`, the last row `train` actually
contains — origin dates now match #29/ARIMA exactly (2,094/2,094 rows, `actual`/`naive` identical
to the bit). Substantive finding is unchanged (VAR does not beat naive at any horizon), but the
significance got *stronger*, not weaker: h=1 and h=5 now both show naive significantly better than
VAR-AIC on RMSE and MAE (previously only MAE at h=1, nothing at h=5). Regenerated
`outputs/r3_patha_rmse_mae_vs_naive.csv` and `outputs/r3_patha_diebold_mariano.csv`; added
`outputs/r3_patha_var_aic_forecasts.csv` (per-origin forecasts, previously not exported) so #50 can
merge on `origin_date`.

**Aug 20 — `#50` (pairwise Diebold-Mariano across all Round 3 baselines) shipped.** Seven arms
compared pairwise at all three horizons — Naive, ARIMA-AIC, ARIMA-BIC, VAR-AIC, VAR-BIC, VECM
(6-var), LSTM (21 pairs x 3 horizons = 63 tests, `notebooks/04_diagnostics/dm_pairwise_comparison.ipynb`,
`outputs/r3_pairwise_diebold_mariano.csv`). Decisions made along the way:

- ARIMA-AIC and ARIMA-BIC kept as separate arms, mirroring the VAR-AIC/VAR-BIC sensitivity-check
  precedent above — no forced winner.
- VECM scored on its 6-variable system only, not the 5-variable system PR #59 calls "primary" —
  the 6-var set matches ARIMA/VAR-AIC/VAR-BIC/LSTM's proposal-correct feature set per the Aug 19
  usdcad decision.
- LSTM's forecasts restricted to the same calendar span as the other baselines before comparison
  (its rolling-CV folds cover ~3,728 origins vs. everyone else's ~698-750).
- Discovered mid-build: ARIMA/VAR-AIC/VAR-BIC (`load_levels()`) and VECM/LSTM
  (`build_gold_features()`) run on two different, unreconciled calendars — cross-pipeline pairs
  need explicit verification to avoid silently pairing forecasts against the wrong outcome. Filed
  as `#63`, not fixed here — #50's numbers are correct for the samples reported, just
  smaller-sample for VECM/LSTM pairs than the same-pipeline ones.

**Aug 20/21 — pre-review fix pass on `#50`'s PR (#64), before sending for review.** A deep review
caught that the cross-pipeline verification above (comparing `actual` values with a float
tolerance) wasn't sufficient: `build_gold_features()` forward-fills yield levels before computing
the target spread, so ~16% of `gold_features.csv` rows repeat their prior value, and two genuinely
different real target dates can coincidentally carry the same `actual` value — 12 of 30
ARIMA-AIC-vs-VECM rows at h=20 were silently mispaired this way. Replaced with real verification:
`src/model_comparison.py`'s `attach_target_date()` walks each pipeline's own calendar forward
`horizon` positions from `origin_date` and requires the two sides' real target dates to match, not
just their `actual` values — correctly shrinks the ARIMA-vs-VECM h=20 sample to 18 origins (below
`dieboldmariano`'s minimum for h=20, now reported as `insufficient_sample` instead of crashing or
silently accepting false ties). Also fixed in the same pass: an off-by-one risk from thin
cross-pipeline samples hitting `dm_test()`'s exception path uncaught; a cwd-relative vs.
project-root-absolute path mismatch between the forecast-producing scripts and the comparison
module; `notebooks/03_models/johansen_vecm.ipynb` had silently drifted out of sync with
`evaluate_vecm()`'s signature and could no longer run; and `src/johansen_vecm.py`'s own inline
verdict-printing loop was missing the "mixed RMSE/MAE result" branch that `#43` was originally
filed to fix elsewhere — consolidated `dm_report()` (`EDA_VAR_AIC`), `dm_report_vecm()`
(`johansen_vecm`), and both files' verdict loops onto the one shared `pairwise_dm()` /
`plain_language_verdict()` in `model_comparison.py` so this doesn't drift a third time. All
re-verified against a fresh top-to-bottom notebook run and the full test suite; substantive
findings unchanged.

**Result:** no model significantly beats Naive in the direction that would support a
forecasting-improvement claim, at any horizon, consistent with each baseline's own individual
finding. h=20 has almost no significant pairwise results anywhere in the table.

## Definition of done

- [x] Round 1 shipped two independent, working collection paths (Giti's sequential pull, Lerneir's
      concurrent client classes) before either was picked — not one built and the other skipped
- [x] Round 1 decision recorded above; `notebooks/02_cleaning` runs top-to-bottom from raw and
      produces everything in `data/processed/` using the chosen (or merged) path
- [x] `data/README.md` data dictionary reflects every column actually in `data/processed/` (no
      `TBD` rows left for columns that exist)
- [x] Round 2 shipped two independent EDA artifacts (Nelson's `01_eda` notebook, Mitchel's Power BI
      exploration), each built from whichever Round 1 output that analyst independently judged best
- [x] Round 2 decision recorded above
- [x] Round 3 shipped two independent baseline notebooks (AIC-lag VAR, BIC-lag VAR), both compared
      against the shared Random Walk benchmark
- [x] Round 3 decision recorded above
- [x] Feature engineering (Lerneir) consolidates a single Gold-layer pipeline in `src/` informed by
      both EDA approaches and both baseline attempts — not a third independent attempt
- [ ] Outcome/early-results plan (Nelson) is a short, honest account of what converged and what's
      still open, not a polished narrative that hides the rounds that didn't get a full answer
- [ ] Every notebook re-runs clean, top-to-bottom, from a fresh kernel, with committed outputs —
      no notebook that only works if cells are run out of order
- [x] `src/` has real functions in it (`gold_feature_pipeline.py`, `orchestrator.py`, `clients/`)
      with clean CLI execution and unit testing

- [ ] `README.md` "Roadmap at a Glance" status line reflects Week 5 / M2, not stale Phase 0 text
- [ ] No raw dataset committed if its license forbids redistribution — access instructions in
      `data/README.md` instead

## What NOT to do here

- Don't backfill `04_diagnostics` with padding to look further along than the project is —
  supervisor feedback on a thin-but-honest diagnostics notebook is more useful than feedback on
  content manufactured to fill the checklist.
- Don't resolve the ARIMA/VECM flag by silently dropping it from the notebook without updating
  `03_analytical_objective.md` / `06_expected_outcomes.md` / `07_project_plan.md` to match — that
  just moves the section 03/05/06/07 mismatch into section 05 vs. the actual repo instead of
  fixing it (see `proposal/build/FLAGS.txt`, Lerneir owns the fix).
- Don't hand-edit `data/processed/` output files directly — regenerate them by re-running
  `02_cleaning` so the pipeline stays reproducible.
- Don't wait until Fri Aug 07 to open the tracking issue — open it Mon Aug 03 (per `TIMELINE.md`)
  so gaps surface with days to fix them, not hours.
