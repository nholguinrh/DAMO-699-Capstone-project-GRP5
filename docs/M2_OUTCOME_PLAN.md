---
title: M2 Outcome / Early-Results Plan
owner: Nelson
reviewer: Mitchel
status: draft
last_rubric_score: null
---

# M2 Outcome / Early-Results Plan

Single-owner close-out task for M2 (issue `#31`) — a short, honest account of what Rounds 1–3
converged on and why, and what's still genuinely open going into Weeks 7–9. Pulls directly from
the convergence-decision log in `M2_CHECKLIST.md` (PR `#46`) rather than re-litigating it. Doesn't
try to look further along than the project actually is — see `M2_CHECKLIST.md`'s own "What NOT to
do here."

## What converged

### Round 1 — Data collection: Giti's sequential pull (Path A)

Both paths shipped working, independent collection code before either was picked — Giti's
sequential/synchronous pull and Lerneir's per-source concurrent client classes. The team chose
Giti's path at the Aug 09 sync. Lerneir's path works too; its one follow-up defect (`#37` — FRED
API key leaking into retry logs, plus a dead StatCan URL) is deferred to the backlog rather than
fixed now, since nothing in the pipeline builds on Path B going forward.

### Round 2 — EDA: keeping both

Nelson's Python-native EDA (`#26`, `notebooks/01_eda`) and Mitchel's Power BI dashboard (`#27`)
were never competing on correctness — one's a reproducible analytical record, the other's a
stakeholder-facing dashboard. Both reviewed and approved Aug 09; the team kept both by design,
per `M2_CHECKLIST.md`'s own rationale for this round.

### Round 3 — Baseline models: keeping both as a sensitivity check

This is the round with an actual empirical finding, not just a process decision. Two VAR
baselines were built against the shared Random Walk benchmark, differing only in lag-order
criterion:

| | AIC path (`#28`) | BIC path (`#29`) |
|---|---|---|
| Feature set | `yield_spread_10y_2y`, `overnight_rate`, `us_treasury_10y`, `fed_funds_rate`, `cpi_yoy` | same (aligned during `#29` review — an earlier draft included `usdcad`, corrected) |
| Selected lag | 10 | 0 (effectively a constant-drift model) |
| Beats naive on RMSE or MAE, any horizon? | No | No |
| Diebold-Mariano significant result | Naive significantly better at h=1, MAE (p=0.0005) | Naive significantly better at h=1, MAE (p=0.0003) |

**Neither lag-order criterion produces a VAR that beats the naive random-walk floor at 1-, 5-, or
20-day horizons**, and the two paths independently replicate the same significant result (naive
wins at h=1 on MAE). Because the two notebooks weren't a perfectly controlled comparison as first
built (see feature-set note above), the team isn't declaring an AIC-vs-BIC winner — both are kept
as a documented sensitivity check. The actionable finding isn't "AIC beats BIC" or vice versa;
it's that **linear VAR, under either criterion, doesn't clear the naive baseline on this dataset**.
That's the honest headline for M2, and it's the reason the nonlinear (LSTM) and cointegration
(Johansen/VECM) work below carries real weight rather than being a formality.

Full detail and rationale: `M2_CHECKLIST.md` §"Convergence decisions."

## What's still preliminary or genuinely open

Everything below is **not started** as of this writing (Aug 16) unless noted. Listed here so
nothing is silently carried forward without a name attached.

| Item | Issue | Owner | Status |
|---|---|---|---|
| Feature engineering — consolidated Gold-layer pipeline in `src/` | `#30` | Lerneir | Not started |
| Johansen cointegration test + conditional VECM | `#47` | Lerneir | Not started (depends on `#30`) |
| IRF / FEVD / Granger causality | `#48` | Giti or Mitchel | Not started (depends on `#30`, Round 3 path) |
| LSTM (rolling-CV architecture) + SHAP explanations | `#49` | Nelson | Not started (depends on `#30`) |
| Diebold-Mariano pairwise testing across the *full* model set (Naive/ARIMA/VAR/LSTM) | `#50` | Nelson | Partially done — the Round 3 VAR-vs-naive DM tests above exist; ARIMA and LSTM aren't built yet, so the full pairwise comparison can't run |
| TCPS 2 certification, all members | `#51` | All | Not started |
| Final executive Power BI dashboard with model forecasts | `#52` | Mitchel | Not started (depends on `#50`) |
| PR `#45` (BIC path) merge | `#29` | Mitchel / Lerneir | Code and fixes pushed; still needs Lerneir's review |

**On `#22`/`#23` (ARIMA/VECM model-list alignment, supervisor feedback):** these were proposal
*document* fixes and are closed — `03_analytical_objective.md` / `05_analytical_approach.md` /
`06_expected_outcomes.md` / `07_project_plan.md` now correctly name VECM and ARIMA as committed
methodology. That's a documentation fix, not an implementation — the actual Johansen/VECM work is
what `#47` tracks, and `#47` is still open. Don't read `#22`/`#23` being closed as "VECM is done."

## What this means going into Weeks 7–9

- The dependency chain from `SPRINT_AUG21_RECOVERY.md` still holds: feature engineering (`#30`) →
  VAR/VECM extensions (`#47`, `#48`) → ARIMA baseline → LSTM+SHAP (`#49`) → full DM comparison
  (`#50`) → final dashboard (`#52`). Nothing past Round 3 can front-run `#30`.
- Because VAR hasn't cleared the naive floor, the report's "early results" framing for M2 should
  be **the naive baseline is currently the strongest candidate, pending Johansen/VECM and LSTM**,
  not "VAR is the working baseline." Overstating VAR's performance here would need walking back
  later.
- ARIMA has no tracking issue yet despite being named in the dependency chain above and in
  `proposal/sections/05_analytical_approach.md` §5.3 — worth opening one alongside `#47`–`#50`
  rather than letting it stay implicit inside "Round 3 baseline models."

## Definition of done (per `#31`)

- [x] Short, honest account of what each round converged on and why
- [x] States what's still preliminary vs. open, including the `#22`/`#23` documentation-vs-implementation
      distinction
- [ ] PR opened, referencing `#31`
- [ ] Mitchel reviews
- [ ] Close once merged
