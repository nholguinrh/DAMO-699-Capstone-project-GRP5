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

**Round 3 — Baseline models:** chosen lag-order criterion — _TBD_ (or: keeping both as a
sensitivity check). Rationale — _TBD_. Carries into Week 6 per `TIMELINE.md` §6.

## Definition of done

- [x] Round 1 shipped two independent, working collection paths (Giti's sequential pull, Lerneir's
      concurrent client classes) before either was picked — not one built and the other skipped
- [x] Round 1 decision recorded above; `notebooks/02_cleaning` runs top-to-bottom from raw and
      produces everything in `data/processed/` using the chosen (or merged) path
- [ ] `data/README.md` data dictionary reflects every column actually in `data/processed/` (no
      `TBD` rows left for columns that exist)
- [x] Round 2 shipped two independent EDA artifacts (Nelson's `01_eda` notebook, Mitchel's Power BI
      exploration), each built from whichever Round 1 output that analyst independently judged best
- [x] Round 2 decision recorded above
- [ ] Round 3 shipped two independent baseline notebooks (AIC-lag VAR, BIC-lag VAR), both compared
      against the shared Random Walk benchmark
- [ ] Round 3 decision recorded above
- [ ] Feature engineering (Lerneir) consolidates a single Gold-layer pipeline in `src/` informed by
      both EDA approaches and both baseline attempts — not a third independent attempt
- [ ] Outcome/early-results plan (Nelson) is a short, honest account of what converged and what's
      still open, not a polished narrative that hides the rounds that didn't get a full answer
- [ ] Every notebook re-runs clean, top-to-bottom, from a fresh kernel, with committed outputs —
      no notebook that only works if cells are run out of order
- [ ] `src/` has real functions in it if any notebook is duplicating logic across cells (not
      required if nothing's been extracted yet — don't manufacture an abstraction just to fill
      the folder)
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
