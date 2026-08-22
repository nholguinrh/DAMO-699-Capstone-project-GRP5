# SPRINT_AUG21_RECOVERY.md — M2 Closeout + Proposal-Gap Recovery Sprint

**Baseline date:** Fri Aug 14, 2026 (M2's own, already-once-revised due date).

**Risk flag before anything else:** this compresses M2's unfinished third of the checklist *and* six previously-untracked proposal-methodology items into 8 days, on top of `TIMELINE.md`'s
own description of this same week as already "the heavy week.". Fag anything wrong at the Aug 14 sync. If real slippage shows up, follow `TIMELINE.md` Standing Rule 3: label `blocked`, raise at standup, re-plan same day — don't let it erode Week 8's report-writing time silently.

---

## 1. Baseline — where things stand at the start of this sprint (Fri Aug 14)

| Workstream | Status | Evidence |
|---|---|---|
| Round 1 — Data collection | `[x]` done | Giti's sequential pull (#25/PR#35) chosen; Lerneir's concurrent path (#24/PR#36) shipped, follow-up #37 deferred to backlog |
| Round 2 — EDA | `[x]` done | Nelson's Python EDA (#26/PR#38, merged) + Mitchel's Power BI (#27/PR#40, merged) — both kept |
| Round 3 — Baseline VAR (AIC vs BIC) | `[ ]` not started | #28 (Giti), #29 (Mitchel) both open, no PRs |
| Feature engineering (Gold-layer) | `[ ]` not started | #30 (Lerneir) open, no PR |
| Outcome / early-results plan | `[ ]` not started | #31 (Nelson) open, no PR |
| Johansen cointegration test → conditional VECM | `[ ]` not scheduled anywhere | Committed in §5.2; absent from `TIMELINE.md` despite `FLAGS.txt` claiming it's allocated |
| IRF / FEVD / Granger causality | `[ ]` not scheduled anywhere | Committed in §5.3; no task line |
| LSTM (architecture, rolling-CV, training) | `[ ]` not scheduled anywhere | Committed in §5.3; buried in generic "main models implemented" |
| SHAP explanations | `[ ]` not scheduled anywhere | Committed in §5.3 and §8; no task line |
| Diebold-Mariano significance testing | `[ ]` not scheduled anywhere | Committed in §5.4 and §7; not named under Wk7 "diagnostics" |
| TCPS 2 certification (all members) | `[ ]` not scheduled anywhere | Committed in §8; only the *section draft* is in `TIMELINE.md`, not the actual certification |
| Final executive Power BI dashboard (model forecasts) | `[ ]` not scheduled anywhere | Committed in §6/§7 as a Wk9–10 deliverable distinct from the Round 2 EDA dashboard |

## 2. Where the six methodology gaps came from

Cross-checked `proposal/sections/05_analytical_approach.md`, `06_expected_outcomes.md`,
`07_project_plan.md`, `08_ethical_considerations.md` against `TIMELINE.md`. Everything §5.3/§5.4
promises as *interpretation and validation machinery* — Johansen, VECM, IRF, FEVD, Granger, SHAP,
Diebold-Mariano — is compressed into one `TIMELINE.md` line ("main models implemented" /
"Diagnostics + validation"), and TCPS certification + the final dashboard have no line at all.
This sprint gives each of them an explicit slot and owner.

## 3. Day-by-day plan

Dependency chain to respect: **Feature engineering (Gold layer) → VAR/VECM (+ Johansen, IRF,
FEVD, Granger) → ARIMA baseline → LSTM (+ SHAP) → Diebold-Mariano comparison → final dashboard.**
LSTM cannot be meaningfully evaluated before VAR exists (per §5.3 — it's the comparison point,
not a substitute), so nothing after Round 3 can front-run Round 3.

### Fri Aug 14 (today)
- `[x]` Commit + PR the pending `TIMELINE.md` / `M2_CHECKLIST.md` edits recording the Aug 09 sync decisions — **Nelson** (PR #46, merged)
- `[~]` Publish this file, confirm owner assignments at today's sync — **Nelson** (file published via PR #41; the sync itself isn't verifiable from the repo alone — see Aug 22 status note below)
- `[x]` Open tracking issues for the 6 methodology gaps (Johansen/VECM, IRF/FEVD/Granger, LSTM+SHAP, Diebold-Mariano, TCPS cert, final dashboard) — **Nelson** (#47, #48, #49, #50, #51, #52 all opened Aug 16)
- `[ ]` Start TCPS 2 certification (self-paced, ~1–2h each) — **All**, target done by Mon Aug 17 (no certificate evidence anywhere in the repo as of Aug 22 — see #51, still open)
- `[x]` Round 3 AIC-VAR implementation continues, reusing ADF/stationarity results already in `01_eda` — **Giti** (#28 closed, PR #42)
- `[x]` Round 3 BIC-VAR implementation continues — **Mitchel** (#29 closed, PR #45)

### Sat Aug 15
- `[x]` Feature engineering: consolidate Gold-layer pipeline in `src/` (#30) — **Lerneir** (closed, PR #55)
- `[x]` Outcome/early-results plan skeleton (#31), structured but not yet filled with real numbers — **Nelson** (closed, PR #53 — content verified honest against `M2_CHECKLIST.md`'s "What NOT to do" bar; see Aug 22 status note)
- `[x]` Fix #37 (redact FRED key from retry logs, drop dead StatCan URL) if bandwidth allows — **Lerneir** (closed, PR #56 — got done despite being flagged low-priority/optional)

### Sun Aug 16
- `[x]` Round 3 PRs opened for #28/#29 — **Giti, Mitchel** (PR #42, PR #45)
- `[x]` Round 3 review (Nelson reviews AIC path, Lerneir reviews BIC path per `M2_CHECKLIST.md`'s reviewer pairing) (verified: PR #42 approved by Nelson, PR #45 approved by Lerneir)
- `[x]` Short sync: converge Round 3 decision — pick one or keep both as sensitivity check (checklist's own fallback) — **All** (decision recorded in `M2_CHECKLIST.md`: keeping both as a sensitivity check)
- `[x]` Johansen cointegration test on Gold-layer level series — **Lerneir** (#47 closed, PR #59; r=1 cointegrating vector found, VECM estimated)

### Mon Aug 17 — `TIMELINE.md` Week 7 begins
- `[~]` M2 formally closes: Round 3 decision recorded in `M2_CHECKLIST.md`, feature eng + outcome plan merged, `data/README.md` TBD resolved, `README.md` Roadmap line updated off "Phase 0" — **Nelson** (first three sub-items done; `README.md`'s Roadmap line is the one still-open piece — as of Aug 22 it still reads "Phase 0 (Setup) · Next deadline: Proposal, Fri Jul 17")
- `[ ]` Confirm all 4 members' TCPS 2 certificates — **All** (not done — #51 open, no evidence)
- `[x]` VECM estimated (only if Johansen found cointegration) — **Lerneir** (done as part of #47/PR #59 above)
- `[~]` IRF, FEVD, Granger causality computed on the chosen VAR/VECM — **Giti or Mitchel** (whichever path converged; both if kept as sensitivity check) — built and PR opened Aug 22 (#65, referencing #48), awaiting Lerneir's review — not yet merged
- `[x]` ARIMA univariate baseline (Auto-ARIMA, AIC/BIC order selection) — **whichever of Giti/Mitchel isn't on VECM extension** (#54 closed, PR #57)

### Tue Aug 18
- `[x]` LSTM: shallow architecture, rolling-window CV, dropout + early stopping — **Nelson** (#49 closed, PR #58)
- `[x]` SHAP values wired to LSTM output — **Nelson** (same PR #58)
- (parallel, per `TIMELINE.md` — not gap-specific) chapters 02–03 drafting — **All** — `[ ]` **not started as of Aug 22**: `report/chapters/` contains only `NOTES.md`, no chapter drafts exist yet

### Wed Aug 19
- `[x]` Diebold-Mariano pairwise tests: Naive vs. ARIMA vs. VAR/VECM vs. LSTM, across 1/5/20-day horizons — **Nelson** (#50 closed, PR #64 — 63 pairwise tests across 7 model arms)
- `[ ]` Diagnostics/validation batch (VIF, residual/fit tests, CV) per `TIMELINE.md` Wk7 — **model owners, each on their own model** — partially covered (LSTM has rolling-CV; ADF/cointegration diagnostics exist for VAR/VECM) but **no VIF check exists anywhere in the repo**, despite the Gold-layer feature store carrying exact multicollinearity (`yield_spread_10y_2y ≡ yield_10y − yield_2y`, and the two other spread columns) that a VIF pass would catch

### Thu Aug 20
- `[ ]` Final executive Power BI dashboard updated with model forecasts, DM results, uncertainty framing — **Mitchel** (#52 still open; `report/power bi/README.md` is still EDA-only — historical spread/yield/volatility charts, no model output ingested)
- `[x]` Outcome/early-results plan finalized with real numbers (honest about what did/didn't converge, per `M2_CHECKLIST.md`'s own "What NOT to do" guidance) — **Nelson** (done as of Aug 16 per `docs/M2_OUTCOME_PLAN.md` — note its "what's still open" table is now stale: it lists #47/#48/#49/#50 as "not started," which was true Aug 16 but isn't anymore; worth a refresh note before report chapters cite it)
- `[!]` Hard checkpoint: any item still `[ ]` or `[~]` here gets flagged `blocked` per Standing Rule 3, not quietly carried — **this checkpoint itself doesn't appear to have happened**; see Aug 22 status note

### Fri Aug 21 — sprint exit
- `[ ]` Run this file's checklist end-to-end against `M2_CHECKLIST.md` DoD and `TIMELINE.md` Wk7 items
- `[ ]` Retro: anything unclosed gets an explicit carry-forward note into Week 8, not a silent drop
- `[ ]` Update `TIMELINE.md` Week 8 row if any carry-forward changes its scope

**Aug 22 status note (this checklist reconciled against live GitHub + repo state, since the Aug 21
sprint-exit sync didn't happen):** the three boxes above are still genuinely open — no dry run, no
retro, no Week 8 update has occurred. Cross-checking everything above against GitHub issue/PR state
directly (not just this file's own checkboxes) turned up two things worth flagging on their own:
`#48` was checked off in the `#32` tracking issue before its PR (`#65`) existed — now corrected, PR
opened and pending Lerneir's review — and this file's own checkboxes had drifted well behind actual
merged work (most of Aug 15–20 was in fact completed on schedule). The real open items carried into
Week 8 are: `#48` review/merge, `#51` (TCPS certs — no evidence for any member), `#52` (final
dashboard — not started), the VIF/diagnostics gap, and chapters 02+ (report writing hasn't started).

---

## 4. Explicitly out of scope for this sprint

- PR #39 (§3 refinement) — stale but not M2/methodology-critical; handle opportunistically
- Issue #37 (API key log redaction) — backlog per Aug 09 team decision, only in this file as a stretch item
- Chapters 05–09 and the deck/poster — `TIMELINE.md` Weeks 8–11 already own these; this file stops at Week 7's boundary

## 5. University deadlines beyond this sprint (the horizon this week's work is protecting)

Everything above exists to protect these — official, graded, and (per `TIMELINE.md` §1) **immovable**.
Reproduced here from `TIMELINE.md` §1 / `docs/course/DAMO_699-Capstone_Instructions_Summer_2026.md`
so the team sees the full runway, not just this sprint's 8 days.

| # | Milestone | Due | Weight | Governing document |
|---|-----------|-----|--------|--------------------|
| M1 | Project Proposal | Fri Jul 17 ✅ done | 10% | Student Guide Wk 2; Instructions II |
| M2 | Preliminary analysis visible in repo | **Fri Aug 14** (this sprint closes it out through Aug 21) | supervisor review | Student Guide Wk 5 |
| M3 | **Final Report + Dataset + full repo** | **Sun Sep 13** | **60%** | Student Guide Wk 10; Instructions III |
| M3b | Presentation deck submitted | **Wk 10, by Sep 11** | 10% (with discussion) | Presentation Strategies Assessments |
| M4 | Poster presentation | **Wk 11, Sep 14–18 — exact session date still TBD** | (part of 10%) | Poster Instructions §5–6 |
| M5 | Reflection paper | **Wed Sep 16** | 20% | Student Guide Wk 11 |

M3 alone is 60% of the grade — it's the reason this sprint exists. Nothing in Aug 14–21 is worth
protecting at the expense of that date.

### 5.1 Highlighted: report review, visual design, and printing

These three are called out separately because they're easy to treat as "polish at the end" — the
course docs and `TIMELINE.md` both treat them as gated steps with their own lead time, not
buffer-free finishing touches.

**Report review (before Sun Sep 13).**
- The official guidance ("Writing the Final Report") says to *"consult your faculty advisor
  regularly"* — this isn't a one-time pre-submission check, it should be an ongoing touchpoint
  through Weeks 7–9, not compressed into Week 10.
- Internally, this repo's own artifact model already enforces the review gate that matters:
  every chapter is owner + reviewer, and per `CLAUDE.md` nothing is `done` until it scores **≥85
  ("Excellent")** on its mapped rubric criteria. Don't let Sprint-Aug21 pressure skip that gate on
  chapters 02–04 just because they were drafted fast.
- `TIMELINE.md` Week 10 (Wed) already schedules a holistic rubric pass + fix pass before the Sep
  13 submission — treat that as the *final* check, not the *only* check.

**Visual design (deck + poster + report figures).**
- Report figures: `TIMELINE.md` Weeks 7–8 already require each figure to ship as chart + APA
  caption + interpretation paragraph, not a bare screenshot.
- Deck: 18-slide template, built + rubric/design-passed in Week 10 (Thu–Fri) per `TIMELINE.md`.
- Poster: per `docs/course/DAMO 699_Poster Presentation Instructions.pdf` — 36"×48" landscape
  (24"×36" minimum acceptable), professional layout (PowerPoint/Canva/Illustrator), fonts
  readable from 3–4 feet, high contrast, bullet points over dense paragraphs, "professional
  conference standard." Every chart on the poster must support a specific claim — no raw output
  screenshots. Assessed on analytical rigor, clarity, visual communication quality, *and*
  professionalism during Q&A.

**Printing (poster — has its own lead time, don't treat it as a Week-11 task).**
- `TIMELINE.md` M4 row already flags this: poster must go to the printer **≥2 business days
  before the session**, and the printer is **closed weekends**. If the actual Sep 14–18 session
  date lands early in that window (Mon/Tue), printing has to shift *into Week 10*, not Week 11 —
  `TIMELINE.md` §9 already calls this out but the session date is still **TBD**, which blocks
  knowing which week actually owns it.
- The Poster Instructions doc names a specific vendor (TPH The Printing House, 6700 Morrison St
  Unit 3, Niagara Falls — walking distance from campus, ~$25+tax for a colour poster, submit as
  PDF) with a hard **"must drop off by [day] to pick up by Friday, not open weekends"** pattern —
  the exact date in that doc is a leftover from a prior term, so it's a *format/lead-time*
  reference, not this term's literal deadline. Use of that vendor isn't mandatory (the doc says
  so explicitly) but the ≥2-business-day, no-weekend-service constraint should be assumed for
  whichever printer is used.
- **Action needed soon, not urgent this week:** confirm the actual M4 session date with the
  instructor/LMS so the printing deadline can get a real date in `TIMELINE.md` §9 instead of
  "TBD" — this should happen before Week 10, since printing lead time eats into whichever week the
  session falls in.
