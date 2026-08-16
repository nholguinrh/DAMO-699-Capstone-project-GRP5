# TIMELINE.md — DAMO 699 Capstone Roadmap (Source of Truth)

> **Doctrine:** This file defines all dates. The GitHub Projects Roadmap view mirrors it, never the reverse.
> Any date change = commit to this file first → then "Sync the board".
> Status legend: `[ ]` planned · `[~]` in progress · `[x]` done · `[!]` blocked/at risk
>
> **Revision (Sun Aug 09):** Team sync decided to absorb the M2 slippage rather than let it
> bleed into M3. M2's repo-checkpoint due date moves **Fri Aug 07 → Fri Aug 14** (§6); Week 6's
> original modeling/report scope folds into Week 7 instead of getting its own week (§7). M1/M3/
> M3b/M4/M5 graded dates are untouched. Same sync: Round 1 canonical path = sequential pull (§6
> note); Round 2 keeps both EDA approaches (§6 note).

**Team:** Giti · Mitchel · Lerneir · Nelson  | 
**Repo:** https://github.com/nholguinrh/DAMO-699-Capstone-project-GRP5

---

## 1. Milestones (graded deadlines — immovable)

| # | Milestone | Due | Weight | Governing document |
|---|-----------|-----|--------|--------------------|
| M1 | Project Proposal | **Fri Jul 17** | 10% | Student Guide Wk 2; Instructions II; Proposal Template (≤5 pp) |
| M1b | Proposal R&R (contingent) | Within **48h** of supervisor feedback (Wk 3) | — | Student Guide Wk 3 |
| M2 | Preliminary analysis visible in repo | **Fri Aug 14** (revised Aug 09; was Fri Aug 07) | supervisor review | Student Guide Wk 5 |
| M3 | Final Report + Dataset + full repo | **Sun Sep 13** | **60%** | Student Guide Wk 10; Instructions III |
| M3b | Presentation deck submitted | **Wk 10 (by Sep 11)** | 10% (with discussion) | Presentation Strategies Assessments |
| M4 | Poster presentation (poster to printer ≥2 business days prior; printer closed weekends) | **Wk 11, Sep 14–18** — exact session date **TBD** | (part of 10%) | Poster Instructions 5 & printing note |
| M5 | Reflection paper | **Wed Sep 16** | 20% | Student Guide Wk 11 |

---

## 2. Week 1 — Jul 06–10 · Phase 0: Setup + topic lock

**Week outcome:** repo scaffold live, skills 1–2 authored, Project board live, **topic + dataset + RQ candidates decided** (blocks everything in M1).

| Date | Task | Owner | Reviewer | Artifact / Issue |
|------|------|-------|----------|------------------|
| Mon Jul 06 | `[ ]` Attend capstone intro session; record supervisor name + meeting cadence in this file | All | — | — |
| Mon Jul 06 | `[ ]` Commit scaffold (6 structure), issue/PR templates |  |  | `scaffold` |
| Mon Jul 06 | `[ ]` Each member brings ≥1 topic/dataset candidate to Tuesday debate (1-para pitch + data link) | All | — | `docs/topic-candidates.md` |
| Tue Jul 07 | `[ ]` **Topic debate:** score ≥3 candidates against Problem Identification + Data Feasibility rubric criteria; decide by majority | All | — | `docs/topic-decision.md` |
| Tue Jul 07 | `[ ]` Verify chosen dataset is actually downloadable + license permits redistribution (Student Guide: datasets in repo "when permitted") |  |  | `data/README.md` stub |
| Wed Jul 08 | `[ ]` Author `evaluation-criteria` skill (rubrics verbatim + grading procedure) |  |  |  |
| Wed Jul 08 | `[ ]` Watch academic-writing workshop + complete ungraded assessment (Student Guide Wk 1) | All | — | — |
| Thu Jul 09 | `[ ]` Author `collaboration-workflow` skill; create Project board, milestones M1–M5, labels |  |  | |
| Thu Jul 09 | `[ ]` Draft `TEAM_CHARTER.md` (availability, review SLA 24h, escalation) |  | All | `TEAM_CHARTER.md` |
| Fri Jul 10 | `[ ]` Open all 10 proposal-section issues (00–09) with owners, DoD, due dates from Week 2 rows |  | — | issues |
| Fri Jul 10 | `[ ]` Week-1 retro (15 min): confirm Week 2 assignments below still hold | All | — | meeting note |

## 3. Week 2 — Jul 13–17 · Phase 1: Proposal build → **M1 due Fri Jul 17**

**Week outcome:** all 10 section artifacts through draft → PR review → rubric gate (≥85) → assembled, ≤5 pages, PDF submitted.

Section ownership (3-3-4 split; each section reviewed by a different member):

| Date | Task | Owner | Reviewer |
|------|------|-------|----------|
| Mon Jul 13 | `[ ]` Author `capstone-proposal` skill |  |  |
| Mon Jul 13 | `[ ]` Watch literature-review workshop + ungraded assessment (Student Guide Wk 2) | All | — |
| Mon–Tue | `[ ]` Draft `01_introduction`, `02_problem_definition`, `03_analytical_objective` |  |  |
| Mon–Tue | `[ ]` Draft `04_data_sources`, `05_analytical_approach` |  |  |
| Mon–Tue | `[ ]` Draft `06_expected_outcomes`, `07_project_plan`, `08_ethical_considerations` (incl. TCPS confirmation) |  |  |
| Wed Jul 15 | `[ ]` Draft `00_title_page`, `09_references` (APA); all PRs open by EOD |  |  |
| Wed Jul 15 | `[ ]` Peer reviews complete (24h SLA); rubric gate every section ≥85 | All | — |
| Thu Jul 16 | `[ ]` Assemble → `proposal/build/proposal.md` → PDF; **page-limit check ≤5 pp excl. cover/refs**; holistic rubric pass |  | All |
| Thu Jul 16 | `[ ]` Fix pass from holistic review; freeze by 21:00 | owner-per-edit | — |
| **Fri Jul 17** | `[ ]` **Submit Proposal (M1)**; open contingency issue `proposal-R&R` (closed if approved) |  | — |

## 4. Week 3 — Jul 20–24 · Phase 2 begins: data acquisition (+ R&R contingency)

- `[ ]` **If proposal not approved:** R&R within **48h of feedback** + supervisor meeting to finalize scope (Student Guide Wk 3). All other tasks yield to this.
- `[ ]` Acquire raw dataset → `data/raw/` (or documented access instructions if license forbids redistribution) — 
- `[ ]` Data dictionary + APA dataset citation in `data/README.md` — 
- `[ ]` Cleaning notebook `02_cleaning` started; reproducible from raw — 

## 5. Week 4 — Jul 27–31 · Cleaning + EDA

- `[ ]` Cleaning notebook complete; `data/processed/` committed — 
- `[ ]` EDA notebook `01_eda`: distributions, missingness, correlations, ≥3 candidate figures — 
- `[ ]` Baseline/naïve model as benchmark — 
- `[ ]` Supervisor touchpoint (optional per Student Guide Wk 4) — book only if scope questions remain

## 6. Week 5–6 — Aug 03–14 · **M2: repo checkpoint, revised → Fri Aug 14**

Week 5 (Aug 03–07) ran long across the M2 parallel-track rounds; Aug 09 team sync moved the
checkpoint into Week 6 rather than let it eat into Week 7's report work. Round decisions made at
that sync:

- **Round 1 (data collection) — chosen path: Giti's sequential/synchronous pull.** Lerneir's
  concurrent-client path shipped too, but its follow-up fix (`#37` — FRED key leaking into retry
  logs + dead StatCan URL) is deferred: moved to backlog rather than fixed now, since the team
  isn't building further on that path.
- **Round 2 (EDA) — keeping both.** Mitchel's Power BI dashboard (`#27`) reviewed and approved
  alongside Nelson's Python notebook (`#26`) — reproducible record + stakeholder dashboard, per
  the parallel-track model's own rationale for this round.
- Round 3 (baseline models) decision still open — carries into Week 6.

Remaining Week 6 (Aug 10–14) work:

- `[ ]` Round 3 baseline models converge (AIC vs. BIC lag order, or keep both) — Giti, Mitchel
- `[ ]` Feature engineering: consolidated Gold-layer pipeline — Lerneir
- `[ ]` Outcome / early-results plan — Nelson
- `[ ]` Repo visibly shows: preprocessing, feature engineering, baseline models, early results (Student Guide Wk 5 list) — All
- `[ ]` README status section updated; notebooks run top-to-bottom — 
- `[ ]` Attend second online class — All
- `[ ]` Harvest supervisor feedback into issues within 24h — 

## 7. Weeks 7–9 — Aug 17 – Sep 04 · Phase 3: modeling + report chapters

Week 6's original scope (main models + chapters 02–03) folds into Week 7 below instead of taking
its own week — the tradeoff for absorbing the M2 slip without moving M3 (Sun Sep 13, immovable).
Week 7 is now the heavy week; watch it at daily standups per Standing Rule 3.

Weekly rhythm (refine into daily rows at the start of each week):

- **Wk 7 (Aug 17–21):** `[ ]` Author `final-report` skill —  · `[ ]` main models implemented —  · `[ ]` chapters 02–03 drafted —  · `[ ]` Diagnostics + validation (VIF, fit tests, CV as applicable) —  · `[ ]` chapter 04 drafted —  · `[ ]` figure artifacts batch 1 (chart + APA caption + interpretation ¶ each) — 
- **Wk 8 (Aug 24–28):** `[ ]` chapter 05 (analytics application) —  · `[ ]` chapter 06 (findings & discussion) —  · `[ ]` figure batch 2 — 
- **Wk 9 (Aug 31–Sep 04):** `[ ]` chapter 07 (conclusion & recommendations) —  · `[ ]` chapters 08–09 (references, appendices) —  · `[ ]` **chapter 01 executive summary — written LAST, starts only when 02–07 merged** —  · `[ ]` all chapters through rubric gate

## 8. Week 10 — Sep 07–11 · Phase 4: **M3 Sun Sep 13 + M3b deck**

- `[ ]` Mon–Tue: author `final-presentation` skill — ; assemble report → holistic rubric pass → fix pass — All
- `[ ]` Wed: final PDF build; dataset package verified; repo cleanup (no orphan branches, all issues closed/triaged) — 
- `[ ]` Thu–Fri: 18-slide deck built per template slide counts; deck rubric/design pass; **submit deck (M3b)** —  + All
- `[ ]` **Sun Sep 13: submit Final Report + Dataset + repo (M3, 60%)** — 
- `[ ]` Poster draft started in parallel (source file, 36×48" landscape) — 

## 9. Week 11 — Sep 14–18 · **M4 poster + M5 reflection Wed Sep 16**

- `[ ]` Mon Sep 14: poster final PDF; **to printer** (≥2 business days before session; printer closed weekends — if session is Mon/Tue this collapses: confirm session date in Week 10 and shift printing into Week 10 accordingly) — 
- `[ ]` Mon–Tue: Q&A rehearsal ×2, 5–7 min walkthrough each member — All
- `[ ]` **Wed Sep 16: Reflection paper due (M5, 20%)** — inputs harvested from meeting notes + PR history —  drafts, all contribute
- `[ ]` Poster session day: arrive **30 min early** for setup (mandatory, graded) — All

---

## 10. Standing rules

1. Daily granularity is mandatory for the *current and next* week only; farther weeks stay weekly until they come into range (re-planned at each Friday retro).
2. Every `[ ]` row that produces a file maps 1:1 to a GitHub issue with the same owner/reviewer/due date.
3. Slippage >1 day on any M1/M3 predecessor task = label `blocked`, raise at daily standup, re-plan same day. **Protect the 60%:** report tasks outrank presentation tasks in any conflict.
4. This file is itself an artifact: changes go through PR like everything else (exception: status-checkbox flips may be committed directly to main).
