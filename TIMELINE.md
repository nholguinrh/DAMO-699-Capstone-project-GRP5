# TIMELINE.md — DAMO 699 Capstone Roadmap (Source of Truth)

> **Doctrine:** This file defines all dates. The GitHub Projects Roadmap view mirrors it, never the reverse.
> Any date change = commit to this file first → then "Sync the board".
> Status legend: `[ ]` planned · `[~]` in progress · `[x]` done · `[!]` blocked/at risk

**Team:** Giti · Mitchel · Lerneir · Nelson  | 
**Repo:** https://github.com/nholguinrh/DAMO-699-Capstone-project-GRP5

---

## 1. Milestones (graded deadlines — immovable)

| # | Milestone | Due | Weight | Governing document |
|---|-----------|-----|--------|--------------------|
| M1 | Project Proposal | **Fri Jul 17** | 10% | Student Guide Wk 2; Instructions §II; Proposal Template (≤5 pp) |
| M1b | Proposal R&R (contingent) | Within **48h** of supervisor feedback (Wk 3) | — | Student Guide Wk 3 |
| M2 | Preliminary analysis visible in repo | **Fri Aug 07** | supervisor review | Student Guide Wk 5 |
| M3 | Final Report + Dataset + full repo | **Sun Sep 13** | **60%** | Student Guide Wk 10; Instructions §III |
| M3b | Presentation deck submitted | **Wk 10 (by Sep 11)** | 10% (with discussion) | Presentation Strategies §Assessments |
| M4 | Poster presentation (poster to printer ≥2 business days prior; printer closed weekends) | **Wk 11, Sep 14–18** — exact session date **TBD** | (part of 10%) | Poster Instructions §5 & printing note |
| M5 | Reflection paper | **Wed Sep 16** | 20% | Student Guide Wk 11 |

---

## 2. Week 1 — Jul 06–10 · Phase 0: Setup + topic lock

**Week outcome:** repo scaffold live, skills 1–2 authored, Project board live, **topic + dataset + RQ candidates decided** (blocks everything in M1).

| Date | Task | Owner | Reviewer | Artifact / Issue |
|------|------|-------|----------|------------------|
| Mon Jul 06 | `[ ]` Attend capstone intro session; record supervisor name + meeting cadence in this file | All | — | — |
| Mon Jul 06 | `[ ]` Commit scaffold (§6 structure), issue/PR templates |  |  | `scaffold` |
| Mon Jul 06 | `[ ]` Each member brings ≥1 topic/dataset candidate to Tuesday debate (1-para pitch + data link) | All | — | `docs/topic-candidates.md` |
| Tue Jul 07 | `[ ]` **Topic debate:** score ≥3 candidates against Problem Identification + Data Feasibility rubric criteria; decide by majority | All | — | `docs/topic-decision.md` |
| Tue Jul 07 | `[ ]` Verify chosen dataset is actually downloadable + license permits redistribution (Student Guide: datasets in repo "when permitted") | Mitchel | Nelson | `data/README.md` stub |
| Wed Jul 08 | `[ ]` Author `evaluation-criteria` skill (rubrics verbatim + grading procedure) | Lerneir | Mitchel |  |
| Wed Jul 08 | `[ ]` Watch academic-writing workshop + complete ungraded assessment (Student Guide Wk 1) | All | — | — |
| Thu Jul 09 | `[ ]` Author `collaboration-workflow` skill; create Project board, milestones M1–M5, labels |  |  | |
| Thu Jul 09 | `[ ]` Draft `TEAM_CHARTER.md` (availability, review SLA 24h, escalation) | Mitchel | All | `TEAM_CHARTER.md` |
| Fri Jul 10 | `[ ]` Open all 10 proposal-section issues (00–09) with owners, DoD, due dates from Week 2 rows | Nelson | — | issues |
| Fri Jul 10 | `[ ]` Week-1 retro (15 min): confirm Week 2 assignments below still hold | All | — | meeting note |

## 3. Week 2 — Jul 13–17 · Phase 1: Proposal build → **M1 due Fri Jul 17**

**Week outcome:** all 10 section artifacts through draft → PR review → rubric gate (≥85) → assembled, ≤5 pages, PDF submitted.

Section ownership (3-3-4 split; each section reviewed by a different member):

| Date | Task | Owner | Reviewer |
|------|------|-------|----------|
| Mon Jul 13 | `[ ]` Author `capstone-proposal` skill | Lerneir | Nelson |
| Mon Jul 13 | `[ ]` Watch literature-review workshop + ungraded assessment (Student Guide Wk 2) | All | — |
| Mon–Tue | `[ ]` Draft `01_introduction`, `02_problem_definition`, `03_analytical_objective` | Nelson | Lerneir |
| Mon–Tue | `[ ]` Draft `04_data_sources`, `05_analytical_approach` | Lerneir | Mitchel |
| Mon–Tue | `[ ]` Draft `06_expected_outcomes`, `07_project_plan`, `08_ethical_considerations` (incl. TCPS confirmation) | Mitchel | Nelson |
| Wed Jul 15 | `[ ]` Draft `00_title_page`, `09_references` (APA); all PRs open by EOD | Mitchel | Nelson |
| Wed Jul 15 | `[ ]` Peer reviews complete (24h SLA); rubric gate every section ≥85 | All | — |
| Thu Jul 16 | `[ ]` Assemble → `proposal/build/proposal.md` → PDF; **page-limit check ≤5 pp excl. cover/refs**; holistic rubric pass | Nelson | All |
| Thu Jul 16 | `[ ]` Fix pass from holistic review; freeze by 21:00 | owner-per-edit | — |
| **Fri Jul 17** | `[ ]` **Submit Proposal (M1)**; open contingency issue `proposal-R&R` (closed if approved) | Nelson | — |

## 4. Week 3 — Jul 20–24 · Phase 2 begins: data acquisition (+ R&R contingency)

- `[ ]` **If proposal not approved:** R&R within **48h of feedback** + supervisor meeting to finalize scope (Student Guide Wk 3). All other tasks yield to this.
- `[ ]` Acquire raw dataset → `data/raw/` (or documented access instructions if license forbids redistribution) — Mitchel
- `[ ]` Data dictionary + APA dataset citation in `data/README.md` — Lerneir
- `[ ]` Cleaning notebook `02_cleaning` started; reproducible from raw — Nelson

## 5. Week 4 — Jul 27–31 · Cleaning + EDA

- `[ ]` Cleaning notebook complete; `data/processed/` committed — Nelson
- `[ ]` EDA notebook `01_eda`: distributions, missingness, correlations, ≥3 candidate figures — Lerneir
- `[ ]` Baseline/naïve model as benchmark — Mitchel
- `[ ]` Supervisor touchpoint (optional per Student Guide Wk 4) — book only if scope questions remain

## 6. Week 5 — Aug 03–07 · **M2: repo checkpoint Fri Aug 07**

- `[ ]` Repo visibly shows: preprocessing, feature engineering, baseline models, early results (Student Guide Wk 5 list) — All
- `[ ]` README status section updated; notebooks run top-to-bottom — Nelson
- `[ ]` Attend second online class — All
- `[ ]` Harvest supervisor feedback into issues within 24h — Lerneir

## 7. Weeks 6–9 — Aug 10 – Sep 04 · Phase 3: modeling + report chapters

Weekly rhythm (refine into daily rows at the start of each week):

- **Wk 6 (Aug 10–14):** `[ ]` Author `final-report` skill — Lerneir · `[ ]` main models implemented — Mitchel · `[ ]` chapters 02–03 drafted — Nelson
- **Wk 7 (Aug 17–21):** `[ ]` Diagnostics + validation (VIF, fit tests, CV as applicable) — Mitchel · `[ ]` chapter 04 drafted — Lerneir · `[ ]` figure artifacts batch 1 (chart + APA caption + interpretation ¶ each) — Nelson
- **Wk 8 (Aug 24–28):** `[ ]` chapter 05 (analytics application) — Mitchel · `[ ]` chapter 06 (findings & discussion) — Nelson · `[ ]` figure batch 2 — Lerneir
- **Wk 9 (Aug 31–Sep 04):** `[ ]` chapter 07 (conclusion & recommendations) — Lerneir · `[ ]` chapters 08–09 (references, appendices) — Mitchel · `[ ]` **chapter 01 executive summary — written LAST, starts only when 02–07 merged** — Nelson · `[ ]` all chapters through rubric gate

## 8. Week 10 — Sep 07–11 · Phase 4: **M3 Sun Sep 13 + M3b deck**

- `[ ]` Mon–Tue: author `final-presentation` skill — Nelson; assemble report → holistic rubric pass → fix pass — All
- `[ ]` Wed: final PDF build; dataset package verified; repo cleanup (no orphan branches, all issues closed/triaged) — Mitchel
- `[ ]` Thu–Fri: 18-slide deck built per template slide counts; deck rubric/design pass; **submit deck (M3b)** — Lerneir + All
- `[ ]` **Sun Sep 13: submit Final Report + Dataset + repo (M3, 60%)** — Nelson
- `[ ]` Poster draft started in parallel (source file, 36×48" landscape) — Mitchel

## 9. Week 11 — Sep 14–18 · **M4 poster + M5 reflection Wed Sep 16**

- `[ ]` Mon Sep 14: poster final PDF; **to printer** (≥2 business days before session; printer closed weekends — if session is Mon/Tue this collapses: confirm session date in Week 10 and shift printing into Week 10 accordingly) — Mitchel
- `[ ]` Mon–Tue: Q&A rehearsal ×2, 5–7 min walkthrough each member — All
- `[ ]` **Wed Sep 16: Reflection paper due (M5, 20%)** — inputs harvested from meeting notes + PR history — Lerneir drafts, all contribute
- `[ ]` Poster session day: arrive **30 min early** for setup (mandatory, graded) — All

---

## 10. Standing rules

1. Daily granularity is mandatory for the *current and next* week only; farther weeks stay weekly until they come into range (re-planned at each Friday retro).
2. Every `[ ]` row that produces a file maps 1:1 to a GitHub issue with the same owner/reviewer/due date.
3. Slippage >1 day on any M1/M3 predecessor task = label `blocked`, raise at daily standup, re-plan same day. **Protect the 60%:** report tasks outrank presentation tasks in any conflict.
4. This file is itself an artifact: changes go through PR like everything else (exception: status-checkbox flips may be committed directly to main).
