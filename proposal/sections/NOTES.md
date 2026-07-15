# proposal/sections/ — Artifact Directory Notes

## What this directory is

Ten independent, individually-owned Markdown files — one per proposal section from the official
template. Each file is a standalone artifact: one owner drafts it, a different teammate reviews
it via PR, and it must clear the rubric gate (≥85, Excellent band) before merge. `proposal/build/`
later assembles all ten into the submitted PDF — nothing in this folder is submitted directly.

**Do not** write the proposal as one big document. **Do** keep each section in its own file so
grading, review, and revision stay independent.

## What "done" looks like for this folder

- [ ] All 10 files below exist, non-empty, and follow the file template (see below).
- [ ] Every file has an open→merged PR carrying a rubric scorecard (per criterion: band + justification).
- [ ] Every file scores ≥85 on every rubric criterion that applies to it (see mapping table).
- [ ] Every corresponding GitHub issue is closed with the final grade noted.
- [ ] No `TBD` / `[PLACEHOLDER]` text remains (supervisor name, group number, dataset specifics, etc.).

## The 10 section artifacts — team of 4

| # | File | Section (per template) | Owner | Reviewer | Rubric criteria | Target length | Due |
|---|------|------------------------|-------|----------|------------------|----------------|-----|
| 00 | `00_title_page.md` | Title Page | **Giti** | Mitchel | Organization & Writing | title block only | Wed Jul 15 |
| 01 | `01_introduction.md` | 1. Introduction (Project Context) | Nelson | Lerneir | Problem ID & Context | ~250 words / 0.5 pp | Tue Jul 14 |
| 02 | `02_problem_definition.md` | 2. Problem Definition | Nelson | Lerneir | Problem ID & Context | ~250 words / 0.5 pp | Tue Jul 14 |
| 03 | `03_analytical_objective.md` | 3. Analytical Objective (incl. RQ + hypotheses) | Nelson | **Giti** | RQ / Analytical Objective | ~375 words / 0.75 pp | Tue Jul 14 |
| 04 | `04_data_sources.md` | 4. Proposed Data Sources | Lerneir | Mitchel | Data Sources & Feasibility | ~375 words / 0.75 pp | Tue Jul 14 |
| 05 | `05_analytical_approach.md` | 5. Proposed Analytical Approach | Lerneir | **Giti** | Analytical Approach & Project Plan | ~375 words / 0.75 pp | Tue Jul 14 |
| 06 | `06_expected_outcomes.md` | 6. Expected Outcomes and Contributions | Mitchel | Nelson | Analytical Approach & Project Plan | ~250 words / 0.5 pp | Tue Jul 14 |
| 07 | `07_project_plan.md` | 7. Project Plan and Timeline | Mitchel | Nelson | Analytical Approach & Project Plan | ~250 words / 0.5 pp | Tue Jul 14 |
| 08 | `08_ethical_considerations.md` | 8. Ethical Considerations | Mitchel | **Giti** | Data Sources & Feasibility + Organization & Writing | ~125 words / 0.25 pp | Tue Jul 14 |
| 09 | `09_references.md` | 9. References (APA) | **Giti** | Nelson | Organization & Writing | grows with citations | Wed Jul 15 |

**Ownership load:** Nelson 3 sections (01,02,03) · Lerneir 2 (04,05) · Mitchel 3 (06,07,08) ·
Giti 2 (00,09) — a 3-2-3-2 split, replacing the old 3-person 3-3-4 imbalance. Nobody reviews a
section they own. 

Length budget for content sections (01–08) totals ~2,250 words / ~4.5 pages, leaving a buffer
inside the 5-page cap (excl. cover + references). These are targets, not hard limits — the real
gate is the rendered page count of the assembled PDF, checked in `proposal/build/` on Thu Jul 16.

## Rubric criteria this folder is graded against

From the official proposal rubric (Instructions Summer 2026):

1. **Problem Identification and Context** (CLO 1)
2. **Research Question / Analytical Objective** (CLO 1)
3. **Data Sources and Feasibility** (CLO 2)
4. **Analytical Approach and Project Plan** (CLO 2, CLO 3)
5. **Project Organization and Professional Writing** (CLO 4, CLO 5)

Full band tables (Fail / Developing / Satisfactory / Good / Excellent, 0–100) belong in an
`evaluation-criteria` reference file (`docs/evaluation-criteria.md`). Grade manually against the rubric text in the Instructions PDF, or paste
a drafted section into a ai tool chat and ask for a rubric scorecard against these five criteria.

## File template — use this frontmatter + structure in every section file

```markdown
---
section: 01
title: Introduction
owner: Nelson
reviewer: Lerneir
status: draft            # draft -> in-review -> rubric-gate -> done
rubric_criteria: [Problem Identification and Context]
target_words: 250
last_rubric_score: null  # filled in at Rubric Gate, e.g. "Excellent (90) — see PR #12"
---

<!-- Section content starts here. Formal academic register, APA 7, active voice,
     no first person unless the course docs explicitly allow it. -->
```

## Workflow for each section (per artifact) — now manual.

1. **Draft** against the template requirements in `DAMO 699_Template Proposal.pdf` for that section.
2. **Self-check** — re-read the rubric criteria mapped above; would a strict grader place this in
   Excellent (85–100)? If not, revise before opening a PR.
3. **Open PR** into `main` referencing the section's GitHub issue (branch: `artifact/section-0X-slug`).
4. **Peer review** by the assigned reviewer within the 24h SLA — reviewer adds the rubric
   scorecard (criterion → band → one-line justification) as a PR comment. 
5. **Rubric Gate**: if any criterion scores below 85, request changes and loop back to step 1.
6. **Merge** once ≥85 on every applicable criterion + 1 approval. Close the issue with the grade.

## What NOT to do here

- Don't write the cover page fields or APA reference format from memory — pull them exactly from
  `DAMO 699_Template Proposal.pdf` (Title Page section) and the APA guide referenced in the Final
  Report writing module.
- Don't duplicate content across sections (e.g., restating the RQ in both §1 and §3) — link back
  instead ("as introduced in Section 1...").
- Don't merge a section without a reviewer's rubric scorecard on the PR, even if it "looks done."
