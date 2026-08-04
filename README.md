# DAMO 699 Capstone Project

![Course](https://img.shields.io/badge/Course-DAMO%20699-003366)
![Program](https://img.shields.io/badge/Program-Master%20of%20Data%20Analytics-0a7d3f)
![University](https://img.shields.io/badge/University-Niagara%20Falls-6b1f2a)
![Term](https://img.shields.io/badge/Term-Summer%202026-555555)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

> An end-to-end data analytics consulting project applying the full analytics lifecycle to a real-world problem.

---

## Goal

Working as a team of four data analytics consultants, we apply the complete data analytics lifecycle: problem definition-> data collection and preparation -> modeling -> diagnostics -> interpretation -> and communication. To a real-world problem. Every graded deliverable is decomposed into small, peer-reviewed artifacts and held to the course rubric. Our explicit target is the **Excellent band (≥ 85)** on *every* applicable rubric criterion, verified before any work is called done.

## Team

| Member | Role | GitHub |
|---|---|---|
| Mitchel | Contributor & reviewer (rotating) | [@MitchelP](https://github.com/MitchelP) |
| Giti | Contributor & reviewer (rotating) | [@gitiabhilasa007](https://github.com/gitiabhilasa007) |
| Lerneir | Contributor & reviewer (rotating) | [@Lerneir](https://github.com/Lerneir) |
| Nelson | Contributor & reviewer (rotating) | [@nholguinrh](https://github.com/nholguinrh) |


Specialized roles (lead, data, writing/QA) are assigned in the [Team Charter](./TEAM_CHARTER.md); ownership and review rotate per artifact.

## Roadmap at a Glance
 
**Current status:** 🟢 On track — Phase 0 (Setup) · Next deadline: **Proposal, Fri Jul 17**
 
| Milestone | Due | Weight |
|-----------|-----|--------|
| Proposal | Fri **Jul 17** | 10% |
| Preliminary analysis in repo (supervisor review) | Fri **Aug 07** | checkpoint |
| Final Report + Dataset + repo | Sun **Sep 13** | **60%** |
| Presentation deck | by Fri **Sep 11** | 10% |
| Poster presentation | Week 11 (**Sep 14–18**, date TBD) | — |
| Reflection paper | Wed **Sep 16** | 20% |
 
➡️ Full day-by-day plan: [`TIMELINE.md`](./TIMELINE.md) · Live board: [GitHub Project Roadmap view]([../../projects/6](https://github.com/users/nholguinrh/projects/6))
 
> `TIMELINE.md` is the source of truth for all dates. The Project board mirrors it.
> Update the status line above at every Friday retro.

## Tools & Stack

- **Python** — data preparation, analysis, and modeling (notebooks + reusable modules in `src/`).
- **Power BI** — dashboards and visual communication of results.
- **GitHub** — single source of truth for code, documents, and history.
- **GitHub Projects** — Kanban board for the artifact backlog.
- **GitHub Actions** — automated checks that gate every pull request.
- **`gh` CLI** — issue, PR, and project automation from the terminal.

## Way of Work (collaboration model)

This is how we work — read this before contributing.

- **Everything is an artifact.** Every unit of work is one artifact: **one file = one GitHub issue = one owner + one reviewer.** A proposal section, a report chapter, a figure, a slide — each is its own file, issue, and PR.
- **Branches and PRs only.** All collaboration happens through branches and pull requests. **Two people never edit the same file directly**, and nothing lands on `main` without review.
- **Branch naming:** `artifact/<id>-<slug>` (e.g. `artifact/04-data_sources`). Commits follow **Conventional Commits** (`feat:`, `docs:`, `fix:`, `chore:` …).
- **Kanban flow (GitHub Projects):**

  ```
  Backlog → Ready → In Progress → In Review → Rubric Gate → Done
  ```

  An artifact only reaches **Rubric Gate** with a passing scorecard, and only reaches **Done** after merge.
- **Quality gates — every PR must:**
  1. **Pass peer review** by one teammate other than the author.
  2. **Pass GitHub Actions checks.**
  3. **Score ≥ 85 (Excellent)** on every applicable criterion of the official course rubric, using the `evaluation-criteria` skill. **The scorecard is recorded in the PR** (see the PR template).
- **Issues track every artifact** with a Definition-of-Done checklist, milestone, owner, and due date. If it isn't an issue, it isn't planned; if it isn't Done by the DoD, it isn't done.

## Repository structure

```
.
├── proposal/           # Proposal deliverable — sections/ (00–09) → build/ (assembled PDF)
├── report/             # Final report — chapters/, figures/ (image + caption pairs), build/
├── presentation/       # deck/ (slides → .pptx) and poster/ (source + PDF)
├── data/               # raw/ and processed/ datasets (+ data dictionary & APA citation)
├── notebooks/          # 01_eda, 02_cleaning, 03_models, 04_diagnostics
├── src/                # Reusable Python modules imported by the notebooks
├── .agent/skills/      # Code skills: evaluation-criteria, capstone-proposal, …
├── docs/course/        # Official DAMO 699 course documents (source of truth)
├── TEAM_CHARTER.md     # Team social contract
└── TIMELINE.md         # Schedule and milestones
```

## Key documents

- **Team Charter (Social Contract):** [./TEAM_CHARTER.md](./TEAM_CHARTER.md) — 🚧 in progress
- **Schedule:** [./TIMELINE.md](./TIMELINE.md) — 🚧 in progress
- **Environment & API Setup:** [./docs/SETUP.md](./docs/SETUP.md) — API credentials and local setup
- **Backlog:** [GitHub Project board](https://github.com/users/nholguinrh/projects/6)

## Milestones

| Deliverable | Weight | Due |
|---|---|---|
| Proposal | 10% | Fri, Jul 17, 2026 |
| Final Report & Dataset | 60% | Sun, Sep 13, 2026 |
| Presentation & Poster | 10% | Week 11 (Sep 14–20, 2026) |
| Reflective Paper | 20% | Wed, Sep 16, 2026 |

---

<sub>DAMO 699 — Master of Data Analytics, University of Niagara Falls · Summer 2026</sub>

