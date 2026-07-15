# proposal/build/ — Assembly & Submission Notes

## What this directory is

The assembly point, nothing else. Nothing is authored here — this folder only ever *combines*
the ten finished, rubric-gated files from `proposal/sections/` into the single document that gets
submitted. If you're drafting content, you're in the wrong folder — go to `proposal/sections/`.

## Preconditions before touching this folder

Do not start assembly until **all** of the following are true:

- [ ] All 10 files in `proposal/sections/` are `status: done` (check each file's frontmatter).
- [ ] Every section PR is merged to `main`.
- [ ] `09_references.md` includes every source cited anywhere in sections 01–08 (cross-check).

## What we're going to build

| Output | Path | Format |
|---|---|---|
| Assembled source | `proposal/build/proposal.md` | Markdown, sections concatenated in template order |
| Submission file | `proposal/build/proposal.pdf` | PDF, APA 7, ≤5 pages excl. cover page + references |

### Assembly order (matches the template — do not reorder)

1. `00_title_page.md`
2. `01_introduction.md`
3. `02_problem_definition.md`
4. `03_analytical_objective.md`
5. `04_data_sources.md`
6. `05_analytical_approach.md`
7. `06_expected_outcomes.md`
8. `07_project_plan.md`
9. `08_ethical_considerations.md`
10. `09_references.md`

### How to assemble.

Concatenate the section files in the order above, stripping frontmatter, inserting a page break
before the Title Page and before References (both excluded from the 5-page count), into
`proposal/build/proposal.md`. Then render to PDF with a plain local command, e.g.:

```bash
pandoc proposal/build/proposal.md -o proposal/build/proposal.pdf \
  --pdf-engine=xelatex -V mainfont="Times New Roman" -V fontsize=12pt
```

Whoever runs this should keep the exact command (and any reference/template doc used) noted at
the bottom of this file so it's reproducible by anyone else on the team, not just the person who
first ran it. 

## Checks to run on the assembled PDF

1. **Page limit** — content pages (01–08 only; cover and references excluded) must be **≤ 5
   pages**. If over, cut at the lowest-scoring section first, not proportionally across all.
2. **Holistic rubric pass** — grade the *whole* proposal, not just the sum of its sections,
   against all 5 criteria (see sections NOTES). Cross-section coherence (does §3's RQ match §5's
   method match §7's plan?) is itself part of "Project Organization and Professional Writing."
3. **APA 7 spot-check** — every in-text citation has a matching entry in the reference list; no
   orphaned citations either direction.
4. **No placeholders** — search the assembled file for `TBD`, `[PLACEHOLDER]`, `XX`, etc.
5. **File naming** — confirm the exact filename the submission dropbox expects (e.g.
   `DAMO699_Group5_Proposal.pdf`) against the course dropbox instructions before final export.

## Fix pass (Thu Jul 16, per TIMELINE.md)

Any issue found in the checks above gets fixed by whoever owns the affected section — not by
whoever ran the assembly. Freeze all edits by **21:00 Thu Jul 16** so Friday is submission-only.

## Who runs assembly

Not yet assigned in `TIMELINE.md` (it only names owners down to the section level). Suggest Giti,
since Giti isn't carrying a Mon–Tue drafting section this week and has the most slack before
Thursday — confirm with the team.

## Submission (Fri Jul 17)

- [ ] Upload `proposal.pdf` to the course (Project Proposal, 10%, due Week 3 per the
  Presentation Strategies module's Assessments & Reminders list).
- [ ] Commit the final `proposal.pdf` and `proposal.md` to `proposal/build/` in the repo.
- [ ] Close the milestone `Proposal (Jul 17)` in GitHub Projects.
- [ ] Open a contingency issue `proposal-R&R` (unassigned, left open) in case the supervisor
  requires revise-and-resubmit within 48h, per the Student Guide's Week 3 instructions. Close it
  without action if the proposal is approved as-is.

## What NOT to do here

- Don't hand-edit content inside `proposal.md` after assembly — fix the source section file in
  `proposal/sections/` and re-assemble, or the two will drift out of sync.
- Don't submit before the page-limit check passes, even under deadline pressure — an over-length
  proposal risks a penalty regardless of content quality.
