#!/usr/bin/env bash
# Assembles proposal/sections/*.md into proposal/build/proposal.md in template order.
# Strips YAML frontmatter and HTML authoring comments; inserts page breaks before the
# Title Page and before References (both excluded from the 5-page count).
# Nothing is authored here — fix content in proposal/sections/ and re-run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sections="$repo_root/proposal/sections"
out="$repo_root/proposal/build/proposal.md"

order=(
  00_title_page.md
  01_introduction.md
  02_problem_definition.md
  03_analytical_objective.md
  04_data_sources.md
  05_analytical_approach.md
  06_expected_outcomes.md
  07_project_plan.md
  08_ethical_considerations.md
  09_references.md
)

# Drop leading YAML frontmatter block, then HTML comments, then squeeze blank runs.
strip() {
  awk 'NR==1 && $0=="---" {fm=1; next} fm && $0=="---" {fm=0; next} !fm' "$1" \
    | perl -0777 -pe 's/<!--.*?-->//gs'
}

: > "$out"
for f in "${order[@]}"; do
  case "$f" in
    00_title_page.md|09_references.md) printf '\\newpage\n\n' >> "$out" ;;
  esac
  strip "$sections/$f" | sed -e 's/[[:space:]]*$//' >> "$out"
  printf '\n' >> "$out"
done

# Collapse 3+ blank lines to one blank line; trim leading blank lines.
perl -0777 -i -pe 's/\n{3,}/\n\n/g; s/\A\s*\n//' "$out"

echo "wrote $out ($(wc -w < "$out") words)"
