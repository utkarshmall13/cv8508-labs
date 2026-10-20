#!/bin/sh
# usage: tools/new_lab.sh 2 "Color constancy"   -> creates lab2/ with data/ and a README
set -e
n="$1"; title="$2"; [ -n "$n" ] && [ -n "$title" ] || { echo 'usage: tools/new_lab.sh <number> "<title>"'; exit 1; }
cd "$(dirname "$0")/.."
mkdir -p "lab$n/data" "instructor/lab$n"
[ -e "lab$n/README.md" ] || printf '# Lab %s: %s\nNotebook: `lab%s_<name>_student.ipynb`. Data packs: `data/`.\n' "$n" "$title" "$n" > "lab$n/README.md"
python3 tools/make_index.py
echo "created lab$n/ (student files + data/ go here; answer keys go in instructor/lab$n/, which git ignores)"
