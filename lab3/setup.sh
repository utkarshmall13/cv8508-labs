#!/bin/sh
# Lab 3 setup: fetch lab3_tools.py and build .venv
# options: --no-install (skip .venv); SRC=<local folder> or SITE=<url> to change the source
set -e
SITE="${SITE:-https://utkarshmall13.github.io/cv8508-labs/lab3}"
cd "$(dirname "$0")"
if [ -f lab3_tools.py ]; then echo "have lab3_tools.py"; elif [ -n "$SRC" ]; then cp "$SRC/lab3_tools.py" lab3_tools.py; else
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o lab3_tools.py "$SITE/lab3_tools.py"; else wget -q -O lab3_tools.py "$SITE/lab3_tools.py"; fi
fi
if [ "$1" = "--no-install" ]; then echo "skipped Python environment"; exit 0; fi
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy matplotlib
python - <<'PY'
import lab3_tools as t
c = t.cone_sensitivities()
print("ready: cones", c.shape, "| peaks (nm):", ", ".join(f"{n}={t.WL[c[:, i].argmax()]}" for i, n in enumerate("LMS")))
PY
echo "Done. Activate with: source .venv/bin/activate"
