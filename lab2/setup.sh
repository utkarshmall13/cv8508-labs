#!/bin/sh
# CV8508 Lab 2 setup: downloads lab2_tools.py and the two data packs, unzips them into ./data, builds a Python environment.
#   sh setup.sh                          normal use (needs internet)
#   sh setup.sh --no-install             data only, skip the Python environment
#   SRC=/path/to/lab2 sh setup.sh        copy lab2_tools.py and data/*.zip from a local folder instead of downloading
#   SITE=https://.../lab2 sh setup.sh    override the download location
set -e
SITE="${SITE:-https://utkarshmall13.github.io/cv8508-labs/lab2}"
ZIPS="iiw confusing"
cd "$(dirname "$0")"
mkdir -p data
fetch() {  # fetch <relative path under SITE or SRC> <destination>
  if [ -n "$SRC" ]; then cp "$SRC/$1" "$2"; else
    echo "downloading $1"
    if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$2" "$SITE/$1"; else wget -q -O "$2" "$SITE/$1"; fi
  fi
}
[ -f lab2_tools.py ] && echo "have lab2_tools.py" || fetch lab2_tools.py lab2_tools.py
for z in $ZIPS; do
  f="cv8508_lab2_$z.zip"
  KEEP=0; [ -f "data/$f" ] && KEEP=1   # never delete a zip that was already there
  if [ -f "data/$f" ] || [ -f ".done_$z" ]; then echo "have $f"; else fetch "data/$f" "data/$f"; fi
  if [ -f "data/$f" ]; then
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('data')" "data/$f"
    { [ "$KEEP" = 0 ] && rm -f "data/$f"; } 2>/dev/null || true; touch ".done_$z"; echo "unzipped $z"
  fi
done
echo "data/ contains: $(ls data | tr '\n' ' ')"
if [ "$1" = "--no-install" ]; then echo "skipped Python environment"; exit 0; fi
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch numpy scipy pillow matplotlib
python - <<'PY'
import torch, lab2_tools as t
tr, te, cf = t.load_iiw("train"), t.load_iiw("test"), t.load_confusing()
print("ready: torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| mps", torch.backends.mps.is_available())
print(f"IIW subset: {len(tr)} train / {len(te)} test images, {sum(len(r['comps']) for r in tr)} train judgments")
print("confusing scenes:", ", ".join(cf))
PY
echo "Done. Activate with: source .venv/bin/activate"
