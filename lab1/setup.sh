#!/bin/sh
# CV8508 Lab 1 setup: downloads the data zips, unzips them into ./data, and builds a Python environment.
#   sh setup.sh                  normal use (needs internet)
#   sh setup.sh --no-install     data only, skip the Python environment
#   ZIPS_SRC=/path/to/zips sh setup.sh   copy zips from a local folder instead of downloading
#   BASE_URL=https://.../lab1/data sh setup.sh   override the download location
set -e
BASE_URL="${BASE_URL:-https://YOUR-USER.github.io/cv8508-labs/lab1/data}"
ZIPS="human_faces animal_faces backgrounds_objects backgrounds_web fit_test fit_train"
cd "$(dirname "$0")"
mkdir -p data
for z in $ZIPS; do
  f="cv8508_lab1_$z.zip"
  KEEP=0; [ -f "data/$f" ] && KEEP=1   # never delete a zip that was already there
  if [ -f "data/$f" ] || [ -f ".done_$z" ]; then echo "have $f"; else
    if [ -n "$ZIPS_SRC" ]; then cp "$ZIPS_SRC/$f" "data/$f"; else
      echo "downloading $f"
      if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "data/$f" "$BASE_URL/$f"; else wget -q -O "data/$f" "$BASE_URL/$f"; fi
    fi
  fi
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
pip install -q torch torchvision scikit-learn pandas pillow numpy
python - <<'PY'
import torch, torchvision, sklearn, pandas
print("ready: torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| mps", torch.backends.mps.is_available())
PY
echo "Done. Activate with: source .venv/bin/activate"
