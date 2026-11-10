#!/bin/sh
# CV8508 Lab 4 setup: gets lab4_tools.py, builds .venv and downloads the two local depth models (about 5 GB).
#   sh setup.sh                 normal use
#   sh setup.sh --no-install    lab4_tools.py only
#   SRC=/path/to/lab4 sh setup.sh      copy lab4_tools.py from a folder instead of downloading
set -e
SITE="${SITE:-https://utkarshmall13.github.io/cv8508-labs/lab4}"
cd "$(dirname "$0")"
mkdir -p images pairs segments generated models
if [ -f lab4_tools.py ]; then :
elif [ -n "$SRC" ]; then cp "$SRC/lab4_tools.py" lab4_tools.py
else curl -fL --retry 3 -o lab4_tools.py "$SITE/lab4_tools.py" || wget -q -O lab4_tools.py "$SITE/lab4_tools.py"; fi
[ "$1" = "--no-install" ] && exit 0
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision transformers diffusers accelerate safetensors huggingface_hub \
    numpy scipy pillow matplotlib opencv-python-headless
export HF_HOME="$PWD/models"
python3 - <<'PY'
import lab4_tools as t
from huggingface_hub import snapshot_download
for name, (kind, hub, what) in t.DEPTH_MODELS.items():
    print("fetching", name, hub); snapshot_download(hub)
PY
echo "done:  . .venv/bin/activate && export HF_HOME=\$PWD/models OPENROUTER_API_KEY=..."
