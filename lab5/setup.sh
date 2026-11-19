#!/bin/sh
# Lab 5 setup: fetch lab5_tools.py, the image pack and the two ResNet-50 weight files, then build .venv.
# Works from a clone (git clone https://github.com/utkarshmall13/cv8508-labs; cd cv8508-labs/lab5; sh setup.sh)
# or alone (curl -LO .../lab5/setup.sh; sh setup.sh). Safe to run again.
# options: --no-install (tools, images and weights only, no Python environment)
#   SRC=<local folder> | SITE=<url> override where files come from
set -e
SITE="${SITE:-https://utkarshmall13.github.io/cv8508-labs/lab5}"
cd "$(dirname "$0")"
fetch() {  # fetch <url> <dest>
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -C - -o "$2" "$1"; else wget -c -q -O "$2" "$1"; fi
}
get() {  # get <file> <dest>   (from SRC, else SITE)
  if [ -f "$2" ]; then return; fi
  mkdir -p "$(dirname "$2")"
  if [ -n "$SRC" ]; then cp "$SRC/$1" "$2"; else fetch "$SITE/$1" "$2"; fi
}
get lab5_tools.py lab5_tools.py
get data/cv8508_lab5_images.zip data/cv8508_lab5_images.zip
if [ ! -f data/images/labels.csv ]; then python3 -c "import zipfile; zipfile.ZipFile('data/cv8508_lab5_images.zip').extractall('data')"; fi
mkdir -p out

# weights: stored in this repository as 40 MB parts (data/<name>.part0, part1, ...), joined here and checked against the
# first 8 hex digits of the sha256 in the filename; the official servers are the fallback
CK="$PWD/models/hub/checkpoints"; mkdir -p "$CK"
weight() {  # weight <file> <official url>
  name="$1"; f="$CK/$name"; h="${name##*-}"; h="${h%.pth}"
  ok() { [ -f "$f" ] && { if command -v sha256sum >/dev/null 2>&1; then s=$(sha256sum "$f" | cut -c1-8); else s=$(shasum -a 256 "$f" | cut -c1-8); fi; [ "$s" = "$h" ]; }; }
  ok && return
  rm -f "$f"; : > "$f.tmp"; i=0
  while :; do
    p="data/$name.part$i"
    if [ ! -f "$p" ]; then
      if [ -n "$SRC" ] && [ -f "$SRC/$p" ]; then cp "$SRC/$p" "$p"
      else fetch "$SITE/$p" "$p" 2>/dev/null || { rm -f "$p"; break; }; fi
    fi
    cat "$p" >> "$f.tmp"; i=$((i+1))
  done
  mv "$f.tmp" "$f"
  ok && return
  rm -f "$f"; fetch "$2" "$f" || true
  ok || { echo "could not get $name with a matching checksum; ask for the file in class" >&2; rm -f "$f"; exit 1; }
}
weight resnet50-0676ba61.pth https://download.pytorch.org/models/resnet50-0676ba61.pth
weight resnet50_lpf4_finetune-cad66808.pth https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4_finetune-cad66808.pth

if [ "$1" = "--no-install" ]; then echo "skipped Python environment"; exit 0; fi
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision antialiased-cnns numpy scipy pillow matplotlib
export TORCH_HOME="$PWD/models"
python lab5_tools.py
echo "Done. Activate with:  source .venv/bin/activate"
