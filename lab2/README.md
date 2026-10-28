# Lab 2: Light or paint? Two ways to split an image into reflectance and shading

One hour. You build two intrinsic-image methods and see where they disagree with each other, with the truth, and with a recent multimodal model.

## Start

```sh
mkdir lab2 && cd lab2
curl -LO https://utkarshmall13.github.io/cv8508-labs/lab2/setup.sh
sh setup.sh
. .venv/bin/activate
```

`setup.sh` downloads `lab2_tools.py` and two small data packs, unzips them into `data/`, creates a Python environment and runs a self-check. Nothing else to install.

## What you get

- `lab2_tools.py`: data loaders (`load_iiw`, `load_adelson`), the human-judgment score `whdr`, the probe measurement `probe_ratio`, and colour helpers `srgb_to_linear` / `linear_to_srgb`.
- `data/iiw/`: a small Intrinsic Images in the Wild subset with human "which point is darker paint" judgments.
- `data/adelson/`: the checker-shadow image (512x390) with probe boxes on the squares A and B and the true albedo ratio (`adelson.json`).

The handout (`Lab2_Light_or_Paint_Intrinsic_Images.pdf`) has the tasks. Solutions are not in this repository.

## Data note

The IIW subset is a downsized sample for teaching. IIW is by Bell, Bala and Snavely (SIGGRAPH 2014); please cite it if you use it beyond the lab.
