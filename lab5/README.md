# Lab 5: Aliasing in networks: shifts, resizing and dithering

Shift, shrink, enlarge and dither 50 ImageNet photographs and watch how a standard ResNet-50 and an anti-aliased ResNet-50 change their minds. Tasks are in `Lab5_Aliasing_in_Networks.pdf`.

```sh
git clone --depth 1 https://github.com/utkarshmall13/cv8508-labs.git && cd cv8508-labs/lab5
sh setup.sh
source .venv/bin/activate
```

(or, without cloning: `mkdir lab5 && cd lab5 && curl -LO https://utkarshmall13.github.io/cv8508-labs/lab5/setup.sh && sh setup.sh`.)

`setup.sh` is safe to run again. It gets `lab5_tools.py` and the 50-image pack (already here in a clone), joins the two ResNet-50 weight files (about 200 MB, stored in `data/` as 40 MB parts, so a clone already has them) and checks their checksums (the official PyTorch and antialiased-cnns servers are only a fallback), creates `.venv` (torch, torchvision, antialiased-cnns, numpy, scipy, pillow, matplotlib) and runs a self-check. `sh setup.sh --no-install` stops before the Python environment. No account, key or login is needed. Runs on a laptop CPU; every experiment in the handout takes a few minutes at most.

For a coding agent: run the commands above in `lab5/`, then `python lab5_tools.py` should print the top-1 class names of five images for both networks (all `tench`) and `dither ok`.

- `lab5_tools.py`: `load_images`, `get_models`, `crop_batch` (shifted windows), `crop_rotated` (small rotations), `probs`, `ptrue`, `top1`; `decimate`, `resize`; `quantize` and `dither` (fewer bits per colour channel, with and without error diffusion), `view_from_far`. It finds the downloaded weights by itself; nothing needs to be exported.
- `data/images/`: 50 photographs, 256x256, five from each of ten ImageNet classes, with `labels.csv`.

Solutions are not in this repository. The images are a small sample of ImageNet validation photographs for teaching; ImageNet's terms apply (non-commercial research and education). The anti-aliased model is from Zhang (ICML 2019), code and weights CC BY-NC-SA; the standard ResNet-50 weights are torchvision's.
