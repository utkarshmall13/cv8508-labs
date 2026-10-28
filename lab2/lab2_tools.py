"""CV8508 Lab 2 helpers: data loading, the WHDR metric and probe measurements.
There is no decomposition code here on purpose. Read the functions, then write your own methods."""
import json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def load_image(path):
    """sRGB image file -> float32 array (H, W, 3) in [0, 1] (still sRGB-encoded)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_iiw(split="train"):
    """Intrinsic Images in the Wild subset (Bell, Bala, Snavely 2014), downsized.
    split: 'train' or 'test'. Returns a list of dicts:
      id      : str
      image   : (H, W, 3) float32, sRGB-encoded in [0, 1]
      points  : (P, 2) float32, (x, y) as fractions of width / height, so they survive resizing
      comps   : (M, 4) float32 rows (i, j, darker, weight): i, j index `points`;
                darker = 0 -> human says equal reflectance, 1 -> point i is darker, 2 -> point j is darker;
                weight = human confidence (use it as the weight in the WHDR)
    """
    d = os.path.join(DATA, "iiw", split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} not found: run `sh setup.sh` first")
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        meta = json.load(open(os.path.join(d, f)))
        stem = f[:-5]
        img = [p for p in (stem + ".jpg", stem + ".png") if os.path.exists(os.path.join(d, p))][0]
        out.append(dict(id=stem, image=load_image(os.path.join(d, img)),
                        points=np.array(meta["points"], np.float32).reshape(-1, 2),
                        comps=np.array(meta["comparisons"], np.float32).reshape(-1, 4)))
    return out


def sample(img, points):
    """Read a (H, W) or (H, W, C) map at normalised (x, y) points (nearest pixel)."""
    H, W = img.shape[:2]
    xs = np.clip((points[:, 0] * W).astype(int), 0, W - 1)
    ys = np.clip((points[:, 1] * H).astype(int), 0, H - 1)
    return img[ys, xs]


def luminance(img):
    """(H, W, 3) linear RGB -> (H, W) luminance; a 2-D input is returned unchanged."""
    if img.ndim == 2:
        return img
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def whdr(reflectance, rec, delta=0.10):
    """Weighted Human Disagreement Rate of one reflectance estimate (lower is better).
    `reflectance` is in LINEAR light (H, W) or (H, W, 3); only ratios matter, any positive scale works.
    For each human judgment: predict 'equal' if the two reflectances are within a factor 1+delta,
    otherwise the smaller one is 'darker'; add the judgment's weight to the error if it disagrees."""
    r = np.maximum(luminance(np.asarray(reflectance, np.float32)), 1e-6)
    v = sample(r, rec["points"])
    i, j, darker, w = rec["comps"][:, 0].astype(int), rec["comps"][:, 1].astype(int), rec["comps"][:, 2].astype(int), rec["comps"][:, 3]
    ri, rj = v[i], v[j]
    pred = np.where(rj / ri > 1 + delta, 1, np.where(ri / rj > 1 + delta, 2, 0))
    return float((w * (pred != darker)).sum() / max(w.sum(), 1e-9))


def whdr_dataset(fn, records):
    """Mean WHDR over records. fn(image_srgb) must return a linear reflectance map."""
    return float(np.mean([whdr(fn(r["image"]), r) for r in records]))


def load_confusing():
    """Synthetic scenes with known albedo. Returns {name: dict(image, albedo_gt, probes, pairs, ratios, note)}.
    image / albedo_gt are sRGB-encoded float arrays in [0, 1]; probes are boxes (x0, y0, x1, y1) in fractions of
    the image size; ratios[ 'q/p' ] gives the ground-truth albedo ratio and the pixel ratio of two probes."""
    d = os.path.join(DATA, "confusing")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} not found: run `sh setup.sh` first")
    meta = json.load(open(os.path.join(d, "scenes.json")))
    return {n: dict(image=load_image(os.path.join(d, n + ".png")), albedo_gt=load_image(os.path.join(d, n + "_albedo_gt.png")), **m)
            for n, m in meta.items()}


def probe_mean(linear_map, box):
    """Mean of a linear-light map inside a probe box (works at any resolution)."""
    H, W = linear_map.shape[:2]
    x0, y0, x1, y1 = box
    a = linear_map[int(round(y0 * H)):max(int(round(y1 * H)), int(round(y0 * H)) + 1), int(round(x0 * W)):max(int(round(x1 * W)), int(round(x0 * W)) + 1)]
    return float(luminance(a).mean())


def probe_ratio(linear_map, probes, q, p):
    """mean(map in probe q) / mean(map in probe p), for a LINEAR-light albedo (or image) map.
    If you have an sRGB picture (for example one saved from a chat model), convert it with srgb_to_linear first."""
    return probe_mean(linear_map, probes[q]) / probe_mean(linear_map, probes[p])
