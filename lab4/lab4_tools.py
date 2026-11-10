"""Lab 4 helpers: monocular depth models, depth-order scoring, image editing, vanishing-point fitting.

    from lab4_tools import *

* Every depth function returns a `closeness` map: float32 (H, W) in [0, 1], 1 = nearest. A relative map has
  no metric scale: compare order, not values.
* Points are (x, y) in image pixels (x right, y down). A segment is (x1, y1, x2, y2).
* A vanishing point is a homogeneous unit 3-vector (a, b, c); the image point is (a/c, b/c), and c ~ 0 means
  the lines stay parallel in the image.
* OpenRouter functions need  export OPENROUTER_API_KEY=...
"""
import base64, io, json, math, os, urllib.request
import numpy as np
from PIL import Image, ImageDraw

# =====================================================================================================
# 0. Images
# =====================================================================================================
def load_image(path, max_side=768):
    """Load an RGB PIL image; resize so the longer side is <= max_side (depth models are slow on big images)."""
    im = Image.open(path).convert("RGB")
    s = max(im.size)
    if s > max_side:
        im = im.resize((round(im.width * max_side / s), round(im.height * max_side / s)), Image.LANCZOS)
    return im

def _norm01(a, lo=2, hi=98):
    a = np.asarray(a, dtype=np.float32)
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / max(h - l, 1e-6), 0, 1).astype(np.float32)


# =====================================================================================================
# 1. Three monocular depth models (all return a `closeness` map)
# =====================================================================================================
_CACHE = {}
def _device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEPTH_MODELS = {
    "depth-anything-v2": ("direct",     "depth-anything/Depth-Anything-V2-Small-hf", "feed-forward ViT regressor (relative inverse depth)"),
    "marigold":          ("generative", "prs-eth/marigold-depth-lcm-v1-0",           "Stable-Diffusion prior fine-tuned for depth"),
}

def depth_direct(img, hub_id=None):
    """Direct regression: one forward pass of Depth Anything V2 (transformers 'depth-estimation' pipeline)."""
    from transformers import pipeline
    hub_id = hub_id or DEPTH_MODELS["depth-anything-v2"][1]
    key = ("direct", hub_id)
    if key not in _CACHE:
        _CACHE[key] = pipeline("depth-estimation", model=hub_id, device=_device())
    out = _CACHE[key](img)
    d = np.asarray(out["predicted_depth"].squeeze().cpu() if hasattr(out["predicted_depth"], "cpu") else out["predicted_depth"], dtype=np.float32)
    d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))   # (H, W); bigger = nearer (inverse depth)
    return _norm01(d)

def depth_generative(img, hub_id=None, steps=4, ensemble=1, seed=0):
    """Generative prior: Marigold (diffusers). Few-step LCM variant by default; raise steps/ensemble for quality."""
    import torch
    from diffusers import MarigoldDepthPipeline
    hub_id = hub_id or DEPTH_MODELS["marigold"][1]
    key = ("generative", hub_id)
    if key not in _CACHE:
        dev = _device()
        _CACHE[key] = MarigoldDepthPipeline.from_pretrained(hub_id, torch_dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = _CACHE[key](img, num_inference_steps=steps, ensemble_size=ensemble, generator=gen)
    d = np.asarray(out.prediction, dtype=np.float32).squeeze()          # affine-invariant depth in [0, 1], bigger = farther
    d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))
    return _norm01(1.0 - d, 0, 100)

# ----- multimodal LLM through OpenRouter: image editing and the third depth model -----
OR_BASE = "https://openrouter.ai/api/v1"
OR_IMAGE_MODELS = {
    "gemini":     "google/gemini-2.5-flash-image",
    "gpt-image":  "openai/gpt-image-1",
    "seedream":   "bytedance-seed/seedream-4.5",
    "flux":       "black-forest-labs/flux.2-pro",
}

def _or_call(path, payload=None, timeout=240):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise RuntimeError("set OPENROUTER_API_KEY first")
    req = urllib.request.Request(OR_BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter {e.code}: {e.read().decode()[:300]}") from None

def or_key_status():
    """Budget left on the shared key (USD)."""
    d = _or_call("/key")["data"]
    return {k: d.get(k) for k in ("limit", "limit_remaining", "usage")}

def or_image(prompt, img=None, model="gemini", aspect_ratio=None):
    """Text-to-image (img=None) or image-to-image edit (img = PIL image to transform). Returns a PIL image.
    `model` is a key of OR_IMAGE_MODELS or any OpenRouter image-model slug."""
    body = {"model": OR_IMAGE_MODELS.get(model, model), "prompt": prompt}
    if aspect_ratio: body["aspect_ratio"] = aspect_ratio
    if img is not None:
        b = io.BytesIO(); img.convert("RGB").save(b, "PNG")
        body["input_references"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}]
    out = _or_call("/images", body)
    return Image.open(io.BytesIO(base64.b64decode(out["data"][0]["b64_json"]))).convert("RGB")

DEPTH_PROMPT = ("Turn this photograph into a grayscale depth map of the same scene with exactly the same framing: "
                "white for the nearest surfaces, black for the farthest, smooth gradients, no text.")

def mllm_depth(img, model="gemini"):
    """Ask a multimodal LLM to draw the depth map. Returns a closeness map resized to img (white = near).
    The drawing is not guaranteed to line up with the photo: look at it before you score it."""
    d = or_image(DEPTH_PROMPT, img, model).convert("L").resize(img.size, Image.BILINEAR)
    return np.asarray(d, np.float32) / 255.0

# =====================================================================================================
# 2. Scoring depth order at hand-annotated point pairs
# =====================================================================================================
# A pair annotation: {"a": [x, y], "b": [x, y], "closer": "a" or "b"}  (YOUR judgement of which is nearer,
# made from the scene, not from a model). Save the list as JSON next to the image.
def load_pairs(path):  return json.load(open(path))
def save_pairs(pairs, path): json.dump(pairs, open(path, "w"), indent=1)

def sample_at(closeness, pt, r=2):
    x, y = int(round(pt[0])), int(round(pt[1])); h, w = closeness.shape
    return float(np.median(closeness[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]))

def pair_scores_map(closeness, pairs, margin=0.0):
    """Per pair: +1 if the map agrees with your annotation, -1 if not, 0 if the two values differ by < margin."""
    out = []
    for p in pairs:
        d = sample_at(closeness, p["a"]) - sample_at(closeness, p["b"])      # >0 means a nearer
        truth = 1 if p["closer"] == "a" else -1
        out.append(0 if abs(d) < margin else int(np.sign(d)) * truth)
    return np.array(out)

def accuracy(scores):
    s = np.asarray(scores); return float((s == 1).mean()) if len(s) else float("nan")

def rank_agreement(maps, n=4000, seed=0):
    """Spearman correlation between maps at the same random pixels: a quick 'do the models agree?' matrix."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed); h, w = next(iter(maps.values())).shape
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    names = list(maps); M = np.zeros((len(names),) * 2)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = spearmanr(maps[a][ys, xs], maps[b][ys, xs])[0]
    return names, M

def column_profile(closeness, x, y0=None, y1=None):
    """Closeness along one image column (useful for the height-in-the-picture-plane / ground-plane prior)."""
    h = closeness.shape[0]; y0 = 0 if y0 is None else y0; y1 = h if y1 is None else y1
    ys = np.arange(y0, y1); return ys, closeness[ys, int(x)]


# =====================================================================================================
# 3. Cue edits: change one cue, re-run the models, compare
# =====================================================================================================
def flip_vertical(img):   return img.transpose(Image.FLIP_TOP_BOTTOM)
def flip_horizontal(img): return img.transpose(Image.FLIP_LEFT_RIGHT)

def add_haze(img, strength=0.6, direction="up", airlight=(200, 210, 225)):
    """Depth-independent aerial-perspective probe: blend toward `airlight`, more strongly toward the top
    ('up': a ground scene whose far part is high in the picture) or toward the bottom. strength in [0, 1]."""
    a = np.asarray(img, np.float32); h = a.shape[0]
    t = np.linspace(1, 0, h) if direction == "up" else np.linspace(0, 1, h)
    t = (strength * t)[:, None, None]
    return Image.fromarray(np.clip(a * (1 - t) + np.array(airlight, np.float32) * t, 0, 255).astype(np.uint8))

def reduce_contrast(img, factor=0.5):
    a = np.asarray(img, np.float32); m = a.mean((0, 1), keepdims=True)
    return Image.fromarray(np.clip(m + factor * (a - m), 0, 255).astype(np.uint8))

def blur_texture(img, sigma=3):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(sigma))

def paste_scaled(img, box, dest_xy, scale=1.0, feather=4):
    """Copy the region box=(x0,y0,x1,y1) (e.g. a person or a can), resize by `scale`, paste with its centre at
    dest_xy. Familiar-size probe: same object, different retinal size / different height in the picture."""
    from PIL import ImageFilter
    patch = img.crop(box); w, h = patch.size; nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    patch = patch.resize((nw, nh), Image.LANCZOS)
    mask = Image.new("L", (nw, nh), 0); ImageDraw.Draw(mask).rectangle([feather, feather, nw - feather, nh - feather], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
    out = img.copy(); out.paste(patch, (round(dest_xy[0] - nw / 2), round(dest_xy[1] - nh / 2)), mask); return out

def height_prior_map(img):
    """Trivial 'prior only' baseline: closeness grows linearly toward the bottom of the picture (height in the
    picture plane, nothing else). Score it with pair_scores_map like any model: if a real model barely beats it on
    your pairs, the pairs mostly test this one prior."""
    h, w = img.height, img.width
    return np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))

def tilt_flip_probe(img, closeness_fn, region=None):
    """Return (map on the image, map on the vertically flipped image flipped back). A model that uses
    'the ground is at the bottom' will change a lot between them."""
    a = closeness_fn(img); b = closeness_fn(flip_vertical(img))[::-1]
    return a, b

def side_by_side(img, maps, titles=None, path=None):
    """Figure: image | each model's closeness map (viridis, bright = near). Returns the matplotlib figure."""
    import matplotlib; import matplotlib.pyplot as plt
    n = 1 + len(maps); fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.4 * img.height / img.width + 0.4))
    ax[0].imshow(img); ax[0].set_title("input", fontsize=9)
    for a, (k, m) in zip(ax[1:], maps.items()):
        a.imshow(m, cmap="viridis", vmin=0, vmax=1); a.set_title(k if titles is None else titles.get(k, k), fontsize=9)
    for a in ax: a.axis("off")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig


# =====================================================================================================
# 4. Vanishing points
# =====================================================================================================
def _hline(seg):
    x1, y1, x2, y2 = map(float, seg); l = np.cross([x1, y1, 1.0], [x2, y2, 1.0]); return l / (np.linalg.norm(l[:2]) + 1e-12)

def _angles(seg_arr, v):
    """Angle (degrees, 0..90) between each segment and the direction from its midpoint to the VP v (homogeneous)."""
    s = np.asarray(seg_arr, float).reshape(-1, 4); mid = (s[:, :2] + s[:, 2:]) / 2; d = s[:, 2:] - s[:, :2]
    to = np.stack([v[0] - v[2] * mid[:, 0], v[1] - v[2] * mid[:, 1]], 1)      # works for v at infinity (v[2] = 0)
    cr = np.abs(d[:, 0] * to[:, 1] - d[:, 1] * to[:, 0]); dt = np.abs((d * to).sum(1))
    return np.degrees(np.arctan2(cr, dt))

def fit_vp(segments, robust=True):
    """Least-squares vanishing point of a group of segments that are parallel in 3-D.
    Returns (v, residuals): v = unit homogeneous 3-vector, residuals = per-segment angle error in degrees
    (0 for a perfect perspective image of parallel lines; ~0.3-1 deg from clicking noise and lens distortion)."""
    from scipy.optimize import least_squares
    segs = np.asarray(segments, float).reshape(-1, 4)
    if len(segs) < 2: raise ValueError("need at least 2 segments")
    L = np.stack([_hline(s) for s in segs]); v0 = np.linalg.svd(L)[2][-1]
    f = lambda p: np.radians(_angles(segs, p / np.linalg.norm(p)))
    p = least_squares(f, v0 / np.linalg.norm(v0), loss="soft_l1" if robust and len(segs) > 3 else "linear", f_scale=np.radians(1.0)).x
    v = p / np.linalg.norm(p); return v, _angles(segs, v)

def vp_xy(v):
    """Image coordinates of a homogeneous VP, or (inf, inf) at infinity."""
    return (v[0] / v[2], v[1] / v[2]) if abs(v[2]) > 1e-9 else (math.inf, math.inf)

def detect_segments(img, min_len=40):
    """Automatic line segments (OpenCV LSD, falling back to probabilistic Hough). Returns an (N, 4) array."""
    import cv2
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    try:
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
        s = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4)
    except Exception:
        e = cv2.Canny(g, 60, 160); l = cv2.HoughLinesP(e, 1, np.pi / 360, 60, minLineLength=min_len, maxLineGap=6)
        s = np.zeros((0, 4)) if l is None else l.reshape(-1, 4).astype(float)
    ln = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]); return s[ln >= min_len]

def ransac_vps(segments, k=3, thresh_deg=1.5, iters=3000, min_support=6, seed=0):
    """Greedy multi-VP RANSAC on automatic segments: returns a list of (v, inlier_index_array), best first."""
    rng = np.random.default_rng(seed); segs = np.asarray(segments, float).reshape(-1, 4)
    L = np.stack([_hline(s) for s in segs]); remaining = np.arange(len(segs)); out = []
    for _ in range(k):
        if len(remaining) < max(min_support, 2): break
        best = (None, np.array([], int))
        for _ in range(iters):
            i, j = rng.choice(remaining, 2, replace=False); v = np.cross(L[i], L[j]); n = np.linalg.norm(v)
            if n < 1e-12: continue
            v = v / n; inl = remaining[_angles(segs[remaining], v) < thresh_deg]
            if len(inl) > len(best[1]): best = (v, inl)
        if best[0] is None or len(best[1]) < min_support: break
        v, _ = fit_vp(segs[best[1]]); inl = remaining[_angles(segs[remaining], v) < thresh_deg]
        out.append((v, inl)); remaining = np.setdiff1d(remaining, inl)
    return out

def draw_vp_figure(img, groups, vps=None, pad=1.0, path=None):
    """Draw segments, extend them to their VP, and show the VP even when it is outside the picture (pad = extra
    canvas as a multiple of the image size). Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    cols = plt.cm.tab10.colors; W, H = img.size
    fig, ax = plt.subplots(figsize=(7, 7 * (H * (1 + 2 * pad)) / (W * (1 + 2 * pad))))
    ax.imshow(img, extent=[0, W, H, 0]); ax.set_xlim(-pad * W, (1 + pad) * W); ax.set_ylim((1 + pad) * H, -pad * H)
    for i, (k, segs) in enumerate(groups.items()):
        v = vps[i] if vps is not None else fit_vp(segs)[0]
        for s in np.asarray(segs, float).reshape(-1, 4):
            ax.plot([s[0], s[2]], [s[1], s[3]], color=cols[i % 10], lw=2)
            if abs(v[2]) > 1e-9:
                m = (s[:2] + s[2:]) / 2; ax.plot([m[0], v[0] / v[2]], [m[1], v[1] / v[2]], color=cols[i % 10], lw=0.7, alpha=0.7)
        x, y = vp_xy(v)
        if np.isfinite(x): ax.plot(x, y, "o", color=cols[i % 10], mec="k", ms=9, label=k)
    ax.legend(fontsize=8); ax.set_aspect("equal"); fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig

def annotate_segments(img_path, n_per_group=5, groups=("floor lines", "ceiling lines"), out_json=None):
    """Click two end points per line (needs an interactive matplotlib backend: on a Mac this works from a terminal).
    Press Enter with no click to skip a group. Returns {group: [(x1,y1,x2,y2), ...]} and optionally saves JSON."""
    import matplotlib.pyplot as plt
    im = Image.open(img_path).convert("RGB"); res = {}
    for g in groups:
        fig = plt.figure(figsize=(9, 9 * im.height / im.width)); plt.imshow(im); plt.title(f"{g}: click 2 points per line, {n_per_group} lines")
        pts = plt.ginput(2 * n_per_group, timeout=0); plt.close(fig)
        res[g] = [tuple(map(float, (*pts[i], *pts[i + 1]))) for i in range(0, len(pts) - 1, 2)]
    if out_json: json.dump(res, open(out_json, "w"), indent=1)
    return res

def load_segments(path): return {k: [tuple(s) for s in v] for k, v in json.load(open(path)).items()}


# =====================================================================================================
# 5. Example prompts for Task 4
# =====================================================================================================
GEN_PROMPTS = {
    "indoor_hall":   "A photograph of a long, empty hallway with a tiled floor, ceiling lights and doors on both sides, taken at eye level looking straight down the hallway.",
    "indoor_corner": "A photograph of a rectangular living room seen from one corner, showing the floor, two walls, the ceiling and a rectangular rug, taken at eye level.",
    "outdoor_street": "A photograph of a straight city street between tall buildings with parked cars on both sides, taken from the middle of the road at eye level.",
    "outdoor_rails": "A photograph of two straight railway tracks with wooden sleepers going to the horizon across a flat field.",
    "outdoor_facade": "A photograph of the corner of a tall rectangular building with a regular grid of windows, seen from the street at an oblique angle.",
    "indoor_library": "A photograph of a library aisle between two long rows of bookshelves, a polished floor, taken at eye level.",
}
