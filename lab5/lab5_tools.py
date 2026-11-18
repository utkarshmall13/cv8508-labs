"""CV8508 Lab 5 tools: aliasing in networks (shifts, resizing, dithering).

Read this file first. Everything works on float tensors x of shape (B, 3, H, W) with values in [0, 1]
(sRGB, not normalised); `probs` does the ImageNet normalisation for you.

    imgs, labels, names = load_images()          # 50 uint8 images, 256x256, with ImageNet class index
    models = get_models()                        # {'resnet50': ..., 'resnet50_aa': ...}
    x = crop_batch(imgs, dx=0)                   # (50, 3, 224, 224)
    p = probs(models['resnet50'], x)             # (50, 1000) softmax
    ptrue(p, labels), top1(p)                    # score of the right class, the model's answer

Models: 'resnet50' is torchvision's standard ResNet-50 (stride-2 convolutions and max pooling, no
low-pass filter). 'resnet50_aa' is the anti-aliased ResNet-50 of Zhang (2019), fine-tuned from the same
standard weights: BlurPool = blur with [1 3 3 1]/8 (separably), then subsample.
"""
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
STORE = 256      # stored image size
CROP = 224       # network input size
BASE = 8         # crop window origin at shift 0; shifts 0..16 stay inside the 256 px image


# ----------------------------------------------------------------------------- data
def load_images(root=None, n=None):
    """Return (imgs, labels, names). imgs: uint8 array (N, 256, 256, 3); labels: ImageNet-1k class index;
    names: file stems. Reads data/images/*.png and data/images/labels.csv (file, class_index, class_name)."""
    root = root or os.path.join(HERE, "data", "images")
    rows = list(csv.DictReader(open(os.path.join(root, "labels.csv"))))
    rows = rows[:n] if n else rows
    imgs = np.stack([np.array(Image.open(os.path.join(root, r["file"])).convert("RGB")) for r in rows])
    labels = np.array([int(r["class_index"]) for r in rows])
    names = [r["class_name"] + ":" + os.path.splitext(r["file"])[0] for r in rows]
    return imgs, labels, names


def to_tensor(imgs_uint8):
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(imgs_uint8)).permute(0, 3, 1, 2).float() / 255.0


def to_uint8(x):
    """float (N, 3, H, W) -> uint8 (N, H, W, 3), for plotting or saving."""
    return (x.clamp(0, 1) * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()


def crop_batch(imgs_uint8, dx=0, dy=0, size=CROP):
    """The 224x224 window at horizontal shift dx and vertical shift dy (pixels, 0..16) of the 256x256 images."""
    assert 0 <= dx <= 16 and 0 <= dy <= 16, "shifts must stay inside the stored image (0..16)"
    ox, oy = BASE + dx, BASE + dy
    return to_tensor(imgs_uint8[:, oy:oy + size, ox:ox + size])


# ----------------------------------------------------------------------------- models
def get_models(names=("resnet50", "resnet50_aa")):
    """Load the models (eval mode, on DEVICE). Weights are cached under $TORCH_HOME (setup.sh sets models/)."""
    import torchvision
    out = {}
    for name in names:
        if name == "resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        elif name == "resnet50_aa":
            import antialiased_cnns
            m = antialiased_cnns.resnet50(pretrained=True, filter_size=4)
        elif name == "vit_b_16":                      # optional task
            m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(name)
        out[name] = m.eval().to(DEVICE)
    return out


@torch.no_grad()
def probs(model, x, batch=25):
    """Softmax over the 1000 ImageNet classes for a float batch (B, 3, H, W) in [0, 1]. Any H, W >= 64
    works for the ResNets (they are fully convolutional with global pooling); the ViT needs 224."""
    out = []
    for i in range(0, len(x), batch):
        xb = ((x[i:i + batch] - MEAN) / STD).to(DEVICE)
        out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def ptrue(p, labels):
    """Probability the model gives to the right class, per image: the 'score' of this lab."""
    return p[np.arange(len(labels)), labels]


def top1(p):
    """The model's answer: index of the most probable class, per image."""
    return p.argmax(1)


def class_name(idx):
    import torchvision
    return torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"][int(idx)]


def consistency(pred_a, pred_b):
    """Fraction of images where two sets of answers agree."""
    return float(np.mean(np.asarray(pred_a) == np.asarray(pred_b)))


# ----------------------------------------------------------------------------- sampling and resizing
def gaussian_kernel1d(sigma, radius=None):
    radius = radius or max(1, int(np.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur_separable(x, k1d):
    """Blur (B, C, H, W) with the 1-D kernel k1d along both axes (reflect padding)."""
    k1d = torch.as_tensor(k1d, dtype=torch.float32, device=x.device)
    c = x.shape[1]
    kh = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    padl, padr = (len(k1d) - 1) // 2, len(k1d) // 2
    x = F.pad(x, (padl, padr, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (0, 0, padl, padr), mode="reflect")
    return F.conv2d(x, kv, groups=c)


def gaussian_blur(x, sigma):
    return x if sigma <= 0 else blur_separable(x, gaussian_kernel1d(sigma))


def blurpool_filter(size=4):
    """The binomial low-pass used by BlurPool: [1 3 3 1]/8 for size 4 (also 2, 3, 5, 6, 7)."""
    a = {1: [1.], 2: [1., 1.], 3: [1., 2., 1.], 4: [1., 3., 3., 1.], 5: [1., 4., 6., 4., 1.],
         6: [1., 5., 10., 10., 5., 1.], 7: [1., 6., 15., 20., 15., 6., 1.]}[size]
    a = torch.tensor(a)
    return a / a.sum()


def decimate(x, factor, phase=0):
    """Direct subsampling: keep every `factor`-th pixel, starting at row/column `phase`. No filtering."""
    return x[..., phase::factor, phase::factor]


def resize(x, size, mode="bilinear", antialias=False):
    """Resize to size=(h, w) with torch's interpolate. mode: 'nearest', 'bilinear', 'bicubic', 'area'.
    antialias=True widens the bilinear/bicubic kernel when shrinking (the PIL / OpenCV INTER_AREA behaviour);
    antialias=False reads only the 2x2 (or 4x4) neighbours of each output pixel, however big the shrink."""
    kw = {} if mode in ("nearest", "area") else {"align_corners": False, "antialias": antialias}
    return F.interpolate(x, size=size, mode=mode, **kw)


# ----------------------------------------------------------------------------- test patterns
def zone_plate(n=512):
    """Zone plate: horizontal and vertical frequency rise linearly from the centre; reaches 0.5 cycle/pixel
    (the Nyquist limit) at the edge midpoints. Returns a (1, 1, n, n) float tensor in [0, 1]."""
    y, x = np.mgrid[-n // 2:n // 2, -n // 2:n // 2].astype(np.float64)
    r2 = x ** 2 + y ** 2
    img = 0.5 + 0.5 * np.cos(np.pi * r2 / n)       # instantaneous frequency = r/n cycles per pixel
    return torch.from_numpy(img).float()[None, None]


def chirp_1d(n=512):
    """Horizontal chirp, (1, 1, 64, n): frequency rises from 0 to 0.5 cycle/pixel along x."""
    x = np.arange(n, dtype=np.float64)
    row = 0.5 + 0.5 * np.sin(np.pi * x ** 2 / (2 * n))
    return torch.from_numpy(np.tile(row, (64, 1))).float()[None, None]


# ----------------------------------------------------------------------------- colour helpers
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def to_gray(x):
    """Luminance of an sRGB batch, in linear light, as (B, 1, H, W)."""
    lin = srgb_to_linear(x)
    return (0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2] + 0.0722 * lin[:, 2:3])


# ----------------------------------------------------------------------------- dithering
def bayer_matrix(n):
    """n x n ordered-dither matrix (n = 2, 4, 8, 16), entries 0..n*n-1."""
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    assert m.shape[0] == n, "n must be a power of two"
    return m


def _floyd_steinberg(img):
    """Error diffusion on one (H, W) array in [0, 1] (serpentine scan). Returns {0, 1}."""
    a = img.astype(np.float64).copy()
    h, w = a.shape
    out = np.zeros_like(a)
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new
            e = old - new
            if 0 <= x + d < w:
                a[y, x + d] += e * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    a[y + 1, x - d] += e * 3 / 16
                a[y + 1, x] += e * 5 / 16
                if 0 <= x + d < w:
                    a[y + 1, x + d] += e * 1 / 16
    return out


def dither(x, method="floyd", linear=True, seed=0):
    """Reduce a batch to two levels (black or white), returned as 3 identical channels, values in {0, 1}.
    method: 'threshold' (fixed cut at middle gray), 'random' (random threshold per pixel), 'bayer2', 'bayer4',
    'bayer8' (ordered dither with that matrix), 'floyd' (Floyd-Steinberg error diffusion, slow: about 0.1 s
    per 224x224 image). linear=True dithers the linear-light luminance, so the fraction of white pixels equals
    the brightness (blurring the result then gives the right gray); linear=False dithers the sRGB numbers.
    The output is in the same encoding the networks expect (sRGB 0 or 1)."""
    g = to_gray(x) if linear else (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3])
    g = g.clamp(0, 1)
    b, _, h, w = g.shape
    if method == "threshold":
        out = ((linear_to_srgb(g) if linear else g) >= 0.5).float()   # cut at middle gray as displayed
    elif method == "random":
        rng = torch.Generator().manual_seed(seed)
        out = (g >= torch.rand(g.shape, generator=rng)).float()
    elif method.startswith("bayer"):
        n = int(method[5:])
        t = (torch.from_numpy(bayer_matrix(n)).float() + 0.5) / (n * n)
        t = t.repeat(h // n + 1, w // n + 1)[:h, :w]
        out = (g >= t).float()
    elif method == "floyd":
        out = torch.from_numpy(np.stack([_floyd_steinberg(gi[0].numpy()) for gi in g])).float()[:, None]
    else:
        raise ValueError(method)
    return out.repeat(1, 3, 1, 1)


def view_from_far(x, sigma, linear=True):
    """Blur an image the way distance does: a Gaussian of `sigma` pixels, applied in linear light (light adds),
    then converted back to sRGB. sigma = 0 returns x."""
    if sigma <= 0:
        return x
    if not linear:
        return gaussian_blur(x, sigma)
    return linear_to_srgb(gaussian_blur(srgb_to_linear(x), sigma))


# ----------------------------------------------------------------------------- self-check
if __name__ == "__main__":
    imgs, labels, names = load_images(n=5)
    ms = get_models()
    x = crop_batch(imgs)
    for name, m in ms.items():
        p = probs(m, x)
        print(f"{name:12s} top-1 {[class_name(i) for i in top1(p)]}  p(true) {np.round(ptrue(p, labels), 2)}")
    d = dither(x[:1], "floyd")
    print("dither ok:", tuple(d.shape), "fraction white", float(d.mean()), "| device:", DEVICE)
