"""CV8508 Lab 5 tools: aliasing in networks (shifts, resizing, dithering).

Read this file first. Everything works on float tensors x of shape (B, 3, H, W) with values in [0, 1]
(sRGB, not normalised); `probs` does the ImageNet normalisation for you.

    imgs, labels, names = load_images()          # 50 uint8 images, 256x256, with ImageNet class index
    models = get_models()                        # {'resnet50': ..., 'resnet50_aa': ...}
    x = crop_batch(imgs, dx=0)                   # (50, 3, 224, 224)
    p = probs(models['resnet50'], x)             # (50, 1000) softmax
    ptrue(p, labels), top1(p)                    # score of the right class, the model's answer

The weights are looked up in lab5/models (TORCH_HOME is set for you).

Models: 'resnet50' is torchvision's standard ResNet-50 (stride-2 convolutions and max pooling, no
low-pass filter). 'resnet50_aa' is the anti-aliased ResNet-50 of Zhang (2019), fine-tuned from the same
standard weights: BlurPool = blur with [1 3 3 1]/8 (separably), then subsample.
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "models"))   # weights downloaded by setup.sh live here

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


def rotate(x, degrees):
    """Rotate a float batch (B, 3, H, W) about its center by `degrees` (counter-clockwise), bilinear interpolation,
    edges reflected. 0 degrees returns x unchanged."""
    if degrees == 0:
        return x
    th = torch.tensor(degrees * np.pi / 180.0)
    c, s_ = torch.cos(th), torch.sin(th)
    A = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0]]).unsqueeze(0).repeat(len(x), 1, 1).float()
    grid = F.affine_grid(A, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def crop_rotated(imgs_uint8, degrees, size=CROP):
    """The 224x224 center window of the 256x256 images after rotating them by `degrees` (use 0 to 5)."""
    x = rotate(to_tensor(imgs_uint8), degrees)
    o = (x.shape[-1] - size) // 2
    return x[..., o:o + size, o:o + size]


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


def decimate(x, factor, phase=0):
    """Direct subsampling: keep every `factor`-th pixel, starting at row/column `phase`. No filtering."""
    return x[..., phase::factor, phase::factor]


def resize(x, size, mode="bilinear", antialias=False):
    """Resize to size=(h, w) with torch's interpolate. mode: 'nearest', 'bilinear', 'bicubic', 'area'.
    antialias=True widens the bilinear/bicubic kernel when shrinking (the PIL / OpenCV INTER_AREA behaviour);
    antialias=False reads only the 2x2 (or 4x4) neighbours of each output pixel, however big the shrink."""
    kw = {} if mode in ("nearest", "area") else {"align_corners": False, "antialias": antialias}
    return F.interpolate(x, size=size, mode=mode, **kw)


# ----------------------------------------------------------------------------- colour helpers
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)



# ----------------------------------------------------------------------------- reducing colour depth
def quantize(x, bits=4):
    """Round every colour channel to 2**bits levels (8 bits = the original, 4 bits = 16 levels per channel).
    Smooth gradients turn into visible bands."""
    n = 2 ** bits - 1
    return torch.round(x.clamp(0, 1) * n) / n


def _floyd_steinberg(img, n):
    """Error diffusion on one (H, W) array in [0, 1] with n+1 output levels (serpentine scan)."""
    a = img.astype(np.float64).copy()
    h, w = a.shape
    out = np.zeros_like(a)
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            old = a[y, x]
            new = min(1.0, max(0.0, round(old * n) / n))
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


def dither(x, bits=4):
    """Reduce every colour channel to 2**bits levels like `quantize`, but spread the rounding error to the
    neighbouring pixels (Floyd-Steinberg error diffusion, each channel on its own), so that bands turn into fine
    noise whose local average is the original colour. Slow: about 0.2 s per 224x224 image. Returns (B, 3, H, W)."""
    n = 2 ** bits - 1
    xs = x.clamp(0, 1).cpu().numpy()
    out = np.stack([[_floyd_steinberg(img[c], n) for c in range(3)] for img in xs])
    return torch.from_numpy(out).float()


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
    d = dither(x[:1], 2)
    print("dither ok:", tuple(d.shape), "levels per channel", len(torch.unique(d)), "| device:", DEVICE)
