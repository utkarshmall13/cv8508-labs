"""Lab 3 helpers: the three cone sensitivities (clipped Gaussians) and a colour function for plots."""
import numpy as np

WL = np.arange(380, 781)                         # wavelength grid, nm
PEAKS = {"L": 560.0, "M": 530.0, "S": 420.0}     # nm
SIGMAS = {"L": 45.0, "M": 40.0, "S": 30.0}       # nm

# LMS -> XYZ, only used to colour plots
LMS_TO_XYZ = np.array([[1.81178, -1.30800, 0.36481],
                       [0.60480, 0.41726, -0.03938],
                       [-0.04974, 0.08113, 1.82140]])
_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def cone_sensitivities(wl=WL):
    """Sensitivity of the L, M and S cones at wavelengths wl (nm), shape (len(wl), 3), each curve peaking at 1.
    Gaussians in wavelength, zero outside 380-780 nm."""
    wl = np.asarray(wl, dtype=float)
    out = np.stack([np.exp(-0.5 * ((wl - PEAKS[k]) / SIGMAS[k]) ** 2) for k in "LMS"], axis=-1)
    return out * ((wl >= 380) & (wl <= 780))[..., None]


def lms_to_srgb(lms):
    """Approximate screen colour of LMS triples, (..., 3) -> sRGB in [0, 1]. Brightness is normalised;
    colours the screen cannot show are desaturated by adding white."""
    lms = np.asarray(lms, dtype=float)
    lin = lms @ (_XYZ_TO_SRGB @ LMS_TO_XYZ).T
    lin = lin + np.maximum(0.0, -lin.min(axis=-1, keepdims=True))
    lin = lin / np.maximum(lin.max(axis=-1, keepdims=True), 1e-12)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055).clip(0, 1)
