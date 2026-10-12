"""Stage 0 - retina: luminance adaptation.

Photoreceptors and horizontal cells divide the local signal by a spatially
pooled estimate of the surround luminance, so downstream units see *contrast*
rather than absolute intensity. For an absorbing medium the physically correct
version of that operation is the Beer-Lambert optical density

    OD(x) = -log( I(x) / I_background(x) )

which is (a) invariant to illumination gain, vignetting and exposure, and (b)
*additive* where vessels overlap - the property the junction classifier later
uses to tell crossovers (OD adds) from bifurcations (it doesn't).

The background is estimated with a morphological closing (for dark vessels)
whose structuring element is wider than the largest vessel, i.e. the "surround"
ignores anything vessel-sized, followed by Gaussian smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .config import Config


def to_float(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:  # RGB: use the channel with the best vessel contrast in
        # most absorption imaging (green); callers can pre-select otherwise
        img = img[..., 1] if img.shape[-1] >= 3 else img[..., 0]
    img = img.astype(np.float64)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        img /= np.iinfo(np.asarray(image).dtype).max
    return img


def estimate_background(img: np.ndarray, cfg: Config) -> np.ndarray:
    size = int(2 * np.ceil(2.0 * max(cfg.radii)) + 1)
    if cfg.polarity == "dark":
        bg = ndi.grey_closing(img, size=(size, size))
    else:
        bg = ndi.grey_opening(img, size=(size, size))
    # second, smaller pass removes the blocky look of the rectangular SE
    bg = ndi.uniform_filter(bg, size=size)
    return ndi.gaussian_filter(bg, cfg.adapt_sigma)


def adapt(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Return background-normalized vessel signal: >0 on vessels, ~0 elsewhere.

    dark   (absorption)  optical density -log(I / I_bg): Beer-Lambert, so
                         overlapping vessels add.
    bright (emission)    (I - I_bg) / median(I_bg): fluorescence adds linearly,
                         and a log would amplify noise on the dark background.
    """
    img = to_float(image)
    floor = max(1e-6, float(np.percentile(img, 0.5)) * 0.5, 1e-3 * float(img.max()))
    img = np.maximum(img, floor)
    bg = np.maximum(estimate_background(img, cfg), floor)
    if cfg.polarity == "dark":
        od = -np.log(img / bg)
    else:
        od = (img - bg) / max(float(np.median(bg)), floor)
    return od.astype(np.float32)


def pixel_noise(od: np.ndarray) -> float:
    """Robust per-pixel noise std from the finest-scale residual (MAD)."""
    r = od - ndi.gaussian_filter(od, 1.0)
    mad = np.median(np.abs(r - np.median(r)))
    # residual of white noise after subtracting a sigma=1 blur keeps ~0.76 std
    return float(1.4826 * mad / 0.76) + 1e-12
