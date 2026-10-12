"""Stage 3 - centerlines by non-maximum suppression *in orientation space*.

Each orientation channel is thinned independently across its own normal
direction (and must also be a local maximum over neighbouring orientations).
Because crossing vessels live in different channels, both centerlines survive
through the crossing instead of being merged into one blob, which is the usual
failure of segment-then-skeletonize approaches.

The surviving ridge strength is then hysteresis-thresholded (seed at t_high,
grow down to t_low, 8-connected), bridged over 1-px breaks, and thinned to a
one-pixel skeleton.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .config import Config
from .cortex import OrientationScore

EIGHT = np.ones((3, 3), dtype=bool)


def _shift_bilinear(a: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """a evaluated at (y + dy, x + dx), edge-replicated."""
    return ndi.shift(a, (-dy, -dx), order=1, mode="nearest", prefilter=False)


def orientation_nms(score: OrientationScore, min_sep: int = 3,
                    secondary_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (ridge strength map, ridge channel mask (N,H,W)).

    The dominant orientation is thinned across its normal everywhere. Another
    channel contributes a ridge only where it is a *distinct* orientation
    peak - at least ``min_sep`` bins from the dominant one and at least
    ``secondary_ratio`` of its strength - i.e. a second vessel crossing.
    Merely neighbouring channels on a curved vessel would otherwise lay
    slightly offset parallel ridges that fuse into blobs.
    """
    U = score.U if score.U is not None else score.z
    N = U.shape[0]
    dom = U.argmax(0)
    Udom = np.take_along_axis(U, dom[None], 0)[0]
    keep = np.zeros(U.shape, dtype=bool)
    for i, th in enumerate(score.thetas):
        ny, nx = np.cos(th), -np.sin(th)  # across-vessel normal (y, x)
        a = _shift_bilinear(U[i], ny, nx)
        b = _shift_bilinear(U[i], -ny, -nx)
        ridge = (U[i] >= a) & (U[i] > b)
        omax = (U[i] >= U[(i - 1) % N]) & (U[i] >= U[(i + 1) % N])
        sep = np.minimum((i - dom) % N, (dom - i) % N)
        distinct = (sep == 0) | ((sep >= min_sep) & (U[i] >= secondary_ratio * Udom))
        keep[i] = ridge & omax & distinct
    strength = np.where(keep, U, 0).max(0)
    # With reflect padding the image is mirror-symmetric about its border
    # pixels: odd (edge) and front/back asymmetry responses vanish there and
    # noise is correlated with its mirror image, so the outermost pixels read
    # as unsuppressed lines. Vessels leaving the field just stop 3 px short.
    b = 3
    strength[:b] = strength[-b:] = 0
    strength[:, :b] = strength[:, -b:] = 0
    return strength, keep


def hysteresis(strength: np.ndarray, low: float, high: float) -> np.ndarray:
    lowmask = strength > low
    lab, n = ndi.label(lowmask, structure=EIGHT)
    if n == 0:
        return lowmask
    seeded = np.zeros(n + 1, dtype=bool)
    seeded[np.unique(lab[strength > high])] = True
    seeded[0] = False
    return seeded[lab]


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndi.label(mask, structure=EIGHT)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    ok = sizes >= min_size
    ok[0] = False
    return ok[lab]


def extract_centerlines(score: OrientationScore, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (1-px skeleton, ridge strength)."""
    strength, _ = orientation_nms(score, cfg.nms_min_sep, cfg.nms_secondary_ratio)
    mask = hysteresis(strength, cfg.t_low, cfg.t_high)
    # bridge single-pixel breaks left by NMS where orientation turns sharply
    mask = ndi.binary_closing(mask, structure=EIGHT) | mask
    skel = skeletonize(mask)
    skel = remove_small(skel, cfg.min_length)
    return skel, strength
