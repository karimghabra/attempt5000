"""Synthetic vascular networks with exact ground truth.

A fixed-weight system can't be trained to fit a dataset, so it has to be shown
to work across conditions. This generator gives full control over the factors
the user listed - length, tortuosity, branching rate, diameter, optical
density, blur, overlap - plus illumination, texture and sensor noise.

Physics: each vessel is a cylinder imaged in transmission. Its optical density
is the chord length times an absorption coefficient, OD(d) = 2 k sqrt(r^2-d^2),
so thin vessels are fainter, as in real images. Separate trees are separate
depth layers: each is blurred by its own defocus PSF and their transmittances
multiply (optical densities add where vessels cross). Within a tree, lumens are
united (max), so bifurcations do not darken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Branch:
    pts: np.ndarray           # (K, 2) float, (x, y)
    radii: np.ndarray         # (K,)
    tree: int
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    exits: bool = False       # ended by leaving the field of view
    contrast: float = 0.0     # median noise-free OD on its centerline (after blur)


@dataclass
class TreeStyle:
    kappa: float              # absorption coefficient (OD per px of chord)
    blur: float               # defocus sigma (px)
    reflex: float = 0.0       # central light reflex depth (0..1)


@dataclass
class Network:
    shape: tuple[int, int]
    branches: list[Branch]
    styles: list[TreeStyle]
    junctions: list[dict]     # {'x','y','type','radius', ...}

    def gt_points(self, types=("bifurcation", "crossover")) -> list[dict]:
        return [j for j in self.junctions if j["type"] in types]


# ------------------------------------------------------------------ growth
def _murray_daughters(r, rng, asym_range):
    a = rng.uniform(*asym_range)               # r2 / r1
    r1 = r / (1 + a**3) ** (1 / 3)
    r2 = a * r1
    c1 = np.clip((r**4 + r1**4 - r2**4) / (2 * r**2 * r1**2), -1, 1)
    c2 = np.clip((r**4 + r2**4 - r1**4) / (2 * r**2 * r2**2), -1, 1)
    return (r1, np.arccos(c1)), (r2, np.arccos(c2))


def grow_network(
    shape=(512, 512),
    n_trees=3,
    seed=0,
    r_root=(2.5, 5.5),
    r_min=0.9,
    tortuosity=0.03,
    persistence=0.97,
    branch_rate=1 / 70,
    taper=0.0015,
    asym_range=(0.45, 1.0),
    angle_jitter=0.25,
    max_branches=120,
    kappa_range=(0.04, 0.10),
    blur_range=(0.0, 1.5),
    reflex_prob=0.0,
    meander_amp=0.0,
    meander_wavelength=(25.0, 50.0),
) -> Network:
    """Grow n_trees vessel trees by persistent random walks with Murray branching.

    tortuosity          std of curvature (rad / px) of the AR(1) heading process
                        (large values coil vessels into knots)
    meander_amp         amplitude (rad) of a serpentine heading modulation -
                        the clinical kind of tortuosity, wiggling around a
                        mean course
    meander_wavelength  range (px) of the meander wavelength
    branch_rate         expected bifurcations per px of length
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    margin = 6
    branches: list[Branch] = []
    junctions: list[dict] = []
    styles: list[TreeStyle] = []

    for tree in range(n_trees):
        styles.append(TreeStyle(
            kappa=float(rng.uniform(*kappa_range)),
            blur=float(rng.uniform(*blur_range)),
            reflex=float(rng.uniform(0.3, 0.6)) if rng.random() < reflex_prob else 0.0,
        ))
        side = rng.integers(4)
        t = rng.uniform(0.15, 0.85)
        if side == 0:
            p0, hdg = (t * w, -2.0), np.pi / 2
        elif side == 1:
            p0, hdg = (w + 1.0, t * h), np.pi
        elif side == 2:
            p0, hdg = (t * w, h + 1.0), -np.pi / 2
        else:
            p0, hdg = (-2.0, t * h), 0.0
        hdg += rng.uniform(-0.5, 0.5)
        queue = [(np.array(p0, float), hdg, float(rng.uniform(*r_root)), None, 0.0)]
        n_tree_branches = 0
        while queue and n_tree_branches < max_branches:
            p, hdg, r, parent, kappa0 = queue.pop(0)
            pts, rad = [p.copy()], [r]
            kappa = kappa0
            base = hdg
            if meander_amp > 0:  # draw only when used, so seeds stay stable
                lam = rng.uniform(*meander_wavelength)
                phase0 = rng.uniform(0, 2 * np.pi)
            else:
                lam, phase0 = 1.0, 0.0
            wiggle0 = meander_amp * np.sin(phase0)
            min_len = max(12.0, 8.0 * r)
            L = 0.0
            exits = False
            split = False
            while True:
                kappa = persistence * kappa + np.sqrt(1 - persistence**2) * rng.normal(0, tortuosity)
                base += kappa
                hdg = base + meander_amp * np.sin(2 * np.pi * (L + 1) / lam + phase0) - wiggle0
                p = p + np.array([np.cos(hdg), np.sin(hdg)])
                r *= 1 - taper
                L += 1
                pts.append(p.copy())
                rad.append(r)
                if p[0] < -margin or p[0] > w + margin or p[1] < -margin or p[1] > h + margin:
                    exits = True
                    break
                if r < r_min or L > 3 * (h + w):
                    break
                if L > min_len and rng.random() < branch_rate:
                    split = True
                    break
            bid = len(branches)
            branches.append(Branch(np.array(pts), np.array(rad), tree, parent, [], exits))
            if parent is not None:
                branches[parent].children.append(bid)
            n_tree_branches += 1
            if split:
                (r1, a1), (r2, a2) = _murray_daughters(r, rng, asym_range)
                s = 1 if rng.random() < 0.5 else -1
                a1 += rng.normal(0, angle_jitter * a1 + 0.05)
                a2 += rng.normal(0, angle_jitter * a2 + 0.05)
                inside = 0 <= p[0] < w and 0 <= p[1] < h
                if r2 >= r_min * 0.9:
                    if inside:
                        junctions.append(dict(x=float(p[0]), y=float(p[1]), type="bifurcation",
                                              radius=float(r), tree=tree, branch=bid))
                    queue.append((p.copy(), hdg + s * a1, r1, bid, 0.0))
                    queue.append((p.copy(), hdg - s * a2, r2, bid, 0.0))
                else:  # daughter too small to see: just continue
                    queue.append((p.copy(), hdg, r1, bid, kappa))
            elif not exits:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    junctions.append(dict(x=float(p[0]), y=float(p[1]), type="endpoint",
                                          radius=float(r), tree=tree, branch=bid))

    net = Network(tuple(shape), branches, styles, junctions)
    net.junctions.extend(_find_crossovers(net))
    return net


def _segments(net: Network, step: int = 2):
    segs, owner, rads, idx = [], [], [], []
    for bi, b in enumerate(net.branches):
        P = b.pts[::step]
        if len(P) < 2 or not np.allclose(P[-1], b.pts[-1]):
            P = np.vstack([P, b.pts[-1:]])
        R = np.interp(np.linspace(0, len(b.radii) - 1, len(P)), np.arange(len(b.radii)), b.radii)
        for k in range(len(P) - 1):
            segs.append((P[k], P[k + 1]))
            owner.append(bi)
            rads.append(0.5 * (R[k] + R[k + 1]))
            idx.append(k)
    return np.array(segs), np.array(owner), np.array(rads), np.array(idx)


def _related(net: Network, a: int, b: int) -> np.ndarray | None:
    """Shared node of two branches (parent/child or siblings), else None."""
    A, B = net.branches[a], net.branches[b]
    if A.parent == b:
        return A.pts[0]
    if B.parent == a:
        return B.pts[0]
    if A.parent is not None and A.parent == B.parent:
        return A.pts[0]
    return None


def _find_crossovers(net: Network) -> list[dict]:
    segs, owner, rads, idx = _segments(net)
    if len(segs) == 0:
        return []
    h, w = net.shape
    cell = 16
    grid: dict[tuple[int, int], list[int]] = {}
    lo = np.minimum(segs[:, 0], segs[:, 1])
    hi = np.maximum(segs[:, 0], segs[:, 1])
    for s in range(len(segs)):
        for gx in range(int(lo[s, 0] // cell), int(hi[s, 0] // cell) + 1):
            for gy in range(int(lo[s, 1] // cell), int(hi[s, 1] // cell) + 1):
                grid.setdefault((gx, gy), []).append(s)
    found = []
    seen = set()
    for members in grid.values():
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                s, t = members[ii], members[jj]
                if (s, t) in seen:
                    continue
                seen.add((s, t))
                a, b = owner[s], owner[t]
                if a == b and abs(idx[s] - idx[t]) < 6:
                    continue
                p, p2 = segs[s]
                q, q2 = segs[t]
                r_, s_ = p2 - p, q2 - q
                den = r_[0] * s_[1] - r_[1] * s_[0]
                if abs(den) < 1e-12:
                    continue
                qp = q - p
                tt = (qp[0] * s_[1] - qp[1] * s_[0]) / den
                uu = (qp[0] * r_[1] - qp[1] * r_[0]) / den
                if not (0 <= tt < 1 and 0 <= uu < 1):
                    continue
                X = p + tt * r_
                shared = _related(net, a, b) if a != b else None
                if shared is not None and np.hypot(*(X - shared)) < 3 * max(rads[s], rads[t]) + 3:
                    continue
                if not (0 <= X[0] < w and 0 <= X[1] < h):
                    continue
                ang = np.rad2deg(np.arccos(abs(np.dot(r_, s_)) / (np.hypot(*r_) * np.hypot(*s_) + 1e-12)))
                found.append(dict(x=float(X[0]), y=float(X[1]), type="crossover",
                                  radius=float(max(rads[s], rads[t])),
                                  radii=(float(rads[s]), float(rads[t])),
                                  angle=float(ang), branches=(int(a), int(b)),
                                  pos=(float(2 * (idx[s] + tt)), float(2 * (idx[t] + uu)))))
    # de-duplicate (a crossing can be hit by several segment pairs)
    out = []
    for f in found:
        if all(np.hypot(f["x"] - o["x"], f["y"] - o["y"]) > 2.0 for o in out):
            out.append(f)
    return out


# --------------------------------------------------------------- rendering
def _branch_od(b: Branch, shape, kappa: float, reflex: float):
    """Pixels and OD of one branch (cylinder chord profile, 2x2 supersampled)."""
    h, w = shape
    P, R = b.pts, b.radii
    rmax = float(R.max())
    pad = int(np.ceil(rmax + 2))
    x0 = int(max(0, np.floor(P[:, 0].min()) - pad))
    x1 = int(min(w, np.ceil(P[:, 0].max()) + pad + 1))
    y0 = int(max(0, np.floor(P[:, 1].min()) - pad))
    y1 = int(min(h, np.ceil(P[:, 1].max()) + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    bh, bw = y1 - y0, x1 - x0
    # rasterize polyline densely, remember segment index per pixel
    t = np.linspace(0, len(P) - 1, max(2, 4 * len(P)))
    seg_of = np.minimum(np.floor(t).astype(int), len(P) - 2)
    xs = np.interp(t, np.arange(len(P)), P[:, 0]) - x0
    ys = np.interp(t, np.arange(len(P)), P[:, 1]) - y0
    ok = (xs >= 0) & (xs < bw) & (ys >= 0) & (ys < bh)
    if not ok.any():
        return None
    seed = np.ones((bh, bw), dtype=bool)
    segmap = np.zeros((bh, bw), dtype=np.int64)
    xi, yi = np.round(xs[ok]).astype(int).clip(0, bw - 1), np.round(ys[ok]).astype(int).clip(0, bh - 1)
    seed[yi, xi] = False
    segmap[yi, xi] = seg_of[ok]
    dist, (iy, ix) = ndi.distance_transform_edt(seed, return_indices=True)
    cand = dist <= rmax + 1.5
    cy, cx = np.nonzero(cand)
    near = segmap[iy[cy, cx], ix[cy, cx]]
    od = np.zeros(len(cy))
    for oy, ox in ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)):
        px = cx + x0 + ox
        py = cy + y0 + oy
        best_d = np.full(len(cy), np.inf)
        best_r = np.zeros(len(cy))
        for dk in (-2, -1, 0, 1, 2):
            k = np.clip(near + dk, 0, len(P) - 2)
            a, bb = P[k], P[k + 1]
            ab = bb - a
            L2 = np.maximum((ab**2).sum(1), 1e-12)
            tt = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / L2, 0, 1)
            d = np.hypot(px - (a[:, 0] + tt * ab[:, 0]), py - (a[:, 1] + tt * ab[:, 1]))
            rr = R[k] + tt * (R[k + 1] - R[k])
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_r = np.where(better, rr, best_r)
        chord = 2 * np.sqrt(np.clip(best_r**2 - best_d**2, 0, None))
        if reflex > 0:
            chord *= 1 - reflex * np.exp(-0.5 * (best_d / (0.3 * best_r)) ** 2)
        od += 0.25 * kappa * chord
    keep = od > 0
    return cy[keep] + y0, cx[keep] + x0, od[keep]


def _annotate_visibility(net: Network, layers, noise: float, snr: float = 3.0):
    """Measure each branch's rendered contrast and flag junctions whose
    participating vessels are below `snr` x noise as not visible."""
    h, w = net.shape
    for b in net.branches:
        xi = np.round(b.pts[:, 0]).astype(int)
        yi = np.round(b.pts[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        b.contrast = float(np.median(layers[b.tree][yi[ok], xi[ok]])) if ok.any() else 0.0
    floor = snr * max(noise, 1e-6)
    for j in net.junctions:
        if j["type"] == "bifurcation":
            ids = [j["branch"]] + net.branches[j["branch"]].children
        elif j["type"] == "crossover":
            ids = list(j["branches"])
        else:
            ids = [j["branch"]]
        j["contrast"] = float(min(net.branches[i].contrast for i in ids))
        j["visible"] = bool(j["contrast"] >= floor)
        if j["type"] == "crossover" and j["visible"]:
            # A crossing is only recognizable as an X if both vessels visibly
            # continue on both sides of it, clear of the other vessel, and if
            # they actually cross rather than run side by side. Otherwise it
            # reads as a T, a Y or an end - for a human annotator too - so it
            # is scored as "don't care".
            (ba, bb), (pa, pb) = j["branches"], j["pos"]
            ra = float(np.interp(pa, np.arange(len(net.branches[ba].radii)), net.branches[ba].radii))
            rb = float(np.interp(pb, np.arange(len(net.branches[bb].radii)), net.branches[bb].radii))
            for bi, pos, r_self, r_other in ((ba, pa, ra, rb), (bb, pb, rb, ra)):
                c_local, stub = _continuation(net, layers, bi, pos)
                if c_local < floor:
                    j["visible"], j["ambiguous"] = False, "faint"
                elif stub < max(8.0, 3.0 * r_self, r_other + 6.0):
                    j["visible"], j["ambiguous"] = False, "stub"
            if j["visible"] and _overlap_length(net, ba, pa, bb) > 5.0 * (ra + rb + 1.0):
                j["visible"], j["ambiguous"] = False, "tangential"


def _overlap_length(net: Network, ba: int, pa: float, bb: int) -> float:
    """Length of branch ba, around arc position pa, lying within the lumen
    of branch bb (centerline distance < r_a + r_b + 1)."""
    A, B = net.branches[ba], net.branches[bb]
    k = int(round(pa))
    lo, hi = max(0, k - 60), min(len(A.pts), k + 61)
    P = A.pts[lo:hi]
    D = np.hypot(P[:, None, 0] - B.pts[None, :, 0], P[:, None, 1] - B.pts[None, :, 1])
    j = D.argmin(1)
    close = D[np.arange(len(P)), j] < A.radii[lo:hi] + B.radii[j] + 1.0
    c = k - lo
    if not (0 <= c < len(close)) or not close[c]:
        return 0.0
    i0 = c
    while i0 > 0 and close[i0 - 1]:
        i0 -= 1
    i1 = c
    while i1 < len(close) - 1 and close[i1 + 1]:
        i1 += 1
    return float(i1 - i0 + 1)


def _continuation(net: Network, layers, bi: int, pos: float):
    """(min local contrast beside the crossing, shortest in-view continuation)."""
    h, w = net.shape
    b = net.branches[bi]
    k = int(round(pos))
    k = min(max(k, 0), len(b.pts) - 1)
    inside = (b.pts[:, 0] >= 0) & (b.pts[:, 0] < w) & (b.pts[:, 1] >= 0) & (b.pts[:, 1] < h)
    big = 1e9
    # backward
    back = 0
    i = k
    while i > 0 and inside[i - 1]:
        back += 1
        i -= 1
    if i == 0 and b.parent is not None and inside[0]:
        back = big
    fwd = 0
    i = k
    while i < len(b.pts) - 1 and inside[i + 1]:
        fwd += 1
        i += 1
    if i == len(b.pts) - 1 and b.children and inside[-1]:
        fwd = big
    r = float(b.radii[k])
    off = int(round(2 * r + 3))
    vals = []
    for q in (k - off, k + off):
        if 0 <= q < len(b.pts) and inside[q]:
            x, y = int(round(b.pts[q, 0])), int(round(b.pts[q, 1]))
            if 0 <= x < w and 0 <= y < h:
                vals.append(layers[b.tree][y, x])
    return (float(min(vals)) if vals else 0.0), float(min(back, fwd))


def _smooth_field(shape, sigma, rng):
    f = ndi.gaussian_filter(rng.normal(size=shape), sigma, mode="wrap")
    return f / (np.abs(f).max() + 1e-12)


def render(
    net: Network,
    seed=0,
    background=0.75,
    illumination=0.25,
    texture=0.04,
    texture_scale=4.0,
    noise=0.015,
    flow: float = 0.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Render a transmission image in [0, 1].

    flow/phase: optional travelling OD modulation along vessels (to mimic blood
    flow in videos); flow is the relative modulation depth.
    """
    rng = np.random.default_rng(seed)
    h, w = net.shape
    T = np.ones(net.shape)
    layers = []
    for tree, style in enumerate(net.styles):
        od = np.zeros(net.shape)
        for b in net.branches:
            if b.tree != tree:
                continue
            res = _branch_od(b, net.shape, style.kappa, style.reflex)
            if res is None:
                continue
            yy, xx, v = res
            if flow > 0:
                v = v * (1 + flow * np.sin(0.35 * (xx + yy) - phase + 1.7 * tree))
            np.maximum.at(od, (yy, xx), v)
        trans = np.exp(-od)
        if style.blur > 0.05:
            trans = ndi.gaussian_filter(trans, style.blur)
        T *= trans
        layers.append(-np.log(np.maximum(trans, 1e-9)))
    _annotate_visibility(net, layers, noise)
    yy, xx = np.mgrid[:h, :w]
    vign = 1 - 0.35 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    I0 = background * vign * (1 + illumination * _smooth_field(net.shape, 0.2 * max(h, w), rng))
    I0 *= 1 + texture * _smooth_field(net.shape, texture_scale, rng)
    img = I0 * T
    img = img + rng.normal(size=img.shape) * noise * np.sqrt(np.maximum(img, 1e-6) / background) * background
    return np.clip(img, 0, 1).astype(np.float32)


def centerline_truth(net: Network, min_radius: float = 0.0):
    """(mask, radius map) of ground-truth centerlines inside the image."""
    h, w = net.shape
    mask = np.zeros(net.shape, dtype=bool)
    rad = np.zeros(net.shape, dtype=np.float32)
    for b in net.branches:
        t = np.linspace(0, len(b.pts) - 1, 2 * len(b.pts))
        xs = np.interp(t, np.arange(len(b.pts)), b.pts[:, 0])
        ys = np.interp(t, np.arange(len(b.pts)), b.pts[:, 1])
        rs = np.interp(t, np.arange(len(b.pts)), b.radii)
        xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (rs >= min_radius)
        mask[yi[ok], xi[ok]] = True
        rad[yi[ok], xi[ok]] = np.maximum(rad[yi[ok], xi[ok]], rs[ok])
    return mask, rad


def make_sample(seed=0, shape=(512, 512), grow_kw=None, render_kw=None):
    net = grow_network(shape=shape, seed=seed, **(grow_kw or {}))
    img = render(net, seed=seed + 1000, **(render_kw or {}))
    return img, net
