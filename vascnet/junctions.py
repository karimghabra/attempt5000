"""Stage 4 - junction analysis: reading the 3-D structure out of a 2-D junction.

A junction in a projection image is ambiguous: four arms meeting at a point can
be one vessel passing over another (a crossover: two vessels at different
depths) or a vessel splitting twice. The visual system resolves this with a
small number of cues, all of which are available without learning:

1. Good continuation  - arms of the same vessel continue straight through
                        (Gestalt; the association field already encodes it).
2. Similarity         - arms of one vessel share width and contrast;
                        at a true bifurcation radii obey Murray's law
                        r_p^3 ~ r_1^3 + r_2^3 instead.
3. Transparency       - in an absorbing medium optical densities ADD where
                        two vessels overlap, so a crossover's centre is darker
                        than either vessel; a bifurcation's centre is not.
                        (The perceptual analogue is Metelli transparency.)
4. Amodal completion  - if a straight pair exists, look for the missing fourth
                        arm on the far side before accepting a T-bifurcation.

Each junction gets a type, a subtype, a confidence in [0, 1] (to triage what a
human annotator should check), its arms (direction, width, contrast) and the
arm pairing that explains it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
from scipy import ndimage as ndi

from .config import Config
from .cortex import OrientationScore
from .graph import VesselGraph

TYPES = ("bifurcation", "crossover", "complex", "contact", "endpoint")


@dataclass
class Arm:
    angle: float          # direction from the junction outward, atan2(dy, dx), rad
    width: float          # vessel radius (px)
    contrast: float       # optical density on the centerline
    strength: float       # orientation-score evidence (noise units)
    length: float         # length of the graph edge (px)
    edge: int | None = None
    virtual: bool = False  # inferred by amodal completion, not on the skeleton
    anchor: tuple[float, float] | None = None  # (y, x) mean of the arm samples
    axis: tuple[float, float] | None = None    # (dy, dx) local vessel direction


@dataclass
class Junction:
    id: int
    x: float
    y: float
    type: str
    subtype: str
    confidence: float
    radius: float
    arms: list[Arm] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    node: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arms"] = [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()}
            for a in d["arms"]
        ]
        d["features"] = {k: (round(float(v), 4) if np.isscalar(v) else v)
                         for k, v in self.features.items()}
        for k in ("x", "y", "confidence", "radius"):
            d[k] = round(float(d[k]), 3)
        return d


# ---------------------------------------------------------------- geometry
def angdiff(a: float, b: float) -> float:
    """Unsigned difference between two directions, in [0, pi]."""
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def wrap_angle_half_pi(a: float) -> float:
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def straight_dev(a: Arm, b: Arm) -> float:
    """0 when two arms point in exactly opposite directions."""
    return np.pi - angdiff(a.angle, b.angle)


def axis_angle(a: Arm, b: Arm) -> float:
    """Orientation (mod pi) of the line through a straight pair."""
    v = np.array([np.cos(a.angle), np.sin(a.angle)]) - np.array([np.cos(b.angle), np.sin(b.angle)])
    return float(np.arctan2(v[1], v[0]) % np.pi)


def pair_cost(a: Arm, b: Arm, cfg: Config) -> float:
    dev = straight_dev(a, b) / np.deg2rad(cfg.straight_tol)
    wr = np.log(max(a.width, 0.5) / max(b.width, 0.5)) / cfg.pair_width_tol
    ca, cb = max(a.contrast, 1e-4), max(b.contrast, 1e-4)
    cr = np.log(ca / cb) / cfg.pair_contrast_tol
    if a.virtual or b.virtual:  # nothing reliable to compare
        wr = cr = 0.0
    return float(dev**2 + 0.5 * wr**2 + 0.5 * cr**2)


def _sample(img: np.ndarray, ys, xs) -> np.ndarray:
    return ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")


# ---------------------------------------------------------------- arms
class ArmReader:
    def __init__(self, score: OrientationScore, od_smooth: np.ndarray, radius: np.ndarray):
        self.U = score.U if score.U is not None else score.z
        self.strength = self.U.max(0)
        self.thetas = score.thetas
        self.od = od_smooth
        self.radius = radius

    def node_geometry(self, g: VesselGraph, nid: int, inc) -> tuple[np.ndarray, float, float]:
        node = g.nodes[nid]
        c = node.yx
        core = 0.0
        if len(node.pixels):
            core = float(np.max(np.hypot(*(node.pixels - c).T)))
        near = [p[: max(2, min(len(p), 6))] for _, p in inc if len(p)]
        pix = np.concatenate([node.pixels] + near) if near else node.pixels
        r = float(np.median(self.radius[pix[:, 0], pix[:, 1]])) if len(pix) else 1.0
        return c, core, r

    def arm(self, c: np.ndarray, path: np.ndarray, r_in: float, r_out: float,
            eid: int | None) -> Arm | None:
        if len(path) == 0:
            return None
        d = np.hypot(path[:, 0] - c[0], path[:, 1] - c[1])
        sel = (d >= r_in) & (d <= r_out)
        if not sel.any():
            sel = d >= 0.5 * r_in
            if not sel.any():
                sel = np.zeros(len(path), bool)
                sel[-1] = True
        pts = path[sel].astype(float)
        anchor = pts.mean(0)
        v = anchor - c
        if np.hypot(*v) < 1e-6:
            return None
        angle = float(np.arctan2(v[0], v[1]))
        if len(pts) >= 3:
            _, _, vt = np.linalg.svd(pts - anchor, full_matrices=False)
            axis = vt[0] if vt[0] @ v >= 0 else -vt[0]
        else:
            axis = v / np.hypot(*v)
        far = (d >= r_in) & (d <= r_in + max(10.0, 4.0 * (r_out - r_in)))
        if not far.any():
            far = sel
        fp = path[far]
        width = float(np.median(self.radius[fp[:, 0], fp[:, 1]]))
        contrast = float(np.median(self.od[fp[:, 0], fp[:, 1]]))
        strength = float(np.median(self.strength[fp[:, 0], fp[:, 1]]))
        return Arm(angle, width, contrast, strength, float(len(path)), eid,
                   anchor=(float(anchor[0]), float(anchor[1])),
                   axis=(float(axis[0]), float(axis[1])))

    def ray_evidence(self, c: np.ndarray, angle: float, r_in: float, r_out: float,
                     cone_deg: float = 0.0, step_deg: float = 5.0) -> float:
        """Line evidence along a ray (or the best ray in a cone), measured in
        the ray's own orientation channel so that vessels merely crossing the
        ray do not count. Returns the 60th percentile along the best ray."""
        t = np.arange(r_in, r_out + 1e-6, 1.0)
        h, w = self.strength.shape
        N = len(self.thetas)
        best = 0.0
        offsets = np.deg2rad(np.arange(-cone_deg, cone_deg + 1e-6, step_deg)) if cone_deg else [0.0]
        for off in offsets:
            a = angle + off
            ys = c[0] + t * np.sin(a)
            xs = c[1] + t * np.cos(a)
            ok = (ys >= 0) & (ys <= h - 1) & (xs >= 0) & (xs <= w - 1)
            if ok.sum() < 2:
                continue
            k = int(round((a % np.pi) / (np.pi / N))) % N
            vals = np.max([_sample(self.U[(k + j) % N], ys[ok], xs[ok]) for j in (-1, 0, 1)], axis=0)
            best = max(best, float(np.percentile(vals, 60)))
        return best

    def center_od(self, c: np.ndarray, r: float) -> float:
        R = max(1.5, r)
        y0, x0 = c
        h, w = self.od.shape
        ylo, yhi = int(max(0, np.floor(y0 - R))), int(min(h, np.ceil(y0 + R) + 1))
        xlo, xhi = int(max(0, np.floor(x0 - R))), int(min(w, np.ceil(x0 + R) + 1))
        win = self.od[ylo:yhi, xlo:xhi]
        yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
        m = np.hypot(yy - y0, xx - x0) <= R
        return float(win[m].max()) if m.any() else float(self.od[int(y0), int(x0)])


# ---------------------------------------------------------- classification
def _best_pairing4(arms: list[Arm], cfg: Config):
    """Pairing of four arms into two vessels with the straightest worst pair
    (ties broken by the full same-vessel cost)."""
    best = None
    for p in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        costs = [pair_cost(arms[i], arms[j], cfg) for i, j in p]
        devs = [straight_dev(arms[i], arms[j]) for i, j in p]
        cand = (max(devs) + 0.05 * sum(costs), p, costs, devs)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


def _crossover_verdict(arms, pairs, costs, center_od, cfg):
    (i, j), (k, l) = pairs
    ax1 = axis_angle(arms[i], arms[j])
    ax2 = axis_angle(arms[k], arms[l])
    cross = float(np.rad2deg(abs(((ax1 - ax2) + np.pi / 2) % np.pi - np.pi / 2)))
    od1 = np.mean([arms[i].contrast, arms[j].contrast])
    od2 = np.mean([arms[k].contrast, arms[l].contrast])
    additivity = center_od / max(od1 + od2, 1e-6)
    geom = float(np.exp(-0.25 * np.mean(costs)))
    s_add = float(np.exp(-0.5 * ((min(additivity, 1.3) - 1.0) / 0.35) ** 2))
    conf = 0.75 * geom + 0.25 * s_add
    feats = dict(crossing_angle=cross, additivity=additivity, geometry_score=geom)
    return cross, conf, feats


def classify(arms: list[Arm], center_od: float, reader: ArmReader, c, r_in, r_out,
             cfg: Config) -> tuple[str, str, float, list, dict]:
    n = len(arms)
    tol = np.deg2rad(cfg.straight_tol)
    evid = float(np.clip(np.median([a.strength for a in arms]) / (2 * cfg.t_high), 0, 1))

    if n == 1:
        return "endpoint", "terminal", evid, [], {}

    if n == 2:  # should have been dissolved; report as a bend
        return "complex", "bend", 0.2 * evid, [], {}

    if n == 3:
        cand = sorted(
            ((pair_cost(arms[i], arms[j], cfg), (i, j)) for i, j in combinations(range(3), 2)),
            key=lambda t: t[0],
        )
        cost, (i, j) = cand[0]
        k = ({0, 1, 2} - {i, j}).pop()
        # T: one vessel runs straight through and the third arm leaves it at a
        # large angle (a side branch - or half of a crossing vessel). A Y whose
        # daughter happens to continue the parent's line within a few tens of
        # degrees is not a T.
        thru = axis_angle(arms[i], arms[j])
        side = abs(wrap_angle_half_pi(arms[k].angle - thru))
        straight = (straight_dev(arms[i], arms[j]) <= np.deg2rad(cfg.t_straight_max)
                    and side >= np.deg2rad(cfg.t_side_min))
        if straight:
            # amodal completion: is the third arm's vessel continuing on the far side?
            opp = (arms[k].angle + np.pi) % (2 * np.pi)
            if opp > np.pi:
                opp -= 2 * np.pi
            # look beyond the through vessel, where the far half would emerge
            w_thru = 0.5 * (arms[i].width + arms[j].width)
            start = max(r_in, w_thru + 2.0)
            ev = reader.ray_evidence(c, opp, start, start + max(6.0, 3.0 * arms[k].width),
                                     cone_deg=cfg.completion_cone)
            # the hidden half must look like the visible half of the same vessel
            if ev >= max(cfg.t_high, cfg.completion_ratio * arms[k].strength):
                virt = Arm(float(opp), arms[k].width, arms[k].contrast, ev, 0.0, None, True)
                arms.append(virt)
                pairs = ((i, j), (k, 3))
                costs = [pair_cost(arms[i], arms[j], cfg), pair_cost(arms[k], virt, cfg)]
                cross, conf, feats = _crossover_verdict(arms, pairs, costs, center_od, cfg)
                feats["completion_evidence"] = ev
                if cross < cfg.min_cross_angle:
                    return "contact", "tangential", conf * evid, list(pairs), feats
                return "crossover", "incomplete", 0.8 * conf * evid, list(pairs), feats
        # bifurcation: the parent sits opposite the two daughters, i.e. it has
        # the largest angular gap to its neighbours; width (Murray: the parent
        # is the widest) breaks near-ties.
        gaps = [min(angdiff(arms[q].angle, arms[o].angle) for o in range(3) if o != q)
                for q in range(3)]
        wmax = max(a.width for a in arms)
        score = [gaps[q] / np.pi + 0.5 * np.log(arms[q].width / wmax) for q in range(3)]
        p = int(np.argmax(score))
        d1, d2 = [q for q in range(3) if q != p]
        rp, r1, r2 = arms[p].width, arms[d1].width, arms[d2].width
        murray = (r1**3 + r2**3) / max(rp**3, 1e-6)
        branch_angle = float(np.rad2deg(angdiff(arms[d1].angle, arms[d2].angle)))
        sub = "T" if straight else "Y"
        # confidence: strong arms, and not crossover-like (no very straight pair)
        dev_best = straight_dev(arms[i], arms[j])
        margin = 1.0 if not straight else 0.6 + 0.4 * min(1.0, dev_best / tol)
        additivity = center_od / max(arms[p].contrast, 1e-6)
        feats = dict(murray_ratio=murray, branch_angle=branch_angle, parent=p,
                     additivity=additivity)
        return "bifurcation", sub, evid * margin, [], feats

    if n == 4:
        total, pairs, costs, devs = _best_pairing4(arms, cfg)
        if _is_crossing4(arms, cfg):
            cross, conf, feats = _crossover_verdict(arms, pairs, costs, center_od, cfg)
            if cross < cfg.min_cross_angle:
                return "contact", "tangential", conf * evid, list(pairs), feats
            return "crossover", "X", conf * evid, list(pairs), feats
        return "complex", "trifurcation", 0.6 * evid, [], {"best_pair_cost": total}

    # n >= 5: peel off straight-through vessels greedily
    remaining = list(range(n))
    pairs = []
    while len(remaining) >= 2:
        best = None
        for i, j in combinations(remaining, 2):
            if straight_dev(arms[i], arms[j]) <= tol:
                cst = pair_cost(arms[i], arms[j], cfg)
                if best is None or cst < best[0]:
                    best = (cst, (i, j))
        if best is None:
            break
        pairs.append(best[1])
        remaining = [a for a in remaining if a not in best[1]]
    left = len(remaining)
    if pairs and left == 0:
        return "crossover", f"{len(pairs)}-way", 0.7 * evid, pairs, {}
    if pairs and left == 3:
        return "complex", "crossover+bifurcation", 0.6 * evid, pairs, {}
    return "complex", f"{len(pairs)} through + {left} arms", 0.4 * evid, pairs, {}


def _arm_window(core: float, r: float) -> tuple[float, float]:
    r_in = core + 0.5 * r + 1.0
    return r_in, r_in + max(5.0, 2.5 * r)


def _node_arms(g: VesselGraph, reader: ArmReader, nid: int, exclude: int | None = None):
    inc = [(eid, p) for eid, p in g.incident(nid) if eid != exclude]
    c, core, r = reader.node_geometry(g, nid, inc)
    r_in, r_out = _arm_window(core, r)
    arms = [a for a in (reader.arm(c, p, r_in, r_out, eid) for eid, p in inc) if a is not None]
    return arms, c, core, r, r_in, r_out


def _is_crossing4(arms: list[Arm], cfg: Config, across: bool = False) -> bool:
    """Two straight-through pairs, clearly better than the alternatives.
    With ``across``, each pair must take one arm from arms[:2] and one from
    arms[2:] (used when fusing two nodes)."""
    _, pairs, _, devs = _best_pairing4(arms, cfg)
    alt = sorted(max(straight_dev(arms[a], arms[b]) for a, b in p)
                 for p in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))))
    ok = max(devs) <= np.deg2rad(cfg.cross_dev_max) and alt[1] - max(devs) >= np.deg2rad(cfg.cross_margin)
    if across:
        ok &= all((i < 2) != (j < 2) for i, j in pairs)
    return bool(ok)


def fuse_crossings(g: VesselGraph, reader: ArmReader, radius: np.ndarray, cfg: Config) -> int:
    """Fuse pairs of 3-way junctions that are one crossover seen as two T/Y's.

    A thin vessel crossing a thick one obliquely meets the thick centerline at
    two points a few radii apart. The pair is fused only on good continuation:
    seen from the midpoint, the four outer arms form two straight-through
    pairs, each taking one arm from each node - both vessels continue across
    the short shared segment. Two genuine bifurcations in a row fail this.
    """
    from .graph import _local_radius, fuse_nodes

    n_fused = 0
    changed = True
    while changed:
        changed = False
        cands = []
        for e in g.edges.values():
            if e.u == e.v or g.nodes[e.u].kind != "junction" or g.nodes[e.v].kind != "junction":
                continue
            if g.degree(e.u) != 3 or g.degree(e.v) != 3:
                continue
            r = max(_local_radius(radius, e.path), 1.0)
            if len(e) <= max(cfg.merge_min, cfg.crossing_merge_factor * max(cfg.radii)):
                cands.append((len(e) / r, e.id))
        for _, eid in sorted(cands):
            e = g.edges.get(eid)
            if e is None or e.u not in g.nodes or e.v not in g.nodes:
                continue
            if g.degree(e.u) != 3 or g.degree(e.v) != 3:
                continue
            A, ca, *_ = _node_arms(g, reader, e.u, eid)
            B, cb, *_ = _node_arms(g, reader, e.v, eid)
            if len(A) != 2 or len(B) != 2:
                continue
            r_thick = max(a.width for a in A + B)
            if len(e) > max(cfg.merge_min, cfg.crossing_merge_factor * r_thick):
                continue
            mid = 0.5 * (ca + cb)
            arms = A + B
            _reaim(arms, mid)
            if _is_crossing4(arms, cfg, across=True):
                fuse_nodes(g, [e.u, e.v], [eid], center=(mid[0], mid[1]))
                n_fused += 1
                changed = True
    return n_fused


def refine_center(c: np.ndarray, arms: list[Arm], core: float, r: float) -> np.ndarray:
    """Least-squares meeting point of the arms' axes.

    The skeleton puts a junction where the merged ridges split, which for
    acute branchings is several px downstream of where the vessels actually
    meet. Extrapolating each arm's local axis back and intersecting them
    recovers the true point; a weak prior toward the skeleton node keeps
    near-parallel configurations well-posed.
    """
    A = np.zeros((2, 2))
    b = np.zeros(2)
    used = 0
    for a in arms:
        if a.virtual or a.anchor is None:
            continue
        d = np.asarray(a.axis)
        P = np.eye(2) - np.outer(d, d)
        w = max(a.strength, 1e-3) ** 0.5
        A += w * P
        b += w * P @ np.asarray(a.anchor)
        used += 1
    if used < 2:
        return c
    lam = 0.05 * np.trace(A)
    x = np.linalg.solve(A + lam * np.eye(2), b + lam * c)
    if np.hypot(*(x - c)) > core + 2.0 * r + 2.0:
        return c
    return x


def _reaim(arms: list[Arm], c: np.ndarray):
    for a in arms:
        if a.anchor is not None and not a.virtual:
            a.angle = float(np.arctan2(a.anchor[0] - c[0], a.anchor[1] - c[1]))


def analyze(g: VesselGraph, score: OrientationScore, od_smooth: np.ndarray,
            radius: np.ndarray, cfg: Config, border: int = 4) -> list[Junction]:
    reader = ArmReader(score, od_smooth, radius)
    fuse_crossings(g, reader, radius, cfg)
    h, w = od_smooth.shape
    out: list[Junction] = []
    for nid, node in g.nodes.items():
        if not g.incident(nid):
            continue
        if node.kind == "end" and (
            node.y < border or node.x < border or node.y >= h - border or node.x >= w - border
        ):
            continue  # vessel leaves the field of view
        arms, c, core, r, r_in, r_out = _node_arms(g, reader, nid)
        if not arms:
            continue
        if len(arms) >= 3:
            c = refine_center(c, arms, core, r)
            _reaim(arms, c)
        cod = reader.center_od(c, r)
        typ, sub, conf, pairs, feats = classify(arms, cod, reader, c, r_in, r_out, cfg)
        feats["center_od"] = cod
        jr = max(a.width for a in arms)
        out.append(Junction(len(out), float(c[1]), float(c[0]), typ, sub,
                            float(np.clip(conf, 0, 1)), jr, arms, [tuple(p) for p in pairs],
                            feats, nid))
    return out
