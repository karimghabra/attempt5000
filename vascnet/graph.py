"""Skeleton -> vessel graph, with the topological clean-up a human annotator
does implicitly:

* bridge free ends to a nearby centerline when they point at it (closes the
  1-3 px gap that NMS leaves where a side branch meets its parent, which would
  otherwise turn a bifurcation into a dangling end),
* prune short spurs created by skeletonization,
* dissolve degree-2 "junctions",
* fuse junctions that are closer than a few vessel radii into one compound
  node (a crossover usually skeletonizes as two Y's joined by a short segment).

Nodes keep their pixels; edges keep an ordered pixel path from u to v that
excludes node pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

from .config import Config

RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
ORTH = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIAG = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Node:
    id: int
    y: float
    x: float
    kind: str                       # 'junction' | 'end'
    pixels: np.ndarray              # (P, 2) int (y, x)
    members: list[int] = field(default_factory=list)

    @property
    def yx(self) -> np.ndarray:
        return np.array([self.y, self.x])


@dataclass
class Edge:
    id: int
    u: int
    v: int
    path: np.ndarray                # (K, 2) int (y, x), ordered u -> v

    def __len__(self) -> int:
        return len(self.path)


class VesselGraph:
    def __init__(self, shape):
        self.shape = shape
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self._inc: dict[int, list[int]] = {}   # node -> edge ids (loops twice)
        self._nid = 0
        self._eid = 0

    # ------------------------------------------------------------ mutation
    def add_node(self, y, x, kind, pixels, members=None) -> int:
        nid = self._nid
        self._nid += 1
        self.nodes[nid] = Node(nid, float(y), float(x), kind,
                               np.asarray(pixels, dtype=int).reshape(-1, 2),
                               list(members) if members else [nid])
        self._inc[nid] = []
        return nid

    def add_edge(self, u, v, path) -> int:
        eid = self._eid
        self._eid += 1
        self.edges[eid] = Edge(eid, u, v, np.asarray(path, dtype=int).reshape(-1, 2))
        self._inc[u].append(eid)
        self._inc[v].append(eid)
        return eid

    def remove_edge(self, eid: int):
        e = self.edges.pop(eid)
        self._inc[e.u].remove(eid)
        self._inc[e.v].remove(eid)

    def remove_node(self, nid: int):
        for eid in list(set(self._inc[nid])):
            if eid in self.edges:
                self.remove_edge(eid)
        del self.nodes[nid]
        del self._inc[nid]

    def relink(self, eid: int, u: int, v: int):
        e = self.edges[eid]
        self._inc[e.u].remove(eid)
        self._inc[e.v].remove(eid)
        e.u, e.v = u, v
        self._inc[u].append(eid)
        self._inc[v].append(eid)

    # ------------------------------------------------------------- queries
    def incident(self, nid: int) -> list[tuple[int, np.ndarray]]:
        """(edge id, path oriented away from nid) for every edge-end at nid."""
        out = []
        for eid in dict.fromkeys(self._inc[nid]):
            e = self.edges[eid]
            if e.u == nid:
                out.append((e.id, e.path))
            if e.v == nid:
                out.append((e.id, e.path[::-1]))
        return out

    def degree(self, nid: int) -> int:
        return len(self._inc[nid])

    def other(self, e: Edge, nid: int) -> int:
        return e.v if e.u == nid else e.u

    def junctions(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "junction"]

    def endpoints(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "end"]

    def to_mask(self) -> np.ndarray:
        m = np.zeros(self.shape, dtype=bool)
        for n in self.nodes.values():
            if len(n.pixels):
                m[n.pixels[:, 0], n.pixels[:, 1]] = True
        for e in self.edges.values():
            if len(e.path):
                m[e.path[:, 0], e.path[:, 1]] = True
        return m


# ------------------------------------------------------------------ build
def _ring_stack(mask: np.ndarray):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return [p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in RING]


def _walk(pixset: set, start) -> list:
    path = [start]
    seen = {start}
    cur = start
    while True:
        nxt = None
        for dy, dx in ORTH + DIAG:
            q = (cur[0] + dy, cur[1] + dx)
            if q in pixset and q not in seen:
                nxt = q
                break
        if nxt is None:
            return path
        path.append(nxt)
        seen.add(nxt)
        cur = nxt


def build_graph(skel: np.ndarray) -> VesselGraph:
    skel = skel.astype(bool)
    h, w = skel.shape
    # Any pixel with >= 3 skeleton neighbours is a junction pixel. This also
    # flags staircase corners and 2-px knots that thinning leaves behind, but
    # those come out as degree-2 nodes and are dissolved in prune(); in return
    # every remaining segment is guaranteed to be a simple path.
    nb = sum(r.astype(np.int8) for r in _ring_stack(skel))
    junc = skel & (nb >= 3)
    jlab, nj = ndi.label(junc, structure=EIGHT)

    g = VesselGraph(skel.shape)
    jnode = {}
    if nj:
        for lbl, sl in enumerate(ndi.find_objects(jlab), start=1):
            ys, xs = np.nonzero(jlab[sl] == lbl)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            jnode[lbl] = g.add_node(ys.mean(), xs.mean(), "junction", np.c_[ys, xs])

    seg = skel & ~junc
    slab, ns = ndi.label(seg, structure=EIGHT)
    sring = _ring_stack(seg)
    nb_seg = sum(r.astype(np.int8) for r in sring)
    jpad = np.pad(jlab, 1)

    def adjacent_junction(p):
        y, x = p
        for dy, dx in RING:
            lbl = jpad[y + 1 + dy, x + 1 + dx]
            if lbl:
                return jnode[lbl]
        return None

    for lbl, sl in enumerate(ndi.find_objects(slab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(slab[sl] == lbl)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pix = list(zip(ys.tolist(), xs.tolist()))
        terms = [p for p in pix if nb_seg[p] <= 1]
        if not terms:
            # closed loop: attach to a junction if it touches one, else drop
            touching = [p for p in pix if adjacent_junction(p) is not None]
            if not touching:
                continue
            terms = [touching[0]]
        path = _walk(set(pix), terms[0])
        a, b = path[0], path[-1]
        u = adjacent_junction(a)
        if u is None:
            u = g.add_node(a[0], a[1], "end", [a])
        v = adjacent_junction(b) if len(path) > 1 else None
        if v is None:
            if len(path) == 1:
                # single-pixel segment: may touch two junction clusters
                y, x = a
                lbls = {jpad[y + 1 + dy, x + 1 + dx] for dy, dx in RING} - {0}
                others = [jnode[l] for l in lbls if jnode[l] != u]
                v = others[0] if others else g.add_node(b[0], b[1], "end", [b])
            else:
                v = g.add_node(b[0], b[1], "end", [b])
        g.add_edge(u, v, path)

    for nid in list(g.nodes):
        if g.degree(nid) == 0 and g.nodes[nid].kind == "junction":
            # isolated junction knot (e.g. a tiny closed blob)
            g.remove_node(nid)
    return g


# --------------------------------------------------------------- clean-up
def _local_radius(radius: np.ndarray, pixels: np.ndarray) -> float:
    if len(pixels) == 0:
        return 1.0
    return float(np.median(radius[pixels[:, 0], pixels[:, 1]]))


def _edge_dir(path: np.ndarray, length: float) -> np.ndarray | None:
    """Unit vector (y, x) pointing out of the free end at path[-1]."""
    if len(path) < 2:
        return None
    k = int(min(len(path) - 1, max(2, round(length))))
    d = path[-1].astype(float) - path[-1 - k].astype(float)
    n = np.hypot(*d)
    return d / n if n > 0 else None


def link_free_ends(g: VesselGraph, skel: np.ndarray, evidence: np.ndarray,
                   radius: np.ndarray, cfg: Config) -> tuple[np.ndarray, int]:
    """Draw straight bridges from free ends to centerlines they point at.

    A free end may stop at the *edge* of a wider vessel it actually joins or
    passes under (its own ridge is swamped there), so the reach is the end's
    own allowance plus the radius of the vessel it would connect to. The
    bridge must run over dense line evidence (max_theta U), not empty space.
    """
    skel = skel.copy()
    h, w = skel.shape
    if not skel.any():
        return skel, 0
    owner = np.full(skel.shape, -1, dtype=np.int64)
    for e in g.edges.values():
        owner[e.path[:, 0], e.path[:, 1]] = e.id
    cos_cone = np.cos(np.deg2rad(cfg.link_cone))
    rmax = float(max(cfg.radii))
    n_links = 0
    for n in g.endpoints():
        inc = g.incident(n.id)
        if len(inc) != 1:
            continue
        eid, out = inc[0]
        path = out[::-1]  # ends at the free end
        r = _local_radius(radius, path[-min(len(path), 8):])
        d = _edge_dir(path, 2 * r + 3)
        if d is None:
            continue
        own_reach = max(cfg.link_min, cfg.link_factor * r)
        R = int(np.ceil(own_reach + rmax + 1))
        p0 = path[-1].astype(float)
        y0, x0 = int(p0[0]), int(p0[1])
        ylo, yhi = max(0, y0 - R), min(h, y0 + R + 1)
        xlo, xhi = max(0, x0 - R), min(w, x0 + R + 1)
        cy, cx = np.nonzero(skel[ylo:yhi, xlo:xhi])
        cy += ylo
        cx += xlo
        vec = np.c_[cy - p0[0], cx - p0[1]]
        dist = np.hypot(vec[:, 0], vec[:, 1])
        reach = own_reach + radius[cy, cx]
        ok = (owner[cy, cx] != eid) & (dist > 0.5) & (dist <= reach)
        if not ok.any():
            continue
        cosang = (vec @ d) / np.maximum(dist, 1e-9)
        ok &= cosang >= cos_cone
        if not ok.any():
            continue
        cost = dist[ok] / reach[ok] * (2.0 - cosang[ok])
        j = np.flatnonzero(ok)[np.argmin(cost)]
        ty, tx = int(cy[j]), int(cx[j])
        rr, cc = draw_line(y0, x0, ty, tx)
        if np.mean(evidence[rr, cc] > cfg.t_low) < 0.6:
            continue
        skel[rr, cc] = True
        n_links += 1
    return skel, n_links


def _dissolve(g: VesselGraph, nid: int):
    """Replace a degree-2 node and its two edges by a single edge."""
    inc = g.incident(nid)
    if len(inc) != 2:
        return
    (e1, p1), (e2, p2) = inc
    node = g.nodes[nid]
    if e1 == e2:  # a loop hanging on this node only: drop it
        g.remove_node(nid)
        return
    a = g.other(g.edges[e1], nid)
    b = g.other(g.edges[e2], nid)
    mid = node.pixels
    if len(mid) and len(p1) and len(p2):
        d = (p2[0] - p1[0]).astype(float)
        mid = mid[np.argsort((mid - p1[0]) @ d)]
    path = np.concatenate([p1[::-1], mid, p2])
    g.remove_node(nid)
    g.add_edge(a, b, path)


def _settle(g: VesselGraph, nid: int) -> bool:
    """Fix up a junction node after its degree changed. True if changed."""
    if nid not in g.nodes or g.nodes[nid].kind != "junction":
        return False
    deg = g.degree(nid)
    if deg == 0:
        g.remove_node(nid)
    elif deg == 1:
        g.nodes[nid].kind = "end"
    elif deg == 2:
        _dissolve(g, nid)
    else:
        return False
    return True


def prune(g: VesselGraph, radius: np.ndarray, cfg: Config) -> VesselGraph:
    changed = True
    while changed:
        changed = False
        for nid in list(g.nodes):
            changed |= _settle(g, nid)
        for eid in list(g.edges):
            e = g.edges.get(eid)
            if e is None:
                continue
            ku, kv = g.nodes[e.u].kind, g.nodes[e.v].kind
            if ku == "end" and kv == "end":
                if len(e) < cfg.min_length:
                    u, v = e.u, e.v
                    g.remove_node(u)
                    if v != u:
                        g.remove_node(v)
                    changed = True
                continue
            if ku == kv:
                continue
            end, jn = (e.u, e.v) if ku == "end" else (e.v, e.u)
            if g.degree(jn) < 3:
                continue
            r = _local_radius(radius, g.nodes[jn].pixels)
            if len(e) < max(cfg.spur_min, cfg.spur_factor * r):
                g.remove_node(end)
                _settle(g, jn)
                changed = True
    return g


def fuse_nodes(g: VesselGraph, members: list[int], internal: list[int],
               center: tuple[float, float] | None = None) -> int:
    """Replace ``members`` (and the ``internal`` edges joining them) by one
    compound junction node; returns its id."""
    pix = [g.nodes[m].pixels for m in members] + [g.edges[e].path for e in internal]
    if center is None:
        center = (np.mean([g.nodes[m].y for m in members]),
                  np.mean([g.nodes[m].x for m in members]))
    mem = sum((g.nodes[m].members for m in members), [])
    new = g.add_node(center[0], center[1], "junction", np.concatenate(pix), mem)
    for eid in internal:
        if eid in g.edges:
            g.remove_edge(eid)
    mset = set(members)
    for m in members:
        for eid in list(dict.fromkeys(g._inc[m])):
            e = g.edges[eid]
            g.relink(eid, new if e.u in mset else e.u, new if e.v in mset else e.v)
        g.remove_node(m)
    return new


def merge_close_junctions(g: VesselGraph, radius: np.ndarray, cfg: Config) -> VesselGraph:
    """Unconditionally fuse junctions joined by an edge shorter than about one
    vessel radius: at that distance they are one junction smeared by the
    skeleton. (Longer separations are only fused when the arms say so, see
    junctions.fuse_crossings.)"""
    parent = {n: n for n in g.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    internal = []
    for e in g.edges.values():
        if e.u == e.v:
            continue
        if g.nodes[e.u].kind != "junction" or g.nodes[e.v].kind != "junction":
            continue
        r = max(_local_radius(radius, e.path), 1.0)
        if len(e) <= max(cfg.merge_min, cfg.merge_factor * r):
            parent[find(e.u)] = find(e.v)
            internal.append(e.id)

    groups: dict[int, list[int]] = {}
    for n in g.nodes:
        groups.setdefault(find(n), []).append(n)
    internal_by_root: dict[int, list[int]] = {}
    for eid in internal:
        internal_by_root.setdefault(find(g.edges[eid].u), []).append(eid)
    touched = [fuse_nodes(g, m, internal_by_root.get(root, []))
               for root, m in groups.items() if len(m) > 1]
    for nid in touched:
        _settle(g, nid)
    return g


def vessel_graph(skel: np.ndarray, evidence: np.ndarray, radius: np.ndarray,
                 cfg: Config) -> tuple[VesselGraph, np.ndarray]:
    """Full graph construction. ``evidence`` is the dense max_theta U map.
    Returns (graph, linked skeleton)."""
    from skimage.morphology import skeletonize

    g = build_graph(skel)
    skel2, n = link_free_ends(g, skel, evidence, radius, cfg)
    if n:
        skel2 = skeletonize(skel2)
        g = build_graph(skel2)
    g = prune(g, radius, cfg)
    g = merge_close_junctions(g, radius, cfg)
    return g, skel2
