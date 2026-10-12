"""Metrics against synthetic (or manually annotated) ground truth."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

JUNCTION_TYPES = ("bifurcation", "crossover", "complex", "contact")


def centerline_metrics(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0,
                       gt_weight_mask: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    d_gt = ndi.distance_transform_edt(~gt)
    d_pr = ndi.distance_transform_edt(~pred)
    precision = float(np.mean(d_gt[pred] <= tol))
    sel = gt if gt_weight_mask is None else gt & gt_weight_mask
    recall = float(np.mean(d_pr[sel] <= tol)) if sel.any() else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(precision=precision, recall=recall, f1=f1)


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray, gate: np.ndarray | float):
    """Hungarian matching with a per-GT gating radius. Returns list of (i_pred, j_gt)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return []
    D = np.hypot(pred_xy[:, None, 0] - gt_xy[None, :, 0], pred_xy[:, None, 1] - gt_xy[None, :, 1])
    gate = np.broadcast_to(np.asarray(gate, float), (len(gt_xy),))
    big = 1e6
    C = np.where(D <= gate[None, :], D, big)
    r, c = linear_sum_assignment(C)
    return [(i, j) for i, j in zip(r, c) if C[i, j] < big]


def cluster_truth(gt: list[dict], factor: float = 2.0, min_dist: float = 5.0,
                  types=("bifurcation", "crossover")) -> list[dict]:
    """Fuse ground-truth junctions closer than max(min_dist, factor*(r_i+r_j))
    into one 'complex' junction - the way a human annotator would mark a
    vessel crossing right at a branch point, or two crossings that touch."""
    G = [dict(g) for g in gt if g["type"] in types]
    parent = list(range(len(G)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            d = np.hypot(G[i]["x"] - G[j]["x"], G[i]["y"] - G[j]["y"])
            if d <= max(min_dist, factor * (G[i].get("radius", 1) + G[j].get("radius", 1))):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(G)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(G[members[0]])
            continue
        ms = [G[m] for m in members]
        out.append(dict(
            x=float(np.mean([m["x"] for m in ms])), y=float(np.mean([m["y"] for m in ms])),
            type="complex", radius=float(max(m.get("radius", 1) for m in ms)),
            visible=all(m.get("visible", True) for m in ms),
            members=[m["type"] for m in ms],
        ))
    return out


def junction_metrics(pred: list, gt: list[dict], tol: float = 6.0, radius_factor: float = 2.0,
                     types=("bifurcation", "crossover", "complex"), cluster: bool = True,
                     only_visible: bool = True, min_contrast: float | None = None) -> dict:
    """Detection (any junction type) and type accuracy on matched pairs.

    pred: Junction objects (or dicts with x, y, type)
    gt:   dicts with x, y, type, radius[, visible]
    A prediction counts as a junction if its type is in JUNCTION_TYPES.
    Ground truth flagged visible=False (a participating vessel is below
    3x noise) is "don't care": matching it is neither a hit nor a false alarm.
    min_contrast re-derives visibility from each junction's rendered contrast
    (e.g. for temporally integrated video, where the effective noise is lower);
    geometric ambiguity flags (stub, tangential) are kept.
    """
    if min_contrast is not None:
        gt = [dict(g, visible=bool(g.get("contrast", np.inf) >= min_contrast
                                   and g.get("ambiguous") not in ("stub", "tangential")))
              for g in gt]
    P = [p if isinstance(p, dict) else {"x": p.x, "y": p.y, "type": p.type} for p in pred]
    P = [p for p in P if p["type"] in JUNCTION_TYPES]
    G = cluster_truth(gt) if cluster else [g for g in gt if g["type"] in types]
    G = [g for g in G if g["type"] in types]
    if only_visible:
        dont_care = [g for g in G if not g.get("visible", True)]
        G = [g for g in G if g.get("visible", True)]
    else:
        dont_care = []
    pxy = np.array([[p["x"], p["y"]] for p in P]).reshape(-1, 2)
    gxy = np.array([[g["x"], g["y"]] for g in G]).reshape(-1, 2)
    gate = np.array([max(tol, radius_factor * g.get("radius", 0) + 2) for g in G])
    m = match_points(pxy, gxy, gate)
    if dont_care:  # unmatched predictions sitting on invisible truth are ignored
        matched = {i for i, _ in m}
        rest = [i for i in range(len(P)) if i not in matched]
        dxy = np.array([[g["x"], g["y"]] for g in dont_care])
        dgate = np.array([max(tol, radius_factor * g.get("radius", 0) + 2) for g in dont_care])
        ign = {rest[i] for i, _ in match_points(pxy[rest], dxy, dgate)} if rest else set()
    else:
        ign = set()
    tp = len(m)
    n_pred = len(P) - len(ign)
    prec = tp / max(n_pred, 1)
    rec = tp / max(len(G), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    conf = Counter((G[j]["type"], P[i]["type"]) for i, j in m)
    per_type = {}
    for t in types:
        n_gt = sum(g["type"] == t for g in G)
        hit = sum(1 for i, j in m if G[j]["type"] == t)
        correct = conf.get((t, t), 0)
        per_type[t] = dict(n_gt=n_gt, detected=hit, recall=hit / max(n_gt, 1),
                           type_correct=correct, type_acc=correct / max(hit, 1))
    type_acc = sum(conf.get((t, t), 0) for t in types) / max(tp, 1)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, n_pred=n_pred, n_gt=len(G),
                type_accuracy=type_acc, per_type=per_type,
                confusion={f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
                matches=m)
