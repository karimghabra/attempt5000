"""Benchmark the fixed pipeline across imaging conditions on synthetic data.

    python scripts/benchmark.py                    # held-out seeds, all conditions
    python scripts/benchmark.py --seeds 0-5 --conditions default,noisy
    python scripts/benchmark.py --json out.json

Seeds 0-49 are the development range (used while designing the pipeline);
report on seeds >= 1000, which were never looked at during design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vascnet import Config, VesselPipeline  # noqa: E402
from vascnet.evaluate import centerline_metrics, junction_metrics  # noqa: E402
from vascnet.synth import centerline_truth, make_sample  # noqa: E402

CONDITIONS = {
    "default":  ({}, {}),
    "noisy":    ({}, {"noise": 0.04}),
    "blurry":   ({"blur_range": (1.5, 3.0)}, {}),
    "faint":    ({"kappa_range": (0.02, 0.045)}, {}),
    "tortuous": ({"meander_amp": 0.6, "meander_wavelength": (20.0, 45.0)}, {}),
    "knotted":  ({"tortuosity": 0.08}, {}),
    "dense":    ({"n_trees": 6, "branch_rate": 1 / 45}, {}),
    "reflex":   ({"reflex_prob": 1.0, "r_root": (4.0, 6.5)}, {}),
    "texture":  ({}, {"texture": 0.08, "illumination": 0.5}),
}


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_one(args):
    cond, seed, cfg_kw, shape = args
    grow_kw, render_kw = CONDITIONS[cond]
    img, net = make_sample(seed=seed, shape=shape, grow_kw=grow_kw, render_kw=render_kw)
    pipe = VesselPipeline(Config(**cfg_kw))
    t0 = time.perf_counter()
    res = pipe(img)
    dt = time.perf_counter() - t0
    gt, rad = centerline_truth(net)
    vis = np.zeros_like(gt)
    for b in net.branches:  # recall only on vessels above 3x noise
        if b.contrast >= 3 * render_kw.get("noise", 0.015):
            m, _ = centerline_truth(type(net)(net.shape, [b], net.styles, []))
            vis |= m
    vis[:3] = vis[-3:] = False  # border band is not traced (see centerline.py)
    vis[:, :3] = vis[:, -3:] = False
    cl = centerline_metrics(res.centerline, gt, gt_weight_mask=vis)
    jm = junction_metrics(res.junctions, net.junctions)
    jm.pop("matches")
    return cond, seed, dt, cl, jm


def aggregate(rows):
    out = {}
    for cond in dict.fromkeys(r[0] for r in rows):
        rs = [r for r in rows if r[0] == cond]
        tp = sum(r[4]["tp"] for r in rs)
        npred = sum(r[4]["n_pred"] for r in rs)
        ngt = sum(r[4]["n_gt"] for r in rs)
        p, rc = tp / max(npred, 1), tp / max(ngt, 1)
        per = {}
        for t in ("bifurcation", "crossover", "complex"):
            n = sum(r[4]["per_type"][t]["n_gt"] for r in rs)
            d = sum(r[4]["per_type"][t]["detected"] for r in rs)
            c = sum(r[4]["per_type"][t]["type_correct"] for r in rs)
            per[t] = dict(n=n, recall=d / max(n, 1), type_acc=c / max(d, 1))
        out[cond] = dict(
            n_images=len(rs),
            sec_per_image=float(np.mean([r[2] for r in rs])),
            centerline_precision=float(np.mean([r[3]["precision"] for r in rs])),
            centerline_recall=float(np.mean([r[3]["recall"] for r in rs])),
            centerline_f1=float(np.mean([r[3]["f1"] for r in rs])),
            junction_precision=p, junction_recall=rc,
            junction_f1=2 * p * rc / max(p + rc, 1e-12),
            type_accuracy=sum(r[4]["tp"] * r[4]["type_accuracy"] for r in rs) / max(tp, 1),
            per_type=per,
        )
    return out


def print_table(agg):
    hdr = (f"{'condition':10s} {'CL-P':>5s} {'CL-R':>5s} {'CL-F1':>5s} | {'J-P':>5s} {'J-R':>5s} "
           f"{'J-F1':>5s} {'type':>5s} | {'bif R':>5s} {'bif T':>5s} {'xov R':>5s} {'xov T':>5s} "
           f"{'cpx R':>5s} | {'s/img':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for cond, a in agg.items():
        pt = a["per_type"]
        print(f"{cond:10s} {a['centerline_precision']:5.2f} {a['centerline_recall']:5.2f} "
              f"{a['centerline_f1']:5.2f} | {a['junction_precision']:5.2f} {a['junction_recall']:5.2f} "
              f"{a['junction_f1']:5.2f} {a['type_accuracy']:5.2f} | "
              f"{pt['bifurcation']['recall']:5.2f} {pt['bifurcation']['type_acc']:5.2f} "
              f"{pt['crossover']['recall']:5.2f} {pt['crossover']['type_acc']:5.2f} "
              f"{pt['complex']['recall']:5.2f} | {a['sec_per_image']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1007")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--json", default=None)
    ap.add_argument("--cfg", default="{}", help="JSON dict of Config overrides")
    a = ap.parse_args()
    cfg_kw = json.loads(a.cfg)
    if "radii" in cfg_kw:
        cfg_kw["radii"] = tuple(cfg_kw["radii"])
    jobs = [(c, s, cfg_kw, (a.size, a.size)) for c in a.conditions.split(",") for s in parse_seeds(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, jobs))
    agg = aggregate(rows)
    print_table(agg)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(config=Config(**cfg_kw).to_dict(), seeds=a.seeds, results=agg), f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
