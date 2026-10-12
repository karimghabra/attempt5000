"""Synthetic stabilized clip: single-frame vs temporally integrated + tracked.

    python scripts/demo_video.py --frames 24 --noise 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vascnet import Config, VesselPipeline  # noqa: E402
from vascnet.evaluate import junction_metrics  # noqa: E402
from vascnet.synth import grow_network, render  # noqa: E402
from vascnet.video import VideoAnnotator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--out", default="docs/video_demo.json")
    a = ap.parse_args()

    net = grow_network(shape=(a.size, a.size), seed=a.seed)
    frames = [render(net, seed=a.seed * 1000 + k, noise=a.noise, flow=0.15, phase=0.6 * k)
              for k in range(a.frames)]

    # score everything against the structure that integration makes visible:
    # EMA with alpha=1/tau leaves noise / sqrt(2 tau - 1)
    floor = 3 * a.noise / np.sqrt(2 * a.tau - 1)
    single = VesselPipeline(Config())(frames[-1])
    m1 = junction_metrics(single.junctions, net.junctions, min_contrast=floor)

    t0 = time.perf_counter()
    va = VideoAnnotator(Config(), tau=a.tau, min_hits=3)
    vr = va.run(frames)
    dt = (time.perf_counter() - t0) / a.frames
    m2 = junction_metrics(vr.last.junctions, net.junctions, min_contrast=floor)
    cons = vr.consensus()
    m3 = junction_metrics(cons, net.junctions, min_contrast=floor)

    def row(name, m):
        print(f"{name:34s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
              f"type={m['type_accuracy']:.2f}  (pred {m['n_pred']}, gt {m['n_gt']})")

    print(f"{a.frames} frames, {a.size}x{a.size}, noise={a.noise}, {dt:.2f} s/frame")
    row("single frame", m1)
    row(f"integrated (tau={a.tau}), last frame", m2)
    row("clip consensus (tracked)", m3)
    ids = [len(f["tracks"]) for f in vr.frames]
    print("confirmed tracks per frame:", ids)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(consensus=cons, per_frame=vr.frames), f, indent=1)
    print("annotations written to", a.out)


if __name__ == "__main__":
    main()
