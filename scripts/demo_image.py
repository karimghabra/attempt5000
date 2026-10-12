"""Run the pipeline on one image (yours or a synthetic one) and save figures
and JSON annotations.

    python scripts/demo_image.py                         # synthetic example
    python scripts/demo_image.py --image frame.png --polarity dark
    python scripts/demo_image.py --image frame.tif --radii 1.5 3 6 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vascnet import Config, VesselPipeline  # noqa: E402
from vascnet.evaluate import centerline_metrics, junction_metrics  # noqa: E402
from vascnet.synth import centerline_truth, make_sample  # noqa: E402


def load(path):
    from PIL import Image

    return np.asarray(Image.open(path))


def junction_gallery(image, res, path, per_type=4, R=26):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vascnet.viz import STYLE

    picks = []
    for t in ("bifurcation", "crossover", "complex"):
        js = sorted((j for j in res.junctions if j.type == t), key=lambda j: -j.confidence)
        picks += js[:per_type]
    if not picks:
        return
    cols = min(len(picks), 6)
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    cl = res.centerline
    rgba = np.zeros(cl.shape + (4,))
    rgba[cl] = (1.0, 0.85, 0.1, 0.55)
    for ax, j in zip(axes.ravel(), picks):
        st = STYLE[j.type]
        ax.imshow(image, cmap="gray")
        ax.imshow(rgba, interpolation="nearest")
        for a in j.arms:
            L = 3 * a.width + 8
            ax.plot([j.x, j.x + L * np.cos(a.angle)], [j.y, j.y + L * np.sin(a.angle)],
                    color=st["color"], lw=1.6, ls=":" if a.virtual else "-")
        for p, q in j.pairs:  # same-vessel pairs
            a, b = j.arms[p], j.arms[q]
            ax.plot([j.x + 6 * np.cos(a.angle), j.x + 6 * np.cos(b.angle)],
                    [j.y + 6 * np.sin(a.angle), j.y + 6 * np.sin(b.angle)], color="w", lw=0.8)
        ax.scatter([j.x], [j.y], s=40, marker=st["marker"], color=st["color"])
        ax.set_xlim(j.x - R, j.x + R)
        ax.set_ylim(j.y + R, j.y - R)
        extra = ""
        if j.type == "bifurcation" and "murray_ratio" in j.features:
            extra = f"\nMurray {j.features['murray_ratio']:.2f}"
        if j.type == "crossover" and "crossing_angle" in j.features:
            extra = f"\nangle {j.features['crossing_angle']:.0f}°  OD add {j.features['additivity']:.2f}"
        ax.set_title(f"{j.type}/{j.subtype}  c={j.confidence:.2f}{extra}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--polarity", default="dark", choices=["dark", "bright"])
    ap.add_argument("--radii", type=float, nargs="*", default=None,
                    help="vessel radii (px) the filter bank spans")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    kw = {"polarity": a.polarity}
    if a.radii:
        kw["radii"] = tuple(a.radii)
    cfg = Config(**kw)

    net = None
    if a.image:
        image = load(a.image)
        stem = os.path.splitext(os.path.basename(a.image))[0]
    else:
        image, net = make_sample(seed=a.seed)
        stem = f"synthetic_{a.seed}"

    res = VesselPipeline(cfg)(image)
    print("timings (s):", {k: round(v, 2) for k, v in res.timings.items()})
    from collections import Counter

    print("junctions:", dict(Counter(j.type for j in res.junctions)))
    if net is not None:
        gt, _ = centerline_truth(net)
        print("centerline:", {k: round(v, 3) for k, v in centerline_metrics(res.centerline, gt).items()})
        m = junction_metrics(res.junctions, net.junctions)
        print("junctions vs truth: P={precision:.2f} R={recall:.2f} F1={f1:.2f} type={type_accuracy:.2f}".format(**m))

    from vascnet.viz import panel

    img_disp = image if image.ndim == 2 else image[..., 1]
    panel(img_disp, res, net, path=os.path.join(a.out, f"{stem}_panel.png"))
    junction_gallery(img_disp, res, os.path.join(a.out, f"{stem}_junctions.png"))
    with open(os.path.join(a.out, f"{stem}_annotations.json"), "w") as f:
        json.dump(res.annotations(), f)
    print("wrote", os.path.join(a.out, f"{stem}_*"))


if __name__ == "__main__":
    main()
