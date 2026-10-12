"""Overlays and diagnostic panels (matplotlib)."""

from __future__ import annotations

import numpy as np

STYLE = {
    "bifurcation": dict(marker="o", color="#21c45d", label="bifurcation"),
    "crossover": dict(marker="X", color="#ff3fb4", label="crossover"),
    "complex": dict(marker="s", color="#ff9a1f", label="complex"),
    "contact": dict(marker="D", color="#b48cff", label="contact"),
    "endpoint": dict(marker=".", color="#3fa7ff", label="endpoint"),
}


def orientation_rgb(score, gamma: float = 0.6, vmax: float | None = None) -> np.ndarray:
    """Hue = dominant orientation, value = strength."""
    from matplotlib.colors import hsv_to_rgb

    U = score.U if score.U is not None else score.z
    s = U.max(0)
    hue = score.thetas[U.argmax(0)] / np.pi
    vmax = vmax or np.percentile(s, 99.5)
    val = np.clip(s / (vmax + 1e-9), 0, 1) ** gamma
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), val], -1))


def draw_junctions(ax, junctions, size=60, show_end=False, show_arms=False, min_conf=0.0):
    seen = set()
    for j in junctions:
        if j.type == "endpoint" and not show_end:
            continue
        if j.confidence < min_conf:
            continue
        st = STYLE.get(j.type, STYLE["complex"])
        lab = st["label"] if st["label"] not in seen else None
        seen.add(st["label"])
        ax.scatter([j.x], [j.y], s=size if j.type != "endpoint" else size / 3,
                   marker=st["marker"], facecolors="none" if st["marker"] in "os" else st["color"],
                   edgecolors=st["color"], linewidths=1.6, label=lab)
        if show_arms:
            for a in j.arms:
                L = 3 * a.width + 6
                ax.plot([j.x, j.x + L * np.cos(a.angle)], [j.y, j.y + L * np.sin(a.angle)],
                        color=st["color"], lw=0.8, ls=":" if a.virtual else "-")


def overlay(ax, image, result, title=None, show_end=False, show_arms=False):
    ax.imshow(image, cmap="gray")
    cl = result.centerline
    rgba = np.zeros(cl.shape + (4,))
    rgba[cl] = (1.0, 0.85, 0.1, 0.9)
    ax.imshow(rgba, interpolation="nearest")
    draw_junctions(ax, result.junctions, show_end=show_end, show_arms=show_arms)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def gt_overlay(ax, image, net, title=None):
    ax.imshow(image, cmap="gray")
    for g in net.junctions:
        st = STYLE.get(g["type"])
        if st is None or g["type"] == "endpoint":
            continue
        ax.scatter([g["x"]], [g["y"]], s=60, marker=st["marker"],
                   facecolors="none" if st["marker"] in "os" else st["color"],
                   edgecolors=st["color"], linewidths=1.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def panel(image, result, net=None, path=None, dpi=80):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 4 if net is None else 5
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("input", fontsize=10)
    axes[0].set_axis_off()
    axes[1].imshow(result.od, cmap="magma", vmin=0, vmax=np.percentile(result.od, 99.7))
    axes[1].set_title("retina: optical density", fontsize=10)
    axes[1].set_axis_off()
    axes[2].imshow(orientation_rgb(result.score))
    axes[2].set_title("cortex: orientation score (hue=θ)", fontsize=10)
    axes[2].set_axis_off()
    overlay(axes[3], image, result, "centerlines + junctions")
    axes[3].legend(loc="lower right", fontsize=7, framealpha=0.7)
    if net is not None:
        gt_overlay(axes[4], image, net, "ground truth")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return fig
