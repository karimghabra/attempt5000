"""Watch a fit happen: record its states every few candidates, and render the record as an animation.

A fit (the sign moves of the flux network, the knot refinement of a spline, the rounds of the recursion) is a
sequence of candidates, some taken and some refused. A `Recorder` is handed every candidate as a small dict (the
caller decides what a state holds: fluxes, splines, a residual picture) and keeps one in `every`, plus every
taken one if asked, plus the last. `animate` then draws the kept states with a drawing function the caller
supplies (one state in, one RGB frame out) and writes a GIF, so that the fit can be watched as a film: what was
proposed, what was taken, what it changed.

Recording is cheap (a state is a few thousand numbers) and drawing is done afterwards, so watching costs the fit
nothing but memory. The notebook's `create_animation` (section 2) plays grayscale stacks; this one plays RGB
frames drawn with matplotlib, at a fixed size, and is independent of the notebook.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Recorder:
    """Keeps every `every`-th candidate state, every taken one when `keep_taken`, and always the last.

    A state is whatever dict the fit hands over; the recorder adds `candidate` (its number in the fit) and keeps
    the caller's `taken` flag. `states` is the record, in order."""
    every: int = 5
    keep_taken: bool = True
    states: list[dict] = field(default_factory=list)
    count: int = 0

    def __call__(self, state: dict) -> None:
        self.count += 1
        state = dict(state, candidate=self.count)
        if (self.count - 1) % max(self.every, 1) == 0 or (self.keep_taken and state.get("taken")) or state.get("last"):
            self.states.append(state)

    def __len__(self) -> int:
        return len(self.states)


def frame_of(figure) -> np.ndarray:
    """A matplotlib figure as an RGB array (H, W, 3), drawn at its own size and dpi."""
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())
    return buffer[..., :3].copy()


def animate(states: list[dict], draw: Callable[[dict], np.ndarray], path: str, fps: float = 4.0,
            hold_last: int = 6) -> str:
    """Draw every state with `draw` (state -> RGB frame of one size) and write the frames as a GIF at `path`,
    the last frame held for `hold_last` extra ticks so that the end can be seen. Returns the path."""
    from PIL import Image

    frames = [draw(state) for state in states]
    if not frames:
        raise ValueError("nothing recorded")
    frames = frames + [frames[-1]] * max(hold_last, 0)
    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    duration = int(round(1000 / max(fps, 1e-3)))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
    return path


def flux_frame(state: dict, positions: dict[int, tuple[float, float]], edges: list[tuple[int, int]],
               true: np.ndarray | None = None, size: tuple[float, float] = (6.0, 4.5), dpi: int = 80) -> np.ndarray:
    """A frame of the flux network's sign moves: the graph drawn with each edge's current direction (an arrow),
    its colour the flux's sign against the truth where one is given (green right, red wrong, grey unmeasured),
    the move of this candidate highlighted (the edges it would turn, in orange; a lumen it would set aside, in
    purple), and a caption with the candidate's number, kind, gain and whether it was taken."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.asarray(state["q"])
    measured = np.asarray(state["measured"])
    moved = set(state.get("edges", []))
    figure, axis = plt.subplots(figsize=size, dpi=dpi)
    for number, (a, b) in enumerate(edges):
        (xa, ya), (xb, yb) = positions[a], positions[b]
        if q[number] < 0:
            (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
        if true is not None and measured[number]:
            colour = "tab:green" if np.sign(q[number]) == np.sign(true[number]) else "tab:red"
        elif true is not None:
            colour = "tab:green" if np.sign(q[number]) == np.sign(true[number]) else "lightcoral"
        else:
            colour = "black" if measured[number] else "grey"
        width = 1.0 + 2.5 * min(abs(q[number]) / (np.max(np.abs(q)) + 1e-9), 1.0)
        if number in moved:
            axis.annotate("", xy=(xb, yb), xytext=(xa, ya),
                          arrowprops=dict(arrowstyle="-", color="orange", lw=width + 4, alpha=0.5))
        axis.annotate("", xy=(xb, yb), xytext=(xa, ya),
                      arrowprops=dict(arrowstyle="->", color=colour, lw=width, shrinkA=6, shrinkB=6))
    for vertex, (x, y) in positions.items():
        axis.plot(x, y, "o", color="white", markeredgecolor="black", markersize=7, zorder=3)
    for lumen in state.get("aside_lumens", []):  # a lumen is ((edge, sign), ...): mark its arms' midpoints
        for edge, _ in lumen:
            (xa, ya), (xb, yb) = positions[edges[edge][0]], positions[edges[edge][1]]
            axis.plot((xa + xb) / 2, (ya + yb) / 2, "s", color="purple", markersize=12, alpha=0.4, zorder=2)
    kind = state.get("kind", "")
    taken = state.get("taken", False)
    caption = (f"candidate {state.get('candidate', '?')}: {kind}"
               + (f" of edges {sorted(moved)}" if moved and kind == "flip" else "")
               + (f", gain {state['gain']:+.2f}" if "gain" in state else "")
               + (" — taken" if taken else (" — refused" if kind not in ("start", "done") else "")))
    axis.set_title(caption, fontsize=9)
    axis.set_aspect("equal")
    axis.axis("off")
    frame = frame_of(figure)
    plt.close(figure)
    return frame
