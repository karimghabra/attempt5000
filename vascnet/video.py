"""Video: temporal integration and junction tracking for stabilized sequences.

Stabilized video gives the visual system two things a single frame doesn't:

1. Temporal integration. Photoreceptors integrate over ~tens of ms; here a
   leaky integrator on the optical-density frames (an exponential moving
   average) raises SNR by ~sqrt(2*tau) for static structure without blurring
   it, because stabilization keeps the vessels in place.

2. Persistence. A real junction is present frame after frame at the same
   place; a noise-induced one flickers. Junctions are tracked with gated
   Hungarian assignment, their type is decided by vote over the track, and a
   track is only reported once it has been seen in ``min_hits`` frames. Every
   structure keeps one ID for the whole clip, which is what an annotation tool
   needs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Config
from .junctions import Junction
from .pipeline import Result, VesselPipeline
from .retina import adapt


class TemporalIntegrator:
    """Leaky integrator (EMA) with time constant ``tau`` frames."""

    def __init__(self, tau: float = 4.0):
        self.alpha = 1.0 / max(tau, 1.0)
        self.state: np.ndarray | None = None
        self.n = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        self.n += 1
        if self.state is None:
            self.state = frame.astype(np.float32).copy()
        else:
            # running mean until the window fills, then exponential forgetting
            a = max(self.alpha, 1.0 / self.n)
            self.state += a * (frame - self.state)
        return self.state


@dataclass
class Track:
    id: int
    x: float
    y: float
    radius: float
    votes: Counter = field(default_factory=Counter)
    confs: list = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    first: int = 0
    last: int = 0
    history: list = field(default_factory=list)   # (frame, x, y, type, conf)

    @property
    def type(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "unknown"

    @property
    def confidence(self) -> float:
        if not self.votes:
            return 0.0
        agreement = self.votes.most_common(1)[0][1] / sum(self.votes.values())
        return float(agreement * np.mean(self.confs[-20:]))

    def to_dict(self) -> dict:
        return dict(id=self.id, x=round(self.x, 2), y=round(self.y, 2), type=self.type,
                    confidence=round(self.confidence, 3), radius=round(self.radius, 2),
                    hits=self.hits, first_frame=self.first, last_frame=self.last,
                    type_votes=dict(self.votes))


class JunctionTracker:
    def __init__(self, gate: float = 5.0, gate_factor: float = 1.5, min_hits: int = 3,
                 max_misses: int = 8, smoothing: float = 0.2,
                 types=("bifurcation", "crossover", "complex", "contact")):
        self.gate, self.gate_factor = gate, gate_factor
        self.min_hits, self.max_misses = min_hits, max_misses
        self.smoothing = smoothing
        self.types = types
        self.active: list[Track] = []
        self.archive: list[Track] = []
        self._next = 0

    def update(self, frame: int, junctions: list[Junction]) -> list[Track]:
        dets = [j for j in junctions if j.type in self.types]
        T = self.active
        if T and dets:
            D = np.hypot(np.array([t.x for t in T])[:, None] - np.array([d.x for d in dets])[None],
                         np.array([t.y for t in T])[:, None] - np.array([d.y for d in dets])[None])
            gate = np.array([max(self.gate, self.gate_factor * t.radius) for t in T])[:, None]
            C = np.where(D <= gate, D, 1e6)
            rows, cols = linear_sum_assignment(C)
            pairs = [(r, c) for r, c in zip(rows, cols) if C[r, c] < 1e6]
        else:
            pairs = []
        hit_t = {r for r, _ in pairs}
        hit_d = {c for _, c in pairs}
        a = self.smoothing
        for r, c in pairs:
            t, d = T[r], dets[c]
            t.x += a * (d.x - t.x)
            t.y += a * (d.y - t.y)
            t.radius += a * (d.radius - t.radius)
            t.votes[d.type] += 1
            t.confs.append(d.confidence)
            t.hits += 1
            t.misses = 0
            t.last = frame
            t.history.append((frame, d.x, d.y, d.type, d.confidence))
        for i, t in enumerate(T):
            if i not in hit_t:
                t.misses += 1
        for c, d in enumerate(dets):
            if c not in hit_d:
                t = Track(self._next, d.x, d.y, d.radius, Counter({d.type: 1}), [d.confidence],
                          1, 0, frame, frame, [(frame, d.x, d.y, d.type, d.confidence)])
                self._next += 1
                self.active.append(t)
        keep = []
        for t in self.active:
            (keep if t.misses <= self.max_misses else self.archive).append(t)
        self.active = keep
        return [t for t in self.active if t.hits >= self.min_hits and t.misses == 0]

    def all_tracks(self, min_hits: int | None = None) -> list[Track]:
        m = self.min_hits if min_hits is None else min_hits
        return [t for t in self.archive + self.active if t.hits >= m]


@dataclass
class VideoResult:
    frames: list[dict]          # per processed frame: {'frame', 'tracks': [...]} confirmed & visible
    tracks: list[Track]         # every confirmed track over the clip
    last: Result | None = None  # full result of the last processed frame

    def consensus(self, min_fraction: float = 0.3) -> list[dict]:
        """One annotation per persistent structure for the whole clip."""
        n = max(len(self.frames), 1)
        return [t.to_dict() for t in self.tracks if t.hits / n >= min_fraction]


class VideoAnnotator:
    """Run the fixed pipeline over a stabilized sequence.

    tau    temporal integration constant (frames); 1 disables integration
    every  run the spatial pipeline every k-th frame (tracks persist between)
    """

    def __init__(self, cfg: Config | None = None, tau: float = 4.0, every: int = 1, **tracker_kw):
        self.cfg = cfg or Config()
        self.pipe = VesselPipeline(self.cfg)
        self.integ = TemporalIntegrator(tau)
        self.every = max(1, every)
        self.tracker = JunctionTracker(**tracker_kw)

    def run(self, frames: Iterable[np.ndarray], callback=None) -> VideoResult:
        out, last = [], None
        for k, frame in enumerate(frames):
            od = self.integ(adapt(frame, self.cfg))
            if k % self.every:
                continue
            last = self.pipe.from_od(od.copy())
            confirmed = self.tracker.update(k, last.junctions)
            out.append(dict(frame=k, tracks=[t.to_dict() for t in confirmed]))
            if callback:
                callback(k, last, confirmed)
        return VideoResult(out, self.tracker.all_tracks(), last)
