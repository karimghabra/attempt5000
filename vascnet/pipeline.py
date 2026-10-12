"""End-to-end, fixed-weight pipeline:

    image ─▶ retina (OD) ─▶ V1 even/odd filter bank ─▶ scale competition
          ─▶ recurrent horizontal interactions (×n_iter)
          ─▶ orientation-space NMS + hysteresis ─▶ skeleton
          ─▶ vessel graph (bridge, prune, fuse) ─▶ junction analysis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .centerline import extract_centerlines
from .config import Config
from .cortex import OrientationScore, feedforward, recurrent
from .graph import VesselGraph, vessel_graph
from .junctions import Junction, analyze
from .retina import adapt


@dataclass
class Result:
    od: np.ndarray
    score: OrientationScore
    strength: np.ndarray
    skeleton: np.ndarray
    radius: np.ndarray
    graph: VesselGraph
    junctions: list[Junction]
    timings: dict = field(default_factory=dict)

    @property
    def centerline(self) -> np.ndarray:
        return self.graph.to_mask()

    def of_type(self, *types: str) -> list[Junction]:
        return [j for j in self.junctions if j.type in types]

    def annotations(self) -> dict:
        """JSON-serializable annotation record."""
        segs = []
        for e in self.graph.edges.values():
            p = e.path
            segs.append({
                "id": e.id, "from": e.u, "to": e.v,
                "points_xy": p[:, ::-1].tolist(),
                "mean_radius": round(float(np.median(self.radius[p[:, 0], p[:, 1]])), 3),
            })
        return {
            "junctions": [j.to_dict() for j in self.junctions],
            "segments": segs,
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.annotations(), f)


class VesselPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

    def __call__(self, image: np.ndarray) -> Result:
        cfg = self.cfg
        t = {}
        t0 = time.perf_counter()
        od = adapt(image, cfg)
        t["retina"] = time.perf_counter() - t0
        return self.from_od(od, t)

    def from_od(self, od: np.ndarray, t: dict | None = None) -> Result:
        cfg = self.cfg
        t = {} if t is None else t
        t0 = time.perf_counter()
        score = feedforward(od, cfg)
        t["v1"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        recurrent(score, cfg)
        t["horizontal"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        skel, strength = extract_centerlines(score, cfg)
        radius = score.radius_map()
        t["centerline"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        g, skel = vessel_graph(skel, score.strength, radius, cfg)
        od_s = ndi.gaussian_filter(od, 1.0)
        junctions = analyze(g, score, od_s, radius, cfg)
        t["graph+junctions"] = time.perf_counter() - t0
        return Result(od, score, strength, skel, radius, g, junctions, t)
