"""Watch the flux network's sign step as an animation.

    python scripts/watch_flux_demo.py                 # writes docs/flux_fit_demo.gif
    python scripts/watch_flux_demo.py --out my.gif --every 1

Two fits are filmed one after the other: the toy graph of tests/test_fluxnet.py with two measurements read the
wrong way round, and a 6 x 6 anastomotic grid with harmonic flow, 70% of its edges measured and 15% of those
reversed. The sign step hands the recorder every candidate move it considers, with the directions that move
would give; the recorder keeps one in a few (three for the toy graph, forty for the grid) and every move taken.
In each frame the edges the candidate would turn are highlighted in orange, a reading it would set aside in
purple, arrows show the directions the candidate would give, and the colour says whether a direction is right
(green) or wrong (red) against the truth. The caption says the candidate's gain and whether it was taken.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from vascnet.fluxnet import lumens_of, solve_flux_network  # noqa: E402
from vascnet.watch import Recorder, animate, flux_frame  # noqa: E402


def toy_case():
    from test_fluxnet import ARMS, EDGES, READING, TRUE, measured
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(1, 7))
    positions = {0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (1, 0), 4: (2, 2), 5: (3, 1), 6: (3, 2), 7: (4, 1)}
    recorder = Recorder(every=3)
    solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), sigma_flux=0.3, watch=recorder)
    return [flux_frame(state, positions, EDGES, TRUE) for state in recorder.states]


def grid_case(n=6, seed=1):
    from test_fluxnet import grid_case as make
    recorder = Recorder(every=40)
    case = make(n, seed, watch=recorder)
    positions = {i * n + j: (j, -i) for i in range(n) for j in range(n)}
    return [flux_frame(state, positions, case["edges"], case["true"]) for state in recorder.states]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "flux_fit_demo.gif"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    frames = toy_case() + grid_case()
    states = [{"frame": frame} for frame in frames]
    path = animate(states, lambda state: state["frame"], args.out, fps=args.fps)
    print(f"{len(frames)} frames written to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
