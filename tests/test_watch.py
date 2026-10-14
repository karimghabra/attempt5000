"""Watching a fit: the recorder keeps what it should, the flux network reports its moves, and a GIF is written."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_fluxnet import ARMS, EDGES, READING, TRUE, measured  # noqa: E402

from vascnet.fluxnet import lumens_of, solve_flux_network  # noqa: E402
from vascnet.watch import Recorder, animate, flux_frame  # noqa: E402


def test_the_recorder_keeps_every_nth_candidate_every_taken_one_and_the_last():
    recorder = Recorder(every=3)
    for number in range(10):
        recorder({"taken": number in (4, 5), "last": number == 9})
    kept = [state["candidate"] for state in recorder.states]
    assert kept == [1, 4, 5, 6, 7, 10]  # 1, 4, 7, 10 by every third; 5, 6 taken (candidates 5 and 6); 10 the last


def test_the_flux_network_reports_its_moves_to_a_recorder():
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(1,))
    recorder = Recorder(every=1)
    solution = solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), watch=recorder)
    kinds = [state["kind"] for state in recorder.states]
    assert kinds[0] == "start" and kinds[-1] == "done"
    taken = [state for state in recorder.states if state["taken"]]
    assert len(taken) == 1 and taken[0]["kind"] == "flip" and taken[0]["edges"] == [1]
    assert taken[0]["gain"] < 0
    assert solution.flipped[1]
    assert recorder.states[-1]["flipped"][1], "the last state carries the final signs"


def test_a_film_of_the_fit_is_written(tmp_path):
    q_hat, sigma_hat = measured(0, 1, 2, 3, 4, 5, 7, flip=(1,))
    recorder = Recorder(every=1)
    solve_flux_network(q_hat, sigma_hat, lumens_of(READING, ARMS), watch=recorder)
    positions = {0: (0, 1), 1: (1, 1), 2: (2, 1), 3: (1, 0), 4: (2, 2), 5: (3, 1), 6: (3, 2), 7: (4, 1)}
    frames = [flux_frame(state, positions, EDGES, TRUE, size=(3, 2.25), dpi=50) for state in recorder.states]
    assert all(frame.shape == frames[0].shape and frame.ndim == 3 for frame in frames)
    path = animate([{"frame": frame} for frame in frames], lambda state: state["frame"], str(tmp_path / "fit.gif"), fps=4)
    assert os.path.getsize(path) > 1000
    from PIL import Image
    with Image.open(path) as film:
        # PIL folds identical consecutive frames into one with their durations added: the held last frame shows
        # as one frame that lasts seven ticks
        assert film.n_frames >= len(frames) - 1
        film.seek(film.n_frames - 1)
        assert film.info["duration"] >= 7 * 250 - 1
