"""Acceptance step 7: the recursion of section 39 (notebook cell `recursion-functions`): J, the moves it takes and
refuses, and the bridges it only reports."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import definitions  # noqa: E402

SHAPE = (200, 260)
BLUR = 1.5


@pytest.fixture(scope="module")
def nb():
    return definitions()


def piece(nb, start, end, radius=3.0):
    points = nb["line_points"](start, end)
    count = len(points)
    half = float(np.sqrt((nb["CYLINDER"] * radius) ** 2 + BLUR ** 2))
    theta = np.full(count, np.arctan2(*(np.asarray(end, float) - np.asarray(start, float))[::-1]) % np.pi)
    return {"x": points[:, 0].astype(np.float32), "y": points[:, 1].astype(np.float32), "radius": np.full(count, half, np.float32),
            "theta": theta.astype(np.float32), "strength": np.full(count, 10.0, np.float32), "flow share": 0.0,
            "length": float(count - 1), "loose": (True, True)}


def scene_of(nb, lumens, seed):
    picture = nb["render_check"](lumens, BLUR, seed, shape=SHAPE)
    frame = nb["scoring_frame"](picture["density"], picture["valid"], picture["quiet"],
                                noise=nb["robust_sd"](picture["density"][picture["quiet"]]))
    return nb["recursion_scene"](picture["density"], picture["valid"], frame), picture


def two_lines(nb):
    return [[(nb["line_points"]((10, 60), (250, 60)), 3.0)], [(nb["line_points"]((10, 140), (250, 140)), 3.0)]]


def test_the_price_counts_the_parameters_of_the_drawn_splines(nb):
    scene, _ = scene_of(nb, two_lines(nb), 1)
    pieces = [piece(nb, (10, 60), (250, 60)), piece(nb, (10, 140), (250, 140))]
    state = nb["fresh_state"](pieces, scene, BLUR)
    zero = np.zeros((nb["N_ORIENT"], len(nb["HYPER_WIDTHS"])))
    total, chi, price = nb["objective"](scene, state, zero, 2.0)
    assert total == pytest.approx(chi + price)
    assert price == pytest.approx(2.0 * scene["ln n"] * sum(nb["spline_parameters"](p) for p in pieces))
    state["drawn"][1] = False
    assert nb["objective"](scene, state, zero, 2.0)[2] == pytest.approx(price / 2)


def test_a_right_spline_is_not_removed(nb):
    scene, _ = scene_of(nb, two_lines(nb), 2)
    state = nb["fresh_state"]([piece(nb, (10, 60), (250, 60)), piece(nb, (10, 140), (250, 140))], scene, BLUR)
    zero = nb["round_zero"](scene, state)
    before = nb["objective"](scene, state, zero, 1.0)[0]
    after = nb["objective"](scene, nb["applied_move"](scene, state, {"kind": "remove", "piece": 1}), zero, 1.0)[0]
    assert after > before


def test_a_missing_vessel_is_added_and_a_false_spline_removed(nb):
    scene, picture = scene_of(nb, two_lines(nb), 3)
    # the model has the first vessel, not the second, and a spline across the empty middle where nothing is
    state = nb["fresh_state"]([piece(nb, (10, 60), (250, 60)), piece(nb, (200, 85), (200, 115))], scene, BLUR)
    # the false spline drawn as dark as the true vessel, as a fit on a noise ridge might draw it
    state["kappa"][1] = np.full(len(state["kappa"][1]), float(np.median(state["kappa"][0])))
    state["radii"][1] = np.full(len(state["radii"][1]), float(np.median(state["radii"][0])))
    state = nb["fresh_render"](scene, state)
    zero = np.zeros((nb["N_ORIENT"], len(nb["HYPER_WIDTHS"])))
    final, history = nb["recursion_rounds"](scene, state, zero, 1.0, rounds=2)
    taken = [move for move in history["moves"] if move["taken"]]
    kinds = {move["kind"] for move in taken}
    assert "remove" in kinds and kinds & {"add", "extend"}
    assert not final["drawn"][1] and final["drawn"][0]
    added = [p for p, origin in enumerate(final["origin"]) if origin in ("add", "extend") and final["drawn"][p]]
    on_second = [np.mean(np.abs(final["pieces"][p]["y"] - 140) <= 2) for p in added]
    assert on_second and max(on_second) > 0.9
    # J falls at every taken move, and the J tracked through the moves is the J of the final model
    values = [history["J"][0]] + [move["J"] for move in taken]
    assert all(b < a for a, b in zip(values[:-1], values[1:]))
    again = nb["fresh_render"](scene, final)
    assert nb["objective"](scene, again, zero, 1.0)[0] == pytest.approx(values[-1], rel=1e-3)


def test_bridges_are_reported_and_never_drawn(nb):
    lumens = [[(nb["line_points"]((10, 100), (120, 100)), 2.5)], [(nb["line_points"]((134, 100), (250, 100)), 2.5)]]
    scene, _ = scene_of(nb, lumens, 4)
    state = nb["fresh_state"]([piece(nb, (10, 100), (120, 100), 2.5), piece(nb, (134, 100), (250, 100), 2.5)], scene, BLUR)
    count = len(state["pieces"])
    bridges = nb["bridge_candidates"](scene, state, np.zeros(SHAPE, np.float32))
    assert len(bridges) == 1 and {bridges[0]["ends"][0][0], bridges[0]["ends"][1][0]} == {0, 1}
    assert len(state["pieces"]) == count and state["drawn"].all()


def test_the_core_filter_proposes_one_lumen_at_an_anastomosis(nb):
    a, b = nb["line_points"]((30, 100), (230, 100)), nb["line_points"]((130, 10), (130, 190))
    for composition, lumens in (("two vessels", [[(a, 3.0)], [(b, 3.0)]]), ("one lumen", [[(a, 3.0), (b, 3.0)]])):
        scene, picture = scene_of(nb, lumens, 5)
        state = nb["fresh_state"]([piece(nb, (30, 100), (230, 100)), piece(nb, (130, 10), (130, 190))], scene, BLUR)
        proposed = nb["composition_candidates"](scene, state, picture["quiet"])
        flips = [c for c in proposed if c["kind"] == "one lumen"]
        assert (len(flips) == 1) == (composition == "one lumen"), (composition, [c["z"] for c in proposed])
