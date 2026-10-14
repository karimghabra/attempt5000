"""Acceptance step 5: spline_to_graph and the render-to-score pipeline (notebook cell `render-functions`), with the
tests Part III of the plan names for them."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import definitions  # noqa: E402


@pytest.fixture(scope="module")
def nb():
    return definitions()


def spline(start, end, radius=2.0):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = int(np.ceil(np.hypot(*(end - start)))) + 1
    points = start + np.linspace(0, 1, count)[:, None] * (end - start)
    return {"x": points[:, 0], "y": points[:, 1], "radius": np.full(count, radius)}


def degrees(view):
    return sorted(len(vertex["ends"]) for vertex in view["vertices"])


def test_two_ends_close_together_are_one_vertex_of_two_arms(nb):
    view = nb["spline_to_graph"]([spline((0, 50), (40, 50)), spline((41, 50), (90, 50))])
    assert degrees(view) == [1, 1, 2] and len(view["edges"]) == 2


def test_an_end_on_a_body_splits_it_into_a_t(nb):
    view = nb["spline_to_graph"]([spline((0, 50), (100, 50)), spline((50, 51), (50, 100))])
    assert degrees(view) == [1, 1, 1, 3] and len(view["edges"]) == 3


def test_two_bodies_that_cross_make_a_vertex_of_four_arms(nb):
    view = nb["spline_to_graph"]([spline((0, 50), (100, 50)), spline((50, 0), (50, 100))])
    assert degrees(view) == [1, 1, 1, 1, 4] and len(view["edges"]) == 4


def test_a_glancing_contact_makes_no_vertex(nb):
    view = nb["spline_to_graph"]([spline((0, 50), (100, 50)), spline((10, 45), (90, 55))])  # 7 degrees apart, crossing at x = 50
    assert degrees(view) == [1, 1, 1, 1]


def test_a_vessel_beside_a_longer_one_is_a_twin_and_no_vertex(nb):
    view = nb["spline_to_graph"]([spline((0, 50), (100, 50), 3.0), spline((30, 54), (70, 54), 2.0)])
    assert view["twins"] == [1] and degrees(view) == [1, 1, 1, 1]


def test_a_spline_that_crosses_itself_makes_a_loop(nb):
    # a figure of eight (Gerono's lemniscate) that crosses itself at right angles in its middle, its ends 16 px apart
    t = np.linspace(0.2, 2 * np.pi - 0.2, 4000)
    x, y = 60 + 40 * np.cos(t), 60 + 40 * np.sin(t) * np.cos(t)
    along = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    steps = np.arange(0, along[-1], 1.0)
    curve = {"x": np.interp(steps, along, x), "y": np.interp(steps, along, y), "radius": np.full(len(steps), 1.5)}
    view = nb["spline_to_graph"]([curve])
    assert view["cycles"] == 1 and degrees(view) == [1, 1, 4]


def test_spline_to_graph_finds_vessel_graphs_junctions_on_a_synthetic_burst(nb):
    # the plan's test: spline_to_graph's graph equals vessel_graph's where no rule differs. On section 37's pieces,
    # which end at their fork pixels; section 29's splines stop a pixel or two short of their forks (cut without
    # to_forks), so no k ends lie within RIDGE_JOIN there. Where several junctions lie within 30 px of each other the
    # two group them differently (the core rule against junction_candidates); every junction with no other within
    # 30 px has to be a vertex of three or more arms.
    found, isolated = 0, 0
    for seed in (1, 2):
        burst = nb["synthetic_burst_35"](seed)
        front = nb["ridge_front"](burst)
        splines, _ = nb["cyclic_pieces"](front, burst["valid"])
        graph = nb["vessel_graph"](splines, burst["valid"], burst["mean"])
        view = nb["spline_to_graph"](splines)
        junctions = np.array([(node["x"], node["y"]) for node in graph["nodes"] if not node.get("gone") and len(node.get("arms", [])) >= 3])
        vertices = np.array([(v["x"], v["y"]) for v in view["vertices"] if len(v["ends"]) >= 3]).reshape(-1, 2)
        for place in junctions:
            others = np.hypot(*(junctions - place).T)
            if np.sort(others)[1] > 30:
                isolated += 1
                found += np.min(np.hypot(*(vertices - place).T)) <= 5
    assert isolated >= 5 and found == isolated


def test_a_straight_vessel_is_drawn_as_its_blurred_chord(nb):
    edge = spline((10, 40), (110, 40), 3.0)
    box, density = nb["draw_lumen"]([edge], [0.03], 1.5, (80, 120), radii=[np.full(len(edge["x"]), 4.0)])
    profile = density[:, 60 - box[1]]
    rows = np.arange(box[0], box[2]) - 40.0
    fine = np.arange(-200, 201) / 10.0  # the chord on a fine grid, its transmittance blurred
    kernel = np.exp(-0.5 * (np.arange(-60, 61) / 10.0 / 1.5) ** 2)
    transmitted = np.convolve(np.exp(-0.03 * 2 * np.sqrt(np.maximum(16 - fine ** 2, 0))), kernel / kernel.sum(), mode="same")
    # pixel-centre sampling of a vessel of R >= 2 before the blur: within 5% of the peak
    assert np.max(np.abs(profile - np.interp(rows, fine, -np.log(transmitted)))) < 0.05 * profile.max()


def test_the_thin_vessels_kappa_is_recovered_after_the_thick_one_is_taken_off(nb):
    # a thick vessel (R = 5) and a thin one (R = 2) crossing, two lumens, both at kappa 0.025, blurred by 1.5 px
    lumens = [[(nb["line_points"]((5, 80), (155, 80)), 5.0)], [(nb["line_points"]((80, 5), (80, 155)), 2.0)]]
    picture = nb["render_check"](lumens, 1.5, 3)
    edges = []
    for points, radius in ((lumens[0][0][0], 5.0), (lumens[1][0][0], 2.0)):
        half = np.sqrt((nb["CYLINDER"] * radius) ** 2 + 1.5 ** 2)
        edges.append({"x": points[:, 0], "y": points[:, 1], "radius": np.full(len(points), half)})
    kappa, radii, windows, _ = nb["kappa_passes"](edges, np.array([0, 1]), picture["density"], 1.5, [(0.0, 0.0), (0.0, 0.0)])
    thin = windows[1]
    near = np.abs(np.asarray(thin[0]) - 75) < 25  # the windows near the crossing, where the thick vessel's density lies too
    assert near.any()
    assert np.median(thin[1][near]) == pytest.approx(nb["SYNTH_ABSORB"], rel=0.3)
    assert np.median(windows[0][1]) == pytest.approx(nb["SYNTH_ABSORB"], rel=0.2)


def test_the_windowed_rescore_equals_the_full_one_inside_the_window(nb):
    rng = np.random.default_rng(0)
    contrast = cv2_blur(nb, rng.normal(0, 1, (400, 520)).astype(np.float32))
    valid = np.ones(contrast.shape, dtype=bool)
    levels = np.ones(len(nb["HYPER_WIDTHS"]))
    for place in (0, 4, 6, 7):
        full = nb["channel_answers"](contrast, valid, levels, place)
        inner, part = nb["rescore_window"](contrast, valid, levels, place, (140, 200, 260, 320))
        assert np.abs(full[:, inner[0]:inner[2], inner[1]:inner[3]] - part).max() <= 0.005


def cv2_blur(nb, picture):
    return nb["blur"](picture, 1.0) * 5


def test_the_generators_truth_agrees_with_its_data(nb):
    # the plan's measurement: the generator's own noiseless truth, scored with the data's constants, finds the data's
    # ridges and is found in them at 0.996 to 0.999 (measured here 0.993 to 0.9995 on bursts 1 and 2)
    burst = nb["synthetic_burst_35"](2)
    frame = nb["scoring_frame"](burst["mean"], burst["valid"], burst["quiet"], noise=nb["robust_sd"](burst["mean"][burst["quiet"]]))
    data, _ = nb["ridges_of"](nb["to_contrast"](burst["mean"], frame), burst["valid"], frame["levels"])
    truth, _ = nb["ridges_of"](nb["to_contrast"](burst["layers"].sum(axis=0), frame), burst["valid"], frame["levels"])
    agreement = nb["self_consistency"](truth, data, burst["valid"])
    assert all(found >= 0.99 for found, _, _ in agreement.values())


def test_the_topology_residual_shows_a_missing_vessel_and_a_stub(nb):
    lumens = [[(nb["line_points"]((5, 60), (155, 60)), 3.0)], [(nb["line_points"]((5, 110), (155, 110)), 3.0)]]
    picture = nb["render_check"](lumens, 1.5, 4)
    frame = nb["scoring_frame"](picture["density"], picture["valid"], picture["quiet"], noise=nb["robust_sd"](picture["density"][picture["quiet"]]))
    # the model has the first vessel only, and a stub where there is nothing
    half = np.sqrt((nb["CYLINDER"] * 3) ** 2 + 1.5 ** 2)
    model = [{"x": np.arange(5, 156, dtype=float), "y": np.full(151, 60.0), "radius": np.full(151, half)},
             {"x": np.full(30, 120.0), "y": np.arange(10, 40, dtype=float), "radius": np.full(30, half)}]
    render, _ = nb["render_graph"](model, np.array([0, 1]), [np.full(151, 0.025), np.full(30, 0.025)], 1.5, picture["density"].shape,
                                   radii=[np.full(151, 3.0), np.full(30, 3.0)])
    zero = np.zeros((nb["N_ORIENT"], len(nb["HYPER_WIDTHS"])))
    positive, _, negative, _ = nb["topology_residual"](nb["to_contrast"](picture["density"], frame), nb["to_contrast"](render, frame),
                                                       picture["valid"], frame["levels"], zero)
    assert positive[110, 40:120].mean() > nb["T_HIGH"]           # the missing vessel: a positive ridge
    assert negative[15:35, 120].mean() > nb["T_HIGH"]            # the stub: a negative one
    assert positive[60, 40:120].mean() < nb["T_HIGH"] and negative[60, 40:120].mean() < nb["T_HIGH"]  # the right line: neither
