"""Acceptance step 3: the flux network on the whole-vessel graphs (notebook cell `flux-network-functions`), with
`join_provenance` and `sigma_flux_by_kind` as Part III of the plan names them."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import definitions  # noqa: E402
from vascnet.fluxnet import lumens_of, solve_flux_network  # noqa: E402


@pytest.fixture(scope="module")
def nb():
    return definitions()


@pytest.fixture(scope="module")
def tree(nb):
    burst = nb["synthetic_burst_35"](1001)
    front = nb["ridge_front"](burst)
    read = nb["read_burst_ridges"](burst, front, consolidate=True, speeds="whole vessels")
    added, lines = nb["consolidated_extensions"](front["strength"], front["orientation"], front["mask"])
    return {"burst": burst, "front": front, "read": read, "added": added, "lines": lines,
            "provenance": nb["join_provenance"](read["graph"], added)}


def test_the_extensions_rebuild_section_29s_consolidation(nb, tree):
    front = tree["front"]
    assert np.array_equal(tree["lines"], nb["ridge_skeleton"](front["strength"], front["orientation"], front["mask"], consolidate=True))


def test_the_undone_joins_are_the_ones_section_32_counted(nb, tree):
    assert sum(join["provenance"] == "undone" for join in tree["provenance"]) == tree["read"]["graph"]["undone"] > 0


def test_extension_joins_lie_on_the_added_pixels(nb, tree):
    for join in tree["provenance"]:
        if join["provenance"] == "extension":
            y, x = np.nonzero(tree["added"])
            assert np.min(np.hypot(x - join["x"], y - join["y"])) <= 1.5


def test_a_crossings_two_lumens_are_set_aside_together(nb, tree):
    graph = tree["read"]["graph"]
    lumens, kinds, groups, overlaps, places = nb["flux_lumens"](graph, tree["read"]["junctions"], tree["provenance"])
    for kind, group, place in zip(kinds, groups, places):
        same = [k for k, g, p in zip(kinds, groups, places) if g == group]
        assert len(same) == (2 if kind == "crossing" else 1)
    assert not any(kind == "join, undone" for kind in kinds)  # an undone join has no balance


def test_the_truth_agrees_with_most_measured_directions(nb, tree):
    graph = tree["read"]["graph"]
    way, _ = nb["edge_truth"](graph, tree["burst"]["segments"])
    q_hat, _, _ = nb["edge_flux"](graph)
    measured = np.isfinite(q_hat) & (way != 0)
    assert measured.sum() >= 5 and np.mean(np.sign(q_hat[measured]) == way[measured]) >= 0.8


def test_sigma_flux_by_kind_measures_or_falls_back(nb):
    rng = np.random.default_rng(0)
    imbalances = {"bifurcation": list(rng.normal(0, 0.2, 40)), "through": [0.1, -0.1], "join, skeleton": [0.05]}
    by_kind = nb["sigma_flux_by_kind"](imbalances, fallback={"through": (0.3, 12, "measured")})
    assert by_kind["bifurcation"][2] == "measured" and by_kind["bifurcation"][0] == pytest.approx(0.2, abs=0.06)
    assert by_kind["through"] == (0.3, 3, "stands in")  # the joins are through-lumens of one vessel too
    assert by_kind["crossing"][0] == nb["SIGMA_FLUX"]


def test_lumen_imbalances_leave_out_the_unmeasured_and_the_impossible(nb):
    q_hat = np.array([10.0, 6.0, 4.0, np.nan, 3.0])
    lumens = [((0, -1), (1, 1), (2, 1)), ((0, -1), (3, 1)), ((1, 1), (4, 1))]  # balanced; one unmeasured; all out
    out = nb["lumen_imbalances"](lumens, ["bifurcation", "through", "through"], q_hat)
    assert out == {"bifurcation": [0.0]}


def test_a_softness_per_lumen_equal_everywhere_is_the_one_softness():
    edges_arms = {1: [(0, 1), (1, 0), (2, 0)], 2: [(1, 1), (3, 0)]}
    lumens = lumens_of({1: [[0, 1, 2]], 2: [[0, 1]]}, edges_arms)
    q_hat, sigma = np.array([10.0, 6.0, np.nan, 6.5]), np.array([1.0, 1.0, np.nan, 1.0])
    one = solve_flux_network(q_hat, sigma, lumens, sigma_flux=0.3)
    each = solve_flux_network(q_hat, sigma, lumens, sigma_flux=np.full(len(lumens), 0.3))
    assert np.allclose(one.q, each.q) and np.allclose(one.sigma, each.sigma)


def test_direction_scores_count_what_the_network_and_the_junctions_infer(nb, tree):
    graph = tree["read"]["graph"]
    by_kind = {kind: (nb["SIGMA_FLUX"], 0, "") for kind in ("through", "bifurcation", "crossing")}
    run = nb["run_flux_network"](graph, tree["read"]["junctions"], tree["provenance"], by_kind, 0.85, {}, tree["burst"]["mean"])
    way, _ = nb["edge_truth"](graph, tree["burst"]["segments"])
    scores = nb["direction_scores"](run, graph, way)
    assert scores["network right"] <= scores["network inferred"] <= scores["unmeasured"]
    assert scores["junctions right"] <= scores["junctions inferred"] <= scores["unmeasured"]
    assert scores["flips right"] + scores["flips wrong"] == int((run["solution"].flipped & np.isfinite(run["q_hat"]) & (way != 0)).sum())


def test_join_features_tell_a_straight_join_from_an_l(nb):
    straight = {"x": np.arange(0, 20, dtype=np.float32), "y": np.full(20, 10.0, np.float32), "radius": np.full(20, 2.0, np.float32)}
    onward = {"x": np.arange(19, 40, dtype=np.float32), "y": np.full(21, 10.0, np.float32), "radius": np.full(21, 2.0, np.float32)}
    turning = {"x": np.full(20, 19.0, np.float32), "y": np.arange(10, 30, dtype=np.float32), "radius": np.full(20, 4.0, np.float32)}
    graph = {"edges": [straight, onward, turning]}
    density = np.ones((50, 50))
    join = lambda second: {"ends": ((0, 1), (second, 0)), "provenance": "skeleton"}
    on, off = nb["join_features"](graph, join(1), density), nb["join_features"](graph, join(2), density)
    assert on["turn"] == pytest.approx(0, abs=1) and off["turn"] == pytest.approx(90, abs=1)
    assert on["width"] == pytest.approx(0) and off["width"] == pytest.approx(np.log(2))


def test_p_comes_from_the_three_arm_junctions_only_where_there_are_enough(nb):
    few = {"three arms": 2, "all in or all out": 1, "sides": 40, "sides that agree": 38}
    many = {"three arms": 40, "all in or all out": 2, "sides": 40, "sides that agree": 38}
    assert nb["chosen_p"](few) == pytest.approx(nb["reliability"](few)[1])
    assert nb["chosen_p"](many) == pytest.approx(nb["reliability"](many)[0])
