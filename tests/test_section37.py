"""Section 37 of the plan: the clean cyclic graph's units (notebook cell `cyclic-graph-functions`), with the tests
Part III of the plan names for them."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import definitions  # noqa: E402


@pytest.fixture(scope="module")
def nb():
    return definitions()


@pytest.fixture(scope="module")
def burst(nb):
    return nb["synthetic_burst_35"](1, second=True)


@pytest.fixture(scope="module")
def front(nb, burst):
    return nb["ridge_front"](burst)


@pytest.fixture(scope="module")
def joined(nb, front):
    return nb["joined_ridges"](front["strength"], front["orientation"], front["mask"])


def straight(nb, joined, front, burst, places):
    """The pieces that lie on one true segment (90% of their points within 2 px of it) and pass no true junction
    within 15 px: the generator's smooth vessels. With the given sub-pixel places."""
    import scipy.spatial
    owner = np.concatenate([np.full(len(segment["points"]), number) for number, segment in enumerate(burst["segments"])])
    tree = scipy.spatial.cKDTree(np.concatenate([segment["points"] for segment in burst["segments"]]))
    junctions = np.array([[junction["x"], junction["y"]] for junction in burst["junctions"]]).reshape(-1, 2)
    lines = nb["fine_lines"](joined["skeleton"], front["strength"], front["orientation"], front["half_width"], places)
    kept = []
    for line in lines:
        points = np.column_stack([line["x"], line["y"]])
        distance, nearest = tree.query(points)
        on = owner[nearest[distance < 2]]
        far = (np.hypot(points[:, None, 0] - junctions[None, :, 0], points[:, None, 1] - junctions[None, :, 1]) > 15).all()
        if len(points) >= 20 and len(on) > 0.9 * len(points) and np.mean(on == np.bincount(on).argmax()) > 0.95 and far:
            kept.append(line)
    return kept


def test_joined_ridges_keep_what_the_consolidation_drops(nb, front, joined):
    import cv2
    consolidated = nb["ridge_skeleton"](front["strength"], front["orientation"], front["mask"], consolidate=True)
    distance = cv2.distanceTransform((~joined["skeleton"]).astype(np.uint8), cv2.DIST_L2, 5)
    assert np.mean(distance[consolidated] <= 1.5) > 0.99  # everything the consolidation keeps is there
    assert joined["skeleton"].sum() > consolidated.sum()  # and what it drops is kept


def test_added_pixels_are_the_extensions(nb, front, joined):
    extended = nb["extend_ends"](joined["closed"], front["strength"])
    assert np.array_equal(joined["added"], extended & ~joined["closed"])


def test_offsets_lower_the_fit_residual(nb, burst, front, joined):
    places = nb["crest_places"](front["scores"], front["orientation"], burst["valid"])
    none = (np.full(burst["mean"].shape, np.nan, np.float32),) * 2
    residual = {}
    for name, given in (("centres", none), ("crests", places)):
        lines = straight(nb, joined, front, burst, given)
        residual[name] = np.median([nb["spline_vessel"]({**line, "flow share": 0.0}, front["flow_strength"])["residual"]
                                    for line in lines])
    assert residual["crests"] < residual["centres"]
    assert residual["crests"] < 0.28  # the staircase's median, as the plan quotes it


def test_refine_knots_splits_a_corner(nb):
    rng = np.random.default_rng(0)
    arm = np.arange(0, 40, 1.0)
    corner = np.r_[np.column_stack([arm, np.zeros_like(arm)]), np.column_stack([np.full(39, 39.0), arm[1:]])]
    corner += rng.normal(0, 0.1, corner.shape)
    pieces = nb["refine_knots"]({"x": corner[:, 0], "y": corner[:, 1]})
    assert len(pieces) == 2
    assert abs(pieces[0][1] - 39) <= 2  # split at the corner


def test_refine_knots_keeps_a_smooth_arc(nb):
    angle = np.linspace(0, 0.8, 80)
    arc = np.column_stack([100 * np.cos(angle), 100 * np.sin(angle)])
    arc = arc[np.r_[0, np.flatnonzero(np.cumsum(np.hypot(*np.diff(arc, axis=0).T)) >= 0)]]
    assert nb["refine_knots"]({"x": arc[:, 0], "y": arc[:, 1]}) == [(0, len(arc) - 1, nb["KNOT_SPACING"])]


def test_generator_vessels_keep_12_px_knots(nb):
    # the generator's smooth vessels, over eight bursts: nearly all keep 12 px knots (measured on 16 bursts: 24 of 25;
    # the one split is at a bend of more than FUSE_STRAIGHT in the ridge, not in the vessel)
    kept, total = 0, 0
    for seed in range(1, 9):
        burst = nb["synthetic_burst"](seed)
        front = nb["ridge_front"](burst)
        joined = nb["joined_ridges"](front["strength"], front["orientation"], front["mask"])
        places = nb["crest_places"](front["scores"], front["orientation"], burst["valid"])
        for line in straight(nb, joined, front, burst, places):
            total += 1
            kept += nb["refine_knots"](line) == [(0, len(line["x"]) - 1, nb["KNOT_SPACING"])]
    assert total >= 8 and kept / total >= 0.9


def test_edge_repeatability_averages_to_found_again(nb, burst, front):
    vessels, joined = nb["cyclic_pieces"](front, burst["valid"])
    second = burst["second"]
    other = nb["ridge_front"](second)
    other_lines = nb["joined_ridges"](other["strength"], other["orientation"], other["mask"])["skeleton"]
    found, chance, above = nb["edge_repeatability"](vessels, burst["valid"], other_lines, second["valid"], second["to second"])
    counted = np.isfinite(found)
    weights = np.array([len(vessel["x"]) for vessel in vessels])[counted]
    drawn = nb["draw_chains"](vessels, burst["valid"].shape) >= 0
    whole = nb["found_again"](drawn, burst["valid"], other_lines, second["valid"], second["to second"])
    assert np.average(found[counted], weights=weights) == pytest.approx(whole, abs=0.03)
    assert np.nanmean(chance) < np.nanmean(found)


def test_near_pairs_finds_every_pair(nb):
    rng = np.random.default_rng(1)
    x, y, reach = rng.uniform(0, 50, 200), rng.uniform(0, 50, 200), rng.uniform(1, 9, 200)
    i, j, distance = nb["near_pairs"](x, y, reach)
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    expected = {(a, b) for a, b in zip(*np.nonzero(d <= np.maximum(reach[:, None], reach[None]))) if a != b}
    assert set(zip(i.tolist(), j.tolist())) == expected


def line_edge(x, y, radius=2.0, loose=(False, False)):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    theta = np.full(len(x), np.arctan2(y[-1] - y[0], x[-1] - x[0]) % np.pi, dtype=np.float32)
    return {"x": x, "y": y, "theta": theta, "radius": np.full(len(x), radius, np.float32),
            "length": float(np.hypot(x[-1] - x[0], y[-1] - y[0])), "loose": loose}


def test_twins_and_stubs(nb):
    valid = np.ones((200, 200), dtype=bool)
    long_line = line_edge(np.arange(20, 180), np.full(160, 100.0), radius=4.0)
    twin = line_edge(np.arange(60, 120), np.full(60, 104.0), radius=2.0, loose=(True, True))
    stub = line_edge(np.full(8, 60.0), np.arange(97, 89, -1), radius=2.0, loose=(False, True))  # inside the tube
    branch = line_edge(np.full(60, 150.0), np.arange(96, 36, -1), radius=2.0, loose=(False, True))  # far outside it
    kinds = nb["twins_and_stubs"]([long_line, twin, stub, branch], 1.0, valid)
    assert kinds == ["plain", "twin", "stub", "plain"]


def test_closing_joins_finds_a_join_of_two_arms(nb):
    ridges = np.zeros((40, 80), dtype=bool)
    ridges[20, 5:38] = True
    ridges[20, 40:75] = True  # a gap of two pixels
    lines = ridges.copy()
    lines[20, 38:40] = True
    joins, labels = nb["closing_joins"](ridges, lines)
    assert len(joins) == 1 and joins[0]["two arms"] and len(joins[0]["pieces"]) == 2


def test_calibration_orders_a_separating_term(nb):
    rng = np.random.default_rng(2)
    truth = rng.random(400) < 0.6
    rows = [{"strength": float(rng.normal(2.0 if right else 1.0, 0.5)), "kind": "plain" if right else rng.choice(["plain", "twin"])}
            for right in truth]
    model = nb["calibrate_terms"](rows, truth, terms=("strength", "kind"))
    ratio = model["terms"]["strength"]["ratio"][:-1]  # the last bin is for NaN
    assert np.all(np.diff(ratio[np.isfinite(ratio)]) > -0.3)  # rising with the strength, give or take a bin's noise
    assert model["terms"]["kind"]["ratio"][model["terms"]["kind"]["cuts"].index("twin")] < 0
    confidence, _ = nb["edge_confidence"](rows, model, terms=("strength", "kind"))
    assert nb["told_apart"](confidence[truth], confidence[~truth]) > 0.85
    nan_row = [{"strength": float("nan"), "kind": "plain"}]
    assert np.isfinite(nb["edge_confidence"](nan_row, model, terms=("strength", "kind"))[0][0])
