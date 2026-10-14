"""Section 35 of the plan: the generator's extensions (notebook cell `generator-functions`).

The gate of acceptance step 1 is that the old seeds are bit-identical and that every new seed has its counts. The
old seeds are pinned by `data/old_seed_digests.json`, made from the generator of section 28 before anything was
added (one SHA-256 per key of the burst). By default four of the 32 seeds are checked; with VASCNET_ALL_SEEDS=1 all
of them (section 35's cell in the notebook compares all 32 every time it runs).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import burst_digest, definitions  # noqa: E402

DIGESTS = json.loads((Path(__file__).parent / "data" / "old_seed_digests.json").read_text())
OLD_SEEDS = [int(seed) for seed in DIGESTS["seeds"]] if os.environ.get("VASCNET_ALL_SEEDS") else [1, 2, 1001, 1016]
OLD_SEGMENT_KEYS = ("points", "radius", "speed", "tree", "depth", "parent", "pixels", "along", "chord")
OLD_JUNCTION_KEYS = ("x", "y", "type", "radius", "segments", "places", "contrast", "visible", "ambiguous")


@pytest.fixture(scope="module")
def nb():
    return definitions()


@pytest.fixture(scope="module")
def trees(nb):
    return nb["synthetic_burst_35"](nb["TREE_SEEDS"][0], "trees", **nb["SYNTH_THINGS"])


@pytest.fixture(scope="module")
def plexus(nb):
    seed = nb["PLEXUS_SEEDS"][0]
    return {kind: nb["synthetic_burst_35"](seed, kind, **nb["SYNTH_THINGS"]) for kind in ("crossing plexus", "anastomotic plexus")}


def old_part(burst):
    """What section 28's burst holds, without the keys section 35 adds."""
    old = {key: burst[key] for key in DIGESTS["keys"]}
    old["segments"] = [{key: segment[key] for key in OLD_SEGMENT_KEYS} for segment in burst["segments"]]
    old["junctions"] = [{key: junction[key] for key in OLD_JUNCTION_KEYS if key in junction} for junction in burst["junctions"]]
    return old


@pytest.mark.parametrize("seed", OLD_SEEDS)
def test_section_28_generator_unchanged(nb, seed):
    assert burst_digest(nb["synthetic_burst"](seed), DIGESTS["keys"]) == DIGESTS["seeds"][str(seed)]


@pytest.mark.parametrize("seed", OLD_SEEDS)
def test_old_seeds_bit_identical_with_everything_off(nb, seed):
    assert burst_digest(old_part(nb["synthetic_burst_35"](seed)), DIGESTS["keys"]) == DIGESTS["seeds"][str(seed)]


def test_unknown_thing_refused(nb):
    with pytest.raises(ValueError):
        nb["synthetic_burst_35"](1, gapz=0.5)


def test_counts_recorded(nb, trees):
    counts = trees["counts"]
    assert set(counts) == set(nb["COUNTED"])
    assert counts["gaps"] == len(trees["gaps"]) and counts["terminations"] == len(trees["terminations"])
    assert counts["collinear overlaps"] == len(trees["overlaps"]) == nb["SYNTH_OVERLAPS"]
    assert counts["trees"] == 3 + nb["SYNTH_CAPILLARY_TREES"] + nb["SYNTH_OVERLAPS"]
    assert counts["capillary segments"] >= 2 * nb["SYNTH_CAPILLARY_TREES"]


def test_capillaries_have_radius_one_to_two(nb, trees):
    capillary = [segment["radius"] for segment in trees["segments"] if segment["tree"] >= 3 and "host" not in segment]
    assert capillary and min(capillary) >= 0.9 and max(capillary) <= 2.0


def test_every_junction_has_its_arms(nb, trees, plexus):
    for burst in (trees, *plexus.values()):
        for junction in burst["junctions"]:
            expected = {"bifurcation": 3, "crossing": 4, "anastomosis": 4}[junction["type"]]
            assert junction["n"] == expected == len(junction["limbs"])
            assert junction["lumens"] == (2 if junction["type"] == "crossing" else 1)
    complex_junction = {"type": "complex", "members": ["bifurcation", "crossover"]}
    assert nb["junction_arms"](complex_junction) == 7


def test_flux_conserved_at_every_node_of_one_lumen(trees, plexus):
    for burst in (trees, plexus["anastomotic plexus"]):
        for node in burst["nodes"]:
            if len(node["ends"]) >= 3:
                net = sum(burst["segments"][s]["flux"] * (1 if end == 0 else -1) for s, end in node["ends"])
                total = sum(abs(burst["segments"][s]["flux"]) for s, _ in node["ends"])
                assert abs(net) <= 1e-9 * total


def test_loops_make_cycles(nb):
    # the cycle rank of the graph of one lumen (edges - vertices + components) counts the loops
    for seed in nb["TREE_SEEDS"][:4]:
        burst = nb["synthetic_burst_35"](seed, "trees", **{**nb["SYNTH_PLAIN"], "loops": 1.0})
        used = {node for segment in burst["segments"] for node in (segment["start"], segment["end"])}
        parent = {node: node for node in used}

        def find(node):
            while parent[node] != node:
                node = parent[node]
            return node

        for segment in burst["segments"]:
            parent[find(segment["start"])] = find(segment["end"])
        components = len({find(node) for node in used})
        assert len(burst["segments"]) - len(used) + components == burst["counts"]["loops"]
        assert sum(junction["kind"] == "loop" for junction in burst["junctions"]) == burst["counts"]["loops"]
        if burst["counts"]["loops"]:
            return
    pytest.fail("no loop was closed in four seeds with the chance of a loop at 1")


def test_gaps_terminations_overlaps_recorded(nb, trees):
    for gap in trees["gaps"]:
        segment = trees["segments"][gap["segment"]]
        assert segment["gap"]["place"] == gap["place"]
        # the chord is faded in the middle of the gap, to SYNTH_GAP_DEPTH of what the same place has without it
        middle = segment["along"] == gap["place"]
        assert middle.any()
    for end in trees["terminations"]:
        assert nb["inside_frame"](np.array([end["x"], end["y"]]), 30)
    for overlap in trees["overlaps"]:
        hidden = trees["segments"][overlap["segment"]]
        host = trees["segments"][overlap["host"]]
        assert np.allclose(hidden["points"], host["points"][overlap["from"]:overlap["from"] + overlap["length"]])
        assert not any(set(junction["segments"]) == {overlap["segment"], overlap["host"]} for junction in trees["junctions"])


def test_gap_fades_the_chord(nb):
    pixels = np.column_stack([np.arange(200, dtype=float), np.full(200, 50.0)])
    points = np.column_stack([np.arange(200, dtype=float), np.full(200, 50.0)])
    plain = {"points": points, "radius": 3.0}
    gapped = {"points": points, "radius": 3.0, "gap": {"place": 100, "length": 24, "depth": 0.15}}
    for segment in (plain, gapped):
        nb["footprint"](segment, pixels, False)
    at = lambda segment, place: segment["chord"][segment["along"] == place][0]
    assert at(gapped, 100) == pytest.approx(0.15 * at(plain, 100))
    assert at(gapped, 130) == pytest.approx(at(plain, 130))


def test_fine_footprint_keeps_a_thin_vessels_mass(nb):
    # a straight vessel of radius 1 on a pixel's centre and between two pixels: the mass across it (pi R^2 x the
    # absorption) is kept within 3% with `fine`; read at the pixel centres it is not
    y, x = np.mgrid[:40, :60]
    pixels = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    masses = {}
    for fine in (False, True):
        for row in (20.0, 20.5):
            segment = {"points": np.column_stack([np.arange(60, dtype=float), np.full(60, row)]), "radius": 1.0}
            nb["footprint"](segment, pixels, fine)
            picture = np.zeros(40 * 60)
            picture[segment["pixels"]] = segment["chord"]
            masses[(fine, row)] = picture.reshape(40, 60)[:, 30].sum() / nb["SYNTH_ABSORB"]
    assert masses[(True, 20.0)] == pytest.approx(np.pi, rel=0.03)
    assert masses[(True, 20.5)] == pytest.approx(np.pi, rel=0.03)
    assert abs(masses[(False, 20.0)] - masses[(False, 20.5)]) > 0.3


def test_plexus_kinds_share_their_lines(nb, plexus):
    crossing, anastomotic = plexus["crossing plexus"], plexus["anastomotic plexus"]
    assert crossing["counts"]["projected crossings"] > 0 and crossing["counts"]["anastomoses"] == 0
    assert anastomotic["counts"]["anastomoses"] > 0 and anastomotic["counts"]["projected crossings"] == 0
    # the same lines: the anastomotic kind only moves a line's point by under a pixel to put both lines of a meeting
    # through one node, and cuts the lines into pieces there
    import cv2
    for first, second in ((crossing, anastomotic), (anastomotic, crossing)):
        distance = cv2.distanceTransform((~second["lines"]).astype(np.uint8), cv2.DIST_L2, 5)
        assert distance[first["lines"]].max() <= 1.5
    assert len({segment["tree"] for segment in anastomotic["segments"]}) == 1  # one lumen
    assert len({segment["tree"] for segment in crossing["segments"]}) == len(crossing["segments"])  # a depth each


def test_second_fixation_is_the_first_moved(nb, trees):
    second = trees["second"]
    low, high = nb["SYNTH_SHIFT"]
    assert all(low <= abs(value) <= high for value in second["shift"])
    # the first fixation's centerlines, moved by the map, lie on the second's (where it has data)
    import cv2
    moved = cv2.warpAffine(trees["lines"].astype(np.uint8), second["to second"][:2], trees["lines"].shape[::-1],
                           flags=cv2.INTER_NEAREST) > 0
    distance = cv2.distanceTransform((~second["lines"]).astype(np.uint8), cv2.DIST_L2, 5)
    inside = moved & (cv2.erode(second["valid"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    assert inside.any() and np.mean(distance[inside] <= 1.5) > 0.99
    assert not np.array_equal(second["frames"][0][second["valid"]], trees["frames"][0][second["valid"]])


def test_velocity_profile(nb, trees):
    for segment in trees["segments"]:
        blunt = (nb["SYNTH_BLUNT"] + 2) / nb["SYNTH_BLUNT"] if segment["radius"] > nb["SYNTH_FILE"] else 1.0
        assert segment["speed max"] == pytest.approx(segment["speed"] * blunt)
