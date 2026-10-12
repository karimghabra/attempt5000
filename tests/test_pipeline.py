import numpy as np
import pytest

from vascnet import Config, VesselPipeline
from vascnet.evaluate import centerline_metrics, junction_metrics
from vascnet.synth import Branch, Network, TreeStyle, centerline_truth, make_sample, render
from vascnet.video import VideoAnnotator


def _net(polylines, radii, shape=(160, 160), trees=None):
    """Build a hand-made network: each polyline is its own tree (own layer)."""
    branches = []
    for k, (pts, r) in enumerate(zip(polylines, radii)):
        pts = np.asarray(pts, float)
        # resample at 1 px
        seg = np.hypot(*np.diff(pts, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s[-1], 1.0)
        P = np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]
        branches.append(Branch(P, np.full(len(P), r), trees[k] if trees else k))
    n_trees = max(b.tree for b in branches) + 1
    styles = [TreeStyle(kappa=0.12, blur=0.6) for _ in range(n_trees)]
    return Network(shape, branches, styles, [])


def _run(net, seed=0, noise=0.01):
    img = render(net, seed=seed, noise=noise, illumination=0.1, texture=0.02)
    return VesselPipeline(Config())(img)


def _types(res):
    return sorted(j.type for j in res.junctions if j.type != "endpoint")


def test_straight_vessel_has_no_junctions_and_accurate_centerline():
    net = _net([[(-5, 80), (165, 80)]], [2.5])
    res = _run(net)
    assert _types(res) == []
    gt, _ = centerline_truth(net)
    m = centerline_metrics(res.centerline, gt)
    assert m["precision"] > 0.95 and m["recall"] > 0.95


def test_crossover_of_two_vessels():
    net = _net([[(-5, 30), (165, 130)], [(-5, 130), (165, 30)]], [2.5, 1.8])
    res = _run(net)
    assert _types(res) == ["crossover"]
    j = [j for j in res.junctions if j.type == "crossover"][0]
    assert abs(j.x - 80) < 4 and abs(j.y - 80) < 4
    assert len(j.pairs) == 2


def test_bifurcation_y():
    # one tree: parent from the left splitting into two daughters
    parent = [(-5, 80), (80, 80)]
    d1 = [(80, 80), (165, 40)]
    d2 = [(80, 80), (165, 125)]
    net = _net([parent, d1, d2], [3.0, 2.4, 2.0], trees=[0, 0, 0])
    res = _run(net)
    assert _types(res) == ["bifurcation"]
    j = [j for j in res.junctions if j.type == "bifurcation"][0]
    assert abs(j.x - 80) < 5 and abs(j.y - 80) < 5
    assert j.subtype == "Y"
    parent_arm = j.arms[j.features["parent"]]
    assert abs(np.cos(parent_arm.angle) + 1) < 0.15  # parent points left


def test_side_branch_t_is_bifurcation_not_crossover():
    main = [(-5, 70), (165, 70)]
    side = [(80, 70), (80, 165)]
    net = _net([main, side], [2.8, 1.6], trees=[0, 0])
    res = _run(net)
    assert _types(res) == ["bifurcation"]
    assert [j for j in res.junctions if j.type == "bifurcation"][0].subtype == "T"


def test_thin_vessel_under_thick_vessel_is_crossover():
    thick = [(-5, 80), (165, 80)]
    thin = [(30, -5), (130, 165)]
    net = _net([thick, thin], [4.5, 1.4])
    res = _run(net)
    assert "crossover" in _types(res)
    assert "bifurcation" not in _types(res)


@pytest.mark.parametrize("seed", [3, 11])
def test_synthetic_network_regression(seed):
    img, net = make_sample(seed=seed, shape=(384, 384))
    res = VesselPipeline(Config())(img)
    gt, _ = centerline_truth(net)
    cl = centerline_metrics(res.centerline, gt)
    jm = junction_metrics(res.junctions, net.junctions)
    assert cl["f1"] > 0.85
    assert jm["f1"] > 0.55


def test_annotations_are_json_serializable():
    import json

    img, net = make_sample(seed=2, shape=(256, 256))
    res = VesselPipeline()(img)
    json.dumps(res.annotations())


def test_video_tracks_keep_ids():
    net = _net([[(-5, 30), (165, 130)], [(-5, 130), (165, 30)]], [2.5, 1.8])
    frames = [render(net, seed=k, noise=0.03, illumination=0.1) for k in range(8)]
    vr = VideoAnnotator(tau=3, min_hits=2).run(frames)
    cons = vr.consensus()
    assert len(cons) == 1 and cons[0]["type"] == "crossover"
    ids = {t["id"] for f in vr.frames for t in f["tracks"]}
    assert len(ids) == 1


def test_bright_polarity_matches_dark_on_inverted_image():
    net = _net([[(-5, 30), (165, 130)], [(-5, 130), (165, 30)]], [2.5, 1.8])
    img = render(net, seed=0, noise=0.01, illumination=0.1, texture=0.02)
    dark = VesselPipeline(Config(polarity="dark"))(img)
    bright = VesselPipeline(Config(polarity="bright"))(1.0 - img)
    assert _types(bright) == _types(dark) == ["crossover"]
