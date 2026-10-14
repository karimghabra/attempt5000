"""Acceptance step 6: the core's matched filter of section 38b (notebook cell `core-functions`)."""

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
def spectrum(nb):
    return nb["noise_spectrum"](nb["CHECK_SHAPE"])


def test_two_compositions_differ_only_at_the_core(nb):
    lines = nb["core_configuration"]((3.0, 3.0))
    arms = [({"x": p[:, 0], "y": p[:, 1], "radius": np.full(len(p), 3.0)}, 0.025, r) for p, r in lines]
    two, one = nb["core_template"](arms, 1.5, nb["CHECK_SHAPE"])
    difference = two - one
    assert difference.min() > -1e-6 and difference.max() > 0
    rows, columns = np.nonzero(difference > 0.01 * difference.max())
    assert np.hypot(rows - 80, columns - 80).max() < 15  # only around the crossing


def test_z_has_unit_spread_on_noise(nb, spectrum):
    lines = nb["core_configuration"]((3.0, 3.0))
    arms = [({"x": p[:, 0], "y": p[:, 1], "radius": np.full(len(p), 3.0)}, 0.025, r) for p, r in lines]
    two, one = nb["core_template"](arms, 1.5, nb["CHECK_SHAPE"])
    rng = np.random.default_rng(7)
    zs = []
    for _ in range(60):
        texture = nb["cv2"].GaussianBlur(rng.normal(0, 1, nb["CHECK_SHAPE"]).astype(np.float32), (0, 0), nb["SYNTH_GRAIN"])
        noise = texture * nb["SYNTH_TEXTURE"] / texture.std() + rng.normal(0, nb["CHECK_NOISE"], nb["CHECK_SHAPE"])
        zs.append(nb["core_z"](one + noise, arms, 1.5, spectrum)[0])
    assert abs(np.mean(zs)) < 0.5 and 0.7 < np.std(zs) < 1.4


def test_z_tells_two_thick_vessels_from_one_lumen(nb, spectrum):
    lines = nb["core_configuration"]((3.0, 3.0))
    zs = {}
    for composition, lumens in (("two", [[line] for line in lines]), ("one", [lines])):
        picture = nb["render_check"](lumens, 1.5, 1)
        measured = [(points, float(np.sqrt((nb["CYLINDER"] * radius) ** 2 + 1.5 ** 2))) for points, radius in lines]
        arms = nb["fitted_arms"](picture["density"], measured, 1.5, 3.0 + 3.0 + nb["RING_MARGIN"])
        zs[composition] = nb["core_z"](picture["density"], arms, 1.5, spectrum)[0]
    assert zs["two"] >= 3 and abs(zs["one"]) < 3
