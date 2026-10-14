"""Acceptance step 4: the junction cells of section 36 (notebook cell `junction-cell-functions`)."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("psutil")
pytest.importorskip("IPython")

from notebook import definitions  # noqa: E402


@pytest.fixture(scope="module")
def nb():
    return definitions()


def test_the_ring_reads_the_neuron_of_the_radial_direction(nb):
    scores = np.zeros((nb["N_ORIENT"], 81, 81), dtype=np.float32)
    for step in range(1, 40):  # a line leaving the centre at 45 degrees, seen only by the neuron of orientation 45
        scores[4, int(round(40 + step * np.sin(np.pi / 4))), int(round(40 + step * np.cos(np.pi / 4)))] = 10.0
    phi, profile = nb["ring_profile"](scores, 40.0, 40.0, 10)
    assert np.degrees(phi[np.argmax(profile)]) == pytest.approx(45, abs=5)


def test_ring_arms_need_a_peak_above_t_high_and_merge_within_same_arm(nb):
    phi = np.arange(72) * 2 * np.pi / 72
    bump = lambda degrees, height: height * np.exp(-0.5 * (np.angle(np.exp(1j * (phi - np.radians(degrees)))) / np.radians(6)) ** 2)
    profile = bump(0, 8) + bump(40, 6) + bump(180, 3) + bump(270, 9) + bump(280, 8.5)
    arms = nb["ring_arms"](phi, profile)
    angles = sorted(round(np.degrees(angle)) % 360 for angle, _ in arms)
    assert len(angles) == 3 and angles[:2] == [0, 40]  # 180 never reaches T_HIGH
    assert 270 <= angles[2] <= 280  # the bumps at 270 and 280 overlap into one arm


def test_a_thin_vessel_crossing_a_thick_one_is_four_arms_on_the_smallest_ring(nb):
    lumens, (x, y), thin = nb["check_configurations"]()["thin crossing a thick vessel"]
    picture = nb["render_check"](lumens, 1.5, 1)
    cell = nb["junction_cell"](picture["scores"], picture["density"], x, y)
    smallest = cell["rings"][0]
    assert smallest["ring"] == nb["JUNCTION_RINGS"][0] and smallest["n"] == 4
    assert nb["cell_kind"](cell) == "four arms"


def test_two_arms_are_a_line_or_a_bend(nb):
    assert nb["cell_kind"]({"n": 2, "arms": [(0.0, 5.0), (np.pi + 0.2, 5.0)]}) == "line"
    assert nb["cell_kind"]({"n": 2, "arms": [(0.0, 5.0), (np.pi / 2, 5.0)]}) == "bend"


def test_plaid_cells_need_two_strong_orientations_far_apart(nb):
    scores = np.zeros((nb["N_ORIENT"], 3, 3), dtype=np.float32)
    scores[0, 0, 0] = scores[8, 0, 0] = 9.0   # 90 degrees apart
    scores[0, 1, 1] = scores[2, 1, 1] = 9.0   # 22.5 degrees apart
    plaid = nb["plaid_cells"](scores)
    assert plaid[0, 0] and not plaid[1, 1]


def test_the_thin_width_channel_answers_at_a_thin_lines_orientation(nb):
    lumens = [[(nb["line_points"]((80, 5), (80, 155)), 1.5)]]  # a thin vertical line
    picture = nb["render_check"](lumens, 1.0, 1)
    channels = nb["width_channels"](picture["contrast"], picture["valid"], picture["quiet"], nb["HYPER_WIDTHS"][0])
    assert int(np.argmax(channels[:, 80, 80])) == nb["N_ORIENT"] // 2  # orientation 90 degrees
    assert channels[:, 80, 80].max() > nb["T_HIGH"]


def test_the_most_arms_rule_prefers_more_arms_among_strong_rings(nb):
    cell = {"ring": 10, "n": 2, "rate": 90.0, "rings": [{"ring": 10, "n": 2, "rate": 90.0}, {"ring": 16, "n": 3, "rate": 40.0},
                                                         {"ring": 24, "n": 4, "rate": 3.0}]}
    assert nb["most_arms_ring"](cell)["ring"] == 16
