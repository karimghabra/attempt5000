import numpy as np
import pytest

from vascnet import Config
from vascnet.config import PEAK_RESPONSE_PER_OD, SIGMA_PER_RADIUS
from vascnet.cortex import feedforward
from vascnet.kernels import association_field, line_pair, orientations


def _line_image(theta, radius=2.0, size=97, od=0.5, noise=0.01):
    """Bright (OD) cylindrical line through the centre at orientation theta.
    A little noise is added: the z-scoring needs a measurable noise floor."""
    y, x = np.mgrid[:size, :size] - size // 2
    v = -x * np.sin(theta) + y * np.cos(theta)
    img = od / radius * np.sqrt(np.clip(radius**2 - v**2, 0, None))
    img += np.random.default_rng(0).normal(0, noise, img.shape)
    return img.astype(np.float32)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0])
def test_kernels_have_zero_dc_and_correct_symmetry(sigma):
    even, odd, front = line_pair(sigma, 2.5, 0.3)
    assert abs(even.sum()) < 1e-5
    assert abs(odd.sum()) < 1e-5
    np.testing.assert_allclose(even, even[::-1, ::-1], atol=1e-7)   # point-symmetric
    np.testing.assert_allclose(odd, -odd[::-1, ::-1], atol=1e-7)    # antisymmetric
    np.testing.assert_allclose(front + front[::-1, ::-1], even, atol=1e-7)  # halves sum


def test_orientation_tuning_peaks_at_line_orientation():
    cfg = Config(n_orient=16)
    for k in (0, 3, 7, 12):
        theta = orientations(16)[k]
        sc = feedforward(_line_image(theta), cfg)
        c = sc.z.shape[1] // 2
        assert int(np.argmax(sc.z[:, c, c])) == k


def test_scale_selection_recovers_radius():
    cfg = Config(radii=(1.0, 1.8, 3.2, 5.6))
    for r in (1.8, 3.2):
        sc = feedforward(_line_image(0.4, radius=r), cfg)
        c = sc.z.shape[1] // 2
        k = int(np.argmax(sc.z[:, c, c]))
        est = sc.sigma[k, c, c] / SIGMA_PER_RADIUS
        assert abs(est - r) / r < 0.15


def test_line_response_is_width_invariant():
    """Matched-scale peak response depends on optical density, not width."""
    peaks = []
    for r in (1.8, 3.2, 5.6):
        sv = SIGMA_PER_RADIUS * r
        even, _, _ = line_pair(sv, 6.0, 0.0)
        img = _line_image(0.0, radius=r, size=161, od=1.0, noise=0.0)
        from scipy.ndimage import correlate
        peaks.append(correlate(img, even)[80, 80])
    peaks = np.array(peaks)
    assert np.all(np.abs(peaks - PEAK_RESPONSE_PER_OD) < 0.06)


def test_association_field_lobes_are_mirror_images():
    f = association_field(0.5, 0.2, 5.0, 20.0, 0.2, +1)
    b = association_field(0.5, 0.2, 5.0, 20.0, 0.2, -1)
    np.testing.assert_allclose(f, b[::-1, ::-1], atol=1e-6)
    assert f.sum() > 0
