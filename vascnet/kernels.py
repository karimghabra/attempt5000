"""Hand-designed receptive fields.

* ``line_pair``          even/odd quadrature pair: an elongated second-derivative
                         (line / "bar" detector, V1 even simple cell) and first-
                         derivative (edge detector, odd simple cell) of an
                         anisotropic Gaussian.
* ``association_field``  lateral (horizontal) connectivity between orientation
                         columns: co-circular, collinear facilitation in the shape
                         measured psychophysically by Field, Hayes & Hess (1993),
                         split into its two lobes so a downstream unit can
                         require support on *both* sides (Grossberg's bipole cell).

Coordinates: x = column, y = row (pointing down); orientation theta is measured
from +x toward +y and lives in [0, pi). The along-vessel unit vector is
(cos theta, sin theta), the across-vessel normal is (-sin theta, cos theta).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def orientations(n: int) -> np.ndarray:
    return np.arange(n) * np.pi / n


def wrap_half_pi(a):
    """Wrap angles to [-pi/2, pi/2) (orientation differences)."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def _grid(half: int):
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    return x, y


@lru_cache(maxsize=512)
def line_pair(sigma_v: float, elongation: float, theta: float, truncate: float = 3.0):
    """Scale-normalized even (line), odd (edge) and half-even kernels.

    even  = -sigma_v^2 d2G/dv2 : positive at the centre of a bright line; its
            peak response to a line of peak amplitude A is independent of the
            line width when sigma is matched (gamma = 1 normalization).
    odd   =  sigma_v dG/dv     : zero at a line centre, maximal on an edge.
    front = even restricted to the half-plane ahead of the centre (u > 0).
            even - front is the back half. A line running *through* the
            receptive field drives both halves equally; a junction blob or a
            line end drives one - the end-stopped / bipole simple cell.
    """
    sigma_u = elongation * sigma_v
    half = int(np.ceil(truncate * max(sigma_u, sigma_v)))
    x, y = _grid(half)
    c, s = np.cos(theta), np.sin(theta)
    u = x * c + y * s
    v = -x * s + y * c
    g = np.exp(-0.5 * (u / sigma_u) ** 2 - 0.5 * (v / sigma_v) ** 2)
    g /= g.sum()
    even = (1.0 - (v / sigma_v) ** 2) * g
    even -= g * even.sum()  # exact zero DC after discretization
    odd = (v / sigma_v) * g
    odd -= odd.mean()
    front = even * ((u > 1e-9) + 0.5 * (np.abs(u) <= 1e-9))
    for k in (even, odd, front):
        k.setflags(write=False)
    return even.astype(np.float32), odd.astype(np.float32), front.astype(np.float32)


@lru_cache(maxsize=1024)
def association_field(
    theta: float,
    dtheta: float,
    sigma_r: float,
    sigma_beta_deg: float,
    sigma_kappa: float,
    lobe: int,
):
    """Weights from senders tuned to theta+dtheta onto a receiver tuned to theta.

    For a sender at offset d = r (cos phi, sin phi):
      beta      = phi - theta (wrapped)       angle off the receiver's axis
      theta_cc  = theta + 2 beta              orientation that is co-circular
                                              (lies on a common circle) with
                                              the receiver
    w = exp(-r^2/2 sigma_r^2) exp(-beta^2/2 sigma_beta^2)
        exp(-(dtheta - 2 beta)^2 / 2 sigma_kappa^2)

    ``lobe`` = +1 keeps senders ahead of the receiver along theta, -1 behind.
    The field is point-symmetric between lobes, so convolution == correlation.
    Each lobe is normalized so that a uniform field of value z in the
    receiver's own channel returns z.
    """
    half = int(np.ceil(3 * sigma_r))
    x, y = _grid(half)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    beta = wrap_half_pi(phi - theta)
    sb = np.deg2rad(sigma_beta_deg)
    delta = wrap_half_pi(dtheta - 2 * beta)
    w = (
        np.exp(-0.5 * (r / sigma_r) ** 2)
        * np.exp(-0.5 * (beta / sb) ** 2)
        * np.exp(-0.5 * (delta / sigma_kappa) ** 2)
    )
    u = x * np.cos(theta) + y * np.sin(theta)
    w[r < 0.5] = 0.0
    w *= (u > 1e-9) if lobe > 0 else (u < -1e-9)
    # normalization reference: same field, receiver's own channel
    ref = (
        np.exp(-0.5 * (r / sigma_r) ** 2)
        * np.exp(-0.5 * (beta / sb) ** 2)
        * np.exp(-0.5 * (wrap_half_pi(-2 * beta) / sigma_kappa) ** 2)
    )
    ref[r < 0.5] = 0.0
    ref *= (u > 1e-9) if lobe > 0 else (u < -1e-9)
    w = w / ref.sum()
    w.setflags(write=False)
    return w.astype(np.float32)
