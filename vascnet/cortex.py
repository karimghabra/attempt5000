"""Stages 1-2 - a fixed-weight recurrent "cortex" that lifts the image into an
orientation score U(x, y, theta).

Stage 1 (feed-forward, V1 simple -> complex cells)
    For each scale s and orientation theta, an even (line) and odd (edge) simple
    cell see the optical-density image. A complex-cell-like unit keeps the line
    evidence that is *not* explained by an edge:  L = [E - alpha |O|]_+ .
    Scales compete (max over s of the scale-normalized response), which selects
    the vessel radius at every point; the winner is expressed in units of that
    scale's own noise floor (a z-score), so every downstream threshold is a
    signal-to-noise ratio rather than an intensity.

    The orientation score is the key representation: a crossover that is a
    single blob in the image becomes two separate ridges living in two
    different orientation channels, so crossing vessels stop interfering.

Stage 2 (recurrent horizontal connections, V1/V2)
    Iterated a few times with fixed kernels:
      * bipole association field  - collinear / co-circular facilitation. Used
        (a) multiplicatively: support boosts driven units (contour integration)
        and (b) additively but AND-gated across the two lobes, so gaps between
        collinear pieces are filled while vessel ends are not extended;
      * untuned cross-orientation inhibition - subtract the mean over
        orientations, which sharpens tuning and suppresses isotropic blobs;
      * divisive normalization - local contrast gain control.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from .config import Config, SIGMA_PER_RADIUS
from .fourier import FourierBank
from .kernels import association_field, line_pair, orientations


@dataclass
class OrientationScore:
    thetas: np.ndarray       # (N,)
    z: np.ndarray            # (N, H, W) feed-forward drive, background-referenced
    sigma: np.ndarray        # (N, H, W) selected filter scale per channel
    noise: np.ndarray        # (S,) noise std of the even response per scale
    U: np.ndarray | None = None  # (N, H, W) after recurrent interactions
    background: tuple[float, float] = (0.0, 1.0)  # (level, spread) of max_theta z

    @property
    def strength(self) -> np.ndarray:
        U = self.U if self.U is not None else self.z
        return U.max(0)

    @property
    def dominant(self) -> np.ndarray:
        U = self.U if self.U is not None else self.z
        return U.argmax(0)

    def radius_map(self) -> np.ndarray:
        """Vessel radius (px) implied by the selected scale at the dominant channel."""
        idx = self.dominant[None]
        sig = np.take_along_axis(self.sigma, idx, 0)[0]
        return sig / SIGMA_PER_RADIUS


# ---------------------------------------------------------------- stage 1
@lru_cache(maxsize=16)
def _line_bank(shape, sigma_v, elongation, n_orient) -> FourierBank:
    thetas = orientations(n_orient)
    ks = [line_pair(float(sigma_v), float(elongation), float(t)) for t in thetas]
    return FourierBank([k[0] for k in ks] + [k[1] for k in ks] + [k[2] for k in ks], shape)


def _robust_std(a: np.ndarray) -> float:
    sub = a[..., ::2, ::2].ravel()
    med = np.median(sub)
    return float(1.4826 * np.median(np.abs(sub - med))) + 1e-12


def feedforward(od: np.ndarray, cfg: Config) -> OrientationScore:
    N = cfg.n_orient
    shape = od.shape
    sigmas = np.asarray(cfg.sigmas, dtype=np.float64)
    S = len(sigmas)
    Ls = np.empty((S, N) + shape, dtype=np.float32)
    noises = np.empty(S)
    for s, sv in enumerate(sigmas):
        bank = _line_bank(shape, float(sv), cfg.elongation, N)
        resp = bank.apply(bank.spectrum(od))
        E, O, Ef = resp[:N], resp[N : 2 * N], resp[2 * N :]
        noises[s] = _robust_std(E)
        asym = np.abs(2 * Ef - E)  # |front - back|
        Ls[s] = np.maximum(E - cfg.edge_suppression * np.abs(O) - cfg.end_stopping * asym, 0.0)
    # scale competition: winner by scale-normalized amplitude
    sidx = Ls.argmax(0)
    best = np.take_along_axis(Ls, sidx[None], 0)[0]
    z = best / noises[sidx].astype(np.float32)
    # sub-scale estimate: parabola through the winner and its neighbours in
    # log(sigma) (radius is otherwise quantized to the bank's scales)
    logs = np.log(sigmas)
    lsig = logs[sidx]
    if S >= 3:
        k = np.clip(sidx, 1, S - 2)
        y0 = np.take_along_axis(Ls, (k - 1)[None], 0)[0]
        y1 = np.take_along_axis(Ls, k[None], 0)[0]
        y2 = np.take_along_axis(Ls, (k + 1)[None], 0)[0]
        x0, x1, x2 = logs[k - 1], logs[k], logs[k + 1]
        d0, d2 = x0 - x1, x2 - x1
        denom = d0 * d2 * (d0 - d2)
        a = ((y0 - y1) * d2 - (y2 - y1) * d0) / np.where(denom == 0, 1, denom)
        b = ((y2 - y1) * d0**2 - (y0 - y1) * d2**2) / np.where(denom == 0, 1, denom)
        interior = (sidx >= 1) & (sidx <= S - 2) & (a < 0)
        vert = x1 - b / np.where(a == 0, -1, 2 * a)
        lsig = np.where(interior, np.clip(vert, x0, x2), lsig)
    sig = np.exp(lsig).astype(np.float32)
    # Firing threshold. z is a max over orientations x scales of rectified
    # responses, so even pure noise has a positive floor (extreme-value
    # statistics) plus whatever line-like texture the image has. Measure that
    # floor on this image - the median and MAD of max_theta z, dominated by
    # background pixels - and express the drive as robust z-scores above it.
    # Units cancel, so the same thresholds hold for any modality and SNR.
    S0 = z.max(0)[::2, ::2]
    level = float(np.median(S0))
    spread = float(1.4826 * np.median(np.abs(S0 - level))) + 1e-12
    drive = np.maximum(z - level, 0.0) / spread
    return OrientationScore(orientations(N), drive.astype(np.float32), sig,
                            np.asarray(noises), background=(level, spread))


# ---------------------------------------------------------------- stage 2
class AssociationField:
    """Bipole horizontal connections as one multi-channel FFT convolution."""

    def __init__(self, shape, cfg: Config):
        N = cfg.n_orient
        K = cfg.assoc_orient_bins
        thetas = orientations(N)
        dth = np.pi / N
        kernels, self.routing = [], {+1: [], -1: []}
        for lobe in (+1, -1):
            for i, th in enumerate(thetas):
                terms = []
                for k in range(-K, K + 1):
                    w = association_field(
                        float(th), float(k * dth), float(cfg.assoc_sigma),
                        float(cfg.assoc_angle), float(dth), lobe,
                    )
                    terms.append(((i + k) % N, len(kernels)))
                    kernels.append(w)
                self.routing[lobe].append(terms)
        self.bank = FourierBank(kernels, shape)

    def __call__(self, U: np.ndarray):
        specs = self.bank.spectrum(U)
        front = self.bank.apply_mixed(specs, self.routing[+1])
        back = self.bank.apply_mixed(specs, self.routing[-1])
        return np.maximum(front, 0), np.maximum(back, 0)


@lru_cache(maxsize=8)
def _assoc(shape, cfg: Config) -> AssociationField:
    return AssociationField(shape, cfg)


def recurrent(score: OrientationScore, cfg: Config) -> OrientationScore:
    U0 = score.z
    U = U0.copy()
    if cfg.n_iter <= 0:
        score.U = U
        return score
    field = _assoc(U0.shape[1:], cfg)
    a, h = cfg.modulation, cfg.modulation_halfsat
    for _ in range(cfg.n_iter):
        Ff, Fb = field(U)
        Fmod = 0.5 * (Ff + Fb)
        Fand = np.sqrt(Ff * Fb)
        R = U0 * (1.0 + a * Fmod / (h + Fmod)) + cfg.completion * Fand
        R -= cfg.cross_inhibition * U.mean(0, keepdims=True)
        np.maximum(R, 0, out=R)
        if cfg.saturation > 0:
            pool = ndi.gaussian_filter(R.max(0), cfg.pool_sigma)
            R /= 1.0 + cfg.saturation * pool
        U = (1 - cfg.relax) * U + cfg.relax * R
    # Divide out the asymptotic gain on a long straight vessel, (1+a)/(1-b),
    # so thresholds keep the same meaning (robust z above background) whether
    # the horizontal connections are on, off, or retuned.
    gain = (1.0 + a) / max(1.0 - cfg.completion, 1e-3)
    score.U = (U / gain).astype(np.float32)
    return score
