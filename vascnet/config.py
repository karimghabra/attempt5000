"""All tunable constants of the pipeline, in one place.

Nothing in this package is learned. Every parameter below is either

* a *physical* quantity expressed in pixels (vessel radii the filter bank spans,
  reach of the horizontal connections), which you set once per imaging setup
  from the magnification, or
* a *dimensionless* quantity (thresholds in units of the measured noise floor,
  angular tolerances, gain constants) that should transfer across images,
  modalities and SNRs without retuning.

Thresholds are expressed as z-scores against a noise level that is re-estimated
on every image, which is what makes a fixed set of constants generalize.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# Optimal second-derivative-of-Gaussian scale for a cylindrical vessel of radius
# r imaged in transmission (chord-length optical density profile) is
# sigma* = 0.836 r, and the scale-normalized peak response is 0.446 x peak OD.
SIGMA_PER_RADIUS = 0.836
PEAK_RESPONSE_PER_OD = 0.446


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ input
    polarity: str = "dark"
    """'dark' if vessels absorb light (bright-field, fundus, angiography),
    'bright' if they emit (fluorescence)."""

    # ----------------------------------------- stage 0: retina (adaptation)
    adapt_sigma: float = 6.0
    """Smoothing (px) of the background estimate used for luminance adaptation."""

    # ------------------------------------ stage 1: V1 simple / complex cells
    radii: tuple[float, ...] = (1.0, 1.8, 3.2, 5.6)
    """Vessel radii (px) the oriented filter bank is tuned to (log-spaced)."""
    min_sigma: float = 1.0
    """Smallest filter scale; below ~1 px the Gaussian derivative aliases."""
    n_orient: int = 16
    """Orientation channels over [0, pi)."""
    elongation: float = 2.5
    """Receptive-field aspect ratio, sigma_along / sigma_across."""
    edge_suppression: float = 1.0
    """alpha: quadrature-pair line-vs-edge test (across-axis asymmetry)."""
    end_stopping: float = 1.0
    """beta: penalty on front/back half-field asymmetry (along-axis). 1 means
    full bipole AND (2*min of the halves); 0 disables. Removes the radial fans
    elongated filters throw off junctions and line ends."""

    # --------------------------- stage 2: horizontal connections (recurrent)
    n_iter: int = 2
    assoc_sigma: float = 6.0
    """Spatial reach (px) of the association field."""
    assoc_angle: float = 20.0
    """Curvature tolerance (deg): spread of sender direction about the axis."""
    assoc_orient_bins: int = 2
    """Orientation coupling, +- this many neighbouring channels."""
    modulation: float = 0.5
    """Max multiplicative gain from collinear support (modulatory synapses)."""
    modulation_halfsat: float = 3.0
    """Support (z units) at which the modulatory gain is half-saturated."""
    completion: float = 0.0
    """Gain of additive bipole completion (fills gaps only with support on
    *both* sides, so it never extrapolates past a vessel end). Off by default:
    in ablations it bridged neighbouring vessels at junctions more often than
    it closed real gaps; gap closing is done more selectively at the graph
    level (free-end linking). Kept for very fragmented, low-SNR data."""
    cross_inhibition: float = 1.0
    """Subtractive, orientation-untuned inhibition (sharpens tuning)."""
    saturation: float = 0.02
    """1/z_sat of divisive normalization (contrast gain control). 0 = off."""
    pool_sigma: float = 4.0
    relax: float = 0.7
    """Relaxation rate of the recurrent update (1 = no memory)."""

    # ---------------------------------------- stage 3: centerline extraction
    t_high: float = 4.5
    """Hysteresis seed threshold, in robust z-scores above the background
    floor of the orientation score (re-measured on every image)."""
    t_low: float = 2.0
    """Hysteresis continuation threshold (same units)."""
    nms_min_sep: int = 1
    nms_secondary_ratio: float = 0.0
    """Besides the dominant orientation, a channel keeps its ridge only as a
    distinct peak (>= nms_min_sep bins away and >= ratio x dominant strength),
    so crossing vessels survive but near-duplicate channels on curved vessels
    do not lay parallel ridges. min_sep=1, ratio=0 keeps every local peak."""
    min_length: int = 10
    """Discard centerline fragments shorter than this (px)."""

    # ------------------------------------------- stage 4: graph and junctions
    spur_factor: float = 2.0
    spur_min: float = 4.0
    """Terminal branches shorter than max(spur_min, spur_factor*r) are pruned."""
    merge_factor: float = 1.0
    merge_min: float = 3.0
    """Junctions closer than max(merge_min, merge_factor*r) are always fused."""
    crossing_merge_factor: float = 2.5
    """Two 3-way junctions up to max(merge_min, crossing_merge_factor*r) apart
    are fused into one crossover *if* their arms continue straight through
    each other (a crossover usually skeletonizes as two Y's)."""
    link_factor: float = 3.0
    link_min: float = 5.0
    link_cone: float = 35.0
    """Free ends are bridged to a centerline within max(link_min,
    link_factor*r) px inside a +-link_cone deg cone (closes T-gaps)."""

    straight_tol: float = 32.0
    """Max deviation from 180 deg for two arms to count as one vessel."""
    t_straight_max: float = 20.0
    t_side_min: float = 50.0
    """A 3-arm junction is a T (side branch, or a crossing missing one arm)
    only if two arms continue each other within t_straight_max (deg) and the
    third leaves that through vessel at >= t_side_min (deg)."""
    cross_dev_max: float = 45.0
    """A 4-arm junction is a crossover if its best pairing into two vessels
    deviates from straight by at most this much (deg) ..."""
    cross_margin: float = 25.0
    """... and beats the next-best pairing by at least this much (deg)."""
    completion_cone: float = 0.0
    """Half-angle (deg) searched on the far side of a T for the hidden half of
    a crossing vessel (amodal completion)."""
    completion_ratio: float = 0.35
    """The hidden half must reach this fraction of the visible half's strength
    (and at least t_high) to be accepted."""
    min_cross_angle: float = 12.0
    """Below this, a crossover is reported as a tangential 'contact'."""
    pair_width_tol: float = 0.45
    """|log width ratio| scale in the same-vessel pairing cost."""
    pair_contrast_tol: float = 0.55
    """|log contrast ratio| scale in the same-vessel pairing cost."""

    # ---------------------------------------------------------------- helpers
    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(max(self.min_sigma, SIGMA_PER_RADIUS * r) for r in self.radii)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)
