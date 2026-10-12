"""vascnet - a fixed-weight, biologically inspired network for vessel centerlines
and junctions (bifurcations, crossovers, complex junctions). Nothing is learned.
"""

from .config import Config
from .pipeline import Result, VesselPipeline

__all__ = ["Config", "Result", "VesselPipeline"]
