"""
Model middleware adapters.

Currently includes:
- adapt_panoptic_to_laneatt: converts Mask2Former-style panoptic outputs into a
  LaneATT-friendly mask/overlay bundle.
"""

from .panoptic_to_laneatt import adapt_panoptic_to_laneatt  # noqa: F401

