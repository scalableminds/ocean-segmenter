"""Threshold-based segmentation layers and Neuroglancer meshes from ocean Zarr data.

    from ocean_segmenter import SegmentRequest, run
    result = run(SegmentRequest(source_url=..., min=34.9, max=35.1, output_url="out/so.zarr"))
"""

from .models import SegmentInfo, SegmentRequest, SegmentResult, ThresholdRule
from .pipeline import ProgressCallback, run

__all__ = ["ProgressCallback", "SegmentInfo", "SegmentRequest", "SegmentResult", "ThresholdRule", "run"]
