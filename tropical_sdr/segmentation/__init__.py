"""Image segmentation using tropical assembly framework."""

from .feature_extraction import (
    StructureTensorExtractor,
    MultiScaleFeatures,
    extract_structure_tensor
)
from .encoder import (
    SDREncoder,
    EncoderConfig,
    ImageToSDR
)
from .segmenter import (
    TropicalSegmenter,
    SegmentationConfig,
    SegmentationResult
)
from .metrics import (
    SegmentationMetrics,
    compute_iou,
    compute_dice,
    compute_boundary_f1
)

__all__ = [
    'StructureTensorExtractor',
    'MultiScaleFeatures',
    'extract_structure_tensor',
    'SDREncoder',
    'EncoderConfig',
    'ImageToSDR',
    'TropicalSegmenter',
    'SegmentationConfig',
    'SegmentationResult',
    'SegmentationMetrics',
    'compute_iou',
    'compute_dice',
    'compute_boundary_f1'
]