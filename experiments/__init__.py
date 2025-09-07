"""Experiment scripts for tropical SDR framework."""

from .train_segmentation import SegmentationTrainer
from .evaluate import Evaluator
from .ablations import AblationStudy

__all__ = [
    'SegmentationTrainer',
    'Evaluator',
    'AblationStudy'
]