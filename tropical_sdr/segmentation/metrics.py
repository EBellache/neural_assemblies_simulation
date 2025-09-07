"""Evaluation metrics for segmentation.

Implements standard segmentation metrics without requiring
gradient computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional
from functools import partial


class SegmentationMetrics:
    """Compute segmentation evaluation metrics."""
    
    @staticmethod
    @jax.jit
    def compute_iou(pred: jnp.ndarray,
                   target: jnp.ndarray,
                   class_id: int) -> float:
        """Compute Intersection over Union for a class.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            class_id: Class to evaluate
            
        Returns:
            IoU score
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = jnp.sum(pred_mask & target_mask)
        union = jnp.sum(pred_mask | target_mask)
        
        iou = intersection / (union + 1e-10)
        return iou
        
    @staticmethod
    @jax.jit
    def compute_dice(pred: jnp.ndarray,
                    target: jnp.ndarray,
                    class_id: int) -> float:
        """Compute Dice coefficient for a class.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            class_id: Class to evaluate
            
        Returns:
            Dice score
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = jnp.sum(pred_mask & target_mask)
        dice = 2 * intersection / (jnp.sum(pred_mask) + jnp.sum(target_mask) + 1e-10)
        
        return dice
        
    @staticmethod
    def compute_boundary_f1(pred: jnp.ndarray,
                           target: jnp.ndarray,
                           tolerance: int = 2) -> float:
        """Compute boundary F1 score.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            tolerance: Pixel tolerance for boundary matching
            
        Returns:
            Boundary F1 score
        """
        from scipy.ndimage import distance_transform_edt
        
        # Find boundaries
        pred_boundaries = self._find_boundaries(pred)
        target_boundaries = self._find_boundaries(target)
        
        # Compute distance transforms
        pred_dist = distance_transform_edt(~pred_boundaries)
        target_dist = distance_transform_edt(~target_boundaries)
        
        # Count matches within tolerance
        pred_matched = jnp.sum((pred_boundaries > 0) & (target_dist <= tolerance))
        target_matched = jnp.sum((target_boundaries > 0) & (pred_dist <= tolerance))
        
        pred_total = jnp.sum(pred_boundaries > 0)
        target_total = jnp.sum(target_boundaries > 0)
        
        precision = pred_matched / (pred_total + 1e-10)
        recall = target_matched / (target_total + 1e-10)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return float(f1)
        
    @staticmethod
    def _find_boundaries(labels: jnp.ndarray) -> jnp.ndarray:
        """Find boundaries in label image.
        
        Args:
            labels: Label image
            
        Returns:
            Binary boundary image
        """
        # Simple boundary detection using gradients
        dx = jnp.diff(labels, axis=0, prepend=labels[0:1, :])
        dy = jnp.diff(labels, axis=1, prepend=labels[:, 0:1])
        
        boundaries = (jnp.abs(dx) > 0) | (jnp.abs(dy) > 0)
        return boundaries.astype(jnp.float32)
        
    def compute_all_metrics(self,
                           pred: jnp.ndarray,
                           target: jnp.ndarray,
                           n_classes: int) -> Dict[str, float]:
        """Compute all segmentation metrics.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            n_classes: Number of classes
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Per-class IoU and Dice
        ious = []
        dices = []
        
        for class_id in range(n_classes):
            if jnp.sum(target == class_id) > 0:  # Class present in ground truth
                iou = self.compute_iou(pred, target, class_id)
                dice = self.compute_dice(pred, target, class_id)
                
                ious.append(iou)
                dices.append(dice)
                
                metrics[f'iou_class_{class_id}'] = float(iou)
                metrics[f'dice_class_{class_id}'] = float(dice)
                
        # Mean metrics
        metrics['mean_iou'] = float(jnp.mean(jnp.array(ious)))
        metrics['mean_dice'] = float(jnp.mean(jnp.array(dices)))
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = float(jnp.mean(pred == target))
        
        # Boundary F1 (expensive, optional)
        # metrics['boundary_f1'] = self.compute_boundary_f1(pred, target)
        
        return metrics


# Convenience functions
@jax.jit
def compute_iou(pred: jnp.ndarray, 
               target: jnp.ndarray,
               n_classes: int) -> jnp.ndarray:
    """Compute IoU for all classes.
    
    Args:
        pred: Predicted labels (H, W)
        target: Ground truth labels (H, W)
        n_classes: Number of classes
        
    Returns:
        IoU scores for each class
    """
    metrics = SegmentationMetrics()
    ious = []
    
    for c in range(n_classes):
        iou = metrics.compute_iou(pred, target, c)
        ious.append(iou)
        
    return jnp.array(ious)


@jax.jit  
def compute_dice(pred: jnp.ndarray,
                target: jnp.ndarray,
                n_classes: int) -> jnp.ndarray:
    """Compute Dice coefficient for all classes.
    
    Args:
        pred: Predicted labels (H, W)
        target: Ground truth labels (H, W)
        n_classes: Number of classes
        
    Returns:
        Dice scores for each class
    """
    metrics = SegmentationMetrics()
    dices = []
    
    for c in range(n_classes):
        dice = metrics.compute_dice(pred, target, c)
        dices.append(dice)
        
    return jnp.array(dices)


def compute_boundary_f1(pred: jnp.ndarray,
                       target: jnp.ndarray,
                       tolerance: int = 2) -> float:
    """Compute boundary F1 score.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        tolerance: Pixel tolerance
        
    Returns:
        Boundary F1 score
    """
    metrics = SegmentationMetrics()
    return metrics.compute_boundary_f1(pred, target, tolerance)