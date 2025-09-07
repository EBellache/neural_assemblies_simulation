"""Main segmentation pipeline using tropical assemblies.

Implements complete image segmentation without gradient descent.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional, Dict, Tuple, List
from functools import partial
import time

from ..assembly.network import TropicalAssemblyNetwork, NetworkConfig
from ..core.sdr import SDR
from .encoder import ImageToSDR, EncoderConfig
from .metrics import SegmentationMetrics


class SegmentationConfig(NamedTuple):
    """Configuration for segmentation."""
    # Network config
    n_assemblies: int = 7
    min_assemblies: int = 5
    max_assemblies: int = 9
    
    # Processing config
    patch_stride: int = 1
    batch_size: int = 32
    min_confidence: float = 5.0
    
    # Learning config
    online_learning: bool = True
    learning_rate: float = 0.01
    
    # Multi-scale config
    use_multiscale: bool = True
    scale_weights: Tuple[float, float, float] = (0.6, 1.0, 0.8)


class SegmentationResult(NamedTuple):
    """Result of segmentation."""
    labels: jnp.ndarray  # Segmentation labels
    confidence: jnp.ndarray  # Confidence map
    assembly_scores: jnp.ndarray  # Raw scores for each assembly
    uncertainty_mask: jnp.ndarray  # Uncertain regions
    timing: Dict[str, float]  # Timing information


class TropicalSegmenter:
    """Complete segmentation system using tropical assemblies."""
    
    def __init__(self,
                 config: Optional[SegmentationConfig] = None):
        """Initialize segmenter.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config or SegmentationConfig()
        
        # Initialize network
        network_config = NetworkConfig(
            base_assemblies=self.config.n_assemblies,
            min_assemblies=self.config.min_assemblies,
            max_assemblies=self.config.max_assemblies
        )
        self.network = TropicalAssemblyNetwork(network_config)
        
        # Initialize encoder
        self.image_to_sdr = ImageToSDR()
        
        # Metrics tracker
        self.metrics = SegmentationMetrics()
        
        # Cache for multi-scale processing
        self._scale_cache = {}
        
    def segment(self, 
               image: jnp.ndarray,
               mask: Optional[jnp.ndarray] = None) -> SegmentationResult:
        """Segment image using tropical assembly competition.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Optional mask for region of interest
            
        Returns:
            Segmentation result
        """
        timing = {}
        
        # Preprocessing
        t0 = time.time()
        if image.ndim == 3:
            # Convert to grayscale if needed
            image = jnp.mean(image, axis=-1)
            
        if mask is not None:
            image = image * mask
            
        h, w = image.shape
        timing['preprocessing'] = time.time() - t0
        
        # Extract SDRs
        t0 = time.time()
        if self.config.use_multiscale:
            labels, confidence = self._multiscale_segment(image)
        else:
            labels, confidence = self._single_scale_segment(image)
        timing['feature_extraction'] = time.time() - t0
        
        # Post-processing
        t0 = time.time()
        labels, confidence, uncertainty_mask = self._postprocess(
            labels, confidence, image
        )
        timing['postprocessing'] = time.time() - t0
        
        # Get assembly scores for analysis
        assembly_scores = self._get_assembly_statistics(labels)
        
        return SegmentationResult(
            labels=labels,
            confidence=confidence,
            assembly_scores=assembly_scores,
            uncertainty_mask=uncertainty_mask,
            timing=timing
        )
        
    def _single_scale_segment(self, 
                            image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Segment at single scale.
        
        Args:
            image: Input image
            
        Returns:
            (labels, confidence)
        """
        h, w = image.shape
        labels = jnp.zeros((h, w), dtype=jnp.int32)
        confidence = jnp.zeros((h, w), dtype=jnp.float32)
        
        # Process in batches for efficiency
        positions = []
        for x in range(0, h, self.config.patch_stride):
            for y in range(0, w, self.config.patch_stride):
                positions.append((x, y))
                
        positions = jnp.array(positions)
        
        # Batch convert to SDRs
        sdr_matrix = self.image_to_sdr.batch_convert(
            image, positions, self.config.batch_size
        )
        
        # Batch competition
        batch_labels, batch_confidence = self.network.batch_segment(sdr_matrix)
        
        # Fill output arrays
        for i, (x, y) in enumerate(positions):
            labels = labels.at[x, y].set(batch_labels[i])
            confidence = confidence.at[x, y].set(batch_confidence[i])
            
        return labels, confidence
        
    def _multiscale_segment(self,
                          image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Segment using multiple scales with tropical aggregation.
        
        Args:
            image: Input image
            
        Returns:
            (labels, confidence)
        """
        h, w = image.shape
        
        # Process at three scales (8, 27, 125)
        scales = [8, 27, 125]
        scale_results = []
        
        for scale_idx, scale in enumerate(scales):
            # Downsample if needed
            if scale > 8:
                stride = scale // 8
            else:
                stride = 1
                
            scale_labels = jnp.zeros((h // stride, w // stride), dtype=jnp.int32)
            scale_confidence = jnp.zeros((h // stride, w // stride), dtype=jnp.float32)
            
            # Process at this scale
            for x in range(0, h, stride):
                for y in range(0, w, stride):
                    # Extract patch
                    patch_size = scale
                    x_start = max(0, x - patch_size // 2)
                    x_end = min(h, x + patch_size // 2)
                    y_start = max(0, y - patch_size // 2)
                    y_end = min(w, y + patch_size // 2)
                    
                    patch = image[x_start:x_end, y_start:y_end]
                    
                    # Convert to SDR
                    sdr = self.image_to_sdr.convert_patch(image, (x, y))
                    
                    # Run competition
                    winner, conf, _ = self.network.process_input(
                        sdr, learn=self.config.online_learning
                    )
                    
                    scale_labels = scale_labels.at[x // stride, y // stride].set(winner)
                    scale_confidence = scale_confidence.at[x // stride, y // stride].set(conf)
                    
            # Upsample to original resolution
            if stride > 1:
                scale_labels = jax.image.resize(
                    scale_labels, (h, w), method='nearest'
                )
                scale_confidence = jax.image.resize(
                    scale_confidence, (h, w), method='linear'
                )
                
            scale_results.append((scale_labels, scale_confidence))
            
        # Aggregate using tropical operations (maximum)
        weights = jnp.array(self.config.scale_weights)
        
        # Weighted confidence
        weighted_confidence = jnp.stack([
            conf * weights[i] for i, (_, conf) in enumerate(scale_results)
        ])
        
        # Tropical aggregation - select label with max weighted confidence
        best_scale = jnp.argmax(weighted_confidence, axis=0)
        
        labels = jnp.zeros((h, w), dtype=jnp.int32)
        confidence = jnp.zeros((h, w), dtype=jnp.float32)
        
        for i in range(len(scales)):
            mask = (best_scale == i)
            labels = jnp.where(mask, scale_results[i][0], labels)
            confidence = jnp.where(mask, scale_results[i][1], confidence)
            
        return labels, confidence
        
    def _postprocess(self,
                    labels: jnp.ndarray,
                    confidence: jnp.ndarray,
                    image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Post-process segmentation.
        
        Args:
            labels: Initial labels
            confidence: Confidence map
            image: Original image
            
        Returns:
            (processed_labels, processed_confidence, uncertainty_mask)
        """
        # Identify uncertain regions
        uncertainty_mask = confidence < self.config.min_confidence
        
        # Assign uncertain regions to special assembly (index 7)
        labels = jnp.where(uncertainty_mask, 7, labels)
        
        # Optional: spatial coherence without CRF
        # Natural coherence emerges from SDR overlap
        
        # Edge-aware smoothing (optional)
        if False:  # Disabled by default - tropical framework maintains sharp boundaries
            from scipy.ndimage import median_filter
            labels = median_filter(labels, size=3)
            
        return labels, confidence, uncertainty_mask
        
    def _get_assembly_statistics(self, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute assembly activation statistics.
        
        Args:
            labels: Segmentation labels
            
        Returns:
            Assembly statistics
        """
        n_assemblies = self.network.n_active
        stats = jnp.zeros(n_assemblies)
        
        for i in range(n_assemblies):
            stats = stats.at[i].set(jnp.sum(labels == i))
            
        return stats / labels.size
        
    def train(self, 
             images: List[jnp.ndarray],
             masks: Optional[List[jnp.ndarray]] = None):
        """Train segmenter on images (unsupervised).
        
        Args:
            images: List of training images
            masks: Optional masks for each image
        """
        for i, image in enumerate(images):
            mask = masks[i] if masks else None
            
            # Segment with learning enabled
            self.config = self.config._replace(online_learning=True)
            _ = self.segment(image, mask)
            
        # Get learning statistics
        stats = self.network.learning.get_learning_statistics()
        print(f"Training complete: {stats}")
        
    def evaluate(self,
                image: jnp.ndarray,
                ground_truth: jnp.ndarray) -> Dict[str, float]:
        """Evaluate segmentation against ground truth.
        
        Args:
            image: Input image
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        # Segment without learning
        self.config = self.config._replace(online_learning=False)
        result = self.segment(image)
        
        # Compute metrics
        metrics = self.metrics.compute_all_metrics(
            result.labels,
            ground_truth,
            n_classes=self.network.n_active
        )
        
        # Add timing information
        metrics.update(result.timing)
        
        return metrics
        
    def visualize_assemblies(self) -> Dict:
        """Get assembly specializations for visualization.
        
        Returns:
            Dictionary of assembly properties
        """
        viz_data = {}
        
        for i, assembly in enumerate(self.network.assemblies[:self.network.n_active]):
            viz_data[f'assembly_{i}'] = {
                'eigenvalue': assembly.eigenvalue,
                'feature_type': assembly.feature_type,
                'n_patterns': len(assembly.patterns),
                'metabolic_energy': assembly.state.metabolic_energy,
                'recent_wins': assembly.state.recent_winner_count
            }
            
        return viz_data