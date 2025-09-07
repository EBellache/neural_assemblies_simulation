"""Tests for segmentation functionality."""

import pytest
import numpy as np
import jax.numpy as jnp

from tropical_sdr.segmentation.feature_extraction import (
    StructureTensorExtractor,
    TextureFeatureExtractor,
    MultiScaleExtractor
)
from tropical_sdr.segmentation.encoder import SDREncoder, ImageToSDR
from tropical_sdr.segmentation.segmenter import TropicalSegmenter, SegmentationConfig
from tropical_sdr.segmentation.metrics import compute_iou, compute_dice


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_structure_tensor(self):
        """Test structure tensor computation."""
        extractor = StructureTensorExtractor()
        
        # Create test image with edge
        image = jnp.zeros((50, 50))
        image = image.at[20:30, :].set(1.0)
        
        tensor = extractor.compute_structure_tensor(image, window_size=9)
        
        assert tensor.eigenvalues.shape == (2,)
        assert 0 <= tensor.coherence <= 1
        assert tensor.anisotropy > 1  # Should detect edge structure
        
    def test_texture_features(self):
        """Test texture feature extraction."""
        extractor = TextureFeatureExtractor()
        
        # Create textured patch
        patch = jnp.array(np.random.rand(27, 27))
        
        features = extractor.extract_texture_features(patch)
        
        assert len(features) == 8  # Number of texture features
        assert jnp.all(jnp.isfinite(features))
        
    def test_multiscale_extraction(self):
        """Test multi-scale feature extraction."""
        extractor = MultiScaleExtractor()
        
        # Create test image
        image = jnp.array(np.random.rand(150, 150))
        
        features = extractor.extract_multiscale_features(image, (75, 75))
        
        assert features.fine_scale is not None
        assert features.medium_scale is not None
        assert features.coarse_scale is not None
        assert features.texture_features is not None
        assert features.intensity_features is not None


class TestSDREncoding:
    """Test SDR encoding."""
    
    def test_encoder(self):
        """Test SDR encoder."""
        encoder = SDREncoder()
        extractor = MultiScaleExtractor()
        
        # Create test image and extract features
        image = jnp.array(np.random.rand(150, 150))
        features = extractor.extract_multiscale_features(image, (75, 75))
        
        # Encode to SDR
        sdr = encoder.encode_features(features)
        
        assert sdr.config.n_active == 40
        assert len(sdr.sparse) == 40
        assert sdr.sparsity == 0.02 or len(sdr.sparse) == 40
        
    def test_image_to_sdr(self):
        """Test complete image to SDR pipeline."""
        converter = ImageToSDR()
        
        image = jnp.array(np.random.rand(100, 100))
        
        # Convert single patch
        sdr = converter.convert_patch(image, (50, 50))
        assert len(sdr.sparse) == 40
        
        # Convert entire image
        sdr_grid = converter.convert_image(image, stride=10)
        assert len(sdr_grid) == 10  # 100/10
        assert len(sdr_grid[0]) == 10


class TestSegmenter:
    """Test segmentation pipeline."""
    
    def test_segmenter_creation(self):
        """Test segmenter initialization."""
        config = SegmentationConfig(n_assemblies=7)
        segmenter = TropicalSegmenter(config)
        
        assert segmenter.network.n_active == 7
        
    def test_single_scale_segmentation(self):
        """Test single-scale segmentation."""
        segmenter = TropicalSegmenter(
            SegmentationConfig(use_multiscale=False)
        )
        
        # Small test image
        image = jnp.array(np.random.rand(32, 32))
        
        result = segmenter.segment(image)
        
        assert result.labels.shape == (32, 32)
        assert result.confidence.shape == (32, 32)
        assert 0 <= jnp.min(result.labels) <= jnp.max(result.labels) < 10
        
    def test_multiscale_segmentation(self):
        """Test multi-scale segmentation."""
        segmenter = TropicalSegmenter(
            SegmentationConfig(use_multiscale=True)
        )
        
        # Test image
        image = jnp.array(np.random.rand(64, 64))
        
        result = segmenter.segment(image)
        
        assert result.labels.shape == (64, 64)
        assert result.uncertainty_mask.shape == (64, 64)
        
    def test_training(self):
        """Test unsupervised training."""
        segmenter = TropicalSegmenter()
        
        # Create training images
        images = [jnp.array(np.random.rand(32, 32)) for _ in range(3)]
        
        segmenter.train(images)
        
        # Check that patterns were learned
        stats = segmenter.network.get_assembly_statistics()
        total_patterns = sum(s['n_patterns'] for s in stats)
        assert total_patterns > 0


class TestMetrics:
    """Test segmentation metrics."""
    
    def test_iou(self):
        """Test IoU computation."""
        pred = jnp.array([[0, 0, 1, 1],
                         [0, 0, 1, 1],
                         [2, 2, 2, 2],
                         [2, 2, 2, 2]])
        
        target = jnp.array([[0, 0, 0, 1],
                           [0, 0, 1, 1],
                           [2, 2, 1, 1],
                           [2, 2, 2, 2]])
        
        ious = compute_iou(pred, target, n_classes=3)
        
        assert len(ious) == 3
        assert 0 <= ious[0] <= 1  # Class 0 IoU
        assert 0 <= ious[1] <= 1  # Class 1 IoU
        assert 0 <= ious[2] <= 1  # Class 2 IoU
        
    def test_dice(self):
        """Test Dice coefficient."""
        pred = jnp.array([[1, 1, 0],
                         [1, 1, 0],
                         [0, 0, 0]])
        
        target = jnp.array([[1, 1, 1],
                           [1, 1, 0],
                           [0, 0, 0]])
        
        dice_scores = compute_dice(pred, target, n_classes=2)
        
        assert len(dice_scores) == 2
        # Class 1 Dice should be 2*4/(4+5) = 8/9
        assert np.isclose(dice_scores[1], 8/9, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])