"""SDR encoding for image patches.

Converts extracted features into sparse distributed representations
for assembly processing.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional, List, Tuple
from functools import partial

from ..core.sdr import SDR, SDRConfig
from .feature_extraction import MultiScaleFeatures, StructureTensor


class EncoderConfig(NamedTuple):
    """Configuration for SDR encoder."""
    sdr_size: int = 2048
    sparsity: float = 0.02
    n_active: int = 40
    
    # Feature thresholds for bit activation
    anisotropy_thresholds: Tuple[float, float, float] = (1.8, 1.4, 1.1)
    coherence_threshold: float = 0.5
    gradient_threshold: float = 0.1
    
    # Bit allocation
    structure_bits: int = 900  # Bits 0-899 for structure
    texture_bits: int = 300  # Bits 900-1199 for texture
    phase_bits: int = 300  # Bits 1200-1499 for phase
    context_bits: int = 548  # Bits 1500-2047 for context


class SDREncoder:
    """Encode image features into SDRs."""
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        """Initialize encoder.
        
        Args:
            config: Encoder configuration
        """
        self.config = config or EncoderConfig()
        self.sdr_config = SDRConfig(
            size=self.config.sdr_size,
            sparsity=self.config.sparsity,
            n_active=self.config.n_active
        )
        
        # Precompute bit ranges for each feature type
        self._setup_bit_ranges()
        
    def _setup_bit_ranges(self):
        """Setup bit ranges for different feature types."""
        # Structure bits by anisotropy
        self.elongated_bits = jnp.arange(0, 300)  # λ ≈ 2.0
        self.curved_bits = jnp.arange(300, 600)  # λ ≈ 1.5
        self.circular_bits = jnp.arange(600, 900)  # λ ≈ 1.0
        
        # Texture bits
        self.texture_bits = jnp.arange(900, 1200)
        
        # Phase bits (for temporal encoding)
        self.phase_bits = jnp.arange(1200, 1500)
        
        # Context bits
        self.context_bits = jnp.arange(1500, 2048)
        
    def encode_features(self, features: MultiScaleFeatures) -> SDR:
        """Encode multi-scale features into SDR.
        
        Args:
            features: Extracted features
            
        Returns:
            SDR representation
        """
        active_bits = []
        
        # Encode structure at each scale
        active_bits.extend(self._encode_structure(features.fine_scale, scale='fine'))
        active_bits.extend(self._encode_structure(features.medium_scale, scale='medium'))
        active_bits.extend(self._encode_structure(features.coarse_scale, scale='coarse'))
        
        # Encode texture
        active_bits.extend(self._encode_texture(features.texture_features))
        
        # Encode intensity context
        active_bits.extend(self._encode_context(features.intensity_features))
        
        # Ensure exactly n_active bits
        active_bits = np.array(active_bits)
        if len(active_bits) > self.config.n_active:
            # Keep most important bits (could use importance weighting)
            active_bits = np.random.choice(active_bits, 
                                         self.config.n_active,
                                         replace=False)
        elif len(active_bits) < self.config.n_active:
            # Add random context bits to reach sparsity
            available = np.setdiff1d(self.context_bits, active_bits)
            n_add = self.config.n_active - len(active_bits)
            additional = np.random.choice(available, n_add, replace=False)
            active_bits = np.concatenate([active_bits, additional])
            
        return SDR(active_indices=active_bits, config=self.sdr_config)
        
    def _encode_structure(self, 
                         structure: StructureTensor,
                         scale: str) -> List[int]:
        """Encode structure tensor into bits.
        
        Args:
            structure: Structure tensor features
            scale: Scale identifier ('fine', 'medium', 'coarse')
            
        Returns:
            List of active bit indices
        """
        active = []
        
        # Scale-dependent offset
        scale_offset = {'fine': 0, 'medium': 100, 'coarse': 200}[scale]
        
        # Select bits based on anisotropy (eigenvalue ratio)
        if structure.anisotropy >= self.config.anisotropy_thresholds[0]:
            # Elongated structure
            base_bits = self.elongated_bits + scale_offset
        elif structure.anisotropy >= self.config.anisotropy_thresholds[1]:
            # Curved structure
            base_bits = self.curved_bits + scale_offset
        else:
            # Circular structure
            base_bits = self.circular_bits + scale_offset
            
        # Select specific bits based on coherence and orientation
        if structure.coherence > self.config.coherence_threshold:
            # Strong structure - encode orientation
            orientation_bin = int((structure.orientation + np.pi) / (2 * np.pi) * 10)
            active.extend(base_bits[orientation_bin*3:(orientation_bin+1)*3])
            
        # Add bits for eigenvalue magnitudes
        for i, eigenval in enumerate(structure.eigenvalues[:2]):
            if abs(eigenval) > self.config.gradient_threshold:
                active.append(int(base_bits[10 + i]))
                
        return active
        
    def _encode_texture(self, texture_features: jnp.ndarray) -> List[int]:
        """Encode texture features into bits.
        
        Args:
            texture_features: Texture feature vector
            
        Returns:
            List of active bit indices
        """
        active = []
        
        # Quantize texture features
        n_texture_bits = len(self.texture_bits)
        n_features = len(texture_features)
        bits_per_feature = n_texture_bits // n_features
        
        for i, feature in enumerate(texture_features):
            # Normalize feature to [0, 1]
            normalized = (feature - feature.min()) / (feature.max() - feature.min() + 1e-10)
            
            # Select bits based on feature value
            bit_idx = int(normalized * bits_per_feature)
            base_idx = i * bits_per_feature
            active.append(self.texture_bits[base_idx + bit_idx])
            
        return active
        
    def _encode_context(self, intensity_features: jnp.ndarray) -> List[int]:
        """Encode intensity context into bits.
        
        Args:
            intensity_features: Intensity statistics
            
        Returns:
            List of active bit indices
        """
        active = []
        
        # Simple binning of intensity values
        for i, intensity in enumerate(intensity_features):
            # Normalize to [0, 1]
            normalized = jnp.clip(intensity, 0, 1)
            
            # Select context bit
            bit_idx = int(normalized * 100) + i * 100
            if bit_idx < len(self.context_bits):
                active.append(self.context_bits[bit_idx])
                
        return active


class ImageToSDR:
    """Complete pipeline from image to SDR representation."""
    
    def __init__(self,
                 encoder_config: Optional[EncoderConfig] = None):
        """Initialize image to SDR converter.
        
        Args:
            encoder_config: Encoder configuration
        """
        from .feature_extraction import MultiScaleExtractor
        
        self.feature_extractor = MultiScaleExtractor()
        self.encoder = SDREncoder(encoder_config)
        
    def convert_patch(self,
                     image: jnp.ndarray,
                     position: Tuple[int, int]) -> SDR:
        """Convert image patch at position to SDR.
        
        Args:
            image: Input image
            position: Center position of patch
            
        Returns:
            SDR representation
        """
        # Extract multi-scale features
        features = self.feature_extractor.extract_multiscale_features(image, position)
        
        # Encode to SDR
        sdr = self.encoder.encode_features(features)
        
        return sdr
        
    def convert_image(self,
                     image: jnp.ndarray,
                     stride: int = 1) -> List[List[SDR]]:
        """Convert entire image to SDR grid.
        
        Args:
            image: Input image
            stride: Stride between patches
            
        Returns:
            2D grid of SDRs
        """
        h, w = image.shape[:2]
        sdrs = []
        
        for x in range(0, h, stride):
            row = []
            for y in range(0, w, stride):
                sdr = self.convert_patch(image, (x, y))
                row.append(sdr)
            sdrs.append(row)
            
        return sdrs
        
    @partial(jax.jit, static_argnames=['self', 'batch_size'])
    def batch_convert(self,
                     image: jnp.ndarray,
                     positions: jnp.ndarray,
                     batch_size: int = 32) -> jnp.ndarray:
        """Batch convert multiple positions to SDRs.
        
        Args:
            image: Input image
            positions: Array of (x, y) positions
            batch_size: Batch size for processing
            
        Returns:
            Dense SDR matrix (n_positions, sdr_size)
        """
        n_positions = positions.shape[0]
        sdr_matrix = jnp.zeros((n_positions, self.encoder.config.sdr_size))
        
        for i in range(0, n_positions, batch_size):
            batch_positions = positions[i:i+batch_size]
            
            for j, pos in enumerate(batch_positions):
                sdr = self.convert_patch(image, tuple(pos))
                sdr_matrix = sdr_matrix.at[i+j].set(sdr.dense)
                
        return sdr_matrix