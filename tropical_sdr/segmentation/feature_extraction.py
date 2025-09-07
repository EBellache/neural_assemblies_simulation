"""Feature extraction for image segmentation.

Computes structure tensors, eigenvalues, and multi-scale features
for SDR encoding.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Tuple, NamedTuple, Optional, List
from functools import partial


class StructureTensor(NamedTuple):
    """Structure tensor and derived features."""
    tensor: jnp.ndarray  # Structure tensor matrix
    eigenvalues: jnp.ndarray  # Sorted eigenvalues
    eigenvectors: jnp.ndarray  # Corresponding eigenvectors
    coherence: float  # Structure coherence
    orientation: float  # Principal orientation
    anisotropy: float  # Eigenvalue ratio (elongation)


class MultiScaleFeatures(NamedTuple):
    """Features at multiple scales."""
    fine_scale: StructureTensor  # 8×8 window
    medium_scale: StructureTensor  # 27×27 window
    coarse_scale: StructureTensor  # 125×125 window
    texture_features: jnp.ndarray  # Local texture statistics
    intensity_features: jnp.ndarray  # Intensity statistics


class StructureTensorExtractor:
    """Extract structure tensors for geometric analysis."""
    
    def __init__(self,
                 sigma_gradient: float = 1.0,
                 sigma_window: float = 3.0):
        """Initialize extractor.
        
        Args:
            sigma_gradient: Gaussian sigma for gradient computation
            sigma_window: Gaussian sigma for tensor averaging
        """
        self.sigma_gradient = sigma_gradient
        self.sigma_window = sigma_window
        
    @partial(jax.jit, static_argnames=['self', 'window_size'])
    def compute_structure_tensor(self,
                                image: jnp.ndarray,
                                window_size: int = 27) -> StructureTensor:
        """Compute structure tensor for image patch.
        
        Args:
            image: Input image or patch
            window_size: Size of local window
            
        Returns:
            Structure tensor and derived features
        """
        # Compute gradients
        gradients = self._compute_gradients(image)
        
        # Build structure tensor
        Ixx = gradients[0] * gradients[0]
        Iyy = gradients[1] * gradients[1]
        Ixy = gradients[0] * gradients[1]
        
        if image.ndim == 3:  # 3D image
            Izz = gradients[2] * gradients[2]
            Ixz = gradients[0] * gradients[2]
            Iyz = gradients[1] * gradients[2]
            
            # Apply Gaussian window
            Ixx = self._gaussian_smooth(Ixx, self.sigma_window)
            Iyy = self._gaussian_smooth(Iyy, self.sigma_window)
            Izz = self._gaussian_smooth(Izz, self.sigma_window)
            Ixy = self._gaussian_smooth(Ixy, self.sigma_window)
            Ixz = self._gaussian_smooth(Ixz, self.sigma_window)
            Iyz = self._gaussian_smooth(Iyz, self.sigma_window)
            
            # Build 3×3 tensor
            tensor = jnp.array([
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]
            ])
            
        else:  # 2D image
            # Apply Gaussian window
            Ixx = self._gaussian_smooth(Ixx, self.sigma_window)
            Iyy = self._gaussian_smooth(Iyy, self.sigma_window)
            Ixy = self._gaussian_smooth(Ixy, self.sigma_window)
            
            # Build 2×2 tensor
            tensor = jnp.array([
                [Ixx, Ixy],
                [Ixy, Iyy]
            ])
            
        # Compute eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(tensor)
        
        # Sort by magnitude (largest first)
        idx = jnp.argsort(jnp.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute derived features
        coherence = self._compute_coherence(eigenvalues)
        orientation = self._compute_orientation(eigenvectors)
        anisotropy = self._compute_anisotropy(eigenvalues)
        
        return StructureTensor(
            tensor=tensor,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            coherence=coherence,
            orientation=orientation,
            anisotropy=anisotropy
        )
        
    @jax.jit
    def _compute_gradients(self, image: jnp.ndarray) -> jnp.ndarray:
        """Compute image gradients using Gaussian derivatives.
        
        Args:
            image: Input image
            
        Returns:
            Gradient array [dI/dx, dI/dy, dI/dz]
        """
        # Sobel operators for gradient
        sobel_x = jnp.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]) / 8.0
        sobel_y = sobel_x.T
        
        if image.ndim == 2:
            dx = jsp.signal.convolve2d(image, sobel_x, mode='same')
            dy = jsp.signal.convolve2d(image, sobel_y, mode='same')
            return jnp.stack([dx, dy])
            
        else:  # 3D
            dx = jnp.zeros_like(image)
            dy = jnp.zeros_like(image)
            dz = jnp.zeros_like(image)
            
            for z in range(image.shape[2]):
                dx = dx.at[:, :, z].set(
                    jsp.signal.convolve2d(image[:, :, z], sobel_x, mode='same')
                )
                dy = dy.at[:, :, z].set(
                    jsp.signal.convolve2d(image[:, :, z], sobel_y, mode='same')
                )
                
            # Z gradient using forward differences
            dz = jnp.diff(image, axis=2, prepend=image[:, :, 0:1])
            
            return jnp.stack([dx, dy, dz])
            
    @jax.jit
    def _gaussian_smooth(self, 
                        image: jnp.ndarray,
                        sigma: float) -> jnp.ndarray:
        """Apply Gaussian smoothing.
        
        Args:
            image: Input image
            sigma: Gaussian standard deviation
            
        Returns:
            Smoothed image
        """
        # Create Gaussian kernel
        kernel_size = int(2 * jnp.ceil(3 * sigma) + 1)
        x = jnp.arange(kernel_size) - kernel_size // 2
        kernel_1d = jnp.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        
        if image.ndim == 2:
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            return jsp.signal.convolve2d(image, kernel_2d, mode='same')
        else:
            # Separable convolution for efficiency
            result = image
            for z in range(image.shape[2]):
                slice_2d = image[:, :, z]
                kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
                result = result.at[:, :, z].set(
                    jsp.signal.convolve2d(slice_2d, kernel_2d, mode='same')
                )
            return result
            
    @jax.jit
    def _compute_coherence(self, eigenvalues: jnp.ndarray) -> float:
        """Compute structure coherence from eigenvalues.
        
        Coherence measures how well-defined the structure is.
        
        Args:
            eigenvalues: Sorted eigenvalues
            
        Returns:
            Coherence value (0-1)
        """
        if len(eigenvalues) == 2:
            coherence = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1] + 1e-10)
        else:  # 3D
            coherence = (eigenvalues[0] - eigenvalues[2]) / (eigenvalues[0] + eigenvalues[2] + 1e-10)
        return jnp.abs(coherence)
        
    @jax.jit
    def _compute_orientation(self, eigenvectors: jnp.ndarray) -> float:
        """Compute principal orientation from eigenvectors.
        
        Args:
            eigenvectors: Matrix of eigenvectors
            
        Returns:
            Orientation angle in radians
        """
        principal_vector = eigenvectors[:, 0]
        return jnp.arctan2(principal_vector[1], principal_vector[0])
        
    @jax.jit
    def _compute_anisotropy(self, eigenvalues: jnp.ndarray) -> float:
        """Compute anisotropy (eigenvalue ratio).
        
        This corresponds to assembly eigenvalue for specialization.
        
        Args:
            eigenvalues: Sorted eigenvalues
            
        Returns:
            Anisotropy ratio (λ₁/λₙ)
        """
        # Avoid division by zero
        min_eigenvalue = jnp.maximum(jnp.abs(eigenvalues[-1]), 1e-10)
        return jnp.abs(eigenvalues[0]) / min_eigenvalue


@jax.jit
def extract_structure_tensor(image: jnp.ndarray,
                            position: Tuple[int, int],
                            window_size: int = 27) -> StructureTensor:
    """Extract structure tensor at specific position.
    
    Args:
        image: Input image
        position: (x, y) position
        window_size: Size of local window
        
    Returns:
        Structure tensor at position
    """
    x, y = position
    half_window = window_size // 2
    
    # Extract patch with padding
    x_start = max(0, x - half_window)
    x_end = min(image.shape[0], x + half_window + 1)
    y_start = max(0, y - half_window)
    y_end = min(image.shape[1], y + half_window + 1)
    
    patch = image[x_start:x_end, y_start:y_end]
    
    # Compute structure tensor
    extractor = StructureTensorExtractor()
    return extractor.compute_structure_tensor(patch, window_size)


class TextureFeatureExtractor:
    """Extract texture features for SDR encoding."""
    
    @jax.jit
    def extract_texture_features(self,
                                patch: jnp.ndarray) -> jnp.ndarray:
        """Extract texture statistics from image patch.
        
        Args:
            patch: Image patch
            
        Returns:
            Texture feature vector
        """
        features = []
        
        # Basic statistics
        features.append(jnp.mean(patch))
        features.append(jnp.std(patch))
        features.append(jnp.percentile(patch, 25))
        features.append(jnp.percentile(patch, 75))
        
        # Gradient statistics
        dx = jnp.diff(patch, axis=0, prepend=patch[0:1, :])
        dy = jnp.diff(patch, axis=1, prepend=patch[:, 0:1])
        gradient_mag = jnp.sqrt(dx**2 + dy**2)
        
        features.append(jnp.mean(gradient_mag))
        features.append(jnp.std(gradient_mag))
        
        # Local Binary Pattern-like features
        center = patch[patch.shape[0]//2, patch.shape[1]//2]
        lbp_code = jnp.sum(patch > center)
        features.append(lbp_code)
        
        # Entropy (simplified)
        hist, _ = jnp.histogram(patch.flatten(), bins=16)
        hist = hist / (hist.sum() + 1e-10)
        entropy = -jnp.sum(hist * jnp.log(hist + 1e-10))
        features.append(entropy)
        
        return jnp.array(features)


class MultiScaleExtractor:
    """Extract features at multiple scales (8, 27, 125)."""
    
    def __init__(self):
        """Initialize multi-scale extractor."""
        self.structure_extractor = StructureTensorExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        
        # P-adic inspired scales
        self.scales = [8, 27, 125]
        
    def extract_multiscale_features(self,
                                   image: jnp.ndarray,
                                   position: Tuple[int, int]) -> MultiScaleFeatures:
        """Extract features at all scales.
        
        Args:
            image: Input image
            position: Center position
            
        Returns:
            Multi-scale features
        """
        x, y = position
        
        # Extract at each scale
        features = {}
        for scale in self.scales:
            half_scale = scale // 2
            
            # Extract patch
            x_start = max(0, x - half_scale)
            x_end = min(image.shape[0], x + half_scale + 1)
            y_start = max(0, y - half_scale)
            y_end = min(image.shape[1], y + half_scale + 1)
            
            patch = image[x_start:x_end, y_start:y_end]
            
            # Compute structure tensor
            structure = self.structure_extractor.compute_structure_tensor(patch, scale)
            
            if scale == 8:
                features['fine_scale'] = structure
            elif scale == 27:
                features['medium_scale'] = structure
            else:  # 125
                features['coarse_scale'] = structure
                
        # Extract texture features at medium scale
        medium_patch = image[
            max(0, x-13):min(image.shape[0], x+14),
            max(0, y-13):min(image.shape[1], y+14)
        ]
        texture_features = self.texture_extractor.extract_texture_features(medium_patch)
        
        # Extract intensity features
        intensity_features = jnp.array([
            image[x, y],  # Center intensity
            jnp.mean(image[max(0, x-4):x+5, max(0, y-4):y+5]),  # Local mean
            jnp.std(image[max(0, x-4):x+5, max(0, y-4):y+5])  # Local std
        ])
        
        return MultiScaleFeatures(
            fine_scale=features['fine_scale'],
            medium_scale=features['medium_scale'],
            coarse_scale=features['coarse_scale'],
            texture_features=texture_features,
            intensity_features=intensity_features
        )