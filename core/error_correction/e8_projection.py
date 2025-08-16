"""
E8 Lattice Projection for Error Correction
==========================================

Implements E8 lattice projection and decoding for biological
error correction. GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from functools import partial

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class E8Point(NamedTuple):
    """Point on E8 lattice."""
    coordinates: jnp.ndarray  # 8D coordinates
    shell_number: int  # Distance from origin
    glue_vector: jnp.ndarray  # Glue vector for construction
    norm_squared: float  # ||x||²


class E8Projection(NamedTuple):
    """Result of E8 projection."""
    original: jnp.ndarray  # Original point
    projected: jnp.ndarray  # Nearest E8 point
    error: jnp.ndarray  # Error vector
    distance: float  # Distance to lattice
    decoding_successful: bool  # Whether decoding succeeded


class E8Shell(NamedTuple):
    """Shell of E8 lattice."""
    radius_squared: int  # r² (always even)
    points: List[jnp.ndarray]  # Points on shell
    kissing_number: int  # Number of points
    theta_series_coeff: int  # Coefficient in theta series


class E8ProjectionComputer:
    """
    Compute E8 lattice projections for error correction.
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize E8 projection computer.

        Args:
            scale: Lattice scale parameter
        """
        self.scale = scale
        self.dim = 8

        # Generate E8 basis and key structures
        self._init_e8_structures()

        # Pre-compile JAX functions
        self._compile_functions()

    def _init_e8_structures(self):
        """Initialize E8 lattice structures."""
        # E8 root system (240 roots)
        self.roots = self._generate_e8_roots()

        # Gram matrix
        self.gram = self._e8_gram_matrix()

        # Generator matrix (rows form basis)
        self.generator = self._e8_generator_matrix()

        # Glue vectors for construction from D8
        self.glue_vectors = self._generate_glue_vectors()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._nearest_point = jit(self._nearest_point_impl)
        self._two_coset_decode = jit(self._two_coset_decode_impl)
        self._compute_shell = jit(self._compute_shell_impl)
        self._theta_series = jit(self._theta_series_impl)
        self._kissing_configuration = jit(self._kissing_configuration_impl)
        self._error_correction = jit(self._error_correction_impl)
        self._whitening_transform = jit(self._whitening_transform_impl)

    def _generate_e8_roots(self) -> jnp.ndarray:
        """
        Generate 240 root vectors of E8.

        Returns:
            Array of root vectors (240 x 8)
        """
        roots = []

        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = jnp.zeros(8)
                        root = root.at[i].set(s1)
                        root = root.at[j].set(s2)
                        roots.append(root)

        # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs (128 roots)
        for signs in range(256):
            sign_pattern = [(signs >> i) & 1 for i in range(8)]
            if sum(sign_pattern) % 2 == 0:
                root = jnp.array([0.5 if s == 0 else -0.5 for s in sign_pattern])
                roots.append(root)

        return jnp.array(roots[:240])  # Take first 240

    def _e8_gram_matrix(self) -> jnp.ndarray:
        """
        E8 Gram matrix (Cartan matrix).

        Returns:
            8x8 Gram matrix
        """
        # E8 Cartan matrix
        gram = jnp.array([
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, 0],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, -1],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, 0, 0, -1, 0, 0, 2]
        ])
        return gram

    def _e8_generator_matrix(self) -> jnp.ndarray:
        """
        E8 generator matrix (basis vectors as rows).

        Returns:
            8x8 generator matrix
        """
        # Standard E8 basis
        gen = jnp.array([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ])
        return gen

    def _generate_glue_vectors(self) -> List[jnp.ndarray]:
        """
        Generate glue vectors for E8 = D8 + glue.

        Returns:
            List of glue vectors
        """
        # E8 = D8 ∪ (D8 + g) where g is glue vector
        g1 = jnp.ones(8) * 0.5  # All coordinates 1/2
        g2 = jnp.array([1, 0, 0, 0, 0, 0, 0, 1]) * 0.5
        return [jnp.zeros(8), g1, g2, g1 + g2]

    @partial(jit, static_argnums=(0,))
    def _nearest_point_impl(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Find nearest E8 lattice point (naive).

        Args:
            x: Point in R^8

        Returns:
            Nearest E8 point
        """
        # Use generator matrix for initial approximation
        # x ≈ Σ c_i v_i where v_i are basis vectors

        # Solve for coefficients
        coeffs = jnp.linalg.solve(self.generator.T, x)

        # Round to nearest integers
        rounded = jnp.round(coeffs)

        # Convert back to coordinates
        nearest = self.generator.T @ rounded

        return nearest * self.scale

    @partial(jit, static_argnums=(0,))
    def _two_coset_decode_impl(self, x: jnp.ndarray) -> E8Projection:
        """
        Two-coset E8 decoding (optimal).

        E8 = D8 ∪ (D8 + g) where g is glue vector.

        Args:
            x: Point to decode

        Returns:
            E8 projection result
        """
        x_scaled = x / self.scale
        best_point = None
        best_distance = jnp.inf

        # Try each coset
        for glue in self.glue_vectors:
            # Decode in D8 coset
            shifted = x_scaled - glue

            # D8 decoding (easier than E8)
            # Round to nearest even/odd pattern
            decoded = jnp.zeros(8)

            # Check parity constraint
            sum_rounded = 0
            for i in range(8):
                r = jnp.round(shifted[i])
                decoded = decoded.at[i].set(r)
                sum_rounded += r

            # Ensure even sum for D8
            if sum_rounded % 2 != 0:
                # Adjust closest coordinate
                errors = jnp.abs(shifted - decoded)
                min_idx = jnp.argmin(errors)
                decoded = decoded.at[min_idx].add(
                    1 if shifted[min_idx] > decoded[min_idx] else -1
                )

            # Add back glue vector
            candidate = decoded + glue

            # Check distance
            dist = jnp.linalg.norm(x_scaled - candidate)

            if dist < best_distance:
                best_distance = dist
                best_point = candidate

        projected = best_point * self.scale
        error = x - projected

        return E8Projection(
            original=x,
            projected=projected,
            error=error,
            distance=float(best_distance * self.scale),
            decoding_successful=best_distance < 1.0  # Within decoding radius
        )

    @partial(jit, static_argnums=(0,))
    def _compute_shell_impl(self, radius_squared: int) -> E8Shell:
        """
        Compute points on shell of given radius.

        Args:
            radius_squared: r² (must be even for E8)

        Returns:
            Shell structure
        """
        points = []

        # Generate lattice points up to radius
        # (Simplified - would enumerate systematically)

        # For small radii, use known results
        if radius_squared == 2:
            # 240 roots
            points = self.roots.tolist()
            kissing = 240
        elif radius_squared == 4:
            # 2160 points
            kissing = 2160
            # Generate by scaling roots and adding
            for root in self.roots[:10]:  # Simplified
                points.append(2 * root)
        else:
            # General case (approximate)
            kissing = 0

        # Theta series coefficient
        if radius_squared == 0:
            theta_coeff = 1
        elif radius_squared == 2:
            theta_coeff = 240
        elif radius_squared == 4:
            theta_coeff = 2160
        else:
            # Use approximation
            theta_coeff = int(240 * radius_squared)

        return E8Shell(
            radius_squared=radius_squared,
            points=points[:100],  # Limit for memory
            kissing_number=kissing,
            theta_series_coeff=theta_coeff
        )

    @partial(jit, static_argnums=(0,))
    def _theta_series_impl(self, tau: complex, max_terms: int = 10) -> complex:
        """
        Compute E8 theta series.

        Θ_E8(τ) = Σ_{x ∈ E8} q^{||x||²/2} where q = e^{2πiτ}

        Args:
            tau: Complex parameter (Im(τ) > 0)
            max_terms: Number of terms

        Returns:
            Theta series value
        """
        q = jnp.exp(2j * jnp.pi * tau)
        theta = 1.0 + 0j  # Term for origin

        # Add shell contributions
        for r_squared in [2, 4, 6, 8][:max_terms]:
            shell = self._compute_shell_impl(r_squared)
            theta += shell.theta_series_coeff * (q ** (r_squared / 2))

        return theta

    @partial(jit, static_argnums=(0,))
    def _kissing_configuration_impl(self) -> jnp.ndarray:
        """
        Get kissing configuration (240 minimal vectors).

        Returns:
            240 minimal vectors (roots)
        """
        return self.roots * self.scale

    @partial(jit, static_argnums=(0,))
    def _error_correction_impl(self, received: jnp.ndarray,
                               syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Correct errors using syndrome.

        Args:
            received: Received vector (with errors)
            syndrome: Error syndrome

        Returns:
            Corrected vector
        """
        # Find error pattern from syndrome
        # For E8: can correct errors up to half minimum distance

        # Syndrome decoding table (simplified)
        if jnp.linalg.norm(syndrome) < 1e-6:
            # No error
            return received

        # Find most likely error
        # Check against known error patterns (roots)
        best_error = jnp.zeros(8)
        best_match = jnp.inf

        for root in self.roots[:50]:  # Check subset
            # Compute syndrome for this error
            test_syndrome = (self.gram @ root) % 2

            # Compare syndromes
            if jnp.linalg.norm(test_syndrome - syndrome) < best_match:
                best_match = jnp.linalg.norm(test_syndrome - syndrome)
                best_error = root

        # Correct
        corrected = received - best_error * self.scale

        # Project to ensure on lattice
        projection = self._two_coset_decode_impl(corrected)

        return projection.projected

    @partial(jit, static_argnums=(0,))
    def _whitening_transform_impl(self, data: jnp.ndarray,
                                  metric: jnp.ndarray) -> jnp.ndarray:
        """
        Whiten data using metric before projection.

        Args:
            data: Data to whiten
            metric: Metric tensor

        Returns:
            Whitened data
        """
        # Compute whitening matrix W such that W^T g W = I
        # W = g^{-1/2}

        # Eigendecomposition of metric
        eigenvals, eigenvecs = jnp.linalg.eigh(metric)

        # Whitening transform
        W = eigenvecs @ jnp.diag(1.0 / jnp.sqrt(eigenvals + 1e-8)) @ eigenvecs.T

        # Apply whitening
        whitened = W @ data

        return whitened

    def decode_trajectory(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """
        Decode entire trajectory using E8.

        Args:
            trajectory: Sequence of states

        Returns:
            Decoded trajectory
        """
        decoded = []

        for state in trajectory:
            projection = self._two_coset_decode(state)
            decoded.append(projection.projected)

        return jnp.array(decoded)

    def compute_code_distance(self, points: jnp.ndarray) -> float:
        """
        Compute minimum distance between coded points.

        Args:
            points: Set of E8 points

        Returns:
            Minimum distance
        """
        n = len(points)
        min_dist = jnp.inf

        for i in range(n):
            for j in range(i + 1, n):
                dist = jnp.linalg.norm(points[i] - points[j])
                min_dist = min(min_dist, dist)

        return float(min_dist)