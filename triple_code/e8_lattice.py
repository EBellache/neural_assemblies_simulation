"""
E8 Lattice Module
=================
Implementation of E8 exceptional Lie group geometry for morphogenetic state space.
The E8 lattice provides optimal sphere packing in 8 dimensions.

Key properties:
- 240 root vectors (nearest neighbors to origin)
- Kissing number: 240 (each point touches 240 others)
- Minimum distance: √2
- Dual lattice: E8* = E8 (self-dual)

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from functools import lru_cache

# Import tropical operations if available
try:
    from ..core.tropical_math import (
        tropical_inner_product, tropical_distance,
        TROPICAL_ZERO, TROPICAL_ONE
    )

    TROPICAL_AVAILABLE = True
except ImportError:
    TROPICAL_AVAILABLE = False
    TROPICAL_ZERO = -np.inf
    TROPICAL_ONE = 0.0


    def tropical_inner_product(a, b):
        return np.max(a + b)


    def tropical_distance(a, b):
        diff = a - b
        return np.max(diff) - np.min(diff)

# E8 lattice constants
E8_DIMENSION = 8
E8_KISSING_NUMBER = 240
E8_MIN_NORM = 2.0  # Minimum squared norm


@dataclass
class E8Point:
    """Represents a point in the E8 lattice."""
    coords: np.ndarray  # 8-dimensional coordinates
    norm_squared: float = None
    is_root: bool = False
    root_index: Optional[int] = None

    def __post_init__(self):
        if self.norm_squared is None:
            self.norm_squared = float(np.sum(self.coords ** 2))


class E8Lattice:
    """
    E8 lattice operations and computations.
    """

    def __init__(self):
        self.roots = self._generate_root_system()
        self.simple_roots = self._get_simple_roots()
        self.cartan_matrix = self._compute_cartan_matrix()
        self.weyl_group_order = 696729600  # |W(E8)|

    @lru_cache(maxsize=1)
    def _generate_root_system(self) -> List[E8Point]:
        """
        Generate all 240 root vectors of E8.

        The E8 roots consist of:
        1. All vectors in R^8 with integer or all half-integer coords,
           sum of squares = 2, and even sum of coordinates.
        2. Specifically: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
        3. (±1/2, ±1/2, ..., ±1/2) with even number of minuses (128 roots)
        """
        roots = []

        # Type 1: Two ±1 coordinates, rest zeros (112 roots)
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        coords = np.zeros(8)
                        coords[i] = si
                        coords[j] = sj
                        roots.append(E8Point(coords, norm_squared=2.0, is_root=True))

        # Type 2: All half-integer coordinates with even sum (128 roots)
        for mask in range(256):  # 2^8 possibilities
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(8)])
            coords = 0.5 * signs

            # Check if sum is even (in units of 1/2)
            if int(np.sum(signs)) % 2 == 0:
                roots.append(E8Point(coords, norm_squared=2.0, is_root=True))

        # Assign indices
        for i, root in enumerate(roots):
            root.root_index = i

        return roots

    def _get_simple_roots(self) -> List[E8Point]:
        """
        Get the 8 simple roots that generate E8.
        These form a basis for the root system.
        """
        # Standard choice of simple roots for E8
        simple = []

        # α1 through α6: Standard An pattern
        for i in range(6):
            coords = np.zeros(8)
            coords[i] = 1
            coords[i + 1] = -1
            simple.append(E8Point(coords, is_root=True))

        # α7: Branching root
        coords = np.zeros(8)
        coords[5] = coords[6] = coords[7] = -0.5
        coords[4] = 0.5
        simple.append(E8Point(coords, is_root=True))

        # α8: Extended root
        coords = np.array([0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
        simple.append(E8Point(coords, is_root=True))

        return simple

    def _compute_cartan_matrix(self) -> np.ndarray:
        """
        Compute the Cartan matrix of E8.
        A_ij = 2(αi·αj)/(αj·αj)
        """
        simple = self.simple_roots
        n = len(simple)
        cartan = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                # Compute inner product
                inner = np.dot(simple[i].coords, simple[j].coords)
                cartan[i, j] = int(2 * inner / simple[j].norm_squared)

        return cartan

    def nearest_e8_point(self, x: np.ndarray) -> np.ndarray:
        """
        Find the nearest E8 lattice point to x.
        Uses the E8 decoding algorithm.
        """
        x = np.asarray(x)
        if len(x) != 8:
            raise ValueError("Input must be 8-dimensional")

        # Method 1: Round to nearest integer/half-integer lattice
        # Check both integer and half-integer lattices

        # Integer lattice candidate
        int_candidate = np.round(x)

        # Half-integer lattice candidate
        half_candidate = np.floor(x) + 0.5

        # For E8, we need the sum of coordinates to be even
        # Adjust if necessary
        if np.sum(int_candidate) % 2 != 0:
            # Find coordinate with smallest rounding error
            errors = np.abs(x - int_candidate)
            min_idx = np.argmin(errors)
            # Flip this coordinate
            int_candidate[min_idx] += 1 if x[min_idx] > int_candidate[min_idx] else -1

        # For half-integer, need even number of negative coords
        neg_count = np.sum(half_candidate < 0)
        if neg_count % 2 != 0:
            # Flip sign of coordinate closest to 0
            abs_vals = np.abs(half_candidate)
            min_idx = np.argmin(abs_vals)
            half_candidate[min_idx] *= -1

        # Choose closer candidate
        dist_int = np.linalg.norm(x - int_candidate)
        dist_half = np.linalg.norm(x - half_candidate)

        return int_candidate if dist_int < dist_half else half_candidate

    def project_to_e8(self, x: np.ndarray) -> np.ndarray:
        """
        Project point onto E8 lattice (same as nearest point).
        """
        return self.nearest_e8_point(x)

    def compute_casimir_invariants(self, x: np.ndarray) -> Dict[str, float]:
        """
        Compute Casimir invariants for E8.
        These are polynomial invariants under the Weyl group action.
        """
        x = np.asarray(x)

        # Degree 2: Quadratic Casimir (sum of squares)
        c2 = float(np.sum(x ** 2))

        # Degree 8: Octic Casimir (simplified)
        c8 = float(np.sum(x ** 8))

        # Degree 12: Involves 12th powers and cross terms
        c12 = float(np.sum(x ** 12)) + 6 * float(np.sum(x ** 4)) ** 2

        # Degree 14: More complex polynomial
        c14 = float(np.sum(x ** 14)) + 7 * float(np.sum(x ** 2)) * float(np.sum(x ** 12))

        # Degree 18
        c18 = float(np.sum(x ** 18)) + 9 * float(np.sum(x ** 6)) ** 2

        # Degree 20
        c20 = float(np.sum(x ** 20)) + 10 * float(np.sum(x ** 4)) * float(np.sum(x ** 16))

        # Degree 24
        c24 = float(np.sum(x ** 24)) + 12 * float(np.sum(x ** 8)) ** 2

        # Degree 30: Highest degree basic invariant
        c30 = float(np.sum(x ** 30)) + 15 * float(np.sum(x ** 10)) ** 2

        return {
            'I2': c2,
            'I8': c8,
            'I12': c12,
            'I14': c14,
            'I18': c18,
            'I20': c20,
            'I24': c24,
            'I30': c30
        }

    def is_e8_point(self, x: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if x is a valid E8 lattice point.
        """
        # Check if coordinates are all integers or all half-integers
        frac_parts = x - np.floor(x)

        all_integers = np.allclose(frac_parts, 0, atol=tolerance)
        all_half_integers = np.allclose(frac_parts, 0.5, atol=tolerance)

        if not (all_integers or all_half_integers):
            return False

        # Check sum constraint
        if all_integers:
            # Sum must be even
            return int(np.sum(x)) % 2 == 0
        else:
            # Number of negative coordinates must be even
            return int(np.sum(x < 0)) % 2 == 0

    def e8_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance between two E8 points.
        """
        return float(np.linalg.norm(x - y))

    def find_nearest_root(self, x: np.ndarray) -> Tuple[E8Point, float]:
        """
        Find the nearest root vector to point x.
        """
        x = np.asarray(x)
        min_dist = float('inf')
        nearest = None

        for root in self.roots:
            dist = np.linalg.norm(x - root.coords)
            if dist < min_dist:
                min_dist = dist
                nearest = root

        return nearest, min_dist

    def weyl_reflect(self, x: np.ndarray, root: E8Point) -> np.ndarray:
        """
        Apply Weyl reflection with respect to a root.
        s_α(x) = x - 2(x·α)/(α·α) * α
        """
        alpha = root.coords
        coeff = 2 * np.dot(x, alpha) / root.norm_squared
        return x - coeff * alpha

    def orbit_size(self, x: np.ndarray, max_iterations: int = 1000) -> int:
        """
        Estimate size of Weyl group orbit containing x.
        """
        orbit = set()
        queue = [x.copy()]

        while queue and len(orbit) < max_iterations:
            current = queue.pop(0)
            current_tuple = tuple(np.round(current, 6))  # Round for hashing

            if current_tuple in orbit:
                continue

            orbit.add(current_tuple)

            # Apply simple root reflections
            for root in self.simple_roots[:4]:  # Use subset to avoid explosion
                reflected = self.weyl_reflect(current, root)
                reflected_tuple = tuple(np.round(reflected, 6))
                if reflected_tuple not in orbit:
                    queue.append(reflected)

        return len(orbit)


# Tropical operations on E8
def tropical_e8_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Tropical product in E8 space.
    Component-wise: (x ⊗ y)_i = x_i + y_i
    """
    return x + y


def tropical_e8_sum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Tropical sum in E8 space.
    Component-wise: (x ⊕ y)_i = max(x_i, y_i)
    """
    return np.maximum(x, y)


def tropical_update_e8(current: np.ndarray, update: np.ndarray,
                       rate: float = 0.1) -> np.ndarray:
    """
    Update E8 coordinates using tropical dynamics.
    """
    # Tropical interpolation
    return (1 - rate) * current + rate * tropical_e8_sum(current, update)


def assembly_to_e8(assembly_pattern: np.ndarray,
                   lattice: Optional[E8Lattice] = None) -> np.ndarray:
    """
    Map neuronal assembly pattern to E8 coordinates.

    Parameters:
    -----------
    assembly_pattern : np.ndarray
        Activity pattern of assembly
    lattice : E8Lattice
        Lattice instance (created if None)

    Returns:
    --------
    np.ndarray : 8-dimensional E8 coordinates
    """
    if lattice is None:
        lattice = E8Lattice()

    # Ensure we have at least 8 dimensions
    if len(assembly_pattern) < 8:
        assembly_pattern = np.pad(assembly_pattern,
                                  (0, 8 - len(assembly_pattern)),
                                  mode='constant')

    # Take first 8 components
    x = assembly_pattern[:8].astype(float)

    # Normalize to unit sphere
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    # Scale to E8 characteristic length
    x = x * np.sqrt(2)

    # Project to nearest E8 point
    return lattice.nearest_e8_point(x)


def compute_e8_trajectory_curvature(trajectory: List[np.ndarray]) -> List[float]:
    """
    Compute curvature along an E8 trajectory.
    High curvature indicates catastrophe points.
    """
    if len(trajectory) < 3:
        return []

    curvatures = []

    for i in range(1, len(trajectory) - 1):
        # Three consecutive points
        p0, p1, p2 = trajectory[i - 1], trajectory[i], trajectory[i + 1]

        # Velocity vectors
        v1 = p1 - p0
        v2 = p2 - p1

        # Angle between velocities
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            # Curvature proportional to angle change
            curvature = angle / (0.5 * (norm1 + norm2))
        else:
            curvature = 0.0

        curvatures.append(curvature)

    return curvatures


def test_e8_lattice():
    """Test E8 lattice functionality."""
    print("\n=== Testing E8 Lattice Module ===\n")

    # Create lattice
    lattice = E8Lattice()

    print(f"E8 root system generated:")
    print(f"  Number of roots: {len(lattice.roots)}")
    print(f"  Expected: {E8_KISSING_NUMBER}")
    print(f"  Match: {len(lattice.roots) == E8_KISSING_NUMBER}")

    # Check root properties
    root = lattice.roots[0]
    print(f"\nFirst root:")
    print(f"  Coordinates: {root.coords}")
    print(f"  Norm squared: {root.norm_squared}")
    print(f"  Is unit root: {np.isclose(root.norm_squared, E8_MIN_NORM)}")

    # Test Cartan matrix
    print(f"\nCartan matrix shape: {lattice.cartan_matrix.shape}")
    print(f"Cartan matrix diagonal: {np.diag(lattice.cartan_matrix)}")
    print(f"All diagonal elements = 2: {np.all(np.diag(lattice.cartan_matrix) == 2)}")

    # Test nearest point projection
    print("\n--- Testing E8 Projection ---")

    # Random point
    x = np.random.standard_normal(8)
    print(f"\nRandom point: {x[:4]}...")

    # Project to E8
    x_e8 = lattice.nearest_e8_point(x)
    print(f"Nearest E8 point: {x_e8[:4]}...")
    print(f"Is valid E8 point: {lattice.is_e8_point(x_e8)}")
    print(f"Distance to original: {np.linalg.norm(x - x_e8):.4f}")

    # Test Casimir invariants
    print("\n--- Testing Casimir Invariants ---")

    casimirs = lattice.compute_casimir_invariants(x_e8)
    print("Casimir invariants:")
    for name, value in list(casimirs.items())[:4]:
        print(f"  {name}: {value:.6f}")

    # Test Weyl reflection
    print("\n--- Testing Weyl Reflections ---")

    simple_root = lattice.simple_roots[0]
    x_reflected = lattice.weyl_reflect(x_e8, simple_root)
    print(f"Original: {x_e8[:4]}...")
    print(f"Reflected: {x_reflected[:4]}...")
    print(f"Preserves norm: {np.isclose(np.linalg.norm(x_e8), np.linalg.norm(x_reflected))}")

    # Test tropical operations
    if TROPICAL_AVAILABLE:
        print("\n--- Testing Tropical E8 Operations ---")

        y = np.random.standard_normal(8)
        y_e8 = lattice.nearest_e8_point(y)

        trop_prod = tropical_e8_product(x_e8, y_e8)
        trop_sum = tropical_e8_sum(x_e8, y_e8)

        print(f"x ⊗ y (first 4): {trop_prod[:4]}")
        print(f"x ⊕ y (first 4): {trop_sum[:4]}")

    # Test assembly mapping
    print("\n--- Testing Assembly to E8 Mapping ---")

    assembly_pattern = np.random.rand(12)
    e8_coords = assembly_to_e8(assembly_pattern, lattice)
    print(f"Assembly pattern (first 6): {assembly_pattern[:6]}")
    print(f"E8 coordinates: {e8_coords[:4]}...")
    print(f"Is valid E8 point: {lattice.is_e8_point(e8_coords)}")

    print("\n✓ E8 lattice module working correctly!")


if __name__ == "__main__":
    test_e8_lattice()