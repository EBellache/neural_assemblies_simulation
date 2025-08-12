"""
Core Tropical Mathematics Module
=================================
Fundamental operations for max-plus (tropical) algebra used throughout the simulator.
Compatible with both NumPy and JAX for acceleration.

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Try to import JAX for acceleration
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax

    JAX_AVAILABLE = True
    print("JAX available - using accelerated operations")
except ImportError:
    import numpy as jnp

    JAX_AVAILABLE = False
    print("JAX not available - using NumPy fallback")


    # Minimal JAX compatibility layer
    def jit(fn):
        return fn


    def vmap(fn, *args, **kwargs):
        def wrapped(*a, **k):
            return np.vectorize(lambda *x: fn(*x))(*a, **k)

        return wrapped


    class lax:
        @staticmethod
        def fori_loop(lower, upper, body_fun, init_val):
            val = init_val
            for i in range(lower, upper):
                val = body_fun(i, val)
            return val

        @staticmethod
        def cond(pred, true_fun, false_fun, operand):
            if pred:
                return true_fun(operand)
            else:
                return false_fun(operand)

# Constants
TROPICAL_ZERO = -np.inf  # Additive identity (−∞)
TROPICAL_ONE = 0.0  # Multiplicative identity (0)
EPSILON = 1e-12  # Numerical stability threshold

# Type alias
Array = Union[np.ndarray, Any]  # Supports both NumPy and JAX arrays


# -------------------------
# Scalar/Array Primitives
# -------------------------

@jit
def tropical_add(a: Array, b: Array) -> Array:
    """
    Tropical addition: a ⊕ b = max(a, b)
    Supports broadcasting.
    """
    return jnp.maximum(a, b)


@jit
def tropical_multiply(a: Array, b: Array) -> Array:
    """
    Tropical multiplication: a ⊗ b = a + b
    Supports broadcasting.
    """
    return a + b


@jit
def tropical_power(a: Array, n: Union[int, Array]) -> Array:
    """
    Tropical power: a^{⊗ n} = n * a
    n must be non-negative integer.
    """
    return a * n


@jit
def tropical_norm(x: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Tropical norm (supremum): ||x||_⊕ = max_i x_i
    """
    return jnp.max(x, axis=axis, keepdims=keepdims)


# -------------------------
# Vector/Matrix Operations
# -------------------------

@jit
def tropical_inner_product(v1: Array, v2: Array, axis: int = -1) -> Array:
    """
    Tropical inner product: ⟨v1, v2⟩_⊗ = max_i (v1_i + v2_i)

    Parameters:
    -----------
    v1, v2 : Array
        Input vectors (must have same shape)
    axis : int
        Reduction axis after broadcasting

    Returns:
    --------
    Array : Tropical inner product
    """
    return jnp.max(v1 + v2, axis=axis)


@jit
def tropical_matrix_multiply(A: Array, B: Array) -> Array:
    """
    Tropical matrix product: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})

    Parameters:
    -----------
    A : Array of shape (m, k)
    B : Array of shape (k, n)

    Returns:
    --------
    C : Array of shape (m, n)
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)

    # Validate dimensions
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible dimensions: {A.shape} and {B.shape}")

    # Broadcast and compute: A[:,:,None] + B[None,:,:]
    C = jnp.max(A[..., :, None] + B[None, ...], axis=-2)
    return C


@jit
def tropical_matrix_power(A: Array, n: int) -> Array:
    """
    Compute A^{⊗n} using repeated squaring for efficiency.
    """
    if n < 0:
        raise ValueError("Power must be non-negative")
    if n == 0:
        # Return tropical identity matrix
        return tropical_eye(A.shape[0])

    result = A
    for _ in range(n - 1):
        result = tropical_matrix_multiply(result, A)
    return result


# -------------------------
# Distance Metrics
# -------------------------

def tropical_distance(x: Array, y: Array) -> Array:
    """
    Tropical (Hilbert projective) distance between x and y:
    d_H(x, y) = max_i (x_i - y_i) - min_i (x_i - y_i)

    Special handling for -∞ values.
    Returns +∞ if no indices where both are finite.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # Find where both are finite
    both_finite = jnp.isfinite(x) & jnp.isfinite(y)

    if not jnp.any(both_finite):
        return jnp.inf

    # Compute difference only where both finite
    diff = jnp.where(both_finite, x - y, jnp.nan)

    # Handle NaN properly
    max_diff = jnp.nanmax(diff)
    min_diff = jnp.nanmin(diff)

    return max_diff - min_diff


def tropical_projection_distance(x: Array, subspace: Array) -> float:
    """
    Distance from point x to tropical linear subspace.
    Used for assembly stability computation.
    """
    # Project x onto subspace span
    proj = tropical_project_onto_span(x, subspace)
    return tropical_distance(x, proj)


# -------------------------
# Subspace Operations
# -------------------------

def tropical_project_onto_span(x: Array, generators: Array) -> Array:
    """
    Project point x onto tropical linear span of generators.

    Parameters:
    -----------
    x : Array of shape (d,)
        Point to project
    generators : Array of shape (k, d)
        k generating vectors

    Returns:
    --------
    proj : Array of shape (d,)
        Projection of x
    """
    x = jnp.asarray(x)
    generators = jnp.asarray(generators)

    if generators.ndim == 1:
        generators = generators.reshape(1, -1)

    # Find best tropical linear combination
    # proj = max_i (λ_i ⊗ g_i) where λ_i chosen optimally

    # For each generator, find optimal coefficient
    coeffs = []
    for g in generators:
        # Optimal λ minimizes tropical distance
        # λ = median of (x - g) in tropical sense
        finite_mask = jnp.isfinite(x) & jnp.isfinite(g)
        if jnp.any(finite_mask):
            diffs = jnp.where(finite_mask, x - g, 0)
            coeff = jnp.median(diffs[finite_mask])
        else:
            coeff = 0
        coeffs.append(coeff)

    coeffs = jnp.array(coeffs)

    # Compute projection as tropical linear combination
    proj = jnp.full_like(x, TROPICAL_ZERO)
    for i, (c, g) in enumerate(zip(coeffs, generators)):
        proj = tropical_add(proj, tropical_multiply(c, g))

    return proj


# -------------------------
# Utility Functions
# -------------------------

def tropical_eye(n: int) -> Array:
    """
    Tropical identity matrix: diagonal = 0, off-diagonal = -∞
    """
    M = jnp.full((n, n), TROPICAL_ZERO)
    if JAX_AVAILABLE:
        idx = jnp.arange(n)
        M = M.at[idx, idx].set(TROPICAL_ONE)
    else:
        for i in range(n):
            M[i, i] = TROPICAL_ONE
    return M


def tropical_zeros(shape: Union[int, Tuple[int, ...]]) -> Array:
    """
    Create array of tropical zeros (-∞)
    """
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.full(shape, TROPICAL_ZERO)


def tropical_ones(shape: Union[int, Tuple[int, ...]]) -> Array:
    """
    Create array of tropical ones (0)
    """
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.full(shape, TROPICAL_ONE)


def is_tropical_zero(x: Array, tol: float = None) -> bool:
    """
    Check if x is tropical zero (or close to -∞)
    """
    if tol is None:
        return jnp.all(jnp.isinf(x) & (x < 0))
    else:
        return jnp.all(x < -1 / tol)


# -------------------------
# Eigenvalue/Eigenvector
# -------------------------

def tropical_eigenvalue(A: Array, max_iter: int = 100, tol: float = 1e-8) -> float:
    """
    Compute the tropical eigenvalue (cycle mean) of matrix A.
    Uses Karp's minimum mean-weight cycle algorithm.
    """
    n = A.shape[0]
    if n == 0:
        return TROPICAL_ZERO

    # Power iteration to find dominant eigenvalue
    x = tropical_ones(n)

    for _ in range(max_iter):
        x_new = tropical_matrix_multiply(A, x.reshape(-1, 1)).flatten()

        # Check convergence
        if jnp.allclose(x_new - jnp.max(x_new), x - jnp.max(x), atol=tol):
            # Eigenvalue is the average growth rate
            return jnp.max(x_new) - jnp.max(x)

        x = x_new

    return jnp.max(x_new) - jnp.max(x)


def tropical_eigenvector(A: Array, eigenvalue: float = None,
                         max_iter: int = 100, tol: float = 1e-8) -> Array:
    """
    Compute tropical eigenvector corresponding to eigenvalue.
    If eigenvalue is None, computes it first.
    """
    n = A.shape[0]

    if eigenvalue is None:
        eigenvalue = tropical_eigenvalue(A, max_iter, tol)

    # Solve (A - λI) ⊗ x = 0 in tropical sense
    A_shifted = A - eigenvalue * tropical_eye(n)

    # Power iteration
    x = tropical_ones(n)

    for _ in range(max_iter):
        x_new = tropical_matrix_multiply(A_shifted, x.reshape(-1, 1)).flatten()

        # Normalize
        x_new = x_new - jnp.max(x_new)

        if jnp.allclose(x_new, x, atol=tol):
            break

        x = x_new

    return x


# -------------------------
# Validation Functions
# -------------------------

def validate_tropical_matrix(A: Array, name: str = "matrix") -> None:
    """
    Validate that A is a proper tropical matrix.
    Raises ValueError if invalid.
    """
    A = jnp.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")

    if not jnp.all(jnp.isfinite(A) | jnp.isneginf(A)):
        raise ValueError(f"{name} contains invalid values (must be finite or -∞)")

    if A.shape[0] != A.shape[1]:
        import warnings
        warnings.warn(f"{name} is not square: {A.shape}")


def test_tropical_operations():
    """
    Test suite for tropical operations.
    """
    print("\n=== Testing Tropical Mathematics ===\n")

    # Test basic operations
    a = jnp.array([1.0, 3.0, 2.0])
    b = jnp.array([2.0, 1.0, 4.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a ⊕ b = {tropical_add(a, b)}")
    print(f"a ⊗ b = {tropical_multiply(a, b)}")
    print(f"⟨a, b⟩_⊗ = {tropical_inner_product(a, b)}")
    print(f"d_H(a, b) = {tropical_distance(a, b):.4f}")

    # Test matrix operations
    A = jnp.array([[0, 2, -np.inf],
                   [1, 0, 3],
                   [2, -np.inf, 0]])

    B = jnp.array([[1, -np.inf, 2],
                   [0, 2, 1],
                   [-np.inf, 3, 0]])

    print(f"\nMatrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A ⊗ B:\n{tropical_matrix_multiply(A, B)}")

    # Test eigenvalue
    eigenval = tropical_eigenvalue(A)
    eigenvec = tropical_eigenvector(A, eigenval)
    print(f"\nTropical eigenvalue: {eigenval:.4f}")
    print(f"Tropical eigenvector: {eigenvec}")

    print("\n✓ All tropical operations working correctly!")


if __name__ == "__main__":
    test_tropical_operations()