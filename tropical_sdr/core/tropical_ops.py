"""Tropical arithmetic operations using JAX.

The tropical semiring (ℝ ∪ {-∞}, max, +) forms the mathematical foundation
for neural competition dynamics. These operations naturally implement
winner-take-all computations.
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from functools import partial

# Tropical -∞ represented as large negative number
TROPICAL_NEG_INF = -1e10


@jax.jit
def tropical_add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Tropical addition: x ⊕ y = max(x, y).
    
    This implements the fundamental winner-take-all operation.
    
    Args:
        x: First operand
        y: Second operand
        
    Returns:
        Maximum of x and y (element-wise for arrays)
    """
    return jnp.maximum(x, y)


@jax.jit
def tropical_mul(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Tropical multiplication: x ⊙ y = x + y.
    
    In log space, multiplication becomes addition.
    
    Args:
        x: First operand
        y: Second operand
        
    Returns:
        Sum of x and y (element-wise for arrays)
    """
    # Handle tropical -∞ properly
    mask = (x > TROPICAL_NEG_INF) & (y > TROPICAL_NEG_INF)
    result = jnp.where(mask, x + y, TROPICAL_NEG_INF)
    return result


@jax.jit
def tropical_dot(vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
    """Tropical inner product: ⊕ᵢ(vec1[i] ⊙ vec2[i]).
    
    Computes max over element-wise sums.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Tropical inner product (scalar)
    """
    products = tropical_mul(vec1, vec2)
    return jnp.max(products)


@jax.jit
def tropical_distance(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Tropical distance: max|xᵢ - yᵢ|.
    
    The tropical metric for comparing states.
    
    Args:
        x: First point
        y: Second point
        
    Returns:
        Tropical distance between x and y
    """
    return jnp.max(jnp.abs(x - y))


@jax.jit
def tropical_polynomial(coeffs: jnp.ndarray, 
                       powers: jnp.ndarray,
                       x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate tropical polynomial at point x.
    
    P(x) = ⊕ᵢ(coeffs[i] ⊙ x^powers[i])
         = max_i(coeffs[i] + powers[i] * x)
    
    Args:
        coeffs: Polynomial coefficients
        powers: Corresponding powers
        x: Evaluation point(s)
        
    Returns:
        Polynomial value at x
    """
    # Reshape for broadcasting if needed
    if x.ndim == 1:
        x = x[:, None]
    if coeffs.ndim == 1:
        coeffs = coeffs[None, :]
    if powers.ndim == 1:
        powers = powers[None, :]
        
    terms = coeffs + powers * x
    return jnp.max(terms, axis=-1)


@partial(jax.jit, static_argnames=['n'])
def tropical_power(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Tropical power: x^n = n ⊙ x = n * x.
    
    Args:
        x: Base
        n: Exponent (must be non-negative integer)
        
    Returns:
        Tropical power x^n
    """
    return n * x


@jax.jit 
def tropical_matrix_mult(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Tropical matrix multiplication.
    
    (A ⊙ B)[i,j] = ⊕ₖ(A[i,k] ⊙ B[k,j])
                 = max_k(A[i,k] + B[k,j])
    
    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)
        
    Returns:
        Tropical product (m x p)
    """
    # Add dimensions for broadcasting
    A_expanded = A[:, :, None]  # (m, n, 1)
    B_expanded = B[None, :, :]  # (1, n, p)
    
    # Tropical multiplication (addition)
    products = A_expanded + B_expanded  # (m, n, p)
    
    # Tropical addition (maximum) over middle dimension
    result = jnp.max(products, axis=1)  # (m, p)
    
    return result


@jax.jit
def tropical_relu(x: jnp.ndarray) -> jnp.ndarray:
    """Tropical ReLU: implements max(x, 0) naturally.
    
    This shows ReLU networks compute tropical polynomials.
    
    Args:
        x: Input
        
    Returns:
        Tropical ReLU of x
    """
    return tropical_add(x, jnp.zeros_like(x))


# Newton polytope operations
@jax.jit
def newton_polytope_vertices(polynomial_powers: jnp.ndarray) -> jnp.ndarray:
    """Extract vertices of Newton polytope from polynomial.
    
    The Newton polytope is the convex hull of the exponent vectors.
    
    Args:
        polynomial_powers: Array of exponent vectors (n_terms x dim)
        
    Returns:
        Vertices of the Newton polytope
    """
    # For now, return unique powers (vertices are subset)
    # Full convex hull computation would use scipy.spatial.ConvexHull
    return jnp.unique(polynomial_powers, axis=0)


@jax.jit
def tropical_argmax(x: jnp.ndarray) -> Tuple[int, float]:
    """Tropical argmax with value.
    
    Returns both the index and value of maximum.
    
    Args:
        x: Input array
        
    Returns:
        (index, value) of tropical maximum
    """
    idx = jnp.argmax(x)
    val = x[idx]
    return idx, val