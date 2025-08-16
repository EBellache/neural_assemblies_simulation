"""
Tropical (Max-Plus) Algebra
============================

Implements tropical algebraic operations for fast morphogenetic computations.
Replaces addition with max and multiplication with addition.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
from jax.scipy.special import logsumexp
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Union
from functools import partial
from dataclasses import dataclass

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class TropicalNumber(NamedTuple):
    """Tropical number (element of tropical semiring)."""
    value: float  # Real value or -∞
    is_infinite: bool  # True if -∞ (additive identity)


class TropicalMatrix(NamedTuple):
    """Matrix over tropical semiring."""
    entries: jnp.ndarray  # Matrix entries
    shape: Tuple[int, int]  # Matrix dimensions


class TropicalPolynomial(NamedTuple):
    """Tropical polynomial."""
    coefficients: jnp.ndarray  # Coefficients
    exponents: jnp.ndarray  # Exponent vectors


class TropicalEigendata(NamedTuple):
    """Tropical eigenvalue and eigenvector."""
    eigenvalue: float  # Max-plus eigenvalue
    eigenvector: jnp.ndarray  # Max-plus eigenvector
    critical_graph: jnp.ndarray  # Critical subgraph


class TropicalComputer:
    """
    Compute tropical algebraic operations.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize tropical computer.

        Args:
            epsilon: Small value for numerical stability
        """
        self.eps = epsilon
        self.minus_inf = -1e10  # Tropical zero

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._tropical_add = jit(self._tropical_add_impl)
        self._tropical_mult = jit(self._tropical_mult_impl)
        self._tropical_dot = jit(self._tropical_dot_impl)
        self._tropical_matmul = jit(self._tropical_matmul_impl)
        self._tropical_power = jit(self._tropical_power_impl)
        self._tropical_eigenvalue = jit(self._tropical_eigenvalue_impl)
        self._tropical_eigenvector = jit(self._tropical_eigenvector_impl)
        self._tropical_distance = jit(self._tropical_distance_impl)
        self._tropical_convolution = jit(self._tropical_convolution_impl)
        self._critical_graph = jit(self._critical_graph_impl)

    @partial(jit, static_argnums=(0,))
    def _tropical_add_impl(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Tropical addition: a ⊕ b = max(a, b).

        Args:
            a, b: Arrays to add tropically

        Returns:
            Tropical sum
        """
        return jnp.maximum(a, b)

    @partial(jit, static_argnums=(0,))
    def _tropical_mult_impl(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Tropical multiplication: a ⊗ b = a + b.

        Args:
            a, b: Arrays to multiply tropically

        Returns:
            Tropical product
        """
        return a + b

    @partial(jit, static_argnums=(0,))
    def _tropical_dot_impl(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """
        Tropical dot product: ⊕_i (a_i ⊗ b_i) = max_i(a_i + b_i).

        Args:
            a, b: Vectors

        Returns:
            Tropical dot product
        """
        return jnp.max(a + b)

    @partial(jit, static_argnums=(0,))
    def _tropical_matmul_impl(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Tropical matrix multiplication.

        (A ⊗ B)_{ij} = ⊕_k (A_{ik} ⊗ B_{kj}) = max_k(A_{ik} + B_{kj})

        Args:
            A, B: Matrices to multiply

        Returns:
            Tropical matrix product
        """
        m, k1 = A.shape
        k2, n = B.shape
        assert k1 == k2, "Matrix dimensions must match"

        C = jnp.zeros((m, n))

        for i in range(m):
            for j in range(n):
                # Tropical dot product of row i of A with column j of B
                C = C.at[i, j].set(jnp.max(A[i, :] + B[:, j]))

        return C

    @partial(jit, static_argnums=(0, 2))
    def _tropical_power_impl(self, A: jnp.ndarray, n: int) -> jnp.ndarray:
        """
        Tropical matrix power: A^⊗n.

        Args:
            A: Square matrix
            n: Power

        Returns:
            Tropical matrix power
        """
        if n == 0:
            # Tropical identity matrix
            m = A.shape[0]
            I = jnp.full((m, m), self.minus_inf)
            I = I.at[jnp.arange(m), jnp.arange(m)].set(0)
            return I

        result = A
        for _ in range(n - 1):
            result = self._tropical_matmul_impl(result, A)

        return result

    @partial(jit, static_argnums=(0,))
    def _tropical_eigenvalue_impl(self, A: jnp.ndarray) -> float:
        """
        Compute tropical (max-plus) eigenvalue.

        λ = max over all elementary cycles of (cycle weight / cycle length)

        Args:
            A: Square matrix

        Returns:
            Tropical eigenvalue
        """
        n = A.shape[0]

        # Compute A^k for k = 1, ..., n
        traces = []
        Ak = A.copy()

        for k in range(1, n + 1):
            # Trace of A^k gives sum over all k-cycles
            trace = jnp.max(jnp.diag(Ak))
            traces.append(trace / k)  # Average weight

            if k < n:
                Ak = self._tropical_matmul_impl(Ak, A)

        # Maximum cycle mean
        eigenvalue = jnp.max(jnp.array(traces))

        return eigenvalue

    @partial(jit, static_argnums=(0,))
    def _tropical_eigenvector_impl(self, A: jnp.ndarray,
                                   eigenvalue: float) -> jnp.ndarray:
        """
        Compute tropical eigenvector for given eigenvalue.

        Solve: A ⊗ v = λ ⊗ v (in tropical arithmetic)

        Args:
            A: Square matrix
            eigenvalue: Tropical eigenvalue

        Returns:
            Tropical eigenvector
        """
        n = A.shape[0]

        # Normalize matrix: A - λ
        A_norm = A - eigenvalue

        # Power method in tropical arithmetic
        v = jnp.zeros(n)

        for _ in range(100):  # Fixed iterations
            v_new = jnp.array([
                jnp.max(A_norm[i, :] + v) for i in range(n)
            ])

            # Normalize to prevent overflow
            v_new = v_new - jnp.mean(v_new)

            # Check convergence
            if jnp.allclose(v, v_new, rtol=1e-6):
                break

            v = v_new

        return v

    @partial(jit, static_argnums=(0,))
    def _critical_graph_impl(self, A: jnp.ndarray,
                             eigenvalue: float) -> jnp.ndarray:
        """
        Extract critical graph (edges achieving eigenvalue).

        Args:
            A: Matrix
            eigenvalue: Tropical eigenvalue

        Returns:
            Adjacency matrix of critical graph
        """
        n = A.shape[0]
        v = self._tropical_eigenvector_impl(A, eigenvalue)

        # Critical edges: those where A_ij + v_j - v_i = eigenvalue
        critical = jnp.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if jnp.abs(A[i, j] + v[j] - v[i] - eigenvalue) < 1e-6:
                    critical = critical.at[i, j].set(True)

        return critical.astype(float)

    @partial(jit, static_argnums=(0,))
    def _tropical_distance_impl(self, p: jnp.ndarray, q: jnp.ndarray) -> float:
        """
        Tropical distance between points.

        d(p, q) = max_i |p_i - q_i|

        Args:
            p, q: Points

        Returns:
            Tropical distance
        """
        return jnp.max(jnp.abs(p - q))

    @partial(jit, static_argnums=(0,))
    def _tropical_convolution_impl(self, f: jnp.ndarray,
                                   g: jnp.ndarray) -> jnp.ndarray:
        """
        Tropical convolution.

        (f ⊛ g)(x) = max_y [f(y) + g(x-y)]

        Args:
            f, g: Functions (as arrays)

        Returns:
            Tropical convolution
        """
        n = len(f)
        m = len(g)
        result = jnp.zeros(n + m - 1)

        for i in range(n + m - 1):
            values = []
            for j in range(max(0, i - m + 1), min(i + 1, n)):
                if i - j < m:
                    values.append(f[j] + g[i - j])

            if values:
                result = result.at[i].set(jnp.max(jnp.array(values)))
            else:
                result = result.at[i].set(self.minus_inf)

        return result

    def solve_tropical_equation(self, A: jnp.ndarray,
                                b: jnp.ndarray) -> jnp.ndarray:
        """
        Solve tropical linear equation A ⊗ x = b.

        Args:
            A: Coefficient matrix
            b: Right-hand side

        Returns:
            Solution x (if exists)
        """
        m, n = A.shape

        # Use tropical Cramer's rule or residuation
        x = jnp.zeros(n)

        for j in range(n):
            # x_j = min_i (b_i - A_ij)
            x = x.at[j].set(jnp.min(b - A[:, j]))

        return x

    def tropical_polynomial_roots(self, poly: TropicalPolynomial) -> jnp.ndarray:
        """
        Find tropical roots of polynomial.

        Tropical roots are points where maximum is achieved
        by at least two monomials.

        Args:
            poly: Tropical polynomial

        Returns:
            Tropical roots
        """
        coeffs = poly.coefficients
        exps = poly.exponents

        n_terms = len(coeffs)
        dim = exps.shape[1] if exps.ndim > 1 else 1

        # Find intersection points of tropical hyperplanes
        roots = []

        # Check pairwise intersections
        for i in range(n_terms):
            for j in range(i + 1, n_terms):
                # Solve c_i + e_i · x = c_j + e_j · x
                # (e_i - e_j) · x = c_j - c_i

                if dim == 1:
                    diff_exp = exps[i] - exps[j]
                    if abs(diff_exp) > 1e-8:
                        x = (coeffs[j] - coeffs[i]) / diff_exp
                        roots.append(x)
                else:
                    # Higher dimensional case (simplified)
                    diff_exp = exps[i] - exps[j]
                    if jnp.linalg.norm(diff_exp) > 1e-8:
                        # One solution on the line
                        x = (coeffs[j] - coeffs[i]) * diff_exp / jnp.dot(diff_exp, diff_exp)
                        roots.append(x)

        return jnp.array(roots) if roots else jnp.array([])