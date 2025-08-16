"""
Fisher-Souriau Information Metric
==================================

Implements the information-geometric metric on coadjoint orbits,
combining Fisher information with Souriau's geometric thermodynamics.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad, hessian
from jax.scipy import linalg as jax_linalg
from jax.scipy.special import logsumexp
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Callable
from functools import partial
from dataclasses import dataclass

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class InformationMetric(NamedTuple):
    """Fisher-Souriau information metric."""
    metric_tensor: jnp.ndarray  # g_ij
    christoffel_1st: jnp.ndarray  # Γ_ijk
    christoffel_2nd: jnp.ndarray  # Γ^k_ij
    riemann_tensor: Optional[jnp.ndarray] = None  # R^i_jkl
    ricci_tensor: Optional[jnp.ndarray] = None  # R_ij
    scalar_curvature: Optional[float] = None  # R


class MetricState(NamedTuple):
    """State with information metric."""
    coordinates: jnp.ndarray  # Position on manifold
    metric: InformationMetric  # Metric at this point
    entropy: float  # Souriau entropy
    capacity: float  # Information capacity


class Geodesic(NamedTuple):
    """Geodesic path on information manifold."""
    path: jnp.ndarray  # Points along geodesic
    length: float  # Total length
    energy: float  # Energy of path
    parallel_transport: jnp.ndarray  # Parallel transport matrix


class FisherSouriauComputer:
    """
    Compute Fisher-Souriau information metric and geometric quantities.
    """

    def __init__(self, dimension: int = 8, beta: float = 1.0):
        """
        Initialize Fisher-Souriau computer.

        Args:
            dimension: Manifold dimension
            beta: Inverse temperature parameter
        """
        self.dim = dimension
        self.beta = beta

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._fisher_metric = jit(self._fisher_metric_impl)
        self._souriau_metric = jit(self._souriau_metric_impl)
        self._combined_metric = jit(self._combined_metric_impl)
        self._christoffel_symbols = jit(self._christoffel_symbols_impl)
        self._riemann_curvature = jit(self._riemann_curvature_impl)
        self._geodesic_equation = jit(self._geodesic_equation_impl)
        self._parallel_transport = jit(self._parallel_transport_impl)
        self._entropy = jit(self._entropy_impl)
        self._capacity = jit(self._capacity_impl)

    @partial(jit, static_argnums=(0,))
    def _fisher_metric_impl(self, theta: jnp.ndarray,
                            log_prob: Callable) -> jnp.ndarray:
        """
        Compute Fisher information metric.

        g_ij = E[∂_i log p(x|θ) ∂_j log p(x|θ)]

        Args:
            theta: Parameter coordinates
            log_prob: Log probability function

        Returns:
            Fisher metric tensor
        """
        # Compute Hessian of log probability
        hess_log_p = hessian(log_prob)(theta)

        # Fisher metric is negative expected Hessian
        g = -hess_log_p

        # Ensure positive definite
        g = 0.5 * (g + g.T)
        eigvals = jnp.linalg.eigvalsh(g)
        min_eigval = jnp.min(eigvals)
        if min_eigval < 1e-6:
            g = g + (1e-6 - min_eigval) * jnp.eye(self.dim)

        return g

    @partial(jit, static_argnums=(0,))
    def _souriau_metric_impl(self, momentum: jnp.ndarray,
                             hamiltonian: Callable) -> jnp.ndarray:
        """
        Compute Souriau's thermodynamic metric.

        g_ij = -∂²S/∂μ_i∂μ_j where S is entropy

        Args:
            momentum: Momentum coordinates
            hamiltonian: Hamiltonian function

        Returns:
            Souriau metric tensor
        """

        # Entropy S = β(μ·v - H) where v is velocity
        def entropy(mu):
            H = hamiltonian(mu)
            # Velocity from Hamilton's equations
            grad_H = grad(hamiltonian)(mu)
            v = grad_H  # Simplified (full version uses symplectic form)
            return self.beta * (jnp.dot(mu, v) - H)

        # Metric is Hessian of entropy
        g = -hessian(entropy)(momentum)

        # Ensure positive definite
        g = 0.5 * (g + g.T)
        eigvals = jnp.linalg.eigvalsh(g)
        min_eigval = jnp.min(eigvals)
        if min_eigval < 1e-6:
            g = g + (1e-6 - min_eigval) * jnp.eye(self.dim)

        return g

    @partial(jit, static_argnums=(0,))
    def _combined_metric_impl(self, state: jnp.ndarray,
                              log_prob: Callable,
                              hamiltonian: Callable,
                              alpha: float = 0.5) -> jnp.ndarray:
        """
        Combine Fisher and Souriau metrics.

        Args:
            state: Current state
            log_prob: Log probability function
            hamiltonian: Hamiltonian function
            alpha: Mixing parameter (0=pure Fisher, 1=pure Souriau)

        Returns:
            Combined metric tensor
        """
        g_fisher = self._fisher_metric_impl(state, log_prob)
        g_souriau = self._souriau_metric_impl(state, hamiltonian)

        # Convex combination
        g = (1 - alpha) * g_fisher + alpha * g_souriau

        return g

    @partial(jit, static_argnums=(0,))
    def _christoffel_symbols_impl(self, state: jnp.ndarray,
                                  metric_fn: Callable) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Christoffel symbols of the first and second kind.

        Args:
            state: Current state
            metric_fn: Function returning metric at a point

        Returns:
            Christoffel symbols (first kind, second kind)
        """
        g = metric_fn(state)

        # Compute metric derivatives using automatic differentiation
        def metric_component(x, i, j):
            return metric_fn(x)[i, j]

        # First kind: Γ_ijk = 1/2 (∂_k g_ij + ∂_j g_ik - ∂_i g_jk)
        gamma_1 = jnp.zeros((self.dim, self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    dg_ij_dk = grad(lambda x: metric_component(x, i, j))(state)[k]
                    dg_ik_dj = grad(lambda x: metric_component(x, i, k))(state)[j]
                    dg_jk_di = grad(lambda x: metric_component(x, j, k))(state)[i]

                    gamma_1 = gamma_1.at[i, j, k].set(
                        0.5 * (dg_ij_dk + dg_ik_dj - dg_jk_di)
                    )

        # Second kind: Γ^k_ij = g^{kl} Γ_lij
        g_inv = jnp.linalg.inv(g + 1e-8 * jnp.eye(self.dim))
        gamma_2 = jnp.einsum('kl,lij->kij', g_inv, gamma_1)

        return gamma_1, gamma_2

    @partial(jit, static_argnums=(0,))
    def _riemann_curvature_impl(self, state: jnp.ndarray,
                                metric_fn: Callable) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Compute Riemann curvature tensor and derived quantities.

        Args:
            state: Current state
            metric_fn: Metric function

        Returns:
            (Riemann tensor, Ricci tensor, scalar curvature)
        """
        _, gamma = self._christoffel_symbols_impl(state, metric_fn)

        # Riemann tensor: R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
        # Simplified computation for efficiency
        R = jnp.zeros((self.dim, self.dim, self.dim, self.dim))

        # Approximate using finite differences for derivatives
        eps = 1e-4
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        # Simplified: only keep the commutator terms
                        for m in range(self.dim):
                            R = R.at[i, j, k, l].add(
                                gamma[i, m, k] * gamma[m, j, l] -
                                gamma[i, m, l] * gamma[m, j, k]
                            )

        # Ricci tensor: R_ij = R^k_ikj
        ricci = jnp.einsum('kikj->ij', R)

        # Scalar curvature: R = g^{ij} R_ij
        g = metric_fn(state)
        g_inv = jnp.linalg.inv(g + 1e-8 * jnp.eye(self.dim))
        scalar = jnp.einsum('ij,ij->', g_inv, ricci)

        return R, ricci, scalar

    @partial(jit, static_argnums=(0,))
    def _geodesic_equation_impl(self, t: float, y: jnp.ndarray,
                                christoffel_fn: Callable) -> jnp.ndarray:
        """
        Geodesic equation: d²x^i/dt² + Γ^i_jk dx^j/dt dx^k/dt = 0

        Args:
            t: Time parameter
            y: State [position, velocity]
            christoffel_fn: Function returning Christoffel symbols

        Returns:
            Derivatives [velocity, acceleration]
        """
        mid = self.dim
        x = y[:mid]  # Position
        v = y[mid:]  # Velocity

        # Get Christoffel symbols at current position
        _, gamma = christoffel_fn(x)

        # Acceleration: a^i = -Γ^i_jk v^j v^k
        a = -jnp.einsum('ijk,j,k->i', gamma, v, v)

        return jnp.concatenate([v, a])

    @partial(jit, static_argnums=(0,))
    def _parallel_transport_impl(self, vector: jnp.ndarray,
                                 path: jnp.ndarray,
                                 christoffel_fn: Callable) -> jnp.ndarray:
        """
        Parallel transport vector along path.

        Args:
            vector: Initial vector
            path: Points along curve
            christoffel_fn: Christoffel symbol function

        Returns:
            Transported vector
        """
        v = vector.copy()

        for i in range(len(path) - 1):
            x = path[i]
            dx = path[i + 1] - path[i]

            # Get Christoffel symbols
            _, gamma = christoffel_fn(x)

            # Transport equation: dv^i/dt = -Γ^i_jk v^j dx^k/dt
            dv = -jnp.einsum('ijk,j,k->i', gamma, v, dx)

            v = v + dv

        return v

    @partial(jit, static_argnums=(0,))
    def _entropy_impl(self, state: jnp.ndarray,
                      distribution: Callable) -> float:
        """
        Compute Souriau entropy.

        Args:
            state: Current state
            distribution: Probability distribution

        Returns:
            Entropy value
        """
        # Souriau entropy: S = -β⟨H⟩ + log Z
        # where Z is partition function

        # Sample from distribution (simplified)
        probs = distribution(state)

        # Shannon entropy as approximation
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))

        # Add thermodynamic correction
        entropy = entropy / self.beta

        return entropy

    @partial(jit, static_argnums=(0,))
    def _capacity_impl(self, metric: jnp.ndarray) -> float:
        """
        Compute information capacity from metric.

        Capacity ~ sqrt(det(g)) (volume element)

        Args:
            metric: Metric tensor

        Returns:
            Information capacity
        """
        # Regularize metric
        g = metric + 1e-8 * jnp.eye(self.dim)

        # Capacity proportional to sqrt of determinant
        capacity = jnp.sqrt(jnp.abs(jnp.linalg.det(g)))

        return capacity

    def compute_geodesic(self, start: jnp.ndarray, end: jnp.ndarray,
                         metric_fn: Callable, num_steps: int = 100) -> Geodesic:
        """
        Compute geodesic between two points.

        Args:
            start: Starting point
            end: Ending point
            metric_fn: Metric function
            num_steps: Number of integration steps

        Returns:
            Geodesic path
        """
        # Initial velocity (straight line approximation)
        v0 = (end - start) / num_steps

        # Integrate geodesic equation
        def christoffel_wrapper(x):
            return self._christoffel_symbols(x, metric_fn)

        path = [start]
        y = jnp.concatenate([start, v0])

        dt = 1.0 / num_steps
        for _ in range(num_steps):
            dy = self._geodesic_equation(0, y, christoffel_wrapper)
            y = y + dt * dy
            path.append(y[:self.dim])

        path = jnp.array(path)

        # Compute path length
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1] - path[i]
            g = metric_fn(path[i])
            length += jnp.sqrt(jnp.dot(dx, g @ dx))

        # Compute energy
        energy = 0.5 * length ** 2

        # Parallel transport identity along path
        transport = jnp.eye(self.dim)  # Simplified

        return Geodesic(
            path=path,
            length=length,
            energy=energy,
            parallel_transport=transport
        )