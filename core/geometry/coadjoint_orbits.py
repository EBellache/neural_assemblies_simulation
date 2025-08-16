"""
Coadjoint Orbits and Symplectic Geometry
=========================================

Implements the symplectic geometry of coadjoint orbits for morphogenetic
state spaces. GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from functools import partial
from dataclasses import dataclass

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class MomentumCoordinates(NamedTuple):
    """8D momentum coordinates on coadjoint orbit."""
    px: float  # Linear momentum x
    py: float  # Linear momentum y
    Lz: float  # Angular momentum
    Q: float  # Bioelectric charge
    rho: float  # Density/mass
    Jx: float  # Shape moment x
    Jy: float  # Shape moment y
    H: float  # Hamiltonian/Energy

    def to_array(self) -> jnp.ndarray:
        """Convert to JAX array."""
        return jnp.array([self.px, self.py, self.Lz, self.Q,
                          self.rho, self.Jx, self.Jy, self.H])


class CoadjointOrbit(NamedTuple):
    """Coadjoint orbit as symplectic leaf."""
    casimirs: Dict[str, float]  # Casimir invariants labeling the orbit
    dimension: int  # Dimension of the orbit
    symplectic_form: jnp.ndarray  # Kirillov-Kostant-Souriau form
    momentum_map: jnp.ndarray  # Momentum map J: M -> g*


class SymplecticState(NamedTuple):
    """State on a coadjoint orbit."""
    momentum: jnp.ndarray  # 8D momentum coordinates
    orbit: CoadjointOrbit  # Which orbit it lies on
    tangent_vector: Optional[jnp.ndarray] = None  # Tangent vector for dynamics


class CoadjointOrbitComputer:
    """
    Compute coadjoint orbits and symplectic geometry.
    """

    def __init__(self, lie_algebra_dim: int = 8):
        """
        Initialize coadjoint orbit computer.

        Args:
            lie_algebra_dim: Dimension of Lie algebra (default 8)
        """
        self.dim = lie_algebra_dim

        # Precompute structure constants for common groups
        self.structure_constants = self._compute_structure_constants()

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        self._compute_orbit = jit(self._compute_orbit_impl)
        self._symplectic_form = jit(self._symplectic_form_impl)
        self._poisson_bracket = jit(self._poisson_bracket_impl)
        self._momentum_map = jit(self._momentum_map_impl)
        self._casimir_invariants = jit(self._casimir_invariants_impl)
        self._geodesic_flow = jit(self._geodesic_flow_impl)

    def _compute_structure_constants(self) -> jnp.ndarray:
        """
        Compute Lie algebra structure constants.

        Returns:
            Structure constants C^k_{ij}
        """
        # For the semidirect product structure relevant to morphogenesis
        # G = SE(2) × R^5 with appropriate brackets
        C = jnp.zeros((self.dim, self.dim, self.dim))

        # SE(2) part: [px, py, Lz]
        # [Lz, px] = py, [Lz, py] = -px
        C = C.at[2, 0, 1].set(1.0)  # [Lz, px] -> py
        C = C.at[2, 1, 0].set(-1.0)  # [Lz, py] -> -px

        # Bioelectric coupling: [Q, Ji] terms
        C = C.at[3, 5, 1].set(0.1)  # [Q, Jx] weakly couples to py
        C = C.at[3, 6, 0].set(0.1)  # [Q, Jy] weakly couples to px

        # Make antisymmetric
        C = C - jnp.transpose(C, (1, 0, 2))

        return C

    @partial(jit, static_argnums=(0,))
    def _compute_orbit_impl(self, momentum: jnp.ndarray) -> CoadjointOrbit:
        """
        Compute the coadjoint orbit through a point.

        Args:
            momentum: Point in dual Lie algebra g*

        Returns:
            CoadjointOrbit containing the point
        """
        # Compute Casimir invariants
        casimirs = self._casimir_invariants_impl(momentum)

        # Compute orbit dimension (codimension = # independent Casimirs)
        # For our 8D system with specific symmetries
        orbit_dim = 6  # Typical for SE(2) × R^5 quotient

        # Compute symplectic form
        omega = self._symplectic_form_impl(momentum)

        # Momentum map (identity for coadjoint orbits)
        J = jnp.eye(self.dim)

        return CoadjointOrbit(
            casimirs=casimirs,
            dimension=orbit_dim,
            symplectic_form=omega,
            momentum_map=J
        )

    @partial(jit, static_argnums=(0,))
    def _symplectic_form_impl(self, momentum: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Kirillov-Kostant-Souriau symplectic form.

        The KKS form at μ ∈ g* is:
        ω_μ(ad*_X μ, ad*_Y μ) = μ([X,Y])

        Args:
            momentum: Point in g*

        Returns:
            Symplectic form matrix
        """
        omega = jnp.zeros((self.dim, self.dim))

        # Use structure constants to build symplectic form
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    omega = omega.at[i, j].add(
                        momentum[k] * self.structure_constants[k, i, j]
                    )

        # Ensure antisymmetric
        omega = 0.5 * (omega - omega.T)

        return omega

    @partial(jit, static_argnums=(0,))
    def _poisson_bracket_impl(self, f: jnp.ndarray, g: jnp.ndarray,
                              momentum: jnp.ndarray) -> float:
        """
        Compute Poisson bracket {f,g} on coadjoint orbit.

        Args:
            f: Function on orbit (as gradient)
            g: Function on orbit (as gradient)
            momentum: Point on orbit

        Returns:
            Poisson bracket value
        """
        omega = self._symplectic_form_impl(momentum)

        # Invert symplectic form to get Poisson tensor
        # Add small regularization for numerical stability
        omega_reg = omega + 1e-8 * jnp.eye(self.dim)
        try:
            poisson = jnp.linalg.inv(omega_reg)
        except:
            poisson = jnp.linalg.pinv(omega_reg)

        # {f,g} = Π^{ij} ∂_i f ∂_j g
        return jnp.dot(f, poisson @ g)

    @partial(jit, static_argnums=(0,))
    def _momentum_map_impl(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute momentum map J: M -> g*.

        Args:
            state: State on manifold M

        Returns:
            Momentum coordinates in g*
        """
        # For coadjoint orbits, this is essentially the identity
        # but we include it for completeness
        return state[:self.dim] if state.shape[0] > self.dim else state

    @partial(jit, static_argnums=(0,))
    def _casimir_invariants_impl(self, momentum: jnp.ndarray) -> Dict[str, float]:
        """
        Compute Casimir invariants (conserved quantities).

        Args:
            momentum: Point in g*

        Returns:
            Dictionary of Casimir invariants
        """
        px, py, Lz, Q, rho, Jx, Jy, H = momentum

        casimirs = {
            'energy': H,
            'bioelectric_charge': Q,
            'angular_momentum': Lz,
            'mass': rho,
            'helicity': px * Jx + py * Jy,  # Momentum-shape coupling
            'enstrophy': px ** 2 + py ** 2 + Lz ** 2,  # Rotational invariant
            'chirality': Lz * Q / (rho + 1e-8),  # Normalized chirality
            'morphogenic_index': (Jx ** 2 + Jy ** 2) * Q ** 2  # Shape-charge coupling
        }

        return casimirs

    @partial(jit, static_argnums=(0,))
    def _geodesic_flow_impl(self, momentum: jnp.ndarray,
                            hamiltonian_grad: jnp.ndarray,
                            dt: float) -> jnp.ndarray:
        """
        Evolve along geodesic flow on coadjoint orbit.

        Args:
            momentum: Current momentum
            hamiltonian_grad: Gradient of Hamiltonian
            dt: Time step

        Returns:
            Updated momentum
        """
        omega = self._symplectic_form_impl(momentum)

        # Hamilton's equations: dμ/dt = {μ, H} = -ω^{-1} ∇H
        omega_reg = omega + 1e-8 * jnp.eye(self.dim)
        try:
            omega_inv = jnp.linalg.inv(omega_reg)
        except:
            omega_inv = jnp.linalg.pinv(omega_reg)

        velocity = -omega_inv @ hamiltonian_grad

        # Symplectic Euler step
        momentum_new = momentum + dt * velocity

        return momentum_new

    def project_to_orbit(self, point: jnp.ndarray,
                         target_casimirs: Dict[str, float]) -> jnp.ndarray:
        """
        Project point to nearest coadjoint orbit with given Casimirs.

        Args:
            point: Point in g*
            target_casimirs: Target Casimir values

        Returns:
            Projected point on target orbit
        """

        def loss(x):
            """Loss function for projection."""
            current_casimirs = self._casimir_invariants(x)

            error = 0.0
            for key, target_val in target_casimirs.items():
                if key in current_casimirs:
                    error += (current_casimirs[key] - target_val) ** 2

            # Also minimize distance to original point
            error += 0.1 * jnp.sum((x - point) ** 2)

            return error

        # Use gradient descent for projection
        x = point.copy()
        grad_fn = grad(loss)

        for _ in range(100):
            g = grad_fn(x)
            x = x - 0.01 * g

        return x

    def compute_stability_matrix(self, momentum: jnp.ndarray) -> jnp.ndarray:
        """
        Compute stability matrix for linearized dynamics.

        Args:
            momentum: Point on orbit

        Returns:
            Stability matrix (eigenvalues determine stability)
        """
        omega = self._symplectic_form(momentum)

        # For Hamiltonian systems, stability determined by
        # eigenvalues of J*H where J is symplectic structure
        # and H is Hessian of Hamiltonian

        # Approximate Hessian (would be computed from specific Hamiltonian)
        H_hess = jnp.diag(jnp.array([1.0, 1.0, 2.0, 0.5,
                                     1.0, 1.5, 1.5, 0.1]))

        # Stability matrix
        omega_reg = omega + 1e-8 * jnp.eye(self.dim)
        try:
            omega_inv = jnp.linalg.inv(omega_reg)
        except:
            omega_inv = jnp.linalg.pinv(omega_reg)

        stability = omega_inv @ H_hess

        return stability