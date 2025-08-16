"""
Casimir Invariants and Conservation Laws
=========================================

Implements Casimir invariants that remain conserved during morphogenetic
dynamics, providing error correction through conservation laws.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad, hessian
from jax.scipy import linalg as jax_linalg
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


class CasimirSet(NamedTuple):
    """Complete set of Casimir invariants."""
    energy: float  # Hamiltonian
    momentum: jnp.ndarray  # Linear momentum
    angular_momentum: float  # Angular momentum
    charge: float  # Bioelectric charge
    helicity: float  # Chirality measure
    enstrophy: float  # Vorticity squared
    mass: float  # Total mass/density
    morphogenic_index: float  # Combined shape-charge invariant


class ConservationViolation(NamedTuple):
    """Violation of conservation laws."""
    casimir_errors: Dict[str, float]  # Error per Casimir
    total_violation: float  # Total conservation violation
    gradient: jnp.ndarray  # Gradient toward conservation
    critical_casimirs: List[str]  # Most violated invariants


class CasimirManifold(NamedTuple):
    """Manifold defined by Casimir constraints."""
    dimension: int  # Dimension of constraint surface
    tangent_space: jnp.ndarray  # Tangent space basis
    normal_space: jnp.ndarray  # Normal space basis
    metric: jnp.ndarray  # Induced metric


class SymplecticCorrector(NamedTuple):
    """Symplectic correction to restore Casimirs."""
    correction_vector: jnp.ndarray  # Correction to apply
    projection_matrix: jnp.ndarray  # Projection onto Casimir manifold
    restored_state: jnp.ndarray  # State after correction
    convergence_rate: float  # Rate of convergence


class CasimirComputer:
    """
    Compute and enforce Casimir invariants.
    """

    def __init__(self, dimension: int = 8, tolerance: float = 1e-6):
        """
        Initialize Casimir computer.

        Args:
            dimension: State space dimension
            tolerance: Conservation tolerance
        """
        self.dim = dimension
        self.tol = tolerance

        # Define Casimir functions
        self._init_casimir_functions()

        # Pre-compile JAX functions
        self._compile_functions()

    def _init_casimir_functions(self):
        """Initialize Casimir function definitions."""
        self.casimir_definitions = {
            'energy': lambda m: m[7],  # H component
            'linear_momentum': lambda m: jnp.array([m[0], m[1]]),  # px, py
            'angular_momentum': lambda m: m[2],  # Lz
            'charge': lambda m: m[3],  # Q
            'mass': lambda m: m[4],  # ρ
            'shape_x': lambda m: m[5],  # Jx
            'shape_y': lambda m: m[6],  # Jy
            'helicity': lambda m: m[0] * m[5] + m[1] * m[6],  # px*Jx + py*Jy
            'enstrophy': lambda m: m[0] ** 2 + m[1] ** 2 + m[2] ** 2,  # |p|² + L²
            'morphogenic_index': lambda m: (m[5] ** 2 + m[6] ** 2) * m[3] ** 2  # |J|² Q²
        }

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._compute_casimirs = jit(self._compute_casimirs_impl)
        self._check_conservation = jit(self._check_conservation_impl)
        self._project_to_manifold = jit(self._project_to_manifold_impl)
        self._symplectic_correction = jit(self._symplectic_correction_impl)
        self._casimir_bracket = jit(self._casimir_bracket_impl)
        self._find_casimir_manifold = jit(self._find_casimir_manifold_impl)
        self._restoration_flow = jit(self._restoration_flow_impl)
        self._multi_casimir_intersection = jit(self._multi_casimir_intersection_impl)

    @partial(jit, static_argnums=(0,))
    def _compute_casimirs_impl(self, state: jnp.ndarray) -> CasimirSet:
        """
        Compute all Casimir invariants.

        Args:
            state: System state (8D momentum coordinates)

        Returns:
            Complete set of Casimirs
        """
        px, py, Lz, Q, rho, Jx, Jy, H = state

        return CasimirSet(
            energy=H,
            momentum=jnp.array([px, py]),
            angular_momentum=Lz,
            charge=Q,
            helicity=px * Jx + py * Jy,
            enstrophy=px ** 2 + py ** 2 + Lz ** 2,
            mass=rho,
            morphogenic_index=(Jx ** 2 + Jy ** 2) * Q ** 2
        )

    @partial(jit, static_argnums=(0,))
    def _check_conservation_impl(self, state1: jnp.ndarray,
                                 state2: jnp.ndarray) -> ConservationViolation:
        """
        Check conservation between two states.

        Args:
            state1: Initial state
            state2: Final state

        Returns:
            Conservation violation analysis
        """
        casimirs1 = self._compute_casimirs_impl(state1)
        casimirs2 = self._compute_casimirs_impl(state2)

        # Compute violations
        errors = {
            'energy': abs(casimirs2.energy - casimirs1.energy),
            'momentum_x': abs(casimirs2.momentum[0] - casimirs1.momentum[0]),
            'momentum_y': abs(casimirs2.momentum[1] - casimirs1.momentum[1]),
            'angular_momentum': abs(casimirs2.angular_momentum - casimirs1.angular_momentum),
            'charge': abs(casimirs2.charge - casimirs1.charge),
            'helicity': abs(casimirs2.helicity - casimirs1.helicity),
            'enstrophy': abs(casimirs2.enstrophy - casimirs1.enstrophy),
            'mass': abs(casimirs2.mass - casimirs1.mass),
            'morphogenic_index': abs(casimirs2.morphogenic_index - casimirs1.morphogenic_index)
        }

        total_violation = sum(errors.values())

        # Gradient toward conservation (points from state2 back toward conserved values)
        gradient = state1 - state2
        gradient = gradient / (jnp.linalg.norm(gradient) + 1e-8)

        # Find most violated
        critical = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:3]
        critical_casimirs = [c[0] for c in critical]

        return ConservationViolation(
            casimir_errors=errors,
            total_violation=float(total_violation),
            gradient=gradient,
            critical_casimirs=critical_casimirs
        )

    @partial(jit, static_argnums=(0,))
    def _casimir_bracket_impl(self, f: Callable, g: Callable,
                              state: jnp.ndarray) -> float:
        """
        Compute Poisson bracket {f, g} of Casimir functions.

        Should be zero for true Casimirs.

        Args:
            f, g: Functions on phase space
            state: Current state

        Returns:
            Poisson bracket value
        """
        # Compute gradients
        grad_f = grad(f)(state)
        grad_g = grad(g)(state)

        # Simplified Poisson bracket (canonical coordinates)
        # {f,g} = ∂f/∂q · ∂g/∂p - ∂f/∂p · ∂g/∂q

        mid = self.dim // 2
        bracket = (jnp.dot(grad_f[:mid], grad_g[mid:]) -
                   jnp.dot(grad_f[mid:], grad_g[:mid]))

        return bracket

    @partial(jit, static_argnums=(0,))
    def _find_casimir_manifold_impl(self, target_casimirs: CasimirSet,
                                    reference_point: jnp.ndarray) -> CasimirManifold:
        """
        Find manifold defined by Casimir constraints.

        Args:
            target_casimirs: Target Casimir values
            reference_point: Reference point on manifold

        Returns:
            Casimir manifold structure
        """
        # Compute constraint gradients
        constraints = []

        # Energy constraint
        constraints.append(grad(lambda s: s[7] - target_casimirs.energy)(reference_point))

        # Momentum constraints
        constraints.append(grad(lambda s: s[0] - target_casimirs.momentum[0])(reference_point))
        constraints.append(grad(lambda s: s[1] - target_casimirs.momentum[1])(reference_point))

        # Angular momentum
        constraints.append(grad(lambda s: s[2] - target_casimirs.angular_momentum)(reference_point))

        # Charge
        constraints.append(grad(lambda s: s[3] - target_casimirs.charge)(reference_point))

        # Stack constraint gradients
        constraint_matrix = jnp.array(constraints)

        # Find tangent space (null space of constraints)
        u, s, vt = jnp.linalg.svd(constraint_matrix)

        # Tangent space: columns of V corresponding to zero singular values
        rank = jnp.sum(s > 1e-10)
        tangent_basis = vt[rank:].T
        normal_basis = vt[:rank].T

        # Induced metric on manifold
        # g_induced = tangent^T @ g_ambient @ tangent
        g_ambient = jnp.eye(self.dim)  # Euclidean for simplicity
        metric = tangent_basis.T @ g_ambient @ tangent_basis

        return CasimirManifold(
            dimension=self.dim - rank,
            tangent_space=tangent_basis,
            normal_space=normal_basis,
            metric=metric
        )

    @partial(jit, static_argnums=(0,))
    def _project_to_manifold_impl(self, state: jnp.ndarray,
                                  target_casimirs: CasimirSet) -> jnp.ndarray:
        """
        Project state onto Casimir manifold.

        Args:
            state: Current state
            target_casimirs: Target Casimir values

        Returns:
            Projected state
        """

        def loss(s):
            """Loss function for projection."""
            casimirs = self._compute_casimirs_impl(s)

            error = 0.0
            error += (casimirs.energy - target_casimirs.energy) ** 2
            error += jnp.sum((casimirs.momentum - target_casimirs.momentum) ** 2)
            error += (casimirs.angular_momentum - target_casimirs.angular_momentum) ** 2
            error += (casimirs.charge - target_casimirs.charge) ** 2
            error += (casimirs.mass - target_casimirs.mass) ** 2

            # Also minimize distance to original state
            error += 0.1 * jnp.sum((s - state) ** 2)

            return error

        # Gradient descent projection
        projected = state.copy()
        grad_fn = grad(loss)

        for _ in range(100):
            g = grad_fn(projected)
            projected = projected - 0.01 * g

            if jnp.linalg.norm(g) < self.tol:
                break

        return projected

    @partial(jit, static_argnums=(0,))
    def _symplectic_correction_impl(self, state: jnp.ndarray,
                                    target_casimirs: CasimirSet,
                                    symplectic_form: jnp.ndarray) -> SymplecticCorrector:
        """
        Compute symplectic correction to restore Casimirs.

        Args:
            state: Current state
            target_casimirs: Target Casimirs
            symplectic_form: Symplectic 2-form

        Returns:
            Symplectic corrector
        """
        # Project onto Casimir manifold
        projected = self._project_to_manifold_impl(state, target_casimirs)

        # Correction vector
        correction = projected - state

        # Ensure correction is symplectic
        # Use symplectic Gram-Schmidt
        omega = symplectic_form

        # Project correction to be symplectic
        # Find closest symplectic vector
        def symplectic_loss(v):
            # Should satisfy ω(v, ·) = dH for some H
            return jnp.sum((omega @ v) ** 2)

        # Minimize to find symplectic correction
        symplectic_corr = correction.copy()
        for _ in range(10):
            g = grad(symplectic_loss)(symplectic_corr)
            symplectic_corr = symplectic_corr - 0.1 * g

        # Projection matrix onto Casimir manifold
        manifold = self._find_casimir_manifold_impl(target_casimirs, projected)
        projection = manifold.tangent_space @ manifold.tangent_space.T

        # Convergence rate (eigenvalue of linearized flow)
        linearization = jnp.eye(self.dim) - 0.1 * projection
        eigenvals = jnp.linalg.eigvals(linearization)
        convergence_rate = jnp.max(jnp.abs(eigenvals))

        return SymplecticCorrector(
            correction_vector=symplectic_corr,
            projection_matrix=projection,
            restored_state=projected,
            convergence_rate=float(convergence_rate)
        )

    @partial(jit, static_argnums=(0,))
    def _restoration_flow_impl(self, state: jnp.ndarray,
                               target_casimirs: CasimirSet,
                               time_steps: int = 100) -> jnp.ndarray:
        """
        Flow that restores Casimir invariants.

        Args:
            state: Initial state
            target_casimirs: Target Casimirs
            time_steps: Number of flow steps

        Returns:
            Final state after restoration flow
        """
        current = state.copy()
        dt = 0.01

        for _ in range(time_steps):
            # Compute Casimir errors
            current_casimirs = self._compute_casimirs_impl(current)

            # Gradient toward target Casimirs
            grad_energy = grad(lambda s: (s[7] - target_casimirs.energy) ** 2)(current)
            grad_momentum = grad(lambda s: jnp.sum((s[:2] - target_casimirs.momentum) ** 2))(current)
            grad_angular = grad(lambda s: (s[2] - target_casimirs.angular_momentum) ** 2)(current)
            grad_charge = grad(lambda s: (s[3] - target_casimirs.charge) ** 2)(current)

            # Combined gradient
            total_grad = grad_energy + grad_momentum + grad_angular + grad_charge

            # Flow step
            current = current - dt * total_grad

            # Check convergence
            if jnp.linalg.norm(total_grad) < self.tol:
                break

        return current

    @partial(jit, static_argnums=(0,))
    def _multi_casimir_intersection_impl(self, casimir_sets: List[CasimirSet]) -> jnp.ndarray:
        """
        Find intersection of multiple Casimir manifolds.

        Args:
            casimir_sets: List of Casimir constraints

        Returns:
            Point in intersection (if exists)
        """
        # Average Casimirs as initial guess
        avg_energy = jnp.mean(jnp.array([c.energy for c in casimir_sets]))
        avg_momentum = jnp.mean(jnp.array([c.momentum for c in casimir_sets]), axis=0)
        avg_angular = jnp.mean(jnp.array([c.angular_momentum for c in casimir_sets]))
        avg_charge = jnp.mean(jnp.array([c.charge for c in casimir_sets]))
        avg_mass = jnp.mean(jnp.array([c.mass for c in casimir_sets]))

        # Initial state
        state = jnp.array([
            avg_momentum[0], avg_momentum[1], avg_angular, avg_charge,
            avg_mass, 0.0, 0.0, avg_energy
        ])

        # Find intersection via optimization
        def intersection_loss(s):
            """Distance to all Casimir manifolds."""
            loss = 0.0
            for target in casimir_sets:
                casimirs = self._compute_casimirs_impl(s)
                loss += (casimirs.energy - target.energy) ** 2
                loss += jnp.sum((casimirs.momentum - target.momentum) ** 2)
                loss += (casimirs.angular_momentum - target.angular_momentum) ** 2
                loss += (casimirs.charge - target.charge) ** 2
            return loss

        # Optimize
        grad_fn = grad(intersection_loss)
        for _ in range(200):
            g = grad_fn(state)
            state = state - 0.01 * g

            if jnp.linalg.norm(g) < self.tol:
                break

        return state

    def enforce_conservation(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """
        Enforce conservation laws along trajectory.

        Args:
            trajectory: Time series of states

        Returns:
            Corrected trajectory
        """
        if len(trajectory) < 2:
            return trajectory

        # Get initial Casimirs
        initial_casimirs = self._compute_casimirs(trajectory[0])

        # Correct each point
        corrected = [trajectory[0]]

        for i in range(1, len(trajectory)):
            # Project to maintain initial Casimirs
            corrected_state = self._project_to_manifold(
                trajectory[i], initial_casimirs
            )
            corrected.append(corrected_state)

        return jnp.array(corrected)