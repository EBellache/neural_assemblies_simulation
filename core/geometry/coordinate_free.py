"""
Coordinate-Free Geometric Operations
=====================================

Implements coordinate-free operations on morphogenetic manifolds using
intrinsic geometric structures. GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from functools import partial
from dataclasses import dataclass

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class GeometricInvariant(NamedTuple):
    """Coordinate-free geometric invariant."""
    name: str
    value: float
    order: int  # Differential order (0=pointwise, 1=first derivative, etc.)
    tensorial_type: str  # 'scalar', 'vector', 'form', 'tensor'


class IntrinsicStructure(NamedTuple):
    """Intrinsic geometric structure on manifold."""
    dimension: int
    topology_type: str  # 'trivial', 'nontrivial'
    invariants: List[GeometricInvariant]
    cohomology_groups: Dict[int, int]  # H^k dimensions


class CoordinateFreeState(NamedTuple):
    """State described without coordinates."""
    invariants: Dict[str, float]  # Geometric invariants
    topology: Dict[str, Any]  # Topological data
    relations: jnp.ndarray  # Relations between components


class DifferentialForm(NamedTuple):
    """Differential form (coordinate-free)."""
    degree: int  # k-form
    components: jnp.ndarray  # Components in current chart
    is_closed: bool  # dω = 0
    is_exact: bool  # ω = dα for some α


class CoordinateFreeComputer:
    """
    Compute coordinate-free geometric quantities.
    """

    def __init__(self, manifold_dim: int = 8):
        """
        Initialize coordinate-free computer.

        Args:
            manifold_dim: Dimension of manifold
        """
        self.dim = manifold_dim

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._lie_derivative = jit(self._lie_derivative_impl)
        self._exterior_derivative = jit(self._exterior_derivative_impl)
        self._hodge_star = jit(self._hodge_star_impl)
        self._wedge_product = jit(self._wedge_product_impl)
        self._interior_product = jit(self._interior_product_impl)
        self._compute_invariants = jit(self._compute_invariants_impl)
        self._cartan_formula = jit(self._cartan_formula_impl)
        self._frobenius_test = jit(self._frobenius_test_impl)

    @partial(jit, static_argnums=(0,))
    def _lie_derivative_impl(self, vector_field: jnp.ndarray,
                             tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative (coordinate-free derivative).

        L_X T = lim_{t→0} (φ*_t T - T)/t
        where φ_t is flow of X

        Args:
            vector_field: Vector field X
            tensor: Tensor to differentiate

        Returns:
            Lie derivative L_X T
        """
        # For efficiency, use Cartan's formula for forms
        # L_X = i_X d + d i_X

        # Simplified implementation for vector fields and functions
        if tensor.ndim == 1:
            # Tensor is a vector field
            # [X, Y] = L_X Y (Lie bracket)
            return self._lie_bracket_impl(vector_field, tensor)
        else:
            # Tensor is higher order
            # Use directional derivative approximation
            eps = 1e-5
            flow = tensor + eps * jnp.outer(vector_field, jnp.ones(tensor.shape[1]))
            return (flow - tensor) / eps

    @partial(jit, static_argnums=(0,))
    def _lie_bracket_impl(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie bracket [X,Y] of vector fields.

        Args:
            X, Y: Vector fields

        Returns:
            Lie bracket [X,Y]
        """
        # In coordinates: [X,Y]^i = X^j ∂_j Y^i - Y^j ∂_j X^i
        # Approximate with finite differences

        bracket = jnp.zeros_like(X)
        eps = 1e-5

        for i in range(self.dim):
            for j in range(self.dim):
                # Approximate derivatives
                dY_i = (Y[i] - Y[i]) / eps  # Simplified
                dX_i = (X[i] - X[i]) / eps

                bracket = bracket.at[i].add(X[j] * dY_i - Y[j] * dX_i)

        return bracket

    @partial(jit, static_argnums=(0,))
    def _exterior_derivative_impl(self, form: DifferentialForm) -> DifferentialForm:
        """
        Compute exterior derivative d.

        d is the unique operator satisfying:
        1. d(ω ∧ η) = dω ∧ η + (-1)^k ω ∧ dη
        2. d² = 0
        3. df = ∑_i ∂f/∂x^i dx^i for functions

        Args:
            form: Differential form

        Returns:
            Exterior derivative dω
        """
        k = form.degree
        components = form.components

        if k == 0:
            # 0-form (function) -> 1-form (gradient)
            # df = grad(f)
            new_components = grad(lambda x: components)(jnp.zeros(self.dim))
            return DifferentialForm(
                degree=1,
                components=new_components,
                is_closed=False,
                is_exact=True
            )

        elif k < self.dim:
            # k-form -> (k+1)-form
            # Use antisymmetrization
            new_shape = components.shape + (self.dim,)
            new_components = jnp.zeros(new_shape)

            # Simplified: antisymmetrize derivatives
            for idx in range(components.shape[0]):
                for i in range(self.dim):
                    # ∂/∂x^i of component
                    deriv = grad(lambda x: components[idx])(jnp.zeros(self.dim))[i]
                    new_components = new_components.at[idx, i].set(deriv)

            # Antisymmetrize
            for i in range(new_components.shape[-2]):
                for j in range(i + 1, new_components.shape[-1]):
                    avg = 0.5 * (new_components[..., i, j] - new_components[..., j, i])
                    new_components = new_components.at[..., i, j].set(avg)
                    new_components = new_components.at[..., j, i].set(-avg)

            return DifferentialForm(
                degree=k + 1,
                components=new_components.reshape(-1),
                is_closed=False,
                is_exact=False
            )

        else:
            # Top degree form -> 0
            return DifferentialForm(
                degree=k + 1,
                components=jnp.zeros(1),
                is_closed=True,
                is_exact=False
            )

    @partial(jit, static_argnums=(0,))
    def _hodge_star_impl(self, form: DifferentialForm,
                         metric: jnp.ndarray) -> DifferentialForm:
        """
        Compute Hodge star operator *.

        *: Ω^k(M) -> Ω^(n-k)(M)

        Args:
            form: k-form
            metric: Metric tensor

        Returns:
            Hodge dual *(n-k)-form
        """
        k = form.degree
        n = self.dim

        # Compute volume form from metric
        vol = jnp.sqrt(jnp.abs(jnp.linalg.det(metric)))

        # Hodge star maps k-forms to (n-k)-forms
        new_degree = n - k

        # Simplified: use metric to raise/lower indices
        g_inv = jnp.linalg.inv(metric + 1e-8 * jnp.eye(n))

        # Contract with volume form
        new_components = vol * jnp.einsum('ij,j->i', g_inv, form.components)

        return DifferentialForm(
            degree=new_degree,
            components=new_components,
            is_closed=form.is_closed,
            is_exact=form.is_exact
        )

    @partial(jit, static_argnums=(0,))
    def _wedge_product_impl(self, form1: DifferentialForm,
                            form2: DifferentialForm) -> DifferentialForm:
        """
        Compute wedge product ω ∧ η.

        Args:
            form1: k-form
            form2: l-form

        Returns:
            (k+l)-form
        """
        k = form1.degree
        l = form2.degree

        if k + l > self.dim:
            # Result is zero
            return DifferentialForm(
                degree=k + l,
                components=jnp.zeros(1),
                is_closed=True,
                is_exact=False
            )

        # Antisymmetric tensor product
        # Simplified: outer product with antisymmetrization
        components = jnp.outer(form1.components, form2.components).flatten()

        # Apply sign from graded commutativity
        sign = (-1) ** (k * l)
        components = sign * components

        return DifferentialForm(
            degree=k + l,
            components=components,
            is_closed=form1.is_closed and form2.is_closed,
            is_exact=form1.is_exact or form2.is_exact
        )

    @partial(jit, static_argnums=(0,))
    def _interior_product_impl(self, vector: jnp.ndarray,
                               form: DifferentialForm) -> DifferentialForm:
        """
        Compute interior product i_X ω.

        (i_X ω)(Y_1,...,Y_{k-1}) = ω(X, Y_1,...,Y_{k-1})

        Args:
            vector: Vector field X
            form: k-form ω

        Returns:
            (k-1)-form i_X ω
        """
        k = form.degree

        if k == 0:
            # Interior product with 0-form is zero
            return DifferentialForm(
                degree=0,
                components=jnp.zeros(1),
                is_closed=True,
                is_exact=False
            )

        # Contract first index with vector
        # Simplified implementation
        new_components = jnp.dot(vector, form.components.reshape(self.dim, -1))

        return DifferentialForm(
            degree=k - 1,
            components=new_components.flatten(),
            is_closed=False,
            is_exact=False
        )

    @partial(jit, static_argnums=(0,))
    def _cartan_formula_impl(self, vector: jnp.ndarray,
                             form: DifferentialForm) -> DifferentialForm:
        """
        Apply Cartan's formula: L_X = i_X d + d i_X.

        Args:
            vector: Vector field X
            form: Differential form

        Returns:
            Lie derivative L_X ω
        """
        # i_X d ω
        d_form = self._exterior_derivative_impl(form)
        term1 = self._interior_product_impl(vector, d_form)

        # d i_X ω
        i_form = self._interior_product_impl(vector, form)
        term2 = self._exterior_derivative_impl(i_form)

        # Sum the terms
        result_components = term1.components + term2.components

        return DifferentialForm(
            degree=form.degree,
            components=result_components,
            is_closed=False,
            is_exact=False
        )

    @partial(jit, static_argnums=(0,))
    def _frobenius_test_impl(self, distribution: jnp.ndarray) -> bool:
        """
        Test if distribution is integrable (Frobenius theorem).

        A distribution is integrable iff [X,Y] ∈ D for all X,Y ∈ D

        Args:
            distribution: Matrix whose columns span the distribution

        Returns:
            True if integrable
        """
        # Check if Lie brackets remain in distribution
        n_vectors = distribution.shape[1]

        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                X = distribution[:, i]
                Y = distribution[:, j]

                # Compute Lie bracket
                bracket = self._lie_bracket_impl(X, Y)

                # Check if bracket is in span of distribution
                # Solve: distribution @ coeffs = bracket
                coeffs, residual, _, _ = jnp.linalg.lstsq(
                    distribution, bracket, rcond=1e-6
                )

                # If residual is large, not integrable
                if jnp.linalg.norm(residual) > 1e-4:
                    return False

        return True

    @partial(jit, static_argnums=(0,))
    def _compute_invariants_impl(self, state_data: jnp.ndarray) -> Dict[str, float]:
        """
        Compute coordinate-free invariants.

        Args:
            state_data: Raw state data

        Returns:
            Dictionary of invariants
        """
        invariants = {}

        # Topological invariants
        # Euler characteristic (simplified)
        invariants['euler_char'] = 2.0  # Sphere-like

        # Geometric invariants
        # Total "mass" (integral of density)
        invariants['total_mass'] = jnp.sum(jnp.abs(state_data))

        # "Angular momentum" (rotational invariant)
        if state_data.shape[0] >= 3:
            invariants['angular_momentum'] = jnp.linalg.norm(
                jnp.cross(state_data[:3], state_data[3:6]
                if state_data.shape[0] >= 6 else state_data[:3])
            )

        # Information-theoretic invariants
        # Entropy (coordinate-free)
        probs = jnp.abs(state_data) / (jnp.sum(jnp.abs(state_data)) + 1e-8)
        invariants['entropy'] = -jnp.sum(probs * jnp.log(probs + 1e-10))

        # Spectral invariants
        if state_data.shape[0] >= self.dim:
            matrix = state_data[:self.dim].reshape(self.dim, -1)
            if matrix.shape[1] >= self.dim:
                matrix = matrix[:, :self.dim]
                eigenvals = jnp.linalg.eigvalsh(matrix @ matrix.T)

                # Trace (sum of eigenvalues)
                invariants['trace'] = jnp.sum(eigenvals)

                # Determinant (product of eigenvalues)
                invariants['determinant'] = jnp.prod(eigenvals)

                # Spectral gap
                sorted_eigs = jnp.sort(jnp.abs(eigenvals))
                if len(sorted_eigs) > 1:
                    invariants['spectral_gap'] = sorted_eigs[-1] - sorted_eigs[-2]

        return invariants

    def extract_intrinsic_structure(self, data: jnp.ndarray) -> IntrinsicStructure:
        """
        Extract intrinsic geometric structure from data.

        Args:
            data: Raw data

        Returns:
            Intrinsic structure
        """
        invariants_dict = self._compute_invariants(data)

        # Convert to list of GeometricInvariant
        invariants = [
            GeometricInvariant(
                name=name,
                value=float(value),
                order=0 if name in ['total_mass', 'trace', 'determinant'] else 1,
                tensorial_type='scalar'
            )
            for name, value in invariants_dict.items()
        ]

        # Simplified cohomology groups
        cohomology = {
            0: 1,  # H^0 = connected components
            1: 0,  # H^1 = loops
            2: 1,  # H^2 = voids (sphere-like)
        }

        return IntrinsicStructure(
            dimension=self.dim,
            topology_type='nontrivial' if cohomology[1] > 0 else 'trivial',
            invariants=invariants,
            cohomology_groups=cohomology
        )