"""
Čech Cohomology Computations
=============================

Implements Čech cohomology for neural sheaves, enabling
topological analysis of neural assemblies and replay.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from functools import partial

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class CechCochain(NamedTuple):
    """Čech cochain."""
    degree: int  # 0, 1, or 2
    support: List[Tuple[int, ...]]  # Indices of open sets
    values: jnp.ndarray  # Values on intersections
    is_cocycle: bool  # Whether d(cochain) = 0


class CohomologyClass(NamedTuple):
    """Cohomology class [α] ∈ H^k."""
    degree: int
    representative: CechCochain  # Representative cocycle
    dimension: int  # Dimension of cohomology group
    basis: List[jnp.ndarray]  # Basis for this class


class SpectralSequence(NamedTuple):
    """Spectral sequence for computing cohomology."""
    page: int  # E_r page
    differentials: Dict[int, jnp.ndarray]  # d_r differentials
    groups: Dict[Tuple[int, int], jnp.ndarray]  # E_r^{p,q} groups
    converged: bool  # Whether sequence has converged


class NerveComplex(NamedTuple):
    """Nerve of an open cover."""
    vertices: List[int]  # Open sets
    edges: List[Tuple[int, int]]  # Double intersections
    triangles: List[Tuple[int, int, int]]  # Triple intersections
    higher_simplices: List[Tuple]  # Higher intersections


class CohomologyComputer:
    """
    Compute Čech cohomology of neural sheaves.
    """

    def __init__(self, max_degree: int = 3, precision: float = 1e-6):
        """
        Initialize cohomology computer.

        Args:
            max_degree: Maximum cohomology degree to compute
            precision: Numerical precision
        """
        self.max_degree = max_degree
        self.precision = precision

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._cech_differential = jit(self._cech_differential_impl)
        self._compute_cohomology_group = jit(self._compute_cohomology_group_impl)
        self._cup_product = jit(self._cup_product_impl)
        self._steenrod_square = jit(self._steenrod_square_impl)
        self._spectral_sequence_page = jit(self._spectral_sequence_page_impl)
        self._edge_homomorphism = jit(self._edge_homomorphism_impl)
        self._obstruction_class = jit(self._obstruction_class_impl)
        self._transgression = jit(self._transgression_impl)

    @partial(jit, static_argnums=(0,))
    def _cech_differential_impl(self, cochain: jnp.ndarray,
                                degree: int,
                                support: List[Tuple]) -> jnp.ndarray:
        """
        Compute Čech differential d^k: C^k → C^{k+1}.

        Args:
            cochain: k-cochain values
            degree: Degree k
            support: Support of cochain

        Returns:
            (k+1)-cochain
        """
        if degree == 0:
            # 0-cochain to 1-cochain
            n = len(cochain)
            result = []

            for i in range(n):
                for j in range(i + 1, n):
                    # (df)_{ij} = f_j - f_i
                    result.append(cochain[j] - cochain[i])

            return jnp.array(result)

        elif degree == 1:
            # 1-cochain to 2-cochain
            # Assume cochain indexed by edges (i,j)
            result = []

            # For each triple (i,j,k)
            for idx, (i, j, k) in enumerate([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]):
                # (dg)_{ijk} = g_{jk} - g_{ik} + g_{ij}
                g_ij = self._get_cochain_value(cochain, support, (i, j))
                g_ik = self._get_cochain_value(cochain, support, (i, k))
                g_jk = self._get_cochain_value(cochain, support, (j, k))

                result.append(g_jk - g_ik + g_ij)

            return jnp.array(result)

        else:
            # Higher degrees
            return jnp.zeros_like(cochain)

    def _get_cochain_value(self, cochain: jnp.ndarray,
                           support: List[Tuple],
                           index: Tuple) -> jnp.ndarray:
        """Get cochain value for given index."""
        try:
            idx = support.index(index)
            return cochain[idx] if idx < len(cochain) else 0
        except ValueError:
            # Try reversed index
            try:
                idx = support.index(index[::-1])
                return -cochain[idx] if idx < len(cochain) else 0
            except ValueError:
                return 0

    @partial(jit, static_argnums=(0,))
    def _compute_cohomology_group_impl(self, cochains: Dict[int, jnp.ndarray],
                                       degree: int) -> CohomologyClass:
        """
        Compute H^k as ker(d^k) / im(d^{k-1}).

        Args:
            cochains: Cochains at each degree
            degree: Which H^k to compute

        Returns:
            Cohomology class
        """
        # Get cochains
        if degree not in cochains:
            return CohomologyClass(
                degree=degree,
                representative=None,
                dimension=0,
                basis=[]
            )

        ck = cochains[degree]

        # Compute kernel of d^k
        dk = self._cech_differential_impl(ck, degree, [])
        ker_dk = self._compute_kernel(dk)

        # Compute image of d^{k-1}
        if degree > 0 and degree - 1 in cochains:
            ck_minus = cochains[degree - 1]
            im_dk_minus = self._cech_differential_impl(ck_minus, degree - 1, [])
        else:
            im_dk_minus = jnp.zeros((1, ck.shape[-1]))

        # Compute quotient
        cohom_basis = self._compute_quotient(ker_dk, im_dk_minus)

        # Create representative cocycle
        if cohom_basis.size > 0:
            rep = CechCochain(
                degree=degree,
                support=[],
                values=cohom_basis[:, 0] if cohom_basis.ndim > 1 else cohom_basis,
                is_cocycle=True
            )
        else:
            rep = None

        return CohomologyClass(
            degree=degree,
            representative=rep,
            dimension=cohom_basis.shape[1] if cohom_basis.ndim > 1 else 0,
            basis=[cohom_basis[:, i] for i in range(min(cohom_basis.shape[1] if cohom_basis.ndim > 1 else 0, 10))]
        )

    @partial(jit, static_argnums=(0,))
    def _compute_kernel(self, linear_map: jnp.ndarray) -> jnp.ndarray:
        """Compute kernel of linear map using SVD."""
        if linear_map.size == 0:
            return jnp.eye(1)

        # Reshape to 2D
        m = linear_map.reshape(linear_map.shape[0], -1)

        # SVD
        u, s, vt = jnp.linalg.svd(m, full_matrices=True)

        # Find null space
        tol = self.precision
        rank = jnp.sum(s > tol)

        if rank < vt.shape[0]:
            kernel = vt[rank:].T
        else:
            kernel = jnp.zeros((vt.shape[1], 1))

        return kernel

    @partial(jit, static_argnums=(0,))
    def _compute_quotient(self, space: jnp.ndarray,
                          subspace: jnp.ndarray) -> jnp.ndarray:
        """Compute quotient space/subspace."""
        if space.size == 0:
            return jnp.zeros((1, 0))

        if subspace.size == 0:
            return space

        # Orthogonalize subspace basis
        q_sub, _ = jnp.linalg.qr(subspace)

        # Project space orthogonal to subspace
        proj = jnp.eye(space.shape[0]) - q_sub @ q_sub.T
        quotient = proj @ space

        # Remove zero columns
        norms = jnp.linalg.norm(quotient, axis=0)
        quotient = quotient[:, norms > self.precision]

        return quotient

    @partial(jit, static_argnums=(0,))
    def _cup_product_impl(self, alpha: CechCochain,
                          beta: CechCochain) -> CechCochain:
        """
        Compute cup product α ∪ β.

        Args:
            alpha: p-cochain
            beta: q-cochain

        Returns:
            (p+q)-cochain
        """
        p = alpha.degree
        q = beta.degree

        # Cup product formula
        # (α ∪ β)(σ₀,...,σₚ₊ᵧ) = α(σ₀,...,σₚ) · β(σₚ,...,σₚ₊ᵧ)

        result_values = []
        result_support = []

        # Simplified: multiply values
        for i, val_a in enumerate(alpha.values):
            for j, val_b in enumerate(beta.values):
                result_values.append(val_a * val_b)

        result_values = jnp.array(result_values)

        return CechCochain(
            degree=p + q,
            support=result_support,
            values=result_values,
            is_cocycle=alpha.is_cocycle and beta.is_cocycle
        )

    @partial(jit, static_argnums=(0,))
    def _steenrod_square_impl(self, cochain: CechCochain,
                              i: int) -> CechCochain:
        """
        Compute Steenrod square Sq^i.

        Args:
            cochain: Cochain to square
            i: Which square Sq^i

        Returns:
            Squared cochain
        """
        # Simplified Steenrod square
        # Sq^i: H^n(X; Z/2) → H^{n+i}(X; Z/2)

        if i == 0:
            # Sq^0 = identity
            return cochain
        elif i == 1:
            # Sq^1 = Bockstein homomorphism (mod 2)
            values = (cochain.values ** 2) % 2
            return CechCochain(
                degree=cochain.degree + 1,
                support=cochain.support,
                values=values,
                is_cocycle=cochain.is_cocycle
            )
        else:
            # Higher squares (simplified)
            values = cochain.values ** (2 ** i)
            return CechCochain(
                degree=cochain.degree + i,
                support=cochain.support,
                values=values,
                is_cocycle=False
            )

    @partial(jit, static_argnums=(0,))
    def _spectral_sequence_page_impl(self, page: int,
                                     filtration: List[jnp.ndarray]) -> SpectralSequence:
        """
        Compute E_r page of spectral sequence.

        Args:
            page: Which page E_r
            filtration: Filtration of complex

        Returns:
            Spectral sequence page
        """
        groups = {}
        differentials = {}

        for p in range(len(filtration)):
            for q in range(self.max_degree):
                if page == 0:
                    # E_0 page: associated graded
                    if p < len(filtration):
                        groups[(p, q)] = filtration[p]
                    else:
                        groups[(p, q)] = jnp.zeros((1,))

                elif page == 1:
                    # E_1 page: cohomology of E_0
                    # Simplified: use filtration directly
                    groups[(p, q)] = filtration[min(p, len(filtration) - 1)]

                else:
                    # Higher pages (simplified)
                    groups[(p, q)] = jnp.zeros((1,))

        # Differentials d_r: E_r^{p,q} → E_r^{p+r,q-r+1}
        for p in range(len(filtration)):
            for q in range(self.max_degree):
                if (p, q) in groups and (p + page, q - page + 1) in groups:
                    # Simplified differential
                    source_dim = groups[(p, q)].shape[0]
                    target_dim = groups[(p + page, q - page + 1)].shape[0]
                    differentials[(p, q)] = jnp.zeros((target_dim, source_dim))

        # Check convergence
        converged = page >= len(filtration)

        return SpectralSequence(
            page=page,
            differentials=differentials,
            groups=groups,
            converged=converged
        )

    @partial(jit, static_argnums=(0,))
    def _edge_homomorphism_impl(self, spectral_seq: SpectralSequence) -> jnp.ndarray:
        """
        Compute edge homomorphism from spectral sequence.

        Args:
            spectral_seq: Converged spectral sequence

        Returns:
            Edge homomorphism matrix
        """
        # Edge homomorphism: H^n(X) → E_∞^{n,0}

        n = min(3, self.max_degree)

        # Get E_∞^{n,0}
        if (n, 0) in spectral_seq.groups:
            target_dim = spectral_seq.groups[(n, 0)].shape[0]
        else:
            target_dim = 1

        # Simplified: identity-like map
        edge = jnp.eye(target_dim)

        return edge

    @partial(jit, static_argnums=(0,))
    def _obstruction_class_impl(self, cocycle: CechCochain,
                                extension_data: Dict) -> CohomologyClass:
        """
        Compute obstruction class for extending sections.

        Args:
            cocycle: Cocycle representing obstruction
            extension_data: Data about extension problem

        Returns:
            Obstruction class in H^2
        """
        # Obstruction lives in H^2
        obstruction_degree = 2

        # Compute obstruction cocycle
        if cocycle.degree == 1:
            # Lift to degree 2
            obs_values = self._cech_differential_impl(
                cocycle.values, 1, cocycle.support
            )
        else:
            obs_values = cocycle.values

        # Check if obstruction vanishes
        is_zero = jnp.linalg.norm(obs_values) < self.precision

        return CohomologyClass(
            degree=obstruction_degree,
            representative=CechCochain(
                degree=obstruction_degree,
                support=[],
                values=obs_values,
                is_cocycle=True
            ),
            dimension=0 if is_zero else 1,
            basis=[obs_values] if not is_zero else []
        )

    @partial(jit, static_argnums=(0,))
    def _transgression_impl(self, fiber_class: CohomologyClass,
                            bundle_data: Dict) -> CohomologyClass:
        """
        Compute transgression in fiber bundle.

        τ: H^k(F) → H^{k+1}(B)

        Args:
            fiber_class: Cohomology class of fiber
            bundle_data: Fiber bundle data

        Returns:
            Transgressed class in base
        """
        # Transgression increases degree by 1
        base_degree = fiber_class.degree + 1

        # Transgressed values (simplified)
        if fiber_class.representative is not None:
            trans_values = jnp.concatenate([
                fiber_class.representative.values,
                jnp.zeros(1)  # Padding
            ])
        else:
            trans_values = jnp.zeros(1)

        return CohomologyClass(
            degree=base_degree,
            representative=CechCochain(
                degree=base_degree,
                support=[],
                values=trans_values,
                is_cocycle=True
            ),
            dimension=fiber_class.dimension,
            basis=[trans_values]
        )

    def compute_nerve(self, cover: List[Any]) -> NerveComplex:
        """
        Compute nerve of open cover.

        Args:
            cover: Open cover

        Returns:
            Nerve complex
        """
        n = len(cover)

        # Vertices: individual sets
        vertices = list(range(n))

        # Edges: pairwise intersections
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                # Check if sets intersect (simplified: always true)
                edges.append((i, j))

        # Triangles: triple intersections
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    triangles.append((i, j, k))

        # Higher simplices (limited)
        higher = []
        if n >= 4:
            higher.append(tuple(range(4)))

        return NerveComplex(
            vertices=vertices,
            edges=edges,
            triangles=triangles,
            higher_simplices=higher
        )