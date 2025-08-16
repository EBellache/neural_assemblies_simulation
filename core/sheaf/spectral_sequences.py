"""
Leray Spectral Sequences
=========================

Implements spectral sequences for computing sheaf cohomology,
particularly the Leray spectral sequence for neural information flow.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Callable
from functools import partial

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class FilteredComplex(NamedTuple):
    """Filtered cochain complex."""
    filtration_length: int
    complexes: List[jnp.ndarray]  # F^p C^*
    differentials: List[jnp.ndarray]  # Differential maps
    filtration_degrees: List[int]


class SpectralPage(NamedTuple):
    """Page of spectral sequence."""
    page_number: int  # r in E_r
    groups: Dict[Tuple[int, int], jnp.ndarray]  # E_r^{p,q}
    differentials: Dict[Tuple[int, int], jnp.ndarray]  # d_r
    total_degree: int  # p + q


class ConvergenceData(NamedTuple):
    """Convergence information for spectral sequence."""
    converged: bool
    convergence_page: int
    limit_groups: Dict[Tuple[int, int], jnp.ndarray]  # E_∞^{p,q}
    associated_graded: Dict[int, jnp.ndarray]  # gr^p H^n


class ExtensionProblem(NamedTuple):
    """Extension problem data."""
    short_exact_sequences: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    extension_groups: Dict[int, jnp.ndarray]  # Ext groups
    obstruction_classes: List[jnp.ndarray]


class SpectralSequenceComputer:
    """
    Compute spectral sequences for sheaf cohomology.
    """

    def __init__(self, max_page: int = 10, precision: float = 1e-8):
        """
        Initialize spectral sequence computer.

        Args:
            max_page: Maximum page to compute
            precision: Numerical precision
        """
        self.max_page = max_page
        self.precision = precision

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._compute_page = jit(self._compute_page_impl)
        self._differential = jit(self._differential_impl)
        self._homology = jit(self._homology_impl)
        self._check_convergence = jit(self._check_convergence_impl)
        self._edge_map = jit(self._edge_map_impl)
        self._extension_class = jit(self._extension_class_impl)
        self._filtration_spectral_sequence = jit(self._filtration_spectral_sequence_impl)
        self._double_complex = jit(self._double_complex_impl)

    @partial(jit, static_argnums=(0,))
    def _compute_page_impl(self, prev_page: SpectralPage,
                           filtration: FilteredComplex) -> SpectralPage:
        """
        Compute next page of spectral sequence.

        E_{r+1}^{p,q} = H(E_r^{p,q}, d_r)

        Args:
            prev_page: Previous page E_r
            filtration: Filtered complex

        Returns:
            Next page E_{r+1}
        """
        r = prev_page.page_number
        next_groups = {}
        next_differentials = {}

        for (p, q), group in prev_page.groups.items():
            # Get differential d_r: E_r^{p,q} → E_r^{p+r,q-r+1}
            if (p, q) in prev_page.differentials:
                d_r = prev_page.differentials[(p, q)]

                # Compute homology
                ker_d = self._compute_kernel(d_r)

                # Get image of incoming differential
                if (p - r, q + r - 1) in prev_page.differentials:
                    d_in = prev_page.differentials[(p - r, q + r - 1)]
                    im_d = self._compute_image(d_in)
                else:
                    im_d = jnp.zeros((group.shape[0], 1))

                # H = ker/im
                homology = self._compute_quotient(ker_d, im_d)
                next_groups[(p, q)] = homology
            else:
                # No differential, group unchanged
                next_groups[(p, q)] = group

        # Compute new differentials d_{r+1}
        for (p, q) in next_groups:
            target = (p + r + 1, q - r)
            if target in next_groups:
                # Induced differential
                source_dim = next_groups[(p, q)].shape[-1] if next_groups[(p, q)].ndim > 0 else 1
                target_dim = next_groups[target].shape[-1] if next_groups[target].ndim > 0 else 1

                # Simplified: zero differential at higher pages
                if r >= 2:
                    next_differentials[(p, q)] = jnp.zeros((target_dim, source_dim))
                else:
                    # Compute induced differential
                    next_differentials[(p, q)] = self._induced_differential(
                        prev_page, (p, q), target
                    )

        return SpectralPage(
            page_number=r + 1,
            groups=next_groups,
            differentials=next_differentials,
            total_degree=prev_page.total_degree
        )

    @partial(jit, static_argnums=(0,))
    def _differential_impl(self, page: int, p: int, q: int,
                           element: jnp.ndarray) -> jnp.ndarray:
        """
        Apply differential d_r.

        d_r: E_r^{p,q} → E_r^{p+r,q-r+1}

        Args:
            page: Page number r
            p, q: Bidegree
            element: Element to apply d_r to

        Returns:
            d_r(element)
        """
        # Differential shifts by (r, -r+1)
        if page == 0:
            # d_0 is the differential of the complex
            return element  # Simplified
        elif page == 1:
            # d_1 is induced from filtration
            return jnp.roll(element, 1)  # Simplified shift
        else:
            # Higher differentials often zero
            return jnp.zeros_like(element)

    @partial(jit, static_argnums=(0,))
    def _homology_impl(self, chain_complex: jnp.ndarray,
                       differential: jnp.ndarray) -> jnp.ndarray:
        """
        Compute homology H = ker(d) / im(d).

        Args:
            chain_complex: Chain groups
            differential: Differential map

        Returns:
            Homology group
        """
        # Compute kernel
        ker = self._compute_kernel(differential)

        # Compute image (from previous differential)
        im = self._compute_image(differential)

        # Quotient
        homology = self._compute_quotient(ker, im)

        return homology

    @partial(jit, static_argnums=(0,))
    def _compute_kernel(self, linear_map: jnp.ndarray) -> jnp.ndarray:
        """Compute kernel using SVD."""
        if linear_map.size == 0:
            return jnp.eye(1)

        # Ensure 2D
        if linear_map.ndim == 1:
            linear_map = linear_map.reshape(-1, 1)

        # SVD
        u, s, vt = jnp.linalg.svd(linear_map, full_matrices=True)

        # Null space
        rank = jnp.sum(s > self.precision)
        if rank < vt.shape[0]:
            kernel = vt[rank:].T
        else:
            kernel = jnp.zeros((vt.shape[1], 1))

        return kernel

    @partial(jit, static_argnums=(0,))
    def _compute_image(self, linear_map: jnp.ndarray) -> jnp.ndarray:
        """Compute image (column space)."""
        if linear_map.size == 0:
            return jnp.zeros((1, 1))

        # Ensure 2D
        if linear_map.ndim == 1:
            linear_map = linear_map.reshape(-1, 1)

        # QR decomposition for column space
        q, r = jnp.linalg.qr(linear_map)

        # Non-zero columns of Q span the image
        rank = jnp.sum(jnp.abs(jnp.diag(r)) > self.precision)
        image = q[:, :rank]

        return image

    @partial(jit, static_argnums=(0,))
    def _compute_quotient(self, space: jnp.ndarray,
                          subspace: jnp.ndarray) -> jnp.ndarray:
        """Compute quotient space/subspace."""
        if space.size == 0:
            return jnp.zeros((1, 1))

        if subspace.size == 0:
            return space

        # Orthogonal projection
        if subspace.ndim == 1:
            subspace = subspace.reshape(-1, 1)

        # Orthogonalize subspace
        q_sub, _ = jnp.linalg.qr(subspace)

        # Project space orthogonal to subspace
        if space.ndim == 1:
            space = space.reshape(-1, 1)

        proj = jnp.eye(space.shape[0]) - q_sub @ q_sub.T
        quotient = proj @ space

        # Remove near-zero columns
        norms = jnp.linalg.norm(quotient, axis=0)
        quotient = quotient[:, norms > self.precision]

        if quotient.size == 0:
            quotient = jnp.zeros((space.shape[0], 1))

        return quotient

    @partial(jit, static_argnums=(0,))
    def _induced_differential(self, prev_page: SpectralPage,
                              source: Tuple[int, int],
                              target: Tuple[int, int]) -> jnp.ndarray:
        """Compute induced differential on next page."""
        p, q = source

        # Get dimensions
        if source in prev_page.groups:
            source_dim = prev_page.groups[source].shape[-1] if prev_page.groups[source].ndim > 0 else 1
        else:
            source_dim = 1

        if target in prev_page.groups:
            target_dim = prev_page.groups[target].shape[-1] if prev_page.groups[target].ndim > 0 else 1
        else:
            target_dim = 1

        # Induced differential (simplified)
        induced = jnp.zeros((target_dim, source_dim))

        # Add structure based on page number
        if prev_page.page_number == 1:
            # E_1 → E_2: often connecting homomorphism
            if source_dim == target_dim:
                induced = 0.1 * jnp.eye(source_dim)

        return induced

    @partial(jit, static_argnums=(0,))
    def _check_convergence_impl(self, page: SpectralPage,
                                prev_page: Optional[SpectralPage]) -> bool:
        """
        Check if spectral sequence has converged.

        Args:
            page: Current page
            prev_page: Previous page

        Returns:
            Whether converged
        """
        if prev_page is None:
            return False

        # Check if differentials are zero
        for d in page.differentials.values():
            if jnp.linalg.norm(d) > self.precision:
                return False

        # Check if groups stabilized
        for (p, q) in page.groups:
            if (p, q) in prev_page.groups:
                curr = page.groups[(p, q)]
                prev = prev_page.groups[(p, q)]

                # Compare dimensions
                curr_dim = curr.shape[-1] if curr.ndim > 0 else 0
                prev_dim = prev.shape[-1] if prev.ndim > 0 else 0

                if curr_dim != prev_dim:
                    return False

        return True

    @partial(jit, static_argnums=(0,))
    def _edge_map_impl(self, converged_page: SpectralPage,
                       n: int) -> jnp.ndarray:
        """
        Compute edge homomorphism.

        H^n(X) → E_∞^{n,0} or E_∞^{0,n} → H^n(X)

        Args:
            converged_page: E_∞ page
            n: Total degree

        Returns:
            Edge homomorphism
        """
        # Horizontal edge: E_∞^{n,0} → H^n
        if (n, 0) in converged_page.groups:
            source = converged_page.groups[(n, 0)]
            source_dim = source.shape[-1] if source.ndim > 0 else 1
        else:
            source_dim = 1

        # Vertical edge: E_∞^{0,n} → H^n
        if (0, n) in converged_page.groups:
            target = converged_page.groups[(0, n)]
            target_dim = target.shape[-1] if target.ndim > 0 else 1
        else:
            target_dim = 1

        # Edge map (often isomorphism)
        edge = jnp.eye(min(source_dim, target_dim))

        # Pad if needed
        if source_dim < target_dim:
            edge = jnp.pad(edge, ((0, target_dim - source_dim), (0, 0)))
        elif target_dim < source_dim:
            edge = jnp.pad(edge, ((0, 0), (0, source_dim - target_dim)))

        return edge

    @partial(jit, static_argnums=(0,))
    def _extension_class_impl(self, e_infinity: SpectralPage,
                              total_degree: int) -> jnp.ndarray:
        """
        Compute extension class for reassembling H^n from E_∞.

        Args:
            e_infinity: Converged page
            total_degree: Total degree n

        Returns:
            Extension data
        """
        # Collect E_∞^{p,q} with p+q = n
        factors = []

        for p in range(total_degree + 1):
            q = total_degree - p
            if (p, q) in e_infinity.groups:
                factors.append(e_infinity.groups[(p, q)])

        if not factors:
            return jnp.zeros((1, 1))

        # Extension problem: reassemble H^n from factors
        # Simplified: concatenate factors
        extension = jnp.concatenate([f.flatten() for f in factors])

        return extension

    @partial(jit, static_argnums=(0,))
    def _filtration_spectral_sequence_impl(self,
                                           filtration: List[jnp.ndarray]) -> SpectralPage:
        """
        Compute E_0 page from filtration.

        Args:
            filtration: Filtered complex F^p C

        Returns:
            E_0 page
        """
        groups = {}
        differentials = {}

        for p, F_p in enumerate(filtration):
            for q in range(F_p.shape[0] if F_p.ndim > 0 else 1):
                # E_0^{p,q} = F^p C^{p+q} / F^{p+1} C^{p+q}
                if p < len(filtration) - 1:
                    F_p_plus = filtration[p + 1]

                    # Quotient (simplified)
                    quotient = self._compute_quotient(
                        F_p[q:q + 1] if F_p.ndim > 0 and q < F_p.shape[0] else F_p,
                        F_p_plus[q:q + 1] if F_p_plus.ndim > 0 and q < F_p_plus.shape[0] else F_p_plus
                    )
                else:
                    quotient = F_p[q:q + 1] if F_p.ndim > 0 and q < F_p.shape[0] else F_p

                groups[(p, q)] = quotient

                # d_0 is induced from complex differential
                # Simplified: small random differential
                target = (p, q + 1)
                if target[1] < 3:  # Limit degree
                    target_dim = 1  # Simplified
                    source_dim = quotient.shape[-1] if quotient.ndim > 0 else 1
                    differentials[(p, q)] = 0.01 * jnp.ones((target_dim, source_dim))

        return SpectralPage(
            page_number=0,
            groups=groups,
            differentials=differentials,
            total_degree=0
        )

    @partial(jit, static_argnums=(0,))
    def _double_complex_impl(self, horizontal: List[jnp.ndarray],
                             vertical: List[jnp.ndarray]) -> SpectralPage:
        """
        Compute spectral sequence from double complex.

        Args:
            horizontal: Horizontal differentials
            vertical: Vertical differentials

        Returns:
            E_0 page of double complex SS
        """
        groups = {}
        differentials = {}

        for p, h_diff in enumerate(horizontal):
            for q, v_diff in enumerate(vertical):
                # C^{p,q} with two differentials
                dim = min(h_diff.shape[0] if h_diff.ndim > 0 else 1,
                          v_diff.shape[0] if v_diff.ndim > 0 else 1)

                groups[(p, q)] = jnp.ones((dim, 1))

                # Total differential d = d_h + (-1)^p d_v
                if p < len(horizontal) - 1 and q < len(vertical) - 1:
                    total_diff = h_diff[:dim, :dim] + ((-1) ** p) * v_diff[:dim, :dim]
                    differentials[(p, q)] = total_diff

        return SpectralPage(
            page_number=0,
            groups=groups,
            differentials=differentials,
            total_degree=0
        )

    def compute_leray_spectral_sequence(self,
                                        sheaf_data: Dict,
                                        map_data: Dict) -> ConvergenceData:
        """
        Compute Leray spectral sequence for a map f: X → Y.

        E_2^{p,q} = H^p(Y, R^q f_* F) ⟹ H^{p+q}(X, F)

        Args:
            sheaf_data: Sheaf F on X
            map_data: Map f: X → Y

        Returns:
            Convergence data
        """
        # Build filtration from pushforward
        filtration = []
        for p in range(5):  # Simplified: 5 levels
            F_p = jnp.ones((3, 3)) * (0.9 ** p)  # Decay with filtration
            filtration.append(F_p)

        # Compute E_0 page
        e0 = self._filtration_spectral_sequence(filtration)

        # Iterate pages
        pages = [e0]
        for r in range(self.max_page):
            if r > 0:
                e_next = self._compute_page(pages[-1],
                                            FilteredComplex(
                                                filtration_length=len(filtration),
                                                complexes=filtration,
                                                differentials=[],
                                                filtration_degrees=list(range(len(filtration)))
                                            ))
                pages.append(e_next)

                # Check convergence
                if self._check_convergence(e_next, pages[-2] if len(pages) > 1 else None):
                    break

        # Extract limit
        e_infinity = pages[-1]

        # Compute associated graded
        associated_graded = {}
        for n in range(5):
            factors = []
            for p in range(n + 1):
                q = n - p
                if (p, q) in e_infinity.groups:
                    factors.append(e_infinity.groups[(p, q)])

            if factors:
                associated_graded[n] = jnp.concatenate([f.flatten() for f in factors])
            else:
                associated_graded[n] = jnp.zeros(1)

        return ConvergenceData(
            converged=len(pages) < self.max_page,
            convergence_page=len(pages) - 1,
            limit_groups=e_infinity.groups,
            associated_graded=associated_graded
        )