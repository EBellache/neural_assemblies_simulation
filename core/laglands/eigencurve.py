"""
Coleman-Mazur Eigencurve
=========================

Implements the p-adic eigencurve parametrizing overconvergent
modular eigenforms, representing biological state spaces.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
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


class EigencurvePoint(NamedTuple):
    """Point on the eigencurve."""
    weight: complex  # Weight (p-adic)
    eigenform: jnp.ndarray  # Overconvergent eigenform
    up_eigenvalue: complex  # U_p eigenvalue
    hecke_eigenvalues: jnp.ndarray  # Hecke eigenvalues
    slope: float  # p-adic slope
    is_classical: bool  # Whether point is classical


class WeightSpace(NamedTuple):
    """p-adic weight space."""
    dimension: int
    base_prime: int
    disc_radius: float  # Radius of p-adic disc
    center: complex  # Center of disc


class SpectralData(NamedTuple):
    """Spectral data on eigencurve."""
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    spectral_measure: jnp.ndarray
    density_of_states: jnp.ndarray


class EigencurveComponent(NamedTuple):
    """Connected component of eigencurve."""
    points: List[EigencurvePoint]
    genus: int  # Genus of component
    ramification_points: List[complex]  # Ramification over weight space
    is_cuspidal: bool  # Cuspidal vs Eisenstein


class EigencurveComputer:
    """
    Compute the Coleman-Mazur eigencurve.
    """

    def __init__(self, prime: int = 3, tame_level: int = 1):
        """
        Initialize eigencurve computer.

        Args:
            prime: Base prime p
            tame_level: Level prime to p
        """
        self.p = prime
        self.N = tame_level
        self.precision = 20  # p-adic precision

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._fredholm_determinant = jit(self._fredholm_determinant_impl)
        self._characteristic_power_series = jit(self._characteristic_power_series_impl)
        self._newton_polygon = jit(self._newton_polygon_impl)
        self._find_eigencurve_points = jit(self._find_eigencurve_points_impl)
        self._spectral_halo = jit(self._spectral_halo_impl)
        self._ramification_locus = jit(self._ramification_locus_impl)
        self._tangent_space = jit(self._tangent_space_impl)
        self._gauss_manin_connection = jit(self._gauss_manin_connection_impl)

    @partial(jit, static_argnums=(0,))
    def _fredholm_determinant_impl(self, operator: jnp.ndarray,
                                   parameter: complex) -> complex:
        """
        Compute Fredholm determinant det(1 - parameter * operator).

        Args:
            operator: Compact operator (U_p)
            parameter: Complex parameter

        Returns:
            Fredholm determinant
        """
        # Use finite rank approximation
        n = operator.shape[0]

        # det(1 - z*T) = exp(-Σ z^n/n * Tr(T^n))
        det_log = 0.0 + 0j
        T_power = operator.copy()
        z_power = parameter

        for n in range(1, min(20, operator.shape[0])):
            trace = jnp.trace(T_power)
            det_log -= z_power * trace / n

            T_power = T_power @ operator
            z_power *= parameter

            if jnp.abs(z_power) < 1e-10:
                break

        return jnp.exp(det_log)

    @partial(jit, static_argnums=(0,))
    def _characteristic_power_series_impl(self, up_operator: jnp.ndarray) -> jnp.ndarray:
        """
        Compute characteristic power series of U_p.

        P(T) = det(1 - T*U_p) = Σ a_n T^n

        Args:
            up_operator: U_p operator matrix

        Returns:
            Coefficients of characteristic series
        """
        n = up_operator.shape[0]
        coeffs = jnp.zeros(n + 1, dtype=complex)

        # Compute via characteristic polynomial
        char_poly = jnp.poly(up_operator)

        # Reverse for power series
        for i in range(min(len(char_poly), n + 1)):
            coeffs = coeffs.at[i].set(char_poly[-(i + 1)])

        return coeffs

    @partial(jit, static_argnums=(0,))
    def _newton_polygon_impl(self, power_series: jnp.ndarray) -> List[Tuple[int, float]]:
        """
        Compute Newton polygon of power series.

        Args:
            power_series: Coefficients a_n

        Returns:
            Vertices of Newton polygon
        """
        # Find p-adic valuations
        valuations = []
        for i, coeff in enumerate(power_series):
            if jnp.abs(coeff) > 1e-10:
                # v_p(coeff)
                val = 0
                c = jnp.abs(coeff)
                while c < 1:
                    val += 1
                    c *= self.p
                valuations.append((i, val))

        if not valuations:
            return [(0, 0)]

        # Compute lower convex hull
        vertices = [valuations[0]]

        for i in range(1, len(valuations)):
            # Check if point is below current hull
            while len(vertices) > 1:
                # Slope from vertices[-2] to vertices[-1]
                dx1 = vertices[-1][0] - vertices[-2][0]
                dy1 = vertices[-1][1] - vertices[-2][1]

                # Slope from vertices[-2] to valuations[i]
                dx2 = valuations[i][0] - vertices[-2][0]
                dy2 = valuations[i][1] - vertices[-2][1]

                # Check convexity
                if dy1 * dx2 <= dy2 * dx1:
                    vertices.pop()
                else:
                    break

            vertices.append(valuations[i])

        return vertices

    @partial(jit, static_argnums=(0,))
    def _find_eigencurve_points_impl(self, weight_disc: WeightSpace,
                                     up_matrix_family: Callable) -> List[EigencurvePoint]:
        """
        Find eigencurve points over a weight disc.

        Args:
            weight_disc: p-adic disc in weight space
            up_matrix_family: Family of U_p operators

        Returns:
            Points on eigencurve
        """
        points = []

        # Sample weights in disc
        n_samples = 20
        for i in range(n_samples):
            # p-adic weight
            theta = 2 * jnp.pi * i / n_samples
            radius = weight_disc.disc_radius * (0.5 + 0.5 * i / n_samples)
            w = weight_disc.center + radius * jnp.exp(1j * theta)

            # Get U_p at this weight
            up_matrix = up_matrix_family(w)

            # Find eigenvalues/eigenvectors
            eigenvals, eigenvecs = jnp.linalg.eig(up_matrix)

            # Each eigenvalue gives an eigencurve point
            for j, (eval, evec) in enumerate(zip(eigenvals, eigenvecs.T)):
                # p-adic slope
                slope = -jnp.log(jnp.abs(eval)) / jnp.log(self.p)

                # Check if classical (integer weight, slope < k)
                is_classical = (jnp.abs(w - jnp.round(jnp.real(w))) < 1e-6 and
                                slope < jnp.real(w))

                point = EigencurvePoint(
                    weight=w,
                    eigenform=evec,
                    up_eigenvalue=eval,
                    hecke_eigenvalues=jnp.array([eval]),  # Simplified
                    slope=float(slope),
                    is_classical=is_classical
                )
                points.append(point)

        return points

    @partial(jit, static_argnums=(0,))
    def _spectral_halo_impl(self, center_point: EigencurvePoint,
                            radius: float) -> SpectralData:
        """
        Compute spectral halo around eigencurve point.

        Args:
            center_point: Center point on eigencurve
            radius: p-adic radius

        Returns:
            Spectral data in neighborhood
        """
        # Perturb around center point
        n_samples = 10
        all_eigenvals = []
        all_eigenvecs = []

        for i in range(n_samples):
            # Perturbed weight
            theta = 2 * jnp.pi * i / n_samples
            w = center_point.weight + radius * jnp.exp(1j * theta)

            # Mock eigenvalues near center (would compute from U_p)
            perturbation = 0.1 * radius * jnp.exp(1j * theta)
            eigenval = center_point.up_eigenvalue + perturbation

            all_eigenvals.append(eigenval)
            all_eigenvecs.append(center_point.eigenform)

        eigenvalues = jnp.array(all_eigenvals)
        eigenvectors = jnp.array(all_eigenvecs)

        # Spectral measure (density of eigenvalues)
        spectral_measure = jnp.ones(len(eigenvalues)) / len(eigenvalues)

        # Density of states
        dos = jnp.histogram(jnp.real(eigenvalues), bins=10)[0]
        dos = dos / jnp.sum(dos)

        return SpectralData(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_measure=spectral_measure,
            density_of_states=dos
        )

    @partial(jit, static_argnums=(0,))
    def _ramification_locus_impl(self, eigencurve_points: List[EigencurvePoint]) -> List[complex]:
        """
        Find ramification points over weight space.

        Args:
            eigencurve_points: Points on eigencurve

        Returns:
            Ramification points (weights)
        """
        ramification = []

        # Group points by weight
        weight_groups = {}
        for point in eigencurve_points:
            w = complex(point.weight)
            if w not in weight_groups:
                weight_groups[w] = []
            weight_groups[w].append(point)

        # Ramification where multiple sheets meet
        for w, points in weight_groups.items():
            if len(points) > 1:
                # Check if slopes collide
                slopes = [p.slope for p in points]
                if len(set(slopes)) < len(slopes):
                    ramification.append(w)

        return ramification

    @partial(jit, static_argnums=(0,))
    def _tangent_space_impl(self, point: EigencurvePoint) -> jnp.ndarray:
        """
        Compute tangent space to eigencurve at point.

        Args:
            point: Point on eigencurve

        Returns:
            Tangent vector
        """
        # Tangent to eigencurve in (weight, eigenvalue) space
        # Given by implicit differentiation of det(U_p - λ) = 0

        # Approximate tangent via finite difference
        h = 1e-6
        w_perturbed = point.weight + h

        # Linear approximation to eigenvalue change
        # dλ/dw ≈ -Tr(dU_p/dw) / (derivative of characteristic poly)

        # Simplified: assume linear variation
        tangent = jnp.array([1.0, -point.slope])  # [dw, dλ]

        # Normalize
        tangent = tangent / jnp.linalg.norm(tangent)

        return tangent

    @partial(jit, static_argnums=(0,))
    def _gauss_manin_connection_impl(self, path: List[EigencurvePoint]) -> jnp.ndarray:
        """
        Compute Gauss-Manin connection along path.

        Parallel transport of eigenforms.

        Args:
            path: Path on eigencurve

        Returns:
            Parallel transport matrix
        """
        if len(path) < 2:
            return jnp.eye(len(path[0].eigenform))

        n = len(path[0].eigenform)
        transport = jnp.eye(n)

        for i in range(len(path) - 1):
            # Connection 1-form
            v1 = path[i].eigenform
            v2 = path[i + 1].eigenform

            # Parallel transport minimizes |v2 - Av1|
            # A = v2 @ v1^T / |v1|^2
            A = jnp.outer(v2, v1) / (jnp.dot(v1, v1) + 1e-8)

            transport = A @ transport

        return transport

    def biological_state_to_eigencurve(self, state: jnp.ndarray) -> EigencurvePoint:
        """
        Map biological state to point on eigencurve.

        Args:
            state: Biological state vector

        Returns:
            Corresponding eigencurve point
        """
        # Extract spectral data from state
        if state.ndim > 1:
            # Use correlation matrix
            corr = jnp.corrcoef(state)
            eigenvals, eigenvecs = jnp.linalg.eig(corr)
        else:
            # Direct spectral analysis
            n = len(state)
            matrix = jnp.outer(state, state)
            eigenvals, eigenvecs = jnp.linalg.eig(matrix)

        # Dominant eigenvalue gives U_p eigenvalue
        idx = jnp.argmax(jnp.abs(eigenvals))
        up_eval = eigenvals[idx]
        eigenform = eigenvecs[:, idx]

        # p-adic weight from trace
        weight = jnp.trace(corr if state.ndim > 1 else matrix)
        weight = weight % (self.p - 1)  # Reduce mod p-1

        # Slope
        slope = -jnp.log(jnp.abs(up_eval)) / jnp.log(self.p)

        # Classical if integer weight
        is_classical = jnp.abs(weight - jnp.round(weight)) < 1e-6

        return EigencurvePoint(
            weight=complex(weight),
            eigenform=eigenform,
            up_eigenvalue=complex(up_eval),
            hecke_eigenvalues=jnp.array([up_eval]),
            slope=float(slope),
            is_classical=bool(is_classical)
        )

    def predict_transitions(self, current_point: EigencurvePoint,
                            perturbation: jnp.ndarray) -> List[EigencurvePoint]:
        """
        Predict state transitions along eigencurve.

        Args:
            current_point: Current state
            perturbation: External perturbation

        Returns:
            Possible future states
        """
        predictions = []

        # Perturbation changes weight
        weight_change = jnp.sum(perturbation) % (self.p - 1)
        new_weight = current_point.weight + weight_change

        # Follow eigencurve to new weight
        # Multiple sheets possible at new weight

        # Sheet 1: Continue along same component
        new_slope = current_point.slope + 0.1 * weight_change
        new_eval = self.p ** (-new_slope)

        pred1 = EigencurvePoint(
            weight=new_weight,
            eigenform=current_point.eigenform,
            up_eigenvalue=new_eval,
            hecke_eigenvalues=jnp.array([new_eval]),
            slope=new_slope,
            is_classical=False
        )
        predictions.append(pred1)

        # Sheet 2: Jump to different component (if ramified)
        if abs(new_slope - round(new_slope)) < 0.1:
            # Near integer slope - possible ramification
            alt_slope = round(new_slope) + 0.5
            alt_eval = self.p ** (-alt_slope)

            pred2 = EigencurvePoint(
                weight=new_weight,
                eigenform=-current_point.eigenform,  # Opposite sheet
                up_eigenvalue=alt_eval,
                hecke_eigenvalues=jnp.array([alt_eval]),
                slope=alt_slope,
                is_classical=False
            )
            predictions.append(pred2)

        return predictions