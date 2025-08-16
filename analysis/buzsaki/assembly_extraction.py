"""
Assembly Extraction and Analysis
=================================

Extract and analyze neural assemblies from Buzsáki lab data using
coordinate-free geometric methods.
GPU-accelerated using JAX.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad
from jax.scipy import linalg as jax_linalg
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from functools import partial

from core.langlands.padic_modular_forms import PadicModularComputer
from core.langlands.eigencurve import EigencurveComputer, EigencurvePoint

# Configure JAX
try:
    if len(jax.devices('gpu')) > 0:
        jax.config.update('jax_platform_name', 'gpu')
except:
    jax.config.update('jax_platform_name', 'cpu')


class NeuralAssembly(NamedTuple):
    """Neural assembly structure."""
    member_neurons: List[int]  # Neuron indices
    eigenform: jnp.ndarray  # Modular eigenform
    hecke_eigenvalues: Dict[int, complex]  # T_p eigenvalues
    stability: float  # Assembly stability
    hierarchical_level: int  # p-adic level


class AssemblyDynamics(NamedTuple):
    """Assembly dynamics over modular curve."""
    trajectory: List[EigencurvePoint]  # Path on eigencurve
    transitions: List[Tuple[int, int]]  # Assembly transitions
    modular_flow: jnp.ndarray  # Flow on X_0(N)
    lyapunov_exponent: float  # Stability measure


class HierarchicalStructure(NamedTuple):
    """Hierarchical organization of assemblies."""
    levels: Dict[int, List[NeuralAssembly]]  # Assemblies by level
    p_adic_tree: Dict[Tuple[int, int], float]  # Tree structure
    ultrametric_distance: jnp.ndarray  # Distance matrix
    log_periodic_signature: float  # Log-periodic modulation


class AssemblyComputer:
    """
    Extract and analyze neural assemblies.
    """

    def __init__(self, prime: int = 3, modular_level: int = 12):
        """
        Initialize assembly computer.

        Args:
            prime: Base prime for p-adic structure
            modular_level: Level of modular curve
        """
        self.p = prime
        self.N = modular_level

        self.padic_computer = PadicModularComputer(prime=self.p)
        self.eigencurve_computer = EigencurveComputer(prime=self.p, tame_level=self.N // self.p)

        # Pre-compile JAX functions
        self._compile_functions()

    def _compile_functions(self):
        """Pre-compile JAX functions."""
        self._detect_assemblies = jit(self._detect_assemblies_impl)
        self._compute_eigenform = jit(self._compute_eigenform_impl)
        self._hecke_eigenvalues = jit(self._hecke_eigenvalues_impl)
        self._assembly_stability = jit(self._assembly_stability_impl)
        self._hierarchical_decomposition = jit(self._hierarchical_decomposition_impl)
        self._ultrametric_clustering = jit(self._ultrametric_clustering_impl)
        self._log_periodic_analysis = jit(self._log_periodic_analysis_impl)
        self._assembly_overlap = jit(self._assembly_overlap_impl)

    @partial(jit, static_argnums=(0,))
    def _detect_assemblies_impl(self, spike_data: jnp.ndarray,
                                window_size: int = 10) -> List[jnp.ndarray]:
        """
        Detect assemblies from spike data.

        Args:
            spike_data: Binary spike matrix (neurons x time)
            window_size: Sliding window size

        Returns:
            List of assembly patterns
        """
        n_neurons, n_time = spike_data.shape
        assemblies = []

        # Slide window to find recurring patterns
        for t in range(0, n_time - window_size, window_size // 2):
            window = spike_data[:, t:t + window_size]

            # Check if enough neurons active
            n_active = jnp.sum(window)
            if n_active > 0.1 * n_neurons * window_size:
                # Compute coactivation matrix
                coactive = window @ window.T

                # Find strongly connected components (assemblies)
                eigenvals, eigenvecs = jnp.linalg.eigh(coactive)

                # Take top eigenvector as assembly
                assembly = eigenvecs[:, -1]
                assemblies.append(assembly)

        return assemblies[:20]  # Limit number

    @partial(jit, static_argnums=(0,))
    def _compute_eigenform_impl(self, assembly_pattern: jnp.ndarray) -> jnp.ndarray:
        """
        Compute modular eigenform for assembly.

        Args:
            assembly_pattern: Assembly activation pattern

        Returns:
            Modular eigenform coefficients
        """
        # Map assembly to modular form via Fourier transform
        fft = jnp.fft.fft(assembly_pattern)

        # Take first N coefficients as q-expansion
        q_expansion = fft[:self.N]

        # Normalize to have leading coefficient 1
        if jnp.abs(q_expansion[0]) > 1e-10:
            q_expansion = q_expansion / q_expansion[0]

        return q_expansion

    @partial(jit, static_argnums=(0,))
    def _hecke_eigenvalues_impl(self, eigenform: jnp.ndarray) -> Dict[int, complex]:
        """
        Compute Hecke eigenvalues for assembly.

        Args:
            eigenform: Modular eigenform

        Returns:
            Hecke eigenvalues T_p(f) = λ_p f
        """
        eigenvalues = {}

        for p in [2, 3, 5, 7]:
            if self.N % p != 0:
                # Apply Hecke operator T_p
                t_p_form = self.padic_computer._hecke_operator_padic_impl(eigenform, p)

                # Compute eigenvalue
                if jnp.linalg.norm(eigenform) > 1e-10:
                    # λ_p = <T_p f, f> / <f, f>
                    lambda_p = jnp.vdot(t_p_form, eigenform) / jnp.vdot(eigenform, eigenform)
                else:
                    lambda_p = 0 + 0j

                eigenvalues[p] = complex(lambda_p)

        return eigenvalues

    @partial(jit, static_argnums=(0,))
    def _assembly_stability_impl(self, assembly_pattern: jnp.ndarray,
                                 noise_level: float = 0.1) -> float:
        """
        Compute assembly stability under perturbations.

        Args:
            assembly_pattern: Assembly pattern
            noise_level: Perturbation strength

        Returns:
            Stability measure
        """
        # Add noise
        noise = noise_level * jnp.random.normal(0, 1, assembly_pattern.shape)
        perturbed = assembly_pattern + noise

        # Project to E8 lattice for error correction
        # Simplified: round to nearest integer pattern
        corrected = jnp.round(perturbed)

        # Measure recovery
        recovery = 1 - jnp.linalg.norm(corrected - assembly_pattern) / (jnp.linalg.norm(assembly_pattern) + 1e-10)

        # KAM stability check
        # Compute frequency vector
        freqs = jnp.abs(jnp.fft.fft(assembly_pattern))[:4]

        # Check Diophantine condition (simplified)
        min_gap = jnp.min(jnp.abs(jnp.diff(freqs)))
        kam_stable = min_gap > 0.1

        stability = recovery * (2.0 if kam_stable else 1.0)

        return float(stability)

    @partial(jit, static_argnums=(0,))
    def _hierarchical_decomposition_impl(self, assemblies: List[jnp.ndarray]) -> Dict[int, List[int]]:
        """
        Decompose assemblies into p-adic hierarchy.

        Args:
            assemblies: List of assembly patterns

        Returns:
            Hierarchical grouping
        """
        hierarchy = {}

        for i, assembly in enumerate(assemblies):
            # Compute p-adic valuation
            # Use norm as proxy for hierarchical level
            norm = jnp.linalg.norm(assembly)

            # Assign level based on p-adic valuation of norm
            level = 0
            val = norm
            while val < 1.0 and level < 10:
                val *= self.p
                level += 1

            if level not in hierarchy:
                hierarchy[level] = []
            hierarchy[level].append(i)

        return hierarchy

    @partial(jit, static_argnums=(0,))
    def _ultrametric_clustering_impl(self, assemblies: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Compute ultrametric distance matrix.

        Args:
            assemblies: Assembly patterns

        Returns:
            Ultrametric distance matrix
        """
        n = len(assemblies)
        distances = jnp.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Ultrametric distance via p-adic valuation
                diff = assemblies[i] - assemblies[j]

                # Find first non-zero coefficient
                val = 0
                for k in range(len(diff)):
                    if jnp.abs(diff[k]) > 1e-10:
                        val = k
                        break

                # Distance = p^(-valuation)
                dist = self.p ** (-val / 10.0)

                distances = distances.at[i, j].set(dist)
                distances = distances.at[j, i].set(dist)

        return distances

    @partial(jit, static_argnums=(0,))
    def _log_periodic_analysis_impl(self, time_series: jnp.ndarray) -> float:
        """
        Detect log-periodic signatures.

        Args:
            time_series: Activity time series

        Returns:
            Log-periodic strength
        """
        # Compute log-spaced spectrum
        n = len(time_series)
        freqs = jnp.fft.fftfreq(n)
        fft = jnp.fft.fft(time_series)
        power = jnp.abs(fft) ** 2

        # Look for log-periodic peaks
        log_freqs = jnp.log(jnp.abs(freqs) + 1e-10)

        # Find peaks in log-frequency space
        # Simplified: check for regular spacing in log domain
        peaks = []
        for i in range(1, len(power) // 2):
            if power[i] > power[i - 1] and power[i] > power[i + 1]:
                peaks.append(log_freqs[i])

        if len(peaks) > 2:
            # Check spacing
            spacings = jnp.diff(jnp.array(peaks))
            regularity = 1.0 / (jnp.std(spacings) + 1e-10)
        else:
            regularity = 0.0

        return float(regularity)

    @partial(jit, static_argnums=(0,))
    def _assembly_overlap_impl(self, assembly1: jnp.ndarray,
                               assembly2: jnp.ndarray) -> float:
        """
        Compute overlap between assemblies.

        Args:
            assembly1, assembly2: Assembly patterns

        Returns:
            Overlap measure
        """
        # Normalize
        a1_norm = assembly1 / (jnp.linalg.norm(assembly1) + 1e-10)
        a2_norm = assembly2 / (jnp.linalg.norm(assembly2) + 1e-10)

        # Compute overlap
        overlap = jnp.abs(jnp.dot(a1_norm, a2_norm))

        return float(overlap)

    def extract_assemblies(self, spike_data: jnp.ndarray) -> List[NeuralAssembly]:
        """
        Extract neural assemblies from spike data.

        Args:
            spike_data: Spike matrix

        Returns:
            List of neural assemblies
        """
        # Detect assembly patterns
        patterns = self._detect_assemblies(spike_data)

        assemblies = []

        for i, pattern in enumerate(patterns):
            # Compute eigenform
            eigenform = self._compute_eigenform(pattern)

            # Compute Hecke eigenvalues
            hecke_vals = self._hecke_eigenvalues(eigenform)

            # Compute stability
            stability = self._assembly_stability(pattern)

            # Find member neurons
            threshold = 0.5 * jnp.max(jnp.abs(pattern))
            members = [int(j) for j in range(len(pattern))
                       if jnp.abs(pattern[j]) > threshold]

            # Assign hierarchical level
            level = int(jnp.log(i + 1) / jnp.log(self.p))

            assembly = NeuralAssembly(
                member_neurons=members[:20],  # Limit size
                eigenform=eigenform,
                hecke_eigenvalues=hecke_vals,
                stability=stability,
                hierarchical_level=level
            )

            assemblies.append(assembly)

        return assemblies

    def compute_dynamics(self, assemblies: List[NeuralAssembly],
                         neural_data: jnp.ndarray) -> AssemblyDynamics:
        """
        Compute assembly dynamics on eigencurve.

        Args:
            assemblies: List of assemblies
            neural_data: Full neural data

        Returns:
            Assembly dynamics
        """
        # Map assemblies to eigencurve
        trajectory = []

        for assembly in assemblies:
            # Convert to eigencurve point
            if neural_data.ndim > 1:
                state = neural_data[:, :len(assembly.eigenform)].mean(axis=0)
            else:
                state = neural_data[:len(assembly.eigenform)]

            point = self.eigencurve_computer.biological_state_to_eigencurve(state)
            trajectory.append(point)

        # Compute transitions
        transitions = []
        for i in range(len(trajectory) - 1):
            transitions.append((i, i + 1))

        # Compute modular flow
        if trajectory:
            weights = jnp.array([p.weight for p in trajectory])
            eigenvalues = jnp.array([p.up_eigenvalue for p in trajectory])
            flow = jnp.column_stack([jnp.real(weights), jnp.imag(weights),
                                     jnp.real(eigenvalues), jnp.imag(eigenvalues)])
        else:
            flow = jnp.zeros((1, 4))

        # Compute Lyapunov exponent (simplified)
        if len(flow) > 1:
            divergence = jnp.std(jnp.diff(flow, axis=0))
            lyapunov = jnp.log(divergence + 1) / len(flow)
        else:
            lyapunov = 0.0

        return AssemblyDynamics(
            trajectory=trajectory,
            transitions=transitions,
            modular_flow=flow,
            lyapunov_exponent=float(lyapunov)
        )

    def analyze_hierarchy(self, assemblies: List[NeuralAssembly]) -> HierarchicalStructure:
        """
        Analyze hierarchical organization.

        Args:
            assemblies: List of assemblies

        Returns:
            Hierarchical structure
        """
        # Group by level
        levels = {}
        for assembly in assemblies:
            level = assembly.hierarchical_level
            if level not in levels:
                levels[level] = []
            levels[level].append(assembly)

        # Build p-adic tree
        tree = {}
        for i, a1 in enumerate(assemblies):
            for j, a2 in enumerate(assemblies):
                if i < j:
                    # Ultrametric distance
                    overlap = self._assembly_overlap(a1.eigenform, a2.eigenform)
                    dist = self.p ** (-overlap * 10)
                    tree[(i, j)] = float(dist)

        # Compute ultrametric distance matrix
        patterns = [a.eigenform for a in assemblies]
        if patterns:
            ultra_dist = self._ultrametric_clustering(patterns)
        else:
            ultra_dist = jnp.zeros((1, 1))

        # Detect log-periodic signature
        if assemblies:
            # Use first assembly's activity as proxy
            time_series = assemblies[0].eigenform
            log_periodic = self._log_periodic_analysis(time_series)
        else:
            log_periodic = 0.0

        return HierarchicalStructure(
            levels=levels,
            p_adic_tree=tree,
            ultrametric_distance=ultra_dist,
            log_periodic_signature=log_periodic
        )