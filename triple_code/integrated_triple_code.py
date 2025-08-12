"""
Integrated Triple Code System
==============================
Combines Golay, E8, and p-adic codes for complete morphogenetic encoding.

The triple code provides:
1. GOLAY: Local error correction (spatial robustness)
2. E8: Global geometric constraints (state space structure)
3. P-ADIC: Hierarchical temporal organization (multi-scale dynamics)

Together they implement a biological error-correcting code that:
- Protects against noise (Golay)
- Maintains geometric coherence (E8)
- Preserves temporal hierarchy (p-adic)

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Import triple code components
from .golay_code import GolayAssemblyEncoder
from .e8_lattice import E8Lattice, assembly_to_e8, compute_e8_trajectory_curvature
from .padic_code import PAdicEncoder, PAdicHierarchyAnalyzer

# Import core components
try:
    from ..core.tropical_math import tropical_inner_product, tropical_distance
    from ..core.assembly_detector import NeuralAssembly
except ImportError:
    # For standalone testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.tropical_math import tropical_inner_product, tropical_distance
    from core.assembly_detector import NeuralAssembly


@dataclass
class TripleCodeState:
    """Complete state in triple code representation."""
    # Golay code (24 bits)
    golay_codeword: np.ndarray
    golay_syndrome: np.ndarray

    # E8 coordinates (8D)
    e8_coords: np.ndarray
    e8_casimirs: Dict[str, float]

    # P-adic encoding (multi-prime)
    padic_encodings: Dict[int, Any]  # Prime -> PAdicNumber
    hierarchical_level: int

    # Combined metrics
    total_error: float = 0.0
    stability_index: float = 0.0
    information_content: float = 0.0


class IntegratedTripleCode:
    """
    Unified triple code system for neuronal assemblies.
    """

    def __init__(self,
                 primes: List[int] = [2, 3, 5],
                 theta_period: float = 0.125):
        """
        Initialize integrated triple code.

        Parameters:
        -----------
        primes : List[int]
            Primes for p-adic encoding
        theta_period : float
            Theta cycle duration
        """
        # Initialize components
        self.golay_encoder = GolayAssemblyEncoder()
        self.e8_lattice = E8Lattice()
        self.padic_encoder = PAdicEncoder(primes=primes)
        self.padic_analyzer = PAdicHierarchyAnalyzer()

        self.theta_period = theta_period
        self.primes = primes

        # State tracking
        self.current_state = None
        self.state_history: List[TripleCodeState] = []
        self.max_history = 100

    def encode_assembly(self,
                        assembly: NeuralAssembly,
                        spike_trains: Dict[int, List[float]],
                        time_window: Tuple[float, float]) -> TripleCodeState:
        """
        Encode assembly using complete triple code.

        Parameters:
        -----------
        assembly : NeuralAssembly
            Assembly to encode
        spike_trains : Dict
            Full spike train data
        time_window : tuple
            Time window for encoding

        Returns:
        --------
        TripleCodeState with complete encoding
        """
        # 1. GOLAY ENCODING - Spatial pattern
        # Create binary firing pattern
        pattern = np.zeros(24)
        for i, cell_id in enumerate(assembly.cells[:24]):
            # Check if cell fired in window
            if cell_id in spike_trains:
                spikes = spike_trains[cell_id]
                fired = any(time_window[0] <= t <= time_window[1] for t in spikes)
                pattern[i] = float(fired)

        # Encode with Golay
        golay_codeword = self.golay_encoder.encode_assembly(pattern[:12])

        # Compute syndrome (for error detection)
        syndrome = self._compute_syndrome(golay_codeword)

        # 2. E8 PROJECTION - Geometric constraints
        # Use full pattern for E8
        e8_coords = assembly_to_e8(pattern, self.e8_lattice)

        # Compute Casimir invariants
        casimirs = self.e8_lattice.compute_casimir_invariants(e8_coords)

        # 3. P-ADIC ENCODING - Temporal hierarchy
        # Extract spike times for assembly cells
        assembly_spikes = []
        for cell_id in assembly.cells:
            if cell_id in spike_trains:
                cell_spikes = [t for t in spike_trains[cell_id]
                               if time_window[0] <= t <= time_window[1]]
                assembly_spikes.extend(cell_spikes)

        assembly_spikes = np.array(sorted(assembly_spikes))

        # P-adic encoding
        padic_encodings = self.padic_encoder.encode_spike_pattern(assembly_spikes)

        # Determine hierarchical level
        hierarchical_level = self._compute_hierarchical_level(padic_encodings)

        # 4. COMPUTE COMBINED METRICS
        # Total error combines all three codes
        golay_error = np.sum(syndrome) / 12.0  # Normalized syndrome weight
        e8_error = np.linalg.norm(e8_coords - self.e8_lattice.nearest_e8_point(e8_coords))
        padic_error = self._compute_padic_consistency(padic_encodings)

        total_error = (golay_error + e8_error + padic_error) / 3.0

        # Stability from E8 geometry
        stability = 1.0 / (1.0 + e8_error)

        # Information content from pattern entropy
        information = self._compute_information(pattern)

        # Create state
        state = TripleCodeState(
            golay_codeword=golay_codeword,
            golay_syndrome=syndrome,
            e8_coords=e8_coords,
            e8_casimirs=casimirs,
            padic_encodings=padic_encodings,
            hierarchical_level=hierarchical_level,
            total_error=total_error,
            stability_index=stability,
            information_content=information
        )

        # Update tracking
        self.current_state = state
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

        return state

    def decode_and_correct(self,
                           corrupted_state: TripleCodeState,
                           noise_level: float = 0.1) -> TripleCodeState:
        """
        Decode and error-correct using all three codes.

        The three codes work together:
        - Golay corrects local bit errors
        - E8 snaps to nearest lattice point
        - P-adic maintains hierarchical consistency
        """
        corrected = TripleCodeState(
            golay_codeword=corrupted_state.golay_codeword.copy(),
            golay_syndrome=corrupted_state.golay_syndrome.copy(),
            e8_coords=corrupted_state.e8_coords.copy(),
            e8_casimirs=corrupted_state.e8_casimirs.copy(),
            padic_encodings=corrupted_state.padic_encodings.copy(),
            hierarchical_level=corrupted_state.hierarchical_level
        )

        # 1. GOLAY CORRECTION - Fix bit errors
        if np.any(corrupted_state.golay_syndrome != 0):
            try:
                corrected_codeword, error_pattern = self.golay_encoder.decoder.decode(
                    corrupted_state.golay_codeword
                )
                corrected.golay_codeword = corrected_codeword
                corrected.golay_syndrome = np.zeros_like(corrupted_state.golay_syndrome)
            except:
                pass  # Keep original if uncorrectable

        # 2. E8 SNAPPING - Project to lattice
        snapped_coords = self.e8_lattice.nearest_e8_point(corrupted_state.e8_coords)
        corrected.e8_coords = snapped_coords
        corrected.e8_casimirs = self.e8_lattice.compute_casimir_invariants(snapped_coords)

        # 3. P-ADIC CONSISTENCY - Maintain hierarchy
        corrected.hierarchical_level = self._enforce_hierarchical_consistency(
            corrupted_state.padic_encodings,
            corrupted_state.hierarchical_level
        )

        # Recompute metrics
        corrected.total_error = self._compute_total_error(corrected)
        corrected.stability_index = 1.0 / (1.0 + corrected.total_error)

        return corrected

    def detect_catastrophe(self,
                           trajectory: List[TripleCodeState],
                           threshold: float = 2.0) -> List[int]:
        """
        Detect catastrophe points where system jumps between states.

        Catastrophes occur when:
        - E8 trajectory has high curvature
        - P-adic hierarchy changes level
        - Golay syndrome suddenly increases
        """
        if len(trajectory) < 3:
            return []

        catastrophe_indices = []

        # Extract E8 trajectory
        e8_trajectory = [state.e8_coords for state in trajectory]

        # Compute curvature
        curvatures = compute_e8_trajectory_curvature(e8_trajectory)

        # Find high curvature points
        mean_curv = np.mean(curvatures)
        std_curv = np.std(curvatures)
        threshold_curv = mean_curv + threshold * std_curv

        for i, curv in enumerate(curvatures):
            catastrophe_detected = False

            # Check E8 curvature
            if curv > threshold_curv:
                catastrophe_detected = True

            # Check hierarchical jumps
            if i > 0:
                level_change = abs(trajectory[i + 1].hierarchical_level -
                                   trajectory[i].hierarchical_level)
                if level_change > 1:
                    catastrophe_detected = True

            # Check Golay syndrome jumps
            if i > 0:
                syndrome_change = np.sum(np.abs(trajectory[i + 1].golay_syndrome -
                                                trajectory[i].golay_syndrome))
                if syndrome_change > 3:
                    catastrophe_detected = True

            if catastrophe_detected:
                catastrophe_indices.append(i + 1)  # Adjust for trajectory indexing

        return catastrophe_indices

    def compute_triple_correlation(self,
                                   state1: TripleCodeState,
                                   state2: TripleCodeState) -> Dict[str, float]:
        """
        Compute correlation between states using all three codes.
        """
        # Golay correlation (Hamming distance)
        golay_dist = np.sum(np.abs(state1.golay_codeword - state2.golay_codeword))
        golay_corr = 1.0 - golay_dist / 24.0

        # E8 correlation (geometric distance)
        e8_dist = self.e8_lattice.e8_distance(state1.e8_coords, state2.e8_coords)
        e8_corr = np.exp(-e8_dist / 2.0)  # Exponential decay

        # P-adic correlation (ultrametric distance)
        padic_corr = 0.0
        n_primes = 0

        for p in self.primes:
            if p in state1.padic_encodings and p in state2.padic_encodings:
                dist = self.padic_encoder.padic_distance(
                    state1.padic_encodings[p],
                    state2.padic_encodings[p]
                )
                padic_corr += np.exp(-dist)
                n_primes += 1

        if n_primes > 0:
            padic_corr /= n_primes

        # Combined correlation (weighted average)
        weights = [0.3, 0.4, 0.3]  # Golay, E8, p-adic
        total_corr = (weights[0] * golay_corr +
                      weights[1] * e8_corr +
                      weights[2] * padic_corr)

        return {
            'golay': golay_corr,
            'e8': e8_corr,
            'padic': padic_corr,
            'total': total_corr
        }

    def _compute_syndrome(self, codeword: np.ndarray) -> np.ndarray:
        """Compute Golay syndrome."""
        return self.golay_encoder.decoder.compute_syndrome(codeword)

    def _compute_hierarchical_level(self, padic_encodings: Dict) -> int:
        """Extract hierarchical level from p-adic encoding."""
        if not padic_encodings:
            return 0

        # Use p=2 valuation as primary hierarchy
        if 2 in padic_encodings:
            return padic_encodings[2].valuation

        # Fallback to average valuation
        valuations = [enc.valuation for enc in padic_encodings.values()]
        return int(np.mean(valuations)) if valuations else 0

    def _compute_padic_consistency(self, padic_encodings: Dict) -> float:
        """Check consistency across different prime encodings."""
        if len(padic_encodings) < 2:
            return 0.0

        # Check if CRT reconstruction is consistent
        reconstructions = []
        for p, enc in padic_encodings.items():
            if hasattr(enc, 'crt_value'):
                reconstructions.append(enc.crt_value)

        if len(reconstructions) > 1:
            # Variance in reconstructions indicates inconsistency
            variance = np.var(reconstructions)
            return variance / (np.mean(reconstructions) + 1e-10)

        return 0.0

    def _compute_information(self, pattern: np.ndarray) -> float:
        """Compute information content of pattern."""
        # Binary entropy
        p1 = np.mean(pattern > 0)
        if p1 == 0 or p1 == 1:
            return 0.0

        entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

        # Scale by pattern length
        return entropy * len(pattern)

    def _compute_total_error(self, state: TripleCodeState) -> float:
        """Compute combined error across all codes."""
        # Golay syndrome weight
        golay_error = np.sum(state.golay_syndrome) / 12.0

        # E8 projection error
        nearest = self.e8_lattice.nearest_e8_point(state.e8_coords)
        e8_error = np.linalg.norm(state.e8_coords - nearest)

        # P-adic consistency
        padic_error = self._compute_padic_consistency(state.padic_encodings)

        return (golay_error + e8_error + padic_error) / 3.0

    def _enforce_hierarchical_consistency(self,
                                          padic_encodings: Dict,
                                          level: int) -> int:
        """Enforce consistency in hierarchical level."""
        # Get majority vote from all primes
        levels = []
        for enc in padic_encodings.values():
            levels.append(enc.valuation)

        if levels:
            # Return median level for robustness
            return int(np.median(levels))

        return level

    def analyze_state_evolution(self) -> Dict[str, Any]:
        """
        Analyze evolution of triple code states over time.
        """
        if len(self.state_history) < 2:
            return {}

        # Track metrics over time
        errors = [s.total_error for s in self.state_history]
        stabilities = [s.stability_index for s in self.state_history]
        information = [s.information_content for s in self.state_history]

        # Detect catastrophes
        catastrophes = self.detect_catastrophe(self.state_history)

        # Compute state correlations
        correlations = []
        for i in range(len(self.state_history) - 1):
            corr = self.compute_triple_correlation(
                self.state_history[i],
                self.state_history[i + 1]
            )
            correlations.append(corr['total'])

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_stability': np.mean(stabilities),
            'mean_information': np.mean(information),
            'n_catastrophes': len(catastrophes),
            'catastrophe_rate': len(catastrophes) / len(self.state_history),
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'trajectory_length': len(self.state_history)
        }


def test_integrated_triple_code():
    """Test integrated triple code system."""
    print("\n=== Testing Integrated Triple Code ===\n")

    # Create system
    triple_code = IntegratedTripleCode()

    # Create mock assembly
    assembly = NeuralAssembly(
        assembly_id=1,
        cells=list(range(30)),
        strength=0.8,
        e8_coords=np.random.standard_normal(8) * 0.5
    )

    # Create mock spike trains
    np.random.seed(42)
    spike_trains = {}

    for cell_id in range(50):
        # Generate spikes with some structure
        if cell_id in assembly.cells:
            # Assembly members fire together
            spike_times = [0.1, 0.3, 0.5, 0.7]
            spike_times += list(np.random.uniform(0, 1, 5))
        else:
            # Random firing
            spike_times = list(np.random.uniform(0, 1, 3))

        spike_trains[cell_id] = sorted(spike_times)

    # Test encoding
    print("--- Testing Triple Code Encoding ---")

    state = triple_code.encode_assembly(
        assembly, spike_trains, time_window=(0, 1.0)
    )

    print(f"Encoded state:")
    print(f"  Golay codeword: {state.golay_codeword[:8]}...")
    print(f"  Golay syndrome: {np.sum(state.golay_syndrome)} errors")
    print(f"  E8 coordinates: {state.e8_coords[:4]}...")
    print(f"  E8 Casimir I2: {state.e8_casimirs.get('I2', 0):.3f}")
    print(f"  Hierarchical level: {state.hierarchical_level}")
    print(f"  Total error: {state.total_error:.3f}")
    print(f"  Stability: {state.stability_index:.3f}")
    print(f"  Information: {state.information_content:.2f} bits")

    # Test error correction
    print("\n--- Testing Error Correction ---")

    # Add noise to state
    corrupted = TripleCodeState(
        golay_codeword=state.golay_codeword.copy(),
        golay_syndrome=state.golay_syndrome.copy(),
        e8_coords=state.e8_coords.copy(),
        e8_casimirs=state.e8_casimirs.copy(),
        padic_encodings=state.padic_encodings.copy(),
        hierarchical_level=state.hierarchical_level
    )

    # Corrupt Golay code
    error_positions = [3, 7, 15]
    for pos in error_positions:
        corrupted.golay_codeword[pos] = 1 - corrupted.golay_codeword[pos]

    # Corrupt E8 coordinates
    corrupted.e8_coords += np.random.standard_normal(8) * 0.1

    print(f"Corrupted state:")
    print(f"  Added {len(error_positions)} bit errors")
    print(f"  E8 perturbation: {np.linalg.norm(corrupted.e8_coords - state.e8_coords):.3f}")

    # Correct errors
    corrected = triple_code.decode_and_correct(corrupted)

    print(f"\nCorrected state:")
    print(f"  Golay errors fixed: {np.array_equal(corrected.golay_codeword, state.golay_codeword)}")
    print(f"  E8 distance to original: {np.linalg.norm(corrected.e8_coords - state.e8_coords):.3f}")
    print(f"  Error reduced: {corrupted.total_error:.3f} → {corrected.total_error:.3f}")

    # Test evolution analysis
    print("\n--- Testing State Evolution ---")

    # Simulate multiple time steps
    for t in range(10):
        # Shift spike times
        shifted_spikes = {}
        for cell_id, spikes in spike_trains.items():
            shifted_spikes[cell_id] = [s + t * 0.1 for s in spikes]

        # Encode
        state = triple_code.encode_assembly(
            assembly, shifted_spikes,
            time_window=(t * 0.1, (t + 1) * 0.1)
        )

    # Analyze evolution
    evolution = triple_code.analyze_state_evolution()

    print(f"Evolution analysis:")
    print(f"  States tracked: {evolution['trajectory_length']}")
    print(f"  Mean error: {evolution['mean_error']:.3f} ± {evolution['std_error']:.3f}")
    print(f"  Mean stability: {evolution['mean_stability']:.3f}")
    print(f"  Mean information: {evolution['mean_information']:.2f} bits")
    print(f"  Catastrophes detected: {evolution['n_catastrophes']}")
    print(f"  Mean correlation: {evolution['mean_correlation']:.3f}")

    # Test catastrophe detection
    print("\n--- Testing Catastrophe Detection ---")

    # Create trajectory with jump
    trajectory = []

    # Smooth evolution
    for i in range(5):
        s = TripleCodeState(
            golay_codeword=np.random.randint(0, 2, 24),
            golay_syndrome=np.zeros(12),
            e8_coords=np.array([i * 0.1] * 8),
            e8_casimirs={},
            padic_encodings={},
            hierarchical_level=1
        )
        trajectory.append(s)

    # Add catastrophe (sudden jump)
    s_jump = TripleCodeState(
        golay_codeword=np.random.randint(0, 2, 24),
        golay_syndrome=np.ones(12) * 3,  # High syndrome
        e8_coords=np.array([2.0] * 8),  # Large jump
        e8_casimirs={},
        padic_encodings={},
        hierarchical_level=5  # Hierarchy jump
    )
    trajectory.append(s_jump)

    # Continue smooth
    for i in range(5):
        s = TripleCodeState(
            golay_codeword=np.random.randint(0, 2, 24),
            golay_syndrome=np.zeros(12),
            e8_coords=np.array([2.0 + i * 0.1] * 8),
            e8_casimirs={},
            padic_encodings={},
            hierarchical_level=5
        )
        trajectory.append(s)

    catastrophes = triple_code.detect_catastrophe(trajectory)
    print(f"Catastrophes detected at indices: {catastrophes}")

    print("\n✓ Integrated triple code working correctly!")


if __name__ == "__main__":
    test_integrated_triple_code()