"""
Cross-Region Coordinate-Free Tests
===================================

Tests for hippocampal-cortical coordination using the coordinate-free framework
with Neuropixels multi-region recordings.
"""

import numpy as np
import jax.numpy as jnp
from scipy import stats, signal
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from data.loaders.neuropixels_loader import (
    NeuropixelsLoader, MultiRegionRecording, CrossRegionAnalysis
)
from core.sheaf.neural_sheaf import NeuralSheafComputer
from core.sheaf.cohomology import CohomologyComputer
from analysis.buzsaki.assembly_extraction import AssemblyComputer


@dataclass
class CrossRegionTestResult:
    """Cross-region test result."""
    test_name: str
    region_pair: Tuple[str, str]
    passed: bool
    statistic: float
    p_value: float
    details: Dict
    figure: Optional[plt.Figure] = None


class CrossRegionCoordinateFreeTests:
    """
    Test suite for cross-region coordination in coordinate-free framework.
    """

    def __init__(self, alpha: float = 0.05, laptop_mode: bool = True):
        """
        Initialize cross-region test suite.

        Args:
            alpha: Significance level
            laptop_mode: Use computational optimizations
        """
        self.alpha = alpha
        self.laptop_mode = laptop_mode

        # Initialize coordinate-free computers
        self.sheaf_computer = NeuralSheafComputer(modular_level=12, dimension=8)
        self.cohomology_computer = CohomologyComputer(max_degree=2)
        self.assembly_computer = AssemblyComputer(prime=3, modular_level=12)

    def run_cross_region_tests(self,
                               recording: MultiRegionRecording) -> Dict[str, CrossRegionTestResult]:
        """
        Run complete cross-region test suite.

        Args:
            recording: Multi-region Neuropixels recording

        Returns:
            Test results
        """
        results = {}

        print("\n" + "=" * 60)
        print("CROSS-REGION COORDINATE-FREE ANALYSIS")
        print(f"Session: {recording.session_id}")
        print(f"Regions: {list(recording.brain_regions.keys())}")
        print(f"Region pairs: {len(recording.cross_region_pairs)}")
        print("=" * 60)

        # Test 1: Shared sheaf structure across regions
        print("\n[Test 1] Shared sheaf structure...")
        results['shared_sheaf'] = self.test_shared_sheaf_structure(recording)

        # Test 2: Cross-region cohomology
        print("\n[Test 2] Cross-region cohomology...")
        results['cross_cohomology'] = self.test_cross_region_cohomology(recording)

        # Test 3: Hippocampal-cortical assembly coupling
        print("\n[Test 3] HC-cortical assembly coupling...")
        results['hc_coupling'] = self.test_hippocampal_cortical_coupling(recording)

        # Test 4: Information flow via spectral sequence
        print("\n[Test 4] Information flow...")
        results['info_flow'] = self.test_information_flow(recording)

        # Test 5: Phase coordination on modular curve
        print("\n[Test 5] Modular phase coordination...")
        results['phase_coord'] = self.test_modular_phase_coordination(recording)

        # Test 6: Cross-region replay
        print("\n[Test 6] Cross-region replay...")
        results['cross_replay'] = self.test_cross_region_replay(recording)

        # Test 7: Hierarchical nesting
        print("\n[Test 7] Hierarchical nesting...")
        results['hierarchy'] = self.test_hierarchical_nesting(recording)

        # Test 8: Conservation across regions
        print("\n[Test 8] Conservation laws...")
        results['conservation'] = self.test_cross_region_conservation(recording)

        self._print_summary(results)

        return results

    def test_shared_sheaf_structure(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Brain regions share common sheaf structure.

        Prediction: Transition functions between regions satisfy cocycle condition.
        """
        start_time = time.time()

        # Select a hippocampus-cortex pair
        pair = self._select_region_pair(recording, prefer_hc=True)
        if not pair:
            return self._null_result("Shared Sheaf Structure", "No region pairs")

        region1, region2 = pair

        # Get spike data for each region
        spikes1 = self._get_region_spikes(recording, region1, max_units=30)
        spikes2 = self._get_region_spikes(recording, region2, max_units=30)

        if spikes1.size == 0 or spikes2.size == 0:
            return self._null_result("Shared Sheaf Structure", "No spikes")

        # Construct sheaves for each region
        sheaf1 = self.sheaf_computer.construct_sheaf(spikes1)
        sheaf2 = self.sheaf_computer.construct_sheaf(spikes2)

        # Test compatibility of transition functions
        compatibility_scores = []

        for (i, j) in sheaf1.transition_functions:
            if (i, j) in sheaf2.transition_functions:
                trans1 = sheaf1.transition_functions[(i, j)]
                trans2 = sheaf2.transition_functions[(i, j)]

                # Check if transitions are compatible (commute)
                if trans1.shape == trans2.shape:
                    commutator = trans1 @ trans2 - trans2 @ trans1
                    score = 1.0 / (1.0 + jnp.linalg.norm(commutator))
                    compatibility_scores.append(float(score))

        if compatibility_scores:
            mean_compatibility = np.mean(compatibility_scores)

            # Statistical test: are scores significantly high?
            # One-sample t-test against chance (0.5)
            t_stat, p_value = stats.ttest_1samp(compatibility_scores, 0.5,
                                                alternative='greater')

            passed = p_value < self.alpha and mean_compatibility > 0.7
        else:
            mean_compatibility = 0.0
            p_value = 1.0
            passed = False
            t_stat = 0.0

        # Create figure
        fig = None
        if not self.laptop_mode:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Plot compatibility scores
            if compatibility_scores:
                ax1.hist(compatibility_scores, bins=15, alpha=0.7, edgecolor='black')
                ax1.axvline(0.5, color='red', linestyle='--', label='Chance')
                ax1.axvline(mean_compatibility, color='green', label='Mean')
                ax1.set_xlabel('Compatibility Score')
                ax1.set_ylabel('Count')
                ax1.set_title(f'{region1}-{region2} Sheaf Compatibility')
                ax1.legend()

            # Plot sheaf structure
            ax2.set_title('Sheaf Structure')
            ax2.text(0.5, 0.5, f'H⁰: {sheaf1.cohomology_computed}\n' +
                     f'Transitions: {len(sheaf1.transition_functions)}',
                     ha='center', va='center')
            ax2.axis('off')

            plt.tight_layout()

        return CrossRegionTestResult(
            test_name="Shared Sheaf Structure",
            region_pair=pair,
            passed=passed,
            statistic=t_stat,
            p_value=p_value,
            details={
                'mean_compatibility': mean_compatibility,
                'n_compatible_transitions': len(compatibility_scores),
                'computation_time': time.time() - start_time
            },
            figure=fig
        )

    def test_cross_region_cohomology(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Cross-region activity generates non-trivial cohomology.

        Prediction: H^1 captures information flow between regions.
        """
        pair = self._select_region_pair(recording, prefer_hc=True)
        if not pair:
            return self._null_result("Cross-Region Cohomology", "No region pairs")

        region1, region2 = pair

        # Combine spike data from both regions
        spikes1 = self._get_region_spikes(recording, region1, max_units=20)
        spikes2 = self._get_region_spikes(recording, region2, max_units=20)

        # Stack to create cross-region representation
        if spikes1.shape[0] > 0 and spikes2.shape[0] > 0:
            combined = jnp.vstack([spikes1[:20], spikes2[:20]])
        else:
            return self._null_result("Cross-Region Cohomology", "Insufficient data")

        # Compute sheaf and cohomology
        sheaf = self.sheaf_computer.construct_sheaf(combined)

        # Build cochains
        cochains = {}
        if sheaf.local_sections:
            cochains[0] = jnp.stack([s.data for s in sheaf.local_sections.values()])

        if sheaf.transition_functions:
            trans_list = []
            for trans in sheaf.transition_functions.values():
                trans_list.append(trans.flatten())

            if trans_list:
                max_len = max(len(t) for t in trans_list)
                cochains[1] = jnp.zeros((len(trans_list), max_len))
                for i, t in enumerate(trans_list):
                    cochains[1] = cochains[1].at[i, :len(t)].set(t)

        # Compute H^1
        h1_class = self.cohomology_computer._compute_cohomology_group_impl(cochains, 1)

        # Test significance
        h1_dim = h1_class.dimension

        # Compare with single-region cohomology
        sheaf1 = self.sheaf_computer.construct_sheaf(spikes1)
        cochains1 = {}
        if sheaf1.local_sections:
            cochains1[0] = jnp.stack([s.data for s in sheaf1.local_sections.values()])
        h1_single = self.cohomology_computer._compute_cohomology_group_impl(cochains1, 1)

        # Statistical test: is cross-region H^1 larger than single-region?
        if h1_dim > 0 and h1_single.dimension >= 0:
            # Chi-square test for difference
            chi2_stat = (h1_dim - h1_single.dimension) ** 2 / (h1_single.dimension + 1)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

            passed = p_value < self.alpha and h1_dim > h1_single.dimension
        else:
            chi2_stat = 0.0
            p_value = 1.0
            passed = False

        return CrossRegionTestResult(
            test_name="Cross-Region Cohomology",
            region_pair=pair,
            passed=passed,
            statistic=chi2_stat,
            p_value=p_value,
            details={
                'cross_region_h1_dim': h1_dim,
                'single_region_h1_dim': h1_single.dimension,
                'enhancement_ratio': h1_dim / (h1_single.dimension + 1)
            }
        )

    def test_hippocampal_cortical_coupling(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Hippocampal assemblies couple with cortical assemblies.

        Prediction: Assembly eigenforms show phase locking across regions.
        """
        # Find HC-cortical pair
        hc_pair = None
        for pair in recording.cross_region_pairs:
            if ('CA1' in pair[0] or 'CA3' in pair[0] or 'DG' in pair[0]) and \
                    ('VIS' in pair[1] or 'RSP' in pair[1] or 'PPC' in pair[1]):
                hc_pair = pair
                break

        if not hc_pair:
            # Try reverse
            for pair in recording.cross_region_pairs:
                if ('CA1' in pair[1] or 'CA3' in pair[1] or 'DG' in pair[1]) and \
                        ('VIS' in pair[0] or 'RSP' in pair[0] or 'PPC' in pair[0]):
                    hc_pair = (pair[1], pair[0])
                    break

        if not hc_pair:
            return self._null_result("HC-Cortical Coupling", "No HC-cortical pairs")

        hippo_region, cortex_region = hc_pair

        # Get assemblies from each region
        hippo_spikes = self._get_region_spikes(recording, hippo_region)
        cortex_spikes = self._get_region_spikes(recording, cortex_region)

        if hippo_spikes.size == 0 or cortex_spikes.size == 0:
            return self._null_result("HC-Cortical Coupling", "No spikes")

        # Extract assemblies
        hippo_assemblies = self.assembly_computer.extract_assemblies(hippo_spikes)
        cortex_assemblies = self.assembly_computer.extract_assemblies(cortex_spikes)

        if not hippo_assemblies or not cortex_assemblies:
            return self._null_result("HC-Cortical Coupling", "No assemblies found")

        # Test coupling between assemblies
        coupling_scores = []

        for h_assembly in hippo_assemblies[:5]:  # Limit for speed
            for c_assembly in cortex_assemblies[:5]:
                # Compare eigenforms
                h_eigen = h_assembly.eigenform
                c_eigen = c_assembly.eigenform

                if len(h_eigen) > 0 and len(c_eigen) > 0:
                    # Compute phase coupling
                    min_len = min(len(h_eigen), len(c_eigen))
                    h_phase = jnp.angle(h_eigen[:min_len])
                    c_phase = jnp.angle(c_eigen[:min_len])

                    # Phase locking value
                    phase_diff = h_phase - c_phase
                    plv = jnp.abs(jnp.mean(jnp.exp(1j * phase_diff)))
                    coupling_scores.append(float(plv))

        if coupling_scores:
            mean_coupling = np.mean(coupling_scores)

            # Statistical test: Rayleigh test for phase locking
            # Under null hypothesis, phases are random
            n = len(coupling_scores)
            R = mean_coupling * n
            z_stat = n * mean_coupling ** 2
            p_value = np.exp(-z_stat)  # Approximate p-value

            passed = p_value < self.alpha and mean_coupling > 0.3
        else:
            mean_coupling = 0.0
            p_value = 1.0
            z_stat = 0.0
            passed = False

        return CrossRegionTestResult(
            test_name="HC-Cortical Coupling",
            region_pair=hc_pair,
            passed=passed,
            statistic=z_stat,
            p_value=p_value,
            details={
                'mean_phase_locking': mean_coupling,
                'n_hippo_assemblies': len(hippo_assemblies),
                'n_cortex_assemblies': len(cortex_assemblies),
                'n_pairs_tested': len(coupling_scores)
            }
        )

    def test_information_flow(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Information flows according to spectral sequence.

        Prediction: Directional flow from hippocampus to cortex during replay.
        """
        pair = self._select_region_pair(recording, prefer_hc=True)
        if not pair:
            return self._null_result("Information Flow", "No region pairs")

        region1, region2 = pair

        # Compute transfer entropy proxy (simplified)
        spikes1 = self._get_region_spikes(recording, region1)
        spikes2 = self._get_region_spikes(recording, region2)

        if spikes1.size == 0 or spikes2.size == 0:
            return self._null_result("Information Flow", "No spikes")

        # Create rate vectors
        bin_size = 0.05  # 50ms bins
        max_time = min(300.0, recording.metadata.get('duration', 300))
        n_bins = int(max_time / bin_size)

        rate1 = self._compute_rate_vector(spikes1, n_bins, bin_size)
        rate2 = self._compute_rate_vector(spikes2, n_bins, bin_size)

        # Compute directional coupling (simplified transfer entropy)
        flow_1to2 = self._compute_directed_info(rate1, rate2)
        flow_2to1 = self._compute_directed_info(rate2, rate1)

        # Net flow
        net_flow = flow_1to2 - flow_2to1

        # Statistical test: is flow significantly directional?
        # Bootstrap test
        n_bootstrap = 100 if self.laptop_mode else 500
        bootstrap_flows = []

        for _ in range(n_bootstrap):
            # Shuffle time
            shuffled1 = np.random.permutation(rate1)
            shuffled2 = np.random.permutation(rate2)

            boot_1to2 = self._compute_directed_info(shuffled1, shuffled2)
            boot_2to1 = self._compute_directed_info(shuffled2, shuffled1)
            bootstrap_flows.append(boot_1to2 - boot_2to1)

        # P-value for directional flow
        if net_flow > 0:
            p_value = np.mean([f >= net_flow for f in bootstrap_flows])
        else:
            p_value = np.mean([f <= net_flow for f in bootstrap_flows])

        passed = p_value < self.alpha and abs(net_flow) > 0.01

        return CrossRegionTestResult(
            test_name="Information Flow",
            region_pair=pair,
            passed=passed,
            statistic=net_flow,
            p_value=p_value,
            details={
                'flow_1to2': flow_1to2,
                'flow_2to1': flow_2to1,
                'net_flow': net_flow,
                'direction': 'forward' if net_flow > 0 else 'reverse'
            }
        )

    def test_modular_phase_coordination(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Cross-region coordination on modular curve.

        Prediction: Regions maintain fixed phase relationships on X_0(N).
        """
        pair = self._select_region_pair(recording, prefer_hc=True)
        if not pair:
            return self._null_result("Modular Phase Coordination", "No region pairs")

        region1, region2 = pair

        # Get spike data
        spikes1 = self._get_region_spikes(recording, region1, max_units=20)
        spikes2 = self._get_region_spikes(recording, region2, max_units=20)

        if spikes1.size < 10 or spikes2.size < 10:
            return self._null_result("Modular Phase Coordination", "Insufficient data")

        # Map to modular coordinates
        sheaf1 = self.sheaf_computer.construct_sheaf(spikes1)
        sheaf2 = self.sheaf_computer.construct_sheaf(spikes2)

        # Extract modular phases
        phases1 = []
        for section in sheaf1.local_sections.values():
            if section.data.size >= 2:
                tau = complex(section.data[0], abs(section.data[1]) + 0.1)
                phases1.append(np.angle(tau))

        phases2 = []
        for section in sheaf2.local_sections.values():
            if section.data.size >= 2:
                tau = complex(section.data[0], abs(section.data[1]) + 0.1)
                phases2.append(np.angle(tau))

        if not phases1 or not phases2:
            return self._null_result("Modular Phase Coordination", "No phases")

        # Test phase relationship stability
        phases1 = np.array(phases1)
        phases2 = np.array(phases2)

        # Compute phase differences
        min_len = min(len(phases1), len(phases2))
        phase_diffs = phases1[:min_len] - phases2[:min_len]

        # Circular variance as measure of coordination
        mean_vector = np.mean(np.exp(1j * phase_diffs))
        R = np.abs(mean_vector)
        circular_var = 1 - R

        # Rayleigh test for non-uniformity
        n = len(phase_diffs)
        z_stat = n * R ** 2
        p_value = np.exp(-z_stat)

        passed = p_value < self.alpha and R > 0.5

        return CrossRegionTestResult(
            test_name="Modular Phase Coordination",
            region_pair=pair,
            passed=passed,
            statistic=z_stat,
            p_value=p_value,
            details={
                'mean_resultant_length': R,
                'circular_variance': circular_var,
                'n_phases': min_len,
                'mean_phase_diff': np.angle(mean_vector)
            }
        )

    def test_cross_region_replay(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Replay events span multiple regions.

        Prediction: H^1 classes are shared across regions during replay.
        """
        pair = self._select_region_pair(recording, prefer_hc=True)
        if not pair:
            return self._null_result("Cross-Region Replay", "No region pairs")

        # Simplified: look for coordinated high-activity periods
        region1, region2 = pair

        spikes1 = self._get_region_spikes(recording, region1)
        spikes2 = self._get_region_spikes(recording, region2)

        if spikes1.size == 0 or spikes2.size == 0:
            return self._null_result("Cross-Region Replay", "No spikes")

        # Find high-activity periods (potential replay)
        window = 0.05  # 50ms
        n_windows = 1000 if self.laptop_mode else 5000

        replay_scores = []

        for i in range(n_windows):
            t = i * window

            # Count spikes in window for each region
            count1 = jnp.sum((spikes1.flatten() >= t) & (spikes1.flatten() < t + window))
            count2 = jnp.sum((spikes2.flatten() >= t) & (spikes2.flatten() < t + window))

            # High activity in both regions suggests replay
            if count1 > 5 and count2 > 5:
                # Compute coordination score
                score = min(count1, count2) / max(count1, count2)
                replay_scores.append(float(score))

        if replay_scores:
            mean_coordination = np.mean(replay_scores)

            # Test against null of independent activity
            # Binomial test
            n_coordinated = sum(s > 0.5 for s in replay_scores)
            n_total = len(replay_scores)

            p_value = stats.binom_test(n_coordinated, n_total, 0.25,
                                       alternative='greater')

            passed = p_value < self.alpha and mean_coordination > 0.5
        else:
            mean_coordination = 0.0
            p_value = 1.0
            passed = False

        return CrossRegionTestResult(
            test_name="Cross-Region Replay",
            region_pair=pair,
            passed=passed,
            statistic=mean_coordination,
            p_value=p_value,
            details={
                'n_replay_events': len(replay_scores),
                'mean_coordination': mean_coordination,
                'percent_coordinated': 100 * sum(s > 0.5 for s in replay_scores) /
                                       (len(replay_scores) + 1)
            }
        )

    def test_hierarchical_nesting(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Cortical assemblies nest within hippocampal sequences.

        Prediction: Cortical eigenforms are linear combinations of hippocampal ones.
        """
        # Find HC-cortical pair
        hc_pair = None
        for pair in recording.cross_region_pairs:
            if 'CA' in pair[0] and 'VIS' in pair[1]:
                hc_pair = pair
                break

        if not hc_pair:
            return self._null_result("Hierarchical Nesting", "No HC-cortical pair")

        hippo_region, cortex_region = hc_pair

        # Get assemblies
        hippo_spikes = self._get_region_spikes(recording, hippo_region)
        cortex_spikes = self._get_region_spikes(recording, cortex_region)

        if hippo_spikes.size == 0 or cortex_spikes.size == 0:
            return self._null_result("Hierarchical Nesting", "No spikes")

        hippo_assemblies = self.assembly_computer.extract_assemblies(hippo_spikes)
        cortex_assemblies = self.assembly_computer.extract_assemblies(cortex_spikes)

        if len(hippo_assemblies) < 2 or len(cortex_assemblies) < 2:
            return self._null_result("Hierarchical Nesting", "Too few assemblies")

        # Test if cortical eigenforms can be expressed as combinations of hippocampal
        nesting_scores = []

        for c_assembly in cortex_assemblies[:5]:
            c_eigen = c_assembly.eigenform

            if len(c_eigen) > 0:
                # Build basis from hippocampal eigenforms
                h_basis = []
                for h_assembly in hippo_assemblies[:5]:
                    if len(h_assembly.eigenform) >= len(c_eigen):
                        h_basis.append(h_assembly.eigenform[:len(c_eigen)])

                if len(h_basis) >= 2:
                    # Test if c_eigen is in span of h_basis
                    h_matrix = jnp.stack(h_basis).T

                    # Compute projection
                    try:
                        coeffs = jnp.linalg.lstsq(h_matrix, c_eigen)[0]
                        reconstruction = h_matrix @ coeffs

                        # Reconstruction error
                        error = jnp.linalg.norm(c_eigen - reconstruction) / \
                                (jnp.linalg.norm(c_eigen) + 1e-10)
                        score = 1.0 / (1.0 + error)
                        nesting_scores.append(float(score))
                    except:
                        pass

        if nesting_scores:
            mean_nesting = np.mean(nesting_scores)

            # Test significance
            t_stat, p_value = stats.ttest_1samp(nesting_scores, 0.5,
                                                alternative='greater')

            passed = p_value < self.alpha and mean_nesting > 0.6
        else:
            mean_nesting = 0.0
            t_stat = 0.0
            p_value = 1.0
            passed = False

        return CrossRegionTestResult(
            test_name="Hierarchical Nesting",
            region_pair=hc_pair,
            passed=passed,
            statistic=t_stat,
            p_value=p_value,
            details={
                'mean_nesting_score': mean_nesting,
                'n_cortical_tested': len(nesting_scores),
                'n_hippo_basis': len(hippo_assemblies)
            }
        )

    def test_cross_region_conservation(self, recording: MultiRegionRecording) -> CrossRegionTestResult:
        """
        Test: Casimir invariants are conserved across regions.

        Prediction: Same conservation laws apply to all regions.
        """
        if len(recording.cross_region_pairs) == 0:
            return self._null_result("Cross-Region Conservation", "No region pairs")

        # Test conservation for multiple pairs
        conservation_scores = []

        for pair in recording.cross_region_pairs[:3]:  # Test first 3 pairs
            region1, region2 = pair

            spikes1 = self._get_region_spikes(recording, region1, max_units=10)
            spikes2 = self._get_region_spikes(recording, region2, max_units=10)

            if spikes1.size > 0 and spikes2.size > 0:
                # Compute simple "invariants"
                inv1 = {
                    'mean_rate': float(jnp.mean(spikes1)),
                    'variance': float(jnp.var(spikes1)),
                    'skewness': float(stats.skew(spikes1.flatten()))
                }

                inv2 = {
                    'mean_rate': float(jnp.mean(spikes2)),
                    'variance': float(jnp.var(spikes2)),
                    'skewness': float(stats.skew(spikes2.flatten()))
                }

                # Compare invariants
                for key in inv1:
                    if inv1[key] > 0 and inv2[key] > 0:
                        ratio = min(inv1[key], inv2[key]) / max(inv1[key], inv2[key])
                        conservation_scores.append(ratio)

        if conservation_scores:
            mean_conservation = np.mean(conservation_scores)

            # Test if conservation is significant
            # One-sample t-test against random (0.5)
            t_stat, p_value = stats.ttest_1samp(conservation_scores, 0.5,
                                                alternative='greater')

            passed = p_value < self.alpha and mean_conservation > 0.7
        else:
            mean_conservation = 0.0
            t_stat = 0.0
            p_value = 1.0
            passed = False

        return CrossRegionTestResult(
            test_name="Cross-Region Conservation",
            region_pair=recording.cross_region_pairs[0] if recording.cross_region_pairs else ('', ''),
            passed=passed,
            statistic=t_stat,
            p_value=p_value,
            details={
                'mean_conservation': mean_conservation,
                'n_invariants_tested': len(conservation_scores),
                'n_pairs_tested': min(3, len(recording.cross_region_pairs))
            }
        )

    # Helper methods

    def _select_region_pair(self, recording: MultiRegionRecording,
                            prefer_hc: bool = True) -> Optional[Tuple[str, str]]:
        """Select a region pair for testing."""
        if not recording.cross_region_pairs:
            return None

        if prefer_hc:
            # Prefer hippocampus-cortex pairs
            for pair in recording.cross_region_pairs:
                if ('CA' in pair[0] or 'DG' in pair[0]) and \
                        ('VIS' in pair[1] or 'RSP' in pair[1] or 'PPC' in pair[1]):
                    return pair

        # Return first available pair
        return recording.cross_region_pairs[0]

    def _get_region_spikes(self, recording: MultiRegionRecording,
                           region: str, max_units: int = 50) -> jnp.ndarray:
        """Get spike matrix for a specific region."""
        region_spikes = []

        for probe in recording.probe_data.values():
            if region in probe.brain_regions:
                # Get units from this region
                unit_count = 0
                for unit_id, spike_train in probe.spike_times.items():
                    if unit_count >= max_units:
                        break
                    region_spikes.append(spike_train)
                    unit_count += 1

        if not region_spikes:
            return jnp.array([])

        # Create spike matrix
        # Simple approach: stack spike trains
        max_len = max(len(s) for s in region_spikes)
        matrix = jnp.zeros((len(region_spikes), min(1000, max_len)))

        for i, spikes in enumerate(region_spikes):
            if len(spikes) > 0:
                matrix = matrix.at[i, :min(len(spikes), 1000)].set(spikes[:1000])

        return matrix

    def _compute_rate_vector(self, spikes: jnp.ndarray,
                             n_bins: int, bin_size: float) -> np.ndarray:
        """Compute firing rate vector."""
        if spikes.ndim == 1:
            spikes = spikes.reshape(1, -1)

        rates = np.zeros(n_bins)

        for unit_spikes in spikes:
            if len(unit_spikes) > 0:
                counts, _ = np.histogram(unit_spikes,
                                         bins=n_bins,
                                         range=(0, n_bins * bin_size))
                rates += counts / bin_size

        return rates / (len(spikes) + 1)

    def _compute_directed_info(self, source: np.ndarray,
                               target: np.ndarray, lag: int = 1) -> float:
        """Compute directed information measure."""
        if len(source) <= lag or len(target) <= lag:
            return 0.0

        # Simple lagged correlation as proxy
        source_lagged = source[:-lag]
        target_future = target[lag:]

        if np.std(source_lagged) > 0 and np.std(target_future) > 0:
            corr = np.corrcoef(source_lagged, target_future)[0, 1]
            return abs(corr)

        return 0.0

    def _null_result(self, test_name: str, reason: str) -> CrossRegionTestResult:
        """Return null result when test cannot be performed."""
        return CrossRegionTestResult(
            test_name=test_name,
            region_pair=('', ''),
            passed=False,
            statistic=0.0,
            p_value=1.0,
            details={'error': reason}
        )

    def _print_summary(self, results: Dict[str, CrossRegionTestResult]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("CROSS-REGION TEST SUMMARY")
        print("=" * 60)

        n_passed = sum(1 for r in results.values() if r.passed)
        n_total = len(results)

        print(f"\nPassed: {n_passed}/{n_total} tests")
        print(f"Success rate: {100 * n_passed / n_total:.1f}%")

        print("\nDetailed Results:")
        print("-" * 60)

        for name, result in results.items():
            status = "✓" if result.passed else "✗"
            regions = f"{result.region_pair[0]}-{result.region_pair[1]}"
            print(f"{status} {result.test_name} ({regions}):")
            print(f"  Statistic: {result.statistic:.4f}")
            print(f"  P-value: {result.p_value:.4f}")

            for key, value in list(result.details.items())[:3]:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def run_neuropixels_analysis(dataset: str = 'allen_visual_coding',
                             session_id: Optional[str] = None,
                             laptop_mode: bool = True):
    """
    Run cross-region analysis on Neuropixels data.

    Args:
        dataset: Which dataset to use
        session_id: Specific session ID (optional)
        laptop_mode: Use computational optimizations

    Returns:
        Test results
    """
    print("\n" + "=" * 60)
    print("NEUROPIXELS CROSS-REGION COORDINATE-FREE ANALYSIS")
    print(f"Dataset: {dataset}")
    print("=" * 60)

    # Initialize loader
    loader = NeuropixelsLoader(
        dataset=dataset,
        max_channels=100 if laptop_mode else 384,
        max_duration=300 if laptop_mode else 1800
    )

    # Load session
    if dataset == 'allen_visual_coding':
        session_id = session_id or 'example_session'
        recording = loader.load_allen_visual_coding_session(session_id)

    elif dataset == 'steinmetz_2019':
        # Example path - adjust to your data location
        session_path = session_id or './steinmetz_data/session1'
        recording = loader.load_steinmetz_session(session_path)

    elif dataset == 'ibl_brainwide':
        eid = session_id or 'example_eid'
        recording = loader.load_ibl_session(eid)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Run cross-region tests
    test_suite = CrossRegionCoordinateFreeTests(laptop_mode=laptop_mode)
    results = test_suite.run_cross_region_tests(recording)

    return results, recording