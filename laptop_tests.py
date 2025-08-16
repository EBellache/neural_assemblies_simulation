"""
Lightweight Statistical Tests for Laptop Execution
===================================================

Optimized statistical validation suite for running on laptop
with HC-5 dataset.
"""

import numpy as np
import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import gc
import warnings

from data.loaders.buzsaki_hc5_loader import HC5LaptopLoader, LightweightSession, ProcessedChunk


@dataclass
class LaptopTestResult:
    """Lightweight test result."""
    test_name: str
    passed: bool
    score: float
    computation_time: float
    memory_used_mb: float
    details: Dict


class LaptopTestSuite:
    """
    Optimized test suite for laptop execution with HC-5 data.
    """

    def __init__(self,
                 quick_mode: bool = True,
                 max_test_time: float = 30.0,  # Max time per test in seconds
                 plot_results: bool = False):  # Disable plots to save memory
        """
        Initialize laptop test suite.

        Args:
            quick_mode: Run faster, less comprehensive tests
            max_test_time: Maximum time for each test
            plot_results: Whether to generate plots
        """
        self.quick_mode = quick_mode
        self.max_test_time = max_test_time
        self.plot_results = plot_results
        self.alpha = 0.05

        # Track resource usage
        self.start_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def run_essential_tests(self, session: LightweightSession) -> Dict[str, LaptopTestResult]:
        """
        Run only essential tests optimized for laptop.

        Args:
            session: Lightweight session data

        Returns:
            Test results
        """
        results = {}

        print("\n" + "=" * 60)
        print("RUNNING ESSENTIAL COORDINATE-FREE TESTS (LAPTOP MODE)")
        print("=" * 60)

        # Test 1: Assembly Structure (Fast)
        print("\n[1/5] Testing assembly structure...")
        results['assembly'] = self._test_assembly_structure(session)

        # Test 2: Hierarchical Organization (Fast)
        print("\n[2/5] Testing hierarchical organization...")
        results['hierarchy'] = self._test_hierarchical_structure(session)

        # Test 3: Replay Detection (Simplified)
        print("\n[3/5] Testing replay detection...")
        results['replay'] = self._test_replay_simplified(session)

        # Test 4: Modular Geometry (Lightweight)
        print("\n[4/5] Testing modular geometry...")
        results['modular'] = self._test_modular_geometry_light(session)

        # Test 5: Conservation Laws (Sampling)
        print("\n[5/5] Testing conservation laws...")
        results['conservation'] = self._test_conservation_sampling(session)

        # Summary
        self._print_summary(results)

        return results

    def run_streaming_tests(self, loader: HC5LaptopLoader,
                            session: LightweightSession) -> Dict[str, LaptopTestResult]:
        """
        Run tests using streaming/chunked processing.

        Args:
            loader: HC-5 loader
            session: Session data

        Returns:
            Test results from streaming analysis
        """
        results = {}

        print("\n" + "=" * 60)
        print("RUNNING STREAMING ANALYSIS TESTS")
        print("=" * 60)

        # Process in chunks to avoid memory issues
        chunk_results = []

        for i, chunk in enumerate(loader.process_chunks(session)):
            if i >= 10:  # Limit chunks for laptop
                break

            print(f"\rProcessing chunk {i + 1}/10...", end='')

            # Test on each chunk
            chunk_test = self._test_chunk(chunk)
            chunk_results.append(chunk_test)

            # Clear memory
            gc.collect()

        print("\n")

        # Aggregate results
        results['streaming'] = self._aggregate_chunk_results(chunk_results)

        return results

    def _test_assembly_structure(self, session: LightweightSession) -> LaptopTestResult:
        """Fast assembly structure test."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Create simplified spike matrix
        n_units = len(session.spike_trains)
        if n_units < 3:
            return LaptopTestResult(
                test_name="Assembly Structure",
                passed=False,
                score=0.0,
                computation_time=0.0,
                memory_used_mb=0.0,
                details={'error': 'Too few units'}
            )

        # Sample time windows for efficiency
        window_size = 0.1  # 100ms windows
        n_windows = min(1000, int(session.metadata['duration_seconds'] / window_size))

        # Build coactivation matrix
        coactive_matrix = np.zeros((n_units, n_units), dtype=np.float32)

        for i in range(n_windows):
            if time.time() - start_time > self.max_test_time:
                break

            t_start = i * window_size
            t_end = (i + 1) * window_size

            # Check which units fired in window
            active = np.zeros(n_units, dtype=bool)
            for unit_id, spikes in session.spike_trains.items():
                if unit_id < n_units:
                    active[unit_id] = np.any((spikes >= t_start) & (spikes < t_end))

            # Update coactivation
            coactive_matrix += np.outer(active, active)

        # Normalize
        if n_windows > 0:
            coactive_matrix /= n_windows

        # Test for assembly structure: eigenvalue distribution
        eigenvals = np.linalg.eigvalsh(coactive_matrix)
        eigenvals = eigenvals[eigenvals > 0]

        if len(eigenvals) > 1:
            # Test for heavy-tailed distribution (indicates assemblies)
            # Simplified: check ratio of largest to mean eigenvalue
            ratio = eigenvals[-1] / np.mean(eigenvals)
            score = min(1.0, ratio / 10.0)  # Normalize score

            # Statistical test: is ratio significantly large?
            # Bootstrap null distribution
            n_bootstrap = 100 if self.quick_mode else 500
            null_ratios = []

            for _ in range(n_bootstrap):
                random_matrix = np.random.rand(n_units, n_units).astype(np.float32)
                random_matrix = (random_matrix + random_matrix.T) / 2
                random_eigs = np.linalg.eigvalsh(random_matrix)
                random_eigs = random_eigs[random_eigs > 0]
                if len(random_eigs) > 1:
                    null_ratios.append(random_eigs[-1] / np.mean(random_eigs))

            if null_ratios:
                p_value = np.mean([r >= ratio for r in null_ratios])
                passed = p_value < self.alpha and score > 0.5
            else:
                p_value = 1.0
                passed = False
        else:
            score = 0.0
            p_value = 1.0
            passed = False

        # Resource tracking
        computation_time = time.time() - start_time
        memory_used = self._get_memory_usage() - start_memory

        return LaptopTestResult(
            test_name="Assembly Structure",
            passed=passed,
            score=score,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            details={
                'eigenvalue_ratio': ratio if len(eigenvals) > 1 else 0,
                'p_value': p_value,
                'n_units': n_units,
                'n_windows_analyzed': min(n_windows, int((time.time() - start_time) / window_size))
            }
        )

    def _test_hierarchical_structure(self, session: LightweightSession) -> LaptopTestResult:
        """Fast hierarchical structure test."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Use spike rate profiles as proxy for hierarchy
        n_units = len(session.spike_trains)

        if n_units < 5:
            return LaptopTestResult(
                test_name="Hierarchical Structure",
                passed=False,
                score=0.0,
                computation_time=0.0,
                memory_used_mb=0.0,
                details={'error': 'Too few units'}
            )

        # Compute firing rates at different time scales
        scales = [0.01, 0.1, 1.0]  # 10ms, 100ms, 1s
        rate_profiles = []

        for scale in scales:
            rates = []
            for unit_id, spikes in session.spike_trains.items():
                if unit_id < n_units:
                    rate = len(spikes) / session.metadata['duration_seconds']
                    rates.append(rate)
            rate_profiles.append(rates)

        rate_profiles = np.array(rate_profiles, dtype=np.float32)

        # Test for hierarchical organization
        # Check if units cluster at different scales
        from scipy.spatial.distance import pdist

        # Compute distances at each scale
        scale_distances = []
        for profile in rate_profiles:
            if len(profile) > 1:
                dists = pdist(profile.reshape(-1, 1))
                scale_distances.append(dists)

        if len(scale_distances) > 1:
            # Test for scale-dependent clustering
            # Simplified: check if distance distributions differ across scales
            stat, p_value = stats.kruskal(*scale_distances)

            # Score based on separation between scales
            score = 1.0 / (1.0 + p_value)
            passed = p_value < self.alpha
        else:
            score = 0.0
            p_value = 1.0
            passed = False

        computation_time = time.time() - start_time
        memory_used = self._get_memory_usage() - start_memory

        return LaptopTestResult(
            test_name="Hierarchical Structure",
            passed=passed,
            score=score,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            details={
                'n_scales': len(scales),
                'p_value': p_value,
                'n_units': n_units
            }
        )

    def _test_replay_simplified(self, session: LightweightSession) -> LaptopTestResult:
        """Simplified replay detection test."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Focus on ripple periods only
        n_ripples = len(session.ripple_windows)

        if n_ripples == 0:
            return LaptopTestResult(
                test_name="Replay Detection",
                passed=False,
                score=0.0,
                computation_time=0.0,
                memory_used_mb=0.0,
                details={'error': 'No ripples detected'}
            )

        # Sample ripples for testing
        test_ripples = min(20, n_ripples) if self.quick_mode else min(50, n_ripples)

        replay_scores = []

        for i in range(test_ripples):
            if time.time() - start_time > self.max_test_time:
                break

            ripple_start, ripple_end = session.ripple_windows[i]

            # Count units active during ripple
            active_units = []
            for unit_id, spikes in session.spike_trains.items():
                if np.any((spikes >= ripple_start) & (spikes <= ripple_end)):
                    active_units.append(unit_id)

            # Simple replay score: fraction of units participating
            if len(session.spike_trains) > 0:
                participation = len(active_units) / len(session.spike_trains)
                replay_scores.append(participation)

        if replay_scores:
            mean_score = np.mean(replay_scores)

            # Test against null hypothesis of random participation
            # Binomial test
            expected_rate = 0.1  # Expected 10% participation by chance
            n_high_participation = sum(s > 0.3 for s in replay_scores)

            # Use binomtest for newer scipy versions, fallback to binom_test
            try:
                p_value = stats.binomtest(n_high_participation, len(replay_scores),
                                        expected_rate, alternative='greater').pvalue
            except AttributeError:
                # Fallback for older scipy
                from scipy.stats import binom
                p_value = 1 - binom.cdf(n_high_participation - 1, len(replay_scores), expected_rate)

            score = mean_score
            passed = p_value < self.alpha and mean_score > 0.2
        else:
            score = 0.0
            p_value = 1.0
            passed = False

        computation_time = time.time() - start_time
        memory_used = self._get_memory_usage() - start_memory

        return LaptopTestResult(
            test_name="Replay Detection",
            passed=passed,
            score=score,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            details={
                'mean_participation': mean_score if replay_scores else 0,
                'p_value': p_value,
                'n_ripples_tested': len(replay_scores)
            }
        )

    def _test_modular_geometry_light(self, session: LightweightSession) -> LaptopTestResult:
        """Lightweight modular geometry test."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Use simplified representation
        # Sample spike trains to create phase representation
        n_samples = min(100, len(session.spike_trains))

        if n_samples < 10:
            return LaptopTestResult(
                test_name="Modular Geometry",
                passed=False,
                score=0.0,
                computation_time=0.0,
                memory_used_mb=0.0,
                details={'error': 'Insufficient data'}
            )

        # Compute phase representation
        phases = []

        for unit_id in list(session.spike_trains.keys())[:n_samples]:
            spikes = session.spike_trains[unit_id]
            if len(spikes) > 1:
                # Inter-spike intervals
                isis = np.diff(spikes)
                if len(isis) > 0:
                    # Phase as angle in complex plane
                    mean_isi = np.mean(isis)
                    std_isi = np.std(isis)
                    phase = np.angle(mean_isi + 1j * std_isi)
                    phases.append(phase)

        if len(phases) > 10:
            phases = np.array(phases, dtype=np.float32)

            # Test for modular structure
            # Check if phases cluster at specific values (mod 2π)
            # Simplified Rayleigh test for uniformity
            mean_vector = np.mean(np.exp(1j * phases))
            R = np.abs(mean_vector)
            n = len(phases)

            # Rayleigh statistic
            z = n * R ** 2

            # P-value (approximate)
            p_value = np.exp(-z)

            # Score based on concentration
            score = R  # Ranges from 0 to 1
            passed = p_value < self.alpha and score > 0.3
        else:
            score = 0.0
            p_value = 1.0
            passed = False

        computation_time = time.time() - start_time
        memory_used = self._get_memory_usage() - start_memory

        return LaptopTestResult(
            test_name="Modular Geometry",
            passed=passed,
            score=score,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            details={
                'rayleigh_R': score,
                'p_value': p_value,
                'n_phases': len(phases) if 'phases' in locals() else 0
            }
        )

    def _test_conservation_sampling(self, session: LightweightSession) -> LaptopTestResult:
        """Test conservation laws using sampling."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Sample time points
        n_samples = 50 if self.quick_mode else 200
        duration = session.metadata['duration_seconds']
        sample_times = np.linspace(0, duration, n_samples)

        # Compute simple "Casimir" invariants at each time
        invariants = []

        for t in sample_times:
            if time.time() - start_time > self.max_test_time:
                break

            # Count active units in window
            window = 0.1
            active_count = 0
            total_spikes = 0

            for unit_id, spikes in session.spike_trains.items():
                mask = (spikes >= t) & (spikes < t + window)
                if np.any(mask):
                    active_count += 1
                    total_spikes += np.sum(mask)

            # Simple invariants
            inv = {
                'activity': active_count,
                'total_spikes': total_spikes,
                'density': total_spikes / (active_count + 1)
            }
            invariants.append(inv)

        if len(invariants) > 10:
            # Test conservation
            # Compute coefficient of variation for each invariant
            cvs = []

            for key in ['activity', 'total_spikes', 'density']:
                values = [inv[key] for inv in invariants]
                if np.mean(values) > 0:
                    cv = np.std(values) / np.mean(values)
                    cvs.append(cv)

            if cvs:
                mean_cv = np.mean(cvs)

                # Test: is CV small (indicating conservation)?
                # Compare with shuffled data
                shuffled_cvs = []
                for _ in range(20):  # Reduced for speed
                    shuffled = np.random.permutation(values)
                    if np.mean(shuffled) > 0:
                        shuffled_cv = np.std(shuffled) / np.mean(shuffled)
                        shuffled_cvs.append(shuffled_cv)

                if shuffled_cvs:
                    # One-sided test
                    p_value = np.mean([cv <= mean_cv for cv in shuffled_cvs])
                    score = 1.0 / (1.0 + mean_cv)
                    passed = mean_cv < 0.5  # Relaxed threshold
                else:
                    p_value = 1.0
                    score = 0.0
                    passed = False
            else:
                score = 0.0
                p_value = 1.0
                passed = False
        else:
            score = 0.0
            p_value = 1.0
            passed = False

        computation_time = time.time() - start_time
        memory_used = self._get_memory_usage() - start_memory

        return LaptopTestResult(
            test_name="Conservation Laws",
            passed=passed,
            score=score,
            computation_time=computation_time,
            memory_used_mb=memory_used,
            details={
                'mean_cv': mean_cv if 'mean_cv' in locals() else 0,
                'p_value': p_value if 'p_value' in locals() else 1,
                'n_samples': len(invariants)
            }
        )

    def _test_chunk(self, chunk: ProcessedChunk) -> Dict:
        """Test single chunk."""
        results = {}

        # Quick tests on chunk data

        # Test 1: Neural state dimension
        results['state_dim'] = chunk.neural_state.shape[1] if chunk.neural_state.ndim > 1 else 1

        # Test 2: Assembly activity
        results['assembly_active'] = np.mean(chunk.assembly_indicators) > 0.1

        # Test 3: Modular coordinates bounded
        results['modular_bounded'] = np.all(np.abs(chunk.modular_coords) < 10)

        # Test 4: Cohomology non-trivial
        results['h1_nontrivial'] = chunk.cohomology_features.get(1, 0) > 0

        return results

    def _aggregate_chunk_results(self, chunk_results: List[Dict]) -> LaptopTestResult:
        """Aggregate results from chunks."""
        if not chunk_results:
            return LaptopTestResult(
                test_name="Streaming Analysis",
                passed=False,
                score=0.0,
                computation_time=0.0,
                memory_used_mb=0.0,
                details={'error': 'No chunks processed'}
            )

        # Aggregate scores
        n_chunks = len(chunk_results)
        state_dims = [r['state_dim'] for r in chunk_results]
        assembly_active = sum(r['assembly_active'] for r in chunk_results)
        modular_bounded = sum(r['modular_bounded'] for r in chunk_results)
        h1_nontrivial = sum(r['h1_nontrivial'] for r in chunk_results)

        # Overall score
        score = (assembly_active + modular_bounded + h1_nontrivial) / (3 * n_chunks)
        passed = score > 0.5

        return LaptopTestResult(
            test_name="Streaming Analysis",
            passed=passed,
            score=score,
            computation_time=0.0,
            memory_used_mb=0.0,
            details={
                'n_chunks': n_chunks,
                'mean_state_dim': np.mean(state_dims),
                'assembly_rate': assembly_active / n_chunks,
                'modular_rate': modular_bounded / n_chunks,
                'h1_rate': h1_nontrivial / n_chunks
            }
        )

    def _print_summary(self, results: Dict[str, LaptopTestResult]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        n_passed = sum(1 for r in results.values() if r.passed)
        n_total = len(results)

        print(f"\nPassed: {n_passed}/{n_total} tests")
        print(f"Success rate: {100 * n_passed / n_total:.1f}%")

        total_time = sum(r.computation_time for r in results.values())
        total_memory = sum(r.memory_used_mb for r in results.values())

        print(f"\nTotal computation time: {total_time:.1f} seconds")
        print(f"Total memory used: {total_memory:.1f} MB")

        print("\nDetailed Results:")
        print("-" * 60)

        for name, result in results.items():
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.test_name}:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Time: {result.computation_time:.2f}s")
            print(f"  Memory: {result.memory_used_mb:.1f}MB")

            # Key details
            for key, value in list(result.details.items())[:3]:
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "=" * 60)


def run_hc5_laptop_analysis(data_path: str,
                            session_name: str = None,
                            quick_mode: bool = True):
    """
    Main function to run HC-5 analysis on laptop.

    Args:
        data_path: Path to HC-5 dataset
        session_name: Specific session to analyze (optional)
        quick_mode: Run faster tests

    Returns:
        Test results
    """
    print("\n" + "=" * 60)
    print("HC-5 COORDINATE-FREE ANALYSIS (LAPTOP MODE)")
    print("=" * 60)

    # Initialize loader with laptop settings
    loader = HC5LaptopLoader(
        data_path=data_path,
        max_units=50,  # Limit units
        max_time_seconds=300,  # 5 minutes max
        use_lightweight=True
    )

    # Find or load session
    if session_name is None:
        # Try to find a session
        data_dir = Path(data_path)
        sessions = [d.name for d in data_dir.iterdir() if d.is_dir()]

        if sessions:
            session_name = sessions[0]
            print(f"Found session: {session_name}")
        else:
            session_name = "test_session"
            print("No sessions found, using synthetic data")

    # Load session
    print(f"\nLoading session: {session_name}")
    session = loader.load_session_lightweight(session_name)

    print(f"Loaded {len(session.spike_trains)} units")
    print(f"Duration: {session.metadata['duration_seconds']:.1f} seconds")
    print(f"Ripples detected: {len(session.ripple_windows)}")

    # Run tests
    test_suite = LaptopTestSuite(quick_mode=quick_mode)

    # Essential tests
    print("\nRunning essential tests...")
    essential_results = test_suite.run_essential_tests(session)

    # Streaming tests (optional)
    if not quick_mode:
        print("\nRunning streaming analysis...")
        streaming_results = test_suite.run_streaming_tests(loader, session)
        essential_results.update(streaming_results)

    return essential_results


# Example usage
if __name__ == "__main__":
    # Example path - adjust to your HC-5 location
    data_path = "/path/to/HC-5/dataset"

    # Run analysis
    results = run_hc5_laptop_analysis(
        data_path=data_path,
        quick_mode=True  # Fast mode for laptop
    )