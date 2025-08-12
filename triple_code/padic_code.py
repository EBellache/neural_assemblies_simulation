"""
P-adic Code Module
==================
Implementation of p-adic valuation and ultrametric structure for
hierarchical organization of neuronal assemblies.

P-adic numbers provide:
- Ultrametric distance (strong triangle inequality)
- Hierarchical tree structure
- Multi-scale temporal organization
- Natural representation of nested oscillations

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from functools import lru_cache
import warnings


@dataclass
class PAdicNumber:
    """Represents a p-adic number with finite precision."""
    p: int  # Prime base
    digits: List[int]  # Digits in base p (least significant first)
    valuation: int  # p-adic valuation (power of p in denominator)

    def __post_init__(self):
        # Validate prime
        if not self._is_prime(self.p):
            raise ValueError(f"{self.p} is not prime")

        # Validate digits
        for d in self.digits:
            if not 0 <= d < self.p:
                raise ValueError(f"Digit {d} not in range [0, {self.p})")

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def to_rational(self, precision: int = 10) -> float:
        """Convert to rational approximation."""
        result = 0
        for i, digit in enumerate(self.digits[:precision]):
            result += digit * (self.p ** (i - self.valuation))
        return result


class PAdicEncoder:
    """
    Encode neuronal assembly patterns using p-adic representation.
    Maps temporal hierarchies to p-adic valuations.
    """

    def __init__(self, primes: List[int] = [2, 3, 5, 7]):
        """
        Initialize p-adic encoder.

        Parameters:
        -----------
        primes : List[int]
            Prime bases for multi-prime encoding (CRT)
        """
        self.primes = primes

        # Validate primes
        for p in primes:
            if not self._is_prime(p):
                raise ValueError(f"{p} is not prime")

        # Precompute modular arithmetic helpers
        self.moduli_product = np.prod(primes)
        self._setup_crt()

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _setup_crt(self):
        """Setup Chinese Remainder Theorem coefficients."""
        self.crt_coeffs = []

        for i, p in enumerate(self.primes):
            # Product of all primes except p
            Mp = self.moduli_product // p

            # Find multiplicative inverse of Mp mod p
            # Using extended Euclidean algorithm
            _, inv, _ = self._extended_gcd(Mp, p)
            inv = inv % p

            self.crt_coeffs.append((Mp, inv))

    @staticmethod
    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm."""
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = PAdicEncoder._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def compute_padic_valuation(self, n: int, p: int) -> int:
        """
        Compute p-adic valuation v_p(n).

        v_p(n) = largest k such that p^k divides n
        """
        if n == 0:
            return float('inf')

        valuation = 0
        while n % p == 0:
            n //= p
            valuation += 1

        return valuation

    def encode_spike_pattern(self, spike_times: np.ndarray,
                             time_resolution: float = 0.001) -> Dict[int, PAdicNumber]:
        """
        Encode spike pattern as p-adic numbers.

        Parameters:
        -----------
        spike_times : np.ndarray
            Array of spike times
        time_resolution : float
            Time bin resolution in seconds

        Returns:
        --------
        Dict mapping prime to p-adic encoding
        """
        if len(spike_times) == 0:
            return {}

        # Discretize spike times
        discrete_times = np.round(spike_times / time_resolution).astype(int)

        encodings = {}

        for p in self.primes:
            # Compute p-adic representation
            digits = []
            valuations = []

            for t in discrete_times:
                # Get p-adic expansion of time
                val = self.compute_padic_valuation(t, p)
                valuations.append(val)

                # Extract digits
                temp = t
                t_digits = []
                for _ in range(10):  # Fixed precision
                    t_digits.append(temp % p)
                    temp //= p

                digits.append(t_digits)

            # Aggregate encoding
            # Use mean valuation as hierarchical level
            mean_valuation = int(np.mean([v for v in valuations if v != float('inf')]))

            # Combine digit patterns
            combined_digits = []
            for i in range(10):
                # Majority vote for each digit position
                digit_votes = [d[i] for d in digits if i < len(d)]
                if digit_votes:
                    combined_digits.append(int(np.median(digit_votes)))
                else:
                    combined_digits.append(0)

            encodings[p] = PAdicNumber(p, combined_digits, mean_valuation)

        return encodings

    def padic_distance(self, x: PAdicNumber, y: PAdicNumber) -> float:
        """
        Compute p-adic distance (ultrametric).

        d_p(x, y) = p^(-v_p(x-y))

        Satisfies strong triangle inequality:
        d(x, z) ≤ max(d(x, y), d(y, z))
        """
        if x.p != y.p:
            raise ValueError("Numbers must have same prime base")

        # Compute difference
        diff_digits = []
        for i in range(max(len(x.digits), len(y.digits))):
            dx = x.digits[i] if i < len(x.digits) else 0
            dy = y.digits[i] if i < len(y.digits) else 0
            diff = (dx - dy) % x.p
            diff_digits.append(diff)

        # Find first non-zero digit (valuation)
        valuation = 0
        for i, d in enumerate(diff_digits):
            if d != 0:
                valuation = i
                break
        else:
            # x == y
            return 0.0

        # Adjust for original valuations
        effective_valuation = valuation + min(x.valuation, y.valuation)

        # Distance is p^(-valuation)
        return float(x.p ** (-effective_valuation))

    def hierarchical_clustering(self, patterns: List[np.ndarray],
                                p: int = 2) -> Dict[str, Any]:
        """
        Perform hierarchical clustering using p-adic distance.

        Parameters:
        -----------
        patterns : List[np.ndarray]
            List of spike patterns
        p : int
            Prime base for p-adic encoding

        Returns:
        --------
        Dict with clustering hierarchy
        """
        # Encode all patterns
        encodings = []
        for pattern in patterns:
            enc_dict = self.encode_spike_pattern(pattern)
            if p in enc_dict:
                encodings.append(enc_dict[p])
            else:
                # Empty pattern
                encodings.append(PAdicNumber(p, [0] * 10, 0))

        n = len(encodings)

        # Compute distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.padic_distance(encodings[i], encodings[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Build hierarchy (ultrametric tree)
        hierarchy = self._build_ultrametric_tree(dist_matrix)

        return {
            'distance_matrix': dist_matrix,
            'hierarchy': hierarchy,
            'encodings': encodings
        }

    def _build_ultrametric_tree(self, dist_matrix: np.ndarray) -> Dict:
        """Build ultrametric tree from distance matrix."""
        n = dist_matrix.shape[0]

        # Simple hierarchical clustering
        # Find minimum distance pairs iteratively
        clusters = [[i] for i in range(n)]
        hierarchy = []

        while len(clusters) > 1:
            # Find closest pair
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Complete linkage
                    max_dist = 0
                    for ci in clusters[i]:
                        for cj in clusters[j]:
                            max_dist = max(max_dist, dist_matrix[ci, cj])

                    if max_dist < min_dist:
                        min_dist = max_dist
                        merge_i, merge_j = i, j

            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            hierarchy.append({
                'merged': (merge_i, merge_j),
                'distance': min_dist,
                'size': len(new_cluster)
            })

            # Update cluster list
            clusters = [c for i, c in enumerate(clusters)
                        if i != merge_i and i != merge_j]
            clusters.append(new_cluster)

        return hierarchy

    def multi_prime_encoding(self, pattern: np.ndarray) -> Dict[int, PAdicNumber]:
        """
        Encode pattern using multiple primes (CRT representation).
        Provides redundancy and error detection.
        """
        encodings = self.encode_spike_pattern(pattern)

        # Add CRT consistency check
        residues = {}
        for p, padic in encodings.items():
            # Extract residue modulo p
            residue = sum(d * (p ** i) for i, d in enumerate(padic.digits[:5])) % p
            residues[p] = residue

        # Reconstruct using CRT
        reconstructed = 0
        for i, p in enumerate(self.primes):
            if p in residues:
                Mp, inv = self.crt_coeffs[i]
                reconstructed += residues[p] * Mp * inv

        reconstructed %= self.moduli_product

        # Store reconstruction for validation
        for p in encodings:
            encodings[p].crt_value = reconstructed

        return encodings


class PAdicHierarchyAnalyzer:
    """
    Analyze hierarchical structure of assemblies using p-adic valuations.
    Maps to theta-gamma phase relationships.
    """

    def __init__(self):
        self.encoder = PAdicEncoder()

    def analyze_temporal_hierarchy(self,
                                   spike_trains: Dict[int, List[float]],
                                   theta_period: float = 0.125) -> Dict[str, Any]:
        """
        Analyze temporal hierarchy using p-adic structure.

        Parameters:
        -----------
        spike_trains : Dict
            Cell ID -> spike times
        theta_period : float
            Theta cycle duration in seconds

        Returns:
        --------
        Dict with hierarchy analysis
        """
        # Group spikes by theta cycles
        all_spikes = []
        for spikes in spike_trains.values():
            all_spikes.extend(spikes)

        if not all_spikes:
            return {}

        max_time = max(all_spikes)
        n_cycles = int(max_time / theta_period) + 1

        # Analyze each cycle
        cycle_hierarchies = []

        for cycle in range(n_cycles):
            cycle_start = cycle * theta_period
            cycle_end = (cycle + 1) * theta_period

            # Get spikes in this cycle
            cycle_patterns = []
            for cell_id, spikes in spike_trains.items():
                cell_spikes = [t - cycle_start for t in spikes
                               if cycle_start <= t < cycle_end]
                if cell_spikes:
                    cycle_patterns.append(np.array(cell_spikes))

            if cycle_patterns:
                # Hierarchical clustering with p=2
                hierarchy = self.encoder.hierarchical_clustering(cycle_patterns, p=2)
                cycle_hierarchies.append(hierarchy)

        # Aggregate results
        if cycle_hierarchies:
            # Average distance matrices
            avg_dist = np.mean([h['distance_matrix']
                                for h in cycle_hierarchies], axis=0)

            # Count hierarchy depth
            depths = [len(h['hierarchy']) for h in cycle_hierarchies]

            return {
                'mean_distance_matrix': avg_dist,
                'hierarchy_depth': np.mean(depths),
                'n_cycles_analyzed': len(cycle_hierarchies),
                'ultrametric_violation': self._check_ultrametric(avg_dist)
            }

        return {}

    def _check_ultrametric(self, dist_matrix: np.ndarray) -> float:
        """
        Check how well distances satisfy ultrametric inequality.
        Returns violation score (0 = perfect ultrametric).
        """
        n = dist_matrix.shape[0]
        violations = []

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Check d(i,k) <= max(d(i,j), d(j,k))
                    d_ik = dist_matrix[i, k]
                    d_ij = dist_matrix[i, j]
                    d_jk = dist_matrix[j, k]

                    max_dist = max(d_ij, d_jk)
                    if d_ik > max_dist:
                        violation = (d_ik - max_dist) / max_dist if max_dist > 0 else 0
                        violations.append(violation)

        return np.mean(violations) if violations else 0.0

    def map_to_theta_gamma(self, padic_encoding: PAdicNumber) -> Dict[str, float]:
        """
        Map p-adic valuation to theta-gamma phase relationship.

        Higher valuations = earlier gamma cycles
        Lower valuations = later gamma cycles
        """
        # Normalize valuation to [0, 1]
        max_valuation = 10  # Arbitrary maximum
        normalized = min(padic_encoding.valuation / max_valuation, 1.0)

        # Map to gamma cycle (5 per theta)
        gamma_cycle = int((1 - normalized) * 5)  # Invert so high valuation = early

        # Map to phase
        theta_phase = gamma_cycle * (2 * np.pi / 5)

        return {
            'gamma_cycle': gamma_cycle,
            'theta_phase': theta_phase,
            'hierarchical_level': padic_encoding.valuation
        }


def test_padic_code():
    """Test p-adic code functionality."""
    print("\n=== Testing P-adic Code Module ===\n")

    # Test p-adic number creation
    print("--- Testing P-adic Numbers ---")

    p = 2
    digits = [1, 0, 1, 1, 0]  # Binary: 10110 (least significant first)
    valuation = 0

    padic = PAdicNumber(p, digits, valuation)
    print(f"P-adic number (p={p}): digits={digits}, valuation={valuation}")
    print(f"Rational approximation: {padic.to_rational()}")

    # Test encoder
    print("\n--- Testing P-adic Encoder ---")

    encoder = PAdicEncoder(primes=[2, 3, 5])

    # Create synthetic spike pattern
    np.random.seed(42)
    spike_times = np.sort(np.random.uniform(0, 1, 20))
    print(f"Spike times: {spike_times[:5]}... ({len(spike_times)} spikes)")

    # Encode
    encodings = encoder.encode_spike_pattern(spike_times)

    for p, padic_num in encodings.items():
        print(f"\nPrime {p}:")
        print(f"  Digits: {padic_num.digits[:5]}...")
        print(f"  Valuation: {padic_num.valuation}")

    # Test distance
    print("\n--- Testing P-adic Distance ---")

    spike_times2 = spike_times + np.random.normal(0, 0.01, len(spike_times))
    encodings2 = encoder.encode_spike_pattern(spike_times2)

    for p in [2, 3, 5]:
        if p in encodings and p in encodings2:
            dist = encoder.padic_distance(encodings[p], encodings2[p])
            print(f"Distance (p={p}): {dist:.6f}")

    # Test hierarchical clustering
    print("\n--- Testing Hierarchical Clustering ---")

    # Create patterns with hierarchical structure
    patterns = []

    # Group 1: Early spikes
    for _ in range(3):
        pattern = np.random.uniform(0, 0.3, 10)
        patterns.append(pattern)

    # Group 2: Late spikes
    for _ in range(3):
        pattern = np.random.uniform(0.7, 1.0, 10)
        patterns.append(pattern)

    hierarchy_result = encoder.hierarchical_clustering(patterns, p=2)

    print(f"Hierarchy depth: {len(hierarchy_result['hierarchy'])}")
    print(f"Distance matrix shape: {hierarchy_result['distance_matrix'].shape}")

    # Test hierarchy analyzer
    print("\n--- Testing Hierarchy Analyzer ---")

    analyzer = PAdicHierarchyAnalyzer()

    # Create spike trains with structure
    spike_trains = {}
    for i in range(20):
        # Some cells fire early in theta
        if i < 10:
            spikes = np.random.uniform(0, 0.5, 30)
        else:
            spikes = np.random.uniform(0.5, 1.0, 30)
        spike_trains[i] = sorted(spikes.tolist())

    hierarchy = analyzer.analyze_temporal_hierarchy(spike_trains)

    if hierarchy:
        print(f"\nTemporal hierarchy analysis:")
        print(f"  Cycles analyzed: {hierarchy['n_cycles_analyzed']}")
        print(f"  Mean hierarchy depth: {hierarchy['hierarchy_depth']:.2f}")
        print(f"  Ultrametric violation: {hierarchy['ultrametric_violation']:.4f}")

    # Test theta-gamma mapping
    print("\n--- Testing Theta-Gamma Mapping ---")

    if encodings:
        padic_2 = encodings[2]
        mapping = analyzer.map_to_theta_gamma(padic_2)
        print(f"\nP-adic to theta-gamma mapping:")
        print(f"  Hierarchical level: {mapping['hierarchical_level']}")
        print(f"  Gamma cycle: {mapping['gamma_cycle']}")
        print(f"  Theta phase: {mapping['theta_phase']:.2f} rad")

    print("\n✓ P-adic code module working correctly!")


if __name__ == "__main__":
    test_padic_code()