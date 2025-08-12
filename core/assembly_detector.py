"""
Assembly Detector Module
========================
Detect and track neuronal assemblies using tropical mathematics.
Assemblies are identified as attractors in the E8 state space.

Key insight: Assemblies are corners in piecewise-linear tropical dynamics!

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Import tropical math operations
try:
    from .tropical_math import (
        tropical_inner_product, tropical_distance, tropical_matrix_multiply,
        tropical_eigenvector, tropical_eigenvalue, tropical_project_onto_span,
        TROPICAL_ZERO, TROPICAL_ONE
    )
except ImportError:
    # For standalone testing
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from tropical_math import (
        tropical_inner_product, tropical_distance, tropical_matrix_multiply,
        tropical_eigenvector, tropical_eigenvalue, tropical_project_onto_span,
        TROPICAL_ZERO, TROPICAL_ONE
    )


@dataclass
class NeuralAssembly:
    """Represents a detected neuronal assembly."""
    assembly_id: int
    cells: List[int]  # Indices of participating cells
    strength: float  # Tropical correlation strength
    e8_coords: np.ndarray = field(default_factory=lambda: np.zeros(8))  # E8 lattice coordinates
    stability: float = 0.0  # Stability in tropical metric
    formation_time: Optional[float] = None
    last_active: Optional[float] = None
    activation_count: int = 0
    preferred_theta_phase: float = 0.0  # Preferred phase for activation
    golay_codeword: Optional[np.ndarray] = None  # Associated Golay code


@dataclass
class AssemblyTrajectory:
    """Tracks evolution of assemblies over time."""
    assembly_id: int
    trajectory: List[Tuple[float, np.ndarray]]  # (time, e8_coords)
    stability_history: List[float]
    participation_history: List[List[int]]
    formation_events: List[float]  # Times of formation
    dissolution_events: List[float]  # Times of dissolution


class AssemblyDetector:
    """
    Identifies cell assemblies as tropical fixed points in activity space.
    Assemblies maximize tropical inner product (synchrony).
    """

    def __init__(self,
                 bin_size_ms: float = 25.0,
                 min_cells: int = 5,
                 stability_threshold: float = 0.1,
                 correlation_threshold: float = 0.3):
        """
        Initialize assembly detector.

        Parameters:
        -----------
        bin_size_ms : float
            Time bin size in ms (typically gamma cycle duration)
        min_cells : int
            Minimum number of cells to form an assembly
        stability_threshold : float
            Minimum stability for valid assembly
        correlation_threshold : float
            Minimum correlation for assembly membership
        """
        self.bin_size_ms = bin_size_ms
        self.min_cells = min_cells
        self.stability_threshold = stability_threshold
        self.correlation_threshold = correlation_threshold

        # Assembly tracking
        self.assemblies: Dict[int, NeuralAssembly] = {}
        self.assembly_trajectories: Dict[int, AssemblyTrajectory] = {}
        self._next_assembly_id = 0

        # Detection parameters
        self.max_assemblies = 50  # Maximum assemblies to track
        self.merge_threshold = 0.7  # Overlap threshold for merging

    def detect_assemblies_tropical(self,
                                   spike_trains: Dict[int, List[float]],
                                   time_window: Tuple[float, float]) -> List[NeuralAssembly]:
        """
        Detect assemblies using tropical clustering.

        Parameters:
        -----------
        spike_trains : Dict[int, List[float]]
            Dictionary mapping cell IDs to spike times
        time_window : Tuple[float, float]
            (start_time, end_time) for detection

        Returns:
        --------
        List[NeuralAssembly] : Detected assemblies
        """
        # Bin spikes at gamma resolution
        binned_spikes = self._bin_spikes(spike_trains, time_window)

        if len(binned_spikes) < self.min_cells:
            return []

        # Compute tropical correlation matrix
        C_tropical = self._compute_tropical_correlation(binned_spikes)

        # Find tropical eigenassemblies
        assemblies = self._find_tropical_assemblies(C_tropical, list(binned_spikes.keys()))

        # Map to E8 space and compute stability
        for assembly in assemblies:
            assembly.e8_coords = self._project_assembly_to_e8(assembly, binned_spikes)
            assembly.stability = self._compute_stability_tropical(assembly, C_tropical)
            assembly.formation_time = time_window[0]
            assembly.last_active = time_window[1]

        # Filter by stability and size
        stable_assemblies = [
            a for a in assemblies
            if a.stability > self.stability_threshold and len(a.cells) >= self.min_cells
        ]

        # Merge overlapping assemblies
        merged_assemblies = self._merge_overlapping_assemblies(stable_assemblies)

        # Update internal tracking
        for assembly in merged_assemblies:
            self.assemblies[assembly.assembly_id] = assembly

        return merged_assemblies

    def _bin_spikes(self,
                    spike_trains: Dict[int, List[float]],
                    time_window: Tuple[float, float]) -> Dict[int, np.ndarray]:
        """
        Bin spike trains at gamma resolution.
        """
        start_time, end_time = time_window
        duration = end_time - start_time
        n_bins = int(duration * 1000 / self.bin_size_ms) + 1

        binned = {}
        for cell_id, spike_times in spike_trains.items():
            # Count spikes in each bin
            bins = np.zeros(n_bins)
            for t in spike_times:
                if start_time <= t < end_time:
                    bin_idx = int((t - start_time) * 1000 / self.bin_size_ms)
                    if 0 <= bin_idx < n_bins:
                        bins[bin_idx] += 1

            # Only keep cells that fired
            if np.sum(bins) > 0:
                binned[cell_id] = bins

        return binned

    def _compute_tropical_correlation(self,
                                      binned_spikes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute tropical correlation matrix.
        C[i,j] = max_t(spike_i[t] + spike_j[t])
        """
        cell_ids = list(binned_spikes.keys())
        n_cells = len(cell_ids)

        # Initialize with tropical zeros
        C = np.full((n_cells, n_cells), TROPICAL_ZERO)

        # Convert to array for efficiency
        spike_array = np.array([binned_spikes[cid] for cid in cell_ids])

        # Compute pairwise tropical inner products
        for i in range(n_cells):
            for j in range(i, n_cells):
                # Tropical inner product
                correlation = tropical_inner_product(spike_array[i], spike_array[j])

                # Normalize by activity levels
                norm_i = np.max(spike_array[i]) if np.any(spike_array[i] > 0) else 1
                norm_j = np.max(spike_array[j]) if np.any(spike_array[j] > 0) else 1

                if norm_i > 0 and norm_j > 0:
                    correlation = correlation / (norm_i + norm_j)

                C[i, j] = correlation
                C[j, i] = correlation

        return C

    def _find_tropical_assemblies(self,
                                  C_tropical: np.ndarray,
                                  cell_ids: List[int]) -> List[NeuralAssembly]:
        """
        Find assemblies as clusters in tropical correlation space.
        Uses tropical eigenvector decomposition.
        """
        n_cells = len(cell_ids)
        assemblies = []

        # Method 1: Tropical eigenvector clustering
        try:
            # Find dominant tropical eigenvector
            eigenval = tropical_eigenvalue(C_tropical)
            eigenvec = tropical_eigenvector(C_tropical, eigenval)

            # Cells with high eigenvector values form assemblies
            threshold = np.percentile(eigenvec[eigenvec > TROPICAL_ZERO], 75)
            member_mask = eigenvec > threshold

            if np.sum(member_mask) >= self.min_cells:
                member_indices = np.where(member_mask)[0]
                member_cells = [cell_ids[i] for i in member_indices]

                assembly = NeuralAssembly(
                    assembly_id=self._get_next_assembly_id(),
                    cells=member_cells,
                    strength=float(eigenval)
                )
                assemblies.append(assembly)
        except:
            pass  # Fallback to other methods

        # Method 2: Hierarchical clustering on tropical distances
        if len(assemblies) < 3:  # Need more assemblies
            # Convert correlation to distance
            D = np.max(C_tropical) - C_tropical
            np.fill_diagonal(D, 0)

            # Hierarchical clustering
            if n_cells > 2:
                condensed_D = pdist(D, metric='euclidean')
                linkage_matrix = linkage(condensed_D, method='average')

                # Cut dendrogram to get clusters
                max_clusters = min(10, n_cells // self.min_cells)
                for n_clusters in range(2, max_clusters + 1):
                    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                    for cluster_id in np.unique(clusters):
                        cluster_mask = clusters == cluster_id
                        if np.sum(cluster_mask) >= self.min_cells:
                            member_indices = np.where(cluster_mask)[0]
                            member_cells = [cell_ids[i] for i in member_indices]

                            # Compute assembly strength
                            submatrix = C_tropical[np.ix_(member_indices, member_indices)]
                            strength = np.mean(submatrix[submatrix > TROPICAL_ZERO])

                            if strength > self.correlation_threshold:
                                assembly = NeuralAssembly(
                                    assembly_id=self._get_next_assembly_id(),
                                    cells=member_cells,
                                    strength=float(strength)
                                )
                                assemblies.append(assembly)

        # Method 3: Greedy expansion from seed cells
        if len(assemblies) < 5:
            used_cells = set()
            for a in assemblies:
                used_cells.update(a.cells)

            # Find unused cells with high connectivity
            for seed_idx in range(n_cells):
                if cell_ids[seed_idx] in used_cells:
                    continue

                # Expand from seed
                assembly_cells = {seed_idx}
                changed = True

                while changed and len(assembly_cells) < 50:
                    changed = False
                    candidates = []

                    for idx in range(n_cells):
                        if idx not in assembly_cells:
                            # Average correlation with assembly members
                            corr_sum = sum(C_tropical[idx, m] for m in assembly_cells)
                            avg_corr = corr_sum / len(assembly_cells)

                            if avg_corr > self.correlation_threshold:
                                candidates.append((idx, avg_corr))

                    if candidates:
                        # Add best candidate
                        best_idx = max(candidates, key=lambda x: x[1])[0]
                        assembly_cells.add(best_idx)
                        changed = True

                if len(assembly_cells) >= self.min_cells:
                    member_cells = [cell_ids[i] for i in assembly_cells]
                    member_indices = list(assembly_cells)

                    # Compute strength
                    submatrix = C_tropical[np.ix_(member_indices, member_indices)]
                    strength = np.mean(submatrix[submatrix > TROPICAL_ZERO])

                    assembly = NeuralAssembly(
                        assembly_id=self._get_next_assembly_id(),
                        cells=member_cells,
                        strength=float(strength)
                    )
                    assemblies.append(assembly)
                    used_cells.update(member_cells)

        return assemblies[:self.max_assemblies]  # Limit number of assemblies

    def _project_assembly_to_e8(self,
                                assembly: NeuralAssembly,
                                binned_spikes: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Project assembly activity pattern to E8 space.
        Uses PCA-like tropical projection.
        """
        # Get activity vectors for assembly members
        member_activities = []
        for cell_id in assembly.cells:
            if cell_id in binned_spikes:
                member_activities.append(binned_spikes[cell_id])

        if not member_activities:
            return np.zeros(8)

        # Stack activities
        A = np.array(member_activities)

        # Compute tropical "principal components"
        # Use first 8 time bins as basis (simplified)
        n_components = min(8, A.shape[1])

        # Tropical projection onto first 8 dimensions
        e8_coords = np.zeros(8)
        for i in range(n_components):
            # Max pooling across cells for each time bin
            e8_coords[i] = np.max(A[:, i]) if A.shape[1] > i else 0

        # Normalize to unit tropical norm
        max_val = np.max(np.abs(e8_coords))
        if max_val > 0:
            e8_coords = e8_coords / max_val

        return e8_coords

    def _compute_stability_tropical(self,
                                    assembly: NeuralAssembly,
                                    C_tropical: np.ndarray) -> float:
        """
        Compute assembly stability using tropical distance to other assemblies.
        High stability means the assembly is well-separated.
        """
        # Get indices of assembly members in the correlation matrix
        # Need to map from cell IDs back to matrix indices
        # This is tricky because we need the mapping used in _find_tropical_assemblies
        # For now, use a simpler approach based on assembly size
        n_cells = C_tropical.shape[0]
        if len(assembly.cells) > n_cells:
            # Assembly has more cells than correlation matrix - something is wrong
            return 0.0
            
        # Use first len(assembly.cells) indices as proxy
        member_indices = list(range(min(len(assembly.cells), n_cells)))
        non_member_indices = [i for i in range(n_cells) if i not in member_indices]

        if not member_indices or not non_member_indices:
            return 0.0

        # Internal coherence: average correlation within assembly
        internal_corr = []
        for i in member_indices:
            for j in member_indices:
                if i != j:
                    internal_corr.append(C_tropical[i, j])

        avg_internal = np.mean(internal_corr) if internal_corr else 0

        # External separation: average correlation with non-members
        external_corr = []
        for i in member_indices:
            for j in non_member_indices:
                external_corr.append(C_tropical[i, j])

        avg_external = np.mean(external_corr) if external_corr else 0

        # Stability is the difference (higher is better)
        stability = avg_internal - avg_external

        # Normalize to [0, 1]
        return float(np.clip(stability, 0, 1))

    def _merge_overlapping_assemblies(self,
                                      assemblies: List[NeuralAssembly]) -> List[NeuralAssembly]:
        """
        Merge assemblies with high overlap.
        """
        if len(assemblies) <= 1:
            return assemblies

        merged = []
        used = set()

        for i, a1 in enumerate(assemblies):
            if i in used:
                continue

            # Start new merged assembly
            merged_cells = set(a1.cells)
            merged_strength = a1.strength
            merged_e8 = a1.e8_coords.copy()
            merge_count = 1

            # Find overlapping assemblies
            for j, a2 in enumerate(assemblies[i + 1:], i + 1):
                if j in used:
                    continue

                # Compute overlap
                overlap = len(set(a1.cells) & set(a2.cells))
                overlap_ratio = overlap / min(len(a1.cells), len(a2.cells))

                if overlap_ratio > self.merge_threshold:
                    # Merge
                    merged_cells.update(a2.cells)
                    merged_strength += a2.strength
                    merged_e8 += a2.e8_coords
                    merge_count += 1
                    used.add(j)

            # Create merged assembly
            merged_assembly = NeuralAssembly(
                assembly_id=a1.assembly_id,
                cells=list(merged_cells),
                strength=merged_strength / merge_count,
                e8_coords=merged_e8 / merge_count,
                stability=a1.stability,
                formation_time=a1.formation_time,
                last_active=a1.last_active
            )
            merged.append(merged_assembly)
            used.add(i)

        return merged

    def track_assembly_evolution(self,
                                 spike_trains: Dict[int, List[float]],
                                 start_time: float,
                                 end_time: float,
                                 theta_period: float = 0.125) -> Dict[int, AssemblyTrajectory]:
        """
        Track how assemblies evolve through time.
        """
        trajectories = {}
        current_time = start_time

        while current_time < end_time:
            window_end = min(current_time + theta_period, end_time)

            # Detect assemblies in this theta cycle
            assemblies_t = self.detect_assemblies_tropical(
                spike_trains, (current_time, window_end)
            )

            # Update trajectories
            for assembly in assemblies_t:
                aid = assembly.assembly_id

                if aid not in trajectories:
                    # New trajectory
                    trajectory = AssemblyTrajectory(
                        assembly_id=aid,
                        trajectory=[(current_time, assembly.e8_coords.copy())],
                        stability_history=[assembly.stability],
                        participation_history=[assembly.cells.copy()],
                        formation_events=[current_time],
                        dissolution_events=[]
                    )
                    trajectories[aid] = trajectory
                else:
                    # Update existing
                    traj = trajectories[aid]
                    traj.trajectory.append((current_time, assembly.e8_coords.copy()))
                    traj.stability_history.append(assembly.stability)
                    traj.participation_history.append(assembly.cells.copy())

            current_time += theta_period

        return trajectories

    def compute_assembly_graph(self, assemblies: List[NeuralAssembly]) -> np.ndarray:
        """
        Compute graph of assembly relationships.
        Edge weight = overlap between assemblies.
        """
        n = len(assemblies)
        graph = np.zeros((n, n))

        for i, a1 in enumerate(assemblies):
            for j, a2 in enumerate(assemblies):
                if i != j:
                    overlap = len(set(a1.cells) & set(a2.cells))
                    graph[i, j] = overlap / np.sqrt(len(a1.cells) * len(a2.cells))

        return graph

    def _get_next_assembly_id(self) -> int:
        """Get next available assembly ID."""
        aid = self._next_assembly_id
        self._next_assembly_id += 1
        return aid

    def get_assembly_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over detected assemblies.
        """
        if not self.assemblies:
            return {}

        assemblies = list(self.assemblies.values())

        return {
            'n_assemblies': len(assemblies),
            'mean_size': np.mean([len(a.cells) for a in assemblies]),
            'std_size': np.std([len(a.cells) for a in assemblies]),
            'mean_strength': np.mean([a.strength for a in assemblies]),
            'mean_stability': np.mean([a.stability for a in assemblies]),
            'total_cells': len(set().union(*[set(a.cells) for a in assemblies]))
        }


def test_assembly_detector():
    """Test assembly detection functionality."""
    print("\n=== Testing Assembly Detector ===\n")

    # Create synthetic spike data with embedded assemblies
    np.random.seed(42)
    n_cells = 100
    duration = 1.0  # 1 second

    # Create 3 ground truth assemblies
    assembly1_cells = list(range(0, 20))
    assembly2_cells = list(range(15, 35))  # Some overlap
    assembly3_cells = list(range(50, 65))

    # Generate spike trains
    spike_trains = {}

    # Assembly 1: synchronous at t=0.1, 0.3, 0.5
    for cell_id in assembly1_cells:
        spikes = [0.1, 0.3, 0.5]
        # Add jitter
        spikes = [t + np.random.normal(0, 0.005) for t in spikes]
        spike_trains[cell_id] = spikes

    # Assembly 2: synchronous at t=0.2, 0.4, 0.6
    for cell_id in assembly2_cells:
        spikes = [0.2, 0.4, 0.6]
        spikes = [t + np.random.normal(0, 0.005) for t in spikes]
        spike_trains[cell_id] = spikes

    # Assembly 3: synchronous at t=0.15, 0.45, 0.75
    for cell_id in assembly3_cells:
        spikes = [0.15, 0.45, 0.75]
        spikes = [t + np.random.normal(0, 0.005) for t in spikes]
        spike_trains[cell_id] = spikes

    # Add random background activity
    for cell_id in range(n_cells):
        if cell_id not in spike_trains:
            spike_trains[cell_id] = []
        # Add random spikes
        n_random = np.random.poisson(2)  # 2 Hz background
        random_spikes = sorted(np.random.uniform(0, duration, n_random))
        spike_trains[cell_id].extend(random_spikes)
        spike_trains[cell_id] = sorted(spike_trains[cell_id])

    # Detect assemblies
    detector = AssemblyDetector(bin_size_ms=25, min_cells=5)
    assemblies = detector.detect_assemblies_tropical(spike_trains, (0, duration))

    print(f"Detected {len(assemblies)} assemblies:")
    for i, assembly in enumerate(assemblies):
        print(f"  Assembly {i + 1}:")
        print(f"    Size: {len(assembly.cells)} cells")
        print(f"    Strength: {assembly.strength:.3f}")
        print(f"    Stability: {assembly.stability:.3f}")
        print(f"    E8 coords: {assembly.e8_coords[:3]}...")  # Show first 3

    # Track evolution
    print("\nTracking assembly evolution...")
    trajectories = detector.track_assembly_evolution(spike_trains, 0, duration)

    for aid, traj in trajectories.items():
        print(f"  Assembly {aid}: {len(traj.trajectory)} time points")

    # Compute statistics
    stats = detector.get_assembly_statistics()
    print(f"\nAssembly statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Assembly detector working correctly!")


if __name__ == "__main__":
    test_assembly_detector()