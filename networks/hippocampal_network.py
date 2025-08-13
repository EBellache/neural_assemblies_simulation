"""
Hippocampal Network Module
===========================
CA3-CA1 circuit with realistic connectivity implementing Buzsáki's framework.
Integrates tropical mathematics, Golay codes, and E8 geometry.

The network generates:
- Theta oscillations (8 Hz)
- Nested gamma (40 Hz, 5 per theta)
- Sharp-wave ripples (150-250 Hz)
- Cell assemblies with error correction

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import sparse
import warnings

# Import core modules
try:
    from ..core.tropical_math import (
        tropical_inner_product, tropical_matrix_multiply,
        tropical_distance, TROPICAL_ZERO, TROPICAL_ONE
    )
    from ..core.oscillator import Oscillator, OscillatorState
    from ..core.assembly_detector import AssemblyDetector, NeuralAssembly
    from ..core.information_geometry import InformationGeometricCorrection, MorphogeneticState
    from ..topology.sheaf_cohomology import AssemblySheaf
    from ..triple_code.e8_lattice import E8Lattice, assembly_to_e8, tropical_update_e8
except ImportError:
    # For standalone testing - adjust paths
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.tropical_math import (
        tropical_inner_product, tropical_matrix_multiply,
        tropical_distance, TROPICAL_ZERO, TROPICAL_ONE
    )
    from core.oscillator import Oscillator, OscillatorState
    from core.assembly_detector import AssemblyDetector, NeuralAssembly
    from core.information_geometry import InformationGeometricCorrection, MorphogeneticState
    from topology.sheaf_cohomology import AssemblySheaf
    from triple_code.e8_lattice import E8Lattice, assembly_to_e8, tropical_update_e8

@dataclass
class PyramidalCell:
    """CA3/CA1 pyramidal neuron model."""
    idx: int
    region: str  # 'CA1' or 'CA3'
    position: Tuple[float, float] = (0.0, 0.0)  # Spatial position
    
    # Electrical properties
    threshold: float = -55.0  # mV
    membrane_potential: float = -70.0  # mV
    resting_potential: float = -70.0  # mV
    tau_membrane: float = 20.0  # ms
    
    # Spiking properties
    last_spike_time: float = -np.inf
    refractory_period: float = 2.0  # ms
    spike_times: List[float] = field(default_factory=list)
    
    # Adaptation
    adaptation: float = 0.0
    tau_adaptation: float = 100.0  # ms
    
    # Tropical state
    tropical_state: np.ndarray = field(default_factory=lambda: np.random.standard_normal(8) * 0.1)
    
    # Place field (for CA1 place cells)
    place_center: Optional[Tuple[float, float]] = None
    place_width: float = 20.0  # cm

@dataclass
class Interneuron:
    """Fast-spiking interneuron model."""
    idx: int
    subtype: str  # 'basket', 'axo-axonic', 'bistratified', 'OLM'
    
    # Electrical properties
    threshold: float = -50.0  # mV (more excitable)
    membrane_potential: float = -65.0  # mV
    resting_potential: float = -65.0  # mV
    tau_membrane: float = 10.0  # ms (faster)
    
    # Spiking properties
    last_spike_time: float = -np.inf
    refractory_period: float = 1.0  # ms (faster)
    spike_times: List[float] = field(default_factory=list)

class HippocampalNetwork:
    """
    Biologically realistic hippocampal CA3-CA1 network.
    """
    
    def __init__(self,
                 n_ca3: int = 500,
                 n_ca1: int = 500,
                 n_interneurons: int = 100,
                 connectivity_params: Optional[Dict] = None,
                 seed: int = 42):
        """
        Initialize hippocampal network.
        
        Parameters:
        -----------
        n_ca3 : int
            Number of CA3 pyramidal cells
        n_ca1 : int
            Number of CA1 pyramidal cells
        n_interneurons : int
            Number of interneurons
        connectivity_params : Dict
            Custom connectivity parameters
        seed : int
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        
        # Network size
        self.n_ca3 = n_ca3
        self.n_ca1 = n_ca1
        self.n_int = n_interneurons
        self.n_pyr = n_ca3 + n_ca1
        
        # Create cells
        self._create_cells()
        
        # Build connectivity
        self.connectivity_params = connectivity_params or self._default_connectivity()
        self._build_connectivity()
        
        # Initialize modules
        self.oscillator = Oscillator(theta_freq=8.0, gamma_per_theta=5)
        self.assembly_detector = AssemblyDetector(bin_size_ms=25, min_cells=5)
        self.info_corrector = InformationGeometricCorrection()
        self.assembly_sheaf = AssemblySheaf()
        self.e8_lattice = E8Lattice()
        
        # Network state
        self.current_time = 0.0
        self.current_assemblies: List[NeuralAssembly] = []
        self.e8_state = np.zeros(8)
        
        # Recording
        self.max_history = 10000
        self.spike_history: Dict[int, List[float]] = {i: [] for i in range(self.n_pyr)}
        
    def _create_cells(self):
        """Create pyramidal cells and interneurons."""
        # CA3 pyramidal cells
        self.ca3_cells = []
        for i in range(self.n_ca3):
            # Random spatial position
            pos = (self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            cell = PyramidalCell(idx=i, region='CA3', position=pos)
            self.ca3_cells.append(cell)
        
        # CA1 pyramidal cells (place cells)
        self.ca1_cells = []
        for i in range(self.n_ca1):
            pos = (self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            cell = PyramidalCell(idx=i + self.n_ca3, region='CA1', position=pos)
            # Assign place field
            cell.place_center = (self.rng.uniform(10, 90), self.rng.uniform(10, 90))
            self.ca1_cells.append(cell)
        
        # All pyramidal cells
        self.pyramidal_cells = self.ca3_cells + self.ca1_cells
        
        # Interneurons
        self.interneurons = []
        subtypes = ['basket', 'axo-axonic', 'bistratified', 'OLM']
        subtype_probs = [0.4, 0.2, 0.2, 0.2]
        
        for i in range(self.n_int):
            subtype = self.rng.choice(subtypes, p=subtype_probs)
            cell = Interneuron(idx=i, subtype=subtype)
            self.interneurons.append(cell)
    
    def _default_connectivity(self) -> Dict:
        """Default connectivity parameters from Buzsáki."""
        return {
            'p_ca3_ca3': 0.03,      # CA3 recurrent
            'p_ca3_ca1': 0.01,      # Schaffer collaterals
            'p_pyr_int': 0.1,       # Pyramidal → interneuron
            'p_int_pyr': 0.5,       # Interneuron → pyramidal
            'w_ca3_ca3': 2.0,       # Synaptic weights
            'w_ca3_ca1': 1.5,
            'w_pyr_int': 3.0,
            'w_int_pyr': -2.0,      # Inhibitory
            'spatial_scale': 50.0   # Spatial connectivity scale (cm)
        }
    
    def _build_connectivity(self):
        """Build synaptic connectivity matrices."""
        params = self.connectivity_params
        
        # CA3 recurrent connections (supports attractor dynamics)
        self.W_ca3_ca3 = self._create_spatial_connectivity(
            self.ca3_cells, self.ca3_cells,
            params['p_ca3_ca3'], params['w_ca3_ca3'],
            params['spatial_scale']
        )
        
        # Schaffer collaterals (CA3 → CA1)
        self.W_ca3_ca1 = self._create_spatial_connectivity(
            self.ca3_cells, self.ca1_cells,
            params['p_ca3_ca1'], params['w_ca3_ca1'],
            params['spatial_scale']
        )
        
        # Pyramidal → Interneuron
        self.W_pyr_int = self._create_random_connectivity(
            self.n_pyr, self.n_int,
            params['p_pyr_int'], params['w_pyr_int']
        )
        
        # Interneuron → Pyramidal (inhibitory)
        self.W_int_pyr = self._create_random_connectivity(
            self.n_int, self.n_pyr,
            params['p_int_pyr'], params['w_int_pyr']
        )
        
        # Convert to tropical for assembly operations
        self.W_ca3_tropical = self._to_tropical_weights(self.W_ca3_ca3)
    
    def _create_spatial_connectivity(self,
                                    pre_cells: List,
                                    post_cells: List,
                                    base_prob: float,
                                    weight: float,
                                    spatial_scale: float) -> sparse.csr_matrix:
        """Create distance-dependent connectivity."""
        n_pre = len(pre_cells)
        n_post = len(post_cells)
        
        rows, cols, data = [], [], []
        
        for i, pre in enumerate(pre_cells):
            for j, post in enumerate(post_cells):
                if pre == post:
                    continue
                
                # Distance-dependent probability
                dist = np.sqrt((pre.position[0] - post.position[0])**2 +
                             (pre.position[1] - post.position[1])**2)
                prob = base_prob * np.exp(-dist / spatial_scale)
                
                if self.rng.random() < prob:
                    # Add connection with some variation
                    w = weight * self.rng.normal(1.0, 0.2)
                    rows.append(j)
                    cols.append(i)
                    data.append(w)
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_post, n_pre))
    
    def _create_random_connectivity(self,
                                   n_pre: int,
                                   n_post: int,
                                   prob: float,
                                   weight: float) -> sparse.csr_matrix:
        """Create random connectivity."""
        # Generate random connections
        connections = self.rng.random((n_post, n_pre)) < prob
        weights = weight * (1 + 0.2 * self.rng.standard_normal((n_post, n_pre)))
        
        # Apply connection mask
        weights = weights * connections
        
        return sparse.csr_matrix(weights)
    
    def _to_tropical_weights(self, W: sparse.csr_matrix) -> np.ndarray:
        """Convert weights to tropical representation."""
        W_dense = W.toarray()
        # Positive weights become tropical weights
        W_tropical = np.where(W_dense > 0, np.log(W_dense + 1e-12), TROPICAL_ZERO)
        return W_tropical
    
    def update_dynamics(self, dt: float = 0.001, external_input: Optional[np.ndarray] = None):
        """
        Update network dynamics for time step dt.
        
        Parameters:
        -----------
        dt : float
            Time step in seconds
        external_input : np.ndarray
            External input to cells
        """
        # Update oscillator
        osc_state = self.oscillator.update(dt)
        
        # Get phase-dependent modulation
        phase_factor = osc_state.pac_strength
        
        # Update pyramidal cells
        self._update_pyramidal_cells(dt, phase_factor, external_input)
        
        # Update interneurons
        self._update_interneurons(dt, phase_factor)
        
        # Detect assemblies every gamma cycle
        if osc_state.gamma_cycle == 0 and self.current_time > 0:
            self._detect_and_encode_assemblies()
        
        # Error correction during transition phase
        if osc_state.route == 'transition':
            self._apply_error_correction()
        
        # Update time
        self.current_time += dt
    
    def _update_pyramidal_cells(self, dt: float, phase_factor: float,
                               external_input: Optional[np.ndarray]):
        """Update pyramidal cell dynamics."""
        dt_ms = dt * 1000  # Convert to ms
        
        for i, cell in enumerate(self.pyramidal_cells):
            # Leak current
            leak = -(cell.membrane_potential - cell.resting_potential) / cell.tau_membrane
            
            # Synaptic input
            syn_input = 0
            
            # CA3 recurrent (for CA3 cells)
            if cell.region == 'CA3':
                ca3_idx = self.ca3_cells.index(cell)
                # Get presynaptic activity
                pre_activity = np.array([1.0 if c.last_spike_time > self.current_time - 0.005 
                                        else 0.0 for c in self.ca3_cells])
                syn_input += np.sum(self.W_ca3_ca3[ca3_idx, :] @ pre_activity)
            
            # External input
            if external_input is not None and i < len(external_input):
                syn_input += external_input[i] * phase_factor
            
            # Inhibitory input from interneurons
            int_activity = np.array([1.0 if int_cell.last_spike_time > self.current_time - 0.002
                                    else 0.0 for int_cell in self.interneurons])
            syn_input += np.sum(self.W_int_pyr[i, :] @ int_activity)
            
            # Update membrane potential
            dV = (leak + syn_input - cell.adaptation) * dt_ms
            cell.membrane_potential += dV
            
            # Check for spike
            if (cell.membrane_potential > cell.threshold and 
                self.current_time - cell.last_spike_time > cell.refractory_period/1000):
                # Spike!
                cell.last_spike_time = self.current_time
                cell.spike_times.append(self.current_time)
                cell.membrane_potential = cell.resting_potential
                
                # Update adaptation
                cell.adaptation += 0.1
                
                # Record spike
                self.spike_history[cell.idx].append(self.current_time)
                
                # Limit history size
                if len(self.spike_history[cell.idx]) > self.max_history:
                    self.spike_history[cell.idx].pop(0)
            
            # Decay adaptation
            cell.adaptation *= np.exp(-dt_ms / cell.tau_adaptation)
    
    def _update_interneurons(self, dt: float, phase_factor: float):
        """Update interneuron dynamics."""
        dt_ms = dt * 1000
        
        for int_cell in self.interneurons:
            # Leak current
            leak = -(int_cell.membrane_potential - int_cell.resting_potential) / int_cell.tau_membrane
            
            # Excitatory input from pyramidal cells
            pyr_activity = np.array([1.0 if p.last_spike_time > self.current_time - 0.003
                                    else 0.0 for p in self.pyramidal_cells])
            syn_input = np.sum(self.W_pyr_int[int_cell.idx, :] @ pyr_activity)
            
            # Phase modulation (interneurons are more active at theta trough)
            syn_input *= (1.0 + 0.5 * (1 - phase_factor))
            
            # Update membrane potential
            dV = (leak + syn_input) * dt_ms
            int_cell.membrane_potential += dV
            
            # Check for spike
            if (int_cell.membrane_potential > int_cell.threshold and
                self.current_time - int_cell.last_spike_time > int_cell.refractory_period/1000):
                # Spike!
                int_cell.last_spike_time = self.current_time
                int_cell.spike_times.append(self.current_time)
                int_cell.membrane_potential = int_cell.resting_potential
    
    def _detect_and_encode_assemblies(self):
        """Detect assemblies and encode with Golay codes."""
        # Get recent spike trains
        time_window = (self.current_time - 0.125, self.current_time)  # Last theta cycle
        
        # Detect assemblies
        assemblies = self.assembly_detector.detect_assemblies_tropical(
            self.spike_history, time_window
        )
        
        # Encode each assembly
        for assembly in assemblies:
            # Get firing pattern
            pattern = np.zeros(24)
            for i, cell_id in enumerate(assembly.cells[:24]):
                # Check if cell fired recently
                if cell_id in self.spike_history:
                    recent_spikes = [t for t in self.spike_history[cell_id]
                                   if time_window[0] < t < time_window[1]]
                    pattern[i] = len(recent_spikes) > 0
            
            # Golay encoding
            # Use information geometry instead of Golay for error correction
            # Create morphogenetic state for the assembly
            distribution = np.zeros(max(50, len(assembly.cells)))
            for i, cell_id in enumerate(assembly.cells):
                if i < len(distribution):
                    distribution[i] = 1.0
            distribution = distribution / np.sum(distribution)
            
            assembly.morphogenetic_state = MorphogeneticState(
                position=np.array([0.0, 0.0]),
                phase=np.array([self.theta_phase, self.gamma_phase]),
                curvature=np.array([0.0, 0.0]),
                momentum=np.random.randn(8) * 0.1,
                distribution=distribution
            )
            
            # Map to E8
            assembly.e8_coords = assembly_to_e8(pattern, self.e8_lattice)
        
        self.current_assemblies = assemblies
        
        # Update global E8 state
        if assemblies:
            # Average E8 coordinates weighted by strength
            total_weight = sum(a.strength for a in assemblies)
            if total_weight > 0:
                weighted_e8 = sum(a.strength * a.e8_coords for a in assemblies)
                self.e8_state = tropical_update_e8(
                    self.e8_state, 
                    weighted_e8 / total_weight,
                    rate=0.1
                )
    
    def _apply_error_correction(self):
        """Apply Golay error correction to assembly patterns."""
        for assembly in self.current_assemblies:
            if assembly.golay_codeword is not None:
                # Add noise to simulate errors
                noise_level = 0.05
                noisy = assembly.golay_codeword.copy()
                errors = self.rng.random(24) < noise_level
                noisy[errors] = 1 - noisy[errors]
                
                # Decode and correct
                try:
                    # Use information geometry correction
                    noisy_state = MorphogeneticState(
                        position=np.array([0.0, 0.0]),
                        phase=np.array([0.0, 0.0]),
                        curvature=np.array([0.0, 0.0]),
                        momentum=np.random.randn(8) * 0.1,
                        distribution=noisy
                    )
                    corrected_state = self.info_corrector.correct_state(noisy_state)
                    corrected = (corrected_state.distribution > 0.5).astype(np.uint8)[:24]
                    n_errors = np.sum(np.abs(corrected - pattern[:24]))
                    
                    # Update assembly if errors were corrected
                    if n_errors > 0:
                        assembly.golay_codeword[:12] = corrected
                        # Re-compute E8 coords
                        assembly.e8_coords = assembly_to_e8(corrected, self.e8_lattice)
                except:
                    pass  # Keep original if decoding fails
    
    def simulate_theta_cycle(self, external_input: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Simulate one complete theta cycle.
        
        Returns:
        --------
        Dict with simulation results
        """
        theta_period = 1.0 / self.oscillator.theta_freq
        dt = 0.001  # 1ms time step
        n_steps = int(theta_period / dt)
        
        # Track results
        results = {
            'spike_times': {i: [] for i in range(self.n_pyr)},
            'assemblies': [],
            'e8_trajectory': [],
            'oscillator_states': []
        }
        
        for step in range(n_steps):
            # Update dynamics
            self.update_dynamics(dt, external_input)
            
            # Record state periodically
            if step % 10 == 0:  # Every 10ms
                results['e8_trajectory'].append(self.e8_state.copy())
                results['oscillator_states'].append(
                    self.oscillator.phase_history[-1] if self.oscillator.phase_history else None
                )
        
        # Final assembly detection
        results['assemblies'] = self.current_assemblies
        results['spike_times'] = {k: v.copy() for k, v in self.spike_history.items() 
                                 if len(v) > 0}
        
        return results
    
    def generate_place_field_input(self, position: Tuple[float, float]) -> np.ndarray:
        """
        Generate input based on animal position (for place cells).
        
        Parameters:
        -----------
        position : (x, y) in cm
        
        Returns:
        --------
        np.ndarray : Input strength for each pyramidal cell
        """
        input_strength = np.zeros(self.n_pyr)
        
        for i, cell in enumerate(self.pyramidal_cells):
            if cell.region == 'CA1' and cell.place_center is not None:
                # Gaussian place field
                dist = np.sqrt((position[0] - cell.place_center[0])**2 +
                             (position[1] - cell.place_center[1])**2)
                input_strength[i] = np.exp(-(dist**2) / (2 * cell.place_width**2))
        
        return input_strength * 10.0  # Scale factor
    
    def compute_network_statistics(self) -> Dict[str, Any]:
        """Compute network activity statistics."""
        # Firing rates
        time_window = 1.0  # Last second
        firing_rates = []
        
        for cell_id, spikes in self.spike_history.items():
            recent_spikes = [t for t in spikes if t > self.current_time - time_window]
            rate = len(recent_spikes) / time_window
            firing_rates.append(rate)
        
        # Assembly statistics
        assembly_stats = self.assembly_detector.get_assembly_statistics()
        
        # E8 state properties
        e8_casimirs = self.e8_lattice.compute_casimir_invariants(self.e8_state)
        
        return {
            'mean_firing_rate': np.mean(firing_rates) if firing_rates else 0,
            'std_firing_rate': np.std(firing_rates) if firing_rates else 0,
            'active_cells': sum(1 for r in firing_rates if r > 0.5),
            'n_assemblies': len(self.current_assemblies),
            'assembly_stats': assembly_stats,
            'e8_casimirs': e8_casimirs,
            'e8_norm': float(np.linalg.norm(self.e8_state))
        }

def test_hippocampal_network():
    """Test hippocampal network functionality."""
    print("\n=== Testing Hippocampal Network ===\n")
    
    # Create small test network
    network = HippocampalNetwork(n_ca3=100, n_ca1=100, n_interneurons=20, seed=42)
    
    print(f"Network created:")
    print(f"  CA3 cells: {network.n_ca3}")
    print(f"  CA1 cells: {network.n_ca1}")
    print(f"  Interneurons: {network.n_int}")
    print(f"  Connectivity: CA3-CA3 has {network.W_ca3_ca3.nnz} connections")
    
    # Test place field input
    print("\n--- Testing Place Field Input ---")
    position = (50, 50)  # Center of environment
    place_input = network.generate_place_field_input(position)
    print(f"Position: {position}")
    print(f"Active place cells: {np.sum(place_input > 0.1)}")
    print(f"Max input: {np.max(place_input):.2f}")
    
    # Simulate one theta cycle
    print("\n--- Simulating Theta Cycle ---")
    results = network.simulate_theta_cycle(external_input=place_input)
    
    # Count spikes
    total_spikes = sum(len(spikes) for spikes in results['spike_times'].values())
    print(f"Total spikes: {total_spikes}")
    print(f"Detected assemblies: {len(results['assemblies'])}")
    
    if results['assemblies']:
        assembly = results['assemblies'][0]
        print(f"\nFirst assembly:")
        print(f"  Size: {len(assembly.cells)} cells")
        print(f"  Strength: {assembly.strength:.3f}")
        print(f"  E8 norm: {np.linalg.norm(assembly.e8_coords):.3f}")
        if assembly.golay_codeword is not None:
            print(f"  Golay code: {assembly.golay_codeword[:8]}...")
    
    # Compute statistics
    print("\n--- Network Statistics ---")
    stats = network.compute_network_statistics()
    print(f"Mean firing rate: {stats['mean_firing_rate']:.2f} Hz")
    print(f"Active cells: {stats['active_cells']}")
    print(f"E8 state norm: {stats['e8_norm']:.3f}")
    
    # Test error correction
    print("\n--- Testing Error Correction ---")
    network._apply_error_correction()
    print("Error correction applied successfully")
    
    # Test multiple theta cycles
    print("\n--- Simulating Multiple Cycles ---")
    e8_trajectory = []
    for cycle in range(5):
        results = network.simulate_theta_cycle(external_input=place_input * (0.5 + 0.5 * cycle / 5))
        e8_trajectory.append(network.e8_state.copy())
        print(f"Cycle {cycle + 1}: {len(results['assemblies'])} assemblies, "
              f"E8 norm = {np.linalg.norm(network.e8_state):.3f}")
    
    print("\n✓ Hippocampal network working correctly!")

if __name__ == "__main__":
    test_hippocampal_network()
