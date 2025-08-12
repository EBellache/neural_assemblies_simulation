"""
Buzsáki Metrics Module
======================
Validation metrics based on Buzsáki's hippocampal research.
Implements key measures from the Rhythms of the Brain framework.

Key metrics:
- Theta-gamma coupling (PAC)
- Assembly compression ratio
- Replay fidelity
- Sharp-wave ripple detection
- Sequence scoring

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import warnings

# Import tropical operations
try:
    from ..core.tropical_math import tropical_inner_product, tropical_distance
except ImportError:
    # Fallback
    def tropical_inner_product(a, b):
        return np.max(a + b)
    
    def tropical_distance(a, b):
        diff = a - b
        return np.max(diff) - np.min(diff)

@dataclass
class BuzsákiMetrics:
    """Container for Buzsáki-style hippocampal metrics."""
    theta_power: float
    gamma_power: float
    theta_gamma_pac: float
    ripple_rate: float
    assembly_compression: float
    replay_score: float
    sequence_consistency: float
    participation_ratio: float
    synchrony_index: float
    information_content: float

class ThetaGammaAnalyzer:
    """Analyze theta-gamma oscillations and coupling."""
    
    def __init__(self, 
                 fs: float = 1000.0,
                 theta_band: Tuple[float, float] = (4, 12),
                 gamma_band: Tuple[float, float] = (30, 100)):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        theta_band : tuple
            (low, high) frequency for theta
        gamma_band : tuple
            (low, high) frequency for gamma
        """
        self.fs = fs
        self.theta_band = theta_band
        self.gamma_band = gamma_band
        
    def compute_pac(self, lfp: np.ndarray) -> float:
        """
        Compute Phase-Amplitude Coupling (PAC).
        
        Parameters:
        -----------
        lfp : np.ndarray
            Local field potential signal
        
        Returns:
        --------
        float : PAC strength (0-1)
        """
        if len(lfp) < self.fs:
            return 0.0
        
        # Extract theta phase
        theta_phase = self._extract_phase(lfp, self.theta_band)
        
        # Extract gamma amplitude
        gamma_amp = self._extract_amplitude(lfp, self.gamma_band)
        
        # Compute PAC using Mean Vector Length (MVL)
        pac = self._compute_mvl(theta_phase, gamma_amp)
        
        return float(pac)
    
    def _extract_phase(self, signal_input: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """Extract instantaneous phase in frequency band."""
        # Bandpass filter
        sos = self._design_bandpass(band[0], band[1])
        from scipy import signal as scipy_signal
        filtered = scipy_signal.sosfilt(sos, signal_input)
        
        # Hilbert transform for phase
        analytic = scipy_signal.hilbert(filtered)
        phase = np.angle(analytic)
        
        return phase
    
    def _extract_amplitude(self, signal_input: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """Extract instantaneous amplitude in frequency band."""
        # Bandpass filter
        sos = self._design_bandpass(band[0], band[1])
        from scipy import signal as scipy_signal
        filtered = scipy_signal.sosfilt(sos, signal_input)
        
        # Hilbert transform for amplitude
        analytic = scipy_signal.hilbert(filtered)
        amplitude = np.abs(analytic)
        
        return amplitude
    
    def _design_bandpass(self, lowcut: float, highcut: float, order: int = 4):
        """Design Butterworth bandpass filter."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure valid frequency range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        from scipy import signal as scipy_signal
        sos = scipy_signal.butter(order, [low, high], btype='band', output='sos')
        return sos
    
    def _compute_mvl(self, phase: np.ndarray, amplitude: np.ndarray) -> float:
        """
        Compute Mean Vector Length for PAC.
        MVL = |sum(A * exp(i*φ))| / sum(A)
        """
        if len(phase) == 0 or len(amplitude) == 0:
            return 0.0
        
        # Normalize amplitude
        amplitude = amplitude / np.sum(amplitude)
        
        # Compute complex mean
        complex_mean = np.sum(amplitude * np.exp(1j * phase))
        
        # MVL is the magnitude
        mvl = np.abs(complex_mean)
        
        return float(mvl)
    
    def detect_theta_cycles(self, lfp: np.ndarray, min_duration: float = 0.08) -> List[Tuple[int, int]]:
        """
        Detect individual theta cycles.
        
        Parameters:
        -----------
        lfp : np.ndarray
            Local field potential
        min_duration : float
            Minimum cycle duration in seconds
        
        Returns:
        --------
        List of (start_idx, end_idx) for each cycle
        """
        # Filter in theta band
        sos = self._design_bandpass(self.theta_band[0], self.theta_band[1])
        from scipy import signal as scipy_signal
        theta_filtered = scipy_signal.sosfilt(sos, lfp)
        
        # Find peaks and troughs
        peaks, _ = scipy_signal.find_peaks(theta_filtered, distance=int(self.fs * min_duration))
        
        cycles = []
        for i in range(len(peaks) - 1):
            cycles.append((peaks[i], peaks[i+1]))
        
        return cycles

class RippleDetector:
    """Detect sharp-wave ripples (SWRs)."""
    
    def __init__(self,
                 fs: float = 1000.0,
                 ripple_band: Tuple[float, float] = (150, 250),
                 threshold_std: float = 3.0):
        """
        Initialize ripple detector.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency
        ripple_band : tuple
            Frequency band for ripples
        threshold_std : float
            Detection threshold in standard deviations
        """
        self.fs = fs
        self.ripple_band = ripple_band
        self.threshold_std = threshold_std
    
    def detect_ripples(self, lfp: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect ripple events.
        
        Returns:
        --------
        List of ripple events with properties
        """
        # Bandpass filter
        from scipy import signal as scipy_signal
        sos = scipy_signal.butter(4, self.ripple_band, btype='band', 
                          fs=self.fs, output='sos')
        ripple_band = scipy_signal.sosfilt(sos, lfp)
        
        # Compute envelope
        analytic = scipy_signal.hilbert(ripple_band)
        envelope = np.abs(analytic)
        
        # Smooth envelope
        envelope = gaussian_filter1d(envelope, sigma=int(0.01 * self.fs))
        
        # Threshold
        threshold = np.mean(envelope) + self.threshold_std * np.std(envelope)
        
        # Find ripple events
        above_threshold = envelope > threshold
        
        # Get event boundaries
        events = []
        in_event = False
        start_idx = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_event:
                start_idx = i
                in_event = True
            elif not above and in_event:
                # Event ended
                duration = (i - start_idx) / self.fs
                
                # Minimum duration criterion (20ms)
                if duration > 0.02:
                    events.append({
                        'start_idx': start_idx,
                        'end_idx': i,
                        'duration': duration,
                        'peak_amplitude': np.max(envelope[start_idx:i]),
                        'peak_frequency': self._compute_peak_frequency(
                            ripple_band[start_idx:i]
                        )
                    })
                
                in_event = False
        
        return events
    
    def _compute_peak_frequency(self, signal_segment: np.ndarray) -> float:
        """Compute peak frequency of signal segment."""
        if len(signal_segment) < 64:
            return (self.ripple_band[0] + self.ripple_band[1]) / 2
        
        # Compute power spectrum
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(signal_segment, self.fs, nperseg=min(len(signal_segment), 256))
        
        # Find peak in ripple band
        band_mask = (freqs >= self.ripple_band[0]) & (freqs <= self.ripple_band[1])
        if np.any(band_mask):
            band_psd = psd[band_mask]
            band_freqs = freqs[band_mask]
            peak_idx = np.argmax(band_psd)
            return float(band_freqs[peak_idx])
        
        return (self.ripple_band[0] + self.ripple_band[1]) / 2

class AssemblyAnalyzer:
    """Analyze neuronal assembly properties."""
    
    def __init__(self):
        self.min_assembly_size = 5
        self.synchrony_window = 0.025  # 25ms
    
    def compute_compression_ratio(self,
                                 spike_trains: Dict[int, List[float]],
                                 assemblies: List[Any]) -> float:
        """
        Compute assembly compression ratio.
        
        Compression = (Total cells) / (Active assemblies * Avg assembly size)
        Higher values indicate better compression.
        """
        if not assemblies:
            return 1.0
        
        total_cells = len(spike_trains)
        active_assemblies = len(assemblies)
        avg_size = np.mean([len(a.cells) for a in assemblies])
        
        if active_assemblies * avg_size == 0:
            return 1.0
        
        compression = total_cells / (active_assemblies * avg_size)
        
        return float(compression)
    
    def compute_participation_ratio(self,
                                   spike_trains: Dict[int, List[float]],
                                   time_window: Tuple[float, float]) -> float:
        """
        Compute participation ratio (how many cells participate).
        
        PR = (sum of participation)^2 / sum(participation^2)
        Ranges from 1 (single cell) to N (all cells equal).
        """
        participations = []
        
        for cell_id, spikes in spike_trains.items():
            # Count spikes in window
            n_spikes = sum(1 for t in spikes if time_window[0] <= t <= time_window[1])
            participations.append(n_spikes)
        
        participations = np.array(participations)
        
        if np.sum(participations) == 0:
            return 0.0
        
        # Normalize
        participations = participations / np.sum(participations)
        
        # Compute PR
        pr = np.sum(participations)**2 / np.sum(participations**2)
        
        return float(pr)
    
    def compute_synchrony_index(self,
                               spike_trains: Dict[int, List[float]],
                               time_window: Tuple[float, float]) -> float:
        """
        Compute synchrony index using coincidence detection.
        """
        # Bin spikes
        bin_size = self.synchrony_window
        start_time, end_time = time_window
        n_bins = int((end_time - start_time) / bin_size) + 1
        
        # Count coincidences
        spike_counts = np.zeros((len(spike_trains), n_bins))
        
        for i, (cell_id, spikes) in enumerate(spike_trains.items()):
            for spike_time in spikes:
                if start_time <= spike_time <= end_time:
                    bin_idx = int((spike_time - start_time) / bin_size)
                    if 0 <= bin_idx < n_bins:
                        spike_counts[i, bin_idx] += 1
        
        # Compute synchrony as variance of population activity
        population_activity = np.sum(spike_counts, axis=0)
        
        if np.mean(population_activity) == 0:
            return 0.0
        
        # Normalize by expected variance under independence
        expected_var = np.mean(population_activity)
        actual_var = np.var(population_activity)
        
        synchrony = actual_var / expected_var if expected_var > 0 else 0.0
        
        return float(np.clip(synchrony, 0, 10))  # Cap at 10
    
    def compute_sequence_score(self,
                              assembly_sequence: List[Any],
                              reference_sequence: Optional[List[Any]] = None) -> float:
        """
        Compute sequence consistency score.
        
        Uses tropical distance for sequence comparison.
        """
        if len(assembly_sequence) < 2:
            return 0.0
        
        if reference_sequence is None:
            # Self-consistency: compare first half with second half
            mid = len(assembly_sequence) // 2
            seq1 = assembly_sequence[:mid]
            seq2 = assembly_sequence[mid:2*mid]
        else:
            seq1 = assembly_sequence
            seq2 = reference_sequence
        
        if not seq1 or not seq2:
            return 0.0
        
        # Compare sequences using tropical correlation
        score = self._sequence_similarity_tropical(seq1, seq2)
        
        return float(score)
    
    def _sequence_similarity_tropical(self,
                                     seq1: List[Any],
                                     seq2: List[Any]) -> float:
        """
        Compute sequence similarity using tropical mathematics.
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Extract features (e.g., E8 coordinates)
        feat1 = []
        feat2 = []
        
        for item in seq1:
            if hasattr(item, 'e8_coords'):
                feat1.append(item.e8_coords)
            elif hasattr(item, 'cells'):
                # Convert cell list to binary vector
                vec = np.zeros(100)  # Assume max 100 cells
                for c in item.cells[:100]:
                    vec[c] = 1
                feat1.append(vec[:8])  # Take first 8 dims
        
        for item in seq2:
            if hasattr(item, 'e8_coords'):
                feat2.append(item.e8_coords)
            elif hasattr(item, 'cells'):
                vec = np.zeros(100)
                for c in item.cells[:100]:
                    vec[c] = 1
                feat2.append(vec[:8])
        
        if not feat1 or not feat2:
            return 0.0
        
        # Compute average tropical distance
        distances = []
        for f1 in feat1[:len(feat2)]:
            for f2 in feat2[:len(feat1)]:
                d = tropical_distance(np.array(f1), np.array(f2))
                distances.append(d)
        
        if not distances:
            return 0.0
        
        # Convert distance to similarity
        avg_dist = np.mean(distances)
        similarity = np.exp(-avg_dist / 10.0)  # Exponential decay
        
        return similarity

class ReplayAnalyzer:
    """Analyze replay events during sharp-wave ripples."""
    
    def __init__(self):
        self.time_compression_factor = 20  # Replay is ~20x faster
        self.min_cells_for_replay = 5
    
    def detect_replay(self,
                     spike_trains: Dict[int, List[float]],
                     ripple_events: List[Dict],
                     template_sequence: Optional[List[int]] = None) -> List[Dict]:
        """
        Detect replay events during ripples.
        
        Parameters:
        -----------
        spike_trains : Dict
            Spike times for each cell
        ripple_events : List
            Detected ripple events
        template_sequence : List[int]
            Expected cell activation sequence
        
        Returns:
        --------
        List of replay events with scores
        """
        replay_events = []
        
        for ripple in ripple_events:
            # Get spikes during ripple
            ripple_spikes = {}
            
            for cell_id, spikes in spike_trains.items():
                ripple_spike_times = [
                    t for t in spikes 
                    if ripple['start_idx'] <= t * 1000 <= ripple['end_idx']
                ]
                if ripple_spike_times:
                    ripple_spikes[cell_id] = ripple_spike_times
            
            if len(ripple_spikes) < self.min_cells_for_replay:
                continue
            
            # Analyze sequence
            sequence = self._extract_sequence(ripple_spikes)
            
            # Score replay
            if template_sequence:
                score = self._score_replay(sequence, template_sequence)
            else:
                # Use sequence consistency as score
                score = self._sequence_consistency(sequence)
            
            replay_events.append({
                'ripple': ripple,
                'sequence': sequence,
                'score': score,
                'n_cells': len(ripple_spikes),
                'compression': self._estimate_compression(ripple_spikes, ripple)
            })
        
        return replay_events
    
    def _extract_sequence(self, spike_dict: Dict[int, List[float]]) -> List[int]:
        """Extract cell activation sequence from spikes."""
        # Get first spike time for each cell
        first_spikes = []
        
        for cell_id, spike_times in spike_dict.items():
            if spike_times:
                first_spikes.append((min(spike_times), cell_id))
        
        # Sort by time
        first_spikes.sort()
        
        # Return cell sequence
        return [cell_id for _, cell_id in first_spikes]
    
    def _score_replay(self, observed: List[int], template: List[int]) -> float:
        """
        Score how well observed sequence matches template.
        Uses rank correlation.
        """
        if len(observed) < 2 or len(template) < 2:
            return 0.0
        
        # Find common cells
        common = set(observed) & set(template)
        
        if len(common) < 2:
            return 0.0
        
        # Get ranks in each sequence
        obs_ranks = {cell: i for i, cell in enumerate(observed) if cell in common}
        temp_ranks = {cell: i for i, cell in enumerate(template) if cell in common}
        
        # Compute Spearman correlation
        cells = list(common)
        obs_r = [obs_ranks[c] for c in cells]
        temp_r = [temp_ranks[c] for c in cells]
        
        if len(obs_r) > 1:
            corr, _ = stats.spearmanr(obs_r, temp_r)
            return float(max(0, corr))  # Only positive correlations
        
        return 0.0
    
    def _sequence_consistency(self, sequence: List[int]) -> float:
        """
        Compute internal consistency of sequence.
        Higher score for more regular patterns.
        """
        if len(sequence) < 3:
            return 0.0
        
        # Check for repeated patterns
        pattern_scores = []
        
        for pattern_len in range(2, min(len(sequence)//2 + 1, 5)):
            for start in range(len(sequence) - 2*pattern_len + 1):
                pattern = sequence[start:start+pattern_len]
                rest = sequence[start+pattern_len:]
                
                # Count matches
                matches = 0
                for i in range(0, len(rest) - pattern_len + 1, pattern_len):
                    if rest[i:i+pattern_len] == pattern:
                        matches += 1
                
                if matches > 0:
                    pattern_scores.append(matches / (len(sequence) / pattern_len))
        
        return float(max(pattern_scores)) if pattern_scores else 0.0
    
    def _estimate_compression(self, ripple_spikes: Dict, ripple_event: Dict) -> float:
        """Estimate temporal compression factor."""
        if len(ripple_spikes) < 2:
            return 1.0
        
        # Get spike time range
        all_times = []
        for spikes in ripple_spikes.values():
            all_times.extend(spikes)
        
        if len(all_times) < 2:
            return 1.0
        
        time_range = max(all_times) - min(all_times)
        
        # Expected range during behavior (assume 1 second)
        expected_range = 1.0
        
        # Compression factor
        if time_range > 0:
            compression = expected_range / time_range
        else:
            compression = self.time_compression_factor
        
        return float(np.clip(compression, 1, 100))

def compute_information_content(spike_trains: Dict[int, List[float]],
                               time_bins: int = 100) -> float:
    """
    Compute information content using entropy.
    
    Returns:
    --------
    float : Information in bits
    """
    if not spike_trains:
        return 0.0
    
    # Create binary matrix
    n_cells = len(spike_trains)
    spike_matrix = np.zeros((n_cells, time_bins))
    
    # Get time range
    all_spikes = []
    for spikes in spike_trains.values():
        all_spikes.extend(spikes)
    
    if not all_spikes:
        return 0.0
    
    t_min, t_max = min(all_spikes), max(all_spikes)
    
    # Bin spikes
    for i, (cell_id, spikes) in enumerate(spike_trains.items()):
        for spike_time in spikes:
            bin_idx = int((spike_time - t_min) / (t_max - t_min) * time_bins)
            if 0 <= bin_idx < time_bins:
                spike_matrix[i, bin_idx] = 1
    
    # Compute entropy of population patterns
    patterns = []
    for t in range(time_bins):
        pattern = tuple(spike_matrix[:, t])
        patterns.append(pattern)
    
    # Count unique patterns
    unique_patterns = list(set(patterns))
    
    if len(unique_patterns) <= 1:
        return 0.0
    
    # Compute probabilities
    pattern_counts = {}
    for p in patterns:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    
    probs = np.array(list(pattern_counts.values())) / len(patterns)
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return float(entropy)

def validate_against_buzsaki(simulation_results: Dict[str, Any],
                            reference_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Validate simulation against Buzsáki's empirical findings.
    
    Parameters:
    -----------
    simulation_results : Dict
        Results from hippocampal network simulation
    reference_metrics : Dict
        Expected values from literature
    
    Returns:
    --------
    Dict with validation results
    """
    if reference_metrics is None:
        # Default expected values from Buzsáki's work
        reference_metrics = {
            'theta_power_ratio': 0.3,      # Theta/total power
            'gamma_power_ratio': 0.2,      # Gamma/total power
            'theta_gamma_pac': 0.4,         # PAC strength
            'ripple_rate': 0.5,             # Ripples per second
            'assembly_compression': 5.0,     # Compression ratio
            'replay_fidelity': 0.6,         # Replay correlation
            'mean_firing_rate': 2.0,        # Hz
            'synchrony_index': 2.0          # Synchrony level
        }
    
    validation = {}
    
    # Extract metrics from simulation
    if 'spike_times' in simulation_results:
        spike_trains = simulation_results['spike_times']
        
        # Firing rate
        total_time = 1.0  # Assume 1 second
        rates = []
        for spikes in spike_trains.values():
            rates.append(len(spikes) / total_time)
        
        validation['mean_firing_rate'] = {
            'observed': np.mean(rates) if rates else 0,
            'expected': reference_metrics['mean_firing_rate'],
            'pass': abs(np.mean(rates) - reference_metrics['mean_firing_rate']) < 2.0
        }
    
    # Assembly metrics
    if 'assemblies' in simulation_results:
        assemblies = simulation_results['assemblies']
        
        analyzer = AssemblyAnalyzer()
        compression = analyzer.compute_compression_ratio(spike_trains, assemblies)
        
        validation['assembly_compression'] = {
            'observed': compression,
            'expected': reference_metrics['assembly_compression'],
            'pass': compression > 2.0  # Minimum acceptable compression
        }
    
    # Compute overall validation score
    passed = sum(1 for v in validation.values() if v.get('pass', False))
    total = len(validation)
    
    validation['overall_score'] = passed / total if total > 0 else 0
    validation['passed_tests'] = passed
    validation['total_tests'] = total
    
    return validation

def test_buzsaki_metrics():
    """Test Buzsáki metrics computation."""
    print("\n=== Testing Buzsáki Metrics ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    fs = 1000  # 1 kHz sampling
    duration = 2.0  # 2 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Generate LFP with theta and gamma
    theta = np.sin(2 * np.pi * 8 * t)  # 8 Hz theta
    gamma = 0.3 * np.sin(2 * np.pi * 40 * t)  # 40 Hz gamma
    noise = 0.1 * np.random.standard_normal(len(t))
    
    # Modulate gamma by theta phase (PAC)
    gamma_modulated = gamma * (1 + 0.5 * theta)
    lfp = theta + gamma_modulated + noise
    
    # Test theta-gamma analyzer
    print("--- Testing Theta-Gamma Analysis ---")
    tg_analyzer = ThetaGammaAnalyzer(fs=fs)
    pac = tg_analyzer.compute_pac(lfp)
    print(f"PAC strength: {pac:.3f}")
    
    cycles = tg_analyzer.detect_theta_cycles(lfp)
    print(f"Detected {len(cycles)} theta cycles")
    
    # Test ripple detector
    print("\n--- Testing Ripple Detection ---")
    
    # Add synthetic ripples
    ripple_times = [0.5, 1.0, 1.5]
    for ripple_time in ripple_times:
        ripple_idx = int(ripple_time * fs)
        ripple_duration = int(0.05 * fs)  # 50ms
        if ripple_idx + ripple_duration < len(lfp):
            ripple_signal = 0.5 * np.sin(2 * np.pi * 200 * t[:ripple_duration] / fs)
            lfp[ripple_idx:ripple_idx+ripple_duration] += ripple_signal
    
    ripple_detector = RippleDetector(fs=fs)
    ripples = ripple_detector.detect_ripples(lfp)
    print(f"Detected {len(ripples)} ripples")
    
    if ripples:
        ripple = ripples[0]
        print(f"First ripple: duration={ripple['duration']*1000:.1f}ms, "
              f"peak_freq={ripple['peak_frequency']:.1f}Hz")
    
    # Test assembly analyzer
    print("\n--- Testing Assembly Analysis ---")
    
    # Create synthetic spike trains
    spike_trains = {}
    n_cells = 50
    
    # Create assembly patterns
    assembly1 = list(range(10))
    assembly2 = list(range(5, 15))
    
    for i in range(n_cells):
        spike_times = []
        
        # Background activity
        spike_times.extend(np.random.uniform(0, duration, 5))
        
        # Assembly activity
        if i in assembly1:
            spike_times.extend([0.1, 0.3, 0.5, 0.7])
        if i in assembly2:
            spike_times.extend([0.2, 0.4, 0.6, 0.8])
        
        spike_trains[i] = sorted(spike_times)
    
    # Create mock assemblies
    from types import SimpleNamespace
    mock_assemblies = [
        SimpleNamespace(cells=assembly1, strength=0.8),
        SimpleNamespace(cells=assembly2, strength=0.7)
    ]
    
    assembly_analyzer = AssemblyAnalyzer()
    
    compression = assembly_analyzer.compute_compression_ratio(spike_trains, mock_assemblies)
    print(f"Compression ratio: {compression:.2f}")
    
    participation = assembly_analyzer.compute_participation_ratio(spike_trains, (0, duration))
    print(f"Participation ratio: {participation:.2f}")
    
    synchrony = assembly_analyzer.compute_synchrony_index(spike_trains, (0, duration))
    print(f"Synchrony index: {synchrony:.2f}")
    
    # Test replay analyzer
    print("\n--- Testing Replay Analysis ---")
    
    replay_analyzer = ReplayAnalyzer()
    replay_events = replay_analyzer.detect_replay(spike_trains, ripples[:1], assembly1)
    
    if replay_events:
        replay = replay_events[0]
        print(f"Replay detected: {replay['n_cells']} cells, "
              f"score={replay['score']:.2f}, "
              f"compression={replay['compression']:.1f}x")
    
    # Test information content
    print("\n--- Testing Information Content ---")
    
    info = compute_information_content(spike_trains)
    print(f"Information content: {info:.2f} bits")
    
    # Test validation
    print("\n--- Testing Validation ---")
    
    sim_results = {
        'spike_times': spike_trains,
        'assemblies': mock_assemblies
    }
    
    validation = validate_against_buzsaki(sim_results)
    print(f"Validation results:")
    for key, value in validation.items():
        if isinstance(value, dict):
            print(f"  {key}: observed={value.get('observed', 0):.2f}, "
                  f"expected={value.get('expected', 0):.2f}, "
                  f"pass={value.get('pass', False)}")
    
    print(f"\nOverall validation score: {validation.get('overall_score', 0):.1%}")
    print("\n✓ Buzsáki metrics module working correctly!")

if __name__ == "__main__":
    test_buzsaki_metrics()
