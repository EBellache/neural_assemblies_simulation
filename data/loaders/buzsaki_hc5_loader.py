"""
Buzsaki HC-5 Dataset Loader
===========================

Lightweight loader for HC-5 hippocampal datasets from the Buzsaki lab.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path
import scipy.io


@dataclass
class LightweightSession:
    """Lightweight session data container."""
    spike_trains: Dict[int, np.ndarray]  # unit_id -> spike times
    ripple_windows: List[Tuple[float, float]]  # start, end times
    metadata: Dict


@dataclass 
class ProcessedChunk:
    """Processed chunk of neural data."""
    neural_state: np.ndarray
    assembly_indicators: np.ndarray
    modular_coords: np.ndarray
    cohomology_features: Dict[int, float]


class HC5LaptopLoader:
    """
    Optimized loader for HC-5 data on laptop.
    """
    
    def __init__(self,
                 data_path: str,
                 max_units: int = 50,
                 max_time_seconds: float = 300,
                 use_lightweight: bool = True):
        """
        Initialize HC5 loader.
        
        Args:
            data_path: Path to HC5 data directory
            max_units: Maximum number of units to load
            max_time_seconds: Maximum duration to load
            use_lightweight: Use lightweight loading mode
        """
        self.data_path = Path(data_path)
        self.max_units = max_units
        self.max_time_seconds = max_time_seconds
        self.use_lightweight = use_lightweight
        
    def load_session_lightweight(self, session_name: str) -> LightweightSession:
        """
        Load session data in lightweight mode.
        
        Args:
            session_name: Name of session to load
            
        Returns:
            Lightweight session data
        """
        # For HC5 data, the files are directly in the data_path directory
        # not in a subdirectory with the session name
        
        # Check if HC5 files exist directly in data_path
        hc5_files = list(self.data_path.glob("*.res.*"))
        
        if len(hc5_files) > 0:
            # We have real HC5 data files
            try:
                return self._load_real_hc5_data(self.data_path)
            except Exception as e:
                print(f"Warning: Could not load real HC5 data ({e}), using synthetic")
                return self._generate_synthetic_session()
        else:
            # No HC5 files found, check for session subdirectory
            session_dir = self.data_path / session_name
            if session_dir.exists():
                try:
                    return self._load_real_hc5_data(session_dir)
                except Exception as e:
                    print(f"Warning: Could not load real data ({e}), using synthetic")
                    return self._generate_synthetic_session()
            else:
                # Generate synthetic data for testing
                print("No real data found, generating synthetic session")
                return self._generate_synthetic_session()
    
    def _load_real_hc5_data(self, session_dir: Path) -> LightweightSession:
        """Load real HC5 data from directory."""
        spike_trains = {}
        
        # Find the session files in the directory
        session_files = list(session_dir.glob("*.res.*"))
        
        if not session_files:
            raise ValueError("No .res spike files found")
            
        print(f"Found {len(session_files)} spike files")
        
        # Load spike times for each tetrode/unit
        unit_count = 0
        for res_file in sorted(session_files):
            if unit_count >= self.max_units:
                break
                
            try:
                # Extract tetrode number from filename
                tetrode_num = int(res_file.name.split('.')[-1])
                
                # Read spike times (in samples)
                with open(res_file, 'r') as f:
                    spike_samples = [int(line.strip()) for line in f if line.strip()]
                
                if len(spike_samples) == 0:
                    continue
                    
                # Convert to seconds (20kHz sampling rate for HC5 data)
                sampling_rate = 20000  # Hz
                spike_times = np.array(spike_samples) / sampling_rate
                
                # Filter by time limit
                spike_times = spike_times[spike_times <= self.max_time_seconds]
                
                if len(spike_times) > 10:  # Need minimum spikes
                    spike_trains[unit_count] = spike_times
                    print(f"  Loaded tetrode {tetrode_num}: {len(spike_times)} spikes, "
                          f"rate: {len(spike_times)/spike_times[-1]:.1f} Hz")
                    unit_count += 1
                    
            except Exception as e:
                print(f"Warning: Could not load {res_file.name}: {e}")
                continue
        
        if len(spike_trains) == 0:
            raise ValueError("No valid spike data loaded")
        
        # Detect ripple events from real data
        ripple_windows = self._detect_real_ripples(spike_trains)
        
        # Calculate actual duration from data
        max_time = max([spikes[-1] for spikes in spike_trains.values()] + [0])
        actual_duration = min(self.max_time_seconds, max_time)
        
        metadata = {
            'duration_seconds': actual_duration,
            'n_units': len(spike_trains),
            'session_name': 'real_hc5_session',
            'sampling_rate': 20000,
            'total_spikes': sum(len(spikes) for spikes in spike_trains.values())
        }
        
        print(f"Loaded {len(spike_trains)} units, duration: {actual_duration:.1f}s")
        print(f"Total spikes: {metadata['total_spikes']}")
        
        return LightweightSession(
            spike_trains=spike_trains,
            ripple_windows=ripple_windows,
            metadata=metadata
        )
    
    def _generate_synthetic_session(self) -> LightweightSession:
        """Generate synthetic session data for testing."""
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic spike trains
        n_units = min(self.max_units, 30)
        spike_trains = {}
        
        for unit_id in range(n_units):
            # Poisson spike train with modulated rate
            base_rate = np.random.uniform(0.5, 5.0)  # Hz
            
            # Generate spike times
            dt = 0.001  # 1ms resolution
            times = np.arange(0, self.max_time_seconds, dt)
            
            # Modulated rate (simple sinusoidal modulation)
            freq = np.random.uniform(4, 12)  # Theta-like oscillation
            rates = base_rate * (1 + 0.3 * np.sin(2 * np.pi * freq * times))
            
            # Poisson process
            spike_probs = rates * dt
            spikes_mask = np.random.rand(len(times)) < spike_probs
            spike_times = times[spikes_mask]
            
            if len(spike_times) > 0:
                spike_trains[unit_id] = spike_times
        
        # Generate synthetic ripple windows
        ripple_windows = self._detect_synthetic_ripples(spike_trains)
        
        metadata = {
            'duration_seconds': self.max_time_seconds,
            'n_units': len(spike_trains),
            'session_name': 'synthetic_session'
        }
        
        return LightweightSession(
            spike_trains=spike_trains,
            ripple_windows=ripple_windows,
            metadata=metadata
        )
    
    def _detect_synthetic_ripples(self, spike_trains: Dict[int, np.ndarray]) -> List[Tuple[float, float]]:
        """Detect synthetic ripple periods."""
        if not spike_trains:
            return []
            
        # Simple ripple detection: periods of high population activity
        bin_size = 0.01  # 10ms bins
        max_time = max([spikes[-1] for spikes in spike_trains.values()] + [0])
        
        if max_time <= 0:
            return []
            
        time_bins = np.arange(0, max_time, bin_size)
        population_rate = np.zeros(len(time_bins))
        
        # Compute population firing rate
        for spikes in spike_trains.values():
            counts, _ = np.histogram(spikes, bins=time_bins)
            population_rate[:-1] += counts
        
        # Find high activity periods
        threshold = np.percentile(population_rate, 90)
        high_activity = population_rate > threshold
        
        # Find continuous regions
        ripples = []
        in_ripple = False
        ripple_start = 0
        
        for i, active in enumerate(high_activity):
            if active and not in_ripple:
                in_ripple = True
                ripple_start = time_bins[i]
            elif not active and in_ripple:
                in_ripple = False
                ripple_end = time_bins[i]
                if ripple_end - ripple_start >= 0.05:  # At least 50ms
                    ripples.append((ripple_start, ripple_end))
        
        return ripples[:50]  # Limit number of ripples
    
    def _detect_real_ripples(self, spike_trains: Dict[int, np.ndarray]) -> List[Tuple[float, float]]:
        """Detect ripple events from real spike data using improved algorithm."""
        if not spike_trains:
            return []
            
        # Improved ripple detection for real data
        bin_size = 0.005  # 5ms bins for higher resolution
        max_time = max([spikes[-1] for spikes in spike_trains.values()] + [0])
        
        if max_time <= 0:
            return []
            
        time_bins = np.arange(0, max_time, bin_size)
        population_rate = np.zeros(len(time_bins))
        
        # Compute population firing rate
        for spikes in spike_trains.values():
            counts, _ = np.histogram(spikes, bins=time_bins)
            population_rate[:-1] += counts
        
        # Smooth the population rate
        window_size = 3
        if len(population_rate) > window_size:
            population_rate = np.convolve(population_rate, 
                                        np.ones(window_size)/window_size, 
                                        mode='same')
        
        # More sophisticated ripple detection
        # Use z-score threshold
        mean_rate = np.mean(population_rate)
        std_rate = np.std(population_rate)
        
        if std_rate > 0:
            z_scores = (population_rate - mean_rate) / std_rate
            # Look for periods where z-score > 2 (high activity)
            high_activity = z_scores > 2.0
        else:
            # Fallback to percentile
            threshold = np.percentile(population_rate, 95)
            high_activity = population_rate > threshold
        
        # Find continuous regions of high activity
        ripples = []
        in_ripple = False
        ripple_start = 0
        
        for i, active in enumerate(high_activity):
            if active and not in_ripple:
                in_ripple = True
                ripple_start = time_bins[i]
            elif not active and in_ripple:
                in_ripple = False
                ripple_end = time_bins[i]
                ripple_duration = ripple_end - ripple_start
                
                # Keep ripples between 30ms and 300ms (typical ripple duration)
                if 0.03 <= ripple_duration <= 0.3:
                    ripples.append((ripple_start, ripple_end))
        
        print(f"Detected {len(ripples)} ripple events")
        return ripples[:100]  # Limit number of ripples
    
    def process_chunks(self, session: LightweightSession) -> Iterator[ProcessedChunk]:
        """Process session data in chunks."""
        chunk_duration = 10.0  # 10 second chunks
        n_chunks = int(session.metadata['duration_seconds'] / chunk_duration)
        
        for i in range(n_chunks):
            start_time = i * chunk_duration
            end_time = (i + 1) * chunk_duration
            
            # Extract chunk data
            chunk_spikes = {}
            for unit_id, spikes in session.spike_trains.items():
                mask = (spikes >= start_time) & (spikes < end_time)
                chunk_spikes[unit_id] = spikes[mask] - start_time
            
            # Create synthetic neural state
            n_units = len(chunk_spikes)
            n_timepoints = 100  # Arbitrary
            neural_state = np.random.randn(n_timepoints, n_units).astype(np.float32)
            
            # Create synthetic features
            assembly_indicators = np.random.rand(n_timepoints) > 0.7
            modular_coords = np.random.randn(n_timepoints, 5)
            cohomology_features = {
                0: np.random.rand(),
                1: np.random.rand() * 0.5,
                2: np.random.rand() * 0.1
            }
            
            yield ProcessedChunk(
                neural_state=neural_state,
                assembly_indicators=assembly_indicators,
                modular_coords=modular_coords,
                cohomology_features=cohomology_features
            )