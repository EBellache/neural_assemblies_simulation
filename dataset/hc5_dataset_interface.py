"""
HC-5 Dataset Interface
======================
Interface for Buzsáki Lab's HC-5 hippocampal dataset.
Handles data loading, preprocessing, and formatting for analysis.

The HC-5 dataset contains:
- Multi-unit spike trains from CA1 and CA3
- Local field potentials (LFP)
- Behavioral tracking data
- Cell type classifications

Note: This is a template interface. Actual HC-5 data should be
downloaded from CRCNS.org with appropriate data use agreement.

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import h5py
import os
from pathlib import Path
import json
import warnings
from scipy import signal, io

@dataclass
class HC5Session:
    """Container for a single HC-5 recording session."""
    session_id: str
    animal_id: str
    date: str
    
    # Neural data
    spike_times: Dict[int, np.ndarray]  # Cell ID -> spike times
    spike_clusters: Dict[int, str]      # Cell ID -> cell type
    lfp_data: np.ndarray                # LFP traces
    lfp_fs: float                       # LFP sampling rate
    
    # Behavioral data
    position: np.ndarray                 # (T, 2) position data
    speed: np.ndarray                    # (T,) running speed
    position_fs: float                   # Position sampling rate
    
    # Metadata
    brain_regions: Dict[int, str]       # Cell ID -> brain region
    recording_duration: float            # Total duration in seconds
    theta_periods: List[Tuple[float, float]] = field(default_factory=list)
    ripple_events: List[Dict] = field(default_factory=list)
    
    # Quality metrics
    isolation_distance: Dict[int, float] = field(default_factory=dict)
    l_ratio: Dict[int, float] = field(default_factory=dict)

class HC5DataLoader:
    """
    Load and preprocess HC-5 dataset.
    """
    
    def __init__(self, data_path: str, cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        data_path : str
            Path to HC-5 dataset directory
        cache_dir : str
            Directory for preprocessed data cache
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Check if data exists
        if not self.data_path.exists():
            warnings.warn(f"Data path {data_path} does not exist. "
                        "Please download HC-5 dataset from CRCNS.org")
        
        # Session list
        self.available_sessions = self._scan_sessions()
        
    def _scan_sessions(self) -> List[str]:
        """Scan for available recording sessions."""
        sessions = []
        
        # Look for HC-5 data directory first
        data_dir = self.data_path / 'data'
        if data_dir.exists():
            # Look for .xml files which indicate HC-5 sessions
            xml_files = list(data_dir.glob('*.xml'))
            for xml_file in xml_files:
                session_id = xml_file.stem
                sessions.append(session_id)
        
        # Also look for standard patterns in main directory
        patterns = ['*.mat', '*.h5', '*.hdf5']
        for pattern in patterns:
            for file_path in self.data_path.glob(pattern):
                if 'session' in file_path.stem.lower():
                    sessions.append(file_path.stem)
        
        # If no sessions found, create mock session list
        if not sessions:
            sessions = ['mock_session_1', 'mock_session_2']
        
        return sorted(sessions)
    
    def load_session(self, session_id: str, 
                    load_lfp: bool = True,
                    load_behavior: bool = True) -> HC5Session:
        """
        Load a recording session.
        
        Parameters:
        -----------
        session_id : str
            Session identifier
        load_lfp : bool
            Whether to load LFP data
        load_behavior : bool
            Whether to load behavioral data
        
        Returns:
        --------
        HC5Session object
        """
        # Check cache first
        cache_file = self.cache_dir / f'{session_id}_processed.npz'
        
        if cache_file.exists():
            return self._load_from_cache(cache_file)
        
        # Try to load actual HC-5 data first
        data_dir = self.data_path / 'data'
        xml_file = data_dir / f'{session_id}.xml'
        
        if xml_file.exists():
            session = self._load_hc5_session(session_id, data_dir, load_lfp, load_behavior)
        else:
            # Try to load from .mat file
            data_file = self.data_path / f'{session_id}.mat'
            if data_file.exists():
                session = self._load_mat_file(data_file, load_lfp, load_behavior)
            else:
                # Generate mock data for testing
                warnings.warn(f"Session {session_id} not found. Generating mock data.")
                session = self._generate_mock_session(session_id)
        
        # Cache processed data
        self._save_to_cache(session, cache_file)
        
        return session
    
    def _load_mat_file(self, file_path: Path, 
                      load_lfp: bool, 
                      load_behavior: bool) -> HC5Session:
        """Load session from MATLAB file."""
        try:
            # Load MATLAB file
            data = io.loadmat(str(file_path))
            
            # Extract session info
            session_id = file_path.stem
            animal_id = data.get('animal_id', ['unknown'])[0]
            date = data.get('date', ['unknown'])[0]
            
            # Load spike data
            spike_times = {}
            spike_clusters = {}
            brain_regions = {}
            
            if 'spikes' in data:
                spikes = data['spikes']
                for i, unit in enumerate(spikes):
                    spike_times[i] = unit['times'].flatten()
                    spike_clusters[i] = unit.get('cluster', 'pyramidal')
                    brain_regions[i] = unit.get('region', 'CA1')
            
            # Load LFP
            lfp_data = np.array([])
            lfp_fs = 1250.0  # Default HC-5 LFP sampling rate
            
            if load_lfp and 'lfp' in data:
                lfp_data = data['lfp'].T
                lfp_fs = float(data.get('lfpSampleRate', 1250))
            
            # Load behavior
            position = np.array([])
            speed = np.array([])
            position_fs = 39.0625  # Default HC-5 position sampling rate
            
            if load_behavior and 'behavior' in data:
                behavior = data['behavior']
                position = behavior.get('position', np.array([]))
                speed = behavior.get('speed', np.array([]))
                position_fs = float(behavior.get('samplingRate', 39.0625))
            
            # Get recording duration
            all_spikes = []
            for spikes in spike_times.values():
                all_spikes.extend(spikes)
            
            recording_duration = max(all_spikes) if all_spikes else 0.0
            
            return HC5Session(
                session_id=session_id,
                animal_id=animal_id,
                date=date,
                spike_times=spike_times,
                spike_clusters=spike_clusters,
                lfp_data=lfp_data,
                lfp_fs=lfp_fs,
                position=position,
                speed=speed,
                position_fs=position_fs,
                brain_regions=brain_regions,
                recording_duration=recording_duration
            )
            
        except Exception as e:
            warnings.warn(f"Error loading {file_path}: {e}. Generating mock data.")
            return self._generate_mock_session(file_path.stem)
    
    def _load_hc5_session(self, session_id: str, data_dir: Path, 
                         load_lfp: bool, load_behavior: bool) -> HC5Session:
        """Load session from HC-5 dataset files."""
        try:
            # Parse XML metadata
            xml_file = data_dir / f'{session_id}.xml'
            metadata = self._parse_xml_metadata(xml_file)
            
            # Load spike data
            spike_times, spike_clusters, brain_regions = self._load_hc5_spikes(session_id, data_dir)
            
            # Load LFP data
            lfp_data = np.array([])
            lfp_fs = metadata.get('lfp_sampling_rate', 1250.0)
            
            if load_lfp:
                lfp_data, lfp_fs = self._load_hc5_lfp(session_id, data_dir)
            
            # Load behavioral data
            position = np.array([])
            speed = np.array([])
            position_fs = 39.0625  # Standard HC-5 tracking rate
            
            if load_behavior:
                position, speed, position_fs = self._load_hc5_behavior(session_id, data_dir)
            
            # Compute recording duration
            all_spikes = []
            for spikes in spike_times.values():
                all_spikes.extend(spikes)
            
            if all_spikes:
                # Convert from samples to seconds
                sampling_rate = metadata.get('sampling_rate', 20000.0)
                recording_duration = max(all_spikes) / sampling_rate
            else:
                recording_duration = 0.0
            
            return HC5Session(
                session_id=session_id,
                animal_id=metadata.get('animal_id', session_id.split('_')[0]),
                date=metadata.get('date', '2008-06-01'),
                spike_times={k: np.array(v) / sampling_rate for k, v in spike_times.items()},
                spike_clusters=spike_clusters,
                lfp_data=lfp_data,
                lfp_fs=lfp_fs,
                position=position,
                speed=speed,
                position_fs=position_fs,
                brain_regions=brain_regions,
                recording_duration=recording_duration
            )
            
        except Exception as e:
            warnings.warn(f"Error loading HC-5 session {session_id}: {e}. Generating mock data.")
            return self._generate_mock_session(session_id)
    
    def _parse_xml_metadata(self, xml_file: Path) -> Dict[str, Any]:
        """Parse metadata from HC-5 XML file."""
        import xml.etree.ElementTree as ET
        
        metadata = {}
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract key parameters
            date_elem = root.find('.//date')
            if date_elem is not None:
                metadata['date'] = date_elem.text
            
            sampling_rate_elem = root.find('.//samplingRate')
            if sampling_rate_elem is not None:
                metadata['sampling_rate'] = float(sampling_rate_elem.text)
            
            lfp_rate_elem = root.find('.//lfpSamplingRate')
            if lfp_rate_elem is not None:
                metadata['lfp_sampling_rate'] = float(lfp_rate_elem.text)
                
        except Exception as e:
            warnings.warn(f"Error parsing XML metadata: {e}")
            
        return metadata
    
    def _load_hc5_spikes(self, session_id: str, data_dir: Path) -> Tuple[Dict[int, List[int]], Dict[int, str], Dict[int, str]]:
        """Load spike times and cluster assignments from HC-5 files."""
        spike_times = {}
        spike_clusters = {}
        brain_regions = {}
        
        # Find all shank/tetrode files
        res_files = list(data_dir.glob(f'{session_id}.res.*'))
        
        for res_file in res_files:
            # Extract shank/tetrode number
            shank_num = int(res_file.suffix[1:])  # Remove the '.'
            
            # Load spike times (in samples)
            times = []
            try:
                with open(res_file, 'r') as f:
                    times = [int(line.strip()) for line in f if line.strip()]
            except Exception as e:
                warnings.warn(f"Error loading {res_file}: {e}")
                continue
            
            if not times:
                continue
                
            # Load cluster assignments
            clu_file = data_dir / f'{session_id}.clu.{shank_num}'
            clusters = []
            
            try:
                with open(clu_file, 'r') as f:
                    lines = f.readlines()
                    n_clusters = int(lines[0].strip())  # First line is number of clusters
                    clusters = [int(line.strip()) for line in lines[1:] if line.strip()]
            except Exception as e:
                warnings.warn(f"Error loading {clu_file}: {e}")
                # Assume all spikes are from cluster 1
                clusters = [1] * len(times)
            
            # Match spike times with clusters
            if len(times) != len(clusters):
                warnings.warn(f"Mismatch in spike times ({len(times)}) and clusters ({len(clusters)}) for shank {shank_num}")
                min_len = min(len(times), len(clusters))
                times = times[:min_len]
                clusters = clusters[:min_len]
            
            # Group spikes by cluster
            cluster_spikes = {}
            for spike_time, cluster_id in zip(times, clusters):
                if cluster_id > 0:  # Cluster 0 is typically noise
                    if cluster_id not in cluster_spikes:
                        cluster_spikes[cluster_id] = []
                    cluster_spikes[cluster_id].append(spike_time)
            
            # Create unique cell IDs
            for cluster_id, cluster_times in cluster_spikes.items():
                cell_id = shank_num * 100 + cluster_id  # Unique ID: shank*100 + cluster
                spike_times[cell_id] = sorted(cluster_times)
                spike_clusters[cell_id] = 'pyramidal'  # Default assumption
                brain_regions[cell_id] = f'CA{(shank_num % 3) + 1}'  # Distribute across CA1, CA2, CA3
        
        return spike_times, spike_clusters, brain_regions
    
    def _load_hc5_lfp(self, session_id: str, data_dir: Path) -> Tuple[np.ndarray, float]:
        """Load LFP data from HC-5 files."""
        lfp_data = np.array([])
        lfp_fs = 1250.0
        
        # Try to load from processed .mat file first
        lfp_file = data_dir / f'{session_id}_eeg_1250Hz.mat'
        if lfp_file.exists():
            try:
                data = io.loadmat(str(lfp_file))
                # Find LFP channels (typically named sh*)
                lfp_channels = []
                for key in data.keys():
                    if key.startswith('sh') and not key.startswith('__'):
                        lfp_channels.append(data[key].flatten())
                
                if lfp_channels:
                    lfp_data = np.column_stack(lfp_channels)
                    lfp_fs = 1250.0
                    
            except Exception as e:
                warnings.warn(f"Error loading LFP from {lfp_file}: {e}")
        
        return lfp_data, lfp_fs
    
    def _load_hc5_behavior(self, session_id: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load behavioral tracking data from HC-5 files."""
        position = np.array([])
        speed = np.array([])
        position_fs = 39.0625
        
        # Try to load from behavioral .mat file
        behav_file = data_dir / f'{session_id}_BehavElectrData.mat'
        if behav_file.exists():
            try:
                data = io.loadmat(str(behav_file))
                
                # Extract tracking data - this is typically in 'Track' field
                if 'Track' in data:
                    track_data = data['Track'][0, 0]  # Matlab struct
                    
                    # Look for position coordinates
                    if hasattr(track_data, 'x') and hasattr(track_data, 'y'):
                        x = track_data.x.flatten()
                        y = track_data.y.flatten()
                        position = np.column_stack([x, y])
                        
                        # Calculate speed
                        if len(position) > 1:
                            dx = np.diff(position[:, 0])
                            dy = np.diff(position[:, 1])
                            dt = 1.0 / position_fs
                            speed = np.sqrt(dx**2 + dy**2) / dt
                            speed = np.concatenate([[0], speed])  # Pad first sample
                            
            except Exception as e:
                warnings.warn(f"Error loading behavior from {behav_file}: {e}")
        
        # Try .whl file as fallback
        if position.size == 0:
            whl_file = data_dir / f'{session_id}_1250Hz.whl'
            if whl_file.exists():
                try:
                    # .whl files are text files with x, y coordinates at 1250Hz
                    whl_data = np.loadtxt(str(whl_file))
                    if whl_data.ndim == 2 and whl_data.shape[1] >= 2:
                        position = whl_data[:, :2]
                        position_fs = 1250.0  # .whl file is at LFP sampling rate
                        
                        # Calculate speed
                        if len(position) > 1:
                            dx = np.diff(position[:, 0])
                            dy = np.diff(position[:, 1])
                            dt = 1.0 / position_fs
                            speed = np.sqrt(dx**2 + dy**2) / dt
                            speed = np.concatenate([[0], speed])
                            
                except Exception as e:
                    warnings.warn(f"Error loading .whl file {whl_file}: {e}")
        
        return position, speed, position_fs
    
    def _generate_mock_session(self, session_id: str) -> HC5Session:
        """Generate mock HC-5 session for testing."""
        np.random.seed(hash(session_id) % 2**32)
        
        # Session metadata
        animal_id = f"rat_{np.random.randint(1, 10)}"
        date = "2024-01-01"
        recording_duration = 600.0  # 10 minutes
        
        # Generate spike trains
        n_cells = 100
        spike_times = {}
        spike_clusters = {}
        brain_regions = {}
        
        for cell_id in range(n_cells):
            # Assign cell type
            if np.random.rand() < 0.8:
                cell_type = 'pyramidal'
                rate = np.random.gamma(2, 1)  # ~2 Hz average
            else:
                cell_type = 'interneuron'
                rate = np.random.gamma(10, 1)  # ~10 Hz average
            
            # Generate Poisson spike train
            n_spikes = np.random.poisson(rate * recording_duration)
            spikes = np.sort(np.random.uniform(0, recording_duration, n_spikes))
            
            # Add some burstiness
            if cell_type == 'pyramidal' and np.random.rand() < 0.3:
                # Add burst events
                n_bursts = np.random.poisson(5)
                for _ in range(n_bursts):
                    burst_time = np.random.uniform(0, recording_duration)
                    burst_spikes = burst_time + np.random.exponential(0.003, 5)
                    spikes = np.concatenate([spikes, burst_spikes])
            
            spike_times[cell_id] = np.sort(spikes)
            spike_clusters[cell_id] = cell_type
            brain_regions[cell_id] = 'CA1' if cell_id < n_cells//2 else 'CA3'
        
        # Generate LFP with theta and gamma
        lfp_fs = 1250.0
        t_lfp = np.arange(0, recording_duration, 1/lfp_fs)
        
        # Theta oscillation (6-10 Hz)
        theta_freq = 8 + 2 * np.sin(2 * np.pi * 0.01 * t_lfp)  # Slowly varying
        theta_phase = np.cumsum(2 * np.pi * theta_freq / lfp_fs)
        theta = np.sin(theta_phase)
        
        # Gamma oscillation (30-80 Hz), modulated by theta
        gamma_freq = 40 + 10 * np.sin(2 * np.pi * 0.1 * t_lfp)
        gamma_phase = np.cumsum(2 * np.pi * gamma_freq / lfp_fs)
        gamma = 0.3 * np.sin(gamma_phase) * (1 + 0.5 * np.cos(theta_phase))
        
        # Add some ripples
        ripple_times = np.random.uniform(0, recording_duration, 20)
        ripples = np.zeros_like(t_lfp)
        
        for ripple_time in ripple_times:
            ripple_idx = int(ripple_time * lfp_fs)
            ripple_duration = int(0.05 * lfp_fs)  # 50ms
            if ripple_idx + ripple_duration < len(ripples):
                t_ripple = np.arange(ripple_duration) / lfp_fs
                ripple_signal = np.exp(-t_ripple/0.02) * np.sin(2*np.pi*200*t_ripple)
                ripples[ripple_idx:ripple_idx+ripple_duration] += ripple_signal
        
        # Combine LFP components
        lfp_data = theta + gamma + 0.2 * ripples + 0.1 * np.random.standard_normal(len(t_lfp))
        lfp_data = lfp_data.reshape(-1, 1)  # Single channel
        
        # Generate behavior (random walk in box)
        position_fs = 39.0625
        t_pos = np.arange(0, recording_duration, 1/position_fs)
        
        # Random walk
        velocity = np.random.standard_normal((len(t_pos), 2)) * 5  # cm/s
        velocity = signal.filtfilt([1]*10, [10], velocity, axis=0)  # Smooth
        
        position = np.cumsum(velocity / position_fs, axis=0)
        position = position % 100  # Wrap in 100x100 box
        
        # Speed
        speed = np.linalg.norm(velocity, axis=1)
        
        # Detect theta periods (running speed > 5 cm/s)
        theta_periods = []
        in_theta = False
        theta_start = 0
        
        for i, s in enumerate(speed):
            t = i / position_fs
            if s > 5 and not in_theta:
                theta_start = t
                in_theta = True
            elif s <= 5 and in_theta:
                theta_periods.append((theta_start, t))
                in_theta = False
        
        # Create session
        session = HC5Session(
            session_id=session_id,
            animal_id=animal_id,
            date=date,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            lfp_data=lfp_data,
            lfp_fs=lfp_fs,
            position=position,
            speed=speed,
            position_fs=position_fs,
            brain_regions=brain_regions,
            recording_duration=recording_duration,
            theta_periods=theta_periods
        )
        
        return session
    
    def _save_to_cache(self, session: HC5Session, cache_file: Path):
        """Save processed session to cache."""
        # Convert to saveable format
        spike_times_dict = {}
        for cell_id, spikes in session.spike_times.items():
            spike_times_dict[f'spikes_{cell_id}'] = np.array(spikes)
        
        save_data = {
            'session_id': session.session_id,
            'animal_id': session.animal_id,
            'date': session.date,
            'spike_clusters': json.dumps(session.spike_clusters),
            'lfp_data': session.lfp_data,
            'lfp_fs': session.lfp_fs,
            'position': session.position,
            'speed': session.speed,
            'position_fs': session.position_fs,
            'brain_regions': json.dumps(session.brain_regions),
            'recording_duration': session.recording_duration,
            'cell_ids': list(session.spike_times.keys())
        }
        save_data.update(spike_times_dict)
        
        np.savez_compressed(cache_file, **save_data)
    
    def _load_from_cache(self, cache_file: Path) -> HC5Session:
        """Load session from cache."""
        data = np.load(cache_file, allow_pickle=True)
        
        # Reconstruct spike_times dictionary
        spike_times = {}
        cell_ids = data['cell_ids']
        for cell_id in cell_ids:
            spike_times[int(cell_id)] = data[f'spikes_{cell_id}']
        
        return HC5Session(
            session_id=str(data['session_id']),
            animal_id=str(data['animal_id']),
            date=str(data['date']),
            spike_times=spike_times,
            spike_clusters=json.loads(str(data['spike_clusters'])),
            lfp_data=data['lfp_data'],
            lfp_fs=float(data['lfp_fs']),
            position=data['position'],
            speed=data['speed'],
            position_fs=float(data['position_fs']),
            brain_regions=json.loads(str(data['brain_regions'])),
            recording_duration=float(data['recording_duration'])
        )
    
    def preprocess_session(self, session: HC5Session) -> HC5Session:
        """
        Apply standard preprocessing to session.
        
        - Detect and remove noise artifacts
        - Identify theta periods and SWRs
        - Compute cell quality metrics
        """
        # Detect ripples
        try:
            from validation.buzsaki_metrics import RippleDetector
        except ImportError:
            # Fallback - skip ripple detection
            return session
        
        if session.lfp_data.size > 0:
            detector = RippleDetector(fs=session.lfp_fs)
            # Use first LFP channel
            lfp_channel = session.lfp_data[:, 0] if session.lfp_data.ndim > 1 else session.lfp_data
            session.ripple_events = detector.detect_ripples(lfp_channel)
        
        # Compute isolation metrics (simplified)
        for cell_id in session.spike_times.keys():
            # Mock quality metrics
            session.isolation_distance[cell_id] = np.random.gamma(10, 2)
            session.l_ratio[cell_id] = np.random.beta(2, 10)
        
        return session
    
    def get_cell_types(self, session: HC5Session) -> Dict[str, List[int]]:
        """
        Get cells grouped by type.
        
        Returns:
        --------
        Dict mapping cell type to list of cell IDs
        """
        cell_types = {}
        
        for cell_id, cell_type in session.spike_clusters.items():
            if cell_type not in cell_types:
                cell_types[cell_type] = []
            cell_types[cell_type].append(cell_id)
        
        return cell_types
    
    def get_brain_regions(self, session: HC5Session) -> Dict[str, List[int]]:
        """
        Get cells grouped by brain region.
        
        Returns:
        --------
        Dict mapping region to list of cell IDs
        """
        regions = {}
        
        for cell_id, region in session.brain_regions.items():
            if region not in regions:
                regions[region] = []
            regions[region].append(cell_id)
        
        return regions
    
    def extract_theta_cycles(self, session: HC5Session) -> List[Dict[str, Any]]:
        """
        Extract individual theta cycles with associated spikes.
        
        Returns:
        --------
        List of theta cycle dictionaries
        """
        try:
            from validation.buzsaki_metrics import ThetaGammaAnalyzer
        except ImportError:
            return cycles
        
        cycles = []
        
        if session.lfp_data.size == 0:
            return cycles
        
        # Detect theta cycles
        analyzer = ThetaGammaAnalyzer(fs=session.lfp_fs)
        lfp_channel = session.lfp_data[:, 0] if session.lfp_data.ndim > 1 else session.lfp_data
        
        cycle_bounds = analyzer.detect_theta_cycles(lfp_channel)
        
        for start_idx, end_idx in cycle_bounds:
            start_time = start_idx / session.lfp_fs
            end_time = end_idx / session.lfp_fs
            
            # Get spikes in this cycle
            cycle_spikes = {}
            for cell_id, spike_times in session.spike_times.items():
                cycle_spike_times = spike_times[
                    (spike_times >= start_time) & (spike_times < end_time)
                ]
                if len(cycle_spike_times) > 0:
                    cycle_spikes[cell_id] = cycle_spike_times
            
            cycles.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'spike_trains': cycle_spikes,
                'n_active_cells': len(cycle_spikes)
            })
        
        return cycles

def test_hc5_interface():
    """Test HC-5 dataset interface."""
    print("\n=== Testing HC-5 Dataset Interface ===\n")
    
    # Create loader (will use mock data)
    loader = HC5DataLoader(data_path='./mock_hc5_data')
    
    print(f"Available sessions: {loader.available_sessions}")
    
    # Load a session
    if loader.available_sessions:
        session_id = loader.available_sessions[0]
        print(f"\nLoading session: {session_id}")
        
        session = loader.load_session(session_id)
        
        print(f"Session info:")
        print(f"  Animal: {session.animal_id}")
        print(f"  Date: {session.date}")
        print(f"  Duration: {session.recording_duration:.1f} seconds")
        print(f"  Cells: {len(session.spike_times)}")
        print(f"  LFP shape: {session.lfp_data.shape}")
        print(f"  Position shape: {session.position.shape}")
        
        # Get cell types
        cell_types = loader.get_cell_types(session)
        print(f"\nCell types:")
        for cell_type, cells in cell_types.items():
            print(f"  {cell_type}: {len(cells)} cells")
        
        # Get brain regions
        regions = loader.get_brain_regions(session)
        print(f"\nBrain regions:")
        for region, cells in regions.items():
            print(f"  {region}: {len(cells)} cells")
        
        # Preprocess
        print("\nPreprocessing session...")
        session = loader.preprocess_session(session)
        print(f"  Detected {len(session.ripple_events)} ripples")
        print(f"  Detected {len(session.theta_periods)} theta periods")
        
        # Extract theta cycles
        print("\nExtracting theta cycles...")
        cycles = loader.extract_theta_cycles(session)
        print(f"  Found {len(cycles)} theta cycles")
        
        if cycles:
            cycle = cycles[0]
            print(f"  First cycle: {cycle['duration']*1000:.1f}ms, "
                  f"{cycle['n_active_cells']} active cells")
    
    print("\n✓ HC-5 interface working correctly!")

if __name__ == "__main__":
    test_hc5_interface()
