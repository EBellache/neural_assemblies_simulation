"""
Neuropixels Multi-Region Loader
================================

Loads Neuropixels datasets with simultaneous cortical and hippocampal recordings.
Supports Allen Institute, IBL, and Steinmetz datasets.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import h5py
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple, Any, Union
from dataclasses import dataclass
import warnings
import requests
import json
import time
from scipy import signal
import gc

# Configure JAX for CPU
import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", False)


class BrainRegion(NamedTuple):
    """Brain region information."""
    name: str
    acronym: str
    id: int
    parent: Optional[str]
    coordinates: Optional[Tuple[float, float, float]]  # AP, ML, DV


class NeuropixelsChannel(NamedTuple):
    """Single Neuropixels channel."""
    channel_id: int
    probe_id: int
    vertical_position: float  # Position on probe
    horizontal_position: float
    brain_region: BrainRegion
    unit_ids: List[int]  # Units on this channel


class MultiRegionRecording(NamedTuple):
    """Multi-region Neuropixels recording."""
    session_id: str
    probe_data: Dict[int, 'ProbeRecording']  # Probe ID -> data
    brain_regions: Dict[str, BrainRegion]  # All regions
    cross_region_pairs: List[Tuple[str, str]]  # Region pairs to analyze
    behavior_data: Optional[Dict]  # Task/behavior data
    metadata: Dict[str, Any]


class ProbeRecording(NamedTuple):
    """Single probe recording."""
    probe_id: int
    channels: List[NeuropixelsChannel]
    spike_times: Dict[int, jnp.ndarray]  # Unit ID -> spike times
    spike_clusters: jnp.ndarray
    spike_amplitudes: jnp.ndarray
    lfp_data: Optional[jnp.ndarray]  # Downsampled LFP
    lfp_timestamps: Optional[jnp.ndarray]
    brain_regions: List[str]  # Regions covered


class CrossRegionAnalysis(NamedTuple):
    """Cross-region analysis data."""
    region_pair: Tuple[str, str]
    coherence_spectrum: jnp.ndarray
    phase_coupling: jnp.ndarray
    information_flow: float
    shared_assemblies: List[jnp.ndarray]


# Dataset configurations
DATASET_CONFIGS = {
    'allen_visual_coding': {
        'base_url': 'https://visual-coding-neuropixels.s3-us-west-2.amazonaws.com',
        'manifest': 'manifest_v1.json',
        'regions': ['VISp', 'CA1', 'CA3', 'DG', 'LGd', 'LP'],
        'sampling_rate': 30000.0,
        'lfp_rate': 2500.0
    },
    'steinmetz_2019': {
        'base_url': 'https://figshare.com/ndownloader/files',
        'regions': ['VISp', 'CA1', 'PPC', 'M2', 'S1', 'RSP'],
        'sampling_rate': 30000.0,
        'lfp_rate': 2500.0
    },
    'ibl_brainwide': {
        'base_url': 'https://ibl.flatironinstitute.org',
        'regions': ['CA1', 'DG', 'VISa', 'ACA', 'MOs', 'PL', 'ILA'],
        'sampling_rate': 30000.0,
        'lfp_rate': 2500.0
    }
}


class NeuropixelsLoader:
    """
    Loader for multi-region Neuropixels datasets.
    """

    def __init__(self,
                 dataset: str = 'allen_visual_coding',
                 cache_dir: str = './cache/neuropixels',
                 max_channels: int = 100,  # Limit for laptop
                 max_duration: float = 300.0,  # 5 minutes
                 target_regions: Optional[List[str]] = None):
        """
        Initialize Neuropixels loader.

        Args:
            dataset: Which dataset to load
            cache_dir: Cache directory
            max_channels: Maximum channels to load per probe
            max_duration: Maximum recording duration
            target_regions: Specific regions to focus on
        """
        self.dataset = dataset
        self.config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['allen_visual_coding'])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_channels = max_channels
        self.max_duration = max_duration

        # Default target regions: hippocampus + cortex
        if target_regions is None:
            self.target_regions = ['CA1', 'CA3', 'DG', 'VISp', 'RSP', 'PPC']
        else:
            self.target_regions = target_regions

        # Allen CCF brain region hierarchy
        self._load_brain_regions()

    def _load_brain_regions(self):
        """Load Allen CCF brain region ontology."""
        self.brain_regions = {}

        # Simplified hierarchy for key regions
        regions_data = {
            'CA1': BrainRegion('Hippocampus CA1', 'CA1', 382, 'HPF', (-2.5, 2.0, -2.0)),
            'CA3': BrainRegion('Hippocampus CA3', 'CA3', 463, 'HPF', (-2.0, 2.5, -2.0)),
            'DG': BrainRegion('Dentate Gyrus', 'DG', 726, 'HPF', (-1.8, 2.3, -2.0)),
            'VISp': BrainRegion('Primary Visual Cortex', 'VISp', 385, 'VIS', (-3.5, 2.5, -0.5)),
            'RSP': BrainRegion('Retrosplenial Cortex', 'RSP', 879, 'CTX', (-1.5, 0.5, -0.5)),
            'PPC': BrainRegion('Posterior Parietal Cortex', 'PPC', 425, 'CTX', (-2.0, 1.5, -0.5)),
            'M2': BrainRegion('Secondary Motor Cortex', 'M2', 993, 'MO', (1.0, 1.0, -0.5)),
            'S1': BrainRegion('Primary Somatosensory', 'S1', 453, 'SS', (0.0, 2.5, -0.5)),
            'ACA': BrainRegion('Anterior Cingulate', 'ACA', 31, 'CTX', (0.5, 0.5, -0.5)),
            'LGd': BrainRegion('Lateral Geniculate', 'LGd', 496, 'TH', (-2.0, 2.0, -2.5)),
            'LP': BrainRegion('Lateral Posterior', 'LP', 218, 'TH', (-1.8, 1.5, -2.5)),
        }

        for key, region in regions_data.items():
            self.brain_regions[key] = region

    def load_allen_visual_coding_session(self, session_id: str) -> MultiRegionRecording:
        """
        Load Allen Visual Coding Neuropixels session.

        Args:
            session_id: Session identifier

        Returns:
            Multi-region recording
        """
        cache_file = self.cache_dir / f"allen_vc_{session_id}.npz"

        if cache_file.exists():
            return self._load_cached_recording(cache_file)

        print(f"Loading Allen Visual Coding session {session_id}")

        # Download metadata
        metadata = self._download_allen_metadata(session_id)

        # Load spike data
        probe_recordings = {}

        for probe_id in metadata.get('probes', []):
            probe_data = self._load_allen_probe_data(session_id, probe_id)
            if probe_data:
                probe_recordings[probe_id] = probe_data

        # Identify cross-region pairs
        all_regions = set()
        for probe in probe_recordings.values():
            all_regions.update(probe.brain_regions)

        # Find hippocampus-cortex pairs
        cross_region_pairs = []
        hippo_regions = {'CA1', 'CA3', 'DG'}
        cortex_regions = {'VISp', 'RSP', 'PPC', 'M2', 'S1'}

        for h_region in hippo_regions & all_regions:
            for c_region in cortex_regions & all_regions:
                cross_region_pairs.append((h_region, c_region))

        # Create final recording
        recording = MultiRegionRecording(
            session_id=session_id,
            duration_seconds=metadata.get('duration', self.max_duration),
            brain_regions=list(all_regions),
            probe_recordings=probe_recordings,
            cross_region_pairs=cross_region_pairs,
            stimulus_epochs=metadata.get('stimulus_epochs', []),
            metadata=metadata
        )

        # Cache the recording
        self._cache_recording(recording, cache_file)

        return recording

    def generate_synthetic_multi_region_recording(self, 
                                                regions: List[str] = None,
                                                duration: float = 300.0,
                                                session_id: str = "synthetic_001") -> MultiRegionRecording:
        """
        Generate synthetic multi-region recording for testing.
        
        Args:
            regions: Brain regions to simulate
            duration: Duration in seconds
            session_id: Session identifier
            
        Returns:
            Synthetic multi-region recording
        """
        if regions is None:
            regions = ['VISp', 'CA1', 'CA3', 'LGd']  # Mix of cortex, hippocampus, thalamus
            
        print(f"🧪 Generating synthetic multi-region recording")
        print(f"   Regions: {regions}")
        print(f"   Duration: {duration}s")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        probe_recordings = {}
        all_cross_region_pairs = []
        
        # Generate data for each region
        for region_idx, region in enumerate(regions):
            probe_id = f"probe_{region_idx}"
            
            # Region-specific parameters
            region_params = self._get_region_parameters(region)
            
            # Generate units for this region
            n_units = region_params['n_units']
            units_data = {}
            spike_times_data = {}
            
            print(f"   Generating {region} ({n_units} units)...")
            
            for unit_idx in range(n_units):
                unit_id = f"{region}_{unit_idx}"
                
                # Generate spike train with region-specific characteristics
                spike_times = self._generate_region_specific_spikes(
                    region, duration, unit_idx, region_params
                )
                
                units_data[unit_id] = {
                    'region': region,
                    'layer': region_params.get('layer', None),
                    'cell_type': region_params['cell_types'][unit_idx % len(region_params['cell_types'])],
                    'quality': 'good',
                    'amplitude': np.random.uniform(50, 200),
                    'coordinates': self._get_unit_coordinates(region, unit_idx)
                }
                
                spike_times_data[unit_id] = spike_times
            
            # Create channels for this probe
            channels = []
            for ch_idx in range(min(self.max_channels, len(units_data))):
                channel = NeuropixelsChannel(
                    channel_id=ch_idx,
                    probe_id=probe_id,
                    vertical_position=ch_idx * 20.0,  # 20μm spacing
                    horizontal_position=0.0,
                    brain_region=self.brain_regions[region],
                    unit_ids=[f"{region}_{ch_idx}"] if ch_idx < n_units else []
                )
                channels.append(channel)
            
            # Create probe recording
            probe_recording = ProbeRecording(
                probe_id=int(region_idx),
                channels=channels,
                spike_times={int(k.split('_')[1]): v for k, v in spike_times_data.items()},
                spike_clusters=np.arange(len(units_data)),
                spike_amplitudes=np.random.uniform(50, 200, len(units_data)),
                lfp_data=None,  # No LFP for synthetic data
                lfp_timestamps=None,
                brain_regions=[region]
            )
            
            probe_recordings[probe_id] = probe_recording
        
        # Generate cross-region pairs
        hippo_regions = {'CA1', 'CA3', 'DG'}
        cortex_regions = {'VISp', 'VISl', 'VISal', 'VISam', 'VISpm', 'RSP', 'PPC'}
        thalamus_regions = {'LGd', 'LP'}
        
        regions_set = set(regions)
        
        # Hippocampus-Cortex pairs
        for h_region in hippo_regions & regions_set:
            for c_region in cortex_regions & regions_set:
                all_cross_region_pairs.append((h_region, c_region))
        
        # Thalamus-Cortex pairs
        for t_region in thalamus_regions & regions_set:
            for c_region in cortex_regions & regions_set:
                all_cross_region_pairs.append((t_region, c_region))
        
        # Create final recording
        metadata = {
            'synthetic': True,
            'generated_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_units': sum(len(p.spike_times) for p in probe_recordings.values()),
            'region_types': {
                'hippocampus': list(hippo_regions & regions_set),
                'cortex': list(cortex_regions & regions_set),
                'thalamus': list(thalamus_regions & regions_set)
            }
        }
        
        # Create brain regions dict
        brain_regions_dict = {region: self.brain_regions[region] for region in regions}
        
        recording = MultiRegionRecording(
            session_id=session_id,
            probe_data=probe_recordings,
            brain_regions=brain_regions_dict,
            cross_region_pairs=all_cross_region_pairs,
            behavior_data=None,  # No behavior data for synthetic
            metadata={**metadata, 'duration_seconds': duration}
        )
        
        print(f"✅ Generated synthetic recording:")
        print(f"   Total units: {metadata['total_units']}")
        print(f"   Cross-region pairs: {len(all_cross_region_pairs)}")
        
        return recording

    def _get_region_parameters(self, region: str) -> Dict:
        """Get region-specific parameters for synthetic data generation."""
        
        # Default parameters
        base_params = {
            'n_units': 20,
            'base_rate': 5.0,  # Hz
            'burst_prob': 0.1,
            'cell_types': ['pyramidal', 'interneuron'],
            'layer': None
        }
        
        # Region-specific modifications
        if region in ['VISp', 'VISl', 'VISal', 'VISam', 'VISpm']:  # Visual cortex
            return {
                **base_params,
                'n_units': 30,
                'base_rate': 8.0,
                'burst_prob': 0.15,
                'cell_types': ['L2/3_pyramidal', 'L4_spiny', 'L5_pyramidal', 'L6_pyramidal', 'interneuron'],
                'layer_structure': True,
                'oscillations': [8, 30],  # Alpha, gamma
            }
        elif region in ['CA1', 'CA3']:  # Hippocampus
            return {
                **base_params,
                'n_units': 25,
                'base_rate': 3.0,
                'burst_prob': 0.2,
                'cell_types': ['pyramidal', 'basket', 'oriens'],
                'theta_modulation': True,
                'ripple_events': True,
            }
        elif region == 'DG':  # Dentate gyrus
            return {
                **base_params,
                'n_units': 15,
                'base_rate': 1.0,
                'burst_prob': 0.05,
                'cell_types': ['granule', 'mossy'],
                'sparse_coding': True,
            }
        elif region in ['LGd', 'LP']:  # Thalamus
            return {
                **base_params,
                'n_units': 20,
                'base_rate': 10.0,
                'burst_prob': 0.3,
                'cell_types': ['relay', 'interneuron'],
                'relay_dynamics': True,
            }
        else:
            return base_params

    def _generate_region_specific_spikes(self, region: str, duration: float, 
                                       unit_idx: int, params: Dict) -> np.ndarray:
        """Generate realistic spike trains for specific brain regions."""
        
        dt = 0.001  # 1ms resolution
        times = np.arange(0, duration, dt)
        n_times = len(times)
        
        # Base Poisson process
        base_rate = params['base_rate']
        spike_prob = base_rate * dt
        
        # Add region-specific modulations
        rate_modulation = np.ones(n_times)
        
        # Visual cortex: stimulus-driven responses
        if region.startswith('VIS'):
            # Add periodic visual stimulation responses
            stim_freq = 0.1  # 10s stimulus periods
            stim_response = 1 + 0.5 * np.sin(2 * np.pi * stim_freq * times)
            rate_modulation *= stim_response
            
            # Add gamma oscillations
            gamma_freq = 30 + np.random.uniform(-5, 5)  # 25-35 Hz
            gamma_mod = 1 + 0.2 * np.sin(2 * np.pi * gamma_freq * times)
            rate_modulation *= gamma_mod
            
        # Hippocampus: theta rhythm and ripples
        elif region in ['CA1', 'CA3']:
            # Theta modulation (6-10 Hz)
            theta_freq = 8 + np.random.uniform(-2, 2)
            theta_mod = 1 + 0.3 * np.sin(2 * np.pi * theta_freq * times)
            rate_modulation *= theta_mod
            
            # Add sharp wave ripples (rare, high-frequency events)
            n_ripples = int(duration / 10)  # ~1 ripple per 10s
            for _ in range(n_ripples):
                ripple_time = np.random.uniform(0, duration)
                ripple_idx = int(ripple_time / dt)
                ripple_width = int(0.1 / dt)  # 100ms ripples
                
                if ripple_idx + ripple_width < n_times:
                    ripple_envelope = np.exp(-((np.arange(ripple_width) - ripple_width/2) / (ripple_width/6))**2)
                    rate_modulation[ripple_idx:ripple_idx+ripple_width] *= (1 + 2 * ripple_envelope)
        
        # Thalamus: relay bursts
        elif region in ['LGd', 'LP']:
            # Add burst epochs
            burst_freq = 0.05  # Low frequency bursting
            burst_mod = 1 + 0.8 * (np.sin(2 * np.pi * burst_freq * times) > 0.5)
            rate_modulation *= burst_mod
        
        # Generate spikes
        instantaneous_rates = base_rate * rate_modulation
        spike_probs = instantaneous_rates * dt
        spikes_binary = np.random.rand(n_times) < spike_probs
        
        # Convert to spike times
        spike_times = times[spikes_binary]
        
        # Add some refractory period
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            valid_spikes = np.concatenate([[True], isis > 0.002])  # 2ms refractory
            spike_times = spike_times[valid_spikes]
        
        return spike_times

    def _get_unit_coordinates(self, region: str, unit_idx: int) -> Tuple[float, float, float]:
        """Get anatomical coordinates for a unit."""
        if region in self.brain_regions:
            base_coords = self.brain_regions[region].coordinates
            if base_coords:
                # Add some jitter around the base coordinates
                jitter = np.random.uniform(-0.1, 0.1, 3)
                return tuple(np.array(base_coords) + jitter)
        
        return (0.0, 0.0, 0.0)  # Default coordinates
        behavior_data = self._load_allen_behavior(session_id)

        recording = MultiRegionRecording(
            session_id=session_id,
            probe_data=probe_recordings,
            brain_regions={r: self.brain_regions.get(r,
                                                     BrainRegion(r, r, 0, None, None))
                           for r in all_regions},
            cross_region_pairs=cross_region_pairs,
            behavior_data=behavior_data,
            metadata=metadata
        )

        # Cache for future use
        self._cache_recording(recording, cache_file)

        return recording

    def load_steinmetz_session(self, session_path: str) -> MultiRegionRecording:
        """
        Load Steinmetz et al. 2019 dataset session.

        Args:
            session_path: Path to session data

        Returns:
            Multi-region recording
        """
        session_path = Path(session_path)
        session_id = session_path.name

        cache_file = self.cache_dir / f"steinmetz_{session_id}.npz"
        if cache_file.exists():
            return self._load_cached_recording(cache_file)

        print(f"Loading Steinmetz session from {session_path}")

        # Load spike data
        spikes_file = session_path / 'spikes.times.npy'
        clusters_file = session_path / 'spikes.clusters.npy'

        if not spikes_file.exists():
            raise FileNotFoundError(f"Spike data not found at {spikes_file}")

        spike_times = np.load(spikes_file).astype(np.float32)
        spike_clusters = np.load(clusters_file)

        # Load cluster info
        clusters_info = pd.read_csv(session_path / 'clusters.info.csv')

        # Load channel positions and brain regions
        channels_file = session_path / 'channels.brainLocation.tsv'
        if channels_file.exists():
            channels_df = pd.read_csv(channels_file, sep='\t')
        else:
            # Use default regions
            channels_df = pd.DataFrame({
                'brain_region': ['CA1'] * 50 + ['VISp'] * 50
            })

        # Group by probe (simplified: assume single probe)
        probe_recordings = {}

        # Create channels
        channels = []
        unique_regions = set()

        for idx, row in channels_df.iterrows():
            if idx >= self.max_channels:
                break

            region_name = row.get('brain_region', 'unknown')
            unique_regions.add(region_name)

            channel = NeuropixelsChannel(
                channel_id=idx,
                probe_id=0,
                vertical_position=idx * 20.0,  # 20 um spacing
                horizontal_position=0.0,
                brain_region=self.brain_regions.get(region_name,
                                                    BrainRegion(region_name, region_name, idx, None, None)),
                unit_ids=[]
            )
            channels.append(channel)

        # Process spikes
        spike_trains = {}
        unique_clusters = np.unique(spike_clusters)[:100]  # Limit units

        for cluster_id in unique_clusters:
            mask = spike_clusters == cluster_id
            cluster_spikes = spike_times[mask]

            # Limit duration
            cluster_spikes = cluster_spikes[cluster_spikes < self.max_duration]
            spike_trains[int(cluster_id)] = jnp.array(cluster_spikes)

        probe_data = ProbeRecording(
            probe_id=0,
            channels=channels,
            spike_times=spike_trains,
            spike_clusters=jnp.array(spike_clusters[:len(spike_times)]),
            spike_amplitudes=jnp.ones(len(spike_times)),  # Placeholder
            lfp_data=None,
            lfp_timestamps=None,
            brain_regions=list(unique_regions)
        )

        probe_recordings[0] = probe_data

        # Cross-region pairs
        cross_region_pairs = []
        if 'CA1' in unique_regions and 'VISp' in unique_regions:
            cross_region_pairs.append(('CA1', 'VISp'))

        # Load behavior (stimulus info)
        behavior_data = self._load_steinmetz_behavior(session_path)

        recording = MultiRegionRecording(
            session_id=session_id,
            probe_data=probe_recordings,
            brain_regions={r: self.brain_regions.get(r,
                                                     BrainRegion(r, r, 0, None, None))
                           for r in unique_regions},
            cross_region_pairs=cross_region_pairs,
            behavior_data=behavior_data,
            metadata={'dataset': 'steinmetz_2019'}
        )

        self._cache_recording(recording, cache_file)

        return recording

    def load_ibl_session(self, eid: str) -> MultiRegionRecording:
        """
        Load IBL Brain-wide Map session.

        Args:
            eid: Experiment ID

        Returns:
            Multi-region recording
        """
        from one.api import ONE  # IBL data access

        cache_file = self.cache_dir / f"ibl_{eid}.npz"
        if cache_file.exists():
            return self._load_cached_recording(cache_file)

        print(f"Loading IBL session {eid}")

        # Initialize ONE API
        one = ONE(base_url='https://openalyx.internationalbrainlab.org',
                  password='international', silent=True)

        # Load spike data
        spikes = one.load_object(eid, 'spikes')
        clusters = one.load_object(eid, 'clusters')
        channels = one.load_object(eid, 'channels')

        # Process brain regions
        unique_regions = set()
        if 'brainLocation' in channels:
            brain_locations = channels['brainLocation']

            for location in brain_locations['acronyms']:
                if location:
                    unique_regions.add(location)

        # Create probe recording
        probe_recordings = {}

        # Process spikes by probe
        if 'probe' in spikes:
            probes = np.unique(spikes['probe'])
        else:
            probes = [0]

        for probe_id in probes[:2]:  # Limit to 2 probes for laptop
            if 'probe' in spikes:
                probe_mask = spikes['probe'] == probe_id
                probe_spike_times = spikes['times'][probe_mask]
                probe_spike_clusters = spikes['clusters'][probe_mask]
            else:
                probe_spike_times = spikes['times']
                probe_spike_clusters = spikes['clusters']

            # Limit duration
            time_mask = probe_spike_times < self.max_duration
            probe_spike_times = probe_spike_times[time_mask]
            probe_spike_clusters = probe_spike_clusters[time_mask]

            # Group by cluster
            spike_trains = {}
            unique_clusters = np.unique(probe_spike_clusters)[:50]  # Limit

            for cluster_id in unique_clusters:
                mask = probe_spike_clusters == cluster_id
                spike_trains[int(cluster_id)] = jnp.array(
                    probe_spike_times[mask].astype(np.float32)
                )

            # Create channels (simplified)
            n_channels = min(self.max_channels,
                             len(channels['x']) if 'x' in channels else 100)

            channel_list = []
            for i in range(n_channels):
                region = 'CA1' if i < n_channels // 2 else 'VISp'  # Simplified

                channel = NeuropixelsChannel(
                    channel_id=i,
                    probe_id=int(probe_id),
                    vertical_position=i * 20.0,
                    horizontal_position=0.0,
                    brain_region=self.brain_regions.get(region),
                    unit_ids=[]
                )
                channel_list.append(channel)

            probe_data = ProbeRecording(
                probe_id=int(probe_id),
                channels=channel_list,
                spike_times=spike_trains,
                spike_clusters=jnp.array(probe_spike_clusters),
                spike_amplitudes=jnp.ones(len(probe_spike_times)),
                lfp_data=None,
                lfp_timestamps=None,
                brain_regions=list(unique_regions)
            )

            probe_recordings[int(probe_id)] = probe_data

        # Load behavior
        behavior_data = self._load_ibl_behavior(one, eid)

        # Cross-region pairs
        cross_region_pairs = self._identify_cross_region_pairs(unique_regions)

        recording = MultiRegionRecording(
            session_id=eid,
            probe_data=probe_recordings,
            brain_regions={r: self.brain_regions.get(r,
                                                     BrainRegion(r, r, 0, None, None))
                           for r in unique_regions},
            cross_region_pairs=cross_region_pairs,
            behavior_data=behavior_data,
            metadata={'dataset': 'ibl_brainwide'}
        )

        self._cache_recording(recording, cache_file)

        return recording

    def _download_allen_metadata(self, session_id: str) -> Dict:
        """Download Allen session metadata."""
        # Simplified metadata
        return {
            'session_id': session_id,
            'probes': [0, 1],  # Assume 2 probes
            'stimulus': 'drifting_gratings',
            'mouse_id': 'example'
        }

    def _load_allen_probe_data(self, session_id: str, probe_id: int) -> Optional[ProbeRecording]:
        """Load Allen probe data."""
        # This would download from Allen S3 bucket
        # Simplified for example

        # Generate synthetic multi-region data
        n_units = 50
        duration = min(300, self.max_duration)

        spike_trains = {}
        for i in range(n_units):
            rate = np.random.gamma(2, 2)
            n_spikes = int(rate * duration)
            spike_trains[i] = jnp.sort(jnp.array(
                np.random.uniform(0, duration, n_spikes), dtype=np.float32
            ))

        # Assign channels to regions
        channels = []
        regions = ['CA1'] * 25 + ['VISp'] * 25  # Half hippocampus, half cortex

        for i in range(50):
            channel = NeuropixelsChannel(
                channel_id=i,
                probe_id=probe_id,
                vertical_position=i * 20.0,
                horizontal_position=0.0,
                brain_region=self.brain_regions[regions[i]],
                unit_ids=[i]
            )
            channels.append(channel)

        return ProbeRecording(
            probe_id=probe_id,
            channels=channels,
            spike_times=spike_trains,
            spike_clusters=jnp.arange(n_units),
            spike_amplitudes=jnp.ones(sum(len(s) for s in spike_trains.values())),
            lfp_data=None,
            lfp_timestamps=None,
            brain_regions=['CA1', 'VISp']
        )

    def _load_allen_behavior(self, session_id: str) -> Optional[Dict]:
        """Load Allen behavior data."""
        return {
            'stimulus_presentations': pd.DataFrame({
                'start_time': np.arange(0, 100, 2),
                'stop_time': np.arange(2, 102, 2),
                'stimulus_name': ['drifting_gratings'] * 50,
                'orientation': np.random.choice([0, 45, 90, 135], 50)
            }),
            'running_speed': np.random.randn(1000) * 5 + 10,
            'eye_tracking': None
        }

    def _load_steinmetz_behavior(self, session_path: Path) -> Optional[Dict]:
        """Load Steinmetz behavior data."""
        behavior = {}

        # Load trial data
        trials_file = session_path / 'trials.intervals.npy'
        if trials_file.exists():
            trials = np.load(trials_file)
            behavior['trial_intervals'] = trials

        # Load stimulus
        contrast_file = session_path / 'trials.visualStim_contrastLeft.npy'
        if contrast_file.exists():
            behavior['contrast_left'] = np.load(contrast_file)
            behavior['contrast_right'] = np.load(
                session_path / 'trials.visualStim_contrastRight.npy'
            )

        # Load choice
        choice_file = session_path / 'trials.response_choice.npy'
        if choice_file.exists():
            behavior['choices'] = np.load(choice_file)

        return behavior if behavior else None

    def _load_ibl_behavior(self, one, eid: str) -> Optional[Dict]:
        """Load IBL behavior data."""
        try:
            trials = one.load_object(eid, 'trials')

            return {
                'trial_start': trials.get('intervals', [[]])[0],
                'trial_end': trials.get('intervals', [[]])[1],
                'choice': trials.get('choice'),
                'reward': trials.get('feedbackType'),
                'contrast_left': trials.get('contrastLeft'),
                'contrast_right': trials.get('contrastRight')
            }
        except:
            return None

    def _identify_cross_region_pairs(self, regions: set) -> List[Tuple[str, str]]:
        """Identify hippocampus-cortex region pairs."""
        pairs = []

        hippo = {'CA1', 'CA3', 'DG', 'SUB'}
        cortex = {'VISp', 'VISa', 'RSP', 'PPC', 'M1', 'M2', 'S1', 'ACA', 'PL'}

        for h in hippo & regions:
            for c in cortex & regions:
                pairs.append((h, c))

        # Also add some cortex-cortex pairs
        cortex_list = list(cortex & regions)
        for i in range(len(cortex_list)):
            for j in range(i + 1, min(i + 2, len(cortex_list))):
                pairs.append((cortex_list[i], cortex_list[j]))

        return pairs[:10]  # Limit pairs for computational efficiency

    def _cache_recording(self, recording: MultiRegionRecording, cache_file: Path):
        """Cache recording for faster loading."""
        # Convert to serializable format
        data = {
            'session_id': recording.session_id,
            'metadata': recording.metadata,
            'brain_regions': list(recording.brain_regions.keys()),
            'cross_region_pairs': recording.cross_region_pairs
        }

        # Save probe data
        for probe_id, probe in recording.probe_data.items():
            probe_key = f'probe_{probe_id}'
            data[f'{probe_key}_regions'] = probe.brain_regions

            # Save spike trains
            spike_arrays = []
            spike_ids = []
            for unit_id, spikes in probe.spike_times.items():
                spike_arrays.append(np.array(spikes))
                spike_ids.append(unit_id)

            if spike_arrays:
                data[f'{probe_key}_spikes'] = spike_arrays
                data[f'{probe_key}_spike_ids'] = spike_ids

        np.savez_compressed(cache_file, **data)
        print(f"Cached to {cache_file}")

    def _load_cached_recording(self, cache_file: Path) -> MultiRegionRecording:
        """Load cached recording."""
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)

        # Reconstruct probe data
        probe_data = {}

        for key in data.files:
            if key.startswith('probe_') and key.endswith('_regions'):
                probe_id = int(key.split('_')[1])

                # Load spike trains
                spike_trains = {}
                if f'probe_{probe_id}_spikes' in data:
                    spikes = data[f'probe_{probe_id}_spikes']
                    spike_ids = data[f'probe_{probe_id}_spike_ids']

                    for unit_id, spike_array in zip(spike_ids, spikes):
                        spike_trains[int(unit_id)] = jnp.array(spike_array)

                # Create simplified probe recording
                probe = ProbeRecording(
                    probe_id=probe_id,
                    channels=[],
                    spike_times=spike_trains,
                    spike_clusters=jnp.array([]),
                    spike_amplitudes=jnp.array([]),
                    lfp_data=None,
                    lfp_timestamps=None,
                    brain_regions=list(data[f'probe_{probe_id}_regions'])
                )

                probe_data[probe_id] = probe

        return MultiRegionRecording(
            session_id=str(data['session_id']),
            probe_data=probe_data,
            brain_regions={r: self.brain_regions.get(r,
                                                     BrainRegion(r, r, 0, None, None))
                           for r in data['brain_regions']},
            cross_region_pairs=list(map(tuple, data['cross_region_pairs'])),
            behavior_data=None,
            metadata=data['metadata'].item() if data['metadata'].size == 1 else {}
        )