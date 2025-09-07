"""I/O utilities for saving and loading models and results."""

import pickle
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional
import h5py

from ..assembly.network import TropicalAssemblyNetwork
from ..segmentation.segmenter import SegmentationResult


def save_network(network: TropicalAssemblyNetwork, 
                 filepath: str,
                 include_history: bool = True):
    """Save network to disk.
    
    Args:
        network: Network to save
        filepath: Path to save file
        include_history: Whether to include competition history
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'config': network.config._asdict(),
        'n_active': network.n_active,
        'global_metabolic_state': network.global_metabolic_state,
        'assemblies': []
    }
    
    # Save each assembly
    for assembly in network.assemblies:
        assembly_data = {
            'index': assembly.index,
            'eigenvalue': assembly.eigenvalue,
            'patterns': [p.sparse.tolist() for p in assembly.patterns],
            'pattern_weights': assembly.pattern_weights.tolist(),
            'pattern_counts': assembly.pattern_counts.tolist(),
            'state': {
                'active': assembly.state.active,
                'metabolic_energy': assembly.state.metabolic_energy,
                'phase': assembly.state.phase.tolist(),
                'recent_winner_count': assembly.state.recent_winner_count,
                'activation_history': assembly.state.activation_history.tolist()
            }
        }
        save_data['assemblies'].append(assembly_data)
        
    # Save competition history if requested
    if include_history and hasattr(network.arena, 'competition_history'):
        save_data['competition_history'] = [
            {
                'winner_idx': r.winner_idx,
                'winner_score': float(r.winner_score),
                'runner_up_idx': r.runner_up_idx,
                'runner_up_score': float(r.runner_up_score),
                'confidence': float(r.confidence),
                'all_scores': r.all_scores.tolist(),
                'active_assemblies': r.active_assemblies.tolist()
            }
            for r in network.arena.competition_history[-1000:]  # Last 1000
        ]
        
    # Save timer state
    save_data['timer_phases'] = {
        prime: {
            'current': phase.current,
            'period': phase.period
        }
        for prime, phase in network.timer.phases.items()
    }
    
    # Save as pickle for complex objects
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"Network saved to {filepath}")


def load_network(filepath: str) -> TropicalAssemblyNetwork:
    """Load network from disk.
    
    Args:
        filepath: Path to saved network
        
    Returns:
        Loaded network
    """
    from ..assembly.network import NetworkConfig
    from ..assembly.assembly import TropicalAssembly, AssemblyState
    from ..assembly.competition import CompetitionResult
    from ..core.sdr import SDR, SDRConfig
    from ..core.padic import PadicPhase
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
        
    # Recreate network
    config = NetworkConfig(**save_data['config'])
    network = TropicalAssemblyNetwork(config)
    
    network.n_active = save_data['n_active']
    network.global_metabolic_state = save_data['global_metabolic_state']
    
    # Restore assemblies
    for i, assembly_data in enumerate(save_data['assemblies']):
        assembly = network.assemblies[i]
        assembly.eigenvalue = assembly_data['eigenvalue']
        
        # Restore patterns
        sdr_config = SDRConfig()
        assembly.patterns = [
            SDR(active_indices=indices, config=sdr_config)
            for indices in assembly_data['patterns']
        ]
        assembly.pattern_weights = jnp.array(assembly_data['pattern_weights'])
        assembly.pattern_counts = jnp.array(assembly_data['pattern_counts'])
        
        # Restore state
        state_data = assembly_data['state']
        assembly.state = AssemblyState(
            active=state_data['active'],
            metabolic_energy=state_data['metabolic_energy'],
            phase=jnp.array(state_data['phase']),
            recent_winner_count=state_data['recent_winner_count'],
            activation_history=jnp.array(state_data['activation_history'])
        )
        
    # Restore competition history
    if 'competition_history' in save_data:
        network.arena.competition_history = [
            CompetitionResult(
                winner_idx=r['winner_idx'],
                winner_score=r['winner_score'],
                runner_up_idx=r['runner_up_idx'],
                runner_up_score=r['runner_up_score'],
                confidence=r['confidence'],
                all_scores=jnp.array(r['all_scores']),
                active_assemblies=jnp.array(r['active_assemblies'])
            )
            for r in save_data['competition_history']
        ]
        
    # Restore timer state
    if 'timer_phases' in save_data:
        for prime, phase_data in save_data['timer_phases'].items():
            network.timer.phases[prime] = PadicPhase(
                prime=prime,
                power=3,  # We use cubes
                current=phase_data['current'],
                period=phase_data['period']
            )
            
    print(f"Network loaded from {filepath}")
    return network


def save_segmentation_result(result: SegmentationResult,
                            filepath: str,
                            compress: bool = True):
    """Save segmentation result to HDF5.
    
    Args:
        result: Segmentation result
        filepath: Path to save file
        compress: Whether to compress arrays
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save arrays
        compression = 'gzip' if compress else None
        
        f.create_dataset('labels', data=result.labels, compression=compression)
        f.create_dataset('confidence', data=result.confidence, compression=compression)
        f.create_dataset('assembly_scores', data=result.assembly_scores)
        f.create_dataset('uncertainty_mask', data=result.uncertainty_mask, 
                        compression=compression)
        
        # Save timing as attributes
        for key, value in result.timing.items():
            f.attrs[f'timing_{key}'] = value
            
    print(f"Segmentation saved to {filepath}")


def load_segmentation_result(filepath: str) -> SegmentationResult:
    """Load segmentation result from HDF5.
    
    Args:
        filepath: Path to saved result
        
    Returns:
        Loaded segmentation result
    """
    with h5py.File(filepath, 'r') as f:
        labels = jnp.array(f['labels'][:])
        confidence = jnp.array(f['confidence'][:])
        assembly_scores = jnp.array(f['assembly_scores'][:])
        uncertainty_mask = jnp.array(f['uncertainty_mask'][:])
        
        # Load timing
        timing = {}
        for key in f.attrs:
            if key.startswith('timing_'):
                timing[key[7:]] = f.attrs[key]
                
    return SegmentationResult(
        labels=labels,
        confidence=confidence,
        assembly_scores=assembly_scores,
        uncertainty_mask=uncertainty_mask,
        timing=timing
    )


def export_to_numpy(data: Any, filepath: str):
    """Export JAX arrays to numpy format.
    
    Args:
        data: Data to export (can be nested dict/list)
        filepath: Path to save file
    """
    def convert_to_numpy(obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return np.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_numpy(item) for item in obj)
        else:
            return obj
            
    numpy_data = convert_to_numpy(data)
    np.save(filepath, numpy_data, allow_pickle=True)
    print(f"Data exported to {filepath}")


def import_from_numpy(filepath: str) -> Any:
    """Import numpy arrays as JAX arrays.
    
    Args:
        filepath: Path to numpy file
        
    Returns:
        Loaded data with JAX arrays
    """
    def convert_to_jax(obj):
        if isinstance(obj, np.ndarray):
            return jnp.array(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_jax(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_jax(item) for item in obj)
        else:
            return obj
            
    numpy_data = np.load(filepath, allow_pickle=True)
    
    # Handle case where numpy saves a 0-d array for dicts
    if isinstance(numpy_data, np.ndarray) and numpy_data.shape == ():
        numpy_data = numpy_data.item()
        
    return convert_to_jax(numpy_data)


def save_config(config: Any, filepath: str):
    """Save configuration to JSON.
    
    Args:
        config: Configuration object (NamedTuple)
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, '_asdict'):
        config_dict = config._asdict()
    else:
        config_dict = dict(config)
        
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
        
    print(f"Config saved to {filepath}")


def load_config(filepath: str, config_class: type) -> Any:
    """Load configuration from JSON.
    
    Args:
        filepath: Path to config file
        config_class: Class to instantiate
        
    Returns:
        Configuration object
    """
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
        
    return config_class(**config_dict)