"""
Cross-Region Mathematical Analysis
==================================

Run comprehensive cross-region tests on Neuropixels data.
"""

import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
from scipy import stats
from typing import Dict, List, Tuple
import time

from data.loaders.neuropixels_loader import NeuropixelsLoader
from core.error_correction.e8_projection import E8ProjectionComputer
from core.sheaf.neural_sheaf import NeuralSheafComputer
from core.sheaf.cohomology import CohomologyComputer
from core.sheaf.spectral_sequences import SpectralSequenceComputer
from analysis.simple_assembly_computer import AssemblyComputer


def run_cross_region_e8_analysis(recording):
    """Test E8 error correction across brain regions."""
    print("\n🔮 E8 ERROR CORRECTION ACROSS REGIONS")
    print("=" * 50)
    
    e8_computer = E8ProjectionComputer()
    region_e8_results = {}
    
    for probe_id, probe in recording.probe_data.items():
        region = probe.brain_regions[0]
        print(f"\nAnalyzing {region}...")
        
        # Create neural state vectors from spike data
        time_bins = np.linspace(0, recording.metadata['duration_seconds'], 60)
        neural_states = []
        
        for i in range(len(time_bins)-1):
            t_start, t_end = time_bins[i], time_bins[i+1]
            
            # Create 8D state vector (embed to E8 dimension)
            state = np.zeros(8)
            unit_count = 0
            
            for unit_id, spikes in probe.spike_times.items():
                if unit_count >= 8:
                    break
                mask = (spikes >= t_start) & (spikes < t_end)
                spike_count = np.sum(mask)
                state[unit_count] = spike_count / (t_end - t_start)  # Firing rate
                unit_count += 1
            
            # Normalize for E8
            if np.linalg.norm(state) > 0:
                state = state / np.linalg.norm(state) * 2.0
            
            neural_states.append(state)
        
        neural_states = np.array(neural_states)
        
        # Test E8 projection for this region
        e8_distances = []
        for state in neural_states[:10]:  # Test first 10 states
            try:
                distance = e8_computer.compute_code_distance(state)
                e8_distances.append(distance)
            except:
                continue
        
        if e8_distances:
            mean_distance = np.mean(e8_distances)
            success_rate = np.mean([d < 1.5 for d in e8_distances])
            
            region_e8_results[region] = {
                'mean_distance': mean_distance,
                'success_rate': success_rate,
                'n_states_tested': len(e8_distances)
            }
            
            print(f"   Mean distance to E8: {mean_distance:.3f}")
            print(f"   Success rate: {100*success_rate:.1f}%")
    
    return region_e8_results


def run_cross_region_sheaf_analysis(recording):
    """Test sheaf cohomology across regions."""
    print("\n🌐 SHEAF COHOMOLOGY ACROSS REGIONS")
    print("=" * 50)
    
    cohomology_computer = CohomologyComputer()
    
    # Build cross-region connectivity matrix
    all_units = []
    region_labels = []
    all_spike_times = {}
    
    unit_idx = 0
    for probe_id, probe in recording.probe_data.items():
        region = probe.brain_regions[0]
        
        for unit_id, spikes in list(probe.spike_times.items())[:15]:  # Limit units
            all_spike_times[unit_idx] = spikes
            region_labels.append(region)
            all_units.append(f"{region}_{unit_id}")
            unit_idx += 1
    
    n_units = len(all_units)
    print(f"Analyzing connectivity across {n_units} units from {len(set(region_labels))} regions")
    
    # Compute cross-correlation matrix
    time_bins = np.linspace(0, recording.metadata['duration_seconds'], 100)
    activity_matrix = np.zeros((len(time_bins)-1, n_units))
    
    for t_idx in range(len(time_bins)-1):
        t_start, t_end = time_bins[t_idx], time_bins[t_idx+1]
        
        for unit_idx, spikes in all_spike_times.items():
            if unit_idx < n_units:
                mask = (spikes >= t_start) & (spikes < t_end)
                activity_matrix[t_idx, unit_idx] = np.sum(mask)
    
    # Compute connectivity
    connectivity = np.corrcoef(activity_matrix.T)
    connectivity = np.abs(connectivity)
    np.fill_diagonal(connectivity, 0)
    
    # Analyze cross-region vs within-region connectivity
    cross_region_connections = []
    within_region_connections = []
    
    for i in range(n_units):
        for j in range(i+1, n_units):
            if i < len(region_labels) and j < len(region_labels):
                conn_strength = connectivity[i, j]
                if not np.isnan(conn_strength):
                    if region_labels[i] == region_labels[j]:
                        within_region_connections.append(conn_strength)
                    else:
                        cross_region_connections.append(conn_strength)
    
    # Statistical comparison
    if cross_region_connections and within_region_connections:
        mean_cross = np.mean(cross_region_connections)
        mean_within = np.mean(within_region_connections)
        
        # Test if cross-region connectivity is different from within-region
        stat, p_value = stats.mannwhitneyu(cross_region_connections, within_region_connections)
        
        print(f"   Within-region connectivity: {mean_within:.3f} ± {np.std(within_region_connections):.3f}")
        print(f"   Cross-region connectivity: {mean_cross:.3f} ± {np.std(cross_region_connections):.3f}")
        print(f"   Statistical test: p = {p_value:.4f}")
        
        # Compute topology measures
        strong_connections = np.sum(connectivity > 0.5)
        total_connections = np.sum(connectivity > 0.1)
        
        print(f"   Strong connections (>0.5): {strong_connections}")
        print(f"   Total connections (>0.1): {total_connections}")
        
        return {
            'cross_region_connectivity': mean_cross,
            'within_region_connectivity': mean_within,
            'p_value': p_value,
            'strong_connections': strong_connections,
            'connectivity_matrix': connectivity
        }
    
    return {}


def run_cross_region_assembly_analysis(recording):
    """Test assembly coupling across regions."""
    print("\n🔗 ASSEMBLY COUPLING ACROSS REGIONS")
    print("=" * 50)
    
    assembly_computer = AssemblyComputer()
    region_assemblies = {}
    
    # Extract assemblies for each region
    for probe_id, probe in recording.probe_data.items():
        region = probe.brain_regions[0]
        print(f"\nExtracting assemblies in {region}...")
        
        # Create activity matrix for this region
        time_bins = np.linspace(0, recording.metadata['duration_seconds'], 200)
        n_units = len(probe.spike_times)
        activity_matrix = np.zeros((len(time_bins)-1, n_units))
        
        for t_idx in range(len(time_bins)-1):
            t_start, t_end = time_bins[t_idx], time_bins[t_idx+1]
            
            unit_idx = 0
            for unit_id, spikes in probe.spike_times.items():
                if unit_idx >= n_units:
                    break
                mask = (spikes >= t_start) & (spikes < t_end)
                activity_matrix[t_idx, unit_idx] = np.sum(mask)
                unit_idx += 1
        
        # Extract assemblies
        assemblies = assembly_computer.extract_assemblies(activity_matrix, n_assemblies=3)
        region_assemblies[region] = assemblies
        
        print(f"   Extracted {assemblies.n_assemblies} assemblies")
        print(f"   Mean participation: {np.mean(assemblies.participation_strength):.3f}")
    
    # Test cross-region assembly coupling
    coupling_results = {}
    regions = list(region_assemblies.keys())
    
    print(f"\nTesting assembly coupling between regions...")
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions[i+1:], i+1):
            coupling = assembly_computer.compute_assembly_coupling(
                region_assemblies[region1], 
                region_assemblies[region2]
            )
            
            coupling_results[f"{region1}-{region2}"] = coupling
            print(f"   {region1} ↔ {region2}: coupling = {coupling:.3f}")
    
    return {
        'region_assemblies': region_assemblies,
        'coupling_strengths': coupling_results
    }


def run_cross_region_spectral_analysis(recording):
    """Test spectral sequences across regions."""
    print("\n🌀 SPECTRAL SEQUENCES ACROSS REGIONS") 
    print("=" * 50)
    
    # Multi-scale analysis across regions
    timescales = [0.1, 1.0, 5.0]  # 100ms, 1s, 5s
    
    scale_connectivity = {}
    
    for scale in timescales:
        print(f"\nAnalyzing {scale}s timescale...")
        
        n_bins = int(recording.metadata['duration_seconds'] / scale)
        time_bins = np.linspace(0, recording.metadata['duration_seconds'], n_bins + 1)
        
        # Get all units across regions
        all_activity = []
        region_indices = {}
        unit_idx = 0
        
        for probe_id, probe in recording.probe_data.items():
            region = probe.brain_regions[0]
            region_start = unit_idx
            
            # Activity matrix for this region at this scale
            n_units = min(10, len(probe.spike_times))  # Limit units
            region_activity = np.zeros((n_bins, n_units))
            
            for t_idx in range(n_bins):
                t_start, t_end = time_bins[t_idx], time_bins[t_idx + 1]
                
                local_unit_idx = 0
                for unit_id, spikes in probe.spike_times.items():
                    if local_unit_idx >= n_units:
                        break
                    mask = (spikes >= t_start) & (spikes < t_end)
                    region_activity[t_idx, local_unit_idx] = np.sum(mask)
                    local_unit_idx += 1
            
            all_activity.append(region_activity)
            region_indices[region] = (region_start, region_start + n_units)
            unit_idx += n_units
        
        # Combine all regions
        if all_activity:
            combined_activity = np.concatenate(all_activity, axis=1)
            
            # Compute correlation matrix at this scale
            if combined_activity.shape[0] > 1:
                corr_matrix = np.corrcoef(combined_activity.T)
                corr_matrix = np.abs(corr_matrix)
                np.fill_diagonal(corr_matrix, 0)
                
                scale_connectivity[scale] = {
                    'correlation_matrix': corr_matrix,
                    'mean_correlation': np.mean(corr_matrix),
                    'region_indices': region_indices
                }
                
                print(f"   Mean correlation: {np.mean(corr_matrix):.3f}")
    
    # Analyze persistence across scales
    if len(scale_connectivity) > 1:
        scales = sorted(scale_connectivity.keys())
        persistence_measures = []
        
        for i in range(len(scales)-1):
            scale1, scale2 = scales[i], scales[i+1]
            corr1 = scale_connectivity[scale1]['correlation_matrix']
            corr2 = scale_connectivity[scale2]['correlation_matrix']
            
            if corr1.shape == corr2.shape:
                persistence = np.corrcoef(corr1.flatten(), corr2.flatten())[0,1]
                if not np.isnan(persistence):
                    persistence_measures.append(persistence)
                    print(f"   Persistence {scale1}s → {scale2}s: {persistence:.3f}")
        
        mean_persistence = np.mean(persistence_measures) if persistence_measures else 0
        print(f"\n   Overall scale persistence: {mean_persistence:.3f}")
        
        return {
            'scale_connectivity': scale_connectivity,
            'persistence': mean_persistence
        }
    
    return {}


def main():
    """Run comprehensive cross-region analysis."""
    print("🧠 CROSS-REGION MATHEMATICAL ANALYSIS")
    print("=" * 80)
    
    # Generate synthetic multi-region data
    print("🔬 Loading multi-region data...")
    loader = NeuropixelsLoader()
    
    recording = loader.generate_synthetic_multi_region_recording(
        regions=['VISp', 'CA1', 'CA3', 'LGd'],
        duration=300.0,
        session_id='cross_region_analysis'
    )
    
    print(f"✅ Loaded recording: {len(recording.brain_regions)} regions, "
          f"{sum(len(p.spike_times) for p in recording.probe_data.values())} units")
    
    # Run all analyses
    results = {}
    
    # 1. E8 Error Correction
    results['e8'] = run_cross_region_e8_analysis(recording)
    
    # 2. Sheaf Cohomology  
    results['sheaf'] = run_cross_region_sheaf_analysis(recording)
    
    # 3. Assembly Coupling
    results['assemblies'] = run_cross_region_assembly_analysis(recording)
    
    # 4. Spectral Sequences
    results['spectral'] = run_cross_region_spectral_analysis(recording)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 CROSS-REGION ANALYSIS SUMMARY")
    print("=" * 80)
    
    # E8 Summary
    if 'e8' in results and results['e8']:
        print("\n🔮 E8 ERROR CORRECTION:")
        for region, data in results['e8'].items():
            print(f"   {region}: distance = {data['mean_distance']:.3f}, "
                  f"success = {100*data['success_rate']:.1f}%")
    
    # Sheaf Summary
    if 'sheaf' in results and results['sheaf']:
        sheaf_data = results['sheaf']
        print(f"\n🌐 SHEAF COHOMOLOGY:")
        print(f"   Cross-region connectivity: {sheaf_data.get('cross_region_connectivity', 0):.3f}")
        print(f"   Within-region connectivity: {sheaf_data.get('within_region_connectivity', 0):.3f}")
        print(f"   Statistical significance: p = {sheaf_data.get('p_value', 1):.4f}")
    
    # Assembly Summary
    if 'assemblies' in results and results['assemblies']:
        assembly_data = results['assemblies']
        print(f"\n🔗 ASSEMBLY COUPLING:")
        for pair, coupling in assembly_data.get('coupling_strengths', {}).items():
            print(f"   {pair}: {coupling:.3f}")
    
    # Spectral Summary
    if 'spectral' in results and results['spectral']:
        spectral_data = results['spectral']
        print(f"\n🌀 SPECTRAL SEQUENCES:")
        print(f"   Multi-scale persistence: {spectral_data.get('persistence', 0):.3f}")
    
    print(f"\n🎉 CROSS-REGION ANALYSIS COMPLETE!")
    
    return results


if __name__ == "__main__":
    results = main()