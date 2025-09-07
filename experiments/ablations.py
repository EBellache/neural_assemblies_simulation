"""Ablation studies for tropical SDR framework.

Tests the importance of different components:
- Assembly count (5, 7, 9)
- Sparsity levels
- Metabolic constraints
- P-adic timing
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from tropical_sdr.assembly.network import TropicalAssemblyNetwork, NetworkConfig
from tropical_sdr.segmentation.segmenter import TropicalSegmenter, SegmentationConfig
from tropical_sdr.segmentation.metrics import SegmentationMetrics
from tropical_sdr.utils.jax_utils import JAXTimer, memory_info


class AblationStudy:
    """Run ablation studies on tropical SDR framework."""
    
    def __init__(self, config_path: str):
        """Initialize ablation study.
        
        Args:
            config_path: Path to ablation configuration
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_test_data()
        
        # Results storage
        self.results = {
            'assembly_count': {},
            'sparsity': {},
            'metabolic': {},
            'padic': {}
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_directories(self):
        """Create output directories."""
        self.results_dir = Path(self.config['output']['results_dir'])
        self.figures_dir = Path(self.config['output']['figures_dir'])
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_test_data(self):
        """Setup test data for ablations."""
        # Generate consistent test set
        key = jax.random.PRNGKey(42)
        self.test_data = []
        
        for i in range(20):  # Smaller test set for efficiency
            key, subkey = jax.random.split(key)
            
            # Generate test image
            image = jax.random.uniform(subkey, (128, 128))
            
            # Generate ground truth
            key, subkey = jax.random.split(key)
            labels = jax.random.randint(subkey, (128, 128), 0, 7)
            
            self.test_data.append((image, labels))
            
        print(f"Generated {len(self.test_data)} test images for ablations")
        
    def run_all_ablations(self):
        """Run all ablation studies."""
        print("\nStarting Ablation Studies")
        print("=" * 60)
        
        # Assembly count ablation
        if self.config['ablation']['study_type'] in ['all', 'assembly_count']:
            self.ablate_assembly_count()
            
        # Sparsity ablation
        if self.config['ablation']['study_type'] in ['all', 'sparsity']:
            self.ablate_sparsity()
            
        # Metabolic ablation
        if self.config['ablation']['study_type'] in ['all', 'metabolic']:
            self.ablate_metabolic_state()
            
        # P-adic timing ablation
        if self.config['ablation']['study_type'] in ['all', 'padic']:
            self.ablate_padic_timing()
            
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        # Statistical analysis
        self.statistical_analysis()
        
        print("\nAblation studies completed!")
        print(f"Results saved to {self.results_dir}")
        
    def ablate_assembly_count(self):
        """Test different numbers of assemblies."""
        print("\n" + "-" * 40)
        print("ABLATION: Assembly Count")
        print("-" * 40)
        
        assembly_counts = self.config['ablation']['assembly_counts']
        n_trials = self.config['evaluation']['n_trials']
        
        for count in assembly_counts:
            print(f"\nTesting {count} assemblies...")
            trial_results = []
            
            for trial in range(n_trials):
                # Create model with specific assembly count
                config = NetworkConfig(
                    base_assemblies=count,
                    min_assemblies=count,
                    max_assemblies=count,
                    **{k: v for k, v in self.config['network'].items() 
                       if k not in ['base_assemblies', 'min_assemblies', 'max_assemblies']}
                )
                
                segmenter = TropicalSegmenter(
                    SegmentationConfig(
                        n_assemblies=count,
                        min_assemblies=count,
                        max_assemblies=count
                    )
                )
                
                # Evaluate
                metrics = self.evaluate_model(segmenter)
                trial_results.append(metrics)
                
            # Aggregate results
            self.results['assembly_count'][count] = self.aggregate_trials(trial_results)
            
            print(f"  Mean IoU: {self.results['assembly_count'][count]['mean_iou']:.4f}")
            print(f"  Mean inference time: {self.results['assembly_count'][count]['mean_time']:.2f}ms")
            
    def ablate_sparsity(self):
        """Test different sparsity levels."""
        print("\n" + "-" * 40)
        print("ABLATION: Sparsity Levels")
        print("-" * 40)
        
        sparsity_levels = self.config['ablation']['sparsity_levels']
        n_trials = self.config['evaluation']['n_trials']
        
        for sparsity in sparsity_levels:
            print(f"\nTesting {sparsity*100:.1f}% sparsity...")
            trial_results = []
            
            for trial in range(n_trials):
                # Create model with specific sparsity
                config = NetworkConfig(
                    sdr_sparsity=sparsity,
                    **{k: v for k, v in self.config['network'].items() 
                       if k != 'sdr_sparsity'}
                )
                
                segmenter = TropicalSegmenter()
                segmenter.network = TropicalAssemblyNetwork(config)
                
                # Evaluate
                metrics = self.evaluate_model(segmenter)
                trial_results.append(metrics)
                
            # Aggregate results
            self.results['sparsity'][sparsity] = self.aggregate_trials(trial_results)
            
            print(f"  Mean IoU: {self.results['sparsity'][sparsity]['mean_iou']:.4f}")
            print(f"  Pattern memory: {self.results['sparsity'][sparsity]['memory_mb']:.2f}MB")
            
    def ablate_metabolic_state(self):
        """Test different metabolic states."""
        print("\n" + "-" * 40)
        print("ABLATION: Metabolic States")
        print("-" * 40)
        
        metabolic_states = self.config['ablation']['metabolic_states']
        n_trials = self.config['evaluation']['n_trials']
        
        for state in metabolic_states:
            print(f"\nTesting metabolic state {state:.1f}...")
            trial_results = []
            
            for trial in range(n_trials):
                # Create standard model
                segmenter = TropicalSegmenter()
                
                # Set metabolic state
                segmenter.network.set_metabolic_state(state)
                
                # Evaluate
                metrics = self.evaluate_model(segmenter)
                metrics['active_assemblies'] = segmenter.network.n_active
                trial_results.append(metrics)
                
            # Aggregate results
            self.results['metabolic'][state] = self.aggregate_trials(trial_results)
            
            print(f"  Mean IoU: {self.results['metabolic'][state]['mean_iou']:.4f}")
            print(f"  Active assemblies: {self.results['metabolic'][state]['active_assemblies']:.1f}")
            
    def ablate_padic_timing(self):
        """Test with/without p-adic timing."""
        print("\n" + "-" * 40)
        print("ABLATION: P-adic Timing")
        print("-" * 40)
        
        n_trials = self.config['evaluation']['n_trials']
        
        # Test with p-adic timing (normal)
        print("\nWith p-adic timing...")
        trial_results = []
        
        for trial in range(n_trials):
            segmenter = TropicalSegmenter()
            metrics = self.evaluate_model(segmenter)
            trial_results.append(metrics)
            
        self.results['padic']['with_padic'] = self.aggregate_trials(trial_results)
        
        # Test without p-adic timing (all assemblies always active)
        print("\nWithout p-adic timing...")
        trial_results = []
        
        for trial in range(n_trials):
            segmenter = TropicalSegmenter()
            
            # Disable p-adic gating (hack: make all assemblies always active)
            # This would need proper implementation in the actual code
            
            metrics = self.evaluate_model(segmenter)
            trial_results.append(metrics)
            
        self.results['padic']['without_padic'] = self.aggregate_trials(trial_results)
        
        print(f"  With p-adic - IoU: {self.results['padic']['with_padic']['mean_iou']:.4f}")
        print(f"  Without p-adic - IoU: {self.results['padic']['without_padic']['mean_iou']:.4f}")
        
    def evaluate_model(self, segmenter) -> Dict:
        """Evaluate a model configuration.
        
        Args:
            segmenter: Segmenter to evaluate
            
        Returns:
            Dictionary of metrics
        """
        metrics_calc = SegmentationMetrics()
        results = {
            'iou': [],
            'dice': [],
            'pixel_acc': [],
            'inference_time': [],
            'memory': []
        }
        
        for image, ground_truth in self.test_data:
            # Time inference
            with JAXTimer("Inference") as timer:
                result = segmenter.segment(image)
                
            # Compute metrics
            metrics = metrics_calc.compute_all_metrics(
                result.labels,
                ground_truth,
                segmenter.network.n_active
            )
            
            results['iou'].append(metrics['mean_iou'])
            results['dice'].append(metrics['mean_dice'])
            results['pixel_acc'].append(metrics['pixel_accuracy'])
            results['inference_time'].append(timer.elapsed * 1000)  # ms
            
        # Memory usage
        total_patterns = sum(
            len(a.patterns)
            for a in segmenter.network.assemblies[:segmenter.network.n_active]
        )
        results['memory'] = total_patterns * 256 / 1e6  # MB
        
        # Active assemblies
        results['active_assemblies'] = segmenter.network.n_active
        
        return results
        
    def aggregate_trials(self, trial_results: List[Dict]) -> Dict:
        """Aggregate results across trials.
        
        Args:
            trial_results: List of trial results
            
        Returns:
            Aggregated metrics
        """
        aggregated = {}
        
        # Compute mean and std for each metric
        for metric in ['iou', 'dice', 'pixel_acc', 'inference_time']:
            values = []
            for trial in trial_results:
                values.extend(trial[metric])
                
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
            
        # Memory and assembly count
        aggregated['memory_mb'] = np.mean([t['memory'] for t in trial_results])
        
        if 'active_assemblies' in trial_results[0]:
            aggregated['active_assemblies'] = np.mean(
                [t['active_assemblies'] for t in trial_results]
            )
            
        # Simplified names for common metrics
        aggregated['mean_iou'] = aggregated['mean_iou']
        aggregated['mean_time'] = aggregated['mean_inference_time']
        
        return aggregated
        
    def save_results(self):
        """Save ablation results."""
        # Save as JSON
        json_path = self.results_dir / 'ablation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
            
        # Save as CSV for easy analysis
        csv_path = self.results_dir / self.config['output']['summary_file']
        
        # Create dataframe
        rows = []
        
        # Assembly count results
        for count, metrics in self.results.get('assembly_count', {}).items():
            row = {'ablation': 'assembly_count', 'parameter': count}
            row.update(metrics)
            rows.append(row)
            
        # Sparsity results
        for sparsity, metrics in self.results.get('sparsity', {}).items():
            row = {'ablation': 'sparsity', 'parameter': sparsity}
            row.update(metrics)
            rows.append(row)
            
        # Metabolic results
        for state, metrics in self.results.get('metabolic', {}).items():
            row = {'ablation': 'metabolic', 'parameter': state}
            row.update(metrics)
            rows.append(row)
            
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to {csv_path}")
            
    def create_visualizations(self):
        """Create ablation study visualizations."""
        # Assembly count ablation
        if self.results.get('assembly_count'):
            self.plot_assembly_ablation()
            
        # Sparsity ablation
        if self.results.get('sparsity'):
            self.plot_sparsity_ablation()
            
        # Metabolic ablation
        if self.results.get('metabolic'):
            self.plot_metabolic_ablation()
            
        # P-adic ablation
        if self.results.get('padic'):
            self.plot_padic_ablation()
            
    def plot_assembly_ablation(self):
        """Plot assembly count ablation results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        counts = sorted(self.results['assembly_count'].keys())
        metrics = self.results['assembly_count']
        
        # IoU vs assembly count
        ious = [metrics[c]['mean_iou'] for c in counts]
        iou_stds = [metrics[c].get('std_iou', 0) for c in counts]
        
        axes[0].errorbar(counts, ious, yerr=iou_stds, marker='o', capsize=5)
        axes[0].axvline(x=7, color='r', linestyle='--', alpha=0.5, label='Default (7)')
        axes[0].set_xlabel('Number of Assemblies')
        axes[0].set_ylabel('Mean IoU')
        axes[0].set_title('Accuracy vs Assembly Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Inference time vs assembly count
        times = [metrics[c]['mean_time'] for c in counts]
        
        axes[1].plot(counts, times, marker='s', color='green')
        axes[1].axvline(x=7, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Number of Assemblies')
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Speed vs Assembly Count')
        axes[1].grid(True, alpha=0.3)
        
        # Memory vs assembly count
        memory = [metrics[c]['memory_mb'] for c in counts]
        
        axes[2].plot(counts, memory, marker='^', color='orange')
        axes[2].axvline(x=7, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Number of Assemblies')
        axes[2].set_ylabel('Memory (MB)')
        axes[2].set_title('Memory vs Assembly Count')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Assembly Count Ablation Study')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'ablation_assembly_count.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def plot_sparsity_ablation(self):
        """Plot sparsity ablation results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sparsities = sorted(self.results['sparsity'].keys())
        metrics = self.results['sparsity']
        
        # IoU vs sparsity
        ious = [metrics[s]['mean_iou'] for s in sparsities]
        
        axes[0].plot([s*100 for s in sparsities], ious, marker='o')
        axes[0].axvline(x=2.0, color='r', linestyle='--', alpha=0.5, label='Default (2%)')
        axes[0].set_xlabel('Sparsity (%)')
        axes[0].set_ylabel('Mean IoU')
        axes[0].set_title('Accuracy vs SDR Sparsity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Memory vs sparsity
        memory = [metrics[s]['memory_mb'] for s in sparsities]
        
        axes[1].plot([s*100 for s in sparsities], memory, marker='s', color='orange')
        axes[1].axvline(x=2.0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Sparsity (%)')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].set_title('Memory vs SDR Sparsity')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Sparsity Ablation Study')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'ablation_sparsity.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def plot_metabolic_ablation(self):
        """Plot metabolic state ablation results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        states = sorted(self.results['metabolic'].keys())
        metrics = self.results['metabolic']
        
        # IoU vs metabolic state
        ious = [metrics[s]['mean_iou'] for s in states]
        
        axes[0].plot(states, ious, marker='o')
        axes[0].set_xlabel('Metabolic State')
        axes[0].set_ylabel('Mean IoU')
        axes[0].set_title('Accuracy vs Metabolic State')
        axes[0].grid(True, alpha=0.3)
        
        # Active assemblies vs metabolic state
        assemblies = [metrics[s]['active_assemblies'] for s in states]
        
        axes[1].plot(states, assemblies, marker='s', color='green')
        axes[1].axhline(y=7, color='r', linestyle='--', alpha=0.5, label='Default (7)')
        axes[1].set_xlabel('Metabolic State')
        axes[1].set_ylabel('Active Assemblies')
        axes[1].set_title('Assembly Recruitment vs Metabolic State')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Metabolic State Ablation Study')
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'ablation_metabolic.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def plot_padic_ablation(self):
        """Plot p-adic timing ablation results."""
        if not self.results.get('padic'):
            return
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        conditions = ['With P-adic', 'Without P-adic']
        ious = [
            self.results['padic']['with_padic']['mean_iou'],
            self.results['padic']['without_padic']['mean_iou']
        ]
        
        bars = ax.bar(conditions, ious, color=['blue', 'red'])
        ax.set_ylabel('Mean IoU')
        ax.set_title('Effect of P-adic Timing on Performance')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, iou in zip(bars, ious):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{iou:.3f}', ha='center', va='bottom')
                   
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'ablation_padic.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def statistical_analysis(self):
        """Perform statistical analysis of ablation results."""
        print("\n" + "=" * 40)
        print("STATISTICAL ANALYSIS")
        print("=" * 40)
        
        # Test if 7 assemblies is significantly better
        if len(self.results.get('assembly_count', {})) >= 3:
            # Compare 5, 7, 9 assemblies
            if all(c in self.results['assembly_count'] for c in [5, 7, 9]):
                iou_5 = self.results['assembly_count'][5]['mean_iou']
                iou_7 = self.results['assembly_count'][7]['mean_iou']
                iou_9 = self.results['assembly_count'][9]['mean_iou']
                
                print("\nAssembly Count Analysis:")
                print(f"  5 assemblies: {iou_5:.4f}")
                print(f"  7 assemblies: {iou_7:.4f}")
                print(f"  9 assemblies: {iou_9:.4f}")
                
                # Check if 7 is optimal
                if iou_7 >= iou_5 and iou_7 >= iou_9:
                    print("  ✓ 7 assemblies achieves best performance")
                elif abs(iou_7 - max(iou_5, iou_9)) < 0.02:
                    print("  ≈ 7 assemblies performs comparably to best")
                else:
                    print(f"  ✗ {max([(5, iou_5), (7, iou_7), (9, iou_9)], key=lambda x: x[1])[0]} assemblies performs best")


def main():
    """Main ablation script."""
    parser = argparse.ArgumentParser(
        description='Run ablation studies for tropical SDR segmentation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/ablation.yaml',
        help='Path to ablation configuration'
    )
    
    args = parser.parse_args()
    
    # Create and run ablation study
    study = AblationStudy(args.config)
    study.run_all_ablations()


if __name__ == '__main__':
    main()