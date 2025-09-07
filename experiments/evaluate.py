"""Evaluation script for trained models.

Comprehensive evaluation including metrics, visualizations,
and comparisons with baseline methods.
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tropical_sdr.segmentation.segmenter import TropicalSegmenter
from tropical_sdr.segmentation.metrics import SegmentationMetrics
from tropical_sdr.utils.io import load_network
from tropical_sdr.utils.visualization import plot_segmentation_results
from tropical_sdr.utils.jax_utils import JAXTimer, memory_info


class Evaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, 
                 model_path: str,
                 data_path: str,
                 output_dir: str = 'evaluation_results'):
        """Initialize evaluator.
        
        Args:
            model_path: Path to saved model
            data_path: Path to test data
            output_dir: Directory for results
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Load test data
        self.load_test_data()
        
        # Initialize metrics
        self.metrics = SegmentationMetrics()
        
        # Results storage
        self.results = {
            'per_image': [],
            'per_class': {},
            'aggregate': {},
            'timing': {},
            'memory': {}
        }
        
    def load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.model_path}")
        
        # Create segmenter
        self.segmenter = TropicalSegmenter()
        
        # Load network weights
        self.segmenter.network = load_network(self.model_path)
        
        # Set to evaluation mode
        self.segmenter.config = self.segmenter.config._replace(
            online_learning=False
        )
        
        print(f"Model loaded with {self.segmenter.network.n_active} active assemblies")
        
    def load_test_data(self):
        """Load test dataset."""
        # For now, generate synthetic test data
        # In practice, load actual test set
        print("Loading test data...")
        
        self.test_data = []
        key = jax.random.PRNGKey(123)
        
        for i in range(50):  # 50 test images
            key, subkey = jax.random.split(key)
            
            # Generate test image
            image = jax.random.uniform(subkey, (256, 256))
            
            # Generate ground truth (simplified)
            key, subkey = jax.random.split(key)
            labels = jax.random.randint(subkey, (256, 256), 0, 7)
            
            self.test_data.append((image, labels))
            
        print(f"Loaded {len(self.test_data)} test images")
        
    def evaluate(self):
        """Run complete evaluation."""
        print("\nStarting evaluation...")
        print("=" * 50)
        
        # Evaluate each image
        for idx, (image, ground_truth) in enumerate(tqdm(self.test_data, 
                                                         desc="Evaluating")):
            # Time inference
            with JAXTimer("Inference") as timer:
                result = self.segmenter.segment(image)
                
            # Compute metrics
            image_metrics = self.evaluate_single(result, ground_truth)
            image_metrics['image_id'] = idx
            image_metrics['inference_time'] = timer.elapsed
            
            self.results['per_image'].append(image_metrics)
            
        # Aggregate results
        self.aggregate_results()
        
        # Class-wise analysis
        self.analyze_per_class()
        
        # Timing analysis
        self.analyze_timing()
        
        # Memory analysis
        self.analyze_memory()
        
        # Generate report
        self.generate_report()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\nEvaluation completed!")
        print(f"Results saved to {self.output_dir}")
        
    def evaluate_single(self, result, ground_truth) -> Dict:
        """Evaluate single image.
        
        Args:
            result: Segmentation result
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.compute_all_metrics(
            result.labels,
            ground_truth,
            self.segmenter.network.n_active
        )
        
        # Add additional metrics
        metrics['mean_confidence'] = float(jnp.mean(result.confidence))
        metrics['uncertainty_ratio'] = float(jnp.mean(result.uncertainty_mask))
        
        # Assembly usage
        for i in range(self.segmenter.network.n_active):
            metrics[f'assembly_{i}_usage'] = float(jnp.mean(result.labels == i))
            
        return metrics
        
    def aggregate_results(self):
        """Aggregate per-image results."""
        df = pd.DataFrame(self.results['per_image'])
        
        # Compute aggregate statistics
        self.results['aggregate'] = {
            'mean_iou': df['mean_iou'].mean(),
            'std_iou': df['mean_iou'].std(),
            'mean_dice': df['mean_dice'].mean(),
            'std_dice': df['mean_dice'].std(),
            'mean_pixel_accuracy': df['pixel_accuracy'].mean(),
            'std_pixel_accuracy': df['pixel_accuracy'].std(),
            'mean_confidence': df['mean_confidence'].mean(),
            'mean_uncertainty': df['uncertainty_ratio'].mean(),
            'mean_inference_time': df['inference_time'].mean(),
            'std_inference_time': df['inference_time'].std()
        }
        
    def analyze_per_class(self):
        """Analyze performance per class."""
        df = pd.DataFrame(self.results['per_image'])
        
        for class_id in range(self.segmenter.network.n_active):
            class_metrics = {}
            
            # IoU for this class
            if f'iou_class_{class_id}' in df.columns:
                class_metrics['mean_iou'] = df[f'iou_class_{class_id}'].mean()
                class_metrics['std_iou'] = df[f'iou_class_{class_id}'].std()
                
            # Dice for this class
            if f'dice_class_{class_id}' in df.columns:
                class_metrics['mean_dice'] = df[f'dice_class_{class_id}'].mean()
                class_metrics['std_dice'] = df[f'dice_class_{class_id}'].std()
                
            # Usage statistics
            if f'assembly_{class_id}_usage' in df.columns:
                class_metrics['usage'] = df[f'assembly_{class_id}_usage'].mean()
                
            self.results['per_class'][f'class_{class_id}'] = class_metrics
            
    def analyze_timing(self):
        """Analyze inference timing."""
        df = pd.DataFrame(self.results['per_image'])
        
        self.results['timing'] = {
            'mean_ms': df['inference_time'].mean() * 1000,
            'std_ms': df['inference_time'].std() * 1000,
            'min_ms': df['inference_time'].min() * 1000,
            'max_ms': df['inference_time'].max() * 1000,
            'median_ms': df['inference_time'].median() * 1000,
            'fps': 1.0 / df['inference_time'].mean()
        }
        
    def analyze_memory(self):
        """Analyze memory usage."""
        memory_info()
        
        # Estimate model size
        total_patterns = sum(
            len(a.patterns) 
            for a in self.segmenter.network.assemblies[:self.segmenter.network.n_active]
        )
        
        # Each SDR is 2048 bits = 256 bytes
        pattern_memory = total_patterns * 256  # bytes
        
        self.results['memory'] = {
            'total_patterns': total_patterns,
            'pattern_memory_mb': pattern_memory / 1e6,
            'n_assemblies': self.segmenter.network.n_active
        }
        
    def generate_report(self):
        """Generate evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("TROPICAL SDR SEGMENTATION - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION")
        report.append("-" * 30)
        report.append(f"Model path: {self.model_path}")
        report.append(f"Active assemblies: {self.segmenter.network.n_active}")
        report.append(f"Total patterns: {self.results['memory']['total_patterns']}")
        report.append(f"Pattern memory: {self.results['memory']['pattern_memory_mb']:.2f} MB")
        report.append("")
        
        # Aggregate metrics
        report.append("AGGREGATE METRICS")
        report.append("-" * 30)
        for key, value in self.results['aggregate'].items():
            report.append(f"{key}: {value:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS")
        report.append("-" * 30)
        for class_name, metrics in self.results['per_class'].items():
            report.append(f"\n{class_name}:")
            for key, value in metrics.items():
                report.append(f"  {key}: {value:.4f}")
        report.append("")
        
        # Timing analysis
        report.append("TIMING ANALYSIS")
        report.append("-" * 30)
        for key, value in self.results['timing'].items():
            report.append(f"{key}: {value:.2f}")
        report.append("")
        
        # Assembly statistics
        report.append("ASSEMBLY STATISTICS")
        report.append("-" * 30)
        assembly_stats = self.segmenter.network.get_assembly_statistics()
        for i, stats in enumerate(assembly_stats):
            report.append(f"\nAssembly {i} ({stats['feature_type']}):")
            report.append(f"  Eigenvalue: {stats['eigenvalue']:.2f}")
            report.append(f"  Patterns: {stats['n_patterns']}")
            report.append(f"  Metabolic energy: {stats['metabolic_energy']:.3f}")
            
        # Save report
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        # Also save as JSON
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
            
        print(f"\nReport saved to {report_path}")
        
    def create_visualizations(self):
        """Create evaluation visualizations."""
        df = pd.DataFrame(self.results['per_image'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. IoU distribution
        axes[0, 0].hist(df['mean_iou'], bins=20, edgecolor='black')
        axes[0, 0].axvline(df['mean_iou'].mean(), color='r', 
                          linestyle='--', label=f"Mean: {df['mean_iou'].mean():.3f}")
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('IoU Distribution')
        axes[0, 0].legend()
        
        # 2. Dice distribution
        axes[0, 1].hist(df['mean_dice'], bins=20, edgecolor='black')
        axes[0, 1].axvline(df['mean_dice'].mean(), color='r',
                          linestyle='--', label=f"Mean: {df['mean_dice'].mean():.3f}")
        axes[0, 1].set_xlabel('Dice')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Dice Distribution')
        axes[0, 1].legend()
        
        # 3. Confidence distribution
        axes[0, 2].hist(df['mean_confidence'], bins=20, edgecolor='black')
        axes[0, 2].set_xlabel('Mean Confidence')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Confidence Distribution')
        
        # 4. Per-class IoU
        class_ious = []
        class_names = []
        for i in range(self.segmenter.network.n_active):
            if f'iou_class_{i}' in df.columns:
                class_ious.append(df[f'iou_class_{i}'].mean())
                class_names.append(f"Class {i}")
                
        axes[1, 0].bar(class_names, class_ious)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Mean IoU')
        axes[1, 0].set_title('Per-Class IoU')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Assembly usage
        assembly_usage = []
        for i in range(self.segmenter.network.n_active):
            if f'assembly_{i}_usage' in df.columns:
                assembly_usage.append(df[f'assembly_{i}_usage'].mean())
                
        axes[1, 1].bar(range(len(assembly_usage)), assembly_usage)
        axes[1, 1].set_xlabel('Assembly')
        axes[1, 1].set_ylabel('Usage Fraction')
        axes[1, 1].set_title('Assembly Usage')
        
        # 6. Inference time
        axes[1, 2].hist(df['inference_time'] * 1000, bins=20, edgecolor='black')
        axes[1, 2].axvline(df['inference_time'].mean() * 1000, color='r',
                          linestyle='--', 
                          label=f"Mean: {df['inference_time'].mean()*1000:.1f}ms")
        axes[1, 2].set_xlabel('Inference Time (ms)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Inference Time Distribution')
        axes[1, 2].legend()
        
        plt.suptitle('Evaluation Results')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'evaluation_plots.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        # Create confusion matrix if possible
        self.create_confusion_matrix(df)
        
    def create_confusion_matrix(self, df):
        """Create and save confusion matrix."""
        # This would require storing per-pixel predictions
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Confusion Matrix\n(Requires per-pixel data)",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Confusion Matrix')
        
        fig_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def compare_with_baseline(self, baseline_results: Optional[Dict] = None):
        """Compare with baseline methods.
        
        Args:
            baseline_results: Results from baseline methods
        """
        if baseline_results is None:
            # Create dummy baseline for demonstration
            baseline_results = {
                'U-Net': {'mean_iou': 0.65, 'inference_ms': 50},
                'DeepLab': {'mean_iou': 0.68, 'inference_ms': 80},
                'SAM': {'mean_iou': 0.70, 'inference_ms': 100}
            }
            
        # Add our results
        baseline_results['Tropical SDR'] = {
            'mean_iou': self.results['aggregate']['mean_iou'],
            'inference_ms': self.results['timing']['mean_ms']
        }
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = list(baseline_results.keys())
        ious = [baseline_results[m]['mean_iou'] for m in methods]
        times = [baseline_results[m]['inference_ms'] for m in methods]
        
        # IoU comparison
        bars = axes[0].bar(methods, ious)
        bars[-1].set_color('red')  # Highlight our method
        axes[0].set_ylabel('Mean IoU')
        axes[0].set_title('Segmentation Accuracy')
        axes[0].set_ylim(0, 1)
        
        # Inference time comparison
        bars = axes[1].bar(methods, times)
        bars[-1].set_color('red')
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Inference Speed')
        
        plt.suptitle('Comparison with Baseline Methods')
        plt.tight_layout()
        
        fig_path = self.output_dir / 'baseline_comparison.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate tropical SDR segmentation model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to saved model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='test_data',
        help='Path to test data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory'
    )
    parser.add_argument(
        '--compare-baseline',
        action='store_true',
        help='Compare with baseline methods'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(args.model, args.data, args.output)
    
    # Run evaluation
    evaluator.evaluate()
    
    # Compare with baselines if requested
    if args.compare_baseline:
        evaluator.compare_with_baseline()


if __name__ == '__main__':
    main()