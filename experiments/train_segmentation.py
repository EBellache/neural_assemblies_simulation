"""Training script for tropical SDR segmentation.

This script demonstrates training without gradient descent,
using only Hebbian learning and competitive dynamics.
"""

import os
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

from tropical_sdr.assembly.network import TropicalAssemblyNetwork, NetworkConfig
from tropical_sdr.assembly.learning import LearningConfig
from tropical_sdr.segmentation.segmenter import (
    TropicalSegmenter, 
    SegmentationConfig
)
from tropical_sdr.segmentation.metrics import SegmentationMetrics
from tropical_sdr.utils.io import save_network, save_config
from tropical_sdr.utils.visualization import (
    plot_assembly_specialization,
    plot_segmentation_results
)


class SegmentationTrainer:
    """Trainer for tropical SDR segmentation."""
    
    def __init__(self, config_path: str):
        """Initialize trainer from config file.
        
        Args:
            config_path: Path to YAML configuration
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        
        # Training state
        self.epoch = 0
        self.best_val_score = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': [],
            'n_assemblies': []
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def setup_directories(self):
        """Create necessary directories."""
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.log_dir = Path(self.config['training']['log_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to log directory
        save_config(self.config, self.log_dir / 'config.yaml')
        
    def setup_model(self):
        """Initialize the segmentation model."""
        # Network configuration
        network_config = NetworkConfig(**self.config['network'])
        
        # Segmentation configuration
        seg_config = SegmentationConfig(**self.config['segmentation'])
        
        # Learning configuration
        learning_config = LearningConfig(**self.config['learning'])
        
        # Create segmenter
        self.segmenter = TropicalSegmenter(seg_config)
        
        # Override learning config
        self.segmenter.network.learning.config = learning_config
        
        print(f"Model initialized with {self.segmenter.network.n_active} assemblies")
        
    def setup_data(self):
        """Setup training and validation data."""
        data_config = self.config['data']
        
        if data_config['dataset'] == 'synthetic':
            self.train_data = self.generate_synthetic_data(
                n_samples=self.config['training']['images_per_epoch'],
                image_size=tuple(data_config['image_size']),
                n_classes=data_config['n_classes']
            )
            self.val_data = self.generate_synthetic_data(
                n_samples=int(self.config['training']['images_per_epoch'] * 
                            self.config['training']['validation_split']),
                image_size=tuple(data_config['image_size']),
                n_classes=data_config['n_classes']
            )
        elif data_config['dataset'] == 'medical':
            self.train_data, self.val_data = self.load_medical_data()
        else:
            self.train_data, self.val_data = self.load_custom_data()
            
        print(f"Data loaded: {len(self.train_data)} train, {len(self.val_data)} val")
        
    def generate_synthetic_data(self, 
                               n_samples: int,
                               image_size: Tuple[int, int],
                               n_classes: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Generate synthetic segmentation data.
        
        Args:
            n_samples: Number of samples
            image_size: Image dimensions
            n_classes: Number of segmentation classes
            
        Returns:
            List of (image, label) pairs
        """
        data = []
        key = jax.random.PRNGKey(self.config['experiment']['seed'])
        
        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            
            # Generate random shapes
            image = jnp.zeros(image_size)
            labels = jnp.zeros(image_size, dtype=jnp.int32)
            
            # Add random blobs for each class
            for class_id in range(1, n_classes):
                key, subkey = jax.random.split(key)
                
                # Random center
                center_x = jax.random.uniform(subkey, (), minval=20, maxval=image_size[0]-20)
                key, subkey = jax.random.split(key)
                center_y = jax.random.uniform(subkey, (), minval=20, maxval=image_size[1]-20)
                
                # Random radius
                key, subkey = jax.random.split(key)
                radius = jax.random.uniform(subkey, (), minval=10, maxval=30)
                
                # Create blob
                y, x = jnp.ogrid[:image_size[0], :image_size[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                
                # Add to image and labels
                key, subkey = jax.random.split(key)
                intensity = jax.random.uniform(subkey, (), minval=0.3, maxval=1.0)
                image = jnp.where(mask, intensity, image)
                labels = jnp.where(mask, class_id, labels)
                
            # Add noise
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, image_size) * 0.1
            image = jnp.clip(image + noise, 0, 1)
            
            data.append((image, labels))
            
        return data
        
    def load_medical_data(self) -> Tuple[List, List]:
        """Load medical imaging data.
        
        Returns:
            Training and validation data
        """
        # Placeholder - implement actual medical data loading
        print("Loading medical data (placeholder)...")
        return self.generate_synthetic_data(100, (256, 256), 7), \
               self.generate_synthetic_data(20, (256, 256), 7)
               
    def load_custom_data(self) -> Tuple[List, List]:
        """Load custom dataset.
        
        Returns:
            Training and validation data
        """
        # Placeholder - implement custom data loading
        print("Loading custom data (placeholder)...")
        return self.generate_synthetic_data(100, (256, 256), 7), \
               self.generate_synthetic_data(20, (256, 256), 7)
               
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        metrics = SegmentationMetrics()
        epoch_stats = {
            'iou': [],
            'dice': [],
            'pixel_acc': [],
            'confidence': [],
            'n_patterns': []
        }
        
        # Training mode
        self.segmenter.config = self.segmenter.config._replace(online_learning=True)
        
        pbar = tqdm(self.train_data, desc=f"Epoch {self.epoch+1} Training")
        for image, ground_truth in pbar:
            # Segment with learning
            result = self.segmenter.segment(image)
            
            # Compute metrics
            all_metrics = metrics.compute_all_metrics(
                result.labels,
                ground_truth,
                self.segmenter.network.n_active
            )
            
            epoch_stats['iou'].append(all_metrics['mean_iou'])
            epoch_stats['dice'].append(all_metrics['mean_dice'])
            epoch_stats['pixel_acc'].append(all_metrics['pixel_accuracy'])
            epoch_stats['confidence'].append(float(jnp.mean(result.confidence)))
            
            # Track pattern growth
            total_patterns = sum(
                len(a.patterns) 
                for a in self.segmenter.network.assemblies[:self.segmenter.network.n_active]
            )
            epoch_stats['n_patterns'].append(total_patterns)
            
            # Update progress bar
            pbar.set_postfix({
                'IoU': f"{np.mean(epoch_stats['iou']):.3f}",
                'Patterns': total_patterns
            })
            
        return {
            'mean_iou': np.mean(epoch_stats['iou']),
            'mean_dice': np.mean(epoch_stats['dice']),
            'mean_pixel_acc': np.mean(epoch_stats['pixel_acc']),
            'mean_confidence': np.mean(epoch_stats['confidence']),
            'total_patterns': epoch_stats['n_patterns'][-1]
        }
        
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        metrics = SegmentationMetrics()
        val_stats = {
            'iou': [],
            'dice': [],
            'pixel_acc': [],
            'confidence': []
        }
        
        # Evaluation mode (no learning)
        self.segmenter.config = self.segmenter.config._replace(online_learning=False)
        
        pbar = tqdm(self.val_data, desc="Validation")
        for image, ground_truth in pbar:
            # Segment without learning
            result = self.segmenter.segment(image)
            
            # Compute metrics
            all_metrics = metrics.compute_all_metrics(
                result.labels,
                ground_truth,
                self.segmenter.network.n_active
            )
            
            val_stats['iou'].append(all_metrics['mean_iou'])
            val_stats['dice'].append(all_metrics['mean_dice'])
            val_stats['pixel_acc'].append(all_metrics['pixel_accuracy'])
            val_stats['confidence'].append(float(jnp.mean(result.confidence)))
            
            pbar.set_postfix({'IoU': f"{np.mean(val_stats['iou']):.3f}"})
            
        return {
            'mean_iou': np.mean(val_stats['iou']),
            'mean_dice': np.mean(val_stats['dice']),
            'mean_pixel_acc': np.mean(val_stats['pixel_acc']),
            'mean_confidence': np.mean(val_stats['confidence'])
        }
        
    def train(self):
        """Run full training loop."""
        n_epochs = self.config['training']['n_epochs']
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.training_history['train_iou'].append(train_metrics['mean_iou'])
            self.training_history['val_iou'].append(val_metrics['mean_iou'])
            self.training_history['n_assemblies'].append(
                self.segmenter.network.n_active
            )
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Train IoU: {train_metrics['mean_iou']:.4f}")
            print(f"  Val IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Active Assemblies: {self.segmenter.network.n_active}")
            print(f"  Total Patterns: {train_metrics['total_patterns']}")
            print(f"  Mean Confidence: {train_metrics['mean_confidence']:.3f}")
            
            # Save checkpoint if improved
            if val_metrics['mean_iou'] > self.best_val_score:
                self.best_val_score = val_metrics['mean_iou']
                self.save_checkpoint('best')
                print(f"  ✓ New best model (IoU: {self.best_val_score:.4f})")
                
            # Regular checkpoint
            if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
                
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_history()
                self.visualize_assemblies()
                
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best validation IoU: {self.best_val_score:.4f}")
        
        # Final save
        self.save_checkpoint('final')
        self.plot_training_history()
        self.visualize_assemblies()
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f'{name}.pkl'
        save_network(self.segmenter.network, checkpoint_path)
        
        # Save training history
        history_path = self.checkpoint_dir / f'{name}_history.npy'
        np.save(history_path, self.training_history)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
        
    def plot_training_history(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.training_history['train_iou']) + 1)
        
        # IoU curve
        axes[0, 0].plot(epochs, self.training_history['train_iou'], 
                       'b-', label='Train')
        axes[0, 0].plot(epochs, self.training_history['val_iou'], 
                       'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('IoU')
        axes[0, 0].set_title('IoU Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Assembly count
        axes[0, 1].plot(epochs, self.training_history['n_assemblies'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Active Assemblies')
        axes[0, 1].set_title('Assembly Count Adaptation')
        axes[0, 1].set_ylim(4.5, 9.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pattern growth (placeholder)
        axes[1, 0].text(0.5, 0.5, "Pattern Growth\n(Not tracked)", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Pattern Accumulation')
        
        # Learning statistics
        stats = self.segmenter.network.learning.get_learning_statistics()
        stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Statistics')
        axes[1, 1].axis('off')
        
        plt.suptitle('Training Progress')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.log_dir / 'training_history.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def visualize_assemblies(self):
        """Visualize assembly specializations."""
        fig = plot_assembly_specialization(self.segmenter.network)
        
        # Save figure
        fig_path = self.log_dir / f'assemblies_epoch_{self.epoch+1}.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
    def visualize_sample_predictions(self, n_samples: int = 4):
        """Visualize sample segmentation results.
        
        Args:
            n_samples: Number of samples to visualize
        """
        fig, axes = plt.subplots(n_samples, 4, figsize=(15, n_samples*4))
        
        # Get random samples
        indices = np.random.choice(len(self.val_data), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, ground_truth = self.val_data[idx]
            result = self.segmenter.segment(image)
            
            # Original image
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(ground_truth, cmap='tab10', vmin=0, vmax=9)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(result.labels, cmap='tab10', vmin=0, vmax=9)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Confidence
            im = axes[i, 3].imshow(result.confidence, cmap='viridis')
            axes[i, 3].set_title('Confidence')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
            
        plt.suptitle(f'Sample Predictions (Epoch {self.epoch+1})')
        plt.tight_layout()
        
        # Save figure
        fig_path = self.log_dir / f'predictions_epoch_{self.epoch+1}.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train tropical SDR segmentation model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/baseline.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu', 'tpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Set JAX device
    if args.device == 'gpu':
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    elif args.device == 'tpu':
        os.environ['JAX_PLATFORM_NAME'] = 'tpu'
    else:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
    # Print device info
    print(f"Using device: {jax.devices()[0]}")
    
    # Create trainer and run
    trainer = SegmentationTrainer(args.config)
    trainer.train()
    
    # Final visualization
    trainer.visualize_sample_predictions(n_samples=6)


if __name__ == '__main__':
    main()