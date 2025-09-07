"""Visualization utilities for tropical SDR framework.

Provides functions for visualizing SDRs, polytopes, assembly dynamics,
and segmentation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import jax.numpy as jnp

from ..core.sdr import SDR
from ..core.polytope import TropicalPolytope, Amoeba
from ..assembly.network import TropicalAssemblyNetwork
from ..segmentation.segmenter import SegmentationResult


def plot_sdr(sdr: SDR, 
             title: str = "Sparse Distributed Representation",
             figsize: Tuple[int, int] = (12, 3)) -> plt.Figure:
    """Visualize an SDR as a bitmap.
    
    Args:
        sdr: SDR to visualize
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Reshape SDR to 2D for visualization
    size = sdr.config.size
    width = int(np.sqrt(size))
    height = size // width
    
    # Plot 1: Full SDR as 2D grid
    dense_2d = sdr.dense[:width*height].reshape(height, width)
    axes[0].imshow(dense_2d, cmap='binary', interpolation='nearest')
    axes[0].set_title(f"SDR Bitmap (sparsity={sdr.sparsity:.3f})")
    axes[0].set_xlabel("Bit index (x)")
    axes[0].set_ylabel("Bit index (y)")
    
    # Plot 2: Semantic regions
    semantic_regions = np.zeros((height, width, 3))
    
    # Color code by semantic region
    for bit in sdr.sparse:
        y, x = bit // width, bit % width
        if y >= height or x >= width:
            continue
            
        if bit < 300:  # Edge bits (red)
            semantic_regions[y, x] = [1, 0, 0]
        elif bit < 600:  # Curve bits (green)
            semantic_regions[y, x] = [0, 1, 0]
        elif bit < 900:  # Circle bits (blue)
            semantic_regions[y, x] = [0, 0, 1]
        elif bit < 1200:  # Texture bits (yellow)
            semantic_regions[y, x] = [1, 1, 0]
        elif bit < 1500:  # Phase bits (magenta)
            semantic_regions[y, x] = [1, 0, 1]
        else:  # Context bits (cyan)
            semantic_regions[y, x] = [0, 1, 1]
            
    axes[1].imshow(semantic_regions)
    axes[1].set_title("Semantic Bit Regions")
    axes[1].set_xlabel("Bit index (x)")
    axes[1].set_ylabel("Bit index (y)")
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Edge (0-300)'),
        Patch(facecolor='green', label='Curve (301-600)'),
        Patch(facecolor='blue', label='Circle (601-900)'),
        Patch(facecolor='yellow', label='Texture (901-1200)'),
        Patch(facecolor='magenta', label='Phase (1201-1500)'),
        Patch(facecolor='cyan', label='Context (1501-2048)')
    ]
    axes[1].legend(handles=legend_elements, loc='center left', 
                   bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # Plot 3: Active bit histogram
    hist_data = []
    hist_labels = []
    
    for name, (start, end) in [
        ('Edge', (0, 300)),
        ('Curve', (300, 600)),
        ('Circle', (600, 900)),
        ('Texture', (900, 1200)),
        ('Phase', (1200, 1500)),
        ('Context', (1500, 2048))
    ]:
        count = np.sum((sdr.sparse >= start) & (sdr.sparse < end))
        hist_data.append(count)
        hist_labels.append(name)
        
    axes[2].bar(hist_labels, hist_data, color=['r', 'g', 'b', 'y', 'm', 'c'])
    axes[2].set_title("Active Bits by Region")
    axes[2].set_xlabel("Semantic Region")
    axes[2].set_ylabel("Number of Active Bits")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_polytope(polytope: TropicalPolytope,
                 amoeba: Optional[Amoeba] = None,
                 figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """Visualize tropical polytope and amoeba.
    
    Args:
        polytope: Tropical polytope
        amoeba: Optional amoeba to overlay
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    if polytope.dimension == 2:
        ax = fig.add_subplot(111)
        _plot_2d_polytope(ax, polytope, amoeba)
    elif polytope.dimension == 3:
        ax = fig.add_subplot(111, projection='3d')
        _plot_3d_polytope(ax, polytope, amoeba)
    else:
        # For higher dimensions, plot 2D projections
        n_projections = min(6, polytope.dimension * (polytope.dimension - 1) // 2)
        n_rows = int(np.ceil(np.sqrt(n_projections)))
        n_cols = int(np.ceil(n_projections / n_rows))
        
        proj_idx = 0
        for i in range(polytope.dimension):
            for j in range(i+1, polytope.dimension):
                if proj_idx >= n_projections:
                    break
                    
                ax = fig.add_subplot(n_rows, n_cols, proj_idx + 1)
                _plot_2d_projection(ax, polytope, amoeba, dims=(i, j))
                ax.set_title(f"Projection: dims {i} vs {j}")
                proj_idx += 1
                
    plt.suptitle(f"Tropical Polytope ({polytope.n_vertices} vertices, "
                 f"τ={polytope.amoeba_thickness:.1f})")
    plt.tight_layout()
    
    return fig


def _plot_2d_polytope(ax, polytope, amoeba):
    """Plot 2D polytope."""
    # Plot vertices
    vertices = np.array([v.position[:2] for v in polytope.vertices])
    ax.scatter(vertices[:, 0], vertices[:, 1], 
              c=[v.eigenvalue for v in polytope.vertices],
              cmap='viridis', s=200, zorder=3, edgecolors='black')
    
    # Plot edges
    for i in range(polytope.n_vertices):
        for j in range(i+1, polytope.n_vertices):
            if polytope.edge_matrix[i, j] < np.inf:
                ax.plot([vertices[i, 0], vertices[j, 0]],
                       [vertices[i, 1], vertices[j, 1]],
                       'k-', alpha=0.3, zorder=1)
                
    # Plot amoeba if provided
    if amoeba is not None:
        # Create amoeba boundary (simplified)
        theta = np.linspace(0, 2*np.pi, 100)
        for v in vertices:
            r = amoeba.thickness / 2
            x = v[0] + r * np.cos(theta)
            y = v[1] + r * np.sin(theta)
            ax.fill(x, y, alpha=0.2, color='gray')
            
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)


def _plot_3d_polytope(ax, polytope, amoeba):
    """Plot 3D polytope."""
    from mpl_toolkits.mplot3d import Axes3D
    
    vertices = np.array([v.position[:3] for v in polytope.vertices])
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              c=[v.eigenvalue for v in polytope.vertices],
              cmap='viridis', s=200, edgecolors='black')
    
    # Plot edges
    for i in range(polytope.n_vertices):
        for j in range(i+1, polytope.n_vertices):
            if polytope.edge_matrix[i, j] < np.inf:
                ax.plot([vertices[i, 0], vertices[j, 0]],
                       [vertices[i, 1], vertices[j, 1]],
                       [vertices[i, 2], vertices[j, 2]],
                       'k-', alpha=0.3)
                
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")


def _plot_2d_projection(ax, polytope, amoeba, dims):
    """Plot 2D projection of higher-dimensional polytope."""
    i, j = dims
    vertices = np.array([v.position for v in polytope.vertices])
    
    ax.scatter(vertices[:, i], vertices[:, j],
              c=[v.eigenvalue for v in polytope.vertices],
              cmap='viridis', s=100, edgecolors='black')
    
    ax.set_xlabel(f"Dim {i}")
    ax.set_ylabel(f"Dim {j}")
    ax.grid(True, alpha=0.3)


def plot_assembly_specialization(network: TropicalAssemblyNetwork,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Visualize assembly specializations and statistics.
    
    Args:
        network: Trained network
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    stats = network.get_assembly_statistics()
    
    # Plot 1: Eigenvalue distribution
    eigenvalues = [s['eigenvalue'] for s in stats]
    axes[0].bar(range(len(eigenvalues)), eigenvalues, 
                color=plt.cm.viridis(np.linspace(0, 1, len(eigenvalues))))
    axes[0].set_xlabel("Assembly Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Assembly Eigenvalues (Specialization)")
    
    # Plot 2: Pattern count
    pattern_counts = [s['n_patterns'] for s in stats]
    axes[1].bar(range(len(pattern_counts)), pattern_counts)
    axes[1].set_xlabel("Assembly Index")
    axes[1].set_ylabel("Number of Patterns")
    axes[1].set_title("Learned Patterns per Assembly")
    
    # Plot 3: Metabolic energy
    energies = [s['metabolic_energy'] for s in stats]
    axes[2].plot(energies, 'o-')
    axes[2].set_xlabel("Assembly Index")
    axes[2].set_ylabel("Metabolic Energy")
    axes[2].set_title("Current Energy Levels")
    axes[2].axhline(y=0.1, color='r', linestyle='--', label='Inactive threshold')
    axes[2].legend()
    
    # Plot 4: Recent wins
    wins = [s['recent_wins'] for s in stats]
    axes[3].bar(range(len(wins)), wins, color='orange')
    axes[3].set_xlabel("Assembly Index")
    axes[3].set_ylabel("Recent Win Count")
    axes[3].set_title("Competition Success")
    
    # Plot 5: Feature type distribution
    feature_types = [s['feature_type'] for s in stats]
    unique_types = list(set(feature_types))
    type_counts = [feature_types.count(t) for t in unique_types]
    axes[4].pie(type_counts, labels=unique_types, autopct='%1.1f%%')
    axes[4].set_title("Feature Type Distribution")
    
    # Plot 6: Average activation
    avg_activations = [s['avg_activation'] for s in stats]
    axes[5].plot(avg_activations, 's-', color='green')
    axes[5].set_xlabel("Assembly Index")
    axes[5].set_ylabel("Average Activation")
    axes[5].set_title("Mean Activation History")
    
    # Plot 7: Competition matrix (if available)
    if hasattr(network.arena, 'competition_history') and len(network.arena.competition_history) > 0:
        competition_matrix = np.zeros((network.n_active, network.n_active))
        for result in network.arena.competition_history[-100:]:
            competition_matrix[result.winner_idx, result.runner_up_idx] += 1
            
        im = axes[6].imshow(competition_matrix, cmap='hot', aspect='auto')
        axes[6].set_xlabel("Runner-up Assembly")
        axes[6].set_ylabel("Winner Assembly")
        axes[6].set_title("Competition Patterns (last 100)")
        plt.colorbar(im, ax=axes[6])
    else:
        axes[6].text(0.5, 0.5, "No competition history", 
                    ha='center', va='center', transform=axes[6].transAxes)
        axes[6].set_title("Competition Patterns")
        
    # Plot 8: P-adic phase distribution
    phases = np.array([network.timer.get_phase_vector()])
    axes[7].bar(['8ms (2³)', '27ms (3³)', '125ms (5³)'], phases[0])
    axes[7].set_ylabel("Phase (normalized)")
    axes[7].set_title("Current P-adic Phases")
    
    plt.suptitle(f"Assembly Network Analysis (n_active={network.n_active})")
    plt.tight_layout()
    
    return fig


def plot_segmentation_results(result: SegmentationResult,
                             original_image: Optional[np.ndarray] = None,
                             ground_truth: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Visualize segmentation results.
    
    Args:
        result: Segmentation result
        original_image: Original input image
        ground_truth: Ground truth labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_plots = 4
    if original_image is not None:
        n_plots += 1
    if ground_truth is not None:
        n_plots += 1
        
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Original image
    if original_image is not None:
        axes[plot_idx].imshow(original_image, cmap='gray')
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    # Segmentation
    im = axes[plot_idx].imshow(result.labels, cmap='tab10')
    axes[plot_idx].set_title("Segmentation")
    axes[plot_idx].axis('off')
    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
    plot_idx += 1
    
    # Ground truth
    if ground_truth is not None:
        im = axes[plot_idx].imshow(ground_truth, cmap='tab10')
        axes[plot_idx].set_title("Ground Truth")
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
        plot_idx += 1
        
    # Confidence map
    im = axes[plot_idx].imshow(result.confidence, cmap='viridis')
    axes[plot_idx].set_title("Confidence Map")
    axes[plot_idx].axis('off')
    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
    plot_idx += 1
    
    # Uncertainty mask
    axes[plot_idx].imshow(result.uncertainty_mask, cmap='gray')
    axes[plot_idx].set_title("Uncertain Regions")
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    # Assembly activation bar chart
    axes[plot_idx].bar(range(len(result.assembly_scores)), 
                       result.assembly_scores)
    axes[plot_idx].set_xlabel("Assembly Index")
    axes[plot_idx].set_ylabel("Activation Fraction")
    axes[plot_idx].set_title("Assembly Usage")
    plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
        
    # Add timing information
    if result.timing:
        timing_text = "Timing: " + ", ".join([
            f"{k}: {v:.3f}s" for k, v in result.timing.items()
        ])
        fig.text(0.5, 0.02, timing_text, ha='center', fontsize=10)
        
    plt.suptitle("Segmentation Results")
    plt.tight_layout()
    
    return fig


def visualize_competition_dynamics(network: TropicalAssemblyNetwork,
                                  n_steps: int = 100,
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Visualize competition dynamics over time.
    
    Args:
        network: Network with competition history
        n_steps: Number of recent steps to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not hasattr(network.arena, 'competition_history') or \
       len(network.arena.competition_history) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No competition history available",
               ha='center', va='center', transform=ax.transAxes)
        return fig
        
    history = network.arena.competition_history[-n_steps:]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Winner trajectory
    winners = [r.winner_idx for r in history]
    axes[0, 0].plot(winners, 'o-', markersize=3)
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Winner Assembly Index")
    axes[0, 0].set_title("Winner Trajectory")
    axes[0, 0].set_ylim(-0.5, network.n_active - 0.5)
    
    # Plot 2: Confidence over time
    confidences = [r.confidence for r in history]
    axes[0, 1].plot(confidences, color='green')
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Confidence")
    axes[0, 1].set_title("Competition Confidence")
    axes[0, 1].axhline(y=network.arena.wta.min_confidence, 
                       color='r', linestyle='--', label='Min threshold')
    axes[0, 1].legend()
    
    # Plot 3: Assembly activation heatmap
    activation_matrix = np.zeros((network.n_active, n_steps))
    for t, result in enumerate(history):
        activation_matrix[:, t] = result.all_scores[:network.n_active]
        
    im = axes[1, 0].imshow(activation_matrix, aspect='auto', cmap='hot')
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Assembly Index")
    axes[1, 0].set_title("Assembly Activation Heatmap")
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Phase-locked activity
    phases = []
    for _ in history:
        phases.append(network.timer.get_phase_vector())
        network.timer.tick()
        
    phases = np.array(phases)
    
    for i, label in enumerate(['8ms', '27ms', '125ms']):
        axes[1, 1].plot(phases[:, i], label=label, alpha=0.7)
        
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Phase (normalized)")
    axes[1, 1].set_title("P-adic Phase Evolution")
    axes[1, 1].legend()
    
    plt.suptitle("Competition Dynamics Analysis")
    plt.tight_layout()
    
    return fig


def create_assembly_animation(network: TropicalAssemblyNetwork,
                             input_sequence: List[SDR],
                             interval: int = 100) -> FuncAnimation:
    """Create animation of assembly dynamics.
    
    Args:
        network: Network to animate
        input_sequence: Sequence of input SDRs
        interval: Milliseconds between frames
        
    Returns:
        Matplotlib animation
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initialize plots
    assembly_bars = axes[0, 0].bar(range(network.n_active), 
                                   np.zeros(network.n_active))
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Assembly Index")
    axes[0, 0].set_ylabel("Activation")
    axes[0, 0].set_title("Assembly Activations")
    
    sdr_img = axes[0, 1].imshow(np.zeros((64, 32)), cmap='binary')
    axes[0, 1].set_title("Current Input SDR")
    
    confidence_line, = axes[1, 0].plot([], [], 'g-')
    axes[1, 0].set_xlim(0, len(input_sequence))
    axes[1, 0].set_ylim(0, 50)
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Confidence")
    axes[1, 0].set_title("Competition Confidence")
    
    winner_scatter = axes[1, 1].scatter([], [], c=[], cmap='tab10', s=50)
    axes[1, 1].set_xlim(0, len(input_sequence))
    axes[1, 1].set_ylim(-0.5, network.n_active - 0.5)
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Winner Index")
    axes[1, 1].set_title("Winner Trajectory")
    
    # Animation data
    confidence_history = []
    winner_history = []
    
    def animate(frame):
        if frame >= len(input_sequence):
            return
            
        # Process input
        sdr = input_sequence[frame]
        winner, confidence, result = network.process_input(sdr, learn=False)
        
        # Update assembly bars
        for i, bar in enumerate(assembly_bars):
            bar.set_height(result.all_scores[i] if i < len(result.all_scores) else 0)
            
        # Update SDR display
        sdr_2d = sdr.dense[:2048].reshape(64, 32)
        sdr_img.set_array(sdr_2d)
        
        # Update confidence
        confidence_history.append(confidence)
        confidence_line.set_data(range(len(confidence_history)), confidence_history)
        
        # Update winner
        winner_history.append(winner)
        winner_scatter.set_offsets(np.c_[range(len(winner_history)), winner_history])
        winner_scatter.set_array(np.array(winner_history))
        
        return assembly_bars, sdr_img, confidence_line, winner_scatter
        
    anim = FuncAnimation(fig, animate, frames=len(input_sequence),
                        interval=interval, blit=False)
    
    return anim