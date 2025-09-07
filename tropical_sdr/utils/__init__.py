"""Utility functions for visualization, I/O, and JAX helpers."""

from .visualization import (
    plot_sdr,
    plot_polytope,
    plot_assembly_specialization,
    plot_segmentation_results,
    visualize_competition_dynamics,
    create_assembly_animation
)

from .io import (
    save_network,
    load_network,
    save_segmentation_result,
    load_segmentation_result,
    export_to_numpy,
    import_from_numpy
)

from .jax_utils import (
    create_rng_keys,
    batch_process,
    profile_function,
    memory_info,
    jit_compile_all
)

__all__ = [
    # Visualization
    'plot_sdr',
    'plot_polytope',
    'plot_assembly_specialization',
    'plot_segmentation_results',
    'visualize_competition_dynamics',
    'create_assembly_animation',
    
    # I/O
    'save_network',
    'load_network',
    'save_segmentation_result',
    'load_segmentation_result',
    'export_to_numpy',
    'import_from_numpy',
    
    # JAX utilities
    'create_rng_keys',
    'batch_process',
    'profile_function',
    'memory_info',
    'jit_compile_all'
]