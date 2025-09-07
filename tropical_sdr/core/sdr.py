"""Sparse Distributed Representation implementation.

SDRs are binary vectors with ~2% sparsity that naturally implement
tropical operations through set operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, List, Tuple, Union, NamedTuple
from functools import partial


class SDRConfig(NamedTuple):
    """Configuration for SDR parameters."""
    size: int = 2048
    sparsity: float = 0.02
    n_active: int = 40  # size * sparsity
    
    # Semantic bit ranges
    edge_bits: Tuple[int, int] = (0, 300)
    curve_bits: Tuple[int, int] = (301, 600)
    circle_bits: Tuple[int, int] = (601, 900)
    phase_bits: Tuple[int, int] = (901, 1200)
    context_bits: Tuple[int, int] = (1201, 2048)


class SDR:
    """Sparse Distributed Representation with tropical operations.
    
    Maintains both dense (bit array) and sparse (indices) formats.
    """
    
    def __init__(self, 
                 active_indices: Optional[Union[List[int], jnp.ndarray]] = None,
                 dense: Optional[jnp.ndarray] = None,
                 config: Optional[SDRConfig] = None):
        """Initialize SDR from either sparse or dense format.
        
        Args:
            active_indices: Indices of active bits
            dense: Dense binary array
            config: SDR configuration
        """
        self.config = config or SDRConfig()
        
        if active_indices is not None:
            self.set_sparse(active_indices)
        elif dense is not None:
            self.set_dense(dense)
        else:
            # Initialize empty SDR
            self._dense = jnp.zeros(self.config.size, dtype=jnp.uint8)
            self._sparse = jnp.array([], dtype=jnp.int32)
            
    def set_sparse(self, indices: Union[List[int], jnp.ndarray]):
        """Set SDR from sparse indices."""
        indices = jnp.asarray(indices, dtype=jnp.int32)
        
        # Ensure sparsity constraint
        if len(indices) > self.config.n_active:
            indices = indices[:self.config.n_active]
            
        self._sparse = indices
        self._dense = self._sparse_to_dense(indices)
        
    def set_dense(self, dense: jnp.ndarray):
        """Set SDR from dense array."""
        self._dense = jnp.asarray(dense, dtype=jnp.uint8)
        self._sparse = self._dense_to_sparse(self._dense)
        
    @partial(jax.jit, static_argnames=['self'])
    def _sparse_to_dense(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Convert sparse indices to dense format."""
        dense = jnp.zeros(self.config.size, dtype=jnp.uint8)
        return dense.at[indices].set(1)
        
    @jax.jit
    def _dense_to_sparse(self, dense: jnp.ndarray) -> jnp.ndarray:
        """Convert dense format to sparse indices."""
        return jnp.where(dense > 0)[0].astype(jnp.int32)
        
    @property
    def dense(self) -> jnp.ndarray:
        """Get dense representation."""
        return self._dense
        
    @property
    def sparse(self) -> jnp.ndarray:
        """Get sparse representation (indices)."""
        return self._sparse
        
    @property
    def sparsity(self) -> float:
        """Current sparsity level."""
        return len(self._sparse) / self.config.size
        
    @jax.jit
    def overlap(self, other: 'SDR') -> int:
        """Compute overlap (number of shared active bits).
        
        This measures similarity between SDRs.
        
        Args:
            other: Another SDR
            
        Returns:
            Number of overlapping active bits
        """
        return jnp.sum(self._dense & other._dense)
        
    @jax.jit
    def union(self, other: 'SDR') -> 'SDR':
        """Tropical addition via union operation.
        
        Implements x ⊕ y in SDR space.
        
        Args:
            other: Another SDR
            
        Returns:
            New SDR with union of active bits
        """
        result_dense = self._dense | other._dense
        return SDR(dense=result_dense, config=self.config)
        
    @jax.jit
    def intersection(self, other: 'SDR') -> 'SDR':
        """Set intersection operation.
        
        Args:
            other: Another SDR
            
        Returns:
            New SDR with intersection of active bits
        """
        result_dense = self._dense & other._dense
        return SDR(dense=result_dense, config=self.config)
        
    @jax.jit
    def difference(self, other: 'SDR') -> 'SDR':
        """Set difference operation (self - other).
        
        Args:
            other: Another SDR
            
        Returns:
            New SDR with bits in self but not other
        """
        result_dense = self._dense & (~other._dense)
        return SDR(dense=result_dense, config=self.config)
        
    def add_noise(self, percent_noise: float, key: jax.random.PRNGKey) -> 'SDR':
        """Add noise by flipping some bits.
        
        Args:
            percent_noise: Fraction of bits to flip
            key: JAX random key
            
        Returns:
            New noisy SDR
        """
        n_flip = int(self.config.n_active * percent_noise)
        
        # Remove some active bits
        key1, key2 = jax.random.split(key)
        remove_idx = jax.random.choice(key1, self._sparse, 
                                       shape=(n_flip,), replace=False)
        
        # Add some random bits
        all_indices = jnp.arange(self.config.size)
        inactive = jnp.setdiff1d(all_indices, self._sparse)
        add_idx = jax.random.choice(key2, inactive,
                                    shape=(n_flip,), replace=False)
        
        # Combine
        new_sparse = jnp.concatenate([
            jnp.setdiff1d(self._sparse, remove_idx),
            add_idx
        ])
        
        return SDR(active_indices=new_sparse, config=self.config)
        
    def get_semantic_bits(self, semantic_type: str) -> jnp.ndarray:
        """Extract bits from semantic region.
        
        Args:
            semantic_type: One of 'edge', 'curve', 'circle', 'phase', 'context'
            
        Returns:
            Active bits in that semantic range
        """
        ranges = {
            'edge': self.config.edge_bits,
            'curve': self.config.curve_bits,
            'circle': self.config.circle_bits,
            'phase': self.config.phase_bits,
            'context': self.config.context_bits
        }
        
        start, end = ranges[semantic_type]
        mask = (self._sparse >= start) & (self._sparse < end)
        return self._sparse[mask]
        
    @staticmethod
    @jax.jit
    def batch_overlap(sdrs1: jnp.ndarray, sdrs2: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise overlaps between two sets of SDRs.
        
        Args:
            sdrs1: Dense SDR matrix (n_sdrs1 x sdr_size)
            sdrs2: Dense SDR matrix (n_sdrs2 x sdr_size)
            
        Returns:
            Overlap matrix (n_sdrs1 x n_sdrs2)
        """
        # Use matrix multiplication for efficient overlap computation
        return jnp.dot(sdrs1.astype(jnp.float32), 
                      sdrs2.astype(jnp.float32).T).astype(jnp.int32)
        
    def __repr__(self) -> str:
        return f"SDR(active={len(self._sparse)}/{self.config.size}, sparsity={self.sparsity:.3f})"