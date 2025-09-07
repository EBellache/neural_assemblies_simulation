"""Single neural assembly with SDR pattern dictionary.

Each assembly specializes in detecting specific structural features
based on its eigenvalue position on the tropical polytope.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, NamedTuple, Dict
from functools import partial

from ..core.sdr import SDR, SDRConfig
from ..core.tropical_ops import tropical_add, tropical_mul


class AssemblyState(NamedTuple):
    """Current state of an assembly."""
    active: bool
    metabolic_energy: float  # Current ATP level (0-1)
    phase: jnp.ndarray  # P-adic phase vector
    recent_winner_count: int  # Wins in recent window
    activation_history: jnp.ndarray  # Recent activation levels


class TropicalAssembly:
    """Single neural assembly with tropical dynamics.
    
    Maintains a dictionary of SDR patterns and competes for activation
    through tropical maximum operations.
    """
    
    def __init__(self,
                 index: int,
                 eigenvalue: float,
                 sdr_config: Optional[SDRConfig] = None,
                 max_patterns: int = 200):
        """Initialize assembly.
        
        Args:
            index: Assembly index (0-8)
            eigenvalue: Structural specialization (2.0=elongated, 1.0=circular)
            sdr_config: SDR configuration
            max_patterns: Maximum patterns in dictionary
        """
        self.index = index
        self.eigenvalue = eigenvalue
        self.sdr_config = sdr_config or SDRConfig()
        self.max_patterns = max_patterns
        
        # Pattern dictionary
        self.patterns = []  # List of SDRs
        self.pattern_weights = jnp.ones(0)  # Importance weights
        self.pattern_counts = jnp.zeros(0)  # Usage counts
        
        # Assembly state
        self.state = AssemblyState(
            active=True,
            metabolic_energy=1.0,
            phase=jnp.zeros(3),  # [phase_2, phase_3, phase_5]
            recent_winner_count=0,
            activation_history=jnp.zeros(100)
        )
        
        # Specialization parameters based on eigenvalue
        self._setup_specialization()
        
    def _setup_specialization(self):
        """Configure assembly based on eigenvalue."""
        if self.eigenvalue >= 1.8:
            # Specialized for elongated structures
            self.preferred_bits = (0, 300)
            self.feature_type = "elongated"
            self.min_aspect_ratio = 2.0
        elif self.eigenvalue >= 1.3:
            # Specialized for curved boundaries
            self.preferred_bits = (301, 600)
            self.feature_type = "curved"
            self.min_aspect_ratio = 1.3
        elif self.eigenvalue >= 1.1:
            # Specialized for compact structures
            self.preferred_bits = (601, 900)
            self.feature_type = "circular"
            self.min_aspect_ratio = 1.0
        else:
            # Uncertainty detector
            self.preferred_bits = (901, 2048)
            self.feature_type = "uncertain"
            self.min_aspect_ratio = 0.0
            
    @partial(jax.jit, static_argnames=['self'])
    def compute_response(self, input_sdr: SDR) -> float:
        """Compute assembly response to input using tropical operations.
        
        Args:
            input_sdr: Input SDR pattern
            
        Returns:
            Maximum overlap score (tropical sum over patterns)
        """
        if len(self.patterns) == 0:
            return 0.0
            
        # Convert input to dense for batch operations
        input_dense = input_sdr.dense
        
        # Compute overlaps with all patterns (vectorized)
        overlaps = jnp.array([
            jnp.sum(input_dense & pattern.dense) 
            for pattern in self.patterns
        ])
        
        # Weight overlaps
        weighted_overlaps = overlaps * self.pattern_weights
        
        # Tropical addition (maximum) over all patterns
        response = jnp.max(weighted_overlaps)
        
        # Modulate by metabolic state
        response *= self.state.metabolic_energy
        
        # Apply eigenvalue bias for preferred features
        start, end = self.preferred_bits
        preferred_overlap = jnp.sum(input_dense[start:end])
        response += preferred_overlap * 0.1 * self.eigenvalue
        
        return response
        
    def add_pattern(self, 
                    sdr: SDR,
                    weight: float = 1.0) -> bool:
        """Add new pattern to dictionary.
        
        Args:
            sdr: SDR pattern to add
            weight: Initial importance weight
            
        Returns:
            True if pattern was added
        """
        # Check if pattern is sufficiently unique
        if self._is_redundant(sdr):
            return False
            
        # Add pattern
        self.patterns.append(sdr)
        self.pattern_weights = jnp.append(self.pattern_weights, weight)
        self.pattern_counts = jnp.append(self.pattern_counts, 0)
        
        # Prune if exceeding capacity
        if len(self.patterns) > self.max_patterns:
            self._prune_patterns()
            
        return True
        
    def _is_redundant(self, sdr: SDR, threshold: int = 35) -> bool:
        """Check if pattern is redundant with existing patterns.
        
        Args:
            sdr: Pattern to check
            threshold: Maximum overlap to be considered unique
            
        Returns:
            True if pattern is redundant
        """
        for pattern in self.patterns:
            if sdr.overlap(pattern) > threshold:
                return True
        return False
        
    def _prune_patterns(self):
        """Remove least useful patterns to maintain capacity."""
        if len(self.patterns) <= self.max_patterns:
            return
            
        # Compute utility scores
        utility = self.pattern_weights * jnp.log1p(self.pattern_counts)
        
        # Keep top patterns
        keep_indices = jnp.argsort(utility)[-self.max_patterns:]
        
        self.patterns = [self.patterns[i] for i in keep_indices]
        self.pattern_weights = self.pattern_weights[keep_indices]
        self.pattern_counts = self.pattern_counts[keep_indices]
        
    def update_state(self,
                    won_competition: bool,
                    activation: float,
                    phase: jnp.ndarray):
        """Update assembly state after competition.
        
        Args:
            won_competition: Whether assembly won
            activation: Activation level achieved
            phase: Current p-adic phase
        """
        # Update metabolic energy
        if won_competition:
            # Winner depletes energy
            energy_cost = 0.05 * (2.0 - self.eigenvalue)  # Higher cost for general assemblies
            new_energy = self.state.metabolic_energy * (1.0 - energy_cost)
            recent_wins = self.state.recent_winner_count + 1
        else:
            # Losers recover
            recovery_rate = 0.02
            new_energy = self.state.metabolic_energy * (1.0 + recovery_rate)
            new_energy = jnp.minimum(new_energy, 1.0)
            recent_wins = self.state.recent_winner_count * 0.95  # Decay
            
        # Update activation history
        history = jnp.roll(self.state.activation_history, -1)
        history = history.at[-1].set(activation)
        
        # Create new state
        self.state = AssemblyState(
            active=new_energy > 0.1,  # Inactive if energy too low
            metabolic_energy=new_energy,
            phase=phase,
            recent_winner_count=recent_wins,
            activation_history=history
        )
        
    def differentiate_from(self, other_assembly: 'TropicalAssembly', overlap_threshold: int = 5):
        """Differentiate patterns from another assembly.
        
        Used in competitive learning to maintain uniqueness.
        
        Args:
            other_assembly: Assembly to differentiate from
            overlap_threshold: Maximum acceptable overlap
        """
        if len(self.patterns) == 0 or len(other_assembly.patterns) == 0:
            return
            
        # Find patterns that overlap too much
        patterns_to_modify = []
        
        for i, pattern in enumerate(self.patterns):
            for other_pattern in other_assembly.patterns:
                if pattern.overlap(other_pattern) > overlap_threshold:
                    patterns_to_modify.append(i)
                    break
                    
        # Modify overlapping patterns
        for idx in patterns_to_modify:
            # Reduce weight of overlapping pattern
            self.pattern_weights = self.pattern_weights.at[idx].multiply(0.9)
            
            # Could also modify the SDR itself to reduce overlap
            # For now, just mark for eventual pruning via reduced weight
            
    def get_statistics(self) -> Dict:
        """Get assembly statistics for monitoring.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'index': self.index,
            'eigenvalue': self.eigenvalue,
            'n_patterns': len(self.patterns),
            'avg_weight': float(jnp.mean(self.pattern_weights)) if len(self.patterns) > 0 else 0,
            'total_uses': float(jnp.sum(self.pattern_counts)),
            'metabolic_energy': float(self.state.metabolic_energy),
            'recent_wins': int(self.state.recent_winner_count),
            'avg_activation': float(jnp.mean(self.state.activation_history)),
            'feature_type': self.feature_type
        }
        
    def save_patterns(self, path: str):
        """Save pattern dictionary to file.
        
        Args:
            path: File path for saving
        """
        data = {
            'eigenvalue': self.eigenvalue,
            'patterns': [p.sparse.tolist() for p in self.patterns],
            'weights': self.pattern_weights.tolist(),
            'counts': self.pattern_counts.tolist()
        }
        np.save(path, data)
        
    def load_patterns(self, path: str):
        """Load pattern dictionary from file.
        
        Args:
            path: File path for loading
        """
        data = np.load(path, allow_pickle=True).item()
        
        self.eigenvalue = data['eigenvalue']
        self.patterns = [
            SDR(active_indices=indices, config=self.sdr_config)
            for indices in data['patterns']
        ]
        self.pattern_weights = jnp.array(data['weights'])
        self.pattern_counts = jnp.array(data['counts'])