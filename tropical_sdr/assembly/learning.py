"""Hebbian learning without gradient descent.

Implements competitive Hebbian learning with homeostasis
for pattern dictionary updates.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, NamedTuple
from functools import partial

from .assembly import TropicalAssembly
from .competition import CompetitionResult
from ..core.sdr import SDR


class LearningConfig(NamedTuple):
    """Configuration for Hebbian learning."""
    eta_positive: float = 0.01  # Learning rate for winner
    eta_negative: float = 0.005  # Learning rate for losers
    homeostasis_rate: float = 0.001  # Homeostatic adaptation rate
    min_pattern_weight: float = 0.001  # Minimum weight before pruning
    differentiation_threshold: int = 5  # Max overlap between assemblies
    reinforcement_decay: float = 0.99  # Decay for unused patterns


class HebbianLearning:
    """Pure Hebbian learning without backpropagation.
    
    Updates pattern dictionaries based on competition outcomes
    using local learning rules.
    """
    
    def __init__(self, config: Optional[LearningConfig] = None):
        """Initialize learning module.
        
        Args:
            config: Learning configuration
        """
        self.config = config or LearningConfig()
        
        # Track learning statistics
        self.total_updates = 0
        self.pattern_additions = 0
        self.pattern_deletions = 0
        
    def update(self,
              assemblies: List[TropicalAssembly],
              winner_idx: int,
              input_sdr: SDR,
              competition_result: CompetitionResult):
        """Update assemblies after competition.
        
        Args:
            assemblies: List of assemblies
            winner_idx: Index of winning assembly
            input_sdr: Input that triggered competition
            competition_result: Full competition results
        """
        # Update winner
        self._update_winner(
            assemblies[winner_idx],
            input_sdr,
            assemblies,
            competition_result
        )
        
        # Update losers
        for i, assembly in enumerate(assemblies):
            if i != winner_idx:
                self._update_loser(
                    assembly,
                    assemblies[winner_idx],
                    competition_result.all_scores[i]
                )
                
        # Apply homeostasis
        self._apply_homeostasis(assemblies)
        
        self.total_updates += 1
        
    def _update_winner(self,
                      winner: TropicalAssembly,
                      input_sdr: SDR,
                      all_assemblies: List[TropicalAssembly],
                      result: CompetitionResult):
        """Update winning assembly - strengthen and add patterns.
        
        Args:
            winner: Winning assembly
            input_sdr: Input pattern
            all_assemblies: All assemblies for uniqueness check
            result: Competition result
        """
        # Check if pattern should be added
        best_overlap = 0
        best_pattern_idx = -1
        
        for i, pattern in enumerate(winner.patterns):
            overlap = input_sdr.overlap(pattern)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pattern_idx = i
                
        # If pattern is novel enough, add it
        if best_overlap < 30:  # Less than 75% overlap
            # Extract unique bits not in other assemblies
            unique_sdr = self._extract_unique_features(
                input_sdr,
                winner,
                all_assemblies
            )
            
            if winner.add_pattern(unique_sdr):
                self.pattern_additions += 1
                
        # Strengthen matching pattern
        elif best_pattern_idx >= 0:
            # Hebbian reinforcement
            winner.pattern_weights = winner.pattern_weights.at[best_pattern_idx].multiply(
                1.0 + self.config.eta_positive
            )
            winner.pattern_counts = winner.pattern_counts.at[best_pattern_idx].add(1)
            
    def _update_loser(self,
                     loser: TropicalAssembly,
                     winner: TropicalAssembly,
                     activation: float):
        """Update losing assembly - differentiate from winner.
        
        Args:
            loser: Losing assembly
            winner: Winning assembly
            activation: Loser's activation level
        """
        # If activation was close to winning, differentiate
        if activation > 0.8 * winner.state.activation_history[-1]:
            loser.differentiate_from(winner, self.config.differentiation_threshold)
            
        # Decay unused patterns
        loser.pattern_weights *= self.config.reinforcement_decay
        
        # Prune very weak patterns
        keep_mask = loser.pattern_weights > self.config.min_pattern_weight
        if jnp.sum(~keep_mask) > 0:
            loser.patterns = [p for i, p in enumerate(loser.patterns) if keep_mask[i]]
            loser.pattern_weights = loser.pattern_weights[keep_mask]
            loser.pattern_counts = loser.pattern_counts[keep_mask]
            self.pattern_deletions += jnp.sum(~keep_mask)
            
    def _extract_unique_features(self,
                                input_sdr: SDR,
                                winner: TropicalAssembly,
                                assemblies: List[TropicalAssembly]) -> SDR:
        """Extract features unique to winner.
        
        Args:
            input_sdr: Input pattern
            winner: Winning assembly
            assemblies: All assemblies
            
        Returns:
            SDR with unique features emphasized
        """
        # Compute union of other assemblies' patterns
        other_bits = jnp.zeros(input_sdr.config.size, dtype=jnp.uint8)
        
        for assembly in assemblies:
            if assembly.index != winner.index:
                for pattern in assembly.patterns[:5]:  # Check top patterns
                    other_bits |= pattern.dense
                    
        # Find unique bits
        unique_bits = input_sdr.dense & (~other_bits)
        
        # Ensure sparsity by keeping top bits
        if jnp.sum(unique_bits) < 20:
            # If too few unique bits, include some shared ones
            candidate_bits = input_sdr.dense
        else:
            candidate_bits = unique_bits
            
        # Select top bits based on winner's eigenvalue preference
        start, end = winner.preferred_bits
        preferred_mask = jnp.zeros_like(candidate_bits)
        preferred_mask = preferred_mask.at[start:end].set(1)
        
        # Combine unique and preferred
        final_bits = candidate_bits * (1 + preferred_mask)
        
        # Keep top 40 bits
        top_indices = jnp.argsort(final_bits)[-40:]
        
        return SDR(active_indices=top_indices, config=input_sdr.config)
        
    def _apply_homeostasis(self, assemblies: List[TropicalAssembly]):
        """Apply homeostatic normalization to maintain stability.
        
        Args:
            assemblies: List of assemblies to normalize
        """
        for assembly in assemblies:
            if len(assembly.patterns) == 0:
                continue
                
            # Normalize weights to sum to number of patterns
            total_weight = jnp.sum(assembly.pattern_weights)
            target_weight = len(assembly.patterns)
            
            if total_weight > 0:
                scale = target_weight / total_weight
                # Smooth adaptation
                scale = 1.0 + self.config.homeostasis_rate * (scale - 1.0)
                assembly.pattern_weights *= scale
                
    def get_learning_statistics(self) -> dict:
        """Get learning statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_updates': self.total_updates,
            'pattern_additions': self.pattern_additions,
            'pattern_deletions': self.pattern_deletions,
            'patterns_per_addition': (self.pattern_additions / 
                                     max(1, self.total_updates)),
            'patterns_per_deletion': (self.pattern_deletions / 
                                     max(1, self.total_updates))
        }
        
    @staticmethod
    @jax.jit
    def compute_stdp_weight_change(dt: float,
                                   tau_plus: float = 20.0,
                                   tau_minus: float = 20.0,
                                   a_plus: float = 0.01,
                                   a_minus: float = 0.01) -> float:
        """Compute STDP weight change for precise timing.
        
        Although we use Hebbian learning, this can modulate
        weights based on precise spike timing if needed.
        
        Args:
            dt: Time difference (post - pre)
            tau_plus: Potentiation time constant
            tau_minus: Depression time constant
            a_plus: Potentiation amplitude
            a_minus: Depression amplitude
            
        Returns:
            Weight change
        """
        return jnp.where(
            dt > 0,
            a_plus * jnp.exp(-dt / tau_plus),
            -a_minus * jnp.exp(dt / tau_minus)
        )