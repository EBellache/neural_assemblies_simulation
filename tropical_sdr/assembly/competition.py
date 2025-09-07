"""Winner-take-all competition dynamics.

Implements tropical maximum operations for assembly selection
with metabolic constraints and phase gating.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, NamedTuple
from functools import partial

from ..core.tropical_ops import tropical_add, tropical_argmax
from ..core.padic import PadicTimer, PhaseGate


class CompetitionResult(NamedTuple):
    """Result of assembly competition."""
    winner_idx: int
    winner_score: float
    runner_up_idx: int
    runner_up_score: float
    confidence: float
    all_scores: jnp.ndarray
    active_assemblies: jnp.ndarray


class WinnerTakeAll:
    """Pure winner-take-all competition without gradients.
    
    Implements lateral inhibition through tropical operations.
    """
    
    def __init__(self,
                 inhibition_strength: float = 1.0,
                 min_confidence: float = 5.0):
        """Initialize WTA mechanism.
        
        Args:
            inhibition_strength: Lateral inhibition scaling
            min_confidence: Minimum confidence for valid winner
        """
        self.inhibition_strength = inhibition_strength
        self.min_confidence = min_confidence
        
    @partial(jax.jit, static_argnames=['self'])
    def compete(self,
                scores: jnp.ndarray,
                active_mask: jnp.ndarray) -> CompetitionResult:
        """Run winner-take-all competition.
        
        Args:
            scores: Assembly scores (n_assemblies,)
            active_mask: Boolean mask of active assemblies
            
        Returns:
            Competition result
        """
        # Apply active mask (inactive get -inf)
        masked_scores = jnp.where(active_mask, scores, -jnp.inf)
        
        # Apply lateral inhibition (supralinear)
        total_activation = jnp.sum(jnp.where(active_mask, scores, 0))
        inhibition = self.inhibition_strength * (total_activation ** 1.5)
        
        # Effective scores after inhibition
        effective_scores = masked_scores - inhibition
        
        # Find winner (tropical maximum)
        winner_idx = jnp.argmax(effective_scores)
        winner_score = effective_scores[winner_idx]
        
        # Find runner-up
        runner_up_scores = effective_scores.at[winner_idx].set(-jnp.inf)
        runner_up_idx = jnp.argmax(runner_up_scores)
        runner_up_score = runner_up_scores[runner_up_idx]
        
        # Compute confidence
        confidence = winner_score - runner_up_score
        
        return CompetitionResult(
            winner_idx=winner_idx,
            winner_score=winner_score,
            runner_up_idx=runner_up_idx,
            runner_up_score=runner_up_score,
            confidence=confidence,
            all_scores=effective_scores,
            active_assemblies=active_mask
        )
        
    @jax.jit
    def soft_wta(self,
                 scores: jnp.ndarray,
                 temperature: float = 1.0) -> jnp.ndarray:
        """Soft winner-take-all using tropical algebra.
        
        As temperature → 0, approaches hard WTA.
        
        Args:
            scores: Assembly scores
            temperature: Softness parameter
            
        Returns:
            Soft assignment probabilities
        """
        # Tropical softmax approximation
        max_score = jnp.max(scores)
        exp_scores = jnp.exp((scores - max_score) / temperature)
        return exp_scores / jnp.sum(exp_scores)


class CompetitionArena:
    """Arena for multi-assembly competition with metabolic constraints."""
    
    def __init__(self,
                 n_assemblies: int = 7,
                 timer: Optional[PadicTimer] = None):
        """Initialize competition arena.
        
        Args:
            n_assemblies: Number of competing assemblies
            timer: P-adic timer for phase gating
        """
        self.n_assemblies = n_assemblies
        self.timer = timer or PadicTimer()
        self.phase_gate = PhaseGate(self.timer)
        self.wta = WinnerTakeAll()
        
        # Track competition history
        self.competition_history = []
        
    def run_competition(self,
                       assemblies: List,
                       input_sdr) -> CompetitionResult:
        """Run full competition with phase gating and metabolic constraints.
        
        Args:
            assemblies: List of TropicalAssembly instances
            input_sdr: Input SDR pattern
            
        Returns:
            Competition result
        """
        # Compute raw scores
        scores = jnp.array([
            assembly.compute_response(input_sdr)
            for assembly in assemblies
        ])
        
        # Get metabolic states
        metabolic_states = jnp.array([
            assembly.state.metabolic_energy
            for assembly in assemblies
        ])
        
        # Apply phase gating
        gated_scores = self.phase_gate.gate_assembly_scores(scores, metabolic_states)
        
        # Determine active assemblies
        active_mask = self.timer.get_active_assemblies(self.n_assemblies)
        
        # Run competition
        result = self.wta.compete(gated_scores, active_mask)
        
        # Update timer
        self.timer.tick()
        
        # Store history
        self.competition_history.append(result)
        if len(self.competition_history) > 1000:
            self.competition_history.pop(0)
            
        return result
        
    @jax.jit
    def batch_competition(self,
                         score_matrix: jnp.ndarray,
                         active_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Batch competition across multiple inputs.
        
        Args:
            score_matrix: Scores for each input (n_inputs, n_assemblies)
            active_mask: Active assemblies mask
            
        Returns:
            (winner_indices, confidences) for each input
        """
        # Vectorized competition
        def single_competition(scores):
            masked = jnp.where(active_mask, scores, -jnp.inf)
            winner = jnp.argmax(masked)
            sorted_scores = jnp.sort(masked)
            confidence = sorted_scores[-1] - sorted_scores[-2]
            return winner, confidence
            
        # Map over all inputs
        winners, confidences = jax.vmap(single_competition)(score_matrix)
        
        return winners, confidences
        
    def get_competition_statistics(self) -> dict:
        """Analyze competition patterns.
        
        Returns:
            Statistics dictionary
        """
        if len(self.competition_history) == 0:
            return {}
            
        winners = [r.winner_idx for r in self.competition_history]
        confidences = [r.confidence for r in self.competition_history]
        
        # Count wins per assembly
        win_counts = jnp.zeros(self.n_assemblies)
        for w in winners:
            win_counts = win_counts.at[w].add(1)
            
        return {
            'total_competitions': len(self.competition_history),
            'win_distribution': win_counts / len(winners),
            'avg_confidence': float(jnp.mean(jnp.array(confidences))),
            'min_confidence': float(jnp.min(jnp.array(confidences))),
            'max_confidence': float(jnp.max(jnp.array(confidences))),
            'dominant_assembly': int(jnp.argmax(win_counts))
        }