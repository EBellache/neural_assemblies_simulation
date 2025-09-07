"""Full 7±2 assembly network with tropical dynamics.

Implements the complete framework with metabolic adaptation
and structural specialization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, NamedTuple, Tuple
from functools import partial

from .assembly import TropicalAssembly
from .competition import CompetitionArena, CompetitionResult
from .learning import HebbianLearning
from ..core.sdr import SDR, SDRConfig
from ..core.polytope import TropicalPolytope, Amoeba
from ..core.padic import PadicTimer


class NetworkConfig(NamedTuple):
    """Configuration for assembly network."""
    base_assemblies: int = 7
    min_assemblies: int = 5
    max_assemblies: int = 9
    sdr_size: int = 2048
    sdr_sparsity: float = 0.02
    max_patterns_per_assembly: int = 200
    amoeba_thickness: float = 2.0


class TropicalAssemblyNetwork:
    """Complete neural assembly network with 7±2 assemblies.
    
    Implements tropical competition, metabolic constraints,
    and adaptive assembly recruitment.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize network.
        
        Args:
            config: Network configuration
        """
        self.config = config or NetworkConfig()
        
        # Core structures
        self.polytope = TropicalPolytope(
            n_vertices=self.config.base_assemblies,
            amoeba_thickness=self.config.amoeba_thickness
        )
        self.amoeba = Amoeba(self.polytope)
        self.timer = PadicTimer()
        
        # SDR configuration
        self.sdr_config = SDRConfig(
            size=self.config.sdr_size,
            sparsity=self.config.sdr_sparsity,
            n_active=int(self.config.sdr_size * self.config.sdr_sparsity)
        )
        
        # Initialize assemblies
        self.assemblies = self._initialize_assemblies()
        self.n_active = self.config.base_assemblies
        
        # Competition and learning
        self.arena = CompetitionArena(self.n_active, self.timer)
        self.learning = HebbianLearning()
        
        # Global metabolic state
        self.global_metabolic_state = 1.0
        
    def _initialize_assemblies(self) -> List[TropicalAssembly]:
        """Initialize assemblies with eigenvalue specialization."""
        assemblies = []
        
        # Eigenvalues for specialization
        eigenvalues = jnp.linspace(2.0, 0.9, self.config.max_assemblies)
        
        for i in range(self.config.max_assemblies):
            assembly = TropicalAssembly(
                index=i,
                eigenvalue=float(eigenvalues[i]),
                sdr_config=self.sdr_config,
                max_patterns=self.config.max_patterns_per_assembly
            )
            assemblies.append(assembly)
            
        return assemblies
        
    def process_input(self, 
                     input_sdr: SDR,
                     learn: bool = True) -> Tuple[int, float, CompetitionResult]:
        """Process single input through competition and learning.
        
        Args:
            input_sdr: Input SDR pattern
            learn: Whether to update patterns
            
        Returns:
            (winner_idx, confidence, full_result)
        """
        # Adjust active assemblies based on metabolic state
        self._adapt_assembly_count()
        
        # Get currently active assemblies
        active_assemblies = self.assemblies[:self.n_active]
        
        # Run competition
        result = self.arena.run_competition(active_assemblies, input_sdr)
        
        # Update assembly states
        for i, assembly in enumerate(active_assemblies):
            won = (i == result.winner_idx)
            assembly.update_state(
                won_competition=won,
                activation=float(result.all_scores[i]),
                phase=self.timer.get_phase_vector()
            )
            
        # Learning phase
        if learn and result.confidence > self.arena.wta.min_confidence:
            self.learning.update(
                assemblies=active_assemblies,
                winner_idx=result.winner_idx,
                input_sdr=input_sdr,
                competition_result=result
            )
            
        return result.winner_idx, result.confidence, result
        
    def _adapt_assembly_count(self):
        """Dynamically adjust number of active assemblies."""
        # Compute current complexity/load
        if len(self.arena.competition_history) > 10:
            recent_confidences = [
                r.confidence for r in self.arena.competition_history[-10:]
            ]
            avg_confidence = np.mean(recent_confidences)
            
            # Low confidence suggests need for more assemblies
            if avg_confidence < 5.0 and self.n_active < self.config.max_assemblies:
                self.n_active += 1
            # High confidence suggests we can use fewer
            elif avg_confidence > 20.0 and self.n_active > self.config.min_assemblies:
                self.n_active -= 1
                
        # Metabolic modulation
        thickness = self.amoeba.modulate_thickness(
            self.global_metabolic_state,
            temperature=37.0
        )
        
        # Thickness determines capacity
        capacity = self.amoeba.assembly_capacity(thickness)
        self.n_active = np.clip(capacity, 
                                self.config.min_assemblies,
                                self.config.max_assemblies)
                                
    def segment_image(self,
                     image_sdrs: List[SDR],
                     positions: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Segment image using assembly competition.
        
        Args:
            image_sdrs: SDR for each pixel/patch
            positions: Spatial positions (for coherence)
            
        Returns:
            (labels, confidences) for each position
        """
        labels = []
        confidences = []
        
        for i, sdr in enumerate(image_sdrs):
            winner, confidence, _ = self.process_input(sdr, learn=False)
            labels.append(winner)
            confidences.append(confidence)
            
        return jnp.array(labels), jnp.array(confidences)
        
    def batch_segment(self,
                     sdr_batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Batch segmentation using vectorized operations.
        
        Args:
            sdr_batch: Dense SDR matrix (n_pixels, sdr_size)
            
        Returns:
            (labels, confidences)
        """
        # Compute all assembly responses
        n_pixels = sdr_batch.shape[0]
        score_matrix = jnp.zeros((n_pixels, self.n_active))
        
        for i, assembly in enumerate(self.assemblies[:self.n_active]):
            # Vectorized response computation
            if len(assembly.patterns) > 0:
                pattern_matrix = jnp.stack([p.dense for p in assembly.patterns])
                overlaps = jnp.dot(sdr_batch, pattern_matrix.T)
                weighted = overlaps * assembly.pattern_weights[None, :]
                scores = jnp.max(weighted, axis=1)
            else:
                scores = jnp.zeros(n_pixels)
                
            score_matrix = score_matrix.at[:, i].set(scores)
            
        # Batch competition
        active_mask = self.timer.get_active_assemblies(self.n_active)
        labels, confidences = self.arena.batch_competition(score_matrix, active_mask)
        
        return labels, confidences
        
    def train_on_batch(self,
                      sdrs: List[SDR],
                      labels: Optional[List[int]] = None):
        """Train network on batch of inputs.
        
        Args:
            sdrs: List of input SDRs
            labels: Optional supervision (not used in unsupervised learning)
        """
        for sdr in sdrs:
            self.process_input(sdr, learn=True)
            
    def get_assembly_statistics(self) -> List[dict]:
        """Get statistics for all assemblies.
        
        Returns:
            List of statistics dictionaries
        """
        return [assembly.get_statistics() 
                for assembly in self.assemblies[:self.n_active]]
                
    def set_metabolic_state(self, state: float):
        """Set global metabolic state.
        
        Args:
            state: Metabolic level (0=depleted, 1=full energy)
        """
        self.global_metabolic_state = np.clip(state, 0.0, 1.0)
        
    def save(self, path: str):
        """Save network state.
        
        Args:
            path: Directory path for saving
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save each assembly
        for i, assembly in enumerate(self.assemblies):
            assembly.save_patterns(os.path.join(path, f'assembly_{i}.npy'))
            
        # Save network config
        config_data = {
            'n_active': self.n_active,
            'global_metabolic_state': self.global_metabolic_state,
            'config': self.config._asdict()
        }
        np.save(os.path.join(path, 'network_config.npy'), config_data)
        
    def load(self, path: str):
        """Load network state.
        
        Args:
            path: Directory path for loading
        """
        import os
        
        # Load assemblies
        for i, assembly in enumerate(self.assemblies):
            pattern_path = os.path.join(path, f'assembly_{i}.npy')
            if os.path.exists(pattern_path):
                assembly.load_patterns(pattern_path)
                
        # Load network config
        config_path = os.path.join(path, 'network_config.npy')
        if os.path.exists(config_path):
            config_data = np.load(config_path, allow_pickle=True).item()
            self.n_active = config_data['n_active']
            self.global_metabolic_state = config_data['global_metabolic_state']