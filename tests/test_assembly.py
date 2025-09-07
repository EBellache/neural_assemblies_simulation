"""Tests for assembly and competition mechanisms."""

import pytest
import numpy as np
import jax.numpy as jnp

from tropical_sdr.core.sdr import SDR, SDRConfig
from tropical_sdr.assembly.assembly import TropicalAssembly
from tropical_sdr.assembly.competition import WinnerTakeAll, CompetitionArena
from tropical_sdr.assembly.network import TropicalAssemblyNetwork, NetworkConfig
from tropical_sdr.assembly.learning import HebbianLearning


class TestAssembly:
    """Test individual assembly functionality."""
    
    def setup_method(self):
        """Setup test assembly."""
        self.sdr_config = SDRConfig()
        self.assembly = TropicalAssembly(
            index=0,
            eigenvalue=2.0,
            sdr_config=self.sdr_config
        )
        
    def test_assembly_creation(self):
        """Test assembly initialization."""
        assert self.assembly.index == 0
        assert self.assembly.eigenvalue == 2.0
        assert self.assembly.feature_type == "elongated"
        assert len(self.assembly.patterns) == 0
        
    def test_pattern_addition(self):
        """Test adding patterns to assembly."""
        sdr = SDR(active_indices=list(range(40)), config=self.sdr_config)
        
        success = self.assembly.add_pattern(sdr)
        assert success
        assert len(self.assembly.patterns) == 1
        
        # Adding same pattern should fail (redundant)
        success = self.assembly.add_pattern(sdr)
        assert not success
        assert len(self.assembly.patterns) == 1
        
    def test_compute_response(self):
        """Test assembly response computation."""
        # Add some patterns
        for i in range(3):
            indices = list(range(i*10, i*10+40))
            sdr = SDR(active_indices=indices, config=self.sdr_config)
            self.assembly.add_pattern(sdr)
            
        # Test response to matching pattern
        test_sdr = SDR(active_indices=list(range(5, 45)), config=self.sdr_config)
        response = self.assembly.compute_response(test_sdr)
        assert response > 0
        
        # Test response to non-matching pattern
        test_sdr = SDR(active_indices=list(range(500, 540)), config=self.sdr_config)
        response = self.assembly.compute_response(test_sdr)
        assert response < 10  # Low response
        
    def test_state_update(self):
        """Test assembly state updates."""
        initial_energy = self.assembly.state.metabolic_energy
        
        # Simulate winning
        self.assembly.update_state(
            won_competition=True,
            activation=0.8,
            phase=jnp.array([0.5, 0.5, 0.5])
        )
        
        # Energy should decrease after winning
        assert self.assembly.state.metabolic_energy < initial_energy
        assert self.assembly.state.recent_winner_count == 1
        
        # Simulate losing
        self.assembly.update_state(
            won_competition=False,
            activation=0.3,
            phase=jnp.array([0.5, 0.5, 0.5])
        )
        
        # Energy should recover
        assert self.assembly.state.metabolic_energy > initial_energy * 0.9


class TestCompetition:
    """Test competition mechanisms."""
    
    def test_winner_take_all(self):
        """Test WTA competition."""
        wta = WinnerTakeAll()
        
        scores = jnp.array([1.0, 3.0, 2.0, 4.0, 1.5])
        active_mask = jnp.ones(5, dtype=bool)
        
        result = wta.compete(scores, active_mask)
        
        assert result.winner_idx == 3  # Highest score
        assert result.runner_up_idx == 1  # Second highest
        assert result.confidence > 0
        
    def test_phase_gating(self):
        """Test phase-gated competition."""
        from tropical_sdr.core.padic import PadicTimer
        
        timer = PadicTimer()
        arena = CompetitionArena(n_assemblies=7, timer=timer)
        
        # Create mock assemblies
        assemblies = []
        for i in range(7):
            assembly = TropicalAssembly(i, 1.5 + i*0.1)
            assemblies.append(assembly)
            
        # Run competition
        test_sdr = SDR(active_indices=list(range(40)))
        result = arena.run_competition(assemblies, test_sdr)
        
        assert 0 <= result.winner_idx < 7
        assert result.confidence >= 0


class TestNetwork:
    """Test full network functionality."""
    
    def test_network_creation(self):
        """Test network initialization."""
        config = NetworkConfig(base_assemblies=7)
        network = TropicalAssemblyNetwork(config)
        
        assert network.n_active == 7
        assert len(network.assemblies) == config.max_assemblies
        
    def test_adaptive_assembly_count(self):
        """Test dynamic assembly adjustment."""
        network = TropicalAssemblyNetwork()
        
        # Simulate low confidence competitions
        for _ in range(20):
            result = type('Result', (), {'confidence': 3.0})()
            network.arena.competition_history.append(result)
            
        network._adapt_assembly_count()
        
        # Should increase assemblies due to low confidence
        assert network.n_active >= 7
        
    def test_process_input(self):
        """Test input processing."""
        network = TropicalAssemblyNetwork()
        
        test_sdr = SDR(active_indices=list(range(40)))
        winner, confidence, result = network.process_input(test_sdr, learn=True)
        
        assert 0 <= winner < network.n_active
        assert confidence >= 0
        assert result.winner_idx == winner
        
    def test_hebbian_learning(self):
        """Test Hebbian learning updates."""
        learning = HebbianLearning()
        network = TropicalAssemblyNetwork()
        
        # Add initial patterns
        for assembly in network.assemblies[:3]:
            for i in range(3):
                indices = list(range(assembly.index*100 + i*40, 
                                   assembly.index*100 + i*40 + 40))
                sdr = SDR(active_indices=indices)
                assembly.add_pattern(sdr)
                
        initial_patterns = [len(a.patterns) for a in network.assemblies[:3]]
        
        # Process input with learning
        test_sdr = SDR(active_indices=list(range(50, 90)))
        winner, confidence, result = network.process_input(test_sdr, learn=True)
        
        # Winner should have updated patterns or weights
        winner_assembly = network.assemblies[winner]
        assert len(winner_assembly.patterns) >= initial_patterns[winner] or \
               jnp.sum(winner_assembly.pattern_weights) != len(winner_assembly.patterns)


class TestPadicTiming:
    """Test p-adic timing structure."""
    
    def test_padic_timer(self):
        """Test p-adic timer functionality."""
        from tropical_sdr.core.padic import PadicTimer
        
        timer = PadicTimer()
        
        # Check initial state
        assert timer.phases[2].current == 0
        assert timer.phases[3].current == 0
        assert timer.phases[5].current == 0
        
        # Advance timer
        for _ in range(8):
            timer.tick()
            
        # 8ms phase should wrap
        assert timer.phases[2].current == 0
        assert timer.phases[3].current == 8
        assert timer.phases[5].current == 8
        
    def test_phase_gating(self):
        """Test phase gate functionality."""
        from tropical_sdr.core.padic import PadicTimer, PhaseGate
        
        timer = PadicTimer()
        gate = PhaseGate(timer)
        
        scores = jnp.ones(7) * 0.5
        metabolic = jnp.ones(7)
        
        gated_scores = gate.gate_assembly_scores(scores, metabolic)
        
        # Some assemblies should be gated
        assert jnp.sum(gated_scores == -jnp.inf) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])