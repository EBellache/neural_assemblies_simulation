"""P-adic temporal structure for neural timing.

Implements the three characteristic timescales (8ms, 27ms, 125ms)
corresponding to 2^3, 3^3, and 5^3 in the p-adic framework.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict
from functools import partial


class PadicPhase(NamedTuple):
    """P-adic phase state for a single prime."""
    prime: int
    power: int 
    current: int  # Current phase (0 to prime^power - 1)
    period: int  # prime^power


class PadicTimer:
    """P-adic timer implementing modular temporal structure.
    
    Three nested timescales create hierarchical phase gating:
    - 8ms (2^3): Dendritic integration windows
    - 27ms (3^3): Assembly selection cycles  
    - 125ms (5^3): Inter-region communication
    """
    
    def __init__(self):
        """Initialize p-adic timer with three primes."""
        self.phases = {
            2: PadicPhase(prime=2, power=3, current=0, period=8),
            3: PadicPhase(prime=3, power=3, current=0, period=27),
            5: PadicPhase(prime=5, power=3, current=0, period=125)
        }
        
        # Precompute gate windows
        self.gate_windows = self._compute_gate_windows()
        
    def _compute_gate_windows(self) -> Dict[str, jnp.ndarray]:
        """Precompute phase windows for gating."""
        windows = {}
        
        # Dendritic integration (first half of 8ms cycle)
        windows['dendritic'] = jnp.arange(4)
        
        # Assembly groups (three 9ms windows in 27ms cycle)
        windows['assembly_group_1'] = jnp.arange(0, 9)
        windows['assembly_group_2'] = jnp.arange(9, 18) 
        windows['assembly_group_3'] = jnp.arange(18, 27)
        
        # Inter-region communication phases
        windows['input_phase'] = jnp.arange(0, 30)
        windows['processing_phase'] = jnp.arange(30, 70)
        windows['output_phase'] = jnp.arange(70, 100)
        windows['consolidation_phase'] = jnp.arange(100, 125)
        
        return windows
        
    @jax.jit
    def tick(self) -> Dict[int, PadicPhase]:
        """Advance all phase counters by one timestep.
        
        Returns:
            Updated phase states
        """
        new_phases = {}
        
        for prime, phase in self.phases.items():
            new_current = (phase.current + 1) % phase.period
            new_phases[prime] = PadicPhase(
                prime=phase.prime,
                power=phase.power,
                current=new_current,
                period=phase.period
            )
            
        self.phases = new_phases
        return new_phases
        
    @jax.jit
    def get_active_gates(self) -> Dict[str, bool]:
        """Determine which gates are currently open.
        
        Returns:
            Dictionary of gate states
        """
        gates = {}
        
        # Dendritic gate (8ms cycle)
        phase_8 = self.phases[2].current
        gates['dendritic'] = phase_8 < 4
        
        # Assembly group selection (27ms cycle)
        phase_27 = self.phases[3].current
        gates['assembly_group'] = phase_27 // 9  # 0, 1, or 2
        
        # Inter-region gates (125ms cycle)
        phase_125 = self.phases[5].current
        gates['input'] = phase_125 < 30
        gates['processing'] = 30 <= phase_125 < 70
        gates['output'] = 70 <= phase_125 < 100
        gates['consolidation'] = 100 <= phase_125 < 125
        
        return gates
        
    @partial(jax.jit, static_argnames=['n_assemblies'])
    def get_active_assemblies(self, n_assemblies: int = 7) -> jnp.ndarray:
        """Determine which assemblies can compete at current phase.
        
        Args:
            n_assemblies: Total number of assemblies (5-9)
            
        Returns:
            Boolean array of active assemblies
        """
        gates = self.get_active_gates()
        active = jnp.zeros(n_assemblies, dtype=bool)
        
        # Distribute assemblies across three groups
        group_size = n_assemblies // 3
        remainder = n_assemblies % 3
        
        group = gates['assembly_group']
        
        if group == 0:
            # First group (plus remainder)
            end_idx = group_size + remainder
            active = active.at[:end_idx].set(True)
        elif group == 1:
            # Second group
            start_idx = group_size + remainder
            end_idx = start_idx + group_size
            active = active.at[start_idx:end_idx].set(True)
        else:  # group == 2
            # Third group
            start_idx = 2 * group_size + remainder
            active = active.at[start_idx:].set(True)
            
        # Apply dendritic gate
        if not gates['dendritic']:
            active = active & False  # No assemblies during refractory
            
        return active
        
    @jax.jit
    def compute_phase_coupling(self, 
                              freq1: float, 
                              freq2: float) -> float:
        """Compute coupling strength between two frequencies.
        
        Cross-frequency coupling is strongest when frequencies
        share p-adic structure.
        
        Args:
            freq1: First frequency (Hz)
            freq2: Second frequency (Hz)
            
        Returns:
            Coupling strength (0-1)
        """
        # Convert frequencies to periods (ms)
        period1 = 1000.0 / freq1
        period2 = 1000.0 / freq2
        
        coupling = 0.0
        
        # Check each prime
        for prime, phase in self.phases.items():
            # Check if periods are close to prime^power multiples
            ratio1 = period1 / phase.period
            ratio2 = period2 / phase.period
            
            # Coupling strongest when both are near integer multiples
            score1 = jnp.exp(-jnp.abs(ratio1 - jnp.round(ratio1)))
            score2 = jnp.exp(-jnp.abs(ratio2 - jnp.round(ratio2)))
            
            coupling += score1 * score2
            
        return coupling / len(self.phases)
        
    @jax.jit
    def theta_gamma_coupling(self) -> Tuple[int, float]:
        """Compute theta-gamma coupling parameters.
        
        Returns:
            (n_gamma_cycles, coupling_strength)
        """
        # Theta ~8Hz (125ms), Gamma ~40Hz (25ms)
        theta_phase = self.phases[5].current  # 125ms cycle
        gamma_phase = self.phases[3].current  # 27ms cycle
        
        # Number of gamma cycles per theta cycle
        n_gamma = 125 // 27  # Approximately 4-5
        
        # Coupling strength based on phase alignment
        phase_diff = jnp.abs(theta_phase / 125 - gamma_phase / 27)
        coupling = jnp.exp(-phase_diff * 10)
        
        return n_gamma, coupling
        
    @jax.jit  
    def is_resonant(self, input_period: int) -> bool:
        """Check if input period resonates with p-adic structure.
        
        Args:
            input_period: Input period in milliseconds
            
        Returns:
            True if resonant
        """
        # Check GCD with each characteristic period
        for phase in self.phases.values():
            if jnp.gcd(input_period, phase.period) > 1:
                return True
        return False
        
    def reset_phase(self, prime: int):
        """Reset specific phase counter (for attention/phase reset).
        
        Args:
            prime: Which prime to reset (2, 3, or 5)
        """
        phase = self.phases[prime]
        self.phases[prime] = PadicPhase(
            prime=phase.prime,
            power=phase.power,
            current=0,
            period=phase.period
        )
        
    def get_phase_vector(self) -> jnp.ndarray:
        """Get current phase as vector for SDR encoding.
        
        Returns:
            Phase vector [phase_2, phase_3, phase_5]
        """
        return jnp.array([
            self.phases[2].current / 8,
            self.phases[3].current / 27,
            self.phases[5].current / 125
        ])
        
    def __repr__(self) -> str:
        return (f"PadicTimer(φ₂={self.phases[2].current}/8, "
                f"φ₃={self.phases[3].current}/27, "
                f"φ₅={self.phases[5].current}/125)")


class PhaseGate:
    """Phase-dependent gating for assembly competition."""
    
    def __init__(self, timer: PadicTimer):
        """Initialize phase gate with p-adic timer.
        
        Args:
            timer: P-adic timer instance
        """
        self.timer = timer
        
    @partial(jax.jit, static_argnames=['self'])
    def gate_assembly_scores(self,
                            scores: jnp.ndarray,
                            metabolic_states: jnp.ndarray) -> jnp.ndarray:
        """Apply phase gates to assembly scores.
        
        Args:
            scores: Raw assembly scores (n_assemblies,)
            metabolic_states: Metabolic state per assembly
            
        Returns:
            Gated scores (inactive assemblies get -inf)
        """
        active = self.timer.get_active_assemblies(len(scores))
        gates = self.timer.get_active_gates()
        
        # Apply assembly group gating
        gated_scores = jnp.where(active, scores, -jnp.inf)
        
        # Apply metabolic modulation during consolidation
        if gates['consolidation']:
            gated_scores = gated_scores * metabolic_states
            
        return gated_scores
        
    @jax.jit
    def compute_phase_precession(self,
                                 position: float,
                                 field_size: float = 1.0) -> float:
        """Compute phase precession through place field.
        
        As position moves through field, phase advances from late
        to early in theta cycle.
        
        Args:
            position: Current position in field (0-1)
            field_size: Size of place field
            
        Returns:
            Phase in theta cycle (0-2π)
        """
        # Normalized position
        x = jnp.clip(position / field_size, 0, 1)
        
        # Phase precession from late (3π/2) to early (π/2)
        phase = (3 * jnp.pi / 2) - x * jnp.pi
        
        return phase % (2 * jnp.pi)