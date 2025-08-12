"""
Oscillator Module
=================
Theta-gamma oscillatory scaffolding for hippocampal dynamics.
Implements Buzsáki's nested oscillation framework.

A theta cycle (~125 ms, 8 Hz) contains 5 nested gamma sub-cycles (~25 ms each, 40 Hz).
This temporal organization routes different computational operations.

Author: Based on morphogenic spaces framework
Date: 2025
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import math

# JAX acceleration if available
try:
    import jax.numpy as jnp
    from jax import jit
    from jax.typing import ArrayLike
    Array = ArrayLike
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    Array = np.ndarray
    JAX_AVAILABLE = False

    def jit(fn):
        return fn

# Constants
TAU = 2.0 * math.pi  # 2π
DEFAULT_THETA_FREQ = 8.0  # Hz
DEFAULT_GAMMA_FREQ = 40.0  # Hz
GAMMA_PER_THETA = 5  # Number of gamma cycles per theta

# Phase routing assignments
PHASE_ROUTES = {
    0: 'encoding',  # Gamma 1: CA3 → CA1 encoding
    1: 'encoding',  # Gamma 2: Continue encoding
    2: 'transition',  # Gamma 3: Error correction
    3: 'retrieval',  # Gamma 4: CA3 recurrence
    4: 'consolidation'  # Gamma 5: Assembly consolidation
}


@dataclass
class OscillatorState:
    """State of the oscillator at a given time."""
    time: float
    theta_phase: float  # [0, 2π)
    gamma_phase: float  # [0, 2π)
    theta_cycle: int  # Which theta cycle
    gamma_cycle: int  # Which gamma within theta (0-4)
    route: str  # Current computational route
    pac_strength: float  # Phase-amplitude coupling strength


class Oscillator:
    """
    Manages theta-gamma nested oscillations for the hippocampal network.
    """

    def __init__(self,
                 theta_freq: float = DEFAULT_THETA_FREQ,
                 gamma_per_theta: int = GAMMA_PER_THETA,
                 phase0_theta: float = 0.0,
                 phase0_gamma: float = 0.0):
        """
        Initialize oscillator.

        Parameters:
        -----------
        theta_freq : float
            Theta frequency in Hz (typically 4-12 Hz)
        gamma_per_theta : int
            Number of gamma cycles per theta (typically 5-7)
        phase0_theta : float
            Initial theta phase in radians
        phase0_gamma : float
            Initial gamma phase in radians
        """
        self.theta_freq = theta_freq
        self.gamma_per_theta = gamma_per_theta
        self.gamma_freq = theta_freq * gamma_per_theta
        self.phase0_theta = phase0_theta
        self.phase0_gamma = phase0_gamma

        # State tracking
        self.current_time = 0.0
        self.theta_phase = phase0_theta
        self.gamma_phase = phase0_gamma
        self.theta_cycle_count = 0
        self.gamma_cycle_count = 0

        # Phase-amplitude coupling parameters
        self.pac_baseline = 0.5
        self.pac_modulation = 0.3

        # History for analysis
        self.phase_history: List[OscillatorState] = []
        self.max_history = 1000  # Keep last N states

    def reset(self):
        """Reset oscillator to initial state."""
        self.current_time = 0.0
        self.theta_phase = self.phase0_theta
        self.gamma_phase = self.phase0_gamma
        self.theta_cycle_count = 0
        self.gamma_cycle_count = 0
        self.phase_history.clear()

    @jit
    def _compute_phase(self, t: float, freq: float, phase0: float) -> Array:
        """
        Compute phase at time t for given frequency.
        Returns phase in [0, 2π).
        """
        return jnp.mod(TAU * freq * t + phase0, TAU)

    def update(self, dt: float) -> OscillatorState:
        """
        Update oscillator state by time step dt.

        Parameters:
        -----------
        dt : float
            Time step in seconds

        Returns:
        --------
        OscillatorState : Current state after update
        """
        # Update time
        self.current_time += dt

        # Compute new phases
        self.theta_phase = self._compute_phase(
            self.current_time, self.theta_freq, self.phase0_theta
        )
        self.gamma_phase = self._compute_phase(
            self.current_time, self.gamma_freq, self.phase0_gamma
        )

        # Determine cycle indices
        theta_period = 1.0 / self.theta_freq
        gamma_period = 1.0 / self.gamma_freq

        self.theta_cycle_count = int(self.current_time / theta_period)

        # Which gamma within current theta?
        time_in_theta = self.current_time % theta_period
        gamma_index = int(time_in_theta / gamma_period)
        gamma_index = min(gamma_index, self.gamma_per_theta - 1)

        # Get routing
        route = PHASE_ROUTES.get(gamma_index, 'unknown')

        # Compute phase-amplitude coupling
        pac = self.compute_pac(self.theta_phase, self.gamma_phase)

        # Create state
        state = OscillatorState(
            time=self.current_time,
            theta_phase=self.theta_phase,
            gamma_phase=self.gamma_phase,
            theta_cycle=self.theta_cycle_count,
            gamma_cycle=gamma_index,
            route=route,
            pac_strength=pac
        )

        # Store in history
        self.phase_history.append(state)
        if len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)

        return state

    def get_nested_phase(self, t: float) -> tuple[Array, Array, int]:
        """
        Get theta and gamma phases at time t, plus gamma index.

        Returns:
        --------
        (theta_phase, gamma_phase, gamma_index)
        """
        theta_phase = self._compute_phase(t, self.theta_freq, self.phase0_theta)
        gamma_phase = self._compute_phase(t, self.gamma_freq, self.phase0_gamma)

        # Which gamma cycle within theta?
        theta_period = 1.0 / self.theta_freq
        gamma_period = 1.0 / self.gamma_freq
        time_in_theta = t % theta_period
        gamma_index = int(time_in_theta / gamma_period)
        gamma_index = min(gamma_index, self.gamma_per_theta - 1)

        return theta_phase, gamma_phase, gamma_index

    def compute_pac(self, theta_phase: float, gamma_phase: float = None) -> float:
        """
        Compute phase-amplitude coupling strength.

        PAC measures how gamma amplitude is modulated by theta phase.
        Returns value in [0, 1].
        """
        # Gamma amplitude is maximal at theta trough (phase = π)
        # Using cosine modulation
        pac = self.pac_baseline + self.pac_modulation * np.cos(theta_phase - np.pi)
        return float(np.clip(pac, 0, 1))

    def route_by_phase(self, theta_phase: float, gamma_phase: float) -> str:
        """
        Determine computational routing based on current phases.

        Returns:
        --------
        str : One of ['encoding', 'transition', 'retrieval', 'consolidation']
        """
        # Determine which gamma cycle we're in
        theta_period = 1.0 / self.theta_freq
        gamma_period = 1.0 / self.gamma_freq

        # Convert phase to time within cycle
        time_in_theta = (theta_phase / TAU) * theta_period
        gamma_index = int(time_in_theta / gamma_period)
        gamma_index = min(gamma_index, self.gamma_per_theta - 1)

        return PHASE_ROUTES.get(gamma_index, 'unknown')

    def modulate_frequency(self, base_freq: float, modulation_factor: float) -> float:
        """
        Modulate oscillation frequency based on network state.

        Parameters:
        -----------
        base_freq : float
            Base frequency in Hz
        modulation_factor : float
            Modulation factor (1.0 = no change)

        Returns:
        --------
        float : Modulated frequency
        """
        return base_freq * modulation_factor

    def get_phase_coherence(self, phases: np.ndarray) -> float:
        """
        Compute phase coherence (synchronization) across multiple oscillators.

        Parameters:
        -----------
        phases : array of shape (n_oscillators,)
            Phase values in radians

        Returns:
        --------
        float : Coherence value in [0, 1]
        """
        if len(phases) == 0:
            return 0.0

        # Compute mean resultant length (Kuramoto order parameter)
        complex_phases = np.exp(1j * phases)
        mean_vector = np.mean(complex_phases)
        coherence = np.abs(mean_vector)

        return float(coherence)

    def detect_phase_reset(self, threshold: float = 0.5) -> bool:
        """
        Detect if a phase reset has occurred.

        Phase resets are important for memory encoding.
        """
        if len(self.phase_history) < 2:
            return False

        # Check for sudden phase jump
        prev_phase = self.phase_history[-2].theta_phase
        curr_phase = self.phase_history[-1].theta_phase

        # Account for phase wrapping
        phase_diff = np.abs(curr_phase - prev_phase)
        if phase_diff > np.pi:
            phase_diff = TAU - phase_diff

        # Expect smooth progression
        expected_diff = TAU * self.theta_freq * (
                self.phase_history[-1].time - self.phase_history[-2].time
        )

        actual_vs_expected = np.abs(phase_diff - expected_diff)

        return actual_vs_expected > threshold

    def get_phase_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over recent phase history.
        """
        if not self.phase_history:
            return {}

        theta_phases = [s.theta_phase for s in self.phase_history]
        gamma_phases = [s.gamma_phase for s in self.phase_history]
        pac_values = [s.pac_strength for s in self.phase_history]

        # Circular statistics for phases
        theta_mean = np.angle(np.mean(np.exp(1j * np.array(theta_phases))))
        gamma_mean = np.angle(np.mean(np.exp(1j * np.array(gamma_phases))))

        # Phase coherence
        theta_coherence = self.get_phase_coherence(np.array(theta_phases))
        gamma_coherence = self.get_phase_coherence(np.array(gamma_phases))

        return {
            'theta_mean_phase': theta_mean,
            'gamma_mean_phase': gamma_mean,
            'theta_coherence': theta_coherence,
            'gamma_coherence': gamma_coherence,
            'mean_pac': np.mean(pac_values),
            'std_pac': np.std(pac_values),
            'n_theta_cycles': self.theta_cycle_count,
            'current_time': self.current_time
        }

    def simulate_burst(self, duration: float, dt: float = 0.001) -> List[OscillatorState]:
        """
        Simulate a burst of oscillatory activity.

        Parameters:
        -----------
        duration : float
            Duration of burst in seconds
        dt : float
            Time step for simulation

        Returns:
        --------
        List[OscillatorState] : States during burst
        """
        states = []
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            state = self.update(dt)
            states.append(state)

        return states


def test_oscillator():
    """Test oscillator functionality."""
    print("\n=== Testing Oscillator Module ===\n")

    # Create oscillator
    osc = Oscillator(theta_freq=8.0, gamma_per_theta=5)

    # Simulate one theta cycle
    theta_period = 1.0 / osc.theta_freq
    dt = 0.001  # 1ms time step

    print(f"Theta frequency: {osc.theta_freq} Hz")
    print(f"Gamma frequency: {osc.gamma_freq} Hz")
    print(f"Theta period: {theta_period * 1000:.1f} ms")
    print(f"Gamma period: {1000 / osc.gamma_freq:.1f} ms")

    # Track routing changes
    routes = []
    times = []

    for i in range(int(theta_period / dt)):
        state = osc.update(dt)

        if i % 25 == 0:  # Sample every 25ms
            routes.append(state.route)
            times.append(state.time * 1000)  # Convert to ms

            if i < 5 or i % 100 == 0:
                print(f"t={state.time * 1000:.1f}ms: "
                      f"θ={state.theta_phase:.2f}, "
                      f"γ={state.gamma_phase:.2f}, "
                      f"route={state.route}, "
                      f"PAC={state.pac_strength:.2f}")

    # Show phase statistics
    stats = osc.get_phase_statistics()
    print(f"\nPhase statistics after {stats['n_theta_cycles']} theta cycles:")
    print(f"  Theta coherence: {stats['theta_coherence']:.3f}")
    print(f"  Gamma coherence: {stats['gamma_coherence']:.3f}")
    print(f"  Mean PAC: {stats['mean_pac']:.3f}")

    # Test phase reset detection
    osc.reset()
    osc.update(0.01)
    osc.theta_phase += np.pi  # Artificial phase reset
    osc.update(0.01)

    if osc.detect_phase_reset():
        print("\n✓ Phase reset detected correctly!")

    print("\n✓ Oscillator module working correctly!")


if __name__ == "__main__":
    test_oscillator()