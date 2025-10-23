"""
Quantum Dynamics Simulator
Implements quantum state evolution, decoherence, and measurement using simplified models
For systems with N≤500 nodes, uses efficient approximations instead of full QuTip
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.linalg import expm

from .temporal_spectral import TemporalSpectralState


@dataclass
class QuantumChannelNoise:
    """Noise parameters for quantum channel"""
    photon_loss_rate: float = 0.02
    dark_count_rate: float = 1e-6
    depolarizing_rate: float = 0.01
    dephasing_rate: float = 0.05


class QuantumStateEvolution:
    """
    Quantum state evolution with decoherence
    Uses Lindblad master equation approximation
    """

    def __init__(self,
                 coherence_time_ms: float = 30.0,
                 n_temporal: int = 32,
                 n_spectral: int = 16):
        """
        Initialize quantum state evolution

        Args:
            coherence_time_ms: T2 coherence time in milliseconds
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        self.coherence_time = coherence_time_ms * 1e-3  # Convert to seconds
        self.decoherence_rate = 1.0 / self.coherence_time
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral

    def evolve_state(self,
                     state: TemporalSpectralState,
                     time_step: float,
                     hamiltonian: Optional[np.ndarray] = None) -> TemporalSpectralState:
        """
        Evolve quantum state for one time step

        Args:
            state: Initial quantum state
            time_step: Time step (seconds)
            hamiltonian: Optional Hamiltonian (if None, free evolution)

        Returns:
            Evolved quantum state
        """
        # Apply unitary evolution
        if hamiltonian is not None:
            state = self._apply_unitary_evolution(state, hamiltonian, time_step)

        # Apply decoherence
        state.apply_decoherence(self.decoherence_rate, time_step)

        return state

    def _apply_unitary_evolution(self,
                                 state: TemporalSpectralState,
                                 hamiltonian: np.ndarray,
                                 time_step: float) -> TemporalSpectralState:
        """
        Apply unitary evolution U = exp(-iHt/ℏ)

        Args:
            state: Quantum state
            hamiltonian: Hamiltonian matrix
            time_step: Time step

        Returns:
            Evolved state
        """
        # Get dense state vector
        state_vector = state.to_dense_vector()

        # Compute evolution operator (setting ℏ=1)
        U = expm(-1j * hamiltonian * time_step)

        # Apply evolution
        evolved_vector = U @ state_vector

        # Update state
        new_state = state.copy()
        new_state.from_dense_vector(evolved_vector)

        return new_state


class QuantumMeasurement:
    """
    Quantum measurement with realistic noise
    Implements probe-state solution to measurement paradox
    """

    def __init__(self,
                 probe_fraction: float = 0.08,
                 noise: Optional[QuantumChannelNoise] = None):
        """
        Initialize quantum measurement

        Args:
            probe_fraction: Fraction of states to probe
            noise: Noise parameters
        """
        self.probe_fraction = probe_fraction
        self.noise = noise or QuantumChannelNoise()

    def measure_state(self,
                     state: TemporalSpectralState,
                     is_probe: bool = False) -> Tuple[int, int]:
        """
        Perform quantum measurement

        Args:
            state: Quantum state to measure
            is_probe: If True, this is a probe measurement

        Returns:
            Measured (temporal_bin, spectral_bin)
        """
        # Apply photon loss
        if np.random.random() < self.noise.photon_loss_rate:
            # Photon lost - return invalid measurement
            return (-1, -1)

        # Perform measurement
        temporal_bin, spectral_bin = state.measure(self.probe_fraction)

        # Add dark counts (false detections)
        if np.random.random() < self.noise.dark_count_rate:
            # Random detection
            temporal_bin = np.random.randint(0, state.n_temporal)
            spectral_bin = np.random.randint(0, state.n_spectral)

        return temporal_bin, spectral_bin

    def calculate_qber(self,
                      transmitted_states: List[TemporalSpectralState],
                      received_measurements: List[Tuple[int, int]]) -> float:
        """
        Calculate Quantum Bit Error Rate (QBER)

        Args:
            transmitted_states: List of transmitted quantum states
            received_measurements: List of measured outcomes

        Returns:
            QBER value [0, 1]
        """
        if not transmitted_states or not received_measurements:
            return 1.0

        errors = 0
        total = 0

        for state, measurement in zip(transmitted_states, received_measurements):
            if measurement == (-1, -1):
                # Lost photon - skip
                continue

            # Get most probable mode from state
            prob_dist = state.get_2d_probability_distribution()
            expected_mode = np.unravel_index(np.argmax(prob_dist), prob_dist.shape)

            if expected_mode != measurement:
                errors += 1

            total += 1

        if total == 0:
            return 1.0

        return errors / total


class QuantumChannelSimulator:
    """
    Simulates quantum communication over a noisy channel
    """

    def __init__(self,
                 fidelity: float = 0.95,
                 loss_rate: float = 0.02,
                 coherence_time: float = 30.0,
                 n_temporal: int = 32,
                 n_spectral: int = 16):
        """
        Initialize quantum channel simulator

        Args:
            fidelity: Channel fidelity
            loss_rate: Photon loss rate
            coherence_time: Coherence time (ms)
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        self.fidelity = fidelity
        self.loss_rate = loss_rate
        self.evolution = QuantumStateEvolution(coherence_time, n_temporal, n_spectral)

        # Noise model
        self.noise = QuantumChannelNoise(
            photon_loss_rate=loss_rate,
            dark_count_rate=1e-6,
            depolarizing_rate=(1 - fidelity) / 3,
            dephasing_rate=(1 - fidelity) / 2
        )

        self.measurement = QuantumMeasurement(probe_fraction=0.08, noise=self.noise)

    def transmit_state(self,
                      state: TemporalSpectralState,
                      propagation_time: float = 1e-6) -> TemporalSpectralState:
        """
        Transmit quantum state through channel

        Args:
            state: Input quantum state
            propagation_time: Propagation time (seconds)

        Returns:
            Output quantum state after channel
        """
        # Copy state to avoid modifying original
        output_state = state.copy()

        # Apply photon loss
        output_state.apply_photon_loss(self.loss_rate)

        # Apply decoherence during propagation
        output_state.apply_decoherence(self.evolution.decoherence_rate, propagation_time)

        # Apply depolarizing noise
        output_state = self._apply_depolarizing_noise(output_state, self.noise.depolarizing_rate)

        return output_state

    def _apply_depolarizing_noise(self,
                                  state: TemporalSpectralState,
                                  depolarizing_rate: float) -> TemporalSpectralState:
        """
        Apply depolarizing noise to quantum state

        Args:
            state: Input state
            depolarizing_rate: Depolarizing rate

        Returns:
            Noisy state
        """
        # Simple depolarizing: mix with maximally mixed state
        mixed_prob = depolarizing_rate

        if np.random.random() < mixed_prob:
            # Replace with uniform superposition
            state.set_uniform_superposition()

        return state


class QuantumNetworkSimulator:
    """
    Full quantum network simulator
    Manages quantum states and channels for all nodes
    """

    def __init__(self,
                 n_nodes: int,
                 n_temporal: int = 32,
                 n_spectral: int = 16,
                 coherence_time: float = 30.0):
        """
        Initialize quantum network simulator

        Args:
            n_nodes: Number of nodes
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
            coherence_time: Coherence time (ms)
        """
        self.n_nodes = n_nodes
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral

        # Node states
        self.node_states: Dict[int, TemporalSpectralState] = {}

        # Channel simulators (indexed by edge)
        self.channels: Dict[Tuple[int, int], QuantumChannelSimulator] = {}

        # Measurement system
        self.measurement = QuantumMeasurement(probe_fraction=0.08)

        # Initialize node states
        for i in range(n_nodes):
            state = TemporalSpectralState(n_temporal, n_spectral)
            state.set_uniform_superposition()
            self.node_states[i] = state

    def add_channel(self,
                   source: int,
                   target: int,
                   fidelity: float = 0.95,
                   loss_rate: float = 0.02):
        """
        Add quantum channel between two nodes

        Args:
            source: Source node
            target: Target node
            fidelity: Channel fidelity
            loss_rate: Loss rate
        """
        channel = QuantumChannelSimulator(
            fidelity=fidelity,
            loss_rate=loss_rate,
            coherence_time=30.0,
            n_temporal=self.n_temporal,
            n_spectral=self.n_spectral
        )
        self.channels[(source, target)] = channel
        self.channels[(target, source)] = channel  # Bidirectional

    def set_node_state(self, node_id: int, temporal_bin: int, spectral_bin: int):
        """
        Set node to specific temporal-spectral mode

        Args:
            node_id: Node identifier
            temporal_bin: Temporal bin
            spectral_bin: Spectral bin
        """
        state = TemporalSpectralState(self.n_temporal, self.n_spectral)
        state.set_single_mode(temporal_bin, spectral_bin)
        self.node_states[node_id] = state

    def transmit(self,
                source: int,
                target: int,
                state: Optional[TemporalSpectralState] = None) -> TemporalSpectralState:
        """
        Transmit quantum state from source to target

        Args:
            source: Source node
            target: Target node
            state: State to transmit (if None, use source node state)

        Returns:
            Received state at target
        """
        if state is None:
            state = self.node_states[source]

        # Get channel
        if (source, target) not in self.channels:
            # No channel - return degraded state
            degraded = state.copy()
            degraded.apply_photon_loss(0.5)
            return degraded

        channel = self.channels[(source, target)]
        return channel.transmit_state(state)

    def measure_node(self, node_id: int) -> Tuple[int, int]:
        """
        Measure quantum state at node

        Args:
            node_id: Node to measure

        Returns:
            Measurement outcome (temporal_bin, spectral_bin)
        """
        state = self.node_states[node_id]
        return self.measurement.measure_state(state)

    def calculate_network_fidelity(self) -> float:
        """
        Calculate average fidelity across network

        Returns:
            Average fidelity
        """
        if not self.channels:
            return 1.0

        fidelities = [ch.fidelity for ch in self.channels.values()]
        return np.mean(fidelities)

    def calculate_information_density(self, successful_transmissions: int, total_photons: int) -> float:
        """
        Calculate information density (bits per photon)

        Args:
            successful_transmissions: Number of successful transmissions
            total_photons: Total photons used

        Returns:
            Information density
        """
        if total_photons == 0:
            return 0.0

        # Information per successful transmission
        bits_per_transmission = np.log2(self.n_temporal * self.n_spectral)

        total_bits = successful_transmissions * bits_per_transmission
        return total_bits / total_photons

    def reset_network(self):
        """Reset all node states to uniform superposition"""
        for node_id in self.node_states:
            state = TemporalSpectralState(self.n_temporal, self.n_spectral)
            state.set_uniform_superposition()
            self.node_states[node_id] = state


def create_hamiltonian_free_evolution(n_modes: int) -> np.ndarray:
    """
    Create Hamiltonian for free evolution
    H = Σ ω_k |k⟩⟨k|

    Args:
        n_modes: Number of modes

    Returns:
        Hamiltonian matrix
    """
    # Diagonal Hamiltonian with frequency spectrum
    frequencies = np.arange(n_modes) * 2 * np.pi / n_modes
    H = np.diag(frequencies)
    return H


def create_hamiltonian_interaction(n_modes: int, coupling_strength: float = 0.1) -> np.ndarray:
    """
    Create Hamiltonian with mode interactions
    H = Σ ω_k |k⟩⟨k| + g Σ (|k⟩⟨k+1| + |k+1⟩⟨k|)

    Args:
        n_modes: Number of modes
        coupling_strength: Coupling strength g

    Returns:
        Hamiltonian matrix
    """
    # Free evolution
    H = create_hamiltonian_free_evolution(n_modes)

    # Add nearest-neighbor coupling
    for i in range(n_modes - 1):
        H[i, i+1] = coupling_strength
        H[i+1, i] = coupling_strength

    return H
