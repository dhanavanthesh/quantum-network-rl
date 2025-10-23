"""
Temporal-Spectral State Encoding Module
Implements quantum state representation in time-frequency domain
|ψ⟩ = Σ α(m,k)|m,k,t⟩
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import scipy.sparse as sp


@dataclass
class TemporalSpectralMode:
    """Represents a temporal-spectral mode"""
    temporal_bin: int  # Time bin index (0 to M-1)
    spectral_bin: int  # Frequency bin index (0 to K-1)
    amplitude: complex  # Complex amplitude α(m,k)

    @property
    def probability(self) -> float:
        """Return probability |α|²"""
        return abs(self.amplitude) ** 2


class TemporalSpectralState:
    """
    Quantum state in temporal-spectral representation
    Efficient sparse representation for large mode spaces
    """

    def __init__(self, n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize temporal-spectral state

        Args:
            n_temporal: Number of temporal bins (M)
            n_spectral: Number of spectral bins (K)
        """
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral
        self.n_modes = n_temporal * n_spectral

        # Sparse representation: store only non-zero amplitudes
        # Key: (temporal_bin, spectral_bin), Value: complex amplitude
        self.amplitudes: Dict[Tuple[int, int], complex] = {}

        # Normalization factor
        self._is_normalized = True

    def set_mode(self, temporal_bin: int, spectral_bin: int, amplitude: complex):
        """Set amplitude for a specific mode"""
        if not (0 <= temporal_bin < self.n_temporal):
            raise ValueError(f"Temporal bin {temporal_bin} out of range [0, {self.n_temporal})")
        if not (0 <= spectral_bin < self.n_spectral):
            raise ValueError(f"Spectral bin {spectral_bin} out of range [0, {self.n_spectral})")

        if abs(amplitude) > 1e-10:  # Only store non-zero amplitudes
            self.amplitudes[(temporal_bin, spectral_bin)] = amplitude
        elif (temporal_bin, spectral_bin) in self.amplitudes:
            del self.amplitudes[(temporal_bin, spectral_bin)]

        self._is_normalized = False

    def get_mode(self, temporal_bin: int, spectral_bin: int) -> complex:
        """Get amplitude for a specific mode"""
        return self.amplitudes.get((temporal_bin, spectral_bin), 0.0 + 0.0j)

    def get_probability(self, temporal_bin: int, spectral_bin: int) -> float:
        """Get probability for a specific mode"""
        amp = self.get_mode(temporal_bin, spectral_bin)
        return abs(amp) ** 2

    def normalize(self):
        """Normalize the quantum state"""
        total_prob = sum(abs(amp) ** 2 for amp in self.amplitudes.values())

        if total_prob > 1e-10:
            norm_factor = 1.0 / np.sqrt(total_prob)
            self.amplitudes = {
                key: amp * norm_factor
                for key, amp in self.amplitudes.items()
            }
            self._is_normalized = True
        else:
            # Empty state - set to uniform superposition
            self.set_uniform_superposition()

    def set_uniform_superposition(self):
        """Set state to uniform superposition over all modes"""
        amplitude = 1.0 / np.sqrt(self.n_modes)
        self.amplitudes = {
            (m, k): amplitude
            for m in range(self.n_temporal)
            for k in range(self.n_spectral)
        }
        self._is_normalized = True

    def set_single_mode(self, temporal_bin: int, spectral_bin: int):
        """Set state to single mode (Fock state)"""
        self.amplitudes = {(temporal_bin, spectral_bin): 1.0 + 0.0j}
        self._is_normalized = True

    def set_coherent_state(self, center_temporal: int, center_spectral: int,
                          sigma_t: float = 2.0, sigma_f: float = 2.0):
        """
        Set Gaussian coherent state centered at (center_temporal, center_spectral)

        Args:
            center_temporal: Center temporal bin
            center_spectral: Center spectral bin
            sigma_t: Temporal width
            sigma_f: Spectral width
        """
        self.amplitudes = {}

        for m in range(self.n_temporal):
            for k in range(self.n_spectral):
                # Gaussian envelope
                dt = m - center_temporal
                df = k - center_spectral
                amplitude = np.exp(-(dt**2)/(2*sigma_t**2) - (df**2)/(2*sigma_f**2))

                if amplitude > 1e-3:  # Only store significant amplitudes
                    self.amplitudes[(m, k)] = amplitude + 0.0j

        self.normalize()

    def to_dense_vector(self) -> np.ndarray:
        """Convert to dense complex vector"""
        vector = np.zeros(self.n_modes, dtype=complex)
        for (m, k), amp in self.amplitudes.items():
            idx = m * self.n_spectral + k
            vector[idx] = amp
        return vector

    def from_dense_vector(self, vector: np.ndarray):
        """Load from dense complex vector"""
        self.amplitudes = {}
        for idx, amp in enumerate(vector):
            if abs(amp) > 1e-10:
                m = idx // self.n_spectral
                k = idx % self.n_spectral
                self.amplitudes[(m, k)] = amp
        self._is_normalized = False

    def get_temporal_marginal(self) -> np.ndarray:
        """Get marginal probability distribution over temporal bins"""
        marginal = np.zeros(self.n_temporal)
        for (m, k), amp in self.amplitudes.items():
            marginal[m] += abs(amp) ** 2
        return marginal

    def get_spectral_marginal(self) -> np.ndarray:
        """Get marginal probability distribution over spectral bins"""
        marginal = np.zeros(self.n_spectral)
        for (m, k), amp in self.amplitudes.items():
            marginal[k] += abs(amp) ** 2
        return marginal

    def get_2d_probability_distribution(self) -> np.ndarray:
        """Get 2D probability distribution (temporal x spectral)"""
        dist = np.zeros((self.n_temporal, self.n_spectral))
        for (m, k), amp in self.amplitudes.items():
            dist[m, k] = abs(amp) ** 2
        return dist

    def measure(self, probe_fraction: float = 0.08) -> Tuple[int, int]:
        """
        Perform measurement on the quantum state

        Args:
            probe_fraction: Fraction of modes to probe (for measurement paradox solution)

        Returns:
            Measured (temporal_bin, spectral_bin)
        """
        if not self._is_normalized:
            self.normalize()

        # Get probability distribution
        probs = []
        modes = []
        for (m, k), amp in self.amplitudes.items():
            modes.append((m, k))
            probs.append(abs(amp) ** 2)

        # Normalize probabilities
        probs = np.array(probs)
        probs /= probs.sum()

        # Sample from distribution
        idx = np.random.choice(len(modes), p=probs)
        return modes[idx]

    def apply_decoherence(self, decoherence_rate: float, time_step: float):
        """
        Apply decoherence to the quantum state
        Simple phase damping model

        Args:
            decoherence_rate: Decoherence rate (1/T2)
            time_step: Time step
        """
        decay_factor = np.exp(-decoherence_rate * time_step)

        # Apply decay to off-diagonal coherences
        # For simplicity, apply uniform decay to all amplitudes
        self.amplitudes = {
            key: amp * decay_factor
            for key, amp in self.amplitudes.items()
        }
        self.normalize()

    def apply_photon_loss(self, loss_rate: float):
        """
        Apply photon loss channel

        Args:
            loss_rate: Probability of photon loss
        """
        # Amplitude damping
        survival_prob = 1.0 - loss_rate
        damping_factor = np.sqrt(survival_prob)

        self.amplitudes = {
            key: amp * damping_factor
            for key, amp in self.amplitudes.items()
        }
        self.normalize()

    def fidelity(self, other: 'TemporalSpectralState') -> float:
        """
        Calculate fidelity with another state
        F(ρ, σ) = |⟨ψ|φ⟩|² for pure states

        Args:
            other: Another TemporalSpectralState

        Returns:
            Fidelity [0, 1]
        """
        if not self._is_normalized:
            self.normalize()
        if not other._is_normalized:
            other.normalize()

        # Calculate overlap ⟨ψ|φ⟩
        overlap = 0.0 + 0.0j
        for (m, k), amp1 in self.amplitudes.items():
            amp2 = other.get_mode(m, k)
            overlap += np.conj(amp1) * amp2

        # Fidelity is |overlap|²
        return abs(overlap) ** 2

    def entropy(self) -> float:
        """
        Calculate von Neumann entropy
        S = -Σ p_i log(p_i)

        Returns:
            Entropy in bits
        """
        if not self._is_normalized:
            self.normalize()

        entropy = 0.0
        for amp in self.amplitudes.values():
            p = abs(amp) ** 2
            if p > 1e-10:
                entropy -= p * np.log2(p)

        return entropy

    def copy(self) -> 'TemporalSpectralState':
        """Create a deep copy of the state"""
        new_state = TemporalSpectralState(self.n_temporal, self.n_spectral)
        new_state.amplitudes = self.amplitudes.copy()
        new_state._is_normalized = self._is_normalized
        return new_state


class TemporalSpectralEncoder:
    """Encoder for temporal-spectral quantum states"""

    def __init__(self, n_temporal: int = 32, n_spectral: int = 16):
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral
        self.n_modes = n_temporal * n_spectral

    def encode_classical_data(self, data: np.ndarray) -> TemporalSpectralState:
        """
        Encode classical data into quantum state

        Args:
            data: Classical data (should be normalizable)

        Returns:
            Encoded quantum state
        """
        state = TemporalSpectralState(self.n_temporal, self.n_spectral)

        # Reshape data to temporal x spectral
        if data.shape != (self.n_temporal, self.n_spectral):
            data = data.reshape(self.n_temporal, self.n_spectral)

        # Encode as amplitudes
        for m in range(self.n_temporal):
            for k in range(self.n_spectral):
                if abs(data[m, k]) > 1e-10:
                    state.set_mode(m, k, data[m, k])

        state.normalize()
        return state

    def decode_to_classical(self, state: TemporalSpectralState) -> np.ndarray:
        """
        Decode quantum state to classical data

        Args:
            state: Quantum state

        Returns:
            Classical probability distribution
        """
        return state.get_2d_probability_distribution()

    def create_mode_index_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Create mapping from linear index to (temporal, spectral) bins"""
        mapping = {}
        for m in range(self.n_temporal):
            for k in range(self.n_spectral):
                idx = m * self.n_spectral + k
                mapping[idx] = (m, k)
        return mapping
