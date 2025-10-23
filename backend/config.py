"""
Quantum Network Simulator Configuration
Optimized for standard desktop systems (8-16GB RAM)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class NetworkConfig:
    """Network topology configuration"""
    # Network sizes (optimized for 8-16GB RAM)
    MAX_NODES: int = 500
    DEFAULT_SMALL: int = 50
    DEFAULT_MEDIUM: int = 100
    DEFAULT_LARGE: int = 300

    # Topology types
    TOPOLOGY_TYPES = ['line', 'grid', 'scale_free', 'random_geometric']

    # Grid dimensions
    GRID_2D_MAX: Tuple[int, int] = (20, 20)  # 400 nodes max

    # Scale-free parameters
    SCALE_FREE_M: int = 3  # Number of edges to attach from new node

    # Random geometric parameters
    GEOMETRIC_RADIUS: float = 0.3
    GEOMETRIC_DIM: int = 2


@dataclass
class QuantumConfig:
    """Quantum simulation parameters"""
    # Temporal-spectral modes
    TEMPORAL_BINS_OPTIONS = [16, 32]
    SPECTRAL_BINS_OPTIONS = [8, 16]
    DEFAULT_TEMPORAL: int = 32
    DEFAULT_SPECTRAL: int = 16

    # Total modes
    @property
    def TOTAL_MODES(self) -> int:
        return self.DEFAULT_TEMPORAL * self.DEFAULT_SPECTRAL

    # Quantum parameters
    COHERENCE_TIME_MS: float = 30.0  # T2 coherence time
    DECOHERENCE_RATE: float = 1.0 / 30.0  # 1/T2

    # Quantum state dimensions
    MAX_QUBITS: int = 12  # Memory constraint for density matrices

    # Probe fraction for measurement paradox solution
    PROBE_FRACTION: float = 0.08

    # QBER threshold for quantum security
    QBER_THRESHOLD: float = 0.11  # Standard QKD threshold

    # Photon loss and noise
    PHOTON_LOSS_RATE: float = 0.02
    DARK_COUNT_RATE: float = 1e-6


@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # DQN Architecture
    HIDDEN_LAYERS = [256, 128, 64]

    # Learning parameters
    LEARNING_RATE: float = 0.15
    DISCOUNT_FACTOR: float = 0.92

    # Exploration
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.995

    # Training
    BATCH_SIZE: int = 64  # Can use 32 for very large networks
    REPLAY_BUFFER_SIZE: int = 10000
    TARGET_UPDATE_FREQ: int = 10

    # Epochs
    TRAINING_EPOCHS_SMALL: int = 150
    TRAINING_EPOCHS_MEDIUM: int = 200
    TRAINING_EPOCHS_LARGE: int = 250

    # Convergence criteria
    CONVERGENCE_VARIANCE: float = 0.02  # 2% variance
    CONVERGENCE_WINDOW: int = 10  # epochs

    # Checkpoint frequency
    CHECKPOINT_FREQ: int = 25


@dataclass
class SimulationConfig:
    """Simulation runtime configuration"""
    # Number of independent runs
    NUM_RUNS: int = 50

    # CPU cores (leave 2 for OS)
    NUM_CORES: int = 4

    # Time limits (seconds)
    TIMEOUT_SMALL: int = 1500  # 25 minutes
    TIMEOUT_MEDIUM: int = 4500  # 75 minutes
    TIMEOUT_LARGE: int = 18000  # 5 hours

    # Memory management
    CLEAR_CACHE_FREQ: int = 10  # Clear every N epochs

    # Random seeds for reproducibility
    RANDOM_SEED: int = 42


@dataclass
class MetricsConfig:
    """Performance metrics configuration"""
    # Information density calculation
    BITS_PER_PHOTON_IDEAL: float = 1.0

    # Network efficiency baseline
    EFFICIENCY_BASELINE_TDMA: float = 0.6

    # Metrics to track
    TRACKED_METRICS = [
        'information_density',
        'network_efficiency',
        'collision_rate',
        'convergence_rate',
        'qber',
        'throughput',
        'latency'
    ]

    # Statistical confidence
    CONFIDENCE_LEVEL: float = 0.95


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    # Plot styles
    FIGSIZE: Tuple[int, int] = (12, 8)
    DPI: int = 100

    # Colors
    COLOR_MAP = 'viridis'
    NODE_COLOR_IDLE = '#3498db'  # Blue
    NODE_COLOR_ACTIVE = '#2ecc71'  # Green
    NODE_COLOR_COLLISION = '#e74c3c'  # Red
    NODE_COLOR_LEARNING = '#f39c12'  # Yellow

    # 3D visualization
    NODE_SIZE_3D: float = 20.0
    EDGE_WIDTH_3D: float = 2.0

    # Animation
    FRAME_RATE: int = 30
    ANIMATION_INTERVAL_MS: int = 100


class Config:
    """Main configuration class combining all settings"""
    def __init__(self):
        self.network = NetworkConfig()
        self.quantum = QuantumConfig()
        self.rl = RLConfig()
        self.simulation = SimulationConfig()
        self.metrics = MetricsConfig()
        self.visualization = VisualizationConfig()

    def get_optimal_config_for_network_size(self, n_nodes: int) -> dict:
        """Get optimized configuration based on network size"""
        if n_nodes <= 50:
            return {
                'temporal_bins': 32,
                'spectral_bins': 16,
                'batch_size': 64,
                'epochs': self.rl.TRAINING_EPOCHS_SMALL,
                'timeout': self.simulation.TIMEOUT_SMALL
            }
        elif n_nodes <= 100:
            return {
                'temporal_bins': 32,
                'spectral_bins': 16,
                'batch_size': 64,
                'epochs': self.rl.TRAINING_EPOCHS_MEDIUM,
                'timeout': self.simulation.TIMEOUT_MEDIUM
            }
        elif n_nodes <= 300:
            return {
                'temporal_bins': 16,
                'spectral_bins': 8,
                'batch_size': 32,
                'epochs': self.rl.TRAINING_EPOCHS_LARGE,
                'timeout': self.simulation.TIMEOUT_LARGE
            }
        else:
            raise ValueError(f"Network size {n_nodes} exceeds maximum {self.network.MAX_NODES}")


# Global configuration instance
config = Config()
