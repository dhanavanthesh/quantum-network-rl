"""
Baseline Algorithms for Comparison
Implements TDMA and Fixed Time-Bin allocation strategies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import networkx as nx


class BaselineAlgorithm:
    """Base class for baseline algorithms"""

    def __init__(self, n_nodes: int, n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize baseline algorithm

        Args:
            n_nodes: Number of nodes
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        self.n_nodes = n_nodes
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral
        self.n_modes = n_temporal * n_spectral

    def select_mode(self, node_id: int, time_step: int, network_state: Dict) -> Tuple[int, int]:
        """
        Select temporal-spectral mode for a node

        Args:
            node_id: Node identifier
            time_step: Current time step
            network_state: Current network state

        Returns:
            (temporal_bin, spectral_bin)
        """
        raise NotImplementedError


class TDMABaseline(BaselineAlgorithm):
    """
    Time Division Multiple Access (TDMA) baseline
    Each node gets exclusive access in round-robin fashion
    """

    def __init__(self, n_nodes: int, n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize TDMA baseline

        Args:
            n_nodes: Number of nodes
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        super().__init__(n_nodes, n_temporal, n_spectral)

        # Pre-allocate time slots to nodes
        self.node_schedule = self._create_tdma_schedule()

    def _create_tdma_schedule(self) -> Dict[int, List[int]]:
        """
        Create TDMA schedule

        Returns:
            Dictionary mapping node_id to list of assigned temporal bins
        """
        schedule = {node_id: [] for node_id in range(self.n_nodes)}

        # Distribute temporal bins among nodes
        bins_per_node = self.n_temporal // self.n_nodes
        remainder = self.n_temporal % self.n_nodes

        current_bin = 0
        for node_id in range(self.n_nodes):
            # Allocate bins
            n_bins = bins_per_node + (1 if node_id < remainder else 0)

            for _ in range(n_bins):
                schedule[node_id].append(current_bin)
                current_bin += 1

        return schedule

    def select_mode(self, node_id: int, time_step: int, network_state: Dict) -> Tuple[int, int]:
        """
        Select mode based on TDMA schedule

        Args:
            node_id: Node identifier
            time_step: Current time step
            network_state: Network state (unused)

        Returns:
            (temporal_bin, spectral_bin)
        """
        # Get assigned temporal bins
        assigned_bins = self.node_schedule.get(node_id, [0])

        if not assigned_bins:
            # Fallback: use node_id as temporal bin
            temporal_bin = node_id % self.n_temporal
        else:
            # Cycle through assigned bins
            temporal_bin = assigned_bins[time_step % len(assigned_bins)]

        # Use full spectral bandwidth (randomly select spectral bin)
        spectral_bin = np.random.randint(0, self.n_spectral)

        return temporal_bin, spectral_bin


class FixedTimeBinBaseline(BaselineAlgorithm):
    """
    Fixed Time-Bin baseline
    Each node is permanently assigned to a fixed temporal-spectral mode
    """

    def __init__(self, n_nodes: int, n_temporal: int = 32, n_spectral: int = 16,
                 assignment_strategy: str = 'sequential'):
        """
        Initialize Fixed Time-Bin baseline

        Args:
            n_nodes: Number of nodes
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
            assignment_strategy: How to assign modes ('sequential', 'random', 'optimized')
        """
        super().__init__(n_nodes, n_temporal, n_spectral)
        self.assignment_strategy = assignment_strategy

        # Pre-assign modes to nodes
        self.node_assignments = self._create_fixed_assignments()

    def _create_fixed_assignments(self) -> Dict[int, Tuple[int, int]]:
        """
        Create fixed mode assignments

        Returns:
            Dictionary mapping node_id to (temporal_bin, spectral_bin)
        """
        assignments = {}

        if self.assignment_strategy == 'sequential':
            # Sequentially assign modes
            for node_id in range(self.n_nodes):
                mode_idx = node_id % self.n_modes
                temporal_bin = mode_idx // self.n_spectral
                spectral_bin = mode_idx % self.n_spectral
                assignments[node_id] = (temporal_bin, spectral_bin)

        elif self.assignment_strategy == 'random':
            # Randomly assign modes (with replacement)
            for node_id in range(self.n_nodes):
                temporal_bin = np.random.randint(0, self.n_temporal)
                spectral_bin = np.random.randint(0, self.n_spectral)
                assignments[node_id] = (temporal_bin, spectral_bin)

        elif self.assignment_strategy == 'optimized':
            # Try to minimize collisions by spreading nodes
            # Use a greedy approach
            used_modes = set()
            for node_id in range(self.n_nodes):
                # Find least used mode
                best_mode = None
                min_usage = float('inf')

                for t in range(self.n_temporal):
                    for s in range(self.n_spectral):
                        mode = (t, s)
                        usage = sum(1 for m in used_modes if m == mode)
                        if usage < min_usage:
                            min_usage = usage
                            best_mode = mode

                assignments[node_id] = best_mode
                used_modes.add(best_mode)

        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")

        return assignments

    def select_mode(self, node_id: int, time_step: int, network_state: Dict) -> Tuple[int, int]:
        """
        Select fixed mode for node

        Args:
            node_id: Node identifier
            time_step: Current time step (unused)
            network_state: Network state (unused)

        Returns:
            (temporal_bin, spectral_bin)
        """
        if node_id in self.node_assignments:
            return self.node_assignments[node_id]
        else:
            # Fallback
            temporal_bin = node_id % self.n_temporal
            spectral_bin = 0
            return temporal_bin, spectral_bin


class ColoringBaseline(BaselineAlgorithm):
    """
    Graph Coloring baseline
    Assigns modes based on graph coloring to minimize neighbor collisions
    """

    def __init__(self, n_nodes: int, graph: nx.Graph,
                 n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize Coloring baseline

        Args:
            n_nodes: Number of nodes
            graph: Network graph
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        super().__init__(n_nodes, n_temporal, n_spectral)
        self.graph = graph

        # Compute graph coloring
        self.node_colors = self._compute_graph_coloring()

        # Map colors to modes
        self.color_to_mode = self._assign_colors_to_modes()

    def _compute_graph_coloring(self) -> Dict[int, int]:
        """
        Compute graph coloring using greedy algorithm

        Returns:
            Dictionary mapping node_id to color
        """
        # Use NetworkX greedy coloring
        coloring = nx.greedy_color(self.graph, strategy='largest_first')
        return coloring

    def _assign_colors_to_modes(self) -> Dict[int, Tuple[int, int]]:
        """
        Assign each color to a temporal-spectral mode

        Returns:
            Dictionary mapping color to (temporal_bin, spectral_bin)
        """
        n_colors = max(self.node_colors.values()) + 1
        color_to_mode = {}

        for color in range(n_colors):
            mode_idx = color % self.n_modes
            temporal_bin = mode_idx // self.n_spectral
            spectral_bin = mode_idx % self.n_spectral
            color_to_mode[color] = (temporal_bin, spectral_bin)

        return color_to_mode

    def select_mode(self, node_id: int, time_step: int, network_state: Dict) -> Tuple[int, int]:
        """
        Select mode based on graph coloring

        Args:
            node_id: Node identifier
            time_step: Current time step (unused)
            network_state: Network state (unused)

        Returns:
            (temporal_bin, spectral_bin)
        """
        color = self.node_colors.get(node_id, 0)
        return self.color_to_mode.get(color, (0, 0))


class RandomBaseline(BaselineAlgorithm):
    """
    Random baseline - randomly select modes
    """

    def __init__(self, n_nodes: int, n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize Random baseline

        Args:
            n_nodes: Number of nodes
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        super().__init__(n_nodes, n_temporal, n_spectral)

    def select_mode(self, node_id: int, time_step: int, network_state: Dict) -> Tuple[int, int]:
        """
        Randomly select mode

        Args:
            node_id: Node identifier
            time_step: Current time step (unused)
            network_state: Network state (unused)

        Returns:
            (temporal_bin, spectral_bin)
        """
        temporal_bin = np.random.randint(0, self.n_temporal)
        spectral_bin = np.random.randint(0, self.n_spectral)
        return temporal_bin, spectral_bin


def create_baseline(baseline_type: str,
                   n_nodes: int,
                   graph: Optional[nx.Graph] = None,
                   n_temporal: int = 32,
                   n_spectral: int = 16,
                   **kwargs) -> BaselineAlgorithm:
    """
    Factory function to create baseline algorithm

    Args:
        baseline_type: Type of baseline ('tdma', 'fixed', 'coloring', 'random')
        n_nodes: Number of nodes
        graph: Network graph (required for coloring)
        n_temporal: Number of temporal bins
        n_spectral: Number of spectral bins
        **kwargs: Additional arguments

    Returns:
        Baseline algorithm instance
    """
    if baseline_type.lower() == 'tdma':
        return TDMABaseline(n_nodes, n_temporal, n_spectral)

    elif baseline_type.lower() in ['fixed', 'fixed_time_bin']:
        strategy = kwargs.get('assignment_strategy', 'sequential')
        return FixedTimeBinBaseline(n_nodes, n_temporal, n_spectral, strategy)

    elif baseline_type.lower() == 'coloring':
        if graph is None:
            raise ValueError("Graph required for coloring baseline")
        return ColoringBaseline(n_nodes, graph, n_temporal, n_spectral)

    elif baseline_type.lower() == 'random':
        return RandomBaseline(n_nodes, n_temporal, n_spectral)

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


class BaselineEvaluator:
    """Evaluator for baseline algorithms"""

    def __init__(self, network_graph: nx.Graph,
                 n_temporal: int = 32,
                 n_spectral: int = 16):
        """
        Initialize baseline evaluator

        Args:
            network_graph: Network graph
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        self.graph = network_graph
        self.n_nodes = len(network_graph.nodes())
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral

    def evaluate_baseline(self,
                         baseline: BaselineAlgorithm,
                         n_time_steps: int = 1000) -> Dict:
        """
        Evaluate baseline algorithm

        Args:
            baseline: Baseline algorithm
            n_time_steps: Number of time steps to simulate

        Returns:
            Evaluation results
        """
        collision_count = 0
        total_transmissions = 0

        # Track mode usage per time step
        collision_history = []

        for t in range(n_time_steps):
            # Get mode selections for all nodes
            node_modes = {}
            for node_id in range(self.n_nodes):
                mode = baseline.select_mode(node_id, t, {})
                node_modes[node_id] = mode

            # Check for collisions on edges
            step_collisions = 0
            for edge in self.graph.edges():
                node1, node2 = edge
                if node_modes[node1] == node_modes[node2]:
                    collision_count += 1
                    step_collisions += 1

            collision_history.append(step_collisions)
            total_transmissions += len(self.graph.edges())

        # Calculate metrics
        collision_rate = collision_count / total_transmissions if total_transmissions > 0 else 0
        network_efficiency = 1.0 - collision_rate

        return {
            'collision_rate': collision_rate,
            'network_efficiency': network_efficiency,
            'total_collisions': collision_count,
            'total_transmissions': total_transmissions,
            'avg_collisions_per_step': np.mean(collision_history),
            'std_collisions_per_step': np.std(collision_history)
        }


class BaselineComparison:
    """Compare RL approach with baseline algorithms"""

    def __init__(self, config):
        """
        Initialize baseline comparison

        Args:
            config: Simulation configuration
        """
        from .quantum_network import create_network

        self.config = config

        # Create network
        self.network = create_network(
            config.n_nodes,
            config.topology,
            config.random_seed
        )

        # Baseline evaluator
        self.evaluator = BaselineEvaluator(
            self.network.graph,
            config.n_temporal,
            config.n_spectral
        )

    def compare_all_baselines(self) -> Dict[str, Dict]:
        """
        Compare all baseline algorithms

        Returns:
            Comparison results
        """
        results = {}

        baseline_types = ['tdma', 'fixed', 'random']

        for baseline_type in baseline_types:
            print(f"Evaluating {baseline_type} baseline...")

            baseline = create_baseline(
                baseline_type,
                self.config.n_nodes,
                self.network.graph,
                self.config.n_temporal,
                self.config.n_spectral
            )

            result = self.evaluator.evaluate_baseline(
                baseline,
                n_time_steps=self.config.episode_length * 10
            )

            results[baseline_type] = result

        return results
