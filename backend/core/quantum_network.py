"""
Quantum Network Topology Generation and Management
Implements various network topologies with quantum channel properties
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.sparse as sp


@dataclass
class QuantumChannel:
    """Represents a quantum channel between two nodes"""
    source: int
    target: int
    fidelity: float  # Channel fidelity
    loss_rate: float  # Photon loss rate
    distance: float  # Physical distance


class QuantumNetwork:
    """Quantum network topology generator and manager"""

    def __init__(self, n_nodes: int, topology_type: str = 'grid', seed: int = 42):
        """
        Initialize quantum network

        Args:
            n_nodes: Number of nodes in the network
            topology_type: Type of topology ('line', 'grid', 'scale_free', 'random_geometric')
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.topology_type = topology_type
        self.seed = seed
        self.graph = None
        self.channels: Dict[Tuple[int, int], QuantumChannel] = {}
        self.node_positions = None

        np.random.seed(seed)
        self._generate_topology()
        self._initialize_quantum_channels()

    def _generate_topology(self):
        """Generate network topology based on type"""
        if self.topology_type == 'line':
            self.graph = self._create_line_topology()
        elif self.topology_type == 'grid':
            self.graph = self._create_grid_topology()
        elif self.topology_type == 'scale_free':
            self.graph = self._create_scale_free_topology()
        elif self.topology_type == 'random_geometric':
            self.graph = self._create_random_geometric_topology()
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")

        # Ensure graph is connected
        if not nx.is_connected(self.graph):
            # Get largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(largest_cc).copy()
            self.n_nodes = len(self.graph.nodes())

    def _create_line_topology(self) -> nx.Graph:
        """Create line/path topology"""
        G = nx.path_graph(self.n_nodes)
        # Set positions for visualization
        self.node_positions = {i: (i, 0, 0) for i in range(self.n_nodes)}
        return G

    def _create_grid_topology(self) -> nx.Graph:
        """Create 2D grid topology"""
        # Determine grid dimensions
        side = int(np.sqrt(self.n_nodes))
        while side * side < self.n_nodes:
            side += 1

        # Create grid
        G = nx.grid_2d_graph(side, side)

        # Relabel nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        # Set 3D positions (z=0 for 2D grid)
        self.node_positions = {}
        for i, (x, y) in enumerate([(i // side, i % side) for i in range(len(G.nodes()))]):
            self.node_positions[i] = (x, y, 0)

        # Trim to exact number of nodes
        if len(G.nodes()) > self.n_nodes:
            nodes_to_remove = list(G.nodes())[self.n_nodes:]
            G.remove_nodes_from(nodes_to_remove)
            for node in nodes_to_remove:
                del self.node_positions[node]

        return G

    def _create_scale_free_topology(self) -> nx.Graph:
        """Create scale-free (Barabási-Albert) topology"""
        m = min(3, self.n_nodes - 1)  # Number of edges to attach from new node
        G = nx.barabasi_albert_graph(self.n_nodes, m, seed=self.seed)

        # Use spring layout for positions
        pos_2d = nx.spring_layout(G, seed=self.seed, dim=3)
        self.node_positions = {i: tuple(pos_2d[i]) for i in G.nodes()}

        return G

    def _create_random_geometric_topology(self) -> nx.Graph:
        """Create random geometric graph"""
        # Adjust radius to ensure connectivity
        radius = 0.3
        G = None

        # Increase radius until graph is connected
        while G is None or not nx.is_connected(G):
            G = nx.random_geometric_graph(self.n_nodes, radius, seed=self.seed, dim=3)
            radius += 0.05
            if radius > 1.0:
                # Fallback to guaranteed connected graph
                G = nx.connected_watts_strogatz_graph(self.n_nodes, 4, 0.3, seed=self.seed)
                pos_2d = nx.spring_layout(G, seed=self.seed, dim=3)
                self.node_positions = {i: tuple(pos_2d[i]) for i in G.nodes()}
                return G

        # Get positions from graph
        self.node_positions = nx.get_node_attributes(G, 'pos')

        return G

    def _initialize_quantum_channels(self):
        """Initialize quantum channels with physical properties"""
        for edge in self.graph.edges():
            source, target = edge

            # Calculate physical distance
            if self.node_positions:
                pos_s = np.array(self.node_positions[source])
                pos_t = np.array(self.node_positions[target])
                distance = np.linalg.norm(pos_s - pos_t)
            else:
                distance = 1.0

            # Channel fidelity decreases with distance
            # F ≈ exp(-distance/L_att) where L_att is attenuation length
            L_att = 10.0  # Attenuation length in arbitrary units
            fidelity = np.exp(-distance / L_att)
            fidelity = max(0.8, min(0.99, fidelity))  # Clamp to reasonable range

            # Loss rate increases with distance
            loss_rate = 0.01 + 0.02 * (distance / L_att)
            loss_rate = min(0.1, loss_rate)  # Cap at 10%

            # Create bidirectional channels
            self.channels[(source, target)] = QuantumChannel(
                source, target, fidelity, loss_rate, distance
            )
            self.channels[(target, source)] = QuantumChannel(
                target, source, fidelity, loss_rate, distance
            )

    def get_adjacency_matrix(self, sparse: bool = True):
        """
        Get adjacency matrix of the network

        Args:
            sparse: If True, return sparse matrix (memory efficient)

        Returns:
            Adjacency matrix (sparse or dense)
        """
        adj = nx.adjacency_matrix(self.graph)
        if sparse:
            return adj
        else:
            return adj.toarray()

    def get_neighbors(self, node: int) -> List[int]:
        """Get list of neighboring nodes"""
        return list(self.graph.neighbors(node))

    def get_channel_fidelity(self, source: int, target: int) -> float:
        """Get fidelity of quantum channel between two nodes"""
        if (source, target) in self.channels:
            return self.channels[(source, target)].fidelity
        return 0.0

    def get_network_statistics(self) -> Dict:
        """Calculate network statistics"""
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else -1,
            'avg_clustering': nx.average_clustering(self.graph),
            'avg_path_length': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else -1,
        }

    def get_node_positions_array(self) -> np.ndarray:
        """Get node positions as numpy array for visualization"""
        if self.node_positions is None:
            # Generate positions using spring layout
            pos_2d = nx.spring_layout(self.graph, seed=self.seed, dim=3)
            self.node_positions = {i: tuple(pos_2d[i]) for i in self.graph.nodes()}

        positions = np.array([self.node_positions[i] for i in sorted(self.graph.nodes())])
        return positions

    def get_edges_array(self) -> np.ndarray:
        """Get edges as numpy array for visualization"""
        return np.array(list(self.graph.edges()))

    def export_to_dict(self) -> Dict:
        """Export network to dictionary for serialization"""
        return {
            'n_nodes': self.n_nodes,
            'topology_type': self.topology_type,
            'edges': list(self.graph.edges()),
            'positions': {int(k): list(v) for k, v in self.node_positions.items()},
            'channels': {
                f"{s}-{t}": {
                    'fidelity': ch.fidelity,
                    'loss_rate': ch.loss_rate,
                    'distance': ch.distance
                }
                for (s, t), ch in self.channels.items()
            },
            'statistics': self.get_network_statistics()
        }


def create_network(n_nodes: int, topology: str = 'grid', seed: int = 42) -> QuantumNetwork:
    """
    Factory function to create quantum network

    Args:
        n_nodes: Number of nodes
        topology: Topology type
        seed: Random seed

    Returns:
        QuantumNetwork instance
    """
    return QuantumNetwork(n_nodes, topology, seed)
