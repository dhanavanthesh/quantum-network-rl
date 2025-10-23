"""
Main Simulation Orchestrator
Coordinates quantum network simulation with distributed RL
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

from .quantum_network import QuantumNetwork, create_network
from .temporal_spectral import TemporalSpectralState, TemporalSpectralEncoder
from .rl_agent import DistributedQLearningAgent, BeaconBroadcaster
from .quantum_simulator import QuantumNetworkSimulator
from .metrics import MetricsCollector, NetworkMetrics
from .baselines import create_baseline, BaselineEvaluator


@dataclass
class SimulationConfig:
    """Configuration for a simulation run"""
    n_nodes: int = 100
    topology: str = 'grid'
    n_temporal: int = 32
    n_spectral: int = 16
    n_episodes: int = 200
    episode_length: int = 100
    learning_rate: float = 0.15
    discount_factor: float = 0.92
    batch_size: int = 64
    probe_fraction: float = 0.08
    random_seed: int = 42


class QuantumNetworkSimulation:
    """Main simulation coordinator"""

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation

        Args:
            config: Simulation configuration
        """
        self.config = config
        np.random.seed(config.random_seed)

        # Create network topology
        self.network = create_network(
            config.n_nodes,
            config.topology,
            config.random_seed
        )

        # Create quantum simulator
        self.quantum_sim = QuantumNetworkSimulator(
            config.n_nodes,
            config.n_temporal,
            config.n_spectral
        )

        # Initialize quantum channels based on network topology
        for (src, tgt), channel in self.network.channels.items():
            self.quantum_sim.add_channel(src, tgt, channel.fidelity, channel.loss_rate)

        # Create RL agents for each node
        self.agents: Dict[int, DistributedQLearningAgent] = {}
        for node_id in range(config.n_nodes):
            n_neighbors = len(self.network.get_neighbors(node_id))
            agent = DistributedQLearningAgent(
                node_id=node_id,
                n_temporal=config.n_temporal,
                n_spectral=config.n_spectral,
                n_neighbors=n_neighbors,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                batch_size=config.batch_size
            )
            self.agents[node_id] = agent

        # Beacon broadcaster for coordination
        self.beacon = BeaconBroadcaster(config.n_nodes)

        # Metrics collector
        self.metrics = MetricsCollector(
            config.n_nodes,
            config.n_temporal,
            config.n_spectral
        )

        # Temporal-spectral encoder
        self.encoder = TemporalSpectralEncoder(
            config.n_temporal,
            config.n_spectral
        )

        # Training statistics
        self.episode_stats = []

    def run_episode(self, episode_num: int, evaluation: bool = False) -> Dict:
        """
        Run one episode of simulation

        Args:
            episode_num: Episode number
            evaluation: If True, no exploration (greedy policy)

        Returns:
            Episode statistics
        """
        episode_rewards = {node_id: 0.0 for node_id in range(self.config.n_nodes)}
        episode_collisions = 0
        episode_successful = 0

        for time_step in range(self.config.episode_length):
            # Each node selects action
            node_modes = {}
            node_states = {}

            for node_id in range(self.config.n_nodes):
                # Get current state
                neighbors = self.network.get_neighbors(node_id)
                neighbor_modes = self.beacon.receive_beacons(node_id, neighbors)

                # Get network state info
                network_state = {
                    'local_efficiency': self.metrics.calculate_network_efficiency(),
                    'local_collision_rate': self.metrics.calculate_collision_rate(),
                    'neighbor_collision_rate': 0.0,  # Could be computed
                    'channel_fidelity_avg': self.network.get_channel_fidelity(
                        node_id, neighbors[0]) if neighbors else 0.9,
                    'network_load': len(neighbor_modes) / max(1, len(neighbors))
                }

                # Current mode (from last step, or -1 if first step)
                current_mode = node_modes.get(node_id, -1)

                # Encode state
                state = self.agents[node_id].encode_state(
                    current_mode, neighbor_modes, network_state
                )
                node_states[node_id] = state

                # Select action
                action = self.agents[node_id].select_action(state, evaluation)
                temporal_bin, spectral_bin = self.agents[node_id].get_mode_from_action(action)

                node_modes[node_id] = action

                # Set quantum state
                self.quantum_sim.set_node_state(node_id, temporal_bin, spectral_bin)

                # Broadcast beacon
                self.beacon.broadcast_beacon(node_id, action, time_step)

            # Detect collisions and compute rewards
            step_collisions = 0
            for node_id in range(self.config.n_nodes):
                neighbors = self.network.get_neighbors(node_id)

                # Check for collisions with neighbors
                collision = self.beacon.detect_collision(
                    node_id, node_modes[node_id], neighbors
                )

                if collision:
                    step_collisions += 1
                    reward = -1.0  # Penalty for collision
                else:
                    # Successful transmission
                    reward = 1.0

                    # Bonus for high channel fidelity
                    if neighbors:
                        avg_fidelity = np.mean([
                            self.network.get_channel_fidelity(node_id, n)
                            for n in neighbors
                        ])
                        reward += 0.5 * avg_fidelity

                episode_rewards[node_id] += reward

                # Store experience (if not evaluation)
                if not evaluation and time_step < self.config.episode_length - 1:
                    # Next state (computed in next iteration)
                    # For now, store with current state as next state
                    # This will be updated in actual implementation
                    next_state = node_states[node_id]  # Placeholder
                    done = (time_step == self.config.episode_length - 1)

                    self.agents[node_id].store_experience(
                        node_states[node_id],
                        node_modes[node_id],
                        reward,
                        next_state,
                        done
                    )

                    # Train agent
                    loss = self.agents[node_id].train_step()

            # Update metrics
            episode_collisions += step_collisions
            episode_successful += (self.config.n_nodes - step_collisions)

            self.metrics.record_transmission(
                success=(step_collisions == 0),
                n_photons=self.config.n_nodes,
                fidelity=self.quantum_sim.calculate_network_fidelity()
            )

        # Episode statistics
        avg_reward = np.mean(list(episode_rewards.values()))
        collision_rate = episode_collisions / (self.config.n_nodes * self.config.episode_length)

        stats = {
            'episode': episode_num,
            'avg_reward': avg_reward,
            'collision_rate': collision_rate,
            'successful_transmissions': episode_successful,
            'network_efficiency': 1.0 - collision_rate
        }

        # Update episode rewards for agents
        for node_id, total_reward in episode_rewards.items():
            self.agents[node_id].update_episode_reward(total_reward)

        # Record episode in metrics
        self.metrics.record_episode(avg_reward, self.config.episode_length)

        return stats

    def train(self, n_episodes: Optional[int] = None, save_dir: str = "models") -> List[Dict]:
        """
        Train RL agents

        Args:
            n_episodes: Number of episodes (if None, use config)
            save_dir: Directory to save trained models

        Returns:
            List of episode statistics
        """
        n_episodes = n_episodes or self.config.n_episodes
        episode_stats = []

        print(f"Starting training for {n_episodes} episodes...")
        start_time = time.time()

        for episode in range(n_episodes):
            stats = self.run_episode(episode, evaluation=False)
            episode_stats.append(stats)

            if (episode + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {stats['avg_reward']:.3f} | "
                      f"Collision Rate: {stats['collision_rate']:.3f} | "
                      f"Efficiency: {stats['network_efficiency']:.3f} | "
                      f"Time: {elapsed:.1f}s")

        self.episode_stats = episode_stats

        # AUTO-SAVE: Save all trained agent models
        self.save_models(save_dir)

        return episode_stats

    def evaluate(self, n_episodes: int = 50) -> Dict:
        """
        Evaluate trained agents

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation results
        """
        print(f"Evaluating for {n_episodes} episodes...")

        eval_stats = []
        for episode in range(n_episodes):
            stats = self.run_episode(episode, evaluation=True)
            eval_stats.append(stats)

        # Compute summary statistics
        avg_reward = np.mean([s['avg_reward'] for s in eval_stats])
        avg_collision = np.mean([s['collision_rate'] for s in eval_stats])
        avg_efficiency = np.mean([s['network_efficiency'] for s in eval_stats])

        return {
            'avg_reward': avg_reward,
            'avg_collision_rate': avg_collision,
            'avg_network_efficiency': avg_efficiency,
            'std_reward': np.std([s['avg_reward'] for s in eval_stats]),
            'std_collision_rate': np.std([s['collision_rate'] for s in eval_stats]),
            'std_network_efficiency': np.std([s['network_efficiency'] for s in eval_stats])
        }

    def get_current_metrics(self) -> NetworkMetrics:
        """Get current network metrics"""
        return self.metrics.compute_current_metrics(time_elapsed=1.0)

    def get_network_state(self) -> Dict:
        """Get current network state for visualization"""
        return {
            'network': self.network.export_to_dict(),
            'metrics': self.metrics.get_metrics_summary(),
            'episode_stats': self.episode_stats[-100:] if self.episode_stats else []
        }

    def save_models(self, save_dir: str = "models") -> str:
        """
        Save all trained agent models

        Args:
            save_dir: Directory to save models

        Returns:
            Path to saved models directory
        """
        import os
        from pathlib import Path
        import json

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each agent
        print(f"\nSaving trained models to {save_path}/...")
        for node_id, agent in self.agents.items():
            model_file = save_path / f"agent_{node_id}.pth"
            agent.save_model(str(model_file))

        # Save training statistics
        stats_file = save_path / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'episode_stats': self.episode_stats,
                'config': self.config.__dict__,
                'n_agents': len(self.agents)
            }, f, indent=2)

        print(f"✓ Saved {len(self.agents)} agent models")
        print(f"✓ Saved training statistics")
        print(f"✓ Models directory: {save_path.absolute()}\n")

        return str(save_path.absolute())

    def load_models(self, save_dir: str = "models"):
        """
        Load trained agent models

        Args:
            save_dir: Directory containing saved models
        """
        from pathlib import Path

        save_path = Path(save_dir)
        if not save_path.exists():
            raise FileNotFoundError(f"Models directory not found: {save_dir}")

        print(f"\nLoading trained models from {save_path}/...")

        # Load each agent
        loaded_count = 0
        for node_id, agent in self.agents.items():
            model_file = save_path / f"agent_{node_id}.pth"
            if model_file.exists():
                agent.load_model(str(model_file))
                loaded_count += 1

        print(f"✓ Loaded {loaded_count}/{len(self.agents)} agent models\n")

        return loaded_count


class BaselineComparison:
    """Compare RL approach with baseline algorithms"""

    def __init__(self, config: SimulationConfig):
        """
        Initialize baseline comparison

        Args:
            config: Simulation configuration
        """
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


def run_full_benchmark(config: SimulationConfig) -> Dict:
    """
    Run full benchmark comparing RL with baselines

    Args:
        config: Simulation configuration

    Returns:
        Complete benchmark results
    """
    results = {
        'config': config.__dict__,
        'rl_results': None,
        'baseline_results': None,
        'comparison': None
    }

    # Run RL training and evaluation
    print("=" * 60)
    print("Running RL-based approach...")
    print("=" * 60)

    rl_sim = QuantumNetworkSimulation(config)
    rl_sim.train(config.n_episodes)
    rl_eval = rl_sim.evaluate(n_episodes=50)

    results['rl_results'] = {
        'training_stats': rl_sim.episode_stats,
        'evaluation': rl_eval,
        'final_metrics': rl_sim.get_current_metrics().__dict__
    }

    # Run baseline comparisons
    print("\n" + "=" * 60)
    print("Running baseline comparisons...")
    print("=" * 60)

    baseline_comp = BaselineComparison(config)
    baseline_results = baseline_comp.compare_all_baselines()

    results['baseline_results'] = baseline_results

    # Generate comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    print(f"\nRL Approach:")
    print(f"  Network Efficiency: {rl_eval['avg_network_efficiency']:.3f} ± {rl_eval['std_network_efficiency']:.3f}")
    print(f"  Collision Rate: {rl_eval['avg_collision_rate']:.3f} ± {rl_eval['std_collision_rate']:.3f}")

    for baseline_name, baseline_result in baseline_results.items():
        print(f"\n{baseline_name.upper()} Baseline:")
        print(f"  Network Efficiency: {baseline_result['network_efficiency']:.3f}")
        print(f"  Collision Rate: {baseline_result['collision_rate']:.3f}")

    results['comparison'] = {
        'rl_efficiency': rl_eval['avg_network_efficiency'],
        'baseline_efficiencies': {
            name: res['network_efficiency']
            for name, res in baseline_results.items()
        }
    }

    return results
