"""
Quantum Network Simulator - Standalone Runner
Run simulations without web interface
"""

import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from backend.core.simulation import SimulationConfig, QuantumNetworkSimulation, run_full_benchmark
from backend.core.baselines import BaselineComparison


def plot_training_progress(episode_stats, save_path=None):
    """Plot training progress"""
    episodes = [s['episode'] for s in episode_stats]
    rewards = [s['avg_reward'] for s in episode_stats]
    collision_rates = [s['collision_rate'] for s in episode_stats]
    efficiencies = [s['network_efficiency'] for s in episode_stats]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Average reward
    axes[0].plot(episodes, rewards, 'b-', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('RL Training Progress - Average Reward')
    axes[0].grid(True, alpha=0.3)

    # Collision rate
    axes[1].plot(episodes, collision_rates, 'r-', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Collision Rate')
    axes[1].set_title('Collision Rate Over Time')
    axes[1].grid(True, alpha=0.3)

    # Network efficiency
    axes[2].plot(episodes, efficiencies, 'g-', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Network Efficiency')
    axes[2].set_title('Network Efficiency Over Time')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()


def plot_baseline_comparison(rl_results, baseline_results, save_path=None):
    """Plot comparison with baselines"""
    algorithms = ['RL'] + list(baseline_results.keys())
    efficiencies = [rl_results['avg_network_efficiency']] + \
                  [baseline_results[b]['network_efficiency'] for b in baseline_results.keys()]
    collision_rates = [rl_results['avg_collision_rate']] + \
                     [baseline_results[b]['collision_rate'] for b in baseline_results.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Network efficiency comparison
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    axes[0].bar(algorithms, efficiencies, color=colors[:len(algorithms)])
    axes[0].set_ylabel('Network Efficiency')
    axes[0].set_title('Network Efficiency Comparison')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(efficiencies):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Collision rate comparison
    axes[1].bar(algorithms, collision_rates, color=colors[:len(algorithms)])
    axes[1].set_ylabel('Collision Rate')
    axes[1].set_title('Collision Rate Comparison')
    axes[1].set_ylim([0, max(collision_rates) * 1.2])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(collision_rates):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def run_quick_simulation(args):
    """Run a quick simulation"""
    print("=" * 70)
    print("Quantum Network Simulator - Quick Simulation")
    print("=" * 70)

    # Create configuration
    config = SimulationConfig(
        n_nodes=args.n_nodes,
        topology=args.topology,
        n_temporal=args.n_temporal,
        n_spectral=args.n_spectral,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        random_seed=args.seed
    )

    print(f"\nConfiguration:")
    print(f"  Nodes: {config.n_nodes}")
    print(f"  Topology: {config.topology}")
    print(f"  Temporal bins: {config.n_temporal}")
    print(f"  Spectral bins: {config.n_spectral}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Episode length: {config.episode_length}")
    print()

    # Create and run simulation
    start_time = time.time()
    simulation = QuantumNetworkSimulation(config)

    print("Training RL agents...")
    episode_stats = simulation.train(config.n_episodes)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")

    # Evaluate
    print("\nEvaluating...")
    eval_results = simulation.evaluate(n_episodes=50)

    print("\nEvaluation Results:")
    print(f"  Average Reward: {eval_results['avg_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"  Network Efficiency: {eval_results['avg_network_efficiency']:.3f} ± {eval_results['std_network_efficiency']:.3f}")
    print(f"  Collision Rate: {eval_results['avg_collision_rate']:.3f} ± {eval_results['std_collision_rate']:.3f}")

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save training plot
        plot_training_progress(
            episode_stats,
            save_path=output_path / "training_progress.png"
        )

        # Save results JSON
        results = {
            'config': config.__dict__,
            'episode_stats': episode_stats,
            'evaluation': eval_results,
            'elapsed_time': elapsed_time
        }

        with open(output_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return simulation, eval_results


def run_benchmark(args):
    """Run full benchmark with baselines"""
    print("=" * 70)
    print("Quantum Network Simulator - Full Benchmark")
    print("=" * 70)

    # Create configuration
    config = SimulationConfig(
        n_nodes=args.n_nodes,
        topology=args.topology,
        n_temporal=args.n_temporal,
        n_spectral=args.n_spectral,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        random_seed=args.seed
    )

    print(f"\nRunning benchmark with {config.n_nodes} nodes...")

    start_time = time.time()

    # Run RL simulation
    print("\n" + "=" * 70)
    print("Phase 1: Training RL-based approach")
    print("=" * 70)
    rl_sim = QuantumNetworkSimulation(config)
    rl_sim.train(config.n_episodes)
    rl_results = rl_sim.evaluate(n_episodes=50)

    # Run baselines
    print("\n" + "=" * 70)
    print("Phase 2: Evaluating baseline algorithms")
    print("=" * 70)
    baseline_comp = BaselineComparison(config)
    baseline_results = baseline_comp.compare_all_baselines()

    elapsed_time = time.time() - start_time

    # Print comparison
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"\nRL Approach:")
    print(f"  Network Efficiency: {rl_results['avg_network_efficiency']:.3f} ± {rl_results['std_network_efficiency']:.3f}")
    print(f"  Collision Rate: {rl_results['avg_collision_rate']:.3f}")

    for baseline_name, baseline_result in baseline_results.items():
        print(f"\n{baseline_name.upper()} Baseline:")
        print(f"  Network Efficiency: {baseline_result['network_efficiency']:.3f}")
        print(f"  Collision Rate: {baseline_result['collision_rate']:.3f}")

    improvement = (rl_results['avg_network_efficiency'] -
                  max([b['network_efficiency'] for b in baseline_results.values()])) * 100
    print(f"\nRL Improvement over best baseline: {improvement:.1f}%")
    print(f"Total time: {elapsed_time:.1f} seconds")

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save plots
        plot_training_progress(
            rl_sim.episode_stats,
            save_path=output_path / "training_progress.png"
        )

        plot_baseline_comparison(
            rl_results,
            baseline_results,
            save_path=output_path / "baseline_comparison.png"
        )

        # Save full results
        full_results = {
            'config': config.__dict__,
            'rl_results': {
                'training_stats': rl_sim.episode_stats,
                'evaluation': rl_results
            },
            'baseline_results': baseline_results,
            'elapsed_time': elapsed_time,
            'improvement_percentage': improvement
        }

        with open(output_path / "benchmark_results.json", 'w') as f:
            json.dump(full_results, f, indent=2)

        print(f"\nBenchmark results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Network Simulator - Distributed RL for Resource Allocation"
    )

    parser.add_argument('--mode', choices=['quick', 'benchmark'], default='quick',
                       help='Simulation mode (quick or benchmark)')
    parser.add_argument('--n-nodes', type=int, default=100,
                       help='Number of network nodes')
    parser.add_argument('--topology', choices=['line', 'grid', 'scale_free', 'random_geometric'],
                       default='grid', help='Network topology')
    parser.add_argument('--n-temporal', type=int, default=32,
                       help='Number of temporal bins')
    parser.add_argument('--n-spectral', type=int, default=16,
                       help='Number of spectral bins')
    parser.add_argument('--n-episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=100,
                       help='Length of each episode')
    parser.add_argument('--learning-rate', type=float, default=0.15,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    if args.mode == 'quick':
        run_quick_simulation(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
