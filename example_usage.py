"""
Example Usage of Quantum Network Simulator
Demonstrates basic API usage with different configurations
"""

import sys
sys.path.insert(0, '.')

from backend.core.simulation import SimulationConfig, QuantumNetworkSimulation
from backend.core.baselines import BaselineComparison
import matplotlib.pyplot as plt
import numpy as np


def example_1_basic_simulation():
    """Example 1: Basic simulation with default parameters"""
    print("\n" + "="*70)
    print("Example 1: Basic Simulation")
    print("="*70)

    config = SimulationConfig(
        n_nodes=50,
        topology='grid',
        n_temporal=32,
        n_spectral=16,
        n_episodes=100,
        random_seed=42
    )

    # Create and run simulation
    sim = QuantumNetworkSimulation(config)
    print("Training for 100 episodes...")
    sim.train(n_episodes=100)

    # Evaluate
    print("\nEvaluating...")
    results = sim.evaluate(n_episodes=20)

    print(f"\nResults:")
    print(f"  Network Efficiency: {results['avg_network_efficiency']:.3f} ± {results['std_network_efficiency']:.3f}")
    print(f"  Collision Rate: {results['avg_collision_rate']:.3f}")


def example_2_compare_topologies():
    """Example 2: Compare different network topologies"""
    print("\n" + "="*70)
    print("Example 2: Comparing Network Topologies")
    print("="*70)

    topologies = ['line', 'grid', 'scale_free']
    results = {}

    for topology in topologies:
        print(f"\nTesting {topology} topology...")

        config = SimulationConfig(
            n_nodes=50,
            topology=topology,
            n_temporal=32,
            n_spectral=16,
            n_episodes=50,
            random_seed=42
        )

        sim = QuantumNetworkSimulation(config)
        sim.train(n_episodes=50)
        eval_result = sim.evaluate(n_episodes=10)

        results[topology] = eval_result['avg_network_efficiency']

    # Print comparison
    print("\n" + "-"*70)
    print("Topology Comparison:")
    for topology, efficiency in results.items():
        print(f"  {topology:15s}: {efficiency:.3f}")


def example_3_parameter_sensitivity():
    """Example 3: Test sensitivity to learning rate"""
    print("\n" + "="*70)
    print("Example 3: Learning Rate Sensitivity")
    print("="*70)

    learning_rates = [0.05, 0.1, 0.15, 0.2]
    results = {}

    for lr in learning_rates:
        print(f"\nTesting learning rate α={lr}...")

        config = SimulationConfig(
            n_nodes=50,
            topology='grid',
            n_temporal=32,
            n_spectral=16,
            n_episodes=50,
            learning_rate=lr,
            random_seed=42
        )

        sim = QuantumNetworkSimulation(config)
        sim.train(n_episodes=50)
        eval_result = sim.evaluate(n_episodes=10)

        results[lr] = {
            'efficiency': eval_result['avg_network_efficiency'],
            'collision': eval_result['avg_collision_rate']
        }

    # Print comparison
    print("\n" + "-"*70)
    print("Learning Rate Analysis:")
    print(f"{'LR':>6} | {'Efficiency':>10} | {'Collision':>10}")
    print("-"*35)
    for lr, res in results.items():
        print(f"{lr:>6.2f} | {res['efficiency']:>10.3f} | {res['collision']:>10.3f}")


def example_4_baseline_comparison():
    """Example 4: Compare with baseline algorithms"""
    print("\n" + "="*70)
    print("Example 4: Baseline Algorithm Comparison")
    print("="*70)

    # Run RL
    print("\nTraining RL-based approach...")
    config = SimulationConfig(
        n_nodes=50,
        topology='grid',
        n_temporal=32,
        n_spectral=16,
        n_episodes=100,
        random_seed=42
    )

    rl_sim = QuantumNetworkSimulation(config)
    rl_sim.train(n_episodes=100)
    rl_results = rl_sim.evaluate(n_episodes=20)

    # Run baselines
    print("\nEvaluating baseline algorithms...")
    baseline_comp = BaselineComparison(config)
    baseline_results = baseline_comp.compare_all_baselines()

    # Print comparison
    print("\n" + "-"*70)
    print("Algorithm Comparison:")
    print(f"{'Algorithm':15s} | {'Efficiency':>10} | {'Collision':>10}")
    print("-"*42)

    # RL results
    print(f"{'RL (Ours)':15s} | {rl_results['avg_network_efficiency']:>10.3f} | "
          f"{rl_results['avg_collision_rate']:>10.3f}")

    # Baseline results
    for name, result in baseline_results.items():
        print(f"{name.upper():15s} | {result['network_efficiency']:>10.3f} | "
              f"{result['collision_rate']:>10.3f}")

    # Calculate improvement
    best_baseline_eff = max([r['network_efficiency'] for r in baseline_results.values()])
    improvement = ((rl_results['avg_network_efficiency'] - best_baseline_eff) /
                   best_baseline_eff * 100)

    print(f"\nRL Improvement over best baseline: {improvement:.1f}%")


def example_5_visualize_training():
    """Example 5: Visualize training progress"""
    print("\n" + "="*70)
    print("Example 5: Visualizing Training Progress")
    print("="*70)

    config = SimulationConfig(
        n_nodes=50,
        topology='grid',
        n_temporal=32,
        n_spectral=16,
        n_episodes=150,
        random_seed=42
    )

    print("Training for 150 episodes...")
    sim = QuantumNetworkSimulation(config)
    sim.train(n_episodes=150)

    # Extract data
    episodes = [s['episode'] for s in sim.episode_stats]
    rewards = [s['avg_reward'] for s in sim.episode_stats]
    efficiencies = [s['network_efficiency'] for s in sim.episode_stats]
    collisions = [s['collision_rate'] for s in sim.episode_stats]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Rewards
    axes[0].plot(episodes, rewards, 'b-', linewidth=2, alpha=0.7)
    axes[0].plot(episodes, np.convolve(rewards, np.ones(10)/10, mode='valid'),
                 'r-', linewidth=2, label='Moving Average')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Progress')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Network Efficiency
    axes[1].plot(episodes, efficiencies, 'g-', linewidth=2, alpha=0.7)
    axes[1].axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
    axes[1].set_ylabel('Network Efficiency')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Collision Rate
    axes[2].plot(episodes, collisions, 'r-', linewidth=2, alpha=0.7)
    axes[2].axhline(y=0.15, color='g', linestyle='--', label='Target (15%)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Collision Rate')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('training_example.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'training_example.png'")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Quantum Network Simulator - Example Usage")
    print("="*70)

    examples = [
        ("Basic Simulation", example_1_basic_simulation),
        ("Compare Topologies", example_2_compare_topologies),
        ("Learning Rate Sensitivity", example_3_parameter_sensitivity),
        ("Baseline Comparison", example_4_baseline_comparison),
        ("Visualize Training", example_5_visualize_training),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    choice = input("\nRun which example? (1-5, or 'all'): ").strip()

    if choice.lower() == 'all':
        for name, func in examples:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice)-1][1]()
    else:
        print("Invalid choice!")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
