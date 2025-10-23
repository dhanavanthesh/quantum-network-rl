"""
Quick test to verify model saving works
"""

from backend.core.simulation import QuantumNetworkSimulation, SimulationConfig
from pathlib import Path

print("=" * 70)
print("Testing Model Saving Functionality")
print("=" * 70)

# Create small simulation
config = SimulationConfig(
    n_nodes=10,           # Small network for quick test
    n_episodes=5,         # Just 5 episodes
    episode_length=10,    # Short episodes
    topology='line'
)

print("\n1. Creating simulation...")
sim = QuantumNetworkSimulation(config)
print(f"   ✓ Created simulation with {config.n_nodes} nodes")

print("\n2. Training for 5 episodes...")
stats = sim.train(save_dir="test_models")
print(f"   ✓ Training complete")
print(f"   ✓ Final collision rate: {stats[-1]['collision_rate']:.3f}")
print(f"   ✓ Final efficiency: {stats[-1]['network_efficiency']:.3f}")

print("\n3. Checking if models were saved...")
model_path = Path("test_models")
if model_path.exists():
    agent_files = list(model_path.glob("agent_*.pth"))
    stats_file = model_path / "training_stats.json"

    print(f"   ✓ Models directory exists: {model_path.absolute()}")
    print(f"   ✓ Found {len(agent_files)} agent model files")
    print(f"   ✓ Training stats saved: {stats_file.exists()}")

    if len(agent_files) == config.n_nodes:
        print("\n✅ SUCCESS! All models saved correctly!")
    else:
        print(f"\n⚠️  WARNING: Expected {config.n_nodes} models, found {len(agent_files)}")
else:
    print("   ❌ ERROR: Models directory not found!")

print("\n4. Testing model loading...")
sim2 = QuantumNetworkSimulation(config)
loaded = sim2.load_models("test_models")
print(f"   ✓ Loaded {loaded} agent models")

if loaded == config.n_nodes:
    print("\n✅ ALL TESTS PASSED!")
    print("\nModel saving and loading is working correctly.")
else:
    print(f"\n⚠️  WARNING: Only loaded {loaded}/{config.n_nodes} models")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
