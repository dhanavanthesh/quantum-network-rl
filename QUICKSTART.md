# Quick Start Guide

Get up and running with the Quantum Network Simulator in 5 minutes!

## Prerequisites

- Python 3.9+ installed
- (Optional) Node.js 18+ for web interface

## Fast Track: Command-Line Simulation

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Your First Simulation

```bash
# Quick test with 50 nodes (15-25 minutes)
python run_simulation.py --mode quick --n-nodes 50 --n-episodes 150 --output-dir results
```

### 3. View Results

Results are saved in the `results/` directory:
- `training_progress.png` - Training curves
- `results.json` - Raw data

## Example Commands

### Small Network Test (15-25 min)
```bash
python run_simulation.py \
  --mode quick \
  --n-nodes 50 \
  --topology grid \
  --n-temporal 32 \
  --n-spectral 16 \
  --n-episodes 150 \
  --output-dir results/small_test
```

### Medium Network (45-75 min)
```bash
python run_simulation.py \
  --mode quick \
  --n-nodes 100 \
  --topology grid \
  --n-episodes 200 \
  --output-dir results/medium_test
```

### Full Benchmark with Baselines (1-2 hours)
```bash
python run_simulation.py \
  --mode benchmark \
  --n-nodes 100 \
  --topology grid \
  --n-episodes 200 \
  --output-dir results/full_benchmark
```

## Understanding Output

After running, you'll see output like:

```
Episode 200/200 | Avg Reward: 12.45 | Collision Rate: 0.08 | Efficiency: 0.92 | Time: 45.3s

Evaluation Results:
  Average Reward: 12.67 ± 0.45
  Network Efficiency: 0.923 ± 0.012
  Collision Rate: 0.077 ± 0.008
```

**Key Metrics:**
- **Network Efficiency**: Higher is better (target: >0.85)
- **Collision Rate**: Lower is better (target: <0.15)
- **Average Reward**: Higher indicates better learning

## Troubleshooting

### Out of Memory?
Reduce network size or mode dimensions:
```bash
python run_simulation.py --n-nodes 50 --n-temporal 16 --n-spectral 8
```

### Too Slow?
Reduce episodes or episode length:
```bash
python run_simulation.py --n-episodes 100 --episode-length 50
```

### Check Installation
```bash
python -c "import torch; import networkx; print('All dependencies OK!')"
```

## Next Steps

1. **Explore parameters**: Try different topologies and configurations
2. **Run benchmarks**: Compare RL with baseline algorithms
3. **Analyze results**: Use matplotlib to visualize training curves
4. **Web interface**: Set up the React frontend for interactive visualization

## Web Interface (Optional)

If you want the 3D visualization:

### Terminal 1 - Backend
```bash
cd backend
python main.py
```

### Terminal 2 - Frontend
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## Getting Help

- Check `README.md` for full documentation
- Example configurations in `backend/config.py`
- API docs at http://localhost:8000/docs (when backend is running)

## Performance Expectations

| Network Size | Episodes | Typical Time | Memory |
|--------------|----------|--------------|--------|
| 50 nodes | 150 | 15-25 min | ~2 GB |
| 100 nodes | 200 | 45-75 min | ~4 GB |
| 300 nodes | 250 | 3-5 hours | ~8 GB |

## Example Results

Typical results for 100-node grid network:

- **RL Approach**: 92% efficiency, 8% collision rate
- **TDMA Baseline**: 85% efficiency, 15% collision rate
- **Fixed Time-Bin**: 78% efficiency, 22% collision rate
- **Random**: 65% efficiency, 35% collision rate

**RL Improvement**: ~7-10% over best baseline

---

Happy simulating! 🚀
