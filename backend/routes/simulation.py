"""
Simulation API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import asyncio
from datetime import datetime
from pathlib import Path
import zipfile
import io

from backend.core.simulation import QuantumNetworkSimulation, SimulationConfig, run_full_benchmark

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

# Global simulation instances (in production, use proper state management)
active_simulations: Dict[str, QuantumNetworkSimulation] = {}
simulation_results: Dict[str, Dict] = {}
simulation_status: Dict[str, str] = {}
simulation_model_paths: Dict[str, str] = {}  # Store paths to saved models


class SimulationRequest(BaseModel):
    """Request model for creating a simulation"""
    n_nodes: int = 100
    topology: str = "grid"
    n_temporal: int = 32
    n_spectral: int = 16
    n_episodes: int = 200
    episode_length: int = 100
    learning_rate: float = 0.15
    discount_factor: float = 0.92
    batch_size: int = 64
    probe_fraction: float = 0.08
    random_seed: int = 42


class SimulationResponse(BaseModel):
    """Response model for simulation creation"""
    simulation_id: str
    status: str
    message: str


@router.post("/create", response_model=SimulationResponse)
async def create_simulation(request: SimulationRequest):
    """Create a new simulation instance"""
    try:
        # Create simulation ID
        sim_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create configuration
        config = SimulationConfig(
            n_nodes=request.n_nodes,
            topology=request.topology,
            n_temporal=request.n_temporal,
            n_spectral=request.n_spectral,
            n_episodes=request.n_episodes,
            episode_length=request.episode_length,
            learning_rate=request.learning_rate,
            discount_factor=request.discount_factor,
            batch_size=request.batch_size,
            probe_fraction=request.probe_fraction,
            random_seed=request.random_seed
        )

        # Create simulation
        simulation = QuantumNetworkSimulation(config)
        active_simulations[sim_id] = simulation
        simulation_status[sim_id] = "created"

        return SimulationResponse(
            simulation_id=sim_id,
            status="created",
            message=f"Simulation created with {request.n_nodes} nodes"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{simulation_id}/train")
async def train_simulation(simulation_id: str, background_tasks: BackgroundTasks):
    """Start training a simulation"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    try:
        simulation = active_simulations[simulation_id]
        simulation_status[simulation_id] = "training"

        # Run training in background
        def run_training():
            try:
                print(f"\n{'='*70}")
                print(f"Training Simulation: {simulation_id}")
                print(f"{'='*70}\n")

                # Train and auto-save models
                model_dir = f"models/{simulation_id}"
                stats = simulation.train(save_dir=model_dir)

                print(f"\n{'='*70}")
                print(f"Training Complete: {simulation_id}")
                print(f"Total Episodes: {len(stats)}")
                print(f"{'='*70}\n")

                # Store model path
                simulation_model_paths[simulation_id] = model_dir

                simulation_results[simulation_id] = {
                    'training_stats': stats,
                    'status': 'completed',
                    'model_path': model_dir
                }
                simulation_status[simulation_id] = "completed"
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"Training Failed: {simulation_id}")
                print(f"Error: {str(e)}")
                print(f"{'='*70}\n")

                simulation_results[simulation_id] = {
                    'error': str(e),
                    'status': 'failed'
                }
                simulation_status[simulation_id] = "failed"

        background_tasks.add_task(run_training)

        return {
            "simulation_id": simulation_id,
            "status": "training_started",
            "message": "Training started in background"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{simulation_id}/status")
async def get_simulation_status(simulation_id: str):
    """Get simulation status"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    status = simulation_status.get(simulation_id, "unknown")
    simulation = active_simulations[simulation_id]

    return {
        "simulation_id": simulation_id,
        "status": status,
        "episode_stats": simulation.episode_stats[-10:] if simulation.episode_stats else [],
        "current_episode": len(simulation.episode_stats)
    }


@router.get("/{simulation_id}/results")
async def get_simulation_results(simulation_id: str):
    """Get simulation results"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = active_simulations[simulation_id]

    return {
        "simulation_id": simulation_id,
        "status": simulation_status.get(simulation_id, "unknown"),
        "episode_stats": simulation.episode_stats,
        "metrics": simulation.metrics.get_metrics_summary(),
        "network_state": simulation.get_network_state()
    }


@router.get("/{simulation_id}/metrics")
async def get_current_metrics(simulation_id: str):
    """Get current metrics"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = active_simulations[simulation_id]
    metrics = simulation.get_current_metrics()

    return {
        "simulation_id": simulation_id,
        "metrics": {
            "information_density": metrics.information_density,
            "network_efficiency": metrics.network_efficiency,
            "collision_rate": metrics.collision_rate,
            "convergence_rate": metrics.convergence_rate,
            "average_fidelity": metrics.average_fidelity,
            "throughput": metrics.throughput,
            "qber": metrics.qber
        }
    }


@router.delete("/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete a simulation"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    del active_simulations[simulation_id]
    if simulation_id in simulation_status:
        del simulation_status[simulation_id]
    if simulation_id in simulation_results:
        del simulation_results[simulation_id]

    return {
        "simulation_id": simulation_id,
        "status": "deleted",
        "message": "Simulation deleted successfully"
    }


@router.get("/list")
async def list_simulations():
    """List all active simulations"""
    return {
        "simulations": [
            {
                "simulation_id": sim_id,
                "status": simulation_status.get(sim_id, "unknown"),
                "n_episodes_completed": len(active_simulations[sim_id].episode_stats),
                "has_trained_model": sim_id in simulation_model_paths
            }
            for sim_id in active_simulations.keys()
        ]
    }


@router.post("/{simulation_id}/evaluate")
async def evaluate_simulation(simulation_id: str, n_episodes: int = 50):
    """Evaluate trained simulation"""
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation_status.get(simulation_id) != "completed":
        raise HTTPException(status_code=400, detail="Simulation must be trained first")

    try:
        simulation = active_simulations[simulation_id]
        eval_results = simulation.evaluate(n_episodes=n_episodes)

        return {
            "simulation_id": simulation_id,
            "evaluation": eval_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{simulation_id}/download")
async def download_models(simulation_id: str):
    """Download trained models as zip file"""
    if simulation_id not in simulation_model_paths:
        raise HTTPException(status_code=404, detail="No trained models found for this simulation")

    try:
        model_dir = Path(simulation_model_paths[simulation_id])
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model directory not found")

        # Create zip file
        zip_path = Path(f"models/{simulation_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in model_dir.glob("**/*"):
                if file.is_file():
                    zipf.write(file, arcname=file.relative_to(model_dir))

        return FileResponse(
            path=str(zip_path),
            filename=f"{simulation_id}_models.zip",
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
