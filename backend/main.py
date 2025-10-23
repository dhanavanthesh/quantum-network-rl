"""
Quantum Network Simulator - FastAPI Backend
Main application entry point
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.routes import simulation, network, metrics


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 60)
    print("Quantum Network Simulator - Backend Starting")
    print("=" * 60)
    yield
    # Shutdown
    print("=" * 60)
    print("Quantum Network Simulator - Backend Shutting Down")
    print("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="Quantum Network Simulator API",
    description="Distributed RL for temporal-spectral resource allocation in quantum networks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simulation.router)
app.include_router(network.router)
app.include_router(metrics.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum Network Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "simulation": "/api/simulation",
            "network": "/api/network",
            "metrics": "/api/metrics",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "quantum-network-simulator"
    }


@app.get("/config")
async def get_config():
    """Get system configuration"""
    from backend.config import config

    return {
        "network": {
            "max_nodes": config.network.MAX_NODES,
            "topology_types": config.network.TOPOLOGY_TYPES
        },
        "quantum": {
            "temporal_bins": config.quantum.DEFAULT_TEMPORAL,
            "spectral_bins": config.quantum.DEFAULT_SPECTRAL,
            "coherence_time_ms": config.quantum.COHERENCE_TIME_MS,
            "probe_fraction": config.quantum.PROBE_FRACTION
        },
        "rl": {
            "hidden_layers": config.rl.HIDDEN_LAYERS,
            "learning_rate": config.rl.LEARNING_RATE,
            "discount_factor": config.rl.DISCOUNT_FACTOR,
            "batch_size": config.rl.BATCH_SIZE
        },
        "simulation": {
            "num_runs": config.simulation.NUM_RUNS,
            "num_cores": config.simulation.NUM_CORES
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Quantum Network Simulator Backend...")
    print("API Documentation: http://localhost:8000/docs")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
