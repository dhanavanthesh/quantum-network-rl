"""
Metrics API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/available")
async def get_available_metrics():
    """Get list of available metrics"""
    return {
        "metrics": [
            {
                "name": "information_density",
                "description": "Bits per photon transmitted",
                "unit": "bits/photon"
            },
            {
                "name": "network_efficiency",
                "description": "Fraction of successful transmissions",
                "unit": "percentage"
            },
            {
                "name": "collision_rate",
                "description": "Fraction of transmission collisions",
                "unit": "percentage"
            },
            {
                "name": "convergence_rate",
                "description": "RL convergence speed",
                "unit": "percentage"
            },
            {
                "name": "average_fidelity",
                "description": "Average quantum state fidelity",
                "unit": "fidelity"
            },
            {
                "name": "qber",
                "description": "Quantum bit error rate",
                "unit": "percentage"
            },
            {
                "name": "throughput",
                "description": "Successful transmissions per second",
                "unit": "tx/s"
            },
            {
                "name": "average_latency",
                "description": "Average communication latency",
                "unit": "ms"
            }
        ]
    }


@router.get("/baselines")
async def get_baseline_algorithms():
    """Get list of baseline algorithms for comparison"""
    return {
        "baselines": [
            {
                "name": "TDMA",
                "description": "Time Division Multiple Access",
                "type": "classical"
            },
            {
                "name": "Fixed Time-Bin",
                "description": "Fixed temporal-spectral mode allocation",
                "type": "static"
            },
            {
                "name": "Random",
                "description": "Random mode selection",
                "type": "random"
            },
            {
                "name": "Graph Coloring",
                "description": "Graph coloring-based allocation",
                "type": "heuristic"
            }
        ]
    }
