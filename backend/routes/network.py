"""
Network Topology API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from backend.core.quantum_network import create_network

router = APIRouter(prefix="/api/network", tags=["network"])


class NetworkRequest(BaseModel):
    """Request model for network creation"""
    n_nodes: int = 100
    topology: str = "grid"
    random_seed: int = 42


@router.post("/generate")
async def generate_network(request: NetworkRequest):
    """Generate a network topology"""
    try:
        network = create_network(
            request.n_nodes,
            request.topology,
            request.random_seed
        )

        return {
            "network": network.export_to_dict(),
            "statistics": network.get_network_statistics()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topologies")
async def get_available_topologies():
    """Get list of available topology types"""
    return {
        "topologies": [
            {
                "name": "line",
                "description": "Linear/path graph topology",
                "max_nodes": 500
            },
            {
                "name": "grid",
                "description": "2D grid topology",
                "max_nodes": 400
            },
            {
                "name": "scale_free",
                "description": "Scale-free Barabási-Albert graph",
                "max_nodes": 300
            },
            {
                "name": "random_geometric",
                "description": "Random geometric graph",
                "max_nodes": 250
            }
        ]
    }


@router.post("/analyze")
async def analyze_network(request: NetworkRequest):
    """Analyze network topology properties"""
    try:
        network = create_network(
            request.n_nodes,
            request.topology,
            request.random_seed
        )

        stats = network.get_network_statistics()

        # Additional analysis
        positions = network.get_node_positions_array()
        edges = network.get_edges_array()

        return {
            "statistics": stats,
            "n_nodes": network.n_nodes,
            "n_edges": len(edges),
            "avg_degree": stats['avg_degree'],
            "diameter": stats['diameter'],
            "clustering_coefficient": stats['avg_clustering'],
            "avg_path_length": stats['avg_path_length']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
