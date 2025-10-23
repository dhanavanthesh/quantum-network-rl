/**
 * Quantum Network Simulator - Main Application
 * React frontend with 3D visualization
 */

import React, { useState, useEffect } from 'react';
import NetworkVisualization3D from './components/NetworkVisualization3D';
import ControlPanel from './components/ControlPanel';
import PerformanceMetrics from './components/PerformanceMetrics';
import ComparisonCharts from './components/ComparisonCharts';
import Comparison3D from './components/Comparison3D';
import { WorkflowGuide, NetworkTopologyExplainer, InfoBox } from './components/HelpTooltips';
import {
  createSimulation,
  getSimulationStatus,
  trainSimulation,
  getSimulationResults,
  getSimulationMetrics
} from './services/api';

function App() {
  const [simulationId, setSimulationId] = useState(null);
  const [simulationStatus, setSimulationStatus] = useState('idle');
  const [networkData, setNetworkData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [episodeStats, setEpisodeStats] = useState([]);

  // Function to update network data from backend results
  const updateNetworkDataFromResults = (results) => {
    if (!results || !results.network_state) return;

    // Get the network topology from backend
    const networkState = results.network_state;

    // Generate visualization data from real backend data
    if (networkState.network && networkState.network.nodes) {
      const realNetworkData = {
        nodes: networkState.network.nodes.map((node, index) => ({
          id: node.id || index,
          position: node.position || generateNodePosition(index, networkState.network.nodes.length),
          status: determineNodeStatus(episodeStats) // Determine from episode stats
        })),
        edges: networkState.network.edges || []
      };

      setNetworkData(realNetworkData);
    }
  };

  // Helper: Generate node position (fallback)
  const generateNodePosition = (index, total) => {
    const gridSize = Math.ceil(Math.sqrt(total));
    const x = (index % gridSize) * 2 - gridSize;
    const y = 0;
    const z = Math.floor(index / gridSize) * 2 - gridSize;
    return [x, y, z];
  };

  // Helper: Determine node status based on training results
  const determineNodeStatus = (stats) => {
    if (!stats || stats.length === 0) return 'idle';

    const latestStats = stats[stats.length - 1];

    // If collision rate is low and efficiency high = active (success)
    if (latestStats.collision_rate < 0.1 && latestStats.network_efficiency > 0.9) {
      return 'active';
    }
    // If collision rate is medium = collision
    else if (latestStats.collision_rate > 0.3) {
      return 'collision';
    }
    // If training in progress = learning
    else if (simulationStatus === 'training') {
      return 'learning';
    }

    return 'idle';
  };

  // Poll for simulation updates
  useEffect(() => {
    if (!simulationId || simulationStatus === 'idle') return;

    const interval = setInterval(async () => {
      try {
        const status = await getSimulationStatus(simulationId);
        setSimulationStatus(status.status);
        setEpisodeStats(status.episode_stats || []);

        // Fetch REAL metrics from backend
        try {
          const metricsData = await getSimulationMetrics(simulationId);
          if (metricsData && metricsData.metrics) {
            setMetrics(metricsData.metrics);
          }
        } catch (metricsError) {
          console.log('Metrics not yet available');
        }

        // If training completed, fetch full results
        if (status.status === 'completed') {
          try {
            const results = await getSimulationResults(simulationId);
            if (results && results.network_state) {
              // Update network data with real simulation data
              updateNetworkDataFromResults(results);
            }
          } catch (resultsError) {
            console.error('Error fetching results:', resultsError);
          }
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [simulationId, simulationStatus]);

  const handleCreateSimulation = async (config) => {
    try {
      const response = await createSimulation(config);
      setSimulationId(response.simulation_id);
      setSimulationStatus('created');

      // Generate initial network data for visualization
      setNetworkData(generateMockNetworkData(config.n_nodes, config.topology));

      return response;
    } catch (error) {
      console.error('Error creating simulation:', error);
      throw error;
    }
  };

  // Update node colors based on current state
  const updateNodeColors = (currentStatus, stats) => {
    if (!networkData) return;

    const latestStats = stats && stats.length > 0 ? stats[stats.length - 1] : null;

    const updatedNodes = networkData.nodes.map(node => {
      let status = 'idle';

      if (currentStatus === 'training') {
        // During training, randomly show some nodes learning/active
        const rand = Math.random();
        if (rand < 0.3) status = 'learning';
        else if (rand < 0.5) status = 'active';
        else status = 'idle';
      } else if (currentStatus === 'completed' && latestStats) {
        // After training, show nodes based on performance
        if (latestStats.network_efficiency > 0.9) {
          // High efficiency - most nodes active (green)
          status = Math.random() < 0.8 ? 'active' : 'idle';
        } else if (latestStats.collision_rate > 0.3) {
          // High collision - show some red
          const rand = Math.random();
          if (rand < 0.3) status = 'collision';
          else if (rand < 0.6) status = 'active';
          else status = 'idle';
        } else {
          // Medium performance
          const rand = Math.random();
          if (rand < 0.5) status = 'active';
          else status = 'idle';
        }
      }

      return { ...node, status };
    });

    setNetworkData({ ...networkData, nodes: updatedNodes });
  };

  // Update node colors when status or episode stats change
  useEffect(() => {
    if (networkData && (simulationStatus === 'training' || simulationStatus === 'completed')) {
      updateNodeColors(simulationStatus, episodeStats);
    }
  }, [simulationStatus, episodeStats]);

  const handleStartTraining = async () => {
    if (!simulationId) return;

    try {
      await trainSimulation(simulationId);
      setSimulationStatus('training');
    } catch (error) {
      console.error('Error starting training:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-blue-400">
            Quantum Network Simulator
          </h1>
          <p className="text-gray-400 mt-1">
            Distributed RL for Temporal-Spectral Resource Allocation
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto p-4">
        {/* 3D Comparison - Show what makes us special */}
        <Comparison3D />

        {/* Workflow Guide - Top Section */}
        <div className="mb-4">
          <WorkflowGuide currentStep={
            simulationStatus === 'idle' ? 1 :
            simulationStatus === 'created' ? 2 :
            simulationStatus === 'training' ? 2 :
            simulationStatus === 'completed' ? 4 : 1
          } />
        </div>

        {/* Info Box */}
        {simulationStatus === 'idle' && (
          <InfoBox title="👋 Welcome! Start by creating a simulation" type="info">
            This simulator trains 100 AI agents to coordinate quantum communication.
            Set your parameters in the Control Panel and click "Create Simulation" to begin.
          </InfoBox>
        )}

        {simulationStatus === 'created' && (
          <InfoBox title="✅ Network Created! Ready to train" type="success">
            Your quantum network is ready. Click "Start Training" to begin the learning process.
            Training will take approximately 45-75 minutes for 100 nodes.
          </InfoBox>
        )}

        {simulationStatus === 'training' && (
          <InfoBox title="🔄 Training in Progress..." type="warning">
            AI agents are learning to avoid collisions. This takes 45-75 minutes.
            Check the backend console to see real-time progress.
          </InfoBox>
        )}

        {simulationStatus === 'completed' && episodeStats.length > 0 && (
          <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 border-2 border-green-500 rounded-lg p-6 mb-4">
            <div className="text-2xl font-bold mb-3 flex items-center gap-2">
              🎉 Training Complete - RESULTS
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded p-4">
                <div className="text-gray-400 text-sm">Network Efficiency</div>
                <div className="text-3xl font-bold text-green-400">
                  {(episodeStats[episodeStats.length - 1].network_efficiency * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">Goal: &gt;90%</div>
              </div>

              <div className="bg-gray-800 rounded p-4">
                <div className="text-gray-400 text-sm">Collision Rate</div>
                <div className="text-3xl font-bold text-blue-400">
                  {(episodeStats[episodeStats.length - 1].collision_rate * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">Goal: &lt;10%</div>
              </div>

              <div className="bg-gray-800 rounded p-4">
                <div className="text-gray-400 text-sm">Average Reward</div>
                <div className="text-3xl font-bold text-yellow-400">
                  {episodeStats[episodeStats.length - 1].avg_reward.toFixed(1)}
                </div>
                <div className="text-xs text-gray-500 mt-1">Higher = Better</div>
              </div>
            </div>

            <div className="mt-4 text-center">
              <div className="text-sm text-gray-300">
                🚀 Models saved and ready for download!
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-12 gap-4">
          {/* Left Panel - Controls */}
          <div className="col-span-3">
            <ControlPanel
              onCreateSimulation={handleCreateSimulation}
              onStartTraining={handleStartTraining}
              simulationStatus={simulationStatus}
              simulationId={simulationId}
            />

            {/* Add Network Topology Explainer */}
            {networkData && <NetworkTopologyExplainer />}
          </div>

          {/* Center - 3D Visualization */}
          <div className="col-span-6">
            <div className="bg-gray-800 rounded-lg p-4 h-[600px]">
              <h2 className="text-xl font-bold mb-2 flex items-center gap-2">
                🌐 Network Topology
                <span className="text-sm font-normal text-gray-400">
                  ({networkData ? networkData.nodes.length : 0} nodes)
                </span>
              </h2>
              <div className="text-sm text-gray-400 mb-3">
                Interactive 3D view: Drag to rotate, scroll to zoom
              </div>
              {!networkData && (
                <div className="flex items-center justify-center h-full text-gray-500">
                  Create a simulation to see the network
                </div>
              )}
              {networkData && (
                <NetworkVisualization3D
                  networkData={networkData}
                  simulationStatus={simulationStatus}
                />
              )}
            </div>
          </div>

          {/* Right Panel - Metrics */}
          <div className="col-span-3">
            <PerformanceMetrics
              metrics={metrics}
              simulationStatus={simulationStatus}
            />
          </div>
        </div>

        {/* Bottom - Charts */}
        <div className="mt-4">
          <ComparisonCharts
            episodeStats={episodeStats}
            simulationStatus={simulationStatus}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 p-4 mt-8">
        <div className="container mx-auto text-center text-gray-400 text-sm">
          Quantum Network Simulator v1.0 | Built with PyTorch + FastAPI + React
        </div>
      </footer>
    </div>
  );
}

// Helper function to generate mock network data for visualization
function generateMockNetworkData(nNodes, topology) {
  const nodes = [];
  const edges = [];

  // Generate nodes with positions
  for (let i = 0; i < nNodes; i++) {
    let x, y, z;

    if (topology === 'grid') {
      const gridSize = Math.ceil(Math.sqrt(nNodes));
      x = (i % gridSize) * 2 - gridSize;
      y = 0;
      z = Math.floor(i / gridSize) * 2 - gridSize;
    } else if (topology === 'line') {
      x = i * 2 - nNodes;
      y = 0;
      z = 0;
    } else {
      // Random positions
      x = (Math.random() - 0.5) * 20;
      y = (Math.random() - 0.5) * 20;
      z = (Math.random() - 0.5) * 20;
    }

    nodes.push({
      id: i,
      position: [x, y, z],
      status: 'idle', // idle, active, collision, learning
    });
  }

  // Generate edges (simplified)
  if (topology === 'grid') {
    const gridSize = Math.ceil(Math.sqrt(nNodes));
    for (let i = 0; i < nNodes; i++) {
      // Connect to right neighbor
      if ((i + 1) % gridSize !== 0 && i + 1 < nNodes) {
        edges.push([i, i + 1]);
      }
      // Connect to bottom neighbor
      if (i + gridSize < nNodes) {
        edges.push([i, i + gridSize]);
      }
    }
  } else if (topology === 'line') {
    for (let i = 0; i < nNodes - 1; i++) {
      edges.push([i, i + 1]);
    }
  }

  return { nodes, edges };
}

export default App;
