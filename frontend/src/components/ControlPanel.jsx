/**
 * Control Panel Component
 * Simulation configuration and controls
 */

import React, { useState } from 'react';
import { Play, Square, Settings, Download, CheckCircle } from 'lucide-react';
import { Tooltip, HELP_TEXT } from './HelpTooltips';

export default function ControlPanel({ onCreateSimulation, onStartTraining, simulationStatus, simulationId }) {
  const [config, setConfig] = useState({
    n_nodes: 100,
    topology: 'grid',
    n_temporal: 32,
    n_spectral: 16,
    n_episodes: 200,
    episode_length: 100,
    learning_rate: 0.15,
    discount_factor: 0.92,
    batch_size: 64,
    random_seed: 42,
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name === 'topology' ? value : parseFloat(value) || parseInt(value)
    }));
  };

  const handleCreate = () => {
    onCreateSimulation(config);
  };

  const handleStart = () => {
    onStartTraining();
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-blue-400" />
        <h2 className="text-xl font-bold">Control Panel</h2>
      </div>

      {/* Configuration Form */}
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Network Size (Quantum Nodes)
            <Tooltip text={HELP_TEXT.networkSize} />
          </label>
          <input
            type="number"
            name="n_nodes"
            value={config.n_nodes}
            onChange={handleInputChange}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            min="10"
            max="500"
          />
          <div className="text-xs text-gray-500 mt-1">100 nodes ≈ 45-75 min training</div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Topology (How Nodes Connect)
            <Tooltip text={HELP_TEXT.topology} />
          </label>
          <select
            name="topology"
            value={config.topology}
            onChange={handleInputChange}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          >
            <option value="grid">Grid (Recommended)</option>
            <option value="line">Line</option>
            <option value="scale_free">Scale-Free</option>
            <option value="random_geometric">Random Geometric</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Temporal Bins (Time Slots)
            <Tooltip text={HELP_TEXT.temporalBins} />
          </label>
          <select
            name="n_temporal"
            value={config.n_temporal}
            onChange={handleInputChange}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          >
            <option value="16">16 (Faster)</option>
            <option value="32">32 (Recommended)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Spectral Bins (Frequencies)
            <Tooltip text={HELP_TEXT.spectralBins} />
          </label>
          <select
            name="n_spectral"
            value={config.n_spectral}
            onChange={handleInputChange}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          >
            <option value="8">8 (Faster)</option>
            <option value="16">16 (Recommended)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Episodes (Training Rounds)
            <Tooltip text={HELP_TEXT.episodes} />
          </label>
          <input
            type="number"
            name="n_episodes"
            value={config.n_episodes}
            onChange={handleInputChange}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            min="50"
            max="500"
          />
          <div className="text-xs text-gray-500 mt-1">200 episodes ≈ 45-75 min</div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Learning Rate (AI Speed)
            <Tooltip text={HELP_TEXT.learningRate} />
          </label>
          <input
            type="number"
            name="learning_rate"
            value={config.learning_rate}
            onChange={handleInputChange}
            step="0.01"
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          />
          <div className="text-xs text-gray-500 mt-1">Default: 0.15 (recommended)</div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-6 space-y-2">
        <button
          onClick={handleCreate}
          disabled={simulationStatus !== 'idle' && simulationStatus !== 'completed'}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-2 px-4 rounded flex items-center justify-center gap-2"
        >
          <Settings className="w-4 h-4" />
          Create Simulation
        </button>

        <button
          onClick={handleStart}
          disabled={simulationStatus !== 'created'}
          className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-2 px-4 rounded flex items-center justify-center gap-2"
        >
          <Play className="w-4 h-4" />
          Start Training
        </button>

        {simulationStatus === 'completed' && simulationId && (
          <a
            href={`http://localhost:8000/api/simulation/${simulationId}/download`}
            download
            className="w-full bg-yellow-600 hover:bg-yellow-700 text-white py-2 px-4 rounded flex items-center justify-center gap-2 block text-center"
          >
            <Download className="w-4 h-4 inline" />
            Download Models
          </a>
        )}
      </div>

      {/* Status */}
      <div className="mt-4 p-3 bg-gray-700 rounded">
        <div className="text-sm text-gray-400">Status</div>
        <div className="font-semibold mt-1">
          {simulationStatus === 'idle' && 'Ready'}
          {simulationStatus === 'created' && 'Simulation Created'}
          {simulationStatus === 'training' && (
            <span className="text-yellow-400">Training...</span>
          )}
          {simulationStatus === 'completed' && (
            <span className="text-green-400">Completed</span>
          )}
        </div>
      </div>
    </div>
  );
}
