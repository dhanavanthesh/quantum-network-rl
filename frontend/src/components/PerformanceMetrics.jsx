/**
 * Performance Metrics Component
 * Displays real-time performance metrics
 */

import React from 'react';
import { Activity, Zap, Target, TrendingUp } from 'lucide-react';

function MetricCard({ title, value, unit, icon: Icon, color }) {
  return (
    <div className="bg-gray-700 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${color}`} />
        <div className="text-xs text-gray-400">{title}</div>
      </div>
      <div className="text-2xl font-bold">
        {value !== null && value !== undefined ? value.toFixed(2) : '--'}
      </div>
      {unit && <div className="text-xs text-gray-500 mt-1">{unit}</div>}
    </div>
  );
}

export default function PerformanceMetrics({ metrics, simulationStatus }) {
  const displayMetrics = metrics || {
    information_density: 0,
    network_efficiency: 0,
    collision_rate: 0,
    convergence_rate: 0,
    average_fidelity: 0,
    throughput: 0,
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-green-400" />
        <h2 className="text-xl font-bold">Real-Time Metrics</h2>
      </div>

      <div className="space-y-3">
        <MetricCard
          title="Information Density"
          value={displayMetrics.information_density}
          unit="bits/photon"
          icon={Zap}
          color="text-yellow-400"
        />

        <MetricCard
          title="Network Efficiency"
          value={displayMetrics.network_efficiency * 100}
          unit="%"
          icon={Target}
          color="text-green-400"
        />

        <MetricCard
          title="Collision Rate"
          value={displayMetrics.collision_rate * 100}
          unit="%"
          icon={Activity}
          color="text-red-400"
        />

        <MetricCard
          title="Convergence"
          value={displayMetrics.convergence_rate * 100}
          unit="%"
          icon={TrendingUp}
          color="text-blue-400"
        />

        <MetricCard
          title="Avg Fidelity"
          value={displayMetrics.average_fidelity}
          unit="F"
          icon={Zap}
          color="text-purple-400"
        />

        <MetricCard
          title="Throughput"
          value={displayMetrics.throughput}
          unit="tx/s"
          icon={Activity}
          color="text-cyan-400"
        />
      </div>

      {/* Status Indicator */}
      <div className="mt-4 p-3 bg-gray-700 rounded">
        <div className="text-sm text-gray-400 mb-1">System Status</div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            simulationStatus === 'training' ? 'bg-green-400 animate-pulse' :
            simulationStatus === 'completed' ? 'bg-blue-400' :
            'bg-gray-400'
          }`}></div>
          <span className="text-sm font-medium">
            {simulationStatus === 'idle' && 'Idle'}
            {simulationStatus === 'created' && 'Ready'}
            {simulationStatus === 'training' && 'Training'}
            {simulationStatus === 'completed' && 'Completed'}
          </span>
        </div>
      </div>
    </div>
  );
}
