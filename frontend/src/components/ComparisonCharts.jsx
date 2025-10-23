/**
 * Comparison Charts Component
 * Displays training progress and comparisons
 */

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart2 } from 'lucide-react';

export default function ComparisonCharts({ episodeStats, simulationStatus }) {
  if (!episodeStats || episodeStats.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <BarChart2 className="w-5 h-5 text-blue-400" />
          <h2 className="text-xl font-bold">Training Progress</h2>
        </div>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No training data available yet
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const chartData = episodeStats.map(stat => ({
    episode: stat.episode,
    reward: stat.avg_reward,
    efficiency: stat.network_efficiency * 100,
    collisionRate: stat.collision_rate * 100,
  }));

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 className="w-5 h-5 text-blue-400" />
        <h2 className="text-xl font-bold">Training Progress</h2>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Average Reward Chart */}
        <div className="bg-gray-700 rounded p-3">
          <h3 className="text-sm font-semibold mb-2 text-gray-300">Average Reward</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="episode" stroke="#888" fontSize={12} />
              <YAxis stroke="#888" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Line type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Network Efficiency Chart */}
        <div className="bg-gray-700 rounded p-3">
          <h3 className="text-sm font-semibold mb-2 text-gray-300">Network Efficiency (%)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="episode" stroke="#888" fontSize={12} />
              <YAxis stroke="#888" fontSize={12} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Line type="monotone" dataKey="efficiency" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Collision Rate Chart */}
        <div className="bg-gray-700 rounded p-3">
          <h3 className="text-sm font-semibold mb-2 text-gray-300">Collision Rate (%)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="episode" stroke="#888" fontSize={12} />
              <YAxis stroke="#888" fontSize={12} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Line type="monotone" dataKey="collisionRate" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Summary Stats */}
        <div className="bg-gray-700 rounded p-3">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Summary Statistics</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Episodes Completed:</span>
              <span className="font-semibold">{episodeStats.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Latest Reward:</span>
              <span className="font-semibold">
                {episodeStats[episodeStats.length - 1]?.avg_reward.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Latest Efficiency:</span>
              <span className="font-semibold text-green-400">
                {(episodeStats[episodeStats.length - 1]?.network_efficiency * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Latest Collision:</span>
              <span className="font-semibold text-red-400">
                {(episodeStats[episodeStats.length - 1]?.collision_rate * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
