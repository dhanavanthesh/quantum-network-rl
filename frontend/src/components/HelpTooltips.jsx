/**
 * Help Tooltips - Explain what everything means
 */

import React, { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';

export function Tooltip({ text, children }) {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-gray-400 hover:text-blue-400 ml-1"
      >
        <HelpCircle className="w-4 h-4 inline" />
      </button>
      {show && (
        <div className="absolute z-50 w-64 p-3 bg-gray-700 border border-gray-600 rounded-lg shadow-lg text-sm left-0 top-6">
          {text}
        </div>
      )}
    </div>
  );
}

export function InfoBox({ title, children, type = 'info' }) {
  const colors = {
    info: 'border-blue-500 bg-blue-900/20',
    success: 'border-green-500 bg-green-900/20',
    warning: 'border-yellow-500 bg-yellow-900/20',
    error: 'border-red-500 bg-red-900/20'
  };

  return (
    <div className={`border-l-4 p-4 mb-4 ${colors[type]}`}>
      <div className="font-bold mb-2">{title}</div>
      <div className="text-sm text-gray-300">{children}</div>
    </div>
  );
}

export const HELP_TEXT = {
  networkSize: "Number of quantum nodes (devices) in the network. Each node is like a quantum computer trying to communicate.",

  topology: `How nodes are connected:
  • Grid: Nodes arranged in a 2D grid (like a checkerboard)
  • Line: Nodes in a straight line
  • Scale-Free: Some nodes have many connections (like social networks)
  • Random: Random connections`,

  temporalBins: "Time slots (32 = 32 different time windows for communication). More slots = more options but slower training.",

  spectralBins: "Frequency channels (16 = 16 different frequencies). Think of it like WiFi channels.",

  episodes: "How many times AI agents practice. More episodes = better learning but takes longer. 200 episodes ≈ 45-75 minutes.",

  learningRate: "How fast AI agents learn. Higher = faster learning but less stable. 0.15 is a good default.",

  nodeStatus: {
    idle: "Node is not communicating (waiting)",
    active: "Node is actively sending quantum data",
    collision: "Two nodes tried to use the same time-frequency slot (DATA LOST!)",
    learning: "AI agent is learning from experience"
  },

  metrics: {
    informationDensity: "How much data is packed per photon (quantum particle). Higher = more efficient.",
    networkEfficiency: "Percentage of successful communications (no collisions). Goal: >90%",
    collisionRate: "Percentage of failed communications. Goal: <10%",
    convergence: "How fast AI is learning. 100% = fully trained",
    avgFidelity: "Quality of quantum states (0-1). Higher = less errors. Goal: >0.95",
    throughput: "Successful transmissions per second. Higher = better performance"
  }
};

export function WorkflowGuide({ currentStep }) {
  const steps = [
    {
      number: 1,
      title: "Create Network",
      description: "Set up the quantum network topology (how nodes are connected)",
      action: "Click 'Create Simulation'",
      duration: "5 seconds"
    },
    {
      number: 2,
      title: "Train AI Agents",
      description: "100 AI agents learn to avoid collisions by trial-and-error",
      action: "Click 'Start Training'",
      duration: "45-75 minutes"
    },
    {
      number: 3,
      title: "View Results",
      description: "See how well the AI learned and compare to traditional methods",
      action: "Charts update automatically",
      duration: "Instant"
    },
    {
      number: 4,
      title: "Download Models",
      description: "Save the trained AI models for later use or research",
      action: "Click 'Download Models'",
      duration: "5 seconds"
    }
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-4">
      <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
        <span className="text-blue-400">📋</span>
        How To Use This Simulator
      </h3>

      <div className="space-y-3">
        {steps.map((step) => {
          const isActive = step.number === currentStep;
          const isCompleted = step.number < currentStep;

          return (
            <div
              key={step.number}
              className={`border-l-4 p-3 ${
                isActive ? 'border-yellow-400 bg-yellow-900/20' :
                isCompleted ? 'border-green-500 bg-green-900/10' :
                'border-gray-600 bg-gray-700/50'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                  isActive ? 'bg-yellow-400 text-black' :
                  isCompleted ? 'bg-green-500 text-white' :
                  'bg-gray-600 text-white'
                }`}>
                  {isCompleted ? '✓' : step.number}
                </div>

                <div className="flex-1">
                  <div className="font-bold">{step.title}</div>
                  <div className="text-sm text-gray-300 mt-1">{step.description}</div>
                  <div className="text-xs text-gray-400 mt-2">
                    <span className="font-semibold">Action:</span> {step.action}
                    <span className="mx-2">•</span>
                    <span className="font-semibold">Time:</span> {step.duration}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function NetworkTopologyExplainer() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-4">
      <h3 className="text-lg font-bold mb-3">🌐 What is "Network Topology"?</h3>

      <div className="text-sm text-gray-300 space-y-2">
        <p>
          The <span className="text-blue-400 font-semibold">3D visualization</span> shows how quantum nodes are connected:
        </p>

        <ul className="list-disc list-inside space-y-1 ml-2">
          <li><span className="font-semibold">Blue Spheres</span> = Quantum nodes (devices)</li>
          <li><span className="font-semibold">White Lines</span> = Quantum communication channels</li>
          <li><span className="font-semibold">Colors</span> = Node status (see legend below)</li>
        </ul>

        <div className="mt-3 p-3 bg-gray-700 rounded">
          <div className="font-semibold mb-2">Node Colors:</div>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500"></div>
              <span><strong>Gray</strong> = Idle (not transmitting)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span><strong>Green</strong> = Active (sending data successfully)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span><strong>Red</strong> = Collision (data lost!)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span><strong>Yellow</strong> = Learning (AI adjusting strategy)</span>
            </div>
          </div>
        </div>

        <p className="mt-3 text-xs text-gray-400 italic">
          💡 Tip: Drag to rotate, scroll to zoom, right-click to pan
        </p>
      </div>
    </div>
  );
}
