/**
 * 3D Network Visualization Component
 * Uses React Three Fiber for WebGL rendering
 */

import React, { useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, Line } from '@react-three/drei';

function Node({ position, status, id }) {
  const colorMap = {
    idle: '#3498db',      // Blue
    active: '#2ecc71',    // Green
    collision: '#e74c3c', // Red
    learning: '#f39c12',  // Yellow
  };

  const color = colorMap[status] || colorMap.idle;

  return (
    <group position={position}>
      <Sphere args={[0.3, 16, 16]}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
      </Sphere>
      {/* Node label would go here in full implementation */}
    </group>
  );
}

function Edge({ start, end }) {
  return (
    <Line
      points={[start, end]}
      color="#4a90e2"
      lineWidth={2}
      opacity={0.6}
      transparent
    />
  );
}

export default function NetworkVisualization3D({ networkData, simulationStatus }) {
  if (!networkData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900 rounded">
        <p className="text-gray-500">No network data available</p>
      </div>
    );
  }

  const { nodes, edges } = networkData;

  return (
    <div className="w-full h-full bg-black rounded">
      <Canvas camera={{ position: [15, 15, 15], fov: 60 }}>
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* Grid floor */}
        <gridHelper args={[40, 40, '#444444', '#222222']} />

        {/* Network edges */}
        {edges && edges.map((edge, idx) => {
          const startNode = nodes.find(n => n.id === edge[0]);
          const endNode = nodes.find(n => n.id === edge[1]);
          if (startNode && endNode) {
            return (
              <Edge
                key={`edge-${idx}`}
                start={startNode.position}
                end={endNode.position}
              />
            );
          }
          return null;
        })}

        {/* Network nodes */}
        {nodes && nodes.map(node => (
          <Node
            key={`node-${node.id}`}
            position={node.position}
            status={node.status}
            id={node.id}
          />
        ))}

        {/* Camera controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={50}
        />
      </Canvas>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 bg-gray-800 bg-opacity-90 p-3 rounded text-xs">
        <div className="font-bold mb-2">Node Status</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#3498db]"></div>
            <span>Idle</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#2ecc71]"></div>
            <span>Active</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#e74c3c]"></div>
            <span>Collision</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#f39c12]"></div>
            <span>Learning</span>
          </div>
        </div>
      </div>
    </div>
  );
}
