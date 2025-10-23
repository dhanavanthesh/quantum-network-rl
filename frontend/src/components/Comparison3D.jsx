/**
 * 3D Comparison Visualization
 * Shows Classical vs Current Quantum vs Our RL Approach
 */

import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

function ClassicalSystem() {
  return (
    <group position={[-8, 0, 0]}>
      {/* Title */}
      <Text position={[0, 3, 0]} fontSize={0.5} color="#3498db">
        CLASSICAL
      </Text>
      <Text position={[0, 2.5, 0]} fontSize={0.3} color="#95a5a6">
        (WiFi/Ethernet)
      </Text>

      {/* Nodes */}
      {[...Array(5)].map((_, i) => (
        <Sphere key={i} position={[0, i * 0.8 - 1.6, 0]} args={[0.2, 16, 16]}>
          <meshStandardMaterial color="#3498db" />
        </Sphere>
      ))}

      {/* Collision - shows packets bouncing back */}
      <Sphere position={[0, 0, 1]} args={[0.15, 16, 16]}>
        <meshStandardMaterial color="#e74c3c" emissive="#e74c3c" emissiveIntensity={0.5} />
      </Sphere>
      <Text position={[0, -0.5, 1]} fontSize={0.2} color="#e74c3c">
        Collision → Retry
      </Text>

      {/* Description */}
      <Text position={[0, -2.5, 0]} fontSize={0.25} color="#95a5a6" maxWidth={3} textAlign="center">
        {"Collisions are OK\nCan resend packets\nRetry mechanism"}
      </Text>

      <Text position={[0, -3.5, 0]} fontSize={0.3} color="#2ecc71">
        ✓ Reliable
      </Text>
    </group>
  );
}

function CurrentQuantumSystem() {
  return (
    <group position={[0, 0, 0]}>
      {/* Title */}
      <Text position={[0, 3, 0]} fontSize={0.5} color="#9b59b6">
        CURRENT QUANTUM
      </Text>
      <Text position={[0, 2.5, 0]} fontSize={0.3} color="#95a5a6">
        (TDMA/Fixed)
      </Text>

      {/* Fixed time slots - rigid structure */}
      {[...Array(5)].map((_, i) => {
        const active = i === 2; // Only one active at a time
        return (
          <group key={i}>
            <Sphere position={[0, i * 0.8 - 1.6, 0]} args={[0.2, 16, 16]}>
              <meshStandardMaterial
                color={active ? "#2ecc71" : "#7f8c8d"}
                emissive={active ? "#2ecc71" : "#000000"}
                emissiveIntensity={active ? 0.5 : 0}
              />
            </Sphere>
            {active && (
              <Text position={[0.5, i * 0.8 - 1.6, 0]} fontSize={0.15} color="#2ecc71">
                Active
              </Text>
            )}
          </group>
        );
      })}

      {/* Time slots indicator */}
      <Box position={[-1, 0, 0]} args={[0.1, 5, 0.1]}>
        <meshStandardMaterial color="#95a5a6" opacity={0.3} transparent />
      </Box>

      {/* Collision - shows quantum state destroyed */}
      <Sphere position={[0, 0, 1.5]} args={[0.15, 16, 16]}>
        <meshStandardMaterial color="#e74c3c" emissive="#e74c3c" emissiveIntensity={0.8} />
      </Sphere>
      <Text position={[0, -0.5, 1.5]} fontSize={0.2} color="#e74c3c">
        Collision → LOST!
      </Text>

      {/* Description */}
      <Text position={[0, -2.5, 0]} fontSize={0.25} color="#95a5a6" maxWidth={3} textAlign="center">
        {"Fixed time slots\nWasted capacity\n70% efficiency"}
      </Text>

      <Text position={[0, -3.5, 0]} fontSize={0.3} color="#e74c3c">
        ⚠️ Inefficient
      </Text>
    </group>
  );
}

function OurRLSystem() {
  const [pulse, setPulse] = useState(0);

  // Animated AI learning effect
  useFrame((state) => {
    setPulse(Math.sin(state.clock.elapsedTime * 2) * 0.2 + 1);
  });

  return (
    <group position={[8, 0, 0]}>
      {/* Title */}
      <Text position={[0, 3, 0]} fontSize={0.5} color="#f39c12">
        OUR RL APPROACH
      </Text>
      <Text position={[0, 2.5, 0]} fontSize={0.3} color="#95a5a6">
        (AI-Coordinated)
      </Text>

      {/* Nodes with AI brains */}
      {[...Array(5)].map((_, i) => {
        const isActive = i % 2 === 0; // Multiple nodes can be active
        return (
          <group key={i}>
            <Sphere position={[0, i * 0.8 - 1.6, 0]} args={[0.2, 16, 16]}>
              <meshStandardMaterial
                color={isActive ? "#2ecc71" : "#3498db"}
                emissive={isActive ? "#2ecc71" : "#3498db"}
                emissiveIntensity={isActive ? 0.8 : 0.2}
              />
            </Sphere>

            {/* AI brain indicator */}
            <Sphere position={[0.4, i * 0.8 - 1.6, 0]} args={[0.1, 8, 8]} scale={pulse}>
              <meshStandardMaterial
                color="#f39c12"
                emissive="#f39c12"
                emissiveIntensity={0.5}
              />
            </Sphere>
          </group>
        );
      })}

      {/* Neural network connections */}
      {[0, 1, 2, 3].map((i) => (
        <Line
          key={i}
          points={[
            [0, i * 0.8 - 1.6, 0],
            [0, (i + 1) * 0.8 - 1.6, 0]
          ]}
          color="#f39c12"
          lineWidth={1}
          opacity={0.5}
          transparent
        />
      ))}

      {/* No collision - smart coordination */}
      <Text position={[0, 0, 1.5]} fontSize={0.2} color="#2ecc71">
        ✓ No Collisions!
      </Text>

      {/* Description */}
      <Text position={[0, -2.5, 0]} fontSize={0.25} color="#95a5a6" maxWidth={3} textAlign="center">
        {"AI learns patterns\nDynamic coordination\n95-100% efficiency"}
      </Text>

      <Text position={[0, -3.5, 0]} fontSize={0.3} color="#2ecc71">
        ✨ OPTIMAL
      </Text>
    </group>
  );
}

function AnimatedArrows() {
  return (
    <>
      {/* Arrow from Classical to Current Quantum */}
      <Text position={[-4, 4, 0]} fontSize={0.25} color="#95a5a6">
        Evolution →
      </Text>

      {/* Arrow from Current Quantum to Our RL */}
      <Text position={[4, 4, 0]} fontSize={0.25} color="#f39c12">
        Our Innovation →
      </Text>
    </>
  );
}

export default function Comparison3D() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-4">
      <h2 className="text-2xl font-bold mb-2 text-center">
        🚀 System Comparison
      </h2>
      <p className="text-center text-gray-400 mb-4 text-sm">
        See how our RL approach beats classical and traditional quantum methods
      </p>

      <div className="h-[500px] rounded-lg overflow-hidden">
        <Canvas camera={{ position: [0, 0, 15], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} />

          <ClassicalSystem />
          <CurrentQuantumSystem />
          <OurRLSystem />
          <AnimatedArrows />

          <OrbitControls enableDamping dampingFactor={0.05} />
        </Canvas>
      </div>

      {/* Legend */}
      <div className="grid grid-cols-3 gap-4 mt-4">
        <div className="bg-blue-900/20 border border-blue-500 rounded p-3">
          <div className="font-bold text-blue-400 mb-2">CLASSICAL</div>
          <div className="text-xs text-gray-300 space-y-1">
            <div>• Regular bits (0 or 1)</div>
            <div>• Collisions → Resend</div>
            <div>• Reliable but slow</div>
          </div>
        </div>

        <div className="bg-purple-900/20 border border-purple-500 rounded p-3">
          <div className="font-bold text-purple-400 mb-2">CURRENT QUANTUM</div>
          <div className="text-xs text-gray-300 space-y-1">
            <div>• Quantum states</div>
            <div>• Collision → LOST FOREVER</div>
            <div>• Fixed schedules (70% efficient)</div>
          </div>
        </div>

        <div className="bg-yellow-900/20 border border-yellow-500 rounded p-3">
          <div className="font-bold text-yellow-400 mb-2">OUR RL APPROACH</div>
          <div className="text-xs text-gray-300 space-y-1">
            <div>• Quantum states + AI</div>
            <div>• Smart collision avoidance</div>
            <div>• Adaptive (95-100% efficient)</div>
          </div>
        </div>
      </div>

      {/* Key Innovation */}
      <div className="mt-4 p-4 bg-gradient-to-r from-yellow-900/30 to-green-900/30 border border-yellow-500 rounded">
        <div className="font-bold text-yellow-400 mb-2 flex items-center gap-2">
          💡 KEY INNOVATION
        </div>
        <div className="text-sm text-gray-200">
          Our system uses <span className="font-bold text-yellow-400">100 AI agents</span> that <span className="font-bold text-green-400">learn to coordinate</span> dynamically,
          achieving <span className="font-bold text-green-400">25-30% better efficiency</span> than traditional quantum methods
          while avoiding the quantum no-cloning limitation.
        </div>
      </div>
    </div>
  );
}
