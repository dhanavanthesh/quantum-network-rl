/**
 * API Service
 * Handles communication with FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Simulation endpoints
export const createSimulation = async (config) => {
  const response = await api.post('/api/simulation/create', config);
  return response.data;
};

export const trainSimulation = async (simulationId) => {
  const response = await api.post(`/api/simulation/${simulationId}/train`);
  return response.data;
};

export const getSimulationStatus = async (simulationId) => {
  const response = await api.get(`/api/simulation/${simulationId}/status`);
  return response.data;
};

export const getSimulationResults = async (simulationId) => {
  const response = await api.get(`/api/simulation/${simulationId}/results`);
  return response.data;
};

export const getSimulationMetrics = async (simulationId) => {
  const response = await api.get(`/api/simulation/${simulationId}/metrics`);
  return response.data;
};

export const deleteSimulation = async (simulationId) => {
  const response = await api.delete(`/api/simulation/${simulationId}`);
  return response.data;
};

export const listSimulations = async () => {
  const response = await api.get('/api/simulation/list');
  return response.data;
};

// Network endpoints
export const generateNetwork = async (config) => {
  const response = await api.post('/api/network/generate', config);
  return response.data;
};

export const getAvailableTopologies = async () => {
  const response = await api.get('/api/network/topologies');
  return response.data;
};

export const analyzeNetwork = async (config) => {
  const response = await api.post('/api/network/analyze', config);
  return response.data;
};

// Metrics endpoints
export const getAvailableMetrics = async () => {
  const response = await api.get('/api/metrics/available');
  return response.data;
};

export const getBaselineAlgorithms = async () => {
  const response = await api.get('/api/metrics/baselines');
  return response.data;
};

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
