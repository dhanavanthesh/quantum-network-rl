"""
Distributed Q-Learning Agent for Temporal-Spectral Resource Allocation
Implements Dec-POMDP with DQN for multi-agent quantum network coordination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Optional
import random


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """Deep Q-Network with configurable architecture"""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [256, 128, 64]):
        """
        Initialize DQN

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
        """
        super(DQN, self).__init__()

        layers = []
        input_dim = state_dim

        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer with fixed size"""

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DistributedQLearningAgent:
    """
    Distributed Q-Learning agent for quantum network node
    Each node maintains local Q-function and coordinates via beacon broadcasting
    """

    def __init__(self,
                 node_id: int,
                 n_temporal: int = 32,
                 n_spectral: int = 16,
                 n_neighbors: int = 4,
                 learning_rate: float = 0.15,
                 discount_factor: float = 0.92,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 target_update_freq: int = 10):
        """
        Initialize distributed Q-learning agent

        Args:
            node_id: Unique identifier for this node
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
            n_neighbors: Average number of neighbors (for state dimension)
            learning_rate: Learning rate α
            discount_factor: Discount factor γ
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Frequency to update target network
        """
        self.node_id = node_id
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral
        self.n_modes = n_temporal * n_spectral

        # State space: local occupation + neighbor occupations + network efficiency
        # Local: one-hot encoded mode (n_modes) + continuous features (4)
        # Neighbors: aggregated features (8)
        # Total: n_modes + 12
        self.state_dim = self.n_modes + 12

        # Action space: select one of the temporal-spectral modes
        self.action_dim = self.n_modes

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Neural networks
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

    def encode_state(self,
                    current_mode: int,
                    neighbor_modes: List[int],
                    network_state: Dict) -> np.ndarray:
        """
        Encode current state into feature vector

        Args:
            current_mode: Current temporal-spectral mode of this node
            neighbor_modes: Modes selected by neighbor nodes
            network_state: Global network statistics

        Returns:
            State feature vector
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # One-hot encode current mode
        if current_mode >= 0:
            state[current_mode] = 1.0

        offset = self.n_modes

        # Extract temporal and spectral bins
        temporal_bin = current_mode // self.n_spectral if current_mode >= 0 else -1
        spectral_bin = current_mode % self.n_spectral if current_mode >= 0 else -1

        # Local features (4 dimensions)
        state[offset] = temporal_bin / self.n_temporal  # Normalized temporal position
        state[offset + 1] = spectral_bin / self.n_spectral  # Normalized spectral position
        state[offset + 2] = network_state.get('local_efficiency', 0.0)
        state[offset + 3] = network_state.get('local_collision_rate', 0.0)
        offset += 4

        # Neighbor features (8 dimensions)
        if neighbor_modes:
            neighbor_temporal = [m // self.n_spectral for m in neighbor_modes]
            neighbor_spectral = [m % self.n_spectral for m in neighbor_modes]

            state[offset] = np.mean(neighbor_temporal) / self.n_temporal
            state[offset + 1] = np.std(neighbor_temporal) / self.n_temporal
            state[offset + 2] = np.mean(neighbor_spectral) / self.n_spectral
            state[offset + 3] = np.std(neighbor_spectral) / self.n_spectral
            state[offset + 4] = len(neighbor_modes) / 10.0  # Normalized neighbor count
            state[offset + 5] = network_state.get('neighbor_collision_rate', 0.0)
            state[offset + 6] = network_state.get('channel_fidelity_avg', 0.9)
            state[offset + 7] = network_state.get('network_load', 0.5)

        return state

    def select_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            evaluation: If True, use greedy policy (no exploration)

        Returns:
            Selected action (mode index)
        """
        if not evaluation and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step

        Returns:
            Loss value if training was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor([e.done for e in batch])

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update training step
        self.training_step += 1

        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def update_episode_reward(self, total_reward: float):
        """Store total episode reward"""
        self.episode_rewards.append(total_reward)

    def get_mode_from_action(self, action: int) -> Tuple[int, int]:
        """
        Convert action index to (temporal_bin, spectral_bin)

        Args:
            action: Action index

        Returns:
            (temporal_bin, spectral_bin)
        """
        temporal_bin = action // self.n_spectral
        spectral_bin = action % self.n_spectral
        return temporal_bin, spectral_bin

    def get_action_from_mode(self, temporal_bin: int, spectral_bin: int) -> int:
        """
        Convert (temporal_bin, spectral_bin) to action index

        Args:
            temporal_bin: Temporal bin
            spectral_bin: Spectral bin

        Returns:
            Action index
        """
        return temporal_bin * self.n_spectral + spectral_bin

    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, filepath)

    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']

    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'node_id': self.node_id,
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'avg_loss_last_100': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'total_episodes': len(self.episode_rewards),
            'buffer_size': len(self.replay_buffer)
        }


class BeaconBroadcaster:
    """
    Classical coordination via beacon broadcasting
    Implements O(log N) overhead protocol for mode coordination
    """

    def __init__(self, n_nodes: int):
        """
        Initialize beacon broadcaster

        Args:
            n_nodes: Number of nodes in network
        """
        self.n_nodes = n_nodes
        self.beacon_messages = {}  # node_id -> beacon_info

    def broadcast_beacon(self, node_id: int, mode: int, timestamp: int):
        """
        Broadcast beacon with selected mode

        Args:
            node_id: Node broadcasting
            mode: Selected temporal-spectral mode
            timestamp: Current time step
        """
        self.beacon_messages[node_id] = {
            'mode': mode,
            'timestamp': timestamp
        }

    def receive_beacons(self, node_id: int, neighbor_ids: List[int]) -> List[int]:
        """
        Receive beacons from neighbors

        Args:
            node_id: Receiving node
            neighbor_ids: List of neighbor node IDs

        Returns:
            List of neighbor modes
        """
        neighbor_modes = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.beacon_messages:
                neighbor_modes.append(self.beacon_messages[neighbor_id]['mode'])

        return neighbor_modes

    def detect_collision(self, node_id: int, mode: int, neighbor_ids: List[int]) -> bool:
        """
        Detect if there's a collision with neighbors

        Args:
            node_id: Current node
            mode: Selected mode
            neighbor_ids: Neighbor node IDs

        Returns:
            True if collision detected
        """
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.beacon_messages:
                if self.beacon_messages[neighbor_id]['mode'] == mode:
                    return True
        return False

    def clear_old_beacons(self, current_timestamp: int, timeout: int = 10):
        """
        Clear old beacon messages

        Args:
            current_timestamp: Current timestamp
            timeout: Beacon timeout
        """
        to_remove = []
        for node_id, beacon in self.beacon_messages.items():
            if current_timestamp - beacon['timestamp'] > timeout:
                to_remove.append(node_id)

        for node_id in to_remove:
            del self.beacon_messages[node_id]
