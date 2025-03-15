"""
Implementation of neural network function approximation for reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Initialize Q-network.
        
        Args:
            input_dim: Input dimension (state dimension)
            output_dim: Output dimension (number of actions)
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            q_values: Q-values for all actions
        """
        return self.network(x)

class DQNAgent:
    """
    Deep Q-Network agent.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-network based on observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])
        
        # Get current Q-value
        q_values = self.q_network(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Get next Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            next_q_value = next_q_values.max(1)[0]
        
        # Calculate target Q-value
        expected_q_value = reward_tensor + (1 - done_tensor) * self.gamma * next_q_value
        
        # Calculate loss
        loss = self.loss_fn(q_value, expected_q_value)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item() 