"""
Implementation of Proximal Policy Optimization (PPO) agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize actor-critic network.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            action_probs: Action probabilities
            state_value: State value
        """
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        """
        Select an action based on the policy.
        
        Args:
            state: State tensor
            
        Returns:
            action: Selected action
            action_prob: Probability of the selected action
            state_value: State value
        """
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_prob = action_probs[action.item()].item()
        return action.item(), action_prob, state_value.item()

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99, 
                 clip_ratio=0.2, epochs=10, batch_size=64):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            epochs: Number of optimization epochs per update
            batch_size: Batch size for optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize actor-critic network
        self.network = ActorCritic(state_dim, action_dim, hidden_dim)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize memory
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, state):
        """
        Select an action using the policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_prob, value = self.network.act(state_tensor)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.values.append(value)
        
        return action
    
    def store_transition(self, reward, done):
        """
        Store reward and done flag.
        
        Args:
            reward: Reward received
            done: Whether the episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """
        Update the agent using PPO.
        
        Returns:
            actor_loss: Actor (policy) loss
            critic_loss: Critic (value) loss
        """
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_action_probs = torch.FloatTensor(self.action_probs)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)
        values = torch.FloatTensor(self.values)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = r + self.gamma * R * (1 - d)
            advantage = R - v
            returns.append(R)
            advantages.append(advantage)
        
        returns = torch.FloatTensor(list(reversed(returns)))
        advantages = torch.FloatTensor(list(reversed(advantages)))
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            # Generate random indices
            indices = torch.randperm(len(states))
            
            # Process in batches
            for start_idx in range(0, len(states), self.batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_action_probs = old_action_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current action probabilities and values
                action_probs, state_values = self.network(batch_states)
                
                # Get probabilities of taken actions
                dist = Categorical(action_probs)
                batch_new_action_probs = dist.probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Calculate ratio
                ratio = batch_new_action_probs / batch_old_action_probs
                
                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Calculate total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return total_actor_loss / self.epochs, total_critic_loss / self.epochs 