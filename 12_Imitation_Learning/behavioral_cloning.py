"""
Implementation of Behavioral Cloning for Imitation Learning.

Behavioral Cloning is a simple approach to imitation learning where we directly
learn a policy from expert demonstrations using supervised learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ExpertDataset(Dataset):
    """
    Dataset for expert demonstrations.
    """
    
    def __init__(self, states, actions):
        """
        Initialize dataset.
        
        Args:
            states: Expert states
            actions: Expert actions
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.states)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.states[idx], self.actions[idx]

class PolicyNetwork(nn.Module):
    """
    Neural network for policy.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State
            
        Returns:
            logits: Action logits
        """
        return self.network(state)

class BehavioralCloningAgent:
    """
    Behavioral Cloning agent for imitation learning.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001):
        """
        Initialize Behavioral Cloning agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, expert_states, expert_actions, batch_size=64, epochs=100):
        """
        Train policy from expert demonstrations.
        
        Args:
            expert_states: Expert states
            expert_actions: Expert actions
            batch_size: Batch size
            epochs: Number of epochs
            
        Returns:
            losses: Training losses
        """
        # Create dataset
        dataset = ExpertDataset(expert_states, expert_actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for states, actions in dataloader:
                # Forward pass
                logits = self.policy(states)
                
                # Calculate loss
                loss = self.criterion(logits, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Store loss
            losses.append(epoch_loss / len(dataloader))
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def select_action(self, state):
        """
        Select action from policy.
        
        Args:
            state: State
            
        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            logits = self.policy(state)
            action = torch.argmax(logits, dim=1).item()
        
        return action

def generate_expert_demonstrations(env, n_episodes=100):
    """
    Generate expert demonstrations using a simple heuristic policy.
    
    Args:
        env: Environment
        n_episodes: Number of episodes
        
    Returns:
        states: Expert states
        actions: Expert actions
    """
    states = []
    actions = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Expert policy for CartPole: move cart in direction of pole tilt
            if state[2] > 0:  # Pole tilting right
                action = 1  # Move cart right
            else:  # Pole tilting left
                action = 0  # Move cart left
            
            # Store state and action
            states.append(state)
            actions.append(action)
            
            # Take action
            next_state, _, done, _ = env.step(action)
            
            # Update state
            state = next_state
    
    return np.array(states), np.array(actions)

def evaluate_agent(env, agent, n_episodes=100):
    """
    Evaluate agent on environment.
    
    Args:
        env: Environment
        agent: Agent
        n_episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
    """
    rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
    
    return rewards

def main():
    """
    Main function to demonstrate behavioral cloning.
    """
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    expert_states, expert_actions = generate_expert_demonstrations(env)
    
    # Initialize agent
    agent = BehavioralCloningAgent(state_dim, action_dim)
    
    # Train agent
    print("Training agent...")
    losses = agent.train(expert_states, expert_actions, epochs=100)
    
    # Evaluate agent
    print("Evaluating agent...")
    rewards = evaluate_agent(env, agent)
    
    # Print results
    print(f"Average reward: {np.mean(rewards):.2f}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Behavioral Cloning Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('behavioral_cloning_loss.png')
    plt.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Behavioral Cloning Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig('behavioral_cloning_rewards.png')
    plt.close()
    
    print("Behavioral cloning results saved as PNG files.")

if __name__ == "__main__":
    main() 