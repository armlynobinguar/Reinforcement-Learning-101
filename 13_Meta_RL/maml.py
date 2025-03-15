"""
Implementation of Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning.

MAML is a meta-learning algorithm that aims to find a good initialization for a model
that can be quickly adapted to new tasks with just a few gradient steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random

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
    
    def get_action(self, state):
        """
        Sample action from policy.
        
        Args:
            state: State
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

class MAML:
    """
    Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, inner_lr=0.1, meta_lr=0.001, gamma=0.99):
        """
        Initialize MAML.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            inner_lr: Inner loop learning rate
            meta_lr: Meta learning rate
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.gamma = gamma
        
        # Initialize meta policy
        self.meta_policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        
        # Initialize meta optimizer
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=meta_lr)
    
    def adapt(self, trajectories):
        """
        Adapt policy to new task using trajectories.
        
        Args:
            trajectories: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            adapted_policy: Adapted policy
        """
        # Clone meta policy
        adapted_policy = PolicyNetwork(self.state_dim, self.action_dim)
        adapted_policy.load_state_dict(self.meta_policy.state_dict())
        
        # Calculate returns
        returns = []
        for trajectory in trajectories:
            states, actions, rewards, _, _ = zip(*trajectory)
            
            # Calculate discounted returns
            discounted_rewards = []
            running_reward = 0
            for reward in reversed(rewards):
                running_reward = reward + self.gamma * running_reward
                discounted_rewards.insert(0, running_reward)
            
            # Normalize returns
            discounted_rewards = torch.FloatTensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
            
            returns.extend(discounted_rewards)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack([t[0] for trajectory in trajectories for t in trajectory]))
        actions = torch.LongTensor(np.array([t[1] for trajectory in trajectories for t in trajectory]))
        returns = torch.FloatTensor(returns)
        
        # Calculate loss
        logits = adapted_policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * returns).mean()
        
        # Update adapted policy
        grads = torch.autograd.grad(loss, adapted_policy.parameters())
        
        # Manual gradient update
        for param, grad in zip(adapted_policy.parameters(), grads):
            param.data.sub_(self.inner_lr * grad)
        
        return adapted_policy
    
    def meta_update(self, task_losses):
        """
        Update meta policy using task losses.
        
        Args:
            task_losses: List of losses for each task
            
        Returns:
            meta_loss: Meta loss
        """
        # Calculate meta loss
        meta_loss = torch.stack(task_losses).mean()
        
        # Update meta policy
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def collect_trajectories(self, env, policy, n_trajectories=10):
        """
        Collect trajectories using policy.
        
        Args:
            env: Environment
            policy: Policy
            n_trajectories: Number of trajectories
            
        Returns:
            trajectories: List of trajectories
            total_rewards: List of total rewards
        """
        trajectories = []
        total_rewards = []
        
        for _ in range(n_trajectories):
            trajectory = []
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = policy.get_action(state_tensor)
                next_state, reward, done, _ = env.step(action)
                
                trajectory.append((state, action, reward, next_state, done))
                
                state = next_state
                total_reward += reward
            
            trajectories.append(trajectory)
            total_rewards.append(total_reward)
        
        return trajectories, total_rewards

def create_cartpole_tasks():
    """
    Create different CartPole tasks by varying the pole length and cart mass.
    
    Returns:
        envs: List of environments
    """
    envs = []
    
    # Default CartPole
    env = gym.make('CartPole-v1')
    envs.append(env)
    
    # CartPole with different pole lengths
    env = gym.make('CartPole-v1')
    env.env.length = 0.3  # Shorter pole
    envs.append(env)
    
    env = gym.make('CartPole-v1')
    env.env.length = 0.7  # Longer pole
    envs.append(env)
    
    # CartPole with different cart masses
    env = gym.make('CartPole-v1')
    env.env.masscart = 0.5  # Lighter cart
    envs.append(env)
    
    env = gym.make('CartPole-v1')
    env.env.masscart = 2.0  # Heavier cart
    envs.append(env)
    
    return envs

def main():
    """
    Main function to demonstrate MAML.
    """
    # Create tasks
    envs = create_cartpole_tasks()
    
    # Get state and action dimensions
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    
    # Initialize MAML
    maml = MAML(state_dim, action_dim)
    
    # Meta-training
    print("Meta-training...")
    meta_losses = []
    task_performances = [[] for _ in range(len(envs))]
    
    for iteration in range(100):
        task_losses = []
        
        # Sample tasks
        task_indices = random.sample(range(len(envs)), 3)
        
        for task_idx in task_indices:
            env = envs[task_idx]
            
            # Collect trajectories with meta policy
            trajectories, _ = maml.collect_trajectories(env, maml.meta_policy, n_trajectories=5)
            
            # Adapt policy to task
            adapted_policy = maml.adapt(trajectories)
            
            # Collect trajectories with adapted policy
            eval_trajectories, eval_rewards = maml.collect_trajectories(env, adapted_policy, n_trajectories=5)
            
            # Calculate loss for meta update
            returns = []
            for trajectory in eval_trajectories:
                states, actions, rewards, _, _ = zip(*trajectory)
                
                # Calculate discounted returns
                discounted_rewards = []
                running_reward = 0
                for reward in reversed(rewards):
                    running_reward = reward + maml.gamma * running_reward
                    discounted_rewards.insert(0, running_reward)
                
                # Normalize returns
                discounted_rewards = torch.FloatTensor(discounted_rewards)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
                
                returns.extend(discounted_rewards)
            
            # Convert to tensors
            states = torch.FloatTensor(np.vstack([t[0] for trajectory in eval_trajectories for t in trajectory]))
            actions = torch.LongTensor(np.array([t[1] for trajectory in eval_trajectories for t in trajectory]))
            returns = torch.FloatTensor(returns)
            
            # Calculate loss
            logits = maml.meta_policy(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            loss = -(log_probs * returns).mean()
            
            task_losses.append(loss)
            
            # Store performance
            task_performances[task_idx].append(np.mean(eval_rewards))
        
        # Meta update
        meta_loss = maml.meta_update(task_losses)
        meta_losses.append(meta_loss)
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}, Meta Loss: {meta_loss:.4f}")
    
    # Meta-testing
    print("Meta-testing...")
    test_performances = []
    
    for task_idx, env in enumerate(envs):
        # Collect trajectories with meta policy
        trajectories, _ = maml.collect_trajectories(env, maml.meta_policy, n_trajectories=5)
        
        # Adapt policy to task
        adapted_policy = maml.adapt(trajectories)
        
        # Evaluate adapted policy
        _, eval_rewards = maml.collect_trajectories(env, adapted_policy, n_trajectories=10)
        
        # Store performance
        test_performances.append(np.mean(eval_rewards))
        
        print(f"Task {task_idx + 1}, Average Reward: {test_performances[-1]:.2f}")
    
    # Plot meta loss
    plt.figure(figsize=(10, 6))
    plt.plot(meta_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Meta Loss')
    plt.title('MAML Meta-Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('maml_meta_loss.png')
    plt.close()
    
    # Plot task performances
    plt.figure(figsize=(10, 6))
    for task_idx in range(len(envs)):
        plt.plot(task_performances[task_idx], label=f'Task {task_idx + 1}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('MAML Task Performances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('maml_task_performances.png')
    plt.close()
    
    # Plot test performances
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(envs) + 1), test_performances)
    plt.xlabel('Task')
    plt.ylabel('Average Reward')
    plt.title('MAML Meta-Testing Performance')
    plt.xticks(range(1, len(envs) + 1))
    plt.grid(True, alpha=0.3)
    plt.savefig('maml_test_performances.png')
    plt.close()
    
    print("MAML results saved as PNG files.")

if __name__ == "__main__":
    main() 