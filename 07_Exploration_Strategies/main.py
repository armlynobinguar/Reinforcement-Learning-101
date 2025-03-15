"""
Implementation of various exploration strategies for reinforcement learning.

This script demonstrates and compares different exploration strategies:
1. Epsilon-Greedy
2. Boltzmann Exploration
3. Upper Confidence Bound (UCB)
4. Thompson Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import defaultdict
import random
import math

class MultiArmedBandit:
    """
    Multi-armed bandit environment.
    """
    
    def __init__(self, n_arms=10):
        """
        Initialize multi-armed bandit.
        
        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.true_values = np.random.normal(0, 1, n_arms)
    
    def pull(self, arm):
        """
        Pull an arm and get reward.
        
        Args:
            arm: Arm to pull
            
        Returns:
            reward: Reward
        """
        return np.random.normal(self.true_values[arm], 1)

class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy.
    """
    
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize epsilon-greedy.
        
        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
    
    def select_action(self):
        """
        Select action using epsilon-greedy.
        
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        """
        Update Q-values.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

class BoltzmannExploration:
    """
    Boltzmann exploration strategy.
    """
    
    def __init__(self, n_arms, temperature=1.0):
        """
        Initialize Boltzmann exploration.
        
        Args:
            n_arms: Number of arms
            temperature: Temperature parameter
        """
        self.n_arms = n_arms
        self.temperature = temperature
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
    
    def select_action(self):
        """
        Select action using Boltzmann exploration.
        
        Returns:
            action: Selected action
        """
        # Compute probabilities using softmax
        exp_values = np.exp(self.q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        
        # Sample action
        return np.random.choice(self.n_arms, p=probs)
    
    def update(self, action, reward):
        """
        Update Q-values.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

class UCB:
    """
    Upper Confidence Bound (UCB) exploration strategy.
    """
    
    def __init__(self, n_arms, c=2.0):
        """
        Initialize UCB.
        
        Args:
            n_arms: Number of arms
            c: Exploration parameter
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0
    
    def select_action(self):
        """
        Select action using UCB.
        
        Returns:
            action: Selected action
        """
        # Initialize all arms if not all have been pulled
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute UCB values
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.counts)
        
        # Select arm with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """
        Update Q-values and time step.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]
        self.t += 1

class ThompsonSampling:
    """
    Thompson Sampling exploration strategy.
    """
    
    def __init__(self, n_arms):
        """
        Initialize Thompson Sampling.
        
        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Success count
        self.beta = np.ones(n_arms)   # Failure count
    
    def select_action(self):
        """
        Select action using Thompson Sampling.
        
        Returns:
            action: Selected action
        """
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)
        
        # Select arm with highest sample
        return np.argmax(samples)
    
    def update(self, action, reward):
        """
        Update Beta distribution parameters.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        # Assuming reward is in [0, 1]
        reward = max(0, min(1, (reward + 5) / 10))  # Normalize to [0, 1]
        
        if reward > 0:
            self.alpha[action] += reward
            self.beta[action] += (1 - reward)
        else:
            self.beta[action] += 1

def run_bandit_experiment(bandit, agent, n_steps=1000):
    """
    Run multi-armed bandit experiment.
    
    Args:
        bandit: Multi-armed bandit environment
        agent: Exploration strategy agent
        n_steps: Number of steps
        
    Returns:
        rewards: List of rewards
        optimal_actions: List of optimal action counts
    """
    rewards = []
    optimal_actions = []
    optimal_arm = np.argmax(bandit.true_values)
    
    for _ in range(n_steps):
        # Select action
        action = agent.select_action()
        
        # Get reward
        reward = bandit.pull(action)
        
        # Update agent
        agent.update(action, reward)
        
        # Store reward
        rewards.append(reward)
        
        # Store optimal action count
        optimal_actions.append(1 if action == optimal_arm else 0)
    
    return rewards, optimal_actions

def run_gridworld_experiment(agent_type, n_episodes=1000):
    """
    Run gridworld experiment.
    
    Args:
        agent_type: Type of exploration strategy
        n_episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
    """
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Initialize Q-values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Initialize rewards list
    rewards = []
    
    for episode in range(n_episodes):
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # Handle gym v26 reset format
            state = state[0]
        
        # Initialize total reward
        total_reward = 0
        
        # Run episode
        done = False
        while not done:
            # Select action based on agent type
            if agent_type == 'epsilon_greedy':
                # Epsilon-greedy
                if np.random.random() < 0.1:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
            elif agent_type == 'boltzmann':
                # Boltzmann exploration
                exp_values = np.exp(Q[state] / 0.5)
                probs = exp_values / np.sum(exp_values)
                action = np.random.choice(env.action_space.n, p=probs)
            elif agent_type == 'ucb':
                # UCB
                counts = defaultdict(lambda: np.zeros(env.action_space.n))
                t = episode + 1
                
                # Initialize all actions if not all have been tried
                for a in range(env.action_space.n):
                    if counts[state][a] == 0:
                        action = a
                        break
                else:
                    # Compute UCB values
                    ucb_values = Q[state] + 2.0 * np.sqrt(np.log(t) / (counts[state] + 1e-5))
                    action = np.argmax(ucb_values)
                
                counts[state][action] += 1
            else:
                # Default to epsilon-greedy
                if np.random.random() < 0.1:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
            
            # Take action
            try:
                # Try new gym API (v26+)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Fall back to old gym API
                next_state, reward, done, info = env.step(action)
            
            # Update Q-value
            Q[state][action] = Q[state][action] + 0.1 * (reward + 0.99 * np.max(Q[next_state]) - Q[state][action])
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
        
        # Store total reward
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Average Reward: {np.mean(rewards[-100:]):.2f}")
    
    return rewards

def main():
    """
    Main function to run experiments.
    """
    # Multi-armed bandit experiments
    print("Running multi-armed bandit experiments...")
    
    bandit = MultiArmedBandit(n_arms=10)
    
    # Epsilon-greedy
    epsilon_greedy = EpsilonGreedy(bandit.n_arms, epsilon=0.1)
    eg_rewards, eg_optimal = run_bandit_experiment(bandit, epsilon_greedy)
    
    # Boltzmann exploration
    boltzmann = BoltzmannExploration(bandit.n_arms, temperature=0.5)
    boltz_rewards, boltz_optimal = run_bandit_experiment(bandit, boltzmann)
    
    # UCB
    ucb = UCB(bandit.n_arms, c=2.0)
    ucb_rewards, ucb_optimal = run_bandit_experiment(bandit, ucb)
    
    # Thompson Sampling
    thompson = ThompsonSampling(bandit.n_arms)
    ts_rewards, ts_optimal = run_bandit_experiment(bandit, thompson)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot average reward
    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(eg_rewards) / np.arange(1, len(eg_rewards) + 1), label='Epsilon-Greedy')
    plt.plot(np.cumsum(boltz_rewards) / np.arange(1, len(boltz_rewards) + 1), label='Boltzmann')
    plt.plot(np.cumsum(ucb_rewards) / np.arange(1, len(ucb_rewards) + 1), label='UCB')
    plt.plot(np.cumsum(ts_rewards) / np.arange(1, len(ts_rewards) + 1), label='Thompson Sampling')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Multi-Armed Bandit: Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot percentage of optimal actions
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(eg_optimal) / np.arange(1, len(eg_optimal) + 1), label='Epsilon-Greedy')
    plt.plot(np.cumsum(boltz_optimal) / np.arange(1, len(boltz_optimal) + 1), label='Boltzmann')
    plt.plot(np.cumsum(ucb_optimal) / np.arange(1, len(ucb_optimal) + 1), label='UCB')
    plt.plot(np.cumsum(ts_optimal) / np.arange(1, len(ts_optimal) + 1), label='Thompson Sampling')
    plt.xlabel('Steps')
    plt.ylabel('Percentage of Optimal Actions')
    plt.title('Multi-Armed Bandit: Percentage of Optimal Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandit_results.png')
    plt.close()
    
    print("Multi-armed bandit results saved as PNG file.")
    
    # Gridworld experiments
    print("\nRunning gridworld experiments...")
    
    # Epsilon-greedy
    print("Epsilon-Greedy:")
    eg_rewards = run_gridworld_experiment('epsilon_greedy')
    
    # Boltzmann exploration
    print("\nBoltzmann Exploration:")
    boltz_rewards = run_gridworld_experiment('boltzmann')
    
    # UCB
    print("\nUCB:")
    ucb_rewards = run_gridworld_experiment('ucb')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(eg_rewards, np.ones(10) / 10, mode='valid'), label='Epsilon-Greedy')
    plt.plot(np.convolve(boltz_rewards, np.ones(10) / 10, mode='valid'), label='Boltzmann')
    plt.plot(np.convolve(ucb_rewards, np.ones(10) / 10, mode='valid'), label='UCB')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Moving Average)')
    plt.title('Gridworld: Exploration Strategies Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gridworld_results.png')
    plt.close()
    
    print("Gridworld results saved as PNG file.")

if __name__ == "__main__":
    main() 