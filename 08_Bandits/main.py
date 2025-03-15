"""
Implementation of multi-armed bandit algorithms.

This script demonstrates and compares different exploration strategies for the
multi-armed bandit problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import exploration strategies
from exploration_strategies.epsilon_greedy import EpsilonGreedy, DecayingEpsilonGreedy
from exploration_strategies.ucb import UCB
from exploration_strategies.thompson_sampling import ThompsonSampling
from exploration_strategies.boltzmann import BoltzmannExploration

class MultiArmedBandit:
    """
    Multi-armed bandit environment.
    """
    
    def __init__(self, n_arms=10, stationary=True):
        """
        Initialize multi-armed bandit.
        
        Args:
            n_arms: Number of arms
            stationary: Whether the environment is stationary
        """
        self.n_arms = n_arms
        self.stationary = stationary
        self.true_values = np.random.normal(0, 1, n_arms)
        self.optimal_arm = np.argmax(self.true_values)
    
    def pull(self, arm):
        """
        Pull an arm and get reward.
        
        Args:
            arm: Arm to pull
            
        Returns:
            reward: Reward
        """
        # Non-stationary environment: true values change over time
        if not self.stationary:
            self.true_values += np.random.normal(0, 0.01, self.n_arms)
            self.optimal_arm = np.argmax(self.true_values)
        
        return np.random.normal(self.true_values[arm], 1)

def run_experiment(bandit, agents, n_steps=1000, n_runs=10):
    """
    Run multi-armed bandit experiment.
    
    Args:
        bandit: Multi-armed bandit environment
        agents: List of exploration strategy agents
        n_steps: Number of steps per run
        n_runs: Number of runs
        
    Returns:
        rewards: Average rewards per step for each agent
        optimal_actions: Average percentage of optimal actions for each agent
    """
    rewards = np.zeros((len(agents), n_steps))
    optimal_actions = np.zeros((len(agents), n_steps))
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        
        # Reset bandit
        bandit.true_values = np.random.normal(0, 1, bandit.n_arms)
        bandit.optimal_arm = np.argmax(bandit.true_values)
        
        # Reset agents
        for agent in agents:
            agent.q_values = np.zeros(bandit.n_arms)
            agent.counts = np.zeros(bandit.n_arms)
            if hasattr(agent, 't'):
                agent.t = 0
            if hasattr(agent, 'alpha'):
                agent.alpha = np.ones(bandit.n_arms)
                agent.beta = np.ones(bandit.n_arms)
        
        for step in range(n_steps):
            for i, agent in enumerate(agents):
                # Select action
                action = agent.select_action()
                
                # Get reward
                reward = bandit.pull(action)
                
                # Update agent
                agent.update(action, reward)
                
                # Store reward
                rewards[i, step] += reward / n_runs
                
                # Store optimal action
                optimal_actions[i, step] += (action == bandit.optimal_arm) / n_runs
    
    return rewards, optimal_actions

def plot_results(rewards, optimal_actions, agent_names):
    """
    Plot experiment results.
    
    Args:
        rewards: Average rewards per step for each agent
        optimal_actions: Average percentage of optimal actions for each agent
        agent_names: Names of agents
    """
    plt.figure(figsize=(15, 10))
    
    # Plot average reward
    plt.subplot(2, 1, 1)
    for i, name in enumerate(agent_names):
        plt.plot(np.cumsum(rewards[i]) / np.arange(1, rewards.shape[1] + 1), label=name)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Multi-Armed Bandit: Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot percentage of optimal actions
    plt.subplot(2, 1, 2)
    for i, name in enumerate(agent_names):
        plt.plot(np.cumsum(optimal_actions[i]) / np.arange(1, optimal_actions.shape[1] + 1), label=name)
    plt.xlabel('Steps')
    plt.ylabel('Percentage of Optimal Actions')
    plt.title('Multi-Armed Bandit: Percentage of Optimal Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandit_results.png')
    plt.show()

def main():
    """
    Main function to run experiments.
    """
    # Create bandit
    n_arms = 10
    bandit = MultiArmedBandit(n_arms=n_arms, stationary=True)
    
    # Create agents
    agents = [
        EpsilonGreedy(n_arms, epsilon=0.1),
        DecayingEpsilonGreedy(n_arms),
        UCB(n_arms, c=2.0),
        ThompsonSampling(n_arms),
        BoltzmannExploration(n_arms, temperature=0.5)
    ]
    
    agent_names = [agent.name for agent in agents]
    
    # Run experiment
    print("Running experiment...")
    start_time = time.time()
    rewards, optimal_actions = run_experiment(bandit, agents, n_steps=1000, n_runs=10)
    end_time = time.time()
    print(f"Experiment completed in {end_time - start_time:.2f} seconds.")
    
    # Plot results
    plot_results(rewards, optimal_actions, agent_names)

if __name__ == "__main__":
    main() 