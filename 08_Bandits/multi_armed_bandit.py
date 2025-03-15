"""
Implementation of the Multi-Armed Bandit problem.
"""

import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    """
    Multi-Armed Bandit environment.
    Each arm returns rewards from a stationary probability distribution.
    """
    
    def __init__(self, n_arms, mean_rewards=None, std_dev=1.0):
        """
        Initialize a multi-armed bandit environment.
        
        Args:
            n_arms: Number of arms (actions)
            mean_rewards: Mean rewards for each arm (if None, randomly generated)
            std_dev: Standard deviation of reward distributions
        """
        self.n_arms = n_arms
        self.std_dev = std_dev
        
        if mean_rewards is None:
            # Generate random mean rewards
            self.mean_rewards = np.random.normal(0, 1, n_arms)
        else:
            self.mean_rewards = np.array(mean_rewards)
        
        # Track the optimal arm
        self.optimal_arm = np.argmax(self.mean_rewards)
    
    def pull(self, arm):
        """
        Pull an arm and get a reward.
        
        Args:
            arm: The arm to pull
            
        Returns:
            reward: The reward received
        """
        return np.random.normal(self.mean_rewards[arm], self.std_dev)
    
    def get_optimal_reward(self):
        """
        Get the expected reward of the optimal arm.
        
        Returns:
            optimal_reward: Expected reward of the optimal arm
        """
        return self.mean_rewards[self.optimal_arm]

def run_bandit_experiment(agents, env, n_steps=1000, n_runs=100):
    """
    Run a multi-armed bandit experiment with multiple agents.
    
    Args:
        agents: List of agents
        env: Bandit environment
        n_steps: Number of steps per run
        n_runs: Number of runs
        
    Returns:
        rewards: Average rewards over runs for each agent
        optimal_actions: Percentage of optimal actions for each agent
    """
    n_agents = len(agents)
    
    # Initialize results arrays
    rewards = np.zeros((n_agents, n_steps))
    optimal_actions = np.zeros((n_agents, n_steps))
    
    for run in range(n_runs):
        # Reset environment for each run
        env = BanditEnvironment(env.n_arms)
        
        # Reset agents
        for agent in agents:
            agent.q_values = np.zeros(env.n_arms)
            agent.counts = np.zeros(env.n_arms)
            if hasattr(agent, 'step'):
                agent.step = 0
            if hasattr(agent, 't'):
                agent.t = 0
        
        # Run steps
        for step in range(n_steps):
            for i, agent in enumerate(agents):
                # Select action
                action = agent.select_action()
                
                # Get reward
                reward = env.pull(action)
                
                # Update agent
                agent.update(action, reward)
                
                # Track results
                rewards[i, step] += reward / n_runs
                optimal_actions[i, step] += (action == env.optimal_arm) / n_runs
    
    return rewards, optimal_actions

def plot_results(rewards, optimal_actions, agent_names):
    """
    Plot the results of a bandit experiment.
    
    Args:
        rewards: Average rewards for each agent
        optimal_actions: Percentage of optimal actions for each agent
        agent_names: Names of the agents
    """
    n_steps = rewards.shape[1]
    
    plt.figure(figsize=(12, 5))
    
    # Plot average rewards
    plt.subplot(1, 2, 1)
    for i, name in enumerate(agent_names):
        plt.plot(rewards[i], label=name)
    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()
    
    # Plot percentage of optimal actions
    plt.subplot(1, 2, 2)
    for i, name in enumerate(agent_names):
        plt.plot(optimal_actions[i], label=name)
    
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.title('Percentage of Optimal Actions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('bandit_results.png')
    plt.show() 