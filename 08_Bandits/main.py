"""
Main script to demonstrate multi-armed bandit algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import BanditEnvironment, run_bandit_experiment, plot_results
import sys
import os

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exploration_strategies.epsilon_greedy import EpsilonGreedy, DecayingEpsilonGreedy
from exploration_strategies.ucb import UCB

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create bandit environment
    n_arms = 10
    env = BanditEnvironment(n_arms)
    
    # Print true mean rewards
    print("True mean rewards:")
    for i, mean in enumerate(env.mean_rewards):
        print(f"Arm {i}: {mean:.4f}")
    print(f"Optimal arm: {env.optimal_arm}")
    
    # Create agents
    agents = [
        EpsilonGreedy(n_arms, epsilon=0.1),
        DecayingEpsilonGreedy(n_arms, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01),
        UCB(n_arms, c=2.0)
    ]
    
    agent_names = [
        "ε-greedy (ε=0.1)",
        "Decaying ε-greedy",
        "UCB"
    ]
    
    # Run experiment
    print("\nRunning bandit experiment...")
    rewards, optimal_actions = run_bandit_experiment(agents, env, n_steps=1000, n_runs=100)
    
    # Plot results
    plot_results(rewards, optimal_actions, agent_names)
    
    # Print final performance
    print("\nFinal performance:")
    for i, name in enumerate(agent_names):
        print(f"{name}:")
        print(f"  Average reward: {rewards[i, -1]:.4f}")
        print(f"  % Optimal action: {optimal_actions[i, -1]*100:.2f}%")

if __name__ == "__main__":
    main() 