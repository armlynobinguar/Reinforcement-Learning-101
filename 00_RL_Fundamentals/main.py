"""
Main script to demonstrate RL fundamentals.
"""

import matplotlib.pyplot as plt
import numpy as np
from mdp import GridWorldMDP
from value_functions import policy_iteration, value_iteration

def visualize_value_function(mdp, V, title):
    """Visualize a value function on a grid."""
    values = np.zeros((mdp.height, mdp.width))
    
    for state, value in V.items():
        x, y = state
        values[mdp.height - 1 - y, x] = value  # Flip y-axis for visualization
    
    plt.figure(figsize=(8, 6))
    plt.imshow(values, cmap='viridis')
    plt.colorbar(label='Value')
    
    # Add grid lines
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks(np.arange(-.5, mdp.width, 1), [])
    plt.yticks(np.arange(-.5, mdp.height, 1), [])
    
    # Add obstacles and terminals
    for x, y in mdp.obstacles:
        plt.text(x, mdp.height - 1 - y, 'X', ha='center', va='center', color='white', fontsize=20)
    
    for x, y in mdp.terminals:
        plt.text(x, mdp.height - 1 - y, 'T', ha='center', va='center', color='white', fontsize=20)
    
    # Add values
    for state, value in V.items():
        x, y = state
        plt.text(x, mdp.height - 1 - y, f'{value:.2f}', ha='center', va='center', color='white', fontsize=10)
    
    plt.title(title)
    plt.tight_layout()
    return plt

def visualize_policy(mdp, policy, title):
    """Visualize a policy on a grid."""
    arrows = ['↑', '→', '↓', '←']
    
    plt.figure(figsize=(8, 6))
    plt.imshow(np.zeros((mdp.height, mdp.width)), cmap='viridis')
    
    # Add grid lines
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks(np.arange(-.5, mdp.width, 1), [])
    plt.yticks(np.arange(-.5, mdp.height, 1), [])
    
    # Add obstacles and terminals
    for x, y in mdp.obstacles:
        plt.text(x, mdp.height - 1 - y, 'X', ha='center', va='center', color='white', fontsize=20)
    
    for x, y in mdp.terminals:
        plt.text(x, mdp.height - 1 - y, 'T', ha='center', va='center', color='white', fontsize=20)
    
    # Add policy arrows
    for state in mdp.states:
        if state in mdp.terminals:
            continue
            
        x, y = state
        action = policy(state)
        plt.text(x, mdp.height - 1 - y, arrows[action], ha='center', va='center', color='white', fontsize=20)
    
    plt.title(title)
    plt.tight_layout()
    return plt

def main():
    # Create a simple grid world
    width, height = 5, 5
    obstacles = [(1, 1), (2, 1), (3, 1)]
    terminals = [(4, 4)]
    rewards = {(4, 4): 1.0}  # Goal state with positive reward
    
    mdp = GridWorldMDP(width, height, obstacles, terminals, rewards, gamma=0.9)
    
    # Solve using policy iteration
    pi_policy, pi_V = policy_iteration(mdp)
    
    # Solve using value iteration
    vi_policy, vi_V = value_iteration(mdp)
    
    # Visualize results
    visualize_value_function(mdp, pi_V, "Value Function (Policy Iteration)")
    plt.savefig("pi_value_function.png")
    
    visualize_policy(mdp, pi_policy, "Optimal Policy (Policy Iteration)")
    plt.savefig("pi_policy.png")
    
    visualize_value_function(mdp, vi_V, "Value Function (Value Iteration)")
    plt.savefig("vi_value_function.png")
    
    visualize_policy(mdp, vi_policy, "Optimal Policy (Value Iteration)")
    plt.savefig("vi_policy.png")
    
    plt.show()

if __name__ == "__main__":
    main() 