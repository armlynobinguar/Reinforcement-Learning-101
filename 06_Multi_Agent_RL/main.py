"""
Main script to demonstrate multi-agent reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import create_prisoners_dilemma, create_matching_pennies
from q_learning import MultiAgentQLearning
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def visualize_policy(q_table, game_name, agent_id):
    """Visualize a policy from a Q-table."""
    policy = np.argmax(q_table, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(policy)), policy)
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.title(f'{game_name} - Agent {agent_id+1} Policy')
    plt.savefig(f'{game_name.lower().replace(" ", "_")}_agent{agent_id+1}_policy.png')

def visualize_rewards(rewards_history, game_name):
    """Visualize rewards over episodes."""
    rewards_history = np.array(rewards_history)
    
    plt.figure(figsize=(10, 6))
    for i in range(rewards_history.shape[1]):
        plt.plot(rewards_history[:, i], label=f'Agent {i+1}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{game_name} - Rewards')
    plt.legend()
    plt.savefig(f'{game_name.lower().replace(" ", "_")}_rewards.png')

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environments
    pd_env = create_prisoners_dilemma()
    mp_env = create_matching_pennies()
    
    # Train on Prisoner's Dilemma
    print("Training on Prisoner's Dilemma...")
    pd_agent = MultiAgentQLearning(
        num_states=1,  # Only one state in matrix games
        num_actions_list=[2, 2],  # Two actions for each agent: Cooperate/Defect
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    pd_rewards = pd_agent.train(pd_env, episodes=10000)
    
    # Train on Matching Pennies
    print("\nTraining on Matching Pennies...")
    mp_agent = MultiAgentQLearning(
        num_states=1,  # Only one state in matrix games
        num_actions_list=[2, 2],  # Two actions for each agent: Heads/Tails
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    mp_rewards = mp_agent.train(mp_env, episodes=10000)
    
    # Visualize results
    for i in range(2):
        visualize_policy(pd_agent.q_tables[i], "Prisoner's Dilemma", i)
        visualize_policy(mp_agent.q_tables[i], "Matching Pennies", i)
    
    visualize_rewards(pd_rewards, "Prisoner's Dilemma")
    visualize_rewards(mp_rewards, "Matching Pennies")
    
    # Print final policies
    print("\nFinal Policies:")
    print("Prisoner's Dilemma:")
    for i in range(2):
        action = np.argmax(pd_agent.q_tables[i][0])
        print(f"Agent {i+1}: {'Cooperate' if action == 0 else 'Defect'}")
    
    print("\nMatching Pennies:")
    for i in range(2):
        action = np.argmax(mp_agent.q_tables[i][0])
        print(f"Agent {i+1}: {'Heads' if action == 0 else 'Tails'}")

if __name__ == "__main__":
    main() 