"""
Introduction to Reinforcement Learning concepts.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_agent_environment_interaction():
    """
    Create a diagram showing the agent-environment interaction loop.
    """
    plt.figure(figsize=(10, 6))
    
    # Create boxes for agent and environment
    agent_box = plt.Rectangle((0.2, 0.5), 0.2, 0.2, fill=True, color='lightblue', alpha=0.7)
    env_box = plt.Rectangle((0.6, 0.5), 0.2, 0.2, fill=True, color='lightgreen', alpha=0.7)
    
    # Add boxes to plot
    plt.gca().add_patch(agent_box)
    plt.gca().add_patch(env_box)
    
    # Add text labels
    plt.text(0.3, 0.6, 'Agent', ha='center', va='center', fontsize=12)
    plt.text(0.7, 0.6, 'Environment', ha='center', va='center', fontsize=12)
    
    # Add arrows
    plt.arrow(0.4, 0.55, 0.18, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.text(0.5, 0.57, 'Action', ha='center', va='center', fontsize=10)
    
    plt.arrow(0.6, 0.65, -0.18, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.text(0.5, 0.67, 'State', ha='center', va='center', fontsize=10)
    
    plt.arrow(0.6, 0.55, -0.18, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.text(0.5, 0.52, 'Reward', ha='center', va='center', fontsize=10)
    
    # Set axis properties
    plt.xlim(0, 1)
    plt.ylim(0.3, 0.8)
    plt.axis('off')
    
    plt.title('Agent-Environment Interaction in Reinforcement Learning')
    plt.tight_layout()
    plt.savefig('agent_environment_interaction.png')
    plt.close()

def plot_reward_discounting():
    """
    Visualize the concept of reward discounting.
    """
    plt.figure(figsize=(10, 6))
    
    # Define time steps and discount factors
    time_steps = np.arange(0, 10)
    gammas = [0.9, 0.8, 0.5]
    
    # Plot discounted values for different gamma values
    for gamma in gammas:
        discounted_values = gamma ** time_steps
        plt.plot(time_steps, discounted_values, 'o-', label=f'γ = {gamma}')
    
    plt.xlabel('Time Steps into the Future')
    plt.ylabel('Discount Factor (γ^t)')
    plt.title('Effect of Discount Factor on Future Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_discounting.png')
    plt.close()

def main():
    """
    Generate visualizations for RL introduction.
    """
    print("Generating RL introduction visualizations...")
    plot_agent_environment_interaction()
    plot_reward_discounting()
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main() 