"""
Main script to demonstrate model-based reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from model import TabularModel, value_iteration
from dyna_q import DynaQ

def discretize_state(state, bins):
    """
    Discretize a continuous state into bins.
    
    Args:
        state: Continuous state
        bins: Number of bins for each dimension
        
    Returns:
        index: Discretized state index
    """
    state_bins = []
    for i, (s, b) in enumerate(zip(state, bins)):
        if i == 0:  # Cart position
            state_bins.append(np.digitize(s, np.linspace(-2.4, 2.4, b)))
        elif i == 1:  # Cart velocity
            state_bins.append(np.digitize(s, np.linspace(-3, 3, b)))
        elif i == 2:  # Pole angle
            state_bins.append(np.digitize(s, np.linspace(-0.2, 0.2, b)))
        elif i == 3:  # Pole angular velocity
            state_bins.append(np.digitize(s, np.linspace(-3, 3, b)))
    
    # Convert to single index
    index = 0
    for i, b in enumerate(state_bins):
        index += b * np.prod(bins[:i])
    
    return int(index)

def run_model_based_experiment(env, bins, episodes=1000):
    """
    Run a model-based RL experiment.
    
    Args:
        env: Gym environment
        bins: Number of bins for discretization
        episodes: Number of episodes
        
    Returns:
        rewards: Episode rewards
        model: Learned model
    """
    # Calculate number of states and actions
    n_states = np.prod(bins)
    n_actions = env.action_space.n
    
    # Initialize model
    model = TabularModel(n_states, n_actions)
    
    # Initialize results
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state_idx = discretize_state(state, bins)
        total_reward = 0
        done = False
        
        while not done:
            # Random policy for data collection
            action = env.action_space.sample()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = discretize_state(next_state, bins)
            
            # Update model
            model.update(state_idx, action, next_state_idx, reward)
            
            # Update state and reward
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    return rewards, model

def run_dyna_q_experiment(env, bins, episodes=1000, planning_steps=5):
    """
    Run a Dyna-Q experiment.
    
    Args:
        env: Gym environment
        bins: Number of bins for discretization
        episodes: Number of episodes
        planning_steps: Number of planning steps per real step
        
    Returns:
        rewards: Episode rewards
        agent: Dyna-Q agent
    """
    # Calculate number of states and actions
    n_states = np.prod(bins)
    n_actions = env.action_space.n
    
    # Initialize agent
    agent = DynaQ(n_states, n_actions, planning_steps=planning_steps)
    
    # Initialize results
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state_idx = discretize_state(state, bins)
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state_idx)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = discretize_state(next_state, bins)
            
            # Update agent
            agent.update(state_idx, action, next_state_idx, reward)
            
            # Update state and reward
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    return rewards, agent

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Define discretization bins
    bins = (3, 3, 6, 3)  # (cart position, cart velocity, pole angle, pole angular velocity)
    
    # Run model-based experiment
    print("Running model-based experiment...")
    model_rewards, model = run_model_based_experiment(env, bins, episodes=100)
    
    # Compute optimal policy using value iteration
    print("Computing optimal policy...")
    V, policy = value_iteration(model, gamma=0.99)
    
    # Run Dyna-Q experiment
    print("\nRunning Dyna-Q experiment...")
    dyna_rewards, dyna_agent = run_dyna_q_experiment(env, bins, episodes=100, planning_steps=5)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(model_rewards, label='Random Policy (Model Building)')
    plt.plot(dyna_rewards, label='Dyna-Q')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Model-Based RL Performance')
    plt.legend()
    plt.savefig('model_based_results.png')
    plt.show()
    
    # Evaluate optimal policy
    print("\nEvaluating optimal policy...")
    eval_rewards = []
    
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_idx = discretize_state(state, bins)
            action = policy[state_idx]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
        
        eval_rewards.append(total_reward)
        print(f"Episode {episode}, Reward: {total_reward}")
    
    print(f"Average reward: {np.mean(eval_rewards):.2f}")

if __name__ == "__main__":
    main() 