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
    Discretize continuous state into bins.
    
    Args:
        state: Continuous state
        bins: List of arrays with bin edges for each dimension
        
    Returns:
        index: Discretized state index
    """
    # Get the bin indices for each dimension
    indices = []
    for i, (s, b) in enumerate(zip(state, bins)):
        # Clip the state to be within the bin range
        clipped_s = np.clip(s, b[0], b[-1])
        # Find the bin index (ensure it's within bounds)
        index = min(np.digitize(clipped_s, b) - 1, len(b) - 2)
        indices.append(index)
    
    # Calculate the flat index
    flat_index = 0
    dims = [len(b) - 1 for b in bins]
    for i, idx in enumerate(indices):
        # Calculate the stride for this dimension
        stride = 1
        for j in range(i+1, len(dims)):
            stride *= dims[j]
        flat_index += idx * stride
    
    # Double-check that the index is within bounds
    n_states = np.prod(dims)
    if flat_index >= n_states:
        print(f"Warning: Calculated index {flat_index} exceeds n_states {n_states}")
        flat_index = n_states - 1
    
    return flat_index

def run_model_based_experiment(env, bins, n_states, episodes=100):
    """
    Run model-based reinforcement learning experiment.
    
    Args:
        env: Environment
        bins: Bins for state discretization
        n_states: Number of discrete states
        episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
        model: Trained model
    """
    # Get action space size
    n_actions = env.action_space.n
    
    # Initialize model
    model = TabularModel(n_states, n_actions)
    
    # Initialize Q-values
    Q = np.zeros((n_states, n_actions))
    
    # Initialize rewards list
    rewards = []
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # Handle gym v26 reset format
            state = state[0]
        state_idx = discretize_state(state, bins)
        
        # Initialize total reward
        total_reward = 0
        
        # Run episode
        done = False
        while not done:
            # Select action using epsilon-greedy policy
            if np.random.random() < 0.1:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_idx])
            
            # Take action
            try:
                # Try new gym API (v26+)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Fall back to old gym API
                next_state, reward, done, info = env.step(action)
            
            next_state_idx = discretize_state(next_state, bins)
            
            # Update model
            model.update(state_idx, action, next_state_idx, reward)
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
            state_idx = next_state_idx
        
        # Store total reward
        rewards.append(total_reward)
        
        # Plan (update Q-values using the model)
        for _ in range(50):  # Number of planning steps
            # Sample random state-action pair
            s = np.random.randint(n_states)
            a = np.random.randint(n_actions)
            
            # Sample transition from model
            next_s, r = model.sample_transition(s, a)
            
            # Update Q-value
            Q[s, a] = Q[s, a] + 0.1 * (r + 0.99 * np.max(Q[next_s]) - Q[s, a])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward}")
    
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

def run_q_learning_experiment(env, bins, n_states, episodes=100):
    """
    Run Q-learning experiment.
    
    Args:
        env: Environment
        bins: Bins for state discretization
        n_states: Number of discrete states
        episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
    """
    # Get action space size
    n_actions = env.action_space.n
    
    # Initialize Q-values
    Q = np.zeros((n_states, n_actions))
    
    # Initialize rewards list
    rewards = []
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):  # Handle gym v26 reset format
            state = state[0]
        state_idx = discretize_state(state, bins)
        
        # Initialize total reward
        total_reward = 0
        
        # Run episode
        done = False
        while not done:
            # Select action using epsilon-greedy policy
            if np.random.random() < 0.1:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_idx])
            
            # Take action
            try:
                # Try new gym API (v26+)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                # Fall back to old gym API
                next_state, reward, done, info = env.step(action)
            
            next_state_idx = discretize_state(next_state, bins)
            
            # Update Q-value
            Q[state_idx, action] = Q[state_idx, action] + 0.1 * (reward + 0.99 * np.max(Q[next_state_idx]) - Q[state_idx, action])
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
            state_idx = next_state_idx
        
        # Store total reward
        rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward}")
    
    return rewards

def main():
    """
    Main function to run experiments.
    """
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Define bins for state discretization
    # CartPole: cart position, cart velocity, pole angle, pole velocity
    cart_pos_bins = np.linspace(-2.4, 2.4, 3)
    cart_vel_bins = np.linspace(-3, 3, 3)
    pole_ang_bins = np.linspace(-0.2, 0.2, 3)
    pole_vel_bins = np.linspace(-3, 3, 3)
    
    bins = [cart_pos_bins, cart_vel_bins, pole_ang_bins, pole_vel_bins]
    
    # Calculate total number of discrete states
    n_states = 1
    for b in bins:
        n_states *= (len(b) - 1)
    
    print(f"Total number of discrete states: {n_states}")
    
    # Run experiments
    print("Running model-based experiment...")
    model_rewards, model = run_model_based_experiment(env, bins, n_states=n_states, episodes=100)
    
    print("Running model-free experiment...")
    q_rewards = run_q_learning_experiment(env, bins, n_states=n_states, episodes=100)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(model_rewards, label='Model-Based')
    plt.plot(q_rewards, label='Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Model-Based vs Model-Free RL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_based_vs_model_free.png')
    plt.close()
    
    print("Results saved as PNG file.")

if __name__ == "__main__":
    main() 