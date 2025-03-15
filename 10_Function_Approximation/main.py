"""
Main script to demonstrate function approximation in reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from linear_approximation import LinearApproximator

def mountain_car_features(state, action, n_tilings=8, n_tiles=8):
    """
    Extract tile coding features for Mountain Car.
    
    Args:
        state: State (position, velocity)
        action: Action
        n_tilings: Number of tilings
        n_tiles: Number of tiles per dimension
        
    Returns:
        features: Feature vector
    """
    # Normalize state
    position, velocity = state
    position = (position + 1.2) / 1.7  # Normalize to [0, 1]
    velocity = (velocity + 0.07) / 0.14  # Normalize to [0, 1]
    
    # Initialize features
    features = np.zeros(n_tilings * n_tiles * n_tiles)
    
    for t in range(n_tilings):
        # Add offset for each tiling
        pos_offset = t * 1.0 / n_tilings
        vel_offset = t * 1.0 / n_tilings
        
        # Calculate tile indices
        pos_idx = int(np.floor(position * n_tiles + pos_offset) % n_tiles)
        vel_idx = int(np.floor(velocity * n_tiles + vel_offset) % n_tiles)
        
        # Calculate feature index
        idx = t * n_tiles * n_tiles + pos_idx * n_tiles + vel_idx
        features[idx] = 1.0
    
    return features

def cart_pole_features(state, action, n_bins=3):
    """
    Extract simple binned features for CartPole.
    
    Args:
        state: State (cart position, cart velocity, pole angle, pole velocity)
        action: Action
        n_bins: Number of bins per dimension
        
    Returns:
        features: Feature vector
    """
    # Define bins for each dimension
    cart_pos_bins = np.linspace(-2.4, 2.4, n_bins+1)[1:-1]
    cart_vel_bins = np.linspace(-3, 3, n_bins+1)[1:-1]
    pole_ang_bins = np.linspace(-0.2, 0.2, n_bins+1)[1:-1]
    pole_vel_bins = np.linspace(-3, 3, n_bins+1)[1:-1]
    
    # Digitize state
    cart_pos_idx = np.digitize(state[0], cart_pos_bins)
    cart_vel_idx = np.digitize(state[1], cart_vel_bins)
    pole_ang_idx = np.digitize(state[2], pole_ang_bins)
    pole_vel_idx = np.digitize(state[3], pole_vel_bins)
    
    # Calculate feature index
    idx = cart_pos_idx * n_bins**3 + cart_vel_idx * n_bins**2 + pole_ang_idx * n_bins + pole_vel_idx
    
    # One-hot encoding
    features = np.zeros(n_bins**4)
    features[idx] = 1.0
    
    return features

def run_mountain_car_experiment(episodes=500):
    """
    Run an experiment on Mountain Car using linear function approximation.
    
    Args:
        episodes: Number of episodes
        
    Returns:
        rewards: Episode rewards
        agent: Trained agent
    """
    # Create environment
    env = gym.make('MountainCar-v0')
    
    # Define feature extractor
    n_tilings = 8
    n_tiles = 8
    n_features = n_tilings * n_tiles * n_tiles
    
    def feature_extractor(state, action):
        return mountain_car_features(state, action, n_tilings, n_tiles)
    
    # Initialize agent
    agent = LinearApproximator(
        feature_extractor=feature_extractor,
        n_features=n_features,
        n_actions=env.action_space.n,
        learning_rate=0.1 / n_tilings,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Initialize results
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    return rewards, agent

def run_cart_pole_experiment(episodes=500):
    """
    Run an experiment on CartPole using linear function approximation.
    
    Args:
        episodes: Number of episodes
        
    Returns:
        rewards: Episode rewards
        agent: Trained agent
    """
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Define feature extractor
    n_bins = 3
    n_features = n_bins**4
    
    def feature_extractor(state, action):
        return cart_pole_features(state, action, n_bins)
    
    # Initialize agent
    agent = LinearApproximator(
        feature_extractor=feature_extractor,
        n_features=n_features,
        n_actions=env.action_space.n,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.1
    )
    
    # Initialize results
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    return rewards, agent

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run Mountain Car experiment
    print("Running Mountain Car experiment...")
    mc_rewards, mc_agent = run_mountain_car_experiment(episodes=200)
    
    # Run CartPole experiment
    print("\nRunning CartPole experiment...")
    cp_rewards, cp_agent = run_cart_pole_experiment(episodes=200)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mc_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Mountain Car')
    
    plt.subplot(1, 2, 2)
    plt.plot(cp_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole')
    
    plt.tight_layout()
    plt.savefig('function_approximation_results.png')
    plt.show()
    
    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    
    # Mountain Car
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = mc_agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"Mountain Car final reward: {total_reward}")
    
    # CartPole
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = cp_agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"CartPole final reward: {total_reward}")

if __name__ == "__main__":
    main() 