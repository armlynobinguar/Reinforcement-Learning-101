"""
Main script to demonstrate Deep Q-Network (DQN) in reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from neural_approximation import DQNAgent
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gym_wrapper import create_env, env_reset, env_step

# Add imageio conditionally
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Animations will not be saved.")

def run_dqn_experiment(env_name, episodes=500, render_final=True):
    """
    Run a DQN experiment on a given environment.
    
    Args:
        env_name: Name of the gym environment
        episodes: Number of episodes
        render_final: Whether to render the final episode
        
    Returns:
        rewards: Episode rewards
        losses: Training losses
        agent: Trained agent
    """
    # Create environment
    env = gym.make(env_name)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Initialize results
    rewards = []
    losses = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update agent
            loss = agent.update(state, action, reward, next_state, done)
            episode_losses.append(loss)
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    # Render final episode if requested
    if render_final:
        env = gym.make(env_name, render_mode='rgb_array')
        state, _ = env.reset()
        total_reward = 0
        frames = []
        done = False
        
        while not done:
            # Render frame
            frames.append(env.render())
            
            # Select action (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            total_reward += reward
        
        print(f"Final episode reward: {total_reward}")
        
        # Save animation
        if render_final and IMAGEIO_AVAILABLE:
            try:
                imageio.mimsave(f'{env_name}_dqn.gif', frames, fps=30)
                print(f"Animation saved as {env_name}_dqn.gif")
            except Exception as e:
                print(f"Could not save animation: {e}")
    
    return rewards, losses, agent

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run CartPole experiment
    print("Running CartPole experiment...")
    cp_rewards, cp_losses, cp_agent = run_dqn_experiment('CartPole-v1', episodes=300)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cp_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole Rewards')
    
    plt.subplot(1, 2, 2)
    plt.plot(cp_losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('CartPole Training Loss')
    
    plt.tight_layout()
    plt.savefig('dqn_results.png')
    plt.show()

if __name__ == "__main__":
    main() 