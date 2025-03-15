"""
Main script to train and evaluate the PPO agent.
"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gym_wrapper import create_env, env_reset, env_step

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Train agent
    rewards = agent.train(env, episodes=100, steps_per_epoch=4000)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Progress')
    plt.savefig('ppo_rewards.png')
    plt.show()
    
    # Save the trained model
    torch.save(agent.model.state_dict(), 'ppo_model.pth')
    
    # Evaluate agent
    evaluate_agent(env, agent)

def evaluate_agent(env, agent, episodes=10):
    total_rewards = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Use trained policy for evaluation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs, _ = agent.model(state_tensor)
            action = torch.argmax(action_probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        total_rewards += episode_reward
    
    avg_reward = total_rewards / episodes
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    main() 