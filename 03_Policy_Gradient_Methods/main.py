import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from policy_network import PolicyNetwork
from reinforce import reinforce

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize policy and optimizer
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    # Train the agent
    rewards = reinforce(env, policy, optimizer, num_episodes=500)
    
    # Plot the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training Progress')
    plt.savefig('reinforce_rewards.png')
    plt.show()
    
    # Save the trained model
    torch.save(policy.state_dict(), 'reinforce_policy.pth')
    
    # Evaluate the trained policy
    evaluate_policy(env, policy)

def evaluate_policy(env, policy, num_episodes=10):
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = policy.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    main() 