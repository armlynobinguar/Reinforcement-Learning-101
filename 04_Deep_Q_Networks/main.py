import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

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
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    )
    
    # Train agent
    rewards = agent.train(env, episodes=500)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress')
    plt.savefig('dqn_rewards.png')
    plt.show()
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    
    # Evaluate agent
    evaluate_agent(env, agent)

def evaluate_agent(env, agent, episodes=10):
    total_rewards = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Use greedy policy for evaluation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            
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