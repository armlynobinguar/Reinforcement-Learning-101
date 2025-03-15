import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearningAgent

def main():
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # Initialize agent
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agent
    rewards = agent.train(env, episodes=5000)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Training Progress')
    plt.savefig('q_learning_rewards.png')
    plt.show()
    
    # Print Q-table
    print("\nLearned Q-table:")
    print(agent.q_table)
    
    # Evaluate agent
    evaluate_agent(env, agent)

def evaluate_agent(env, agent, episodes=100):
    total_rewards = 0
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(agent.q_table[state])  # Greedy action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_rewards += reward
    
    avg_reward = total_rewards / episodes
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
    
    # Visualize policy
    policy = np.argmax(agent.q_table, axis=1)
    print("\nLearned Policy:")
    action_symbols = ["←", "↓", "→", "↑"]
    for i in range(4):
        row = []
        for j in range(4):
            s = i * 4 + j
            row.append(action_symbols[policy[s]])
        print("\t".join(row))

if __name__ == "__main__":
    main() 