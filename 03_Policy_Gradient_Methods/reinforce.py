import torch
import numpy as np

def reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99):
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        # Collect trajectory
        while not done:
            action, log_prob = policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        episode_rewards.append(sum(rewards))
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f'Episode {episode}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}')
    
    return episode_rewards 