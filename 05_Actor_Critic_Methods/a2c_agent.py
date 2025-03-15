import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actor_critic_model import ActorCritic

class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        
        # Initialize actor-critic model
        self.model = ActorCritic(state_dim, action_dim, hidden_dim)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, env, episodes=1000):
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            log_probs = []
            values = []
            rewards = []
            
            # Collect trajectory
            while not done:
                action, log_prob, value = self.model.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                state = next_state
            
            episode_rewards.append(sum(rewards))
            
            # Calculate returns and advantages
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns)
            
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Calculate advantages
            advantages = returns - torch.cat(values).detach()
            
            # Calculate losses
            actor_loss = []
            critic_loss = []
            
            for log_prob, value, R, A in zip(log_probs, values, returns, advantages):
                actor_loss.append(-log_prob * A)
                critic_loss.append(nn.MSELoss()(value, torch.tensor([R])))
            
            loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if episode % 10 == 0:
                print(f'Episode {episode}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}')
        
        return episode_rewards 