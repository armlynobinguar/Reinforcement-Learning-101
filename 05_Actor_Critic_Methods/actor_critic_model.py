import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.feature_layer(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action), state_value 