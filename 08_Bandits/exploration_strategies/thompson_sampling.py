"""
Implementation of Thompson Sampling exploration strategy for multi-armed bandits.
"""

import numpy as np

class ThompsonSampling:
    """
    Thompson Sampling exploration strategy.
    """
    
    def __init__(self, n_arms):
        """
        Initialize Thompson Sampling.
        
        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Success count
        self.beta = np.ones(n_arms)   # Failure count
        self.name = "Thompson Sampling"
    
    def select_action(self):
        """
        Select action using Thompson Sampling.
        
        Returns:
            action: Selected action
        """
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)
        
        # Select arm with highest sample
        return np.argmax(samples)
    
    def update(self, action, reward):
        """
        Update Beta distribution parameters.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        # Normalize reward to [0, 1] range for Beta distribution
        normalized_reward = max(0, min(1, (reward + 1) / 2))
        
        self.alpha[action] += normalized_reward
        self.beta[action] += (1 - normalized_reward) 