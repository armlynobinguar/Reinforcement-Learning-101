"""
Implementation of Boltzmann exploration strategy for multi-armed bandits.
"""

import numpy as np

class BoltzmannExploration:
    """
    Boltzmann exploration strategy.
    """
    
    def __init__(self, n_arms, temperature=1.0):
        """
        Initialize Boltzmann exploration.
        
        Args:
            n_arms: Number of arms
            temperature: Temperature parameter
        """
        self.n_arms = n_arms
        self.temperature = temperature
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.name = f"Boltzmann (τ={temperature})"
    
    def select_action(self):
        """
        Select action using Boltzmann exploration.
        
        Returns:
            action: Selected action
        """
        # Compute probabilities using softmax
        exp_values = np.exp(self.q_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        
        # Sample action
        return np.random.choice(self.n_arms, p=probs)
    
    def update(self, action, reward):
        """
        Update Q-values.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action] 