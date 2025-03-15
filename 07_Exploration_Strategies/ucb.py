"""
Implementation of Upper Confidence Bound (UCB) exploration strategy.
"""

import numpy as np
import matplotlib.pyplot as plt

class UCB:
    """
    Upper Confidence Bound (UCB) exploration strategy.
    Selects actions based on their estimated value plus an exploration bonus.
    """
    
    def __init__(self, n_arms, c=2.0, initial_values=0.0):
        """
        Initialize UCB strategy.
        
        Args:
            n_arms: Number of arms (actions)
            c: Exploration parameter
            initial_values: Initial Q-values
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.ones(n_arms) * initial_values
        self.counts = np.zeros(n_arms)
        self.t = 0
    
    def select_action(self):
        """
        Select an action using UCB.
        
        Returns:
            action: Selected action
        """
        self.t += 1
        
        # If some arms haven't been pulled yet, pull them first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Calculate UCB values
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.counts)
        
        # Select arm with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """
        Update Q-values based on observed reward.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action] 