"""
Implementation of Upper Confidence Bound (UCB) exploration strategies for multi-armed bandits.
"""

import numpy as np

class UCB:
    """
    Upper Confidence Bound (UCB) exploration strategy.
    """
    
    def __init__(self, n_arms, c=2.0):
        """
        Initialize UCB.
        
        Args:
            n_arms: Number of arms
            c: Exploration parameter
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0
        self.name = f"UCB (c={c})"
    
    def select_action(self):
        """
        Select action using UCB.
        
        Returns:
            action: Selected action
        """
        # Initialize all arms if not all have been pulled
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Compute UCB values
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.counts)
        
        # Select arm with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """
        Update Q-values and time step.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]
        self.t += 1 