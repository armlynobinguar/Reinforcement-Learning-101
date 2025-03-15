"""
Implementation of epsilon-greedy exploration strategy.
"""

import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy.
    With probability epsilon, take a random action.
    With probability 1-epsilon, take the best action.
    """
    
    def __init__(self, n_arms, epsilon=0.1, initial_values=0.0):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            n_arms: Number of arms (actions)
            epsilon: Exploration probability
            initial_values: Initial Q-values
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.ones(n_arms) * initial_values
        self.counts = np.zeros(n_arms)
    
    def select_action(self):
        """
        Select an action using epsilon-greedy.
        
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best action
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        """
        Update Q-values based on observed reward.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

class DecayingEpsilonGreedy(EpsilonGreedy):
    """
    Epsilon-greedy with decaying exploration rate.
    """
    
    def __init__(self, n_arms, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01, initial_values=0.0):
        """
        Initialize decaying epsilon-greedy strategy.
        
        Args:
            n_arms: Number of arms (actions)
            initial_epsilon: Initial exploration probability
            min_epsilon: Minimum exploration probability
            decay_rate: Rate at which epsilon decays
            initial_values: Initial Q-values
        """
        super().__init__(n_arms, initial_epsilon, initial_values)
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.step = 0
    
    def select_action(self):
        """
        Select an action using decaying epsilon-greedy.
        
        Returns:
            action: Selected action
        """
        # Update epsilon
        self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.step)
        self.step += 1
        
        return super().select_action() 