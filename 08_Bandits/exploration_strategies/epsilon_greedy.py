"""
Implementation of epsilon-greedy exploration strategies for multi-armed bandits.
"""

import numpy as np

class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy.
    """
    
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize epsilon-greedy.
        
        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.name = f"Epsilon-Greedy (ε={epsilon})"
    
    def select_action(self):
        """
        Select action using epsilon-greedy.
        
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        """
        Update Q-values.
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

class DecayingEpsilonGreedy:
    """
    Epsilon-greedy with decaying epsilon.
    """
    
    def __init__(self, n_arms, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01):
        """
        Initialize decaying epsilon-greedy.
        
        Args:
            n_arms: Number of arms
            initial_epsilon: Initial exploration probability
            min_epsilon: Minimum exploration probability
            decay_rate: Decay rate
        """
        self.n_arms = n_arms
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0
        self.name = f"Decaying Epsilon-Greedy (ε₀={initial_epsilon}, min={min_epsilon})"
    
    def select_action(self):
        """
        Select action using epsilon-greedy.
        
        Returns:
            action: Selected action
        """
        # Decay epsilon
        self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.t)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)
    
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