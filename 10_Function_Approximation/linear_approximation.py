"""
Implementation of linear function approximation for reinforcement learning.
"""

import numpy as np

class LinearApproximator:
    """
    Linear function approximation for Q-values.
    Q(s,a) = w^T * phi(s,a)
    """
    
    def __init__(self, feature_extractor, n_features, n_actions, learning_rate=0.01, gamma=0.99, epsilon=0.1):
        """
        Initialize linear function approximator.
        
        Args:
            feature_extractor: Function that extracts features from state and action
            n_features: Number of features
            n_actions: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.feature_extractor = feature_extractor
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize weights
        self.weights = np.zeros((n_actions, n_features))
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: State
            action: Action
            
        Returns:
            q_value: Q-value
        """
        features = self.feature_extractor(state, action)
        return np.dot(self.weights[action], features)
    
    def get_q_values(self, state):
        """
        Get Q-values for all actions in a state.
        
        Args:
            state: State
            
        Returns:
            q_values: Q-values for all actions
        """
        q_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            q_values[action] = self.get_q_value(state, action)
        return q_values
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update weights based on observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get next Q-value
        if done:
            next_q = 0
        else:
            next_q = np.max(self.get_q_values(next_state))
        
        # Calculate TD target and error
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        # Update weights
        features = self.feature_extractor(state, action)
        self.weights[action] += self.lr * td_error * features 