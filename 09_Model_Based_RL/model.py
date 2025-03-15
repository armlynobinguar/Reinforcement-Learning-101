"""
Implementation of model-based reinforcement learning algorithms.
"""

import numpy as np

class TabularModel:
    """
    A tabular model for model-based reinforcement learning.
    Learns the transition and reward functions from experience.
    """
    
    def __init__(self, n_states, n_actions):
        """
        Initialize a tabular model.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
        """
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Initialize transition counts and model
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        self.transition_model = np.ones((n_states, n_actions, n_states)) / n_states
        
        # Initialize reward model
        self.reward_sum = np.zeros((n_states, n_actions, n_states))
        self.reward_model = np.zeros((n_states, n_actions, n_states))
    
    def update(self, state, action, next_state, reward):
        """
        Update the model based on observed transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
        """
        # Update transition counts
        self.transition_counts[state, action, next_state] += 1
        
        # Update transition model
        total_count = np.sum(self.transition_counts[state, action])
        self.transition_model[state, action] = self.transition_counts[state, action] / total_count
        
        # Update reward model
        self.reward_sum[state, action, next_state] += reward
        self.reward_model[state, action, next_state] = self.reward_sum[state, action, next_state] / self.transition_counts[state, action, next_state]
    
    def sample_transition(self, state, action):
        """
        Sample a transition from the model.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            next_state: Sampled next state
            reward: Expected reward
        """
        # Sample next state
        next_state = np.random.choice(self.n_states, p=self.transition_model[state, action])
        
        # Get expected reward
        reward = self.reward_model[state, action, next_state]
        
        return next_state, reward

def value_iteration(model, gamma=0.99, theta=1e-6):
    """
    Value iteration algorithm using a learned model.
    
    Args:
        model: The learned model
        gamma: Discount factor
        theta: Convergence threshold
        
    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    n_states = model.n_states
    n_actions = model.n_actions
    
    # Initialize value function
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        
        # Update each state
        for s in range(n_states):
            v = V[s]
            
            # Try all possible actions
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                # Calculate expected value
                expected_value = 0
                for s_next in range(n_states):
                    # Transition probability
                    p = model.transition_model[s, a, s_next]
                    # Expected reward
                    r = model.reward_model[s, a, s_next]
                    # Update expected value
                    expected_value += p * (r + gamma * V[s_next])
                
                action_values[a] = expected_value
            
            # Update value with best action
            V[s] = np.max(action_values)
            
            # Calculate delta
            delta = max(delta, abs(v - V[s]))
        
        # Check if converged
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            expected_value = 0
            for s_next in range(n_states):
                p = model.transition_model[s, a, s_next]
                r = model.reward_model[s, a, s_next]
                expected_value += p * (r + gamma * V[s_next])
            action_values[a] = expected_value
        policy[s] = np.argmax(action_values)
    
    return V, policy 