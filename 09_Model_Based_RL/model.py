"""
Implementation of a tabular model for model-based reinforcement learning.
"""

import numpy as np

class TabularModel:
    """
    Tabular model for discrete state and action spaces.
    """
    
    def __init__(self, n_states, n_actions):
        """
        Initialize tabular model.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
        """
        # Convert to integers to avoid numpy array dimension errors
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        
        # Initialize transition counts
        self.transition_counts = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Initialize reward sum
        self.reward_sum = np.zeros((self.n_states, self.n_actions))
        
        # Initialize visit counts
        self.visit_counts = np.zeros((self.n_states, self.n_actions))
        
        # For compatibility with value_iteration function
        self.transition_model = None
    
    def update(self, state, action, next_state, reward):
        """
        Update the model with a new transition.
        
        Args:
            state: Current state index
            action: Action taken
            next_state: Next state index
            reward: Reward received
        """
        # Check if indices are within bounds
        if state >= self.n_states or next_state >= self.n_states:
            print(f"Warning: State indices out of bounds. state={state}, next_state={next_state}, n_states={self.n_states}")
            state = min(state, self.n_states - 1)
            next_state = min(next_state, self.n_states - 1)
        
        # Update transition counts
        self.transition_counts[state, action, next_state] += 1
        
        # Update reward sum
        self.reward_sum[state, action] += reward
        
        # Update visit counts
        self.visit_counts[state, action] += 1
        
        # Update transition model for value iteration
        self._update_transition_model()
    
    def _update_transition_model(self):
        """
        Update the transition model based on counts.
        """
        # Initialize transition model
        self.transition_model = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Calculate transition probabilities
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total = np.sum(self.transition_counts[s, a])
                if total > 0:
                    self.transition_model[s, a] = self.transition_counts[s, a] / total
                else:
                    # If no transitions observed, assume uniform distribution
                    self.transition_model[s, a] = np.ones(self.n_states) / self.n_states
    
    def get_transition_probs(self, state, action):
        """
        Get transition probabilities for a state-action pair.
        
        Args:
            state: State index
            action: Action
            
        Returns:
            probs: Transition probabilities
        """
        if self.transition_model is not None:
            return self.transition_model[state, action]
        
        counts = self.transition_counts[state, action]
        total = np.sum(counts)
        
        if total > 0:
            return counts / total
        else:
            # If no transitions observed, assume uniform distribution
            return np.ones(self.n_states) / self.n_states
    
    def get_expected_reward(self, state, action):
        """
        Get expected reward for a state-action pair.
        
        Args:
            state: State index
            action: Action
            
        Returns:
            reward: Expected reward
        """
        if self.visit_counts[state, action] > 0:
            return self.reward_sum[state, action] / self.visit_counts[state, action]
        else:
            # If no transitions observed, assume zero reward
            return 0.0
    
    def sample_transition(self, state, action):
        """
        Sample a transition from the model.
        
        Args:
            state: Current state index
            action: Action
            
        Returns:
            next_state: Sampled next state
            reward: Expected reward
        """
        probs = self.get_transition_probs(state, action)
        next_state = np.random.choice(self.n_states, p=probs)
        reward = self.get_expected_reward(state, action)
        
        return next_state, reward

def value_iteration(model, reward_function, gamma=0.99, theta=1e-6):
    """
    Value iteration algorithm.
    
    Args:
        model: Transition model
        reward_function: Reward function
        gamma: Discount factor
        theta: Convergence threshold
        
    Returns:
        V: Value function
        policy: Optimal policy
    """
    # Initialize value function
    V = np.zeros(model.n_states)
    
    # Initialize policy
    policy = np.zeros(model.n_states, dtype=int)
    
    # Value iteration
    while True:
        delta = 0
        
        for s in range(model.n_states):
            v = V[s]
            
            # Compute value for each action
            action_values = np.zeros(model.n_actions)
            
            for a in range(model.n_actions):
                # Compute expected value
                expected_value = 0
                
                for s_next in range(model.n_states):
                    # Make sure transition_model is available
                    if model.transition_model is None:
                        model._update_transition_model()
                    
                    p = model.transition_model[s, a, s_next]
                    r = reward_function(s, a, s_next)
                    expected_value += p * (r + gamma * V[s_next])
                
                action_values[a] = expected_value
            
            # Update value function
            best_action = np.argmax(action_values)
            V[s] = action_values[best_action]
            policy[s] = best_action
            
            # Update delta
            delta = max(delta, abs(v - V[s]))
        
        # Check convergence
        if delta < theta:
            break
    
    return V, policy 