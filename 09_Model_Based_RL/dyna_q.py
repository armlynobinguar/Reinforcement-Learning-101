"""
Implementation of Dyna-Q algorithm for model-based reinforcement learning.
"""

import numpy as np

class DynaQ:
    """
    Dyna-Q algorithm that combines Q-learning with planning using a learned model.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, 
                 epsilon=0.1, planning_steps=5):
        """
        Initialize Dyna-Q agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            planning_steps: Number of planning steps per real step
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Initialize model
        self.model = {}  # (state, action) -> (next_state, reward)
        self.visited_sa_pairs = set()  # Set of visited (state, action) pairs
    
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
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, next_state, reward):
        """
        Update Q-values and model based on observed transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
        """
        # Direct RL update (Q-learning)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
        # Update model
        self.model[(state, action)] = (next_state, reward)
        self.visited_sa_pairs.add((state, action))
        
        # Planning
        for _ in range(self.planning_steps):
            # Sample a previously observed (state, action) pair
            if not self.visited_sa_pairs:
                break
                
            sa_pair = list(self.visited_sa_pairs)[np.random.randint(len(self.visited_sa_pairs))]
            s, a = sa_pair
            
            # Get predicted next state and reward
            s_next, r = self.model[sa_pair]
            
            # Q-learning update using the model
            best_next_action = np.argmax(self.q_table[s_next])
            td_target = r + self.gamma * self.q_table[s_next, best_next_action]
            td_error = td_target - self.q_table[s, a]
            self.q_table[s, a] += self.lr * td_error 