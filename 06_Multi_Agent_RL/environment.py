"""
Multi-agent reinforcement learning environments.
"""

import numpy as np

class MatrixGame:
    """
    A simple matrix game for multi-agent reinforcement learning.
    Two agents simultaneously choose actions and receive rewards based on a payoff matrix.
    """
    
    def __init__(self, payoff_matrix_1, payoff_matrix_2=None):
        """
        Initialize a matrix game.
        
        Args:
            payoff_matrix_1: Reward matrix for agent 1
            payoff_matrix_2: Reward matrix for agent 2 (if None, use -payoff_matrix_1 for zero-sum game)
        """
        self.payoff_1 = np.array(payoff_matrix_1)
        
        if payoff_matrix_2 is None:
            # Zero-sum game
            self.payoff_2 = -self.payoff_1
        else:
            self.payoff_2 = np.array(payoff_matrix_2)
        
        # Check dimensions
        assert self.payoff_1.shape == self.payoff_2.shape, "Payoff matrices must have the same shape"
        
        self.num_actions_1 = self.payoff_1.shape[0]
        self.num_actions_2 = self.payoff_1.shape[1]
        
        # State is just a placeholder for matrix games
        self.state = 0
    
    def reset(self):
        """Reset the environment."""
        self.state = 0
        return self.state
    
    def step(self, action_1, action_2):
        """
        Take a step in the environment.
        
        Args:
            action_1: Action for agent 1
            action_2: Action for agent 2
            
        Returns:
            next_state: The next state (always 0 for matrix games)
            rewards: Tuple of rewards for both agents
            done: Whether the episode is done (always True for one-shot games)
        """
        reward_1 = self.payoff_1[action_1, action_2]
        reward_2 = self.payoff_2[action_1, action_2]
        
        return self.state, (reward_1, reward_2), True

# Example games
def create_prisoners_dilemma():
    """
    Create a Prisoner's Dilemma game.
    Actions: 0 = Cooperate, 1 = Defect
    """
    # Payoff matrix for agent 1 (row player)
    # [Cooperate, Defect] x [Cooperate, Defect]
    payoff_1 = [
        [-1, -3],  # Cooperate: -1 if both cooperate, -3 if agent 1 cooperates but agent 2 defects
        [0, -2]    # Defect: 0 if agent 1 defects but agent 2 cooperates, -2 if both defect
    ]
    
    # Payoff matrix for agent 2 (column player)
    payoff_2 = [
        [-1, 0],   # Cooperate: -1 if both cooperate, 0 if agent 1 defects but agent 2 cooperates
        [-3, -2]   # Defect: -3 if agent 1 cooperates but agent 2 defects, -2 if both defect
    ]
    
    return MatrixGame(payoff_1, payoff_2)

def create_matching_pennies():
    """
    Create a Matching Pennies game (zero-sum).
    Actions: 0 = Heads, 1 = Tails
    """
    # Payoff matrix for agent 1 (row player)
    # [Heads, Tails] x [Heads, Tails]
    payoff_1 = [
        [1, -1],  # Heads: 1 if both choose heads, -1 if agent 1 chooses heads but agent 2 chooses tails
        [-1, 1]   # Tails: -1 if agent 1 chooses tails but agent 2 chooses heads, 1 if both choose tails
    ]
    
    # Zero-sum game, so payoff_2 = -payoff_1
    return MatrixGame(payoff_1) 