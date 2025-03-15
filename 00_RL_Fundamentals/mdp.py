"""
Markov Decision Process (MDP) implementation.
This module defines the core components of an MDP.
"""

class MDP:
    """
    A Markov Decision Process defined by:
    - States: A set of possible states
    - Actions: A set of possible actions
    - Transitions: P(s'|s,a) - Probability of transitioning to state s' from state s after taking action a
    - Rewards: R(s,a,s') - Reward received after transitioning from state s to s' via action a
    - Discount factor: gamma - How much to discount future rewards
    """
    
    def __init__(self, states, actions, transitions, rewards, gamma=0.9):
        """
        Initialize an MDP.
        
        Args:
            states: List of states
            actions: List of actions
            transitions: Function P(s'|s,a) returning probability of next state
            rewards: Function R(s,a,s') returning reward
            gamma: Discount factor
        """
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
    
    def get_transition_prob(self, state, action, next_state):
        """Get the probability of transitioning to next_state from state via action."""
        return self.transitions(state, action, next_state)
    
    def get_reward(self, state, action, next_state):
        """Get the reward for transitioning to next_state from state via action."""
        return self.rewards(state, action, next_state)
    
    def get_possible_next_states(self, state, action):
        """Get all possible next states from state via action with non-zero probability."""
        return [s for s in self.states if self.get_transition_prob(state, action, s) > 0]


class GridWorldMDP(MDP):
    """
    A simple grid world MDP implementation.
    The agent can move in four directions: up, right, down, left.
    """
    
    def __init__(self, width, height, obstacles=None, terminals=None, rewards=None, gamma=0.9):
        """
        Initialize a grid world MDP.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            obstacles: List of (x,y) coordinates of obstacles
            terminals: List of (x,y) coordinates of terminal states
            rewards: Dictionary mapping (x,y) coordinates to rewards
            gamma: Discount factor
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        self.terminals = terminals or []
        self.rewards_map = rewards or {}
        
        # Define states as (x,y) coordinates
        states = [(x, y) for x in range(width) for y in range(height) if (x, y) not in self.obstacles]
        
        # Define actions: 0=up, 1=right, 2=down, 3=left
        actions = [0, 1, 2, 3]
        
        # Define transitions and rewards functions
        def transitions(state, action, next_state):
            if state in self.terminals:
                return 1.0 if state == next_state else 0.0
            
            x, y = state
            nx, ny = next_state
            
            # Calculate intended next position
            if action == 0:  # up
                intended_next_pos = (x, min(y+1, height-1))
            elif action == 1:  # right
                intended_next_pos = (min(x+1, width-1), y)
            elif action == 2:  # down
                intended_next_pos = (x, max(y-1, 0))
            elif action == 3:  # left
                intended_next_pos = (max(x-1, 0), y)
            
            # Check if intended position is an obstacle
            if intended_next_pos in self.obstacles:
                intended_next_pos = state
            
            # Deterministic transitions
            return 1.0 if intended_next_pos == next_state else 0.0
        
        def rewards(state, action, next_state):
            return self.rewards_map.get(next_state, 0.0)
        
        super().__init__(states, actions, transitions, rewards, gamma) 