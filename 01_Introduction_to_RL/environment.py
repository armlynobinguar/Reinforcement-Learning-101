import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """
    A simple 4x4 gridworld environment for introducing RL concepts.
    """
    def __init__(self):
        # Grid dimensions
        self.height = 4
        self.width = 4
        
        # Define states
        self.n_states = self.height * self.width
        
        # Define actions: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Define rewards
        self.rewards = np.zeros((self.height, self.width))
        self.rewards[0, 0] = -1  # Negative reward state
        self.rewards[self.height-1, self.width-1] = 1  # Goal state
        
        # Define terminal states
        self.terminal_states = [(0, 0), (self.height-1, self.width-1)]
        
        # Current state
        self.reset()
    
    def reset(self):
        # Start in a random non-terminal state
        while True:
            self.state = (np.random.randint(self.height), np.random.randint(self.width))
            if self.state not in self.terminal_states:
                break
        return self.state_to_index(self.state)
    
    def step(self, action):
        # Check if current state is terminal
        if self.state in self.terminal_states:
            return self.state_to_index(self.state), 0, True
        
        # Get action direction
        direction = self.actions[action]
        
        # Calculate new state
        new_state = (
            max(0, min(self.height-1, self.state[0] + direction[0])),
            max(0, min(self.width-1, self.state[1] + direction[1]))
        )
        
        # Update state
        self.state = new_state
        
        # Get reward
        reward = self.rewards[self.state]
        
        # Check if done
        done = self.state in self.terminal_states
        
        return self.state_to_index(self.state), reward, done
    
    def state_to_index(self, state):
        return state[0] * self.width + state[1]
    
    def index_to_state(self, index):
        return (index // self.width, index % self.width)
    
    def render(self, values=None, policy=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(i, color='black', lw=2)
        for j in range(self.width + 1):
            ax.axvline(j, color='black', lw=2)
        
        # Fill in rewards
        for i in range(self.height):
            for j in range(self.width):
                if self.rewards[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j, self.height-i-1), 1, 1, fill=True, color='green', alpha=0.3))
                elif self.rewards[i, j] == -1:
                    ax.add_patch(plt.Rectangle((j, self.height-i-1), 1, 1, fill=True, color='red', alpha=0.3))
        
        # Show values if provided
        if values is not None:
            for i in range(self.height):
                for j in range(self.width):
                    state_idx = self.state_to_index((i, j))
                    plt.text(j + 0.5, self.height - i - 0.5, f"{values[state_idx]:.2f}", 
                             ha='center', va='center', fontsize=12)
        
        # Show policy if provided
        if policy is not None:
            for i in range(self.height):
                for j in range(self.width):
                    if (i, j) in self.terminal_states:
                        continue
                    
                    state_idx = self.state_to_index((i, j))
                    action = policy[state_idx]
                    
                    # Draw arrow
                    dx, dy = self.actions[action]
                    plt.arrow(j + 0.5, self.height - i - 0.5, dx * 0.3, -dy * 0.3, 
                              head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Set limits and labels
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0.5, self.width, 1))
        ax.set_yticks(np.arange(0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.title('GridWorld')
        plt.show() 