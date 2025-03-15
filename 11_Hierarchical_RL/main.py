"""
Main script to demonstrate the Options Framework for Hierarchical Reinforcement Learning.

This script creates a simple four-room gridworld environment and shows how options
(temporally extended actions) can be used to improve learning efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import defaultdict
import time

# Import the Options Framework
from options_framework import Option, OptionsAgent, run_options_agent

class FourRoomsEnv:
    """
    Four-room gridworld environment.
    
    The environment consists of four rooms connected by hallways.
    The agent starts in the bottom-left room and needs to reach the goal
    in the top-right room.
    """
    
    def __init__(self, size=11):
        """
        Initialize four-room environment.
        
        Args:
            size: Size of the grid (size x size)
        """
        self.size = size
        self.grid = np.zeros((size, size))
        
        # Create walls
        self.grid[0, :] = 1  # Top wall
        self.grid[size-1, :] = 1  # Bottom wall
        self.grid[:, 0] = 1  # Left wall
        self.grid[:, size-1] = 1  # Right wall
        
        # Create room dividers
        self.grid[size//2, :] = 1
        self.grid[:, size//2] = 1
        
        # Create hallways
        self.grid[size//2, size//4] = 0  # Left hallway
        self.grid[size//2, 3*size//4] = 0  # Right hallway
        self.grid[size//4, size//2] = 0  # Top hallway
        self.grid[3*size//4, size//2] = 0  # Bottom hallway
        
        # Set start and goal positions
        self.start_pos = (size-2, 1)  # Bottom-left room
        self.goal_pos = (1, size-2)  # Top-right room
        
        # Set current position
        self.pos = self.start_pos
        
        # Define action space
        self.action_space = gym.spaces.Discrete(4)  # Up, Right, Down, Left
        
        # Define observation space
        self.observation_space = gym.spaces.Discrete(size * size)
    
    def reset(self):
        """
        Reset environment.
        
        Returns:
            state: Initial state
        """
        self.pos = self.start_pos
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Up, 1: Right, 2: Down, 3: Left)
            
        Returns:
            next_state: Next state
            reward: Reward
            done: Whether episode is done
            info: Additional information
        """
        # Get current position
        row, col = self.pos
        
        # Compute next position
        if action == 0:  # Up
            next_pos = (row-1, col)
        elif action == 1:  # Right
            next_pos = (row, col+1)
        elif action == 2:  # Down
            next_pos = (row+1, col)
        elif action == 3:  # Left
            next_pos = (row, col-1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if next position is valid
        next_row, next_col = next_pos
        if (0 <= next_row < self.size and 0 <= next_col < self.size and
            self.grid[next_row, next_col] == 0):
            self.pos = next_pos
        
        # Compute reward and done flag
        if self.pos == self.goal_pos:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        """
        Get current state.
        
        Returns:
            state: Current state
        """
        row, col = self.pos
        return row * self.size + col
    
    def render(self):
        """
        Render environment.
        """
        grid = np.copy(self.grid)
        row, col = self.pos
        grid[row, col] = 2  # Agent
        goal_row, goal_col = self.goal_pos
        grid[goal_row, goal_col] = 3  # Goal
        
        # Print grid
        for row in range(self.size):
            for col in range(self.size):
                if grid[row, col] == 0:
                    print(".", end=" ")  # Empty
                elif grid[row, col] == 1:
                    print("#", end=" ")  # Wall
                elif grid[row, col] == 2:
                    print("A", end=" ")  # Agent
                elif grid[row, col] == 3:
                    print("G", end=" ")  # Goal
            print()
        print()

def create_options(env):
    """
    Create options for four-room environment.
    
    Args:
        env: Four-room environment
        
    Returns:
        options: List of options
    """
    options = []
    
    # Define hallway positions
    hallways = [
        (env.size//2, env.size//4),    # Left hallway
        (env.size//2, 3*env.size//4),  # Right hallway
        (env.size//4, env.size//2),    # Top hallway
        (3*env.size//4, env.size//2)   # Bottom hallway
    ]
    
    # Create options to reach each hallway
    for i, (hall_row, hall_col) in enumerate(hallways):
        hall_state = hall_row * env.size + hall_col
        
        # Define initiation set (all states except hallway)
        def init_function(s, hall_state=hall_state):
            return s != hall_state
        
        # Define termination condition (hallway state)
        def term_function(s, hall_state=hall_state):
            return s == hall_state
        
        # Create policy (will be learned)
        policy = {}
        
        # Use positional arguments instead of keyword arguments
        option = Option(init_function, term_function, policy)
        
        options.append(option)
    
    return options

def run_experiment(env, use_options=False, n_episodes=500):
    """
    Run experiment with or without options.
    
    Args:
        env: Environment
        use_options: Whether to use options
        n_episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
        steps: List of steps per episode
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    if use_options:
        # Create options
        options = create_options(env)
        
        # Initialize agent with options
        agent = OptionsAgent(n_states, n_actions, options)
        
        # Define learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.99  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        # Custom implementation of run_options_agent for FourRoomsEnv
        rewards = []
        steps = []
        
        for episode in range(n_episodes):
            # Reset environment
            state = env.reset()
            
            # Initialize total reward and steps
            total_reward = 0
            episode_steps = 0
            
            # Run episode
            done = False
            while not done:
                # Choose between primitive action and option
                if np.random.random() < epsilon:
                    # Random action
                    action = env.action_space.sample()
                    next_state, reward, done, _ = env.step(action)
                    episode_steps += 1
                    total_reward += reward
                    
                    # Update Q-value
                    agent.Q[state, action] += alpha * (
                        reward + gamma * np.max(agent.Q[next_state]) - agent.Q[state, action]
                    )
                    
                    state = next_state
                else:
                    # Choose best option or primitive action
                    option_values = agent.Q_options[state]
                    primitive_values = agent.Q[state]
                    
                    if np.max(option_values) > np.max(primitive_values):
                        # Execute option
                        option_idx = np.argmax(option_values)
                        option = agent.options[option_idx]
                        
                        if option.is_initiation_state(state):
                            option_reward = 0
                            option_steps = 0
                            option_states = [state]
                            
                            while not option.is_termination_state(state) and not done:
                                # Use primitive policy for now (will be learned)
                                action = np.argmax(agent.Q[state])
                                next_state, reward, done, _ = env.step(action)
                                
                                option_reward += reward
                                option_steps += 1
                                option_states.append(next_state)
                                
                                state = next_state
                            
                            # Update option Q-value
                            for s in option_states[:-1]:
                                agent.Q_options[s, option_idx] += alpha * (
                                    option_reward + gamma**option_steps * np.max(agent.Q_options[state]) - 
                                    agent.Q_options[s, option_idx]
                                )
                        else:
                            # Use primitive action
                            action = np.argmax(agent.Q[state])
                            next_state, reward, done, _ = env.step(action)
                            episode_steps += 1
                            total_reward += reward
                            
                            # Update Q-value
                            agent.Q[state, action] += alpha * (
                                reward + gamma * np.max(agent.Q[next_state]) - agent.Q[state, action]
                            )
                            
                            state = next_state
                    else:
                        # Use primitive action
                        action = np.argmax(agent.Q[state])
                        next_state, reward, done, _ = env.step(action)
                        episode_steps += 1
                        total_reward += reward
                        
                        # Update Q-value
                        agent.Q[state, action] += alpha * (
                            reward + gamma * np.max(agent.Q[next_state]) - agent.Q[state, action]
                        )
                        
                        state = next_state
            
            # Store total reward and steps
            rewards.append(total_reward)
            steps.append(episode_steps)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {total_reward}, Steps: {episode_steps}")
    else:
        # Initialize Q-values
        Q = np.zeros((n_states, n_actions))
        
        # Initialize rewards and steps lists
        rewards = []
        steps = []
        
        for episode in range(n_episodes):
            # Reset environment
            state = env.reset()
            
            # Initialize total reward and steps
            total_reward = 0
            episode_steps = 0
            
            # Run episode
            done = False
            while not done:
                # Select action using epsilon-greedy policy
                if np.random.random() < 0.1:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Update Q-value
                Q[state, action] = Q[state, action] + 0.1 * (reward + 0.99 * np.max(Q[next_state]) - Q[state, action])
                
                # Update total reward and steps
                total_reward += reward
                episode_steps += 1
                
                # Update state
                state = next_state
            
            # Store total reward and steps
            rewards.append(total_reward)
            steps.append(episode_steps)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {total_reward}, Steps: {episode_steps}")
    
    return rewards, steps

def main():
    """
    Main function to run experiments.
    """
    # Create environment
    env = FourRoomsEnv()
    
    # Run experiment without options
    print("Running experiment without options...")
    start_time = time.time()
    rewards_no_options, steps_no_options = run_experiment(env, use_options=False, n_episodes=500)
    end_time = time.time()
    print(f"Experiment without options completed in {end_time - start_time:.2f} seconds.")
    
    # Run experiment with options
    print("\nRunning experiment with options...")
    start_time = time.time()
    rewards_options, steps_options = run_experiment(env, use_options=True, n_episodes=500)
    end_time = time.time()
    print(f"Experiment with options completed in {end_time - start_time:.2f} seconds.")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards_no_options, label='Without Options')
    plt.plot(rewards_options, label='With Options')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Hierarchical RL: Total Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot steps
    plt.subplot(2, 1, 2)
    plt.plot(steps_no_options, label='Without Options')
    plt.plot(steps_options, label='With Options')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Hierarchical RL: Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hierarchical_rl_results.png')
    plt.show()
    
    print("Results saved as PNG file.")

if __name__ == "__main__":
    main() 