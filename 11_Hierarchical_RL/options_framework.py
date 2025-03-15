"""
Implementation of the Options Framework for Hierarchical Reinforcement Learning.

The Options framework extends the standard RL setting by introducing temporally
extended actions called "options" that consist of:
1. An initiation set (states where the option can be started)
2. An intra-option policy (how to behave while executing the option)
3. A termination condition (when to stop executing the option)
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import defaultdict

class Option:
    """
    Represents a temporally extended action (option) in hierarchical RL.
    """
    
    def __init__(self, init_set_func, policy_func, term_func, name=""):
        """
        Initialize an option.
        
        Args:
            init_set_func: Function that takes a state and returns True if option can be initiated
            policy_func: Function that takes a state and returns an action
            term_func: Function that takes a state and returns termination probability
            name: Name of the option
        """
        self.init_set_func = init_set_func
        self.policy_func = policy_func
        self.term_func = term_func
        self.name = name
    
    def can_initiate(self, state):
        """Check if option can be initiated in state."""
        return self.init_set_func(state)
    
    def get_action(self, state):
        """Get action from option policy."""
        return self.policy_func(state)
    
    def should_terminate(self, state):
        """Check if option should terminate in state."""
        term_prob = self.term_func(state)
        return random.random() < term_prob

class OptionsAgent:
    """
    Agent that uses options for hierarchical reinforcement learning.
    """
    
    def __init__(self, n_states, n_actions, options, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize options agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of primitive actions
            options: List of Option objects
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.options = options
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-values for primitive actions
        self.Q = np.zeros((n_states, n_actions))
        
        # Q-values for options
        self.Q_options = np.zeros((n_states, len(options)))
        
        # Keep track of current option being executed
        self.current_option = None
        self.option_start_state = None
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy over primitive actions and options.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected primitive action
            is_option: Whether the action came from an option policy
        """
        # If currently executing an option
        if self.current_option is not None:
            # Check if option should terminate
            if self.current_option.should_terminate(state):
                self.current_option = None
                self.option_start_state = None
            else:
                # Continue with option's policy
                return self.current_option.get_action(state), True
        
        # Select new action or option
        if np.random.random() < self.epsilon:
            # Explore: random primitive action
            return np.random.randint(self.n_actions), False
        else:
            # Exploit: best action or option
            
            # Get available options
            available_options = [i for i, opt in enumerate(self.options) if opt.can_initiate(state)]
            
            # Combine Q-values for primitive actions and available options
            combined_Q = list(self.Q[state])
            for i in available_options:
                combined_Q.append(self.Q_options[state, i])
            
            # Select best action or option
            best_idx = np.argmax(combined_Q)
            
            if best_idx < self.n_actions:
                # Selected primitive action
                return best_idx, False
            else:
                # Selected option
                option_idx = available_options[best_idx - self.n_actions]
                self.current_option = self.options[option_idx]
                self.option_start_state = state
                return self.current_option.get_action(state), True
    
    def update(self, state, action, reward, next_state, done, is_option):
        """
        Update Q-values based on observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            is_option: Whether the action came from an option policy
        """
        if not is_option:
            # Update Q-value for primitive action
            if done:
                target = reward
            else:
                # Max over primitive actions and available options
                available_options = [i for i, opt in enumerate(self.options) if opt.can_initiate(next_state)]
                
                primitive_max = np.max(self.Q[next_state])
                option_max = np.max(self.Q_options[next_state, available_options]) if available_options else -np.inf
                
                target = reward + self.gamma * max(primitive_max, option_max)
            
            self.Q[state, action] += self.lr * (target - self.Q[state, action])
        
        # If option terminated or episode ended, update option Q-value
        if (is_option and self.current_option is None) or done:
            if self.option_start_state is not None:
                option_idx = self.options.index(self.current_option)
                
                # Calculate cumulative discounted reward
                if done:
                    target = reward
                else:
                    # Max over primitive actions and available options
                    available_options = [i for i, opt in enumerate(self.options) if opt.can_initiate(next_state)]
                    
                    primitive_max = np.max(self.Q[next_state])
                    option_max = np.max(self.Q_options[next_state, available_options]) if available_options else -np.inf
                    
                    target = reward + self.gamma * max(primitive_max, option_max)
                
                self.Q_options[self.option_start_state, option_idx] += self.lr * (target - self.Q_options[self.option_start_state, option_idx])
                
                self.option_start_state = None

def run_options_agent(env, agent, n_episodes=1000):
    """
    Run options agent on environment.
    
    Args:
        env: Environment
        agent: Options agent
        n_episodes: Number of episodes
        
    Returns:
        rewards: List of total rewards per episode
    """
    rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        state = discretize_state(state, env)
        total_reward = 0
        done = False
        
        while not done:
            action, is_option = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, env)
            
            agent.update(state, action, reward, next_state, done, is_option)
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return rewards

def discretize_state(state, env):
    """
    Discretize a continuous state.
    
    Args:
        state: Continuous state
        env: Environment
        
    Returns:
        index: Discretized state index
    """
    if env.spec.id == 'CartPole-v1':
        # CartPole: cart position, cart velocity, pole angle, pole velocity
        cart_pos_bins = np.linspace(-2.4, 2.4, 3)
        cart_vel_bins = np.linspace(-3, 3, 3)
        pole_ang_bins = np.linspace(-0.2, 0.2, 6)
        pole_vel_bins = np.linspace(-3, 3, 3)
        
        # Digitize state
        cart_pos_idx = np.digitize(state[0], cart_pos_bins)
        cart_vel_idx = np.digitize(state[1], cart_vel_bins)
        pole_ang_idx = np.digitize(state[2], pole_ang_bins)
        pole_vel_idx = np.digitize(state[3], pole_vel_bins)
        
        # Calculate index
        return cart_pos_idx * 3 * 6 * 3 + cart_vel_idx * 6 * 3 + pole_ang_idx * 3 + pole_vel_idx
    
    else:
        # Default: just return the state
        return state

def create_cartpole_options():
    """
    Create options for CartPole environment.
    
    Returns:
        options: List of Option objects
    """
    # Option 1: Balance when pole is leaning right
    def init_set_1(state):
        # Can initiate when pole is leaning right
        cart_pos_bins = np.linspace(-2.4, 2.4, 3)
        cart_vel_bins = np.linspace(-3, 3, 3)
        pole_ang_bins = np.linspace(-0.2, 0.2, 6)
        pole_vel_bins = np.linspace(-3, 3, 3)
        
        cart_pos_idx = np.digitize(state[0], cart_pos_bins)
        cart_vel_idx = np.digitize(state[1], cart_vel_bins)
        pole_ang_idx = np.digitize(state[2], pole_ang_bins)
        pole_vel_idx = np.digitize(state[3], pole_vel_bins)
        
        return pole_ang_idx > 3  # Pole leaning right
    
    def policy_1(state):
        # Move cart right
        return 1
    
    def term_1(state):
        # Terminate when pole is centered
        cart_pos_bins = np.linspace(-2.4, 2.4, 3)
        cart_vel_bins = np.linspace(-3, 3, 3)
        pole_ang_bins = np.linspace(-0.2, 0.2, 6)
        pole_vel_bins = np.linspace(-3, 3, 3)
        
        cart_pos_idx = np.digitize(state[0], cart_pos_bins)
        cart_vel_idx = np.digitize(state[1], cart_vel_bins)
        pole_ang_idx = np.digitize(state[2], pole_ang_bins)
        pole_vel_idx = np.digitize(state[3], pole_vel_bins)
        
        return 1.0 if pole_ang_idx == 3 else 0.0  # Terminate when pole is centered
    
    option_1 = Option(init_set_1, policy_1, term_1, "Balance Right")
    
    # Option 2: Balance when pole is leaning left
    def init_set_2(state):
        # Can initiate when pole is leaning left
        cart_pos_bins = np.linspace(-2.4, 2.4, 3)
        cart_vel_bins = np.linspace(-3, 3, 3)
        pole_ang_bins = np.linspace(-0.2, 0.2, 6)
        pole_vel_bins = np.linspace(-3, 3, 3)
        
        cart_pos_idx = np.digitize(state[0], cart_pos_bins)
        cart_vel_idx = np.digitize(state[1], cart_vel_bins)
        pole_ang_idx = np.digitize(state[2], pole_ang_bins)
        pole_vel_idx = np.digitize(state[3], pole_vel_bins)
        
        return pole_ang_idx < 3  # Pole leaning left
    
    def policy_2(state):
        # Move cart left
        return 0
    
    def term_2(state):
        # Terminate when pole is centered
        cart_pos_bins = np.linspace(-2.4, 2.4, 3)
        cart_vel_bins = np.linspace(-3, 3, 3)
        pole_ang_bins = np.linspace(-0.2, 0.2, 6)
        pole_vel_bins = np.linspace(-3, 3, 3)
        
        cart_pos_idx = np.digitize(state[0], cart_pos_bins)
        cart_vel_idx = np.digitize(state[1], cart_vel_bins)
        pole_ang_idx = np.digitize(state[2], pole_ang_bins)
        pole_vel_idx = np.digitize(state[3], pole_vel_bins)
        
        return 1.0 if pole_ang_idx == 3 else 0.0  # Terminate when pole is centered
    
    option_2 = Option(init_set_2, policy_2, term_2, "Balance Left")
    
    return [option_1, option_2]

def main():
    """
    Main function to demonstrate options framework.
    """
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create options
    options = create_cartpole_options()
    
    # Get state and action dimensions
    n_states = 3 * 3 * 6 * 3  # Discretized state space
    n_actions = env.action_space.n
    
    # Initialize agent
    agent = OptionsAgent(n_states, n_actions, options, learning_rate=0.1, gamma=0.99, epsilon=0.1)
    
    # Run options agent
    rewards = run_options_agent(env, agent, n_episodes=1000)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Options Framework Learning Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig('options_framework_rewards.png')
    plt.close()
    
    print("Options framework learning results saved as PNG file.")

if __name__ == "__main__":
    main() 