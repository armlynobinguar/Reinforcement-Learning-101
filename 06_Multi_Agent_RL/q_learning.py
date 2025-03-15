"""
Multi-agent Q-learning implementation.
"""

import numpy as np

class MultiAgentQLearning:
    """
    Independent Q-learning for multiple agents.
    Each agent learns its own Q-function independently.
    """
    
    def __init__(self, num_states, num_actions_list, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize multi-agent Q-learning.
        
        Args:
            num_states: Number of states
            num_actions_list: List of number of actions for each agent
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
        """
        self.num_agents = len(num_actions_list)
        self.num_states = num_states
        self.num_actions_list = num_actions_list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-tables for each agent
        self.q_tables = [np.zeros((num_states, num_actions)) for num_actions in num_actions_list]
    
    def select_actions(self, state):
        """
        Select actions for all agents using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            actions: List of actions for each agent
        """
        actions = []
        
        for agent_id in range(self.num_agents):
            if np.random.random() < self.epsilon:
                # Explore: random action
                action = np.random.randint(self.num_actions_list[agent_id])
            else:
                # Exploit: best action
                action = np.argmax(self.q_tables[agent_id][state, :])
            
            actions.append(action)
        
        return actions
    
    def update(self, state, actions, rewards, next_state, done):
        """
        Update Q-values for all agents.
        
        Args:
            state: Current state
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_state: Next state
            done: Whether the episode is done
        """
        for agent_id in range(self.num_agents):
            action = actions[agent_id]
            reward = rewards[agent_id]
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_tables[agent_id][next_state, :])
            
            self.q_tables[agent_id][state, action] += self.lr * (target - self.q_tables[agent_id][state, action])
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes=10000):
        """
        Train agents using Q-learning.
        
        Args:
            env: Environment
            episodes: Number of episodes
            
        Returns:
            rewards_history: List of rewards for each episode
        """
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            total_rewards = np.zeros(self.num_agents)
            done = False
            
            while not done:
                actions = self.select_actions(state)
                next_state, rewards, done = env.step(*actions)
                
                self.update(state, actions, rewards, next_state, done)
                
                state = next_state
                total_rewards += rewards
            
            rewards_history.append(total_rewards)
            
            if episode % 1000 == 0:
                avg_rewards = np.mean(rewards_history[-100:], axis=0)
                print(f"Episode {episode}, Average Rewards: {avg_rewards}, Epsilon: {self.epsilon:.2f}")
        
        return rewards_history 