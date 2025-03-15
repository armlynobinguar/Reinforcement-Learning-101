"""
Implementation of Constrained Policy Optimization (CPO) for Safe Reinforcement Learning.

CPO is a policy optimization algorithm that respects safety constraints during training.
It extends Trust Region Policy Optimization (TRPO) to include constraints on the expected
return of a cost function.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import scipy.optimize

class Actor(nn.Module):
    """
    Actor network for continuous action spaces.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_min=-20, log_std_max=2):
        """
        Initialize actor network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log standard deviation
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State
            
        Returns:
            mean: Action mean
            log_std: Log standard deviation
        """
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state):
        """
        Sample action from policy.
        
        Args:
            state: State
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action = torch.tanh(action)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        """
        Get log probability of action.
        
        Args:
            state: State
            action: Action
            
        Returns:
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Calculate log probability
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        return log_prob

class Critic(nn.Module):
    """
    Critic network for value function.
    """
    
    def __init__(self, state_dim, hidden_dim=64):
        """
        Initialize critic network.
        
        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State
            
        Returns:
            value: State value
        """
        return self.network(state)

class CostCritic(nn.Module):
    """
    Critic network for cost function.
    """
    
    def __init__(self, state_dim, hidden_dim=64):
        """
        Initialize cost critic network.
        
        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
        """
        super(CostCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State
            
        Returns:
            cost: State cost
        """
        return self.network(state)

class CPO:
    """
    Constrained Policy Optimization (CPO) for Safe Reinforcement Learning.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, delta=0.01, cost_limit=25.0):
        """
        Initialize CPO.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            gamma: Discount factor
            delta: KL divergence limit
            cost_limit: Cost limit
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.delta = delta
        self.cost_limit = cost_limit
        
        # Initialize actor
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        
        # Initialize critic
        self.critic = Critic(state_dim, hidden_dim)
        
        # Initialize cost critic
        self.cost_critic = CostCritic(state_dim, hidden_dim)
        
        # Initialize optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=0.001)
    
    def get_action(self, state):
        """
        Get action from policy.
        
        Args:
            state: State
            
        Returns:
            action: Action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.actor.get_action(state)
        
        return action.squeeze().numpy()
    
    def update_critics(self, states, actions, rewards, costs, next_states, dones):
        """
        Update critic networks.
        
        Args:
            states: States
            actions: Actions
            rewards: Rewards
            costs: Costs
            next_states: Next states
            dones: Done flags
            
        Returns:
            critic_loss: Critic loss
            cost_critic_loss: Cost critic loss
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        costs = torch.FloatTensor(costs).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Calculate target values
        with torch.no_grad():
            next_values = self.critic(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
            
            next_costs = self.cost_critic(next_states)
            target_costs = costs + self.gamma * next_costs * (1 - dones)
        
        # Update critic
        values = self.critic(states)
        critic_loss = nn.MSELoss()(values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update cost critic
        costs_pred = self.cost_critic(states)
        cost_critic_loss = nn.MSELoss()(costs_pred, target_costs)
        
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()
        
        return critic_loss.item(), cost_critic_loss.item()
    
    def compute_advantages(self, states, rewards, next_states, dones):
        """
        Compute advantages.
        
        Args:
            states: States
            rewards: Rewards
            next_states: Next states
            dones: Done flags
            
        Returns:
            advantages: Advantages
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # TD error
            delta = rewards + self.gamma * next_values * (1 - dones) - values
            
            # GAE
            advantages = delta.detach().numpy()
        
        return advantages
    
    def compute_cost_advantages(self, states, costs, next_states, dones):
        """
        Compute cost advantages.
        
        Args:
            states: States
            costs: Costs
            next_states: Next states
            dones: Done flags
            
        Returns:
            cost_advantages: Cost advantages
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        costs = torch.FloatTensor(costs).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Calculate cost advantages
        with torch.no_grad():
            cost_values = self.cost_critic(states)
            next_cost_values = self.cost_critic(next_states)
            
            # TD error
            delta = costs + self.gamma * next_cost_values * (1 - dones) - cost_values
            
            # GAE
            cost_advantages = delta.detach().numpy()
        
        return cost_advantages
    
    def update_policy(self, states, actions, advantages, cost_advantages):
        """
        Update policy using CPO.
        
        Args:
            states: States
            actions: Actions
            advantages: Advantages
            cost_advantages: Cost advantages
            
        Returns:
            policy_loss: Policy loss
            kl_divergence: KL divergence
            cost_surrogate: Cost surrogate
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        advantages = torch.FloatTensor(advantages)
        cost_advantages = torch.FloatTensor(cost_advantages)
        
        # Get log probabilities
        with torch.no_grad():
            old_log_probs = self.actor.get_log_prob(states, actions)
        
        # Calculate policy loss
        log_probs = self.actor.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Surrogate loss
        surrogate = (ratio * advantages).mean()
        
        # Cost surrogate
        cost_surrogate = (ratio * cost_advantages).mean()
        
        # KL divergence
        kl_divergence = (old_log_probs - log_probs).mean()
        
        # Solve constrained optimization problem
        def objective(x):
            return -surrogate.item() + x[0] * (kl_divergence.item() - self.delta) + x[1] * (cost_surrogate.item() - self.cost_limit)
        
        def constraint(x):
            return [self.delta - kl_divergence.item(), self.cost_limit - cost_surrogate.item()]
        
        # Initial guess
        x0 = [0.0, 0.0]
        
        # Bounds
        bounds = [(0.0, None), (0.0, None)]
        
        # Solve
        result = scipy.optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraint})
        
        # Get Lagrange multipliers
        lambda_kl = result.x[0]
        lambda_cost = result.x[1]
        
        # Update policy
        policy_loss = -surrogate + lambda_kl * kl_divergence + lambda_cost * cost_surrogate
        
        # Backward pass
        policy_loss.backward()
        
        # Update actor parameters
        for param in self.actor.parameters():
            param.data -= 0.01 * param.grad.data
        
        return policy_loss.item(), kl_divergence.item(), cost_surrogate.item()
    
    def collect_trajectories(self, env, n_trajectories=10):
        """
        Collect trajectories.
        
        Args:
            env: Environment
            n_trajectories: Number of trajectories
            
        Returns:
            states: States
            actions: Actions
            rewards: Rewards
            costs: Costs
            next_states: Next states
            dones: Done flags
            total_rewards: Total rewards
            total_costs: Total costs
        """
        states = []
        actions = []
        rewards = []
        costs = []
        next_states = []
        dones = []
        total_rewards = []
        total_costs = []
        
        for _ in range(n_trajectories):
            state = env.reset()
            done = False
            trajectory_rewards = 0
            trajectory_costs = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Get cost from environment
                cost = info.get('cost', 0.0)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                costs.append(cost)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
                trajectory_rewards += reward
                trajectory_costs += cost
            
            total_rewards.append(trajectory_rewards)
            total_costs.append(trajectory_costs)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(costs), np.array(next_states), np.array(dones), total_rewards, total_costs

def create_safety_gym_env():
    """
    Create a Safety Gym environment.
    
    Returns:
        env: Safety Gym environment
    """
    try:
        import safety_gym
        env = gym.make('Safexp-PointGoal1-v0')
    except ImportError:
        # Fallback to a custom environment with safety constraints
        env = gym.make('Pendulum-v1')
        
        # Wrap environment to add safety constraints
        class SafetyWrapper(gym.Wrapper):
            def __init__(self, env):
                super(SafetyWrapper, self).__init__(env)
                self.env = env
            
            def step(self, action):
                next_state, reward, done, info = self.env.step(action)
                
                # Add safety cost: penalize high velocities
                cost = 0.0
                if abs(next_state[1]) > 1.0:  # Angular velocity
                    cost = 1.0
                
                info['cost'] = cost
                
                return next_state, reward, done, info
        
        env = SafetyWrapper(env)
    
    return env

def main():
    """
    Main function to demonstrate CPO.
    """
    # Create environment
    env = create_safety_gym_env()
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize CPO
    cpo = CPO(state_dim, action_dim)
    
    # Training
    print("Training CPO...")
    rewards_history = []
    costs_history = []
    
    for iteration in range(100):
        # Collect trajectories
        states, actions, rewards, costs, next_states, dones, total_rewards, total_costs = cpo.collect_trajectories(env)
        
        # Update critics
        critic_loss, cost_critic_loss = cpo.update_critics(states, actions, rewards, costs, next_states, dones)
        
        # Compute advantages
        advantages = cpo.compute_advantages(states, rewards, next_states, dones)
        cost_advantages = cpo.compute_cost_advantages(states, costs, next_states, dones)
        
        # Update policy
        policy_loss, kl_divergence, cost_surrogate = cpo.update_policy(states, actions, advantages, cost_advantages)
        
        # Store metrics
        avg_reward = np.mean(total_rewards)
        avg_cost = np.mean(total_costs)
        rewards_history.append(avg_reward)
        costs_history.append(avg_cost)
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Cost: {avg_cost:.2f}")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  KL Divergence: {kl_divergence:.4f}")
            print(f"  Cost Surrogate: {cost_surrogate:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('CPO Rewards')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(costs_history)
    plt.axhline(y=cpo.cost_limit, color='r', linestyle='--', label='Cost Limit')
    plt.xlabel('Iteration')
    plt.ylabel('Average Cost')
    plt.title('CPO Costs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cpo_results.png')
    plt.close()
    
    print("CPO results saved as PNG file.")

if __name__ == "__main__":
    main() 