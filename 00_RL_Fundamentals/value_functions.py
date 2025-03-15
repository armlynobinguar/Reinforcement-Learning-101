"""
Value function implementations for reinforcement learning.
"""

import numpy as np

def policy_evaluation(mdp, policy, theta=0.0001):
    """
    Evaluate a policy by computing its value function.
    
    Args:
        mdp: The MDP
        policy: A function mapping states to actions
        theta: Convergence threshold
        
    Returns:
        V: The value function for the policy
    """
    # Initialize value function
    V = {state: 0.0 for state in mdp.states}
    
    while True:
        delta = 0
        
        # Update value function for each state
        for state in mdp.states:
            if state in mdp.terminals:
                continue
                
            v = V[state]
            
            # Get action according to policy
            action = policy(state)
            
            # Calculate new value
            new_v = 0
            for next_state in mdp.get_possible_next_states(state, action):
                prob = mdp.get_transition_prob(state, action, next_state)
                reward = mdp.get_reward(state, action, next_state)
                new_v += prob * (reward + mdp.gamma * V[next_state])
            
            V[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            break
    
    return V

def policy_improvement(mdp, V):
    """
    Improve a policy based on its value function.
    
    Args:
        mdp: The MDP
        V: The value function
        
    Returns:
        policy: The improved policy
    """
    # Initialize policy
    def policy(state):
        if state in mdp.terminals:
            return 0  # Action doesn't matter in terminal states
        
        # Find the best action
        best_action = None
        best_value = float('-inf')
        
        for action in mdp.actions:
            value = 0
            for next_state in mdp.get_possible_next_states(state, action):
                prob = mdp.get_transition_prob(state, action, next_state)
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    return policy

def policy_iteration(mdp, theta=0.0001):
    """
    Find the optimal policy using policy iteration.
    
    Args:
        mdp: The MDP
        theta: Convergence threshold
        
    Returns:
        policy: The optimal policy
        V: The optimal value function
    """
    # Initialize random policy
    def initial_policy(state):
        return np.random.choice(mdp.actions)
    
    policy = initial_policy
    
    while True:
        # Evaluate the policy
        V = policy_evaluation(mdp, policy, theta)
        
        # Improve the policy
        new_policy = policy_improvement(mdp, V)
        
        # Check if policy has changed
        policy_stable = True
        for state in mdp.states:
            if state not in mdp.terminals and new_policy(state) != policy(state):
                policy_stable = False
                break
        
        if policy_stable:
            return new_policy, V
        
        policy = new_policy

def value_iteration(mdp, theta=0.0001):
    """
    Find the optimal value function and policy using value iteration.
    
    Args:
        mdp: The MDP
        theta: Convergence threshold
        
    Returns:
        policy: The optimal policy
        V: The optimal value function
    """
    # Initialize value function
    V = {state: 0.0 for state in mdp.states}
    
    while True:
        delta = 0
        
        # Update value function for each state
        for state in mdp.states:
            if state in mdp.terminals:
                continue
                
            v = V[state]
            
            # Find maximum value over all actions
            values = []
            for action in mdp.actions:
                value = 0
                for next_state in mdp.get_possible_next_states(state, action):
                    prob = mdp.get_transition_prob(state, action, next_state)
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.gamma * V[next_state])
                values.append(value)
            
            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    # Extract policy from value function
    def policy(state):
        if state in mdp.terminals:
            return 0  # Action doesn't matter in terminal states
        
        # Find the best action
        best_action = None
        best_value = float('-inf')
        
        for action in mdp.actions:
            value = 0
            for next_state in mdp.get_possible_next_states(state, action):
                prob = mdp.get_transition_prob(state, action, next_state)
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    return policy, V 