import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Value Iteration algorithm to find the optimal value function.
    
    Args:
        env: The environment
        gamma: Discount factor
        theta: Convergence threshold
        
    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # Initialize value function
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        
        # Update each state
        for s in range(env.n_states):
            state = env.index_to_state(s)
            
            # Skip terminal states
            if state in env.terminal_states:
                continue
            
            v = V[s]
            
            # Try all possible actions
            action_values = []
            for a in range(env.n_actions):
                # Save current state
                old_state = env.state
                
                # Set state to s
                env.state = state
                
                # Take action a
                next_s, r, done = env.step(a)
                
                # Calculate value
                action_values.append(r + gamma * V[next_s])
                
                # Restore state
                env.state = old_state
            
            # Update value with best action
            V[s] = max(action_values)
            
            # Calculate delta
            delta = max(delta, abs(v - V[s]))
        
        # Check if converged
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        state = env.index_to_state(s)
        
        # Skip terminal states
        if state in env.terminal_states:
            continue
        
        # Try all possible actions
        action_values = []
        for a in range(env.n_actions):
            # Save current state
            old_state = env.state
            
            # Set state to s
            env.state = state
            
            # Take action a
            next_s, r, done = env.step(a)
            
            # Calculate value
            action_values.append(r + gamma * V[next_s])
            
            # Restore state
            env.state = old_state
        
        # Choose best action
        policy[s] = np.argmax(action_values)
    
    return V, policy 