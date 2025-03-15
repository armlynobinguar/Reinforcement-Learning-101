"""
Compatibility wrapper for gym/gymnasium environments.
This handles the transition from gym to gymnasium.
"""

try:
    import gymnasium as gym
    USING_GYMNASIUM = True
except ImportError:
    import gym
    USING_GYMNASIUM = False

def create_env(env_id, **kwargs):
    """
    Create a gym environment with proper version handling.
    
    Args:
        env_id: The environment ID
        **kwargs: Additional arguments to pass to gym.make
        
    Returns:
        A gym environment
    """
    if USING_GYMNASIUM:
        # Gymnasium version
        return gym.make(env_id, **kwargs)
    else:
        # Old gym version
        return gym.make(env_id, **kwargs)

def env_reset(env):
    """
    Reset the environment with proper version handling.
    
    Args:
        env: The gym environment
        
    Returns:
        Initial state and info
    """
    if USING_GYMNASIUM:
        # Gymnasium version returns (obs, info)
        return env.reset()
    else:
        # Old gym version returns just obs
        return env.reset(), {}

def env_step(env, action):
    """
    Take a step in the environment with proper version handling.
    
    Args:
        env: The gym environment
        action: The action to take
        
    Returns:
        next_state, reward, terminated, truncated, info
    """
    if USING_GYMNASIUM:
        # Gymnasium version returns (obs, reward, terminated, truncated, info)
        return env.step(action)
    else:
        # Old gym version returns (obs, reward, done, info)
        next_state, reward, done, info = env.step(action)
        return next_state, reward, done, False, info 