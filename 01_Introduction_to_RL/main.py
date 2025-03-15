from environment import GridWorld
from value_iteration import value_iteration

def main():
    # Create environment
    env = GridWorld()
    
    # Run value iteration
    V, policy = value_iteration(env)
    
    # Visualize results
    print("Optimal Value Function:")
    for i in range(env.height):
        row = []
        for j in range(env.width):
            s = env.state_to_index((i, j))
            row.append(f"{V[s]:.2f}")
        print("\t".join(row))
    
    print("\nOptimal Policy:")
    action_symbols = ["↑", "→", "↓", "←"]
    for i in range(env.height):
        row = []
        for j in range(env.width):
            s = env.state_to_index((i, j))
            if (i, j) in env.terminal_states:
                row.append("T")
            else:
                row.append(action_symbols[policy[s]])
        print("\t".join(row))
    
    # Render the environment with value function and policy
    env.render(values=V, policy=policy)

if __name__ == "__main__":
    main() 