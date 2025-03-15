# Key Concepts in Reinforcement Learning

## The RL Framework

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The RL framework consists of:

- **Agent**: The learner or decision-maker
- **Environment**: Everything the agent interacts with
- **State (S)**: The current situation or configuration
- **Action (A)**: What the agent can do
- **Reward (R)**: Feedback signal indicating the desirability of the state-action pair
- **Policy (π)**: The agent's strategy or mapping from states to actions
- **Value Function (V or Q)**: Prediction of future rewards
- **Model**: The agent's representation of the environment

## Markov Decision Processes (MDPs)

MDPs provide a mathematical framework for modeling decision-making problems where outcomes are partly random and partly under the control of a decision-maker. An MDP is defined by:

- A set of states S
- A set of actions A
- Transition probabilities P(s'|s,a)
- Reward function R(s,a,s')
- Discount factor γ ∈ [0,1]

## The Exploration-Exploitation Dilemma

One of the fundamental challenges in RL is balancing:

- **Exploration**: Trying new actions to discover better strategies
- **Exploitation**: Using known information to maximize reward

## Types of RL Algorithms

1. **Model-Based**: Learn a model of the environment and use it for planning
2. **Model-Free**: Learn directly from experience without building a model
   - **Value-Based**: Learn value functions (e.g., Q-learning)
   - **Policy-Based**: Learn policies directly (e.g., REINFORCE)
   - **Actor-Critic**: Combine value and policy learning

## Key Equations

- **Bellman Equation for State Values**:
  V(s) = max_a [R(s,a) + γ ∑_s' P(s'|s,a)V(s')]

- **Bellman Equation for Action Values**:
  Q(s,a) = R(s,a) + γ ∑_s' P(s'|s,a) max_a' Q(s',a')

- **Policy Gradient**:
  ∇J(θ) = E_π [∇log π(a|s;θ) Q(s,a)]

## Applications of RL

- Game playing (Chess, Go, Atari games)
- Robotics and control
- Resource management
- Recommendation systems
- Healthcare
- Finance and trading
- Natural language processing 