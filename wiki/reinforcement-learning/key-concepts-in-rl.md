---
date: 2025-03-11 # YYYY-MM-DD
title: Key Concepts of Reinforcement Learning
---

This tutorial provides an introduction to the fundamental concepts of Reinforcement Learning (RL). RL involves an agent interacting with an environment to learn optimal behaviors through trial and feedback. The main objective of RL is to maximize cumulative rewards over time.

## Main Components of Reinforcement Learning

### Agent and Environment
The agent is the learner or decision-maker, while the environment represents everything the agent interacts with. The agent receives observations from the environment and takes actions that influence the environment's state.

![Humanoid robot agent interacting with its environment](/assets/images/humanoid-robot.drawio.png)
{: .full}

### States and Observations
- A **state** (s) fully describes the world at a given moment.
- An **observation** (o) is a partial view of the state.
- Environments can be **fully observed** (complete information) or **partially observed** (limited information).

### Action Spaces
- The **action space** defines all possible actions an agent can take.
- **Discrete action spaces** (e.g., Atari, Go) have a finite number of actions.
- **Continuous action spaces** (e.g., robotics control) allow real-valued actions.

## Policies
A **policy** determines how an agent selects actions based on states:

- **Deterministic policy**: Always selects the same action for a given state.
  
  $a_t = \mu(s_t)$
  
- **Stochastic policy**: Samples actions from a probability distribution.
  
  $a_t \sim \pi(\cdot | s_t)$
  

### Example: Deterministic Policy in PyTorch
```python
import torch.nn as nn

pi_net = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, act_dim)
)
```

## Trajectories
A **trajectory ($\tau$)** is a sequence of states and actions:
$$
\tau = (s_0, a_0, s_1, a_1, \dots)
$$
State transitions follow deterministic or stochastic rules:
$$
s_{t+1} = f(s_t, a_t)
$$
or
$$
s_{t+1} \sim P(\cdot|s_t, a_t)
$$

## Reward and Return
The **reward function ($R$)** determines the agent's objective:
$$
r_t = R(s_t, a_t, s_{t+1})
$$
### Types of Return
1. **Finite-horizon undiscounted return**:
   $$
   R(\tau) = \sum_{t=0}^T r_t
   $$
2. **Infinite-horizon discounted return**:
   $$
   R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
   $$
   where $\gamma$ (discount factor) balances immediate vs. future rewards.

## Summary
This tutorial introduced fundamental RL concepts, including agents, environments, policies, action spaces, trajectories, and rewards. These components are essential for designing RL algorithms.

## Further Reading
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.

## References
- [Reinforcement Learning Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
