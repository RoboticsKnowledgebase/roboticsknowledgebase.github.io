---
date: 2025-05-04
title: A Taxonomy of Reinforcement Learning Algorithms
---

Reinforcement Learning (RL) is a foundational paradigm in artificial intelligence where agents learn to make decisions through trial and error, guided by rewards. Over the years, a rich variety of RL algorithms have been developed, each differing in the way they represent knowledge, interact with the environment, and generalize from data. This article presents a high-level taxonomy of RL algorithms with an emphasis on design trade-offs, learning objectives, and algorithmic categories. The goal is to provide a structured guide to the RL landscape for students and practitioners.

## Model-Based vs Model-Free Reinforcement Learning

One of the most fundamental distinctions among RL algorithms is whether or not the algorithm uses a model of the environment's dynamics.

### Model-Free RL

Model-free algorithms do not attempt to learn or use an internal model of the environment. Instead, they learn policies or value functions directly from experience. These methods are typically simpler to implement and tune, making them more widely adopted in practice.

**Key Advantages:**
- Easier to apply when the environment is complex or high-dimensional.
- No need for a simulator or model-learning pipeline.

**Drawbacks:**
- High sample complexity: requires many interactions with the real or simulated environment.
- Cannot perform planning or imagination-based reasoning.

**Examples:**
- **DQN (Deep Q-Networks)**: First to combine Q-learning with deep networks for Atari games.
- **PPO (Proximal Policy Optimization)**: A robust policy gradient method widely used in robotics and games.

### Model-Based RL

In contrast, model-based algorithms explicitly learn or use a model of the environment that predicts future states and rewards. The agent can then plan ahead by simulating trajectories using this model.

**Key Advantages:**
- Better sample efficiency through planning and simulation.
- Can separate learning from data collection, enabling "dream-based" training.

**Challenges:**
- Learning accurate models is difficult.
- Errors in the model can lead to compounding errors during planning.

**Use Cases:**
- High-stakes environments where sample efficiency is critical.
- Scenarios requiring imagination or foresight (e.g., robotics, strategic games).

**Examples:**
- **MBVE (Model-Based Value Expansion)**: Uses a learned model to expand the value estimate of real transitions.
- **AlphaZero**: Combines MCTS with learned value/policy networks to dominate board games.

## What to Learn: Policy, Value, Q, or Model?

RL algorithms also differ based on what the agent is trying to learn:

- **Policy** $\pi_\theta(a|s)$: A mapping from state to action, either deterministic or stochastic.
- **Value function** $V^\pi(s)$: The expected return starting from state $s$ under policy $\pi$.
- **Action-Value (Q) function** $Q^\pi(s, a)$: The expected return starting from state $s$ taking action $a$, then following $\pi$.
- **Model**: A transition function $f(s, a) \rightarrow s'$ and reward predictor $r(s, a)$.

### Model-Free Learning Strategies

#### 1. Policy Optimization

These algorithms directly optimize the parameters of a policy using gradient ascent on a performance objective:

$$
J(\pi_\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r_t \right]
$$

They often require estimating the advantage function or value function to reduce variance.

**Characteristics:**
- **On-policy**: Data must come from the current policy.
- **Stable and robust**: Optimizes directly for performance.

**Popular Methods:**
- **A2C / A3C (Asynchronous Advantage Actor-Critic)**: Learns both policy and value function in parallel.
- **PPO (Proximal Policy Optimization)**: Ensures stable updates with clipped surrogate objectives.
- **TRPO (Trust Region Policy Optimization)**: Uses trust regions to prevent catastrophic policy changes.

#### 2. Q-Learning

Instead of learning a policy directly, Q-learning methods aim to learn the optimal action-value function:

$$
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
$$

Once $Q^*(s, a)$ is known, the policy is derived via:

$$
\pi(s) = \arg\max_a Q^*(s, a)
$$

**Characteristics:**
- **Off-policy**: Can use data from any past policy.
- **Data-efficient**, but prone to instability.

**Variants:**
- **DQN**: Introduced experience replay and target networks.
- **C51 / QR-DQN**: Learn a distribution over returns, not just the mean.

> **Trade-Off**: Policy gradient methods are more stable and principled; Q-learning methods are more sample-efficient but harder to stabilize due to the "deadly triad": function approximation, bootstrapping, and off-policy updates.

#### Hybrid Algorithms

Some methods blend policy optimization and Q-learning:

- **DDPG (Deep Deterministic Policy Gradient)**: Actor-Critic method with off-policy Q-learning and deterministic policies.
- **TD3 (Twin Delayed DDPG)**: Addresses overestimation bias in DDPG.
- **SAC (Soft Actor-Critic)**: Adds entropy regularization to encourage exploration and stabilize learning.

### Model-Based Learning Strategies

Model-based RL allows a variety of architectures and learning techniques.

#### 1. Pure Planning (e.g., MPC)

The agent uses a learned or known model to plan a trajectory and execute the first action, then replan. No policy is explicitly learned.

#### 2. Expert Iteration (ExIt)

Combines planning and learning. Planning (e.g., via MCTS) provides strong actions ("experts"), which are used to train a policy via supervised learning.

- **AlphaZero**: Exemplifies this method by using MCTS and neural nets in self-play.

#### 3. Data Augmentation

The learned model is used to synthesize additional training data.

- **MBVE**: Augments true experiences with simulated rollouts.
- **World Models**: Trains entirely on imagined data ("dreaming").

#### 4. Imagination-Augmented Agents (I2A)

Here, planning is embedded as a subroutine inside the policy network. The policy learns when and how to use imagination.

> This technique can mitigate model bias because the policy can learn to ignore poor planning results.

## Summary

The landscape of RL algorithms is broad and evolving, but organizing them into categories based on model usage and learning targets helps build intuition:

| Dimension               | Model-Free RL            | Model-Based RL                          |
|------------------------|--------------------------|------------------------------------------|
| Sample Efficiency      | Low                      | High                                     |
| Stability              | High (Policy Gradient)   | Variable (depends on model quality)      |
| Planning Capability    | None                     | Yes (MPC, MCTS, ExIt)                    |
| Real-World Deployment  | Slower                   | Faster (if model is accurate)            |
| Representative Methods | DQN, PPO, A2C            | AlphaZero, MBVE, World Models, I2A       |

Understanding these trade-offs is key to selecting or designing an RL algorithm for your application.


## Further Reading
- [Spinning Up in Deep RL – OpenAI](https://spinningup.openai.com/en/latest/)
- [RL Course by David Silver](https://www.davidsilver.uk/teaching/)
- [RL Book – Sutton and Barto (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
- [CS285: Deep Reinforcement Learning – UC Berkeley (Sergey Levine)](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [Deep RL Bootcamp (2017) – Stanford](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [Lil’Log – Reinforcement Learning Series by Lilian Weng](https://lilianweng.github.io/lil-log/)
- [RL Algorithms – Denny Britz’s GitHub](https://github.com/dennybritz/reinforcement-learning)
- [Reinforcement Learning Zoo – A curated collection of RL papers and code](https://github.com/instillai/reinforcement-learning-zoo)
- [Distill: Visualizing Reinforcement Learning](https://distill.pub/2019/visual-exploration/)
- [Deep Reinforcement Learning Nanodegree – Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
- [Reinforcement Learning: State-of-the-Art (2019) – Arulkumaran et al.](https://arxiv.org/abs/1701.07274)
- [The RL Baselines3 Zoo – PyTorch Implementations of Popular RL Algorithms](https://github.com/DLR-RM/rl-baselines3-zoo)


## References
- [2] V. Mnih et al., “Asynchronous Methods for Deep Reinforcement Learning,” ICML, 2016.
- [3] J. Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.
- [5] T. Lillicrap et al., “Continuous Control with Deep Reinforcement Learning,” ICLR, 2016.
- [7] T. Haarnoja et al., “Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL,” ICML, 2018.
- [8] V. Mnih et al., “Playing Atari with Deep Reinforcement Learning,” NIPS Deep Learning Workshop, 2013.
- [9] M. Bellemare et al., “A Distributional Perspective on Reinforcement Learning,” ICML, 2017.
- [12] D. Ha and J. Schmidhuber, “World Models,” arXiv:1803.10122, 2018.
- [13] T. Weber et al., “Imagination-Augmented Agents,” NIPS, 2017.
- [14] A. Nagabandi et al., “Neural Network Dynamics for Model-Based Deep RL,” CoRL, 2017.
- [16] D. Silver et al., “Mastering the Game of Go without Human Knowledge,” Nature, 2017.
