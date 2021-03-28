---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-12-06 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Introduction to Reinforcement Learning
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.

---
The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relative simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards.

## Key Concepts

### Bellman Equations

Bellman equations refer to a set of equations that decompose the value function into the immediate reward plus the discounted future values.
$$\begin{aligned}
  V(s) &= \mathbb{E}[G_t | S_t = s]\\
   &= \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})|S_t = s)]\\
 Q(s,a)&=\mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s, A_t = a]
\end{aligned}$$

#### Bellman Expectation Equations

$$\begin{aligned}
  V_{\pi}(s) &= \sum_a \pi(a|s)\sum_{s',r} p(s', r | s, a)[r + \gamma V_{\pi}(s')]\\
  Q_\pi(s, a) &= \sum_{s'}\sum_{r}p(s', r | s, a)[r +\gamma\sum_{a'}\pi(a', s')Q_\pi(s', a')]
\end{aligned}
$$

#### Bellman Optimality Equations

$$\begin{aligned}
  V_*(s) &= \max_{a}\sum_{s'}\sum_{r}p(s', r | s, a)[r + \gamma V_*(s')]\\
  Q_*(s,a) &= \sum_{s'}\sum_{r}p(s', r | s, a)[r +\gamma\max_{a'}Q_*(s', a')]
\end{aligned} $$

## Approaches

### Dynamic Programming

When the model of the environment is known, following Bellman equations, we can use Dynamic Programming (DP) to iteratively evaluate value functions and improve policy.

#### Policy Evaluation

$$
V_{t+1} = \mathbb{E}[r+\gamma V_t(s') | S_t = s] = \sum_a\pi(a|s)\sum_{s', r}p(s', r|s,a)(r+\gamma V_t(s'))
$$

#### Policy Improvement

Given a policy and its value function, we can easily evaluate a change in the policy at a single state to a particular action. It is a natural extension to consider changes at all states and to all possible actions, selecting at each state the action that appears best according to $q_{\pi}(s,a).$ In other words, we make a new policy by acting greedily.

$$
Q_\pi(s, a) = \mathbb{E}[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t = s, A_t = a] = \sum_{s', r} p(s', r|s, a)(r+\gamma V_\pi (s'))
$$

#### Policy Iteration

Once a policy, $\pi$, has been improved using $V_{\pi}$ to yield a better policy, $\pi'$, we can then compute $V_{\pi}'$ and improve it again to yield an even better $\pi''$. We can thus obtain a sequence of monotonically improving policies and value functions:
$$\pi_0 \xrightarrow{E}V_{\pi_0}\xrightarrow{I}\pi_1 \xrightarrow{E}V_{\pi_1}\xrightarrow{I}\pi_2 \xrightarrow{E}\dots\xrightarrow{I}\pi_*\xrightarrow{E}V_{\pi_*}$$
where $\xrightarrow{E}$ denotes a policy evaluation and $\xrightarrow{I}$ denotes a policy improvement.

### Monte-Carlo Methods
Monte-Carlo (MC) methods require only experience --- sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. It learns from actual experience without no prior knowledge of the environment's dynamics. To compute the empirical return $G_t$, MC methods need to learn complete episodes $S_1, A_1, R_2, \dots, S_T$ to compute $G_t = \sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}$ and all the episodes must eventually terminate no matter what actions are selected.

The empirical mean return for state $s$ is:
$$V(s)=\frac{\sum_{t=1}^T\mathbf{1}[S_t=s]G_t}{\sum_{t=1}^T\mathbf{1}[S_t = s]}$$
Each occurrence of state $s$ in an episode is called a visit to $s$. We may count the visit of state $s$ every time so that there could exist multiple visits of one state in one episode ("every-visit"), or only count it the first time we encounter a state in one episode ("first-visit"). In practical, first-visit MC converges faster with lower average root mean squared error. A intuitive explanation is that it ignores data from other visits to $s$ after the first, which breaks the correlation between data resulting in unbiased estimate. 

This way of approximation can be easily extended to action-value functions by counting $(s, a)$ pair.
$$Q(s,a) = \frac{\sum_{t=1}^T\mathbf{1}[S_t = s, A_t = a]G_t}{\sum_{t=1}^T[S_t = s, A_t =a]}$$
To learn the optimal policy by MC, we iterate it by following a similar idea to Generalized Policy iteration (GPI).

1. Improve the policy greedily with respect to the current value function: $$\pi(s) = \arg\max_{a\in A}Q(s,a)$$

2. Generate a new episode with the new policy $\pi$ (i.e. using algorithms like $\epsilon$-greedy helps us balance between exploitation and exploration)

3. Estimate $Q$ using the new episode: $$q_\pi(s, a) = \frac{\sum_{t = 1}^T(\mathbf{1}[S_t = s, A_t = a]\sum_{k = 0}^{T-t-1}\gamma^kR_{t+k+1})}{\sum_{t=1}^T\mathbf{1}[S_t = s, A_t = a]}$$

### Temporal-Difference Learning

Temporal-difference (TD) learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).
Similar to Monte-Carlo methods, Temporal-Difference (TD) Learning is model-free and learns from episodes of experience. However, TD learning can learn from incomplete episodes.
$$Q(s, a) = R(s,a) + \gamma Q^\pi(s',a')$$

#### Comparison between MC and TD}

MC regresses $Q(s,a)$ with targets $y = \sum_i r(s_i, a_i)$. Each rollout has randomness due to stochasticity in policy and environment. Therefore, to estimate $Q(s,a)$, we need to generate many trajectories and average over such stochasticity, which is a high variance estimate. But it is unbiased meaning the return is the true target.

TD estimates $Q(s,s)$ with $y = r(s,a)+\gamma Q^\pi(s',a')$, where $Q^\pi(s',a')$ already accounts for stochasticity of future states and actions. Thus, the estimate has lower variance meaning it needs fewer samples to get a good estimate. But the estimate is biased: if $Q(s', a')$ has approximation errors, the target $y$ has approximation errors; this could lead to unstable training due to error propagation.

#### Bootstrapping

TD learning methods update targets in the following equation with regard to existing estimates rather than exclusively relying on actual rewards and complete returns as in MC methods in Equation (\ref{eq:9}). This approach is known as bootstrapping.
$$
\begin{aligned}
V(S_t) &\leftarrow V(S_t) +\alpha[G_t - V(S_t)]\\
V(S_t) &\leftarrow V(S_t) +\alpha[R_{t+1} +\gamma V(S_t) - V(S_t)]
\end{aligned}
$$

## Summary

Here are some simple methods used in Reinforcement Learning. There are a lot of fancy stuff, but due to limited pages, not included here. Feel free to update the wiki to keep track of the latest algorithms of RL.

## See Also:

<https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>

## Further Reading

- Introduction to Reinforcement Learning, MIT Press

## References

Kaelbling, Leslie Pack, Michael L. Littman, and Andrew W. Moore. "Reinforcement learning: A survey." Journal of artificial intelligence research 4 (1996): 237-285.
