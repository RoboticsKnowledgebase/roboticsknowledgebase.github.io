---
date: 2024-12-09
title: GRPO for Diffusion Policies in Robotics
---

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that optimizes generative models by leveraging group-based reward normalization, eliminating the need for separate value function networks. Originally developed for fine-tuning large language models (LLMs) to align with human preferences, GRPO has been successfully adapted to diffusion models for visual generation tasks and, more recently, to diffusion policies in robotics. This article introduces GRPO, explains its mathematical foundations, and demonstrates how it can be applied to diffusion policies using the Stochastic Differential Equation (SDE) formulation to enable effective policy optimization through stochastic sampling.

## GRPO Origins in Large Language Models

GRPO was originally developed to address challenges in aligning LLMs with human preferences through reinforcement learning. Traditional methods like Proximal Policy Optimization (PPO) require training a separate critic network to estimate value functions, which increases computational overhead and introduces additional sources of variance and instability.

### The Problem with Traditional RLHF

Reinforcement Learning from Human Feedback (RLHF) typically involves:
1. **Reward Modeling**: Training a reward model on human preference data
2. **Policy Optimization**: Using RL algorithms like PPO to optimize the language model policy
3. **Value Function Estimation**: Training a critic network to estimate state values

The critic network in step 3 requires significant memory and computational resources, and its estimation errors can propagate through the policy updates, leading to training instability.

### GRPO's Solution

GRPO eliminates the need for a separate value function by using group-based advantage estimation. For each input prompt, GRPO samples multiple candidate outputs from the current policy, evaluates their rewards, and normalizes these rewards within the group to compute relative advantages. This approach leverages the group statistics as a natural baseline, reducing variance without requiring a learned value function.

## GRPO Fundamentals

### Group-Based Advantage Estimation

Given an input prompt $q$, GRPO samples $G$ outputs $\{o_i\}_{i=1}^{G}$ from the current policy $\pi_{\theta_{\text{old}}}$. Each output $o_i$ receives a reward $r_i$ based on task-specific criteria (e.g., human preference scores, task completion metrics, or learned reward models).

The advantage for each output is computed using group statistics:

$$A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

where:
- $\mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j$ is the mean reward of the group
- $\sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}$ is the standard deviation of group rewards
- $\epsilon$ is a small constant for numerical stability (typically $10^{-8}$)

This normalization ensures that advantages are centered around zero and scaled appropriately, providing stable gradient signals for policy updates.

### Policy Update Objective

The policy is updated using a clipped surrogate loss similar to PPO:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{i \sim [1,G]} \left[ \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) \right]$$

where:
- $\pi_\theta(o_i|q)$ is the probability of output $o_i$ under the updated policy
- $\pi_{\theta_{\text{old}}}(o_i|q)$ is the probability under the old policy
- The clipping mechanism prevents large policy updates that could destabilize training

### Key Advantages

1. **No Value Function Required**: Eliminates the need for a separate critic network, reducing memory usage and computational cost
2. **Reduced Variance**: Group-based normalization provides a natural baseline, reducing variance in advantage estimates
3. **Simplified Training**: Fewer components to tune and maintain
4. **Scalability**: More memory-efficient, enabling training on resource-constrained hardware

## Applying GRPO to Diffusion Models

Applying GRPO to diffusion models requires addressing a fundamental challenge: diffusion models are typically trained to generate samples through a deterministic or near-deterministic process, but GRPO requires stochastic sampling to generate diverse candidate outputs for group-based evaluation.

### The Stochasticity Requirement

GRPO needs to sample multiple diverse outputs from the policy for each input. In language models, this is straightforward—the policy outputs a probability distribution over tokens, and sampling naturally produces diverse outputs. However, diffusion models often use deterministic ODE solvers for efficient sampling, which produce the same output given the same initial noise.

### SDE Formulation for Stochastic Sampling

The solution lies in using the Stochastic Differential Equation (SDE) formulation of diffusion models, which naturally incorporates stochasticity through the Brownian motion term. Recall from the diffusion model formulation that the reverse SDE is:

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + g(t)d\bar{\mathbf{w}}$$

where $d\bar{\mathbf{w}}$ is a reverse-time Wiener process that introduces stochasticity at each step.

For variance-preserving diffusion processes:

$$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}, \quad g(t) = \sqrt{\beta(t)}$$

The stochastic term $g(t)d\bar{\mathbf{w}}$ ensures that different samples from the same initial condition will follow different trajectories, producing diverse outputs suitable for GRPO's group-based evaluation.

### Discrete-Time SDE Sampling

In practice, we discretize the SDE using numerical methods. The Euler-Maruyama method provides a simple discretization:

$$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \left[\mathbf{f}(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)\right] + g(t)\sqrt{\Delta t} \cdot \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is sampled independently at each step, introducing the necessary stochasticity.

## GRPO for Diffusion Policies in Robotics

In robotics, diffusion policies model the action distribution as a denoising process. Applying GRPO to diffusion policies allows us to optimize these policies based on task-specific rewards, such as task completion, efficiency, or safety metrics.

### Problem Setup

Given a dataset of observation-action pairs $\mathcal{D} = \{(\mathbf{o}_i, \mathbf{a}_i)\}$, a diffusion policy learns to generate actions $\mathbf{a}$ conditioned on observations $\mathbf{o}$. GRPO extends this by optimizing the policy to maximize expected rewards:

$$\max_\theta \mathbb{E}_{\mathbf{o} \sim p(\mathbf{o}), \mathbf{a} \sim \pi_\theta(\cdot|\mathbf{o})} [R(\mathbf{o}, \mathbf{a})]$$

where $R(\mathbf{o}, \mathbf{a})$ is a reward function evaluating the quality of action $\mathbf{a}$ given observation $\mathbf{o}$.

### Group Sampling with SDE

For each observation $\mathbf{o}$, GRPO samples $G$ action sequences $\{\mathbf{a}_i^{0:H-1}\}_{i=1}^{G}$ using SDE-based sampling:

1. **Initialize**: Sample $G$ independent noise vectors $\{\mathbf{a}_{T,i}^{0:H-1}\}_{i=1}^{G} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

2. **Reverse SDE Process**: For each sample $i$ and timestep $t = T$ down to $1$:
   - Predict score: $\hat{\boldsymbol{\epsilon}}_i = \boldsymbol{\epsilon}_\theta(\mathbf{a}_{t,i}^{0:H-1}, \mathbf{o}, t)$
   - Sample noise: $\boldsymbol{\epsilon}_{\text{step}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
   - Update: $\mathbf{a}_{t-1,i}^{0:H-1} = \mathbf{a}_{t,i}^{0:H-1} + \Delta t \cdot \text{drift} + g(t)\sqrt{\Delta t} \cdot \boldsymbol{\epsilon}_{\text{step}}$

3. **Evaluate**: Execute each action sequence (or simulate) and compute rewards $\{r_i\}_{i=1}^{G}$

4. **Compute Advantages**: Normalize rewards within the group:
   $$A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

5. **Update Policy**: Update $\theta$ using the GRPO loss with computed advantages

### Reward Functions for Robotics

Common reward functions for robotic tasks include:

- **Task Completion**: Binary or continuous reward based on whether the task was completed successfully
- **Efficiency**: Negative of execution time or energy consumption
- **Safety**: Penalties for collisions, joint limit violations, or unsafe configurations
- **Trajectory Quality**: Smoothness, precision, or similarity to demonstrations
- **Multi-Objective**: Weighted combinations of the above

### Action Sequence Evaluation

Since diffusion policies generate action sequences $\mathbf{a}^{0:H-1}$, we need to evaluate the entire sequence. Options include:

1. **Full Sequence Reward**: Execute the entire sequence and compute a single reward
2. **Cumulative Reward**: Sum rewards from executing each action in sequence
3. **Final State Reward**: Reward based only on the final state reached

The choice depends on the task structure and whether intermediate rewards are meaningful.

## Practical Implementation Considerations

### Sampling Efficiency

SDE-based sampling requires more steps than ODE sampling for comparable quality, increasing computational cost. Strategies to mitigate this:

1. **Adaptive Step Sizes**: Use larger steps in early diffusion stages where noise is high, smaller steps near the end
2. **Reduced Group Size**: Use smaller groups ($G=4$ to $G=8$) for initial experiments, increase if needed
3. **Parallel Sampling**: Generate group samples in parallel on GPU

### Training Stability

GRPO training can be unstable, especially early in training when the policy distribution changes rapidly. Techniques to improve stability:

1. **Learning Rate Scheduling**: Use lower learning rates or cosine annealing
2. **Gradient Clipping**: Clip gradients to prevent large updates
3. **KL Regularization**: Add a KL divergence term to prevent the policy from deviating too far from a reference policy
4. **Warm-up Period**: Start with supervised learning, gradually transition to GRPO

### Reward Shaping

Well-shaped rewards are crucial for GRPO success:

1. **Normalization**: Ensure rewards are on a similar scale across different task components
2. **Dense Rewards**: Provide intermediate rewards when possible, not just final outcomes
3. **Reward Validation**: Monitor reward distributions to detect issues (e.g., all rewards near zero or one)

### Integration with Existing Pipelines

GRPO can be integrated into existing diffusion policy training pipelines:

1. **Pre-training**: Train diffusion policy on demonstration data using standard maximum likelihood objective
2. **Fine-tuning**: Apply GRPO to optimize for task-specific rewards
3. **Hybrid Training**: Alternate between supervised learning and GRPO updates

## DanceGRPO: A Reference Implementation

DanceGRPO demonstrates GRPO's application to diffusion models for visual generation (image and video synthesis). While focused on visual tasks, the mathematical framework and implementation strategies directly translate to diffusion policies in robotics.

### Key Insights from DanceGRPO

1. **Unified Framework**: The same GRPO formulation works across different generative paradigms (diffusion models, rectified flows) by ensuring stochastic sampling through SDE formulation

2. **Stability Techniques**: DanceGRPO employs several techniques to ensure stable training:
   - Careful reward normalization
   - Gradient reweighting strategies
   - Ratio normalization to prevent over-optimization

3. **Scalability**: The framework scales to large models (e.g., Stable Diffusion, FLUX) and diverse reward models (aesthetics, alignment, motion quality)

4. **Performance Gains**: Significant improvements over baselines (up to 181% on some benchmarks), demonstrating GRPO's effectiveness

### Adapting to Robotics

The same principles apply to robotics:

- **Stochastic Sampling**: Use SDE formulation to generate diverse action sequences
- **Reward Design**: Adapt visual rewards (aesthetics, alignment) to robotic rewards (task completion, efficiency, safety)
- **Stability**: Apply similar normalization and gradient techniques
- **Evaluation**: Use task-specific metrics instead of visual quality metrics

## Challenges and Solutions

### Challenge 1: Computational Cost

**Problem**: SDE sampling requires more function evaluations than ODE sampling, and GRPO requires sampling $G$ candidates per update.

**Solutions**:
- Use smaller group sizes ($G=4$ to $G=8$) for initial experiments
- Implement efficient parallel sampling on GPU
- Consider hybrid approaches (ODE for inference, SDE for training)

### Challenge 2: Reward Design

**Problem**: Poorly designed rewards can lead to reward hacking or unstable training.

**Solutions**:
- Start with simple, well-understood rewards (e.g., task completion)
- Validate reward distributions during training
- Use reward normalization and shaping techniques

### Challenge 3: Sample Diversity

**Problem**: If SDE sampling doesn't produce sufficiently diverse samples, group-based advantages become unreliable.

**Solutions**:
- Ensure proper noise injection at each SDE step
- Use appropriate noise schedules $\beta(t)$
- Monitor sample diversity metrics during training

### Challenge 4: Distribution Shift

**Problem**: Optimized policy may deviate significantly from demonstration distribution, losing important behaviors.

**Solutions**:
- Add KL regularization to reference policy
- Use hybrid training (alternate supervised and GRPO updates)
- Monitor policy distribution statistics

## Alternative Approaches

Several alternatives to pure SDE-based GRPO have been developed:

### Neighbor GRPO

Generates diverse candidates by perturbing initial noise conditions of ODE samplers, maintaining ODE efficiency while enabling group-based optimization. Useful when deterministic sampling is preferred but diversity is needed.

### MixGRPO

Uses SDE sampling within a specific time window (typically near the end of diffusion) and ODE sampling elsewhere. Balances stochasticity needs with computational efficiency.

### Direct Group Preference Optimization (DGPO)

Learns directly from group-level preferences without requiring stochastic policies, allowing efficient deterministic ODE samplers. May be suitable when preferences are available instead of scalar rewards.

## Summary

GRPO provides a powerful framework for optimizing diffusion policies in robotics through reinforcement learning. By leveraging group-based reward normalization, GRPO eliminates the need for separate value functions, simplifying training and reducing computational overhead. The key to applying GRPO to diffusion policies lies in using the SDE formulation, which naturally provides the stochasticity needed to generate diverse candidate actions for group-based evaluation. While SDE sampling is computationally more expensive than ODE sampling, the benefits of reward-based optimization often justify the cost. Practical implementations should carefully consider reward design, training stability, and computational efficiency. The success of DanceGRPO in visual generation demonstrates the viability of this approach, and the same principles translate directly to robotic applications where we optimize policies for task-specific rewards like completion, efficiency, and safety.

## See Also:

- [Introduction to Diffusion Models and Diffusion Policy](/wiki/machine-learning/intro-to-diffusion/) - Foundational concepts on diffusion models and their application to robotics
- [Introduction to Reinforcement Learning](/wiki/machine-learning/intro-to-rl/) - Background on policy gradient methods and RL fundamentals

## Further Reading

- **[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)** - The original DanceGRPO paper demonstrating GRPO's application to diffusion models for image and video generation. Provides detailed implementation strategies and experimental results that translate directly to robotics applications.

- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)** - Introduces GRPO as a variant of PPO for optimizing language models, demonstrating its effectiveness for mathematical reasoning. Essential reading for understanding the core GRPO algorithm and its advantages over traditional RLHF methods with value functions.

- **[Neighbor GRPO: Deterministic Policy Optimization with Group Relative Advantage](https://arxiv.org/abs/2511.16955)** - Introduces Neighbor GRPO, which generates diverse candidates through ODE noise perturbation rather than SDE sampling. Relevant for understanding alternatives when computational efficiency is critical.

- **[MixGRPO: Mixed Sampling for Group Relative Policy Optimization](https://arxiv.org/abs/2507.21802)** - Presents a hybrid approach using both SDE and ODE sampling. Useful for understanding how to balance stochasticity needs with computational efficiency in practical deployments.

- **[Direct Group Preference Optimization](https://arxiv.org/abs/2510.08425)** - Introduces DGPO, which optimizes directly from group preferences without requiring stochastic policies. Relevant when preference data is available instead of scalar rewards.

## References

Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. *Robotics: Science and Systems*.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations*.
