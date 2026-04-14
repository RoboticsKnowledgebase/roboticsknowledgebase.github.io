---
date: 2024-12-06
title: Introduction to Diffusion Models and Diffusion Policy
---

Diffusion models have emerged as a powerful class of generative models that have revolutionized image generation, achieving state-of-the-art results in tasks ranging from image synthesis to text-to-image generation. In robotics, diffusion models have been adapted to create **diffusion policies**, which represent a paradigm shift in how robots learn and execute complex manipulation and navigation tasks. Unlike traditional policy learning methods that output deterministic or simple stochastic actions, diffusion policies model the action distribution as a denoising process, allowing robots to capture multi-modal behaviors and handle ambiguous situations more effectively. This article provides a comprehensive introduction to diffusion models, their mathematical foundations through both Ordinary Differential Equation (ODE) and Stochastic Differential Equation (SDE) formulations, and their practical application in robotics through diffusion policies.

## Fundamentals of Diffusion Models

Diffusion models are generative models that learn to generate data by reversing a gradual noising process. The core idea is inspired by non-equilibrium thermodynamics: starting from a simple noise distribution (typically Gaussian), the model learns to iteratively denoise the data until it produces a realistic sample from the target distribution.

### The Forward Diffusion Process

The forward diffusion process gradually adds Gaussian noise to data over a series of timesteps. Given an initial data point $\mathbf{x}_0$ sampled from the data distribution $q(\mathbf{x}_0)$, we define a forward process that produces a sequence of increasingly noisy versions $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$, where $T$ is the total number of diffusion steps.

The forward process is defined as:

$$q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})$$

where each step adds Gaussian noise:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

Here, $\beta_t$ is a variance schedule that controls how much noise is added at each timestep $t$. Typically, $\beta_t$ increases with $t$, meaning more noise is added as we progress through the diffusion process.

A key insight is that we can sample $\mathbf{x}_t$ directly from $\mathbf{x}_0$ without going through all intermediate steps:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. This allows efficient training by sampling random timesteps.

### The Reverse Diffusion Process

The reverse process is what we learn during training. We want to model $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$, which reverses the forward process. The reverse process is parameterized by a neural network $\theta$:

$$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$

where $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ is the prior distribution (pure noise).

The reverse step is also Gaussian:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

### Training Objective

The training objective for diffusion models is derived from variational inference. The key insight is to predict the noise $\boldsymbol{\epsilon}$ that was added to $\mathbf{x}_0$ to obtain $\mathbf{x}_t$, rather than predicting $\mathbf{x}_{t-1}$ directly.

Given that $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the training loss becomes:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$$

where $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ is a neural network that predicts the noise given the noisy sample $\mathbf{x}_t$ and timestep $t$.

## ODE Formulation: Deterministic Sampling

The Ordinary Differential Equation (ODE) formulation provides a deterministic view of the diffusion process, enabling faster and more stable sampling. This formulation is based on the observation that the diffusion process can be described as a continuous-time ODE.

### Probability Flow ODE

In the continuous-time limit, the forward diffusion process can be written as a stochastic differential equation. However, there exists a corresponding deterministic ODE, called the **Probability Flow ODE**, that has the same marginal distributions:

$$\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)\left[\mathbf{x} + \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]$$

where $\beta(t)$ is the continuous-time noise schedule and $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is the score function.

The score function can be approximated using the learned noise predictor:

$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

This leads to the practical ODE:

$$\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)\left[\mathbf{x} - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}\right]$$

### Advantages of ODE Formulation

The ODE formulation offers several practical advantages:

1. **Deterministic Sampling**: Given the same initial noise, the ODE produces the same output, enabling reproducible results.

2. **Faster Sampling**: ODE solvers can use adaptive step sizes and higher-order methods, allowing fewer function evaluations than the standard discrete diffusion steps.

3. **Exact Likelihood Computation**: The ODE formulation enables exact likelihood computation through the change of variables formula, useful for model evaluation and likelihood-based training.

4. **Latent Space Interpolation**: The deterministic nature allows smooth interpolation in the latent space, useful for generating intermediate samples.

### Numerical Integration Methods

Common ODE solvers used with diffusion models include:

- **Euler Method**: Simple first-order method, $\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \frac{d\mathbf{x}}{dt}$
- **Heun's Method**: Second-order Runge-Kutta method for better accuracy
- **DPM-Solver**: Specialized solver designed for diffusion models that can achieve high-quality samples with very few steps (e.g., 10-20 steps instead of 1000)

## SDE Formulation: Stochastic Sampling

The Stochastic Differential Equation (SDE) formulation provides a more general framework that encompasses both the forward and reverse diffusion processes as continuous-time stochastic processes.

### Forward SDE

The forward diffusion process can be written as a forward SDE:

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

where:
- $\mathbf{f}(\mathbf{x}, t)$ is the drift coefficient
- $g(t)$ is the diffusion coefficient  
- $d\mathbf{w}$ is a Wiener process (Brownian motion)

For the variance-preserving diffusion process:

$$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}, \quad g(t) = \sqrt{\beta(t)}$$

### Reverse SDE

The reverse-time SDE that generates samples is:

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + g(t)d\bar{\mathbf{w}}$$

where $d\bar{\mathbf{w}}$ is a reverse-time Wiener process. The key term is $g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, which corrects for the drift to reverse the diffusion process.

### Variance-Exploding vs Variance-Preserving

There are two main classes of diffusion SDEs:

**Variance-Preserving (VP) SDE**: The variance of the noise stays bounded. This is the most common formulation:

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$$

**Variance-Exploding (VE) SDE**: The variance grows unbounded:

$$d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}d\mathbf{w}$$

where $\sigma(t)$ is a time-dependent noise scale.

### Advantages of SDE Formulation

1. **Theoretical Rigor**: Provides a complete mathematical framework connecting forward and reverse processes
2. **Flexibility**: Can model different noise schedules and diffusion types
3. **Stochasticity**: The Brownian motion term introduces randomness, which can help explore the data distribution
4. **Unified Framework**: Connects diffusion models to other generative models (e.g., score-based models)

## Diffusion Policy for Robotics

Diffusion policies adapt the diffusion model framework to learn action distributions for robotic control. Instead of generating images, the model generates action sequences that the robot should execute.

### Problem Formulation

In imitation learning, we have a dataset $\mathcal{D} = \{(\mathbf{o}_i, \mathbf{a}_i)\}$ of observation-action pairs. Traditional methods learn a deterministic policy $\pi(\mathbf{a} | \mathbf{o})$ or a simple Gaussian policy. Diffusion policies model the action distribution as:

$$\pi(\mathbf{a} | \mathbf{o}) = \int p(\mathbf{a}_T) \prod_{t=1}^{T} p_\theta(\mathbf{a}_{t-1} | \mathbf{a}_t, \mathbf{o}) d\mathbf{a}_{1:T}$$

where $\mathbf{a}$ is the action and $\mathbf{o}$ is the observation (e.g., camera image, proprioceptive state).

### Architecture

A diffusion policy network takes as input:
- **Observation** $\mathbf{o}$: Current robot state and sensor readings
- **Noisy action** $\mathbf{a}_t$: The action at diffusion timestep $t$
- **Timestep** $t$: The current diffusion step

The network outputs a prediction of the noise $\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, \mathbf{o}, t)$ that was added to the clean action.

### Action Sequence Generation

For robotics, we often need to generate action sequences rather than single actions. The diffusion policy can generate a sequence of $H$ actions $\mathbf{a}^{0:H-1}$:

1. Sample initial noise: $\mathbf{a}_T^{0:H-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. For $t = T$ down to $1$:
   - Predict noise: $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{a}_t^{0:H-1}, \mathbf{o}, t)$
   - Denoise: $\mathbf{a}_{t-1}^{0:H-1} = \text{denoise}(\mathbf{a}_t^{0:H-1}, \hat{\boldsymbol{\epsilon}}, t)$
3. Return $\mathbf{a}_0^{0:H-1}$ as the action sequence

The robot executes the first action $\mathbf{a}_0^0$, then re-plans at the next timestep (receding horizon control).

### Key Advantages for Robotics

1. **Multi-Modal Action Distributions**: Unlike deterministic policies, diffusion policies can represent multiple valid actions for the same observation, crucial when there are multiple ways to accomplish a task.

2. **Temporal Consistency**: By generating action sequences, the policy naturally maintains temporal smoothness, important for stable robot control.

3. **Robustness**: The iterative denoising process can recover from poor initial samples, making the policy more robust to distribution shift.

4. **Data Efficiency**: Can learn complex behaviors from relatively small demonstration datasets.

## Practical Implications: ODE vs SDE

The choice between ODE and SDE formulations has significant practical implications for deploying diffusion policies in robotics.

### Sampling Speed

**ODE Formulation**:
- Typically requires 10-50 function evaluations for high-quality samples
- Deterministic, reproducible results
- Faster for real-time control applications
- Can use advanced ODE solvers (DPM-Solver, DPM-Solver++) for even fewer steps

**SDE Formulation**:
- Usually requires 50-1000 steps for good quality
- Stochastic sampling introduces variability
- Slower, but can explore the distribution better
- More suitable for offline planning or when sampling speed is not critical

### Sample Quality and Diversity

**ODE Formulation**:
- Produces high-quality, deterministic samples
- Less diversity due to deterministic nature
- Better for tasks requiring precise, repeatable actions
- Suitable for safety-critical applications where consistency matters

**SDE Formulation**:
- Can produce more diverse samples due to stochasticity
- Better exploration of the action distribution
- Useful when multiple valid action modes exist
- May introduce unwanted variability in robot behavior

### Computational Requirements

**ODE Formulation**:
- Lower computational cost per sample (fewer steps)
- Can leverage efficient ODE solvers
- Better suited for on-robot deployment with limited compute
- Enables real-time control at higher frequencies

**SDE Formulation**:
- Higher computational cost (more steps typically needed)
- May require GPU acceleration for real-time performance
- Better suited for offline planning or simulation

### Implementation Considerations

For robotics applications, consider the following:

1. **Real-Time Constraints**: If the robot needs to make decisions at high frequency (e.g., 10-30 Hz), ODE formulation with fast solvers is preferable.

2. **Action Diversity Needs**: If the task benefits from exploring multiple action modes (e.g., manipulation with multiple grasp strategies), SDE formulation may be better.

3. **Hardware Limitations**: Resource-constrained robots should prefer ODE formulation for its efficiency.

4. **Safety Requirements**: Deterministic ODE sampling provides more predictable behavior, important for safety-critical applications.

### Hybrid Approaches

Many practical implementations use hybrid approaches:

- **Training**: Use SDE formulation for better coverage of the action distribution
- **Inference**: Use ODE formulation for faster, deterministic sampling
- **Adaptive Sampling**: Start with ODE for fast initial samples, use SDE refinement when needed

## Training Diffusion Policies

Training a diffusion policy follows similar principles to training image diffusion models, with modifications for the robotics domain.

### Dataset Preparation

The training dataset consists of:
- **Observations**: $\mathbf{o}_i$ - sensor readings, images, proprioceptive state
- **Actions**: $\mathbf{a}_i$ - demonstrated actions (joint velocities, end-effector poses, etc.)
- **Action Sequences**: For sequence-based policies, include action sequences $\mathbf{a}_i^{0:H-1}$

### Loss Function

The training loss is similar to standard diffusion models:

$$\mathcal{L} = \mathbb{E}_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}, t \sim \mathcal{U}(1,T), \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, \mathbf{o}, t)\|^2 \right]$$

where $\mathbf{a}_t = \sqrt{\bar{\alpha}_t}\mathbf{a} + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ is the noisy action.

### Observation Encoding

The observation $\mathbf{o}$ needs to be encoded into a representation that the diffusion model can use. Common approaches:

- **CNN Encoders**: For image observations
- **Transformer Encoders**: For multi-modal observations or sequences
- **Proprioceptive Encoders**: MLPs for joint positions, velocities, etc.

The encoded observation is concatenated with the noisy action and timestep embedding before being fed to the diffusion model.

### Conditioning Strategies

Different conditioning strategies affect how observations influence action generation:

1. **Concatenation**: Simply concatenate observation features with noisy actions
2. **Cross-Attention**: Use attention mechanisms to condition on observations
3. **FiLM (Feature-wise Linear Modulation)**: Modulate intermediate features based on observations

## Applications in Robotics

Diffusion policies have shown success in various robotic domains:

### Manipulation Tasks

- **Grasping**: Learning diverse grasping strategies from demonstrations
- **Assembly**: Complex manipulation sequences for assembly tasks
- **Tool Use**: Using tools with multiple valid manipulation modes

### Mobile Robotics

- **Navigation**: Planning paths with multiple valid routes
- **Obstacle Avoidance**: Generating smooth avoidance trajectories

### Human-Robot Interaction

- **Handover Tasks**: Adapting to different handover scenarios
- **Collaborative Manipulation**: Coordinating with human partners

## Challenges and Limitations

While diffusion policies offer significant advantages, they also face challenges:

1. **Computational Cost**: Even with ODE solvers, diffusion policies require multiple forward passes, making them slower than simple policy networks.

2. **Hyperparameter Sensitivity**: The noise schedule $\beta_t$ and number of steps $T$ significantly affect performance and need careful tuning.

3. **Distribution Shift**: Like all imitation learning methods, performance degrades when test conditions differ from training.

4. **Sequential Inference**: Generating action sequences requires careful handling of observation updates during execution.

## Summary

Diffusion models represent a powerful paradigm for generative modeling that has been successfully adapted to robotics through diffusion policies. The mathematical foundations can be understood through both ODE and SDE formulations, each offering distinct advantages. The ODE formulation provides deterministic, fast sampling suitable for real-time control, while the SDE formulation offers a more complete theoretical framework and better exploration capabilities. In practice, the choice between formulations depends on the specific requirements of the robotic application, including real-time constraints, need for action diversity, and computational resources available. Diffusion policies excel at learning multi-modal action distributions from demonstrations, making them particularly valuable for complex manipulation and navigation tasks where multiple valid solutions exist.

## See Also:

- [Introduction to Reinforcement Learning](/wiki/machine-learning/intro-to-rl/) - Alternative approaches to learning robot policies
- [GRPO for Diffusion Policies in Robotics](/wiki/machine-learning/grpo-diffusion-policies/) - Using reinforcement learning to optimize diffusion policies with reward-based learning
- [NLP for Robotics](/wiki/machine-learning/nlp_for_robotics/) - Other applications of modern ML in robotics

## Further Reading

- **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** by Ho, Jain, and Abbeel (2020). The foundational paper that introduced DDPMs, establishing the core framework for diffusion models. This paper presents the key insight of modeling data generation as a reverse denoising process and demonstrates state-of-the-art image generation results. Essential reading for understanding the basic principles underlying all diffusion models.

- **[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)** by Song, Meng, and Ermon (2021). Introduces DDIMs, which enable faster sampling (10-50x speedup) by using non-Markovian diffusion processes while maintaining the same training procedure as DDPMs. This paper is particularly relevant for understanding how ODE formulations enable efficient sampling, as discussed in this article's ODE section.

- **[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)** by Song et al. (2021). Provides the comprehensive SDE formulation framework for diffusion models, connecting forward and reverse processes through stochastic differential equations. This paper is essential for understanding the theoretical foundations of the SDE formulation covered in this article, including variance-preserving and variance-exploding SDEs.

- **[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://roboticsproceedings.org/rss19/p026.html)** by Chi et al. (2023). The original paper introducing diffusion policies for robotics, demonstrating how diffusion models can be adapted for action generation. This paper shows 46.9% average improvement over existing methods and introduces key techniques like receding horizon control and visual conditioning. Directly relevant to the diffusion policy sections of this article.

- **[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)** by Lu et al. (2022). Introduces DPM-Solver, a high-order ODE solver that enables high-quality sampling with only 10-20 function evaluations. This paper is crucial for understanding practical ODE-based sampling methods and demonstrates the speed advantages of ODE formulations discussed in the practical implications section.

## References

Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. *Robotics: Science and Systems*.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.

Lu, C., et al. (2022). DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. *Advances in Neural Information Processing Systems*, 35, 5775-5787.

Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations*.
