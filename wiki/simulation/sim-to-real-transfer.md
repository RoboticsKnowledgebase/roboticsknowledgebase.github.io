---
date: 2026-04-30
title: "Sim-to-Real Transfer: Domain Randomization and Adaptation Techniques"
---
Sim-to-real transfer is the process of training robotic policies in simulation and deploying them on physical hardware. Simulation offers unlimited data, safe exploration, and rapid iteration, but a persistent gap between simulated and real-world dynamics causes policies that perform well in simulation to fail on real robots. This article covers the primary sources of the reality gap, the two dominant families of techniques for bridging it — domain randomization and domain adaptation — and provides practical implementation guidance with code examples using NVIDIA Isaac Gym and Isaac Sim. Whether you are training a locomotion controller for a quadruped or a manipulation policy for a robotic arm, these techniques are essential for reliable real-world deployment.

## The Reality Gap

The reality gap refers to the systematic discrepancies between a simulated environment and the physical world. Understanding these discrepancies is the first step toward closing them. The gap arises from four primary sources:

### Visual Discrepancies

Rendered images in simulation differ from real camera feeds in texture detail, lighting conditions, reflections, shadows, lens distortion, motion blur, and color balance. A perception model trained exclusively on synthetic images will often fail when confronted with real visual input.

### Dynamics Discrepancies

Physics engines approximate the real world using simplified contact models, rigid-body assumptions, and discrete-time integration. Parameters such as mass, center of gravity, friction coefficients, joint damping, and restitution are never perfectly known. Small errors in these values compound over long rollouts, causing trajectory divergence.

### Sensor Noise and Latency

Real sensors exhibit noise distributions, biases, quantization artifacts, and latency that are difficult to replicate exactly. An IMU in simulation may report perfect angular velocities, whereas the physical IMU adds gyroscope bias drift and accelerometer noise. Similarly, depth cameras produce noisy point clouds with missing data near reflective or transparent surfaces.

### Actuator Modeling

Real actuators have nonlinear torque-speed curves, backlash, gear friction, thermal effects, and communication delays between the controller and the motor driver. Simulation often models actuators as ideal torque or velocity sources, ignoring these effects.

> The reality gap is not a single problem but a collection of mismatches across perception, dynamics, sensing, and actuation. Effective sim-to-real transfer addresses multiple sources simultaneously.

## Domain Randomization

Domain randomization (DR) is the strategy of training a policy across a wide distribution of simulated environments so that the real world appears as just another sample from that distribution. Rather than building a single high-fidelity simulator, DR exposes the policy to enough variation that it learns to be robust to the specific conditions it will encounter on hardware.

### Mathematical Formulation

Let $\xi$ denote the vector of environment parameters (masses, frictions, visual properties, sensor noise levels, etc.). In standard reinforcement learning, a policy $\pi_\theta$ is trained in a single environment with fixed parameters $\xi_0$:

$$\theta^* = \arg\max_\theta \; \mathbb{E}_{\tau \sim \pi_\theta, \xi_0} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

With domain randomization, parameters are sampled from a distribution $P(\xi)$ at the start of each episode (or at each step), and the objective becomes:

$$\theta^* = \arg\max_\theta \; \mathbb{E}_{\xi \sim P(\xi)} \; \mathbb{E}_{\tau \sim \pi_\theta, \xi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

The key insight is that if the real-world parameters $\xi_{\text{real}}$ fall within the support of $P(\xi)$, the trained policy is likely to generalize. The distribution $P(\xi)$ is typically uniform or Gaussian over manually specified ranges.

### Visual Randomization

Visual domain randomization modifies the appearance of the simulated scene to prevent the perception module from overfitting to synthetic rendering artifacts. Common randomizations include:

- **Textures**: Random textures applied to objects, floors, walls, and the robot body.
- **Lighting**: Random number, position, intensity, and color of light sources. Random ambient light levels.
- **Camera intrinsics**: Small perturbations to focal length, principal point, and distortion coefficients.
- **Camera extrinsics**: Jitter in camera position and orientation relative to the nominal mount.
- **Distractors**: Random geometric shapes placed in the scene background.
- **Post-processing**: Random noise, blur, color jitter, and contrast adjustments applied to rendered images.

Visual randomization is particularly important for policies that consume raw image input. For policies that use only proprioceptive state (joint angles, velocities), visual randomization is unnecessary.

### Dynamics Randomization

Dynamics randomization perturbs the physical parameters of the simulation:

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Link masses | ±15–30% | Randomize each link independently |
| Center of mass offsets | ±1–3 cm | Per link, all three axes |
| Joint friction | ±20–50% | Coulomb and viscous components |
| Joint damping | ±20–50% | Often coupled with friction |
| Ground friction | 0.3–1.2 | Coefficient of friction |
| Restitution | 0.0–0.5 | Bounciness of contacts |
| Actuator strength | ±10–20% | Scales applied torques |
| Control latency | 0–20 ms | Simulates communication delay |
| Observation noise | Sensor-dependent | Gaussian noise on joint encoders, IMU |
| External disturbances | Task-dependent | Random forces/torques on the base |

### Practical Example: Domain Randomization in Isaac Gym

The following example demonstrates how to configure dynamics randomization for a quadruped locomotion task using NVIDIA Isaac Gym. The randomization parameters are specified in a configuration dictionary and applied at each environment reset.

```python
import isaacgym
from isaacgym import gymapi, gymutil
import numpy as np
import torch

class QuadrupedDREnv:
    """Quadruped locomotion environment with domain randomization."""

    def __init__(self, num_envs=4096, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # Define randomization ranges as (min, max) tuples
        self.randomization_params = {
            "base_mass_offset": (-1.0, 1.0),       # kg added to base link
            "link_mass_scale": (0.8, 1.2),          # multiplicative scale per link
            "friction_range": (0.3, 1.25),          # ground friction coefficient
            "restitution_range": (0.0, 0.4),        # ground restitution
            "joint_damping_scale": (0.8, 1.3),      # scale factor for joint damping
            "joint_friction": (0.0, 0.05),           # joint-level Coulomb friction (Nm)
            "actuator_strength_scale": (0.85, 1.15), # scale factor on torque commands
            "control_latency_steps": (0, 3),         # discrete steps of added latency
            "observation_noise": {
                "joint_pos": 0.01,    # rad standard deviation
                "joint_vel": 0.15,    # rad/s standard deviation
                "imu_gyro": 0.02,     # rad/s standard deviation
                "imu_accel": 0.05,    # m/s^2 standard deviation
            },
            "push_force_range": (0.0, 30.0),  # random external push force (N)
        }

    def randomize_environment(self, env_ids):
        """Apply domain randomization to specified environments on reset."""
        n = len(env_ids)

        # Randomize base mass by adding an offset
        mass_offsets = torch.uniform(
            self.randomization_params["base_mass_offset"][0],
            self.randomization_params["base_mass_offset"][1],
            size=(n,), device=self.device
        )
        self.apply_mass_offsets(env_ids, mass_offsets)

        # Randomize ground friction per environment
        frictions = torch.uniform(
            self.randomization_params["friction_range"][0],
            self.randomization_params["friction_range"][1],
            size=(n,), device=self.device
        )
        self.apply_ground_friction(env_ids, frictions)

        # Randomize actuator strength (scales the torques sent to joints)
        self.actuator_scales[env_ids] = torch.uniform(
            self.randomization_params["actuator_strength_scale"][0],
            self.randomization_params["actuator_strength_scale"][1],
            size=(n, self.num_joints), device=self.device
        )

        # Randomize control latency (integer number of steps)
        low, high = self.randomization_params["control_latency_steps"]
        self.latency_steps[env_ids] = torch.randint(
            low, high + 1, size=(n,), device=self.device
        )

    def apply_observation_noise(self, obs):
        """Add Gaussian noise to observations to simulate sensor imperfections."""
        noise_cfg = self.randomization_params["observation_noise"]
        obs[:, 0:12] += torch.randn_like(obs[:, 0:12]) * noise_cfg["joint_pos"]
        obs[:, 12:24] += torch.randn_like(obs[:, 12:24]) * noise_cfg["joint_vel"]
        obs[:, 24:27] += torch.randn_like(obs[:, 24:27]) * noise_cfg["imu_gyro"]
        obs[:, 27:30] += torch.randn_like(obs[:, 27:30]) * noise_cfg["imu_accel"]
        return obs
```

The corresponding YAML configuration file allows rapid iteration without modifying code:

```yaml
# domain_randomization.yaml
domain_randomization:
  # Dynamics randomization applied at each episode reset
  dynamics:
    base_mass_offset: [-1.0, 1.0]      # kg, uniform
    link_mass_scale: [0.8, 1.2]         # multiplicative, uniform
    friction_range: [0.3, 1.25]         # ground coefficient, uniform
    restitution_range: [0.0, 0.4]       # ground restitution, uniform
    joint_damping_scale: [0.8, 1.3]     # multiplicative, uniform
    joint_friction: [0.0, 0.05]         # Nm, uniform
    actuator_strength_scale: [0.85, 1.15]
    control_latency_steps: [0, 3]       # integer, uniform

  # Observation noise applied every step
  observation_noise:
    joint_position_std: 0.01            # rad
    joint_velocity_std: 0.15            # rad/s
    imu_gyro_std: 0.02                  # rad/s
    imu_accel_std: 0.05                 # m/s^2

  # External disturbances applied periodically
  disturbances:
    push_force_range: [0.0, 30.0]       # N, applied to base
    push_interval_s: [5.0, 15.0]        # seconds between pushes
    push_duration_s: [0.05, 0.2]        # duration of each push

  # Visual randomization (for vision-based policies)
  visual:
    texture_randomize: true
    light_intensity_range: [0.3, 1.5]
    light_color_temperature: [3000, 7000]  # Kelvin
    camera_fov_jitter: 2.0              # degrees
    camera_position_jitter: 0.005       # meters
```

## Domain Adaptation

While domain randomization broadens the training distribution to encompass reality, domain adaptation takes the opposite approach: it explicitly aligns the simulated and real domains so that a policy trained in one transfers directly to the other. The two approaches are complementary and are often combined.

### System Identification

System identification (SysID) estimates the physical parameters of the real system and configures the simulator to match. This produces a high-fidelity simulation that minimizes the reality gap at its source.

Common approaches include:

1. **Manual measurement**: Weigh links on a scale, measure dimensions with calipers, estimate friction by dragging surfaces.
2. **Trajectory matching**: Record real-world trajectories under known commands, then optimize simulation parameters to minimize the trajectory error:

$$\xi^* = \arg\min_\xi \sum_{t=0}^{T} \| s_t^{\text{real}} - s_t^{\text{sim}}(\xi) \|^2$$

3. **Bayesian estimation**: Use Bayesian optimization or Markov Chain Monte Carlo methods to estimate parameter distributions rather than point estimates, naturally integrating with domain randomization.

System identification reduces the width of the randomization distribution needed, leading to better-performing policies. However, it requires real-world data collection and cannot capture all aspects of the reality gap.

### Transfer Learning and Progressive Nets

Transfer learning fine-tunes a policy pre-trained in simulation using a small amount of real-world data. Progressive neural networks extend this idea by freezing the simulation-trained network and adding lateral connections to a new network that adapts to real data, preventing catastrophic forgetting of simulation knowledge.

The procedure is:

1. Train a policy $\pi_{\text{sim}}$ to convergence in simulation.
2. Freeze the weights of $\pi_{\text{sim}}$.
3. Initialize a new network $\pi_{\text{real}}$ with lateral connections from $\pi_{\text{sim}}$.
4. Fine-tune $\pi_{\text{real}}$ on real-world rollouts with a small learning rate.

This approach is particularly useful when the reality gap is too large for zero-shot transfer but collecting large amounts of real data is impractical.

### Adversarial Domain Adaptation

Adversarial methods learn domain-invariant representations by training a feature extractor that a domain discriminator cannot distinguish between sim and real:

1. **Domain-Adversarial Neural Networks (DANN)**: A gradient reversal layer forces the feature extractor to learn representations that are informative for the task but invariant to the domain (sim vs. real). The training objective combines task loss and domain confusion:

$$\mathcal{L} = \mathcal{L}_{\text{task}}(\theta_f, \theta_y) - \lambda \, \mathcal{L}_{\text{domain}}(\theta_f, \theta_d)$$

where $\theta_f$ are feature extractor parameters, $\theta_y$ are task predictor parameters, $\theta_d$ are domain classifier parameters, and $\lambda$ controls the trade-off.

2. **Image-level adaptation with CycleGAN**: CycleGAN translates simulated images to appear realistic (and vice versa) without paired data. A policy can then be trained on "sim-to-real translated" images, which look like real camera feeds but are generated from simulation. This is effective for bridging the visual domain gap while retaining the ability to generate unlimited training data.

```python
# Pseudocode for CycleGAN-based visual domain adaptation
# Step 1: Collect unpaired images from sim and real
sim_images = collect_sim_images(num=10000)
real_images = collect_real_images(num=500)

# Step 2: Train CycleGAN to translate sim -> real appearance
cyclegan = CycleGAN()
cyclegan.train(sim_images, real_images, epochs=200)

# Step 3: During policy training, translate sim renders before feeding to policy
for episode in range(num_episodes):
    sim_obs = env.render()
    # Translate simulated image to real-looking image
    adapted_obs = cyclegan.sim_to_real(sim_obs)
    action = policy(adapted_obs)
    env.step(action)
```

## Evaluation Methodology

Evaluating sim-to-real transfer requires measuring performance in both domains and analyzing the transfer gap.

### Metrics

| Metric | Description |
|--------|-------------|
| **Sim success rate** | Task success rate in the training simulation |
| **Real success rate** | Task success rate on physical hardware |
| **Transfer ratio** | Real success rate / Sim success rate |
| **Sim-to-real gap** | Sim success rate − Real success rate |
| **Robustness score** | Success rate under perturbations (pushes, payload changes) |

### Evaluation Protocol

1. **Baseline**: Train and evaluate a policy without any randomization or adaptation to establish the naive transfer gap.
2. **Ablation**: Enable randomization categories one at a time (dynamics only, visual only, both) to measure the contribution of each.
3. **Range sweep**: Vary the width of randomization ranges. Too narrow fails to cover reality; too wide makes the task too hard to learn.
4. **Real-world trials**: Run a statistically significant number of real-world trials (typically 20–100 per condition) and report mean success rate with confidence intervals.

## Best Practices

### Start Simple, Randomize Incrementally

Begin by training in a non-randomized simulator with identified parameters. Once the policy works in that setting, add randomization categories one at a time, verifying that the policy still learns. This helps isolate which randomizations are necessary and which degrade performance.

### Validate with Real-World Data

Collect a small set of real-world trajectories early in the project. Use them to:
- Calibrate simulator parameters via system identification.
- Set informed randomization ranges rather than guessing.
- Evaluate candidate policies before committing to extensive real-world testing.

### Automated Domain Randomization (ADR)

Manually specifying randomization ranges is tedious and suboptimal. Automated domain randomization, introduced by OpenAI for the Rubik's cube project, adapts ranges during training based on policy performance:

1. Start with narrow randomization ranges centered on nominal values.
2. If the policy achieves a target success rate, widen the ranges.
3. If the policy performance drops below a threshold, narrow the ranges.

This produces the widest randomization distribution that the policy can still handle, automatically trading off robustness and performance.

$$\Delta \xi_i \leftarrow \begin{cases} \Delta \xi_i + \delta & \text{if success rate} > \tau_{\text{upper}} \\ \Delta \xi_i - \delta & \text{if success rate} < \tau_{\text{lower}} \end{cases}$$

### When NOT to Use Sim-to-Real

Sim-to-real transfer is not always the best approach:

- **Sufficient real data is available**: If you can collect enough real-world demonstrations efficiently (e.g., via teleoperation), imitation learning on real data avoids the sim-to-real gap entirely.
- **The task is contact-rich and hard to simulate**: Tasks like deformable object manipulation, fluid handling, or fine-grained tactile sensing may have reality gaps that are too large to bridge.
- **The real environment is simple**: For well-structured environments (e.g., a fixed industrial cell), it may be faster to build a high-fidelity digital twin than to randomize broadly.

## Case Studies

### OpenAI Rubik's Cube (Dactyl)

OpenAI trained a Shadow Dexterous Hand to solve a Rubik's cube entirely in simulation using massive domain randomization and ADR. The policy was trained across billions of episodes with randomized hand geometry, cube dimensions, friction coefficients, actuator gains, and visual properties. The ADR procedure automatically expanded 37 randomization parameters over the course of training. The resulting policy transferred zero-shot to the real robot, demonstrating that sufficient randomization can bridge even complex dexterous manipulation gaps.

### ANYmal Quadruped Locomotion

ETH Zurich and NVIDIA trained locomotion policies for the ANYmal quadruped robot in Isaac Gym with dynamics randomization including mass, friction, motor strength, and added latency. The policy was trained using PPO with 4096 parallel environments, achieving training times of under 20 minutes on a single GPU. The resulting controller transferred to the real ANYmal, enabling robust locomotion over rough terrain, stairs, and slippery surfaces without any real-world fine-tuning.

### Manipulation Policies with Diffusion Models

Recent work combines diffusion policy architectures with sim-to-real transfer for manipulation tasks. Policies are pre-trained in simulation with visual and dynamics randomization, then optionally fine-tuned with a small number of real-world demonstrations. The multi-modal nature of diffusion policies makes them naturally robust to distributional shift, complementing the robustness provided by domain randomization.

## Summary

Sim-to-real transfer enables training robotic policies in the safety and efficiency of simulation while deploying them on physical hardware. The reality gap — spanning visual appearance, dynamics, sensor noise, and actuator behavior — is the central challenge. Domain randomization addresses this by training across a distribution of simulated environments broad enough to include reality as a plausible sample. Domain adaptation takes the complementary approach of aligning simulated and real domains through system identification, transfer learning, or adversarial methods. In practice, the most successful deployments combine both: use system identification to build an accurate baseline simulator, apply domain randomization to handle residual uncertainty, and optionally fine-tune on real data. Start with narrow randomization ranges, validate against real-world data early, and consider automated domain randomization to systematically expand robustness.

## See Also:
- [NVIDIA Isaac Sim Setup and ROS2 Workflow](/wiki/simulation/simulation-isaacsim-setup/)
- [Building a Light-Weight Custom Simulator](/wiki/simulation/Building-a-Light-Weight-Custom-Simulator/)
- [Introduction to Reinforcement Learning](/wiki/machine-learning/intro-to-rl/)
- [Introduction to Diffusion Models and Diffusion Policy](/wiki/machine-learning/intro-to-diffusion/)
- [Gazebo Simulation](/wiki/tools/gazebo-simulation/)
- [Imitation Learning With a Focus on Humanoids](/wiki/machine-learning/imitation-learning/)

## Further Reading
- [Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning](https://developer.nvidia.com/isaac-gym) — NVIDIA's GPU-accelerated physics simulator purpose-built for RL with thousands of parallel environments.
- [Sim-to-Real Robot Learning from Pixels with Progressive Nets (DeepMind Blog)](https://www.deepmind.com/blog/sim-to-real-robot-learning-from-pixels-with-progressive-nets) — Overview of progressive neural networks for sim-to-real visual policy transfer.
- [OpenAI Solving Rubik's Cube with a Robot Hand](https://openai.com/index/solving-rubiks-cube/) — Detailed account of the Dactyl project and automated domain randomization.
- [Domain Randomization for Sim2Real Transfer (Lilian Weng's Blog)](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/) — Comprehensive survey of domain randomization techniques with references.

## References

Akkaya, I., Andrychowicz, M., Chociej, M., Litwin, M., McGrew, B., Petron, A., Paino, A., Plappert, M., Powell, G., Ribas, R., & Schneider, J. (2019). Solving Rubik's cube with a robot hand. *arXiv preprint arXiv:1910.07113*.

Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 1180–1189.

Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots. *Science Robotics*, 4(26), eaau5872.

Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Ruber, N., Tremblay, J., Murrell, T., Petrenko, O., & State, G. (2021). Isaac Gym: High performance GPU-based physics simulation for robot learning. *arXiv preprint arXiv:2108.10470*.

Rusu, A. A., Vecerik, M., Rothörl, T., Heess, N., Pascanu, R., & Hadsell, R. (2017). Sim-to-real robot learning from pixels with progressive nets. *Proceedings of the 1st Conference on Robot Learning (CoRL)*, 262–270.

Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 23–30.

Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2223–2232.
