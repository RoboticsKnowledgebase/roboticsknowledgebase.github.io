---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-04-26 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: A Comprehensive Overview of Humanoid Robot Planning, Control, and Skill Learning
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
Humanoid robots are uniquely well-suited for executing human-level tasks, as they are designed to closely replicate human motions across a variety of activities. This capability enables them to perform whole-body loco-manipulation tasks, ranging from industrial manufacturing operations to service-oriented applications. Their anthropomorphic structure resembling the human form provides a natural advantage when interacting with environments built for humans, setting them apart from other robotic platforms.

Humanoids are particularly valuable for physical collaboration tasks with humans, such as jointly moving a heavy table upstairs or providing direct human assistance in daily living and healthcare scenarios. However, achieving these intricate tasks is far from straightforward. It requires managing the robot’s highly complex dynamics while ensuring safety and robustness, especially in unstructured, unpredictable environments.

One promising path to address these challenges is to leverage the abundance of human-generated data including motion demonstrations, sensory feedback, and task strategies to accelerate the acquisition of motor and cognitive skills in humanoid robots. By learning from human behavior, humanoids can potentially develop adaptive, generalized capabilities more quickly. Thus, leveraging human knowledge for humanoid embodiment is seen as a fast and effective route toward achieving true embodied intelligence, bridging the gap between current robotic capabilities and natural human-like autonomy.

This blog focuses on a subset of the vast humanoid robotics field. As illustrated in Figure 1 below, we specifically explore two major components critical to whole-body loco-manipulation:
- Traditional Planning and Control Approaches
- Emerging Learning-Based Methods

Humanoid robotics spans a much larger landscape, including mechanical design, perception, and decision-making, but here we narrow the scope to planning, control, and skill learning.

![Scope of This Blog: Traditional Planning and Control vs. Learning-Based Approaches](/assets/images/Humanoid%20robot.drawio.png)

*Figure 1: Scope of this blog. Humanoid robots are complex systems. We focus on two key pillars: Traditional Planning and Control (multi-contact planning, model predictive control, whole-body control) and Learning-Based Approaches (reinforcement learning, imitation learning, and combined learning methods).*

## Foundations of Humanoid Loco-Manipulation (HLM)

Model-based methods serve as the cornerstone for enabling humanoid loco-manipulation (HLM) capabilities. These approaches depend critically on accurate physical models, which greatly influence the quality, speed, and safety guarantees of motion generation and control. Over the past decade, planning and control strategies have converged toward a predictive-reactive control hierarchy, employing a model predictive controller (MPC) at the high level and a whole-body controller (WBC) at the low level.

These techniques are typically formulated as optimal control problems (OCPs) and solved using numerical optimization methods. While these methods are well-established, ongoing research continues to focus on improving computational efficiency, numerical stability, and scalability to high-dimensional humanoid systems.

In parallel, learning-based approaches have witnessed a rapid surge in humanoid robotics, achieving impressive results that are attracting a growing research community. Among them, reinforcement learning (RL) has demonstrated the ability to develop robust motor skills through trial-and-error interactions. However, pure RL remains prohibitively inefficient for HLM tasks, given the high degrees of freedom and sparse reward settings typical in humanoids. To address this, RL is often trained in simulation environments and later transferred to real-world systems, though this introduces challenges in bridging the sim-to-real gap.

On the other hand, imitation learning (IL) from expert demonstrations has proven to be an efficient method for acquiring diverse motor skills. Techniques such as behavior cloning (BC) have shown remarkable capabilities in mimicking a wide array of behaviors. As the quest for versatile and generalizable policies continues, researchers are increasingly focusing on scaling data.

Although collecting robot experience data is highly valuable, it is expensive and time-consuming. Thus, learning from human data abundantly available from Internet videos and public datasets has emerged as a pivotal strategy for humanoid robotics. Leveraging human demonstrations is a unique advantage of humanoid embodiments, as their anthropomorphic form makes direct learning from human behavior feasible.


## Planning and Control

### Multi-Contact Planning

Multi-contact planning is a fundamental and challenging aspect of humanoid loco-manipulation. Planners must compute not only robot state trajectories but also determine contact locations, timings, and forces while maintaining balance and interaction with diverse environments and objects.

The field has produced significant progress over the past decade, but most state-of-the-art (SOTA) methods still rely on pre-planned contact sequences. Solving contact planning and trajectory generation simultaneously, known as contact-implicit planning (CIP), remains computationally intensive due to the combinatorial complexity of contact modes.

As illustrated in Figure 2, multi-contact planning methods can be broadly categorized into three major groups:

- **Search-Based Planning**:  
  These methods expand robot states by exploring feasible contact sequences using heuristics. Techniques like Monte Carlo Tree Search and graph-based search offer practical solutions but face challenges with high computational demands and limited exploration horizons.

- **Optimization-Based Planning**:  
  Contact-Implicit Trajectory Optimization (CITO) integrates contact dynamics directly into trajectory planning, allowing simultaneous optimization of contact modes, forces, and full-body motions. While CITO has been applied to quadrupeds and robotic arms, extending it to humanoid robots in real time remains a significant challenge.

- **Learning-Based Planning**:  
  Learning-based approaches utilize reinforcement learning or supervised learning to predict contact sequences, task goals, or dynamic models, enhancing planning efficiency and enabling more flexible real-time (re)planning.

Additionally, **Pose Optimization (PO)** plays a complementary role by computing optimal whole-body static or quasi-static poses to maximize interaction forces or stability during manipulation tasks. While PO techniques are effective for discrete tasks like object pushing, they are limited when it comes to dynamic, continuous motions — motivating the adoption of dynamic optimization approaches such as model predictive control.

![Multi-Contact Planning Categories](/assets/images/multi_contact_planning.png)

*Figure 2: An overview of multi-contact planning categories: Search-Based, Optimization-Based, and Learning-Based methods, each addressing the challenges of humanoid loco-manipulation planning with different strengths and limitations.*

### Model Predictive Control for Humanoid Loco-Manipulation

Model Predictive Control (MPC) has become a cornerstone of trajectory planning for humanoid loco-manipulation, valued for its flexibility in defining motion objectives, mathematical rigor, and the availability of efficient optimization solvers.

MPC formulates motion planning as an Optimal Control Problem (OCP) over a finite future horizon, optimizing system states, control inputs, and constraint forces while ensuring dynamic feasibility and adherence to task-specific constraints.

Depending on the dynamics model and problem structure, MPC methods generally fall into two categories:

- **Simplified Models (Linear MPC)**:  
  Simplified dynamics such as the Single Rigid-Body Model (SRBM) and the Linear Inverted Pendulum Model (LIPM) allow high-frequency online planning through efficient convex optimization. These models enable dynamic locomotion and loco-manipulation but may sacrifice modeling accuracy.

- **Nonlinear Models (NMPC)**:  
  Nonlinear MPC uses more detailed dynamics, such as Centroidal Dynamics (CD) and Whole-Body Dynamics (WBD), capturing full-body inertia and multi-contact behaviors. Although they improve motion fidelity, these models impose significant computational demands. Accelerated optimization methods like Sequential Quadratic Programming (SQP) and Differential Dynamic Programming (DDP) are often used to make NMPC practical for real-time control.

### Whole-Body Control

Whole-Body Control (WBC) refers to controllers that generate generalized accelerations, joint torques, and constraint forces to achieve dynamic tasks in humanoid robots. WBC is essential when trajectories are generated from reduced-order models, full-order plans are too computationally intensive to track directly, or when disturbances from environment uncertainty must be compensated in real time.

The WBC solves an instantaneous control problem based on Euler-Lagrangian dynamics, optimizing decision variables such as accelerations, constraint forces, and joint torques. Due to the underactuated nature of humanoids, WBC must satisfy contact constraints while maintaining dynamic balance.

Dynamic tasks within WBC are formulated as linear equations of the decision variables and can represent objectives such as:
- Joint acceleration tracking
- End-effector motion tracking
- Centroidal momentum regulation
- Capture point stabilization
- Reaction force optimization
- Collision avoidance

Dynamic tasks can be precomputed (e.g., from MPC), generated online, or commanded interactively through teleoperation, including VR-based interfaces with haptic feedback.

WBC approaches fall into two main categories:

- **Closed-Form Methods**:  
  Early methods like inverse dynamics control and operational space control project system dynamics into constraint-free manifolds, enabling efficient task tracking and strict task prioritization. However, closed-form methods struggle with incorporating inequality constraints such as joint limits and obstacle avoidance.

- **Optimization-Based Methods**:  
  Modern WBC often formulates control as a quadratic programming (QP) problem, offering flexibility to handle multiple equality and inequality tasks. Conflicting tasks are resolved through strict hierarchies or soft weightings. Optimization-based WBC allows better handling of complex task requirements and robot constraints.

Both approaches have contributed to advances in humanoid loco-manipulation, with optimization-based methods gaining popularity for their robustness and extensibility.

## Learning-based Approches
### Reinforcement Learning

Reinforcement Learning (RL), empowered by deep learning algorithms, offers a model-free framework to acquire complex motor skills by rewarding desirable behaviors through environment interaction. RL eliminates the need for prior expert demonstrations but introduces significant challenges related to sample efficiency, reward design, and real-world deployment.

A key advantage of RL is its end-to-end nature: policies can map raw sensory input directly to actuation commands in real time. However, learning from scratch is often inefficient and sensitive to the design of reward functions. Algorithms such as Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) have struggled to achieve whole-body loco-manipulation in humanoids due to the high degrees of freedom, unstable dynamics, and sparse rewards typical in these tasks.

Several strategies have been developed to address RL challenges:
- **Curriculum Learning**: Gradually increases task difficulty during training to accelerate learning.
- **Curiosity-Driven Exploration**: Encourages exploration of novel states to improve learning without heavily engineered rewards.
- **Constrained RL**: Replaces complex reward tuning with task-specific constraints, achieving better locomotion performance.

The sim-to-real gap remains a critical bottleneck in RL for humanoid robotics. While quadrupeds have achieved notable sim-to-real success due to their simpler, more stable dynamics, humanoids face greater challenges including dynamic instability, high-dimensional control, and complex manipulation environments.

Common strategies to bridge the sim-to-real gap include:
- **Domain Randomization**: Training with randomized physical parameters to improve real-world robustness, though it requires careful tuning.
- **System Identification**: Using real-world data to refine simulation parameters for greater fidelity.
- **Domain Adaptation**: Fine-tuning simulation-trained policies on real hardware data with safety measures in place.

RL offers a pathway to learn novel, complex behaviors for humanoid loco-manipulation but remains impractical for direct real-world training due to its inefficiency and sim-to-real difficulties. Consequently, RL is predominantly used in simulation, with real-world deployment heavily relying on additional techniques like imitation learning (IL) to close the gap.

## Imitation Learning from Robot Experience

Imitation Learning (IL) encompasses a range of techniques, including supervised, reinforcement, and unsupervised learning methods, that train policies using expert demonstrations. IL is particularly effective for complex tasks that are difficult to specify explicitly.

The process typically involves three steps: capturing expert demonstrations, retargeting those demonstrations into robot-compatible motions if needed, and training policies using the retargeted data. Demonstrations can come from different sources, with robot experience data divided into two types:  
(i) policy execution and (ii) teleoperation.

**Policy Execution**:  
Executing expert policies, either model-based or learned, can generate training data efficiently in simulation environments. While scalable, the fidelity gap between simulation and hardware introduces sim-to-real challenges.

**Teleoperation**:  
Teleoperation enables human experts to command robot motions in real time, providing smooth, natural, and diverse demonstrations across tasks such as object manipulation, locomotion, and handovers. Full-body teleoperation, however, often requires specialized hardware like IMU suits or exoskeletons and remains challenging for capturing dynamic motions like walking.

Key differences exist between teleoperation and policy execution data:
- Teleoperation data is often **multimodal**, representing multiple valid ways to perform a task.
- Policy execution data tends to be **unimodal**, producing more consistent behaviors.

From these datasets, several IL techniques are applied:
- **Behavior Cloning (BC)**: Casts IL as supervised learning, effective for direct skill transfer.
- **Diffusion Policies**: Learn multimodal, highly versatile skills by modeling action distributions.
- **Inverse Reinforcement Learning (IRL)**: Infers reward structures from expert demonstrations and retrains policies to generalize across environments.
- **Action Chunking Transformers (ACT)**: Address distribution shift and error compounding by predicting sequences of future actions.

Although collecting high-quality robot data is resource-intensive, IL remains a reliable method to achieve expert-level performance. Scaling teleoperation data collection has become a major focus for industrial labs and companies like Tesla, Toyota Research, and Sanctuary, aiming to create large, diverse datasets for training increasingly versatile robotic policies.

### Combining Imitation Learning (IL) and Reinforcement Learning (RL)

One effective approach to sim-to-real transfer combines imitation learning (IL) and pure reinforcement learning (RL) through a two-stage teacher-student paradigm. In this framework, a teacher policy is first trained using pure RL with privileged observations in simulation. A student policy then clones the teacher's behavior using only onboard, partial observations, making it suitable for direct deployment on real hardware.

An alternative two-stage framework reverses the order: IL is used to pre-train a policy from expert demonstrations, which is then fine-tuned using RL to surpass expert-level performance and adapt to varying tasks and environments. This combination leverages the efficiency of IL with the adaptability and performance potential of RL.


## Summary

Humanoid robots offer tremendous potential for performing complex whole-body loco-manipulation tasks, but achieving robust and adaptable behavior remains a significant challenge. In this article, we explored two primary avenues: traditional model-based planning and control methods, and emerging learning-based approaches.

Model-based methods, including multi-contact planning, model predictive control (MPC), and whole-body control (WBC), provide strong theoretical guarantees and precise physical feasibility but face challenges in computational scalability and adaptability to complex environments. Learning-based methods, particularly reinforcement learning (RL) and imitation learning (IL), offer flexibility and scalability but encounter issues such as sample inefficiency and the sim-to-real gap.

Combining these two paradigms — using learning to enhance planning or using model-based structures to guide learning — is emerging as a powerful strategy. Techniques like curriculum learning, domain randomization, and hybrid IL-RL training frameworks show promise for bridging the gap between simulation and real-world deployment, especially for complex humanoid tasks.

Going forward, advancing humanoid robotics will require innovations in both scalable learning techniques and efficient, adaptive control frameworks. Additionally, systematic approaches to address the sim-to-real gap and leverage large-scale data collection will be crucial to enable reliable humanoid performance in real-world, dynamic environments.


## See Also:
- ## See Also:
- [Model Predictive Control (MPC) for Robotics](https://roboticsknowledgebase.com/wiki/actuation/model-predictive-control/)

## Further Reading
- [Sim-to-Real Transfer for Reinforcement Learning Policies in Robotics](https://arxiv.org/abs/2009.13303) — A detailed exploration of strategies to overcome the sim-to-real gap in robotic learning.
- [Curriculum Learning in Deep Reinforcement Learning: A Survey](https://arxiv.org/abs/2003.04664) — A review on curriculum learning approaches that accelerate policy training in complex environments.
- [How Tesla Trains its Humanoid Robot (Tesla AI Day Summary, 2022)](https://www.tesla.com/AI) — Insights into Tesla's Optimus humanoid, focusing on scalable learning from teleoperation and simulation.
- [Deep Reinforcement Learning: Pong from Pixels (OpenAI Blog)](https://openai.com/research/pong-from-pixels) — A simple but powerful introduction to how deep RL can learn complex behavior from raw visual inputs.


## References
[1] Z. Gu, A. Shamsah, J. Li, W. Shen, Z. Xie, S. McCrory, R. Griffin, X. Cheng, C. K. Liu, A. Kheddar, X. B. Peng, G. Shi, X. Wang, and W. Yu, "Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning," arXiv preprint arXiv:2501.02116, 2025.

[2] J. Achiam, "Spinning Up in Deep Reinforcement Learning," OpenAI, 2018. [Online]. Available: https://spinningup.openai.com/en/latest/

