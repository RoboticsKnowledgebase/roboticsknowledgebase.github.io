---
title: Building a Light Weight Custom Simulator
published: true
---

# Overview

Out project makes extensive usage of simulators to test our reinforcement learning algorithm. Due to the fact that different simulators possess different functionalities and characteristics, we decided to use different simulators at different stages of the project.

# Background

While a high-fidelity simulator is always desirable for testing a robotics system from a technical perspective, but having the development process to block the testing of other modules is usually unwanted from a project management perspective. Additionally, it is better to test the system with minimal external noises at the early stage of a project. As a result, the presence of a light-weight low fidelity simulator is usually helpful to a project to allow for rapid prototyings.

# Design Considerations

As specified earlier, the design of this light-weight simulator holds a completely different philosophy from final stage testing simulators like CARLA.
1. Minimal external noises (e.g. Perception Inaccuracy, Controller Error, etc.)
2. Highly customizable (Easy to change: Vehicle Model, Environment, Agent Behavior, etc.)
3. Minimal development effort
4. Simple and reliable architecture

# Minimal External Noises

During project executions, software code base usually cannot deliver desired robustness at the beginning of the project, but it is still essential to test the best case performances of those algorithms. Therefore, as an early-stage testing tool, the light-weight simulator must minimize any noises present in the simulation.

# Highly Customizable

System design is usually not finalized at project early stages, as such many components of the system may change. Thus, the light-weight simulator must be able easily modifiable according to the design changes.

# Minimal Development Effort

Since this simulator is meant to test early stage implementation, its development should not be blocking any other tasks by taking longer than the other systems to be implemented. As a result, this simulator should only be implemented with the required functionalities with no additional workload.

# Simple and Reliable Architecture

The purpose of this simulator is to test other algorithms, and thus it should not take longer to debug than the system to be tested. Most of the time, simpler code structures are more reliable than complex ones, and especially multi-processing structures are the hardest to debug.
