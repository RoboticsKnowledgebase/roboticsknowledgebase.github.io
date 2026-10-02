---
date: 2025-05-04
title: Using g2o for Pose Graph Optimization in C++
---

This article provides a practical introduction to using **g2o** in C++ for pose graph optimization — a foundational tool in SLAM, calibration, and motion estimation. It is intended for robotics students or practitioners implementing multi-frame optimization, particularly in the context of Visual SLAM or sensor fusion. The article explains what g2o is and how to structure a typical optimization problem.

By the end of this article, you'll understand the essential building blocks of g2o, how to configure it for your graph-based problem, and how it fits into your robotic perception system.

## Overview of g2o
[g2o (General Graph Optimization)](https://github.com/RainerKuemmerle/g2o) is a C++ library that solves nonlinear least-squares problems by representing them as graphs. In robotics, this graph typically consists of:
- **Vertices:** Pose variables or landmark positions.
- **Edges:** Constraints between variables, such as relative transformations.

Each vertex holds a state estimate (e.g., `SE3` pose), and each edge encodes an observation or measurement (e.g., odometry, loop closure). Optimization minimizes the total error across the graph.

> g2o is particularly useful when working with keyframes in SLAM or multi-view geometry where optimization improves the global consistency of pose estimates.

## g2o Structure and Components

### 1. SparseOptimizer
The core of the framework is the `g2o::SparseOptimizer`, which manages all vertices and edges and runs the optimization routine.

### 2. Vertices
Use `g2o::VertexSE3` to represent 6-DoF robot poses. Each vertex is assigned a unique ID and an initial estimate (from odometry or SLAM tracking).

### 3. Edges
Use `g2o::EdgeSE3` to define constraints between two vertices. Edges store the relative pose and an information matrix (inverse covariance) to weigh the constraint.

### 4. Solver
g2o supports different solvers like:
- **Levenberg-Marquardt**
- **Gauss-Newton**

You must also configure a linear solver backend like **CSparse** or **Eigen**.

### 5. Execution Flow
The general flow includes:
1. Initialize `SparseOptimizer` and set solver.
2. Add all `VertexSE3` instances (each with `setEstimate()`).
3. Add all `EdgeSE3` instances with relative pose and information.
4. Call `initializeOptimization()` and then `optimize(n_iters)`.

### Basic Pseudocode
```cpp
g2o::SparseOptimizer optimizer;
// Configure solver and backend
// Add VertexSE3 with unique IDs and initial estimates
// Add EdgeSE3 with relative transforms
optimizer.initializeOptimization();
optimizer.optimize(10);
```

## Applications in Robotics

g2o is widely used in robotics for a variety of estimation and optimization tasks:

- **Visual SLAM**: Optimize camera or keyframe pose graphs for improved global consistency.
- **Loop Closure**: Add constraints between distant frames to reduce accumulated drift in the trajectory.
- **Bundle Adjustment**: Refine both 3D landmarks and camera parameters jointly for better reconstruction accuracy.
- **Sensor Calibration**: Estimate rigid-body transformations between sensors or between the robot and the global frame.
- **Multi-Robot Mapping**: Merge and optimize maps from different agents using inter-robot pose constraints.

These applications help robotic systems maintain accurate maps, trajectories, and spatial awareness under noisy measurements and long-term deployment.

## Summary

g2o is an essential optimization tool in modern robotic systems. With its efficient graph representation and support for custom vertex and edge types, it allows scalable and accurate optimization of nonlinear problems in SLAM, calibration, and state estimation. Once understood, it can be modularized and reused across perception, mapping, and planning tasks in both research and production systems.

## See Also:
- [ORB-SLAM2 Optimization Module](https://github.com/raulmur/ORB_SLAM2)
- [OpenSLAM](https://openslam-org.github.io/g2o.html)
- [Python binding (also installable using pip)](https://github.com/miquelmassot/g2o-python)

## Further Reading
- SLAM Book v2 – Chapter 7: Bundle Adjustment and g2o

## References
- R. Kümmerle, G. Grisetti, H. Strasdat, K. Konolige, and W. Burgard, “g2o: A General Framework for Graph Optimization,” *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, 2011. [Paper PDF](https://ais.informatik.uni-freiburg.de/publications/papers/kuemmerle11icra.pdf)
- g2o GitHub Repository. [https://github.com/RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o)