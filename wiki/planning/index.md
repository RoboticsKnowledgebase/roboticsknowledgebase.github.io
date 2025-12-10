---
date: 2024-12-05
title: Planning
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

The **Planning** section offers resources and tutorials on motion planning techniques, ranging from classical algorithms like A* to modern approaches such as Frenet-frame trajectory planning and multi-robot navigation strategies. This section is tailored to help researchers and developers understand, implement, and optimize planning algorithms for diverse robotics applications.

We are actively seeking contributions to expand the resources available in this section.

## Key Subsections and Highlights

- **[A* Implementation Guide](/wiki/planning/astar_planning_implementation_guide/)**
  A step-by-step tutorial on implementing the A* algorithm for robot motion planning. Covers key concepts such as heuristic design, map representation, and non-holonomic motion primitives for Ackermann vehicles.

- **[Behavior Trees](/wiki/planning/behavior_tree/)**
  Comprehensive guide to Behavior Trees for robot decision-making. Covers BT architecture vs FSM/HFSM, node types (Action, Condition, Sequence, Fallback), AirStack implementation, and ROS2 integration with hands-on tutorials.

- **[Coverage Planning Implementation Guide](/wiki/planning/coverage-planning-implementation-guide/)**
  Details cellular decomposition-based coverage planning methods for ensuring full area coverage. Applications include drone monitoring and robotic vacuum cleaning.

- **[Trajectory Planning in the Frenet Space](/wiki/planning/frenet-frame-planning/)**
  Explains Frenet-frame trajectory planning, useful for structured environments like highways. Includes algorithmic steps, transformation techniques, and advantages over Cartesian coordinates.

- **[Move Base Flex](/wiki/planning/move_base_flex/)**
  A comprehensive guide to the ROS-based Move Base Flex navigation stack. Covers its architecture, plugin implementation, and customization for 2.5D/3D maps or custom planners.

- **[Multi-Robot Navigation Stack Design](/wiki/planning/multi-robot-planning/)**
  Compares planning- and control-based approaches for centralized multi-robot navigation. Includes prioritized A* and Lazy Traffic Controller implementations.

- **[Overview of Motion Planning](/wiki/planning/planning-overview/)**
  Introduces key motion planning paradigms, such as search-based and sampling-based methods, with examples like D* Lite and RRT.

- **[Resolved-Rate Motion Control](/wiki/planning/resolved-rates/)**
  A Jacobian-based control scheme for precise Cartesian movements of robot manipulators, ideal for real-time applications like surgical robotics and motion compensation.

## Development Needs
This section seeks contributions in the following areas:
- Advanced sampling-based algorithms (e.g., RRT*, PRM*)
- Hybrid approaches combining search- and sampling-based methods
- Tutorials on implementing planners using modern libraries (e.g., OMPL, ROS Navigation2)
- Case studies on integrating planners with robot systems

