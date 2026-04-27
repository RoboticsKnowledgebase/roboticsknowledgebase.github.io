---
date: 2025-04-30 # YYYY-MM-DD
title: Using xarm_ros with XArm 7 in ROS
---

This article covers the `xarm_ros` package, a valuable tool for controlling xArm robots using ROS 1 or ROS 2. It provides a low-friction setup for motion planning, Cartesian control, and basic manipulations. This entry explains how to integrate the package with your system, key components, and configurations to optimize usage for industrial, academic, or research tasks. Additionally, it discusses common pitfalls and configuration tips for reliable performance.

## Introduction

The `xarm_ros` package offers an easy-to-use interface for controlling the xArm 7 robot within ROS environments. By connecting directly to the hardware or a simulated robot, `xarm_ros` provides essential tools like launch files, URDF/XACRO models, and MoveIt configurations, making it easier to get started with robotics control. The goal is to minimize the complexity of setting up your robot, letting users focus on developing high-level functionalities rather than dealing with hardware details.

## First Steps for Using xarm_ros

To begin using `xarm_ros`, ensure the `xarm_sdk` is properly configured. This SDK is responsible for communicating with the xArm hardware, and proper configuration is essential for smooth operation. The `cxx` folder within the SDK contains important C++ files that must be correctly set up. You can find the files listed on the following [GitHub page](https://github.com/xArm-Developer/xArm-CPLUS-SDK/tree/30640642a40963d93c1cfad417469f6819ff882d).

Once the SDK is set up, you can start integrating it with ROS by installing the `xarm_ros` package.

## xarm_ros Directory Structure

The `xarm_ros` package includes several key directories:

1. **xarm_api**
   - **Purpose:** This is the core API module that facilitates communication between ROS and the xArm controller. It wraps the xArm C++ SDK into ROS services and messages.
   
2. **xarm_controller**
   - **Purpose:** Contains custom controllers for the xArm robots within the ROS control framework.

3. **xarm_description**
   - **Purpose:** Contains the URDF models for the xArm robots.

4. **xarm_gazebo**
   - **Purpose:** Provides Gazebo simulation capabilities for the xArm.
   - **Launch Command:**  
     `roslaunch xarm_gazebo xarm[5/6/7]_gazebo.launch`

5. **xarm_gripper**
   - **Purpose:** Adds support for the xArm gripper.

6. **xarm_msgs**
   - **Purpose:** Custom ROS message and service definitions specific to xArm.

7. **xarm_planner**
   - **Purpose:** Motion planning capabilities for xArm using MoveIt.
   - **Launch Command:**  
     `roslaunch xarm_planner xarm_planner.launch robot:=xarm[5/6/7]`

8. **xarm_vision**
   - **Purpose:** Provides support for vision-related functionality with xArm robots.

9. **xarm_moveit_config**
   - **Purpose:** MoveIt configuration packages for xArm robots.
   - **Launch Command:**  
     `roslaunch xarm[5/6/7]_moveit_config demo.launch`

10. **xarm_controller**
    - **Purpose:** Implements `ros_controllers` for xArm.

## Technical Caveats and Configuration Details

When using the `xarm_ros` package in ROS 1, be aware of several technical caveats that could affect performance, particularly when aiming for reliable, deterministic control or robust trajectory planning.

### SDK and ROS Integration

The `xarm_sdk` serves as the hardware driver layer, and improper configuration can lead to inconsistencies in higher-level nodes like `xarm_api` and `xarm_controller`. Pay special attention to firmware versions to avoid subtle bugs, especially during trajectory streaming. Mismatches may cause the robot to halt or enter protective stop modes unexpectedly.

### Ethernet Configuration

Ensure that the correct Ethernet interface is used during the ROS launch process. Multi-interface machines or virtualized systems can sometimes cause issues, so verify your network settings with `ifconfig` or `ip link`. Ensure that the hardware MAC address aligns with the ROS parameters, and check for firewall issues that might block UDP traffic.

### MoveIt Configuration

It’s crucial to reference the correct manipulator planning group names in the `xarm_planner` and `xarm_moveit_config` packages. If MoveIt throws errors about mismatched planning groups or joint states, double-check your URDF and SRDF definitions, as well as the controller configuration YAML files in `xarm_controller`. Incorrect inertial or collision parameters in XACRO files can mislead the planning pipeline, potentially resulting in self-collisions or infeasible trajectories.

### Debugging and Testing

Test the planning scene in RViz, ensuring that planning request visualization is turned on. Use `tf_monitor` or `rqt_tf_tree` to check for discrepancies in TF frames, as delayed or missing transformations can cause issues in time-critical Cartesian control loops.

Ensure proper sequencing of service calls when using the `xarm_api` service interfaces. Calls to `set_mode` and `set_state` should precede executing joint or Cartesian commands. A common failure is executing `set_mode` without first setting the state to `READY`, which will result in no feedback or errors.

### Gripper Operations

For gripper operations, do not rely solely on joint state publications. The gripper controller uses internal state variables, so it's crucial to initialize the gripper via the corresponding service call to ensure proper operation.

### Controller Spawner Issues

Be cautious of hardcoded parameters in `roslaunch` files, especially in `xarm_moveit_config`. If you are dynamically generating URDFs or modifying controller configurations, ensure that you remap the relevant parameters to avoid node startup crashes or MoveIt planning failures. Additionally, reorder the launch sequence to ensure the `joint_state_publisher` starts before MoveIt to prevent stale joint data.

## Summary

The `xarm_ros` package provides an efficient and straightforward method for controlling xArm robots in ROS, but attention to configuration and sequencing is critical to avoid common pitfalls. Careful setup of the SDK, Ethernet interface, and MoveIt configurations ensures reliable performance and smooth integration.

## See Also:
- [xArm ROS Documentation](https://github.com/xArm-Developer/xArm-ROS)
- [MoveIt Documentation](https://moveit.ros.org/)

## Further Reading
- [xArm C++ SDK GitHub](https://github.com/xArm-Developer/xArm-CPLUS-SDK)
- [ROS Control Tutorials](http://wiki.ros.org/ros_control)

## References
- [XArm ROS Package GitHub](https://github.com/xArm-Developer/xArm-ROS)
- [ROS Wiki](http://wiki.ros.org/)

