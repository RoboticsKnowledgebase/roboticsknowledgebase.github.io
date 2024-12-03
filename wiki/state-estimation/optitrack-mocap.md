---
date: 2024-12-02
title: Setting Up OptiTrack Mocap System
---

Motion capture (mocap) systems, such as OptiTrack, provide highly accurate positional and orientational data, often regarded as ground truth, for rigid and deformable objects. This capability is invaluable in robotics for debugging subsystem modules, verifying controller designs, or comparing perception systems. By employing an array of overhead cameras and reflective markers, mocap systems accurately track movements and provide real-time feedback.

Two major brands dominate the mocap market: VICON and OptiTrack. Both systems operate on similar principles but offer distinct software ecosystems and hardware configurations. In this guide, we will focus on the OptiTrack system, detailing its setup, integration with robotics software frameworks like ROS and PX4, and applications such as PX4 integration for aerial robots.

## Overview of Motion Capture Systems

Motion capture systems rely on reflective markers affixed to objects or bodies. These markers are tracked by high-speed cameras equipped with infrared (IR) lights.

### Working Principles
- **Rigid Bodies**: For rigid objects like drones or ground robots, the relative positions of markers remain constant, allowing accurate calculation of the object’s position and orientation.
- **Deformable Bodies**: For flexible or articulated objects such as wires or human bodies, individual marker positions are tracked to reconstruct motion.

### Applications in Robotics
- **Controller Tuning**: Validate robot control algorithms by comparing mocap data to expected outcomes.
- **Perception System Evaluation**: Benchmark sensor-based localization against mocap-generated ground truth.
- **Real-time Motion Planning**: Use mocap data for precise control in dynamic environments.

**Note**: OptiTrack is particularly popular in the robotics community due to its affordability and ease of integration with open-source tools like ROS.

## OptiTrack System Setup

### Hardware Requirements
Before setting up the system, ensure your workspace meets the following requirements:
- **Camera Placement**: Arrange OptiTrack cameras to achieve overlapping fields of view. This maximizes tracking accuracy and avoids occlusions.
- **Reflective Markers**: Use pre-stitched marker configurations for rigid bodies or distribute markers evenly for deformable objects.
- **Environmental Considerations**: Reduce IR interference (e.g., sunlight or other IR sources) and minimize reflective surfaces that may cause false positives.

### Software Setup: Motive Software in Windows
Motive is the core software used for managing OptiTrack motion capture systems, including configuration, tracking, visualization, and analysis of motion data. It supports various applications such as biomechanics studies, animation, VR, and robotics.

#### Minimum System Requirements
- **Operating System**: Windows 10 or later (64-bit)
- **Processor (CPU)**: Intel Core i5 or equivalent
- **Memory (RAM)**: 8 GB
- **Graphics (GPU)**: Dedicated graphics card with DirectX 11 support (e.g., NVIDIA GeForce GTX 1050 or AMD Radeon RX 560)
- **Storage**: 500 GB or more SSD
- **USB Ports**: USB 3.0
- **Network**: Gigabit Ethernet
- **Display Resolution**: Minimum 1920x1080 (Full HD)
- **Other Requirements**: DirectX 11 or higher

[Installation and Activation Guide](https://docs.optitrack.com/v3.0/motive/installation-and-activation)

### Pulling Mocap Data to a Linux Computer and ROS Workspace
The OptiTrack system is typically operated via a Windows computer connected to the cameras. The data can be streamed to a Linux machine running ROS for further integration.

#### Steps:
1. Configure objects and setup using Motive software on the Windows machine.
2. Install the ROS driver `mocap_optitrack` on the Linux computer. Use these tutorials:
   - [OptiTrack and ROS Tutorial](https://tuw-cpsg.github.io/tutorials/optitrack-and-ros/)
   - [ROS Wiki: mocap_optitrack](http://wiki.ros.org/mocap_optitrack)
3. Ensure both systems are on the same network.
4. Stream data from Motive software by selecting the object name. Adjust broadcast frequency as needed (default: <100 Hz; maximum: 1000 Hz).

> **Note**: The mocap data uses an ENU frame (East-North-Up). This differs from the Motive software visualization where Y is up.

## Fuse Mocap Data into PX4

For aerial robotics, mocap data can be fused into PX4 for precise positioning. This is particularly useful in environments where GPS is unavailable. Detailed documentation is available [here](https://docs.px4.io/v1.12/en/ros/external_position_estimation.html).

By following this guide, you can:
- Treat mocap data as a fake GPS signal.
- Enable PX4's position hold mode without GPS.

### Tips for Successful Integration:
1. **Remap Data**: Map mocap data to `/mavros/vision_pose/pose` at 30–50 Hz in the ENU frame. The `mocap_optitrack` ROS driver handles this conversion. If not, write a script to ensure data consistency.
2. **Frame Alignment**: ROS uses the ENU frame, while PX4 internally uses the NED frame. Mavros ensures compatibility with ENU.
3. **EKF2_EV_DELAY Parameter**: Adjust this parameter in PX4 to synchronize IMU and mocap data. Compare orientations from both sources to determine the correct value.
4. **Bandwidth Management**: Minimize unnecessary network traffic to prevent data transfer bottlenecks, which can adversely affect motion planning.

## Conclusion
OptiTrack motion capture systems offer unparalleled precision for robotics applications. By effectively setting up hardware, configuring Motive software, and integrating mocap data into frameworks like ROS and PX4, roboticists can unlock new possibilities in localization, control, and real-time planning. 

Adhering to best practices, such as proper marker placement and frame alignment, ensures accurate and reliable data for any application. Further exploration of advanced techniques and troubleshooting strategies will solidify OptiTrack’s role in cutting-edge robotics research and development.

## See Also
- [Setting up ROS Workspaces](https://wiki.ros.org/ROS/Tutorials)
- [Motion Capture for Robotics](https://roboticsknowledgebase.com/mocap)

## Further Reading
- [OptiTrack Official Documentation](https://optitrack.com/documentation/)
- [PX4 External Position Estimation](https://docs.px4.io/v1.12/en/ros/external_position_estimation.html)
- [Benchmarking Localization Systems](https://roboticsbenchmarking.com)

## References
- OptiTrack Motive Installation Guide: <https://docs.optitrack.com/v3.0/motive/installation-and-activation>
- TUW CPSG Tutorial: <https://tuw-cpsg.github.io/tutorials/optitrack-and-ros/>
- ROS Wiki - mocap_optitrack: <http://wiki.ros.org/mocap_optitrack>
- PX4 External Position Estimation: <https://docs.px4.io/v1.12/en/ros/external_position_estimation.html>
