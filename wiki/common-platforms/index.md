---
date: 2017-09-13
title: Common Platforms
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

The **Common Platforms** section highlights a wide range of commonly used hardware and software frameworks in robotics. These include platforms like ROS, various robot-specific SDKs, and tools tailored for drones, manipulators, and mobile robots. This section is intended to help researchers, students, and developers gain familiarity with these platforms and effectively use them for their robotics projects.

We encourage contributions to further enhance the knowledge base in this section.

## Key Subsections and Highlights

- **[Asctec UAV Setup Guide](/wiki/common-platforms/asctec-uav-setup-guide/)**
  A detailed tutorial for setting up the Asctec Pelican UAV for autonomous waypoint navigation using ROS. Covers configuring network settings, flashing firmware, and running the ROS package on the onboard Atomboard.

- **[DJI Drone Breakdown for Technical Projects](/wiki/common-platforms/dji-drone-breakdown-for-technical-projects/)**
  Explores the advantages and limitations of DJI drones in research projects. Includes information on flight modes, GPS dependencies, and practical tips for successful drone operations.

- **[Outdoor UAV Navigation with DJI SDK](/wiki/common-platforms/dji-sdk/)**
  An introduction to using DJI's SDK for UAV navigation. Includes insights into coordinate systems, compass usage, and waypoint navigation.

- **[Hello Robot Stretch RE1](/wiki/common-platforms/hello-robot/)**
  Provides guidance on working with the Hello Robot Stretch RE1 mobile manipulator. Covers software configuration, extending the robot's capabilities with custom tools, and leveraging ROS for control.

- **[Husky Interfacing and Communication](/wiki/common-platforms/husky_interfacing_and_communication/)**
  Discusses how to set up communication with the Clearpath Husky robot, including hardware setup and localization using GPS, IMU, and odometry.

- **[Khepera 4 Robot Guide](/wiki/common-platforms/khepera4/)**
  Introduces K-Team's Khepera 4 robot for indoor navigation and swarm robotics research. Includes a quick-start guide for programming and communicating with the robot.

- **[Pixhawk UAV Platform](/wiki/common-platforms/pixhawk/)**
  A guide for using Pixhawk hardware and PX4/APM firmware for UAV control. Covers firmware setup, offboard control, and integrating the platform with ROS.

- **[RC Cars for Autonomous Vehicle Research](/wiki/common-platforms/rccars-the-complete-guide/)**
  Offers a comprehensive guide to transforming RC vehicles into autonomous platforms. Includes chassis options, motor controllers, sensor integration, and control through ROS.

- **[ROS 2 Navigation with Clearpath Husky](/wiki/common-platforms/ros2-navigation-for-clearpath-husky/)**
  A step-by-step tutorial on integrating ROS 1 and ROS 2 for Clearpath Husky navigation, including the configuration of the ROS 2 Nav2 stack.

- **[Unitree Go1 Guide](/wiki/common-platforms/unitree-go1/)**
  Highlights the features and capabilities of the Unitree Go1 Edu quadruped robot. Includes power, sensors, networking configuration, and tips for simulation and real-world control.

- **[UR5e Collaborative Robotic Arm](/wiki/common-platforms/ur5e/)**
  Covers the setup and operation of the Universal Robots UR5e arm. Includes network configuration, running the ROS driver, and controlling the arm using MoveIt.

## Resources

Here is a compiled list of external resources referenced in the subsections:

1. **Asctec UAV**
   - [Asctec Pelican Official Documentation](http://wiki.asctec.de/display/AR/AscTec+Research+Home)
   - [Asctec User Forum](http://asctec-users.986163.n3.nabble.com/)
   - [ROS Wiki: asctec_mav_framework](http://wiki.ros.org/asctec_mav_framework)

2. **DJI Drone SDK**
   - [DJI SDK Documentation](https://developer.dji.com/mobile-sdk/documentation/)
   - [DJI Waypoint Navigation GitHub (Android)](https://github.com/DJI-Mobile-SDK-Tutorials/Android-GSDemo-GoogleMap)
   - [DJI Waypoint Navigation GitHub (iOS)](https://github.com/DJI-Mobile-SDK-Tutorials/iOS-GSDemo)

3. **Hello Robot Stretch RE1**
   - [Hello Robot Documentation](http://docs.hello-robot.com/)
   - [Stretch ROS Package](https://github.com/hello-robot/stretch_ros)

4. **Clearpath Husky**
   - [Clearpath Husky Documentation](http://www.clearpathrobotics.com/assets/guides/husky/)
   - [Husky Localization Guide](http://www.clearpathrobotics.com/assets/guides/husky/HuskyGPSWaypointNav.html)

5. **Khepera 4**
   - [Khepera 4 User Manual](https://www.k-team.com/khepera-iv)

6. **Pixhawk**
   - [PX4 Dev Guide](http://dev.px4.io/)
   - [Ardupilot Documentation](http://ardupilot.org/)

7. **RC Cars**
   - [F1Tenth Autonomous Racing Documentation](https://f1tenth.readthedocs.io/en/stable/)

8. **Unitree Go1**
   - [Unitree Robotics Documentation](https://www.unitree.com/)
   - [Unitree SDK](https://github.com/unitree/unitree_legged_sdk)

9. **UR5e Robotic Arm**
   - [Universal Robots ROS Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
   - [MoveIt Setup for UR5e](http://moveit.ros.org/)

## Development Needs
We seek contributions in the following areas:
- Detailed guides for setting up and integrating additional platforms like Boston Dynamics robots or custom robotic arms.
- Tutorials on advanced use cases for existing platforms (e.g., SLAM, multi-robot coordination).
- Case studies and real-world applications of common platforms in robotics research or industry.
