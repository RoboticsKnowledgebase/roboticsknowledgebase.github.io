---
date: 2024-12-05
title: State Estimation
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

State estimation is a critical aspect of robotics, enabling accurate determination of a robot's position, orientation, and velocity in its environment. This section focuses on the **practical implementation of state estimation techniques** and highlights key algorithms, tools, and configurations used in robotics. From Adaptive Monte Carlo Localization to advanced sensor fusion methods, this section bridges theoretical concepts with real-world applications.

The "State Estimation" section provides a comprehensive understanding of how to estimate a robot's pose and state using various algorithms and sensors. It explores probabilistic methods like particle filters, integration with ROS packages, and advanced SLAM (Simultaneous Localization and Mapping) techniques. Additionally, it discusses scenarios where GPS signals are unavailable, the integration of different sensors, and the setup of navigation stacks. This section is invaluable for implementing accurate and robust localization and mapping systems, crucial for robotics applications in diverse environments.


## Key Subsections and Highlights

- **[Adaptive Monte Carlo Localization (AMCL)](/wiki/state-estimation/adaptive-monte-carlo-localization/)**
  Explains how AMCL uses a particle filter to localize robots in a known map. Includes a discussion on adaptive sampling and configuration details for the ROS AMCL package, with example launch files and parameter tuning recommendations.
  - Steps in particle filtering: re-sampling, sampling, and importance sampling.
  - Application to localization with a predefined map.
  - Configuring and tuning the ROS AMCL package.
  
- **[Cartographer ROS Integration](/wiki/state-estimation/Cartographer-ROS-Integration/)**
  Details the integration of Google Cartographer's SLAM algorithm with ROS for 2D mapping. Discusses the configuration of LiDAR, IMU, and odometry inputs, as well as custom inflation layers for navigation.
  - Sensor compatibility: LiDAR, IMU, odometry.
  - Transform preparation and costmap integration for navigation.
  - Benefits of Cartographer's submap-based large-scale mapping.

- **[GPS-Lacking State Estimation Sensors](/wiki/state-estimation/gps-lacking-state-estimation-sensors/)**
  Highlights techniques for state estimation in GPS-denied environments, including UltraWideband beacons, ultrasonic positioning, and robotic total stations. Emphasizes the pros and cons of each method and their environmental requirements.
  - Comparison of different external systems with pros and cons.
  - Applications in uniform and dynamic environments.

- **[Oculus Prime Navigation](/wiki/state-estimation/OculusPrimeNavigation/)**
  Covers waypoint navigation techniques for the Oculus Prime platform, from ROS command-line methods to graphical waypoint selection using Rviz. Provides examples for remote navigation setup.

- **[ORB SLAM2 Setup Guidance](/wiki/state-estimation/orb-slam2-setup/)**
  Step-by-step tutorial for installing and configuring ORB SLAM2 for stereo vision. Includes details on camera calibration, rectification matrices, and adding ROS publisher support for pose output.
  - Dependencies for setup: Pangolin, OpenCV, DBoW2.
  - Adding ROS publisher support for pose output.

- **[Radar-Camera Sensor Fusion](/wiki/state-estimation/radar-camera-sensor-fusion/)**
  Introduces radar and camera sensor fusion for tracking objects, especially in autonomous vehicles. Covers data association, Kalman filter-based sensor fusion, and tracker evaluation metrics.

- **[ROS Cost Maps](/wiki/state-estimation/ros-cost-maps/)**
  Explains how to configure ROS cost maps for navigation. Discusses layering, inflation, and map updates with examples from a quadrotor equipped with RGB-D sensors.

- **[ROS Mapping and Localization](/wiki/state-estimation/ros-mapping-localization/)**
  Compares popular ROS mapping and localization packages, including Gmapping, Hector Mapping, and AMCL. Highlights use cases and key differences between methods.
  - Mapping with Gmapping and Hector mapping.
  - Localization with AMCL and odometry configuration.

- **[ROS Navigation](/wiki/state-estimation/ros-navigation/)**
  Comprehensive guide for setting up the ROS navigation stack on custom robots. Discusses transform configuration, localization tools, and integration with SLAM algorithms.
  - TF tree setup and debugging tools.
  - Localization and mapping tools like AMCL, Robot_Localization.

- **[SBPL Lattice Planner](/wiki/state-estimation/sbpl-lattice-planner/)**
  Covers implementing the Search-Based Planning Lab’s Lattice Planner in ROS. Provides installation guidance and details on resolving common build errors.

- **[Visual Servoing](/wiki/state-estimation/visual-servoing/)**
  Describes the use of visual servoing for robotic control using image-based and pose-based methods. Includes a detailed application of drone alignment using image-based visual servoing.
  - Pose-based and image-based approaches.
  - Application to drones for precise alignment tasks.

## Resources

- [AMCL ROS Wiki](http://wiki.ros.org/amcl)
- [Cartographer Documentation](https://google-cartographer.readthedocs.io/en/latest/)
- [Dieter Fox's Paper on Adaptive Particle Filters](http://papers.nips.cc/paper/1998-kld-sampling-adaptive-particle-filters.pdf)
- [DecaWave UWB System](https://www.decawave.com/products/)
- [ROS Navigation Tutorials](http://wiki.ros.org/navigation/Tutorials)
- [ORB SLAM2 GitHub](https://github.com/raulmur/ORB_SLAM2)
- [Kalman Filter in Python](https://github.com/balzer82/Kalman)
- [Radar-Camera Sensor Fusion Overview](https://radar-camera-fusion-tutorial.com)
- [Costmap2D ROS Package](http://wiki.ros.org/costmap_2d)
- [Visual Servoing Platform (ViSP)](https://visp.inria.fr/)
- [Search-Based Planning Lab (SBPL)](http://sbpl.net/)