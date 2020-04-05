---
title: ROS Introduction
---

From ROS's homepage:

> ROS (Robot Operating System) provides libraries and tools to help software developers create robot applications. It provides hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. ROS is licensed under an open source, BSD license.

Actually, ROS is not an operating system but a meta operating system, which means it assumes there is an underlying operating system to help carry out its tasks. At its core, ROS provides the functionality for programs to communicate with each other either on the same computer or over a network. Built in to ROS are many functions, such as 2D planning and navigation, robot manipulator planning and navigation, mapping, visualization and more.

**Rviz** is a 3D visualization tool for ROS. With a topic as input, it visualizes that based on the message type being published, which allows us to see the environment from the perspective of the robot.

**Gazebo** is the most popular simulator to work with ROS with good community support. The ability of a robot could be evaluated and tested in a customized 3D scenario without any harm to the real robot.

**MoveIt** is state of the art software for manipulation, incorporating the latest advances in motion planning, manipulation, 3D perception, kinematics, control and navigation. It provides an easy-to-use platform for developing advanced robotics applications, evaluating new robot designs and building integrated robotics products for industrial, commercial, R&D and other domains.

## Further Reading

- ### [Installation](http://wiki.ros.org/ROS/Installation)

  There are several versions to use on the website. You could choose any version that supports your own project. The priorized priority would stability, since you do not want to spend your time debugging potentially non-stable packages.

- ### [Base Tutorials](http://wiki.ros.org/ROS/Tutorials)

  ROS is useless without knowing how it works. Merely reading through the tutorials are not enough; this cannot be stressed enough. Learning ROS takes time and effort, so when going through the tutorials, try to understand what you are seeing, and make sure you follow along by Typing the example code, and run each tutorial to learn what is happening.
  
  One of the most important package in ROS is [navigation stack](ros-navigation). Here are several topics of it covered in this directory for reference.
  1. [Global Planner](ros-global-planner)
  2. [Local Planner](ros-local-planner)
  3. [Costmap](ros-cost-maps)
  4. [Mapping and Localization](ros-mapping-localization)
  5. [Motion Server](ros-motion-server-framework)
