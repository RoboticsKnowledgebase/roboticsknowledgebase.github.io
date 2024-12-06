---
date: 2017-09-13
title: Controls & Actuation 
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

The "Controls & Actuation" section provides a detailed guide to implementing and understanding a range of actuation and control methodologies essential for robotics and automation. Topics span from high-level motion planning to low-level motor controllers, helping engineers bridge the gap between conceptual design and physical motion. The section includes insights into both theoretical foundations and practical implementations.

## Key Subsections and Highlights

- **[Drive-by-wire Conversion for Autonomous Vehicles](/wiki/actuation/drive-by-wire/)**
  Explains the principles and types of drive-by-wire systems, including throttle-by-wire, brake-by-wire, and steer-by-wire, and their role in modern autonomous vehicles.

- **[Linear Actuator Resources and Quick Reference](/wiki/actuation/linear-actuator-resources/)**
  A comprehensive overview of linear actuation systems, covering linear guides, belt drives, screw systems, and rack-and-pinion setups.

- **[Model Predictive Control Introduction and Setup](/wiki/actuation/model-predictive-control/)**
  Discusses Model Predictive Control (MPC), its benefits, and implementation in robotics for handling constraints in multi-input, multi-output systems.

- **[Motor Controller with Feedback](/wiki/actuation/motor-controller-feedback/)**
  Introduces motor controllers with encoder feedback, highlighting the Pololu Jrk 21v3 USB Motor Controller as an example.

- **[MoveIt Motion Planning and HEBI Actuator Setup and Integration](/wiki/actuation/moveit-and-HEBI-integration/)**
  Outlines using MoveIt in ROS for robotic motion planning and integrating it with HEBI actuators for hardware execution.

- **[PID Control on Arduino](/wiki/actuation/pid-control-arduino/)**
  Explains implementing PID control on Arduino platforms, including tips for tuning and integrating Kalman filters for noisy sensors.

- **[Pure-Pursuit Based Controller for Skid Steering Robots](/wiki/actuation/Pure-Pursuit-Controller-for-Skid-Steering-Robot/)**
  Covers the Pure-Pursuit algorithm for trajectory tracking in skid-steering robots, including implementation steps and constraints.

- **[Task Prioritization Control for Advanced Manipulator Control](/wiki/actuation/task-prioritization-control/)**
  Details task prioritization control for robotic manipulators to manage redundant degrees of freedom and prioritize multiple simultaneous tasks.

- **[Using ULN 2003A as a Motor Controller](/wiki/actuation/uln2003a-motor-controller/)**
  Explains how the ULN 2003A Darlington Array can be used as a motor controller for low-power motors and relays.

- **[Vedder Open-Source Electronic Speed Controller](/wiki/actuation/vedder-electronic-speed-controller/)**
  Highlights features and advantages of the Vedder ESC, an open-source electronic speed controller for brushless DC motors.

## Resources

- [Drive-by-wire System Overview](https://carbiketech.com/drive-by-wire-technology-working/)
- [Linear Motion Components - Misumi](https://us.misumi-ec.com/)
- [Model Predictive Control - MATLAB Guide](https://www.mathworks.com/videos/series/understanding-model-predictive-control.html)
- [Pololu Jrk 21v3 USB Motor Controller](https://www.pololu.com/product/1392)
- [MoveIt Tutorials for ROS](https://ros-planning.github.io/moveit_tutorials/)
- [Arduino PID Library](http://playground.arduino.cc/Code/PIDLibrary)
- [Task Priority Control - Robotics Journal](https://journals.sagepub.com/doi/10.1177/027836498700600201)
- [Pure-Pursuit Algorithm Reference](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [ULN 2003A Documentation](https://www.ti.com/lit/ds/symlink/uln2003a.pdf)
- [Vedder ESC Project](http://vedder.se/2015/01/vesc-open-source-esc/)
