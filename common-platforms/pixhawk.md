---
title: Pixhawk UAV Platform
---
This guide is meant for people using UAVs with the Pixhawk hardware and looking to control the UAV through a computer/microprocessor.

## Set Up
### Firmware
This is the most confusing part of the Pixhawk/PX4/APM community. The Pixhawk is a hardware developed after multiple iterations (Older versions of hardware include the PX4 "board", and the "APM" board). The Pixhawk hardware can be set up with the PX4 or the APM firmware (the "choice").
1. PX4 firmware or PX4 stack: Developed by the PX4 dev team led by [Lorenz Meier](https://github.com/LorenzMeier) from ETH Zurich: http://dev.px4.io/
- APM firmware: Developed by the APM team. Usually shipped set up with the 3DR products: http://copter.ardupilot.org/ardupilot/index.html


It is only a matter or choice which firmware to use as both are fully compatible with the Pixhawk hardware and most UAV configurations. Few differences between them:

- PX4 is iterated more often and tested while being developed, although their stable branch should always be in perfect working condition, whereas the APM is updated less often and is assumed to be more stable.
- At the time of testing, APM did not support Z-angular velocity commands to control the UAV. Though, this should be verified against the latest implementation.
- PX4 uses control groups for direct actuator control whereas APM uses the MAVlink `DO_SET_SERVO`. It may take a more time to understand but should not affect your decision.

### Recommendation
If you have a pre-loaded firmware with the UAV, prefer using that. Switching firmware may reset the PID parameters which the manufacturers must have tuned for you.

## Off-board Companion Computer
Most common uses of UAVs require them to be controlled by an external computer or microprocessor which runs the application software. Details about possible hardware, connection and control are given here:
### Hardware
Any computer or microprocessor is suitable for controlling the UAV. It should have a USB or a serial UART port to connect to the Pixhawk. A few systems that have been used:
1. Odroid (recommended due to higher processing power and ease of set up)
- BeagleBone Black
- Raspberry Pi

### Connecting to the Pixhawk
- Processor side:
  - Use an FTDI cable and connect to a USB on the laptop or microprocessor. (recommended)
  - Use a serial UART connection. Make sure the UART is 3.3V IO. 1.8V IO (as on the Odroid) will require a logic level converter in the middle.
- Pixhawk side:
  - TELEM2 port (recommended): Use the other side of the FTDI to connect to the TELEM2 port based on these instructions: PX4 or APM. Might need you to remove your radio for ground control station. Instead you will have to use the USB for connecting to the ground control.
  - USB port on the Pixhawk (Not recommended): USB port is initialized on power up, hence may be unavailable for communication at certain moments.
  - Serial 4: Documented as a viable option, but difficult to setup.

### Recommendations
1. Use Odroid XU4 with FTDI cable and TELEM2 port. This is the most reliable configuration from experience.
- On PX4, set `SYS_COMPANION` to 921600 for highest possible rate of data updates from the UAV.
- To check the setup was done, either run mavros (introduced later) or simply open a serial terminal on the companion computer and listen on the FTDI port (/dev/ttyUSB0?) and appropriate baud rate. If gibberish is seen on the screen, the connection has been set up.

## Offboard Control
In most common applications of the UAV, it needs to be externally controlled using the companion computer. Once the companion computer is set up, the following methods may be used:
1. ### [UAV serial](https://github.com/mavlink/c_uart_interface_example)
  - Uses MAVlink to communicate commands directly to the UAV over serial .
    - Pros:
      - Provides access to low-level control.
      - Recommended by the development lead.
      - Well tested
     - Cons:
      - Requires more time to develop a full application.
      - Limits you to using C++.
      - Requires knowledge of pthreads for most applications.
- ### [MAVROS](http://wiki.ros.org/mavros) (recommended)
  - MAVROS is a ROS framework over the MAVlink for communication with the UAV. It exposes required APIs over ROS topics and services.
  - Set up instructions for the companion computer can be [found here](http://dev.px4.io/ros-mavros-installation.html). It exposes most utility items as ROS topics and services for utmost convenience.
    - Pros:
      - Provides ROS based access to features and utilities.
      - Freedom of programming language - C++ and Python.
      - Enables use of most ROS packages directly (for navigation, planning, and perception).
    - Cons:
      - No low level control of the firmware.
      - May not have access to all APIs exposed by the firmware.

## UAV state monitoring
The UAV's state can be easily monitored by simply subscribing to ROS topics. For example:
1. Local Position / Odometry: Can be obtained by subscribing to `/mavros/local_position/pose` or `/mavros/local_position/odom`. Note: The "local" position decides its origin whenever the Pixhawk boots up.
- Global Position /GPS coordinates: Can be obtained by subscribing to `/mavros/global_position/*``
- State of the UAV: Subscribe to `/mavros/state` and `/mavros/extended_state`
- TF for the local position can be made available for updating the values in the `px4_config.yaml` or `apm_config.yaml`.


## Position Control
Position can be controlled easily by publishing local setpoints to `/mavros/setpoint_position/local`.
1. This control requires the UAV to be in `"OFFBOARD"` mode (PX4) or `"GUIDED"` mode (APM) to move the UAV.
- The positions are maintained in the local frame (with origin at last power cycle point).
- These should be published at a constant rate of at least 10 Hz. To add this functionality to an application, you may need to use a separate thread that can publish the current setpoint at 10 hz.
- Autonomous takeoff can be implemented using this directly. Setting a high waypoint with the same X and Y as the current position of the UAV, makes the UAV take off.

### Tips:
- Use Python with a structure similar to [MAVROS offboard](http://dev.px4.io/ros-mavros-offboard.html) for a quick and robust implementation of threads.
- Autonomous landing can be implemented using the above example or using a separate mode on PX4 `Auto.LAND`.
- Landing state is received in the `extended_state` topic, and can be check to verify when the UAV has landed.

## Velocity Control
Velocity of the UAV can be directly controlled by publishing to the `/mavros/sepoint_velocity/cmd_vel` topic.
1. This control requires the UAV to be in "OFFBOARD" mode (PX4) or "GUIDED" mode (APM) to move the UAV.
- Do note that the velocities are sent in the local frame (frame of UAV attached to he ground where it was powered up). Hence the velocity in x takes the UAV in one direction irrespective of its yaw.
- Can be used for sending velocity commands that come from various ROS planning nodes, but only after appropriate frame conversions.

### Recommendation
- Velocity control is less robust than position control. Prefer position control where possible.
- If you are using MAVROS, look up `mavsetp`, `mavsys`, `mavsafety`, and `mavcmd` on the MAVROS Wiki for easy to use services to test your UAV.

## Setting up a simulator
As testing on the real UAV is often dangerous, set up a simulation environment, which will help you test most of the code beforehand.
1. PX4. Multiple simulators are available:
    1. [SITL](http://dev.px4.io/simulation-sitl.html): Software in the loop. Can use `Jmavsim` as a simulator. Is easy to set up, but a little inconvenient to test.
    - [Gazebo](http://dev.px4.io/simulation-gazebo.html): Similar to SITL, using the Gazebo simulation environment. Very powerful, but a little time consuming to set up.
    - [Connect through ROS](http://dev.px4.io/simulation-ros-interface.html).
- APM
    1. [SITL](http://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html) simulator. See above.  

## Additional Resoursces
- As the documentation on most of these items are still being developed. You may need to take help from the community in case of any trouble.
  - PX4 forums: [px4-users](https://groups.google.com/forum/#!forum/px4users) Google group
  - APM forums: http://forums.ardupilot.org/
  - Read source code for the firmware.
  - Email developers directly.
- Recommended links for MAVROS references:
  - Setup and MAVROS code: https://github.com/mavlink/mavros
  - ROS wiki documentation: http://wiki.ros.org/mavros
