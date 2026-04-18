---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-05-01 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: F1/10th Autonomous Car- Debugging
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
he F1/10th autonomous driving course at Carnegie Mellon University is a multidisciplinary platform used across departments to teach core robotics concepts in perception, planning, and control. Students are provided with 1/10th-scale RC vehicles that are retrofitted with various sensors and compute resources. These cars are capable of operating autonomously through ROS-based systems and custom control pipelines.

The vehicle uses an NVIDIA Jetson Xavier NX as the primary compute unit, which runs ROS nodes for real-time processing. For perception, it is equipped with a Hokuyo 2D LiDAR and an Intel RealSense depth camera. The control of the drivetrain, however, is managed by a key piece of hardware known as the **VESC (Vedder Electronic Speed Controller)**. This open-source ESC is used widely in electric skateboards and robotic platforms due to its configurability, integrated telemetry, and support for high-speed communication protocols.

While the VESC is powerful, it is also sensitive to how data is streamed to and from it—especially under real-time ROS environments. This article highlights a persistent but subtle issue with VESC communication, outlines the diagnostics process undertaken to identify the root cause, and provides best practices for resolving and avoiding the problem in the future.

## Problem Overview

The issue in question occurs during a common development workflow. When the vehicle is powered on, and the Jetson and VESC are connected, the system appears to operate correctly: speed control works, steering commands are honored, and ROS topics are published and subscribed to as expected.

However, after modifying the ROS codebase and running `colcon build`, if the new nodes are executed without a full power cycle, the vehicle becomes unresponsive. The Jetson appears to send messages, and all relevant ROS topics are alive, yet the car does not move. Speed and steering commands appear to be published but are ignored.

> This failure does not indicate a full system crash or an obvious fault like a segmentation fault or kernel panic. Instead, the Jetson continues functioning and interacting with other peripherals correctly, suggesting that the issue lies in a more isolated component—specifically, the communication interface between the VESC and Jetson.

During field testing and parameter tuning (especially during race prep or controller calibration), this behavior introduces a significant delay. Restarting the Jetson takes several minutes and breaks the workflow of iterative development. In scenarios where tuning the pure pursuit controller's lookahead window or steering gain requires repeated test runs, even a 2-minute delay per iteration becomes a serious bottleneck.

## Root Cause Analysis

After multiple instances and testing on different F1/10th vehicles, a pattern—or lack thereof—began to emerge. Not all Jetson-VESC combinations experienced the issue, and no consistent hardware configuration was found to be the cause. However, the problem became repeatable under a specific condition:

1. ROS nodes are running and controlling the car.
2. Code is modified, and `colcon build` is executed.
3. New launch files are executed without restarting or re-initializing the hardware.
4. The VESC becomes unresponsive, despite being powered and detected.

System logs and ROS topic echoing were used extensively to trace the source. The `/vesc/commands/motor` topic was being published, and the `/vesc/sensors/core` topic showed no major deviation. However, inspection of lower-level serial logs showed that the USB communication interface was returning stale or no data.

### Communication Layer Bottleneck

The VESC communicates with the Jetson over a USB serial link. This interface, while typically robust, can enter a **buffer-locked** or **error state** if shutdown is not handled gracefully. This typically occurs when ROS nodes are terminated abruptly (e.g., via Ctrl+C) or crash mid-execution, which causes an unexpected surge of outgoing messages without proper port closure. The USB serial driver fails to recover cleanly, and the buffer within the VESC or Jetson becomes unrecoverable via software alone.

In embedded systems, such issues are well-known: a flooded or "orphaned" communication bus can silently drop packets, get stuck in error-retry loops, or even disable itself to prevent electrical damage. In this case, it's likely that the serial port associated with the VESC enters a non-operational state, which is not reset unless the USB bus is physically disconnected or the entire system is rebooted.

> Embedded systems with real-time communication buses often require explicit close-handling routines, especially in serial communication using USB CDC or UART bridges.

## Resolution Strategies

Given the nature of the issue, both software-level and hardware-level fixes are available. Addressing the root cause proactively can dramatically improve development cycle time and system reliability.

### Software-Based Fixes

1. **Graceful Shutdown Handling in ROS Nodes**  
   Modify your ROS node (especially the one publishing to `/vesc/commands/motor`) to catch termination signals such as `SIGINT` or `SIGTERM`. Upon receiving these, the node should:
   - Stop publishing new messages.
   - Close the serial port cleanly using appropriate APIs (e.g., `serial_port.close()`).
   - Wait for acknowledgment or timeout before exiting.

   Example in Python:
   ```python
   import signal
   import sys

   def signal_handler(sig, frame):
       print("Shutting down cleanly...")
       serial_port.close()
       sys.exit(0)

   signal.signal(signal.SIGINT, signal_handler)
   ```

2. **Rate Limiting Publishers**  
   Use ROS QoS policies (in ROS 2) or set rate limits manually (in ROS 1) to avoid flooding the buffer under high-frequency commands. This is especially important during `rosbag` playback or simulation.

3. **Error Detection Logging**  
   Enhance your node to monitor the status of the USB device file (e.g., `/dev/ttyUSB0`). Implement a timeout or watchdog system that logs errors when no acknowledgments are received within expected time windows.

### Hardware-Based Fixes

1. **USB Reset via Disconnect/Reconnect**  
   The most direct method to restore buffer integrity is to unplug and plug back in the USB cable connecting the Jetson and the VESC. This causes the operating system to reinitialize the USB driver stack, resetting the communication buffer and allowing fresh enumeration.

2. **Inline USB Switch**  
   For more convenience, especially in trackside environments, consider installing a USB kill switch or an inline toggle module. This allows buffer resets without physically removing cables—saving wear and tear on connectors.

3. **Powered USB Hub**  
   In some cases, using a powered USB hub can isolate brownout or current fluctuation issues that contribute to communication instability, especially when peripherals share the same power rail.

## Summary

Debugging communication between embedded systems like the VESC and Jetson Xavier NX requires careful observation of not just functional behavior but also how and when faults occur. In this case, the core issue lies in the USB serial interface between the Jetson and VESC becoming stuck in a buffer overload or error state, usually triggered by improper ROS shutdowns.

Key takeaways:
- Avoid abrupt node terminations—always handle shutdowns gracefully.
- Consider implementing software checks for port health and reset strategies.
- Use physical resets (USB disconnects) only when necessary, and automate if possible.
- Document the issue to help others in your team recognize and handle it efficiently.

## See Also:
- [F1/10th Platform Setup Guide](/f110-setup)
- [ROS Serial Communication Tips](/ros-serial)
- [Jetson Xavier NX UART/USB Handling](/jetson-usb-debug)
- [Pure Pursuit Controller Tuning](/pure-pursuit)

## Further Reading
- https://vesc-project.com/
- https://wiki.ros.org/rosserial
- https://elinux.org/Jetson/HW
- https://docs.nvidia.com/jetson/

## References
- Vedder, B. “VESC Project Documentation,” [Online]. Available: https://vesc-project.com/
- ROS Wiki. [Online]. Available: http://wiki.ros.org/
- NVIDIA Developer Documentation. [Online]. Available: https://developer.nvidia.com/embedded/jetson-xavier-nx