---
date: 2026-05-08 # YYYY-MM-DD
title: Franka Emika Panda
---
![Franka Emika Panda](/assets/images/panda.png)

The Franka Emika Panda has quickly become one of the most influential research robot arms in modern robotics. Over the last few years, it has appeared everywhere from imitation learning papers and reinforcement learning benchmarks to manipulation research labs and startup prototypes. Its combination of high-quality torque control, integrated force sensing, safety-oriented design and comparatively low cost made it one of the default manipulators for academic robotics research.

With the platform now effectively end-of-life (EOL) and Franka Robotics shifting focus toward newer products and enterprise solutions, the once-active ecosystem around the Panda has become fragmented. A large amount of useful documentation exists only across scattered GitHub repositories, outdated tutorials, archived forum posts, etc. Even simple tasks can become frustrating without sufficient background context.

This article aims to consolidate the essential concepts needed to understand and operate the Franka Panda today. Rather than focusing purely on theory, the goal is to bridge the gap between the robot’s low-level architecture and practical modern control frameworks such as FrankaPy:-

## Franka Emika Panda software stack

1. **User Code & High-Level Wrappers (e.g. FrankaPy)** - This is where the user typically writes application logic. Instead of managing real-time loops and complex C++ pointers, the user can specify commands using high-level APIs. Libraries like FrankaPy act as the client-side interface, abstracting away the math.

2. **Middleware & State Management (ROS / franka_ros / franka-interface)** - This middle layer translates discrete high-level commands into continuous trajectories and manages state machines. `franka_ros` provides the actual controllers that handle gravity compensation, joint limits and collision behavior, exposing them as ROS topics and services.

3. **The Low-Level Driver (libfranka)** - This is the official C++ library provided by the manufacturer. It sits at the core of the workstation and calculates the raw joint torques, positions or velocities. Crucially, running `libfranka` reliably requires a computer configured with a Real-Time Linux Kernel (PREEMPT_RT) to guarantee a strict 1 kHz (1ms) control loop without interruption.

4. **The Network Bridge (Franka Control Interface - FCI)** - `libfranka` communicates with the robot's external Control Box over a dedicated Ethernet cable using UDP. The FCI is the feature that enables this rapid, bi-directional communication at 1000 packets per second.

5. **Hardware (Robot Controller & Arm)** - The Control Box receives the digital signals, translates them into electrical currents for the motors in the Panda's joints and continuously reads back high-resolution joint states, returning them up the stack. 


### Hardware Configuration and Safety Protocols

When operating the physical hardware, operators must remain clear of the table workspace and maintain constant access to the emergency stop at all times.

#### State Management via Indicator Lights
The robot's current operational state is communicated through the LED indicator on the base:-

* **Yellow (Locked):** The robot's joints are physically locked, and it cannot be moved. Unlocking requires accessing the Franka web interface.
* **White (Manual Mode):** Programmatic control is disabled. The robot can be safely guided by hand by pressing the grey buttons near the robot hand. This mode is achieved by pressing down on the e-stop.
* **Blue (Program Mode):** The robot is active and listening for programmatic commands. Manual movement is disabled. This state is achieved by twisting and releasing the e-stop.
* **Pink (Minor Error):** Indicates an attempt to manually force the robot while it was active in Blue mode. Resolve this by pressing down on the e-stop to return the robot to Manual mode.
* **Red (Critical Error):** Triggered by a significant collision. Requires a full restart to recover and move the robot again.

#### Manual Manipulation (Guide Mode)
Very useful for bringing the robot back to safe configurations:-
1. Hit the emergency stop button to transition the robot to Manual (white) mode.
2. Gently squeeze the opposing grey buttons on the wrist flange. Note: These are deadman switches; if you press too firmly, it will stop.
3. Reposition the arm. The required force should be minimal, similar to holding a cup of water.
4. Release the wrist buttons and twist the e-stop button to re-engage Blue mode.

#### System Initialization and Shutdown Sequence
**Startup Protocol:**
1. Flip the ON switch on the Franka Control Interface. Turn on both the Control PC and the User PC. Wait for the yellow light on the arm to stop blinking.
2. SSH onto the Control PC from the User PC (`ssh -X [username]@[hostname_or_IP]`).
3. Launch Google Chrome in the terminal and navigate to `172.16.0.2` to access the web interface.
4. Press "Click to unlock" to release the joints. The robot is ready to receive commands once the indicator lights turn blue.

**Shutdown Protocol:**
Hardware must be reliably returned to a known state. 
1. Reset the robot to its home position.
2. Close the terminals related to the Control PC to shut off the remote server.
3. Execute a shutdown via the web interface menu on the Control PC.
4. Wait a full minute for the robot to shutdown before flipping the physical switch on the FCI.

## Remote System Management (Desk API)
For more advanced automation or headless setups, you can interact with the robot’s "Desk" software using REST API calls. These commands are useful for scripting the initialization sequence without manually opening a browser to the web interface.

The following endpoints can be triggered via `curl` or Python's `requests` library:

* **Open Brakes:** Physically unlocks the robot joints for movement.
  `POST /admin/api/open-brakes`
* **Close Brakes:** Locks the joints; recommended for parking or long-term safety.
  `POST /admin/api/close-brakes`
* **Clear Safety Violations:** Resets internal error states (e.g., recovery from minor collisions) without requiring a full system reboot.
  `POST /admin/api/clear-errors`
* **Activate FCI:** Switches the robot to "External Control" mode, which is required to listen for commands from `libfranka` or ROS.
  `POST /admin/api/activate-fci`
* **Deactivate FCI:** Returns the robot to manual "Desk" control mode.
  `POST /admin/api/deactivate-fci`
* **Reboot System:** Performs a soft reboot of the Control unit.
  `POST /admin/api/reboot`

> ### Warning on Factory Resets
> Performing a factory reset on the Panda arm will wipe the **FCI license and configuration files**. Only attempt a reset if you have the original system image and license files provided at the time of purchase. 
> * **For FR3:** Documentation and recovery files are available at [Franka Documents](https://franka.de/documents).
> * **For Panda:** Limited open-source resources are available via the [Franka Robotics GitHub](https://frankarobotics.github.io/). 
> * If these files are lost, you must contact the Franka Robotics developer forums or your local distributor for support.

## Troubleshooting and Performance Tuning
Operating the Panda via FrankaPy and the FCI involves strict timing and safety constraints.

### Communication Failures
* **Real-Time Heartbeat:** If the 1kHz control loop is interrupted, the robot will lock. Ensure no heavy non-real-time processes are running on the Control PC during execution.
* **Docker Connectivity:** If using a Docker environment, ensure `roscore` is initialized in a separate terminal before launching the `start_control_pc.sh` script.
* **Web Interface Hangs:** If the Google Chrome instance or SSH connection hangs, use `killall google-chrome` or restart the terminal session.

## References
* O. Kroemer, "Robot Autonomy Lab (16-662)," Carnegie Mellon University, 2026.
* Franka Robotics GmbH, "Franka Control Interface (FCI) Documentation," 2026.
* Franka Robotics GmbH, "Franka World and Desktop Documentation," [Online]. Available: https://frankarobotics.github.io/.
* IAM Lab, "FrankaPy: A High-Level Python Interface for Franka Emika Panda," [Online]. Available: https://iamlab-cmu.github.io/frankapy/index.html.
