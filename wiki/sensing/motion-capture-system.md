---
date: 2026-04-29
title: Motion Capture Systems -- Setup Guide
---
This guide covers the setup and use of motion capture systems, including OptiTrack and Vicon, for robotics applications. It walks through software installation, hardware configuration, calibration workflow, and ROS 2 integration, and concludes with application-specific notes for humanoids, exoskeletons, and UAVs.

## Software Installation

### OptiTrack
Download **Motive** from [optitrack.com/support/downloads](https://optitrack.com/support/downloads). Motive handles calibration, rigid body tracking, and data streaming. It requires a Windows host PC (i7+, 32 GB RAM, dedicated GPU, gigabit NIC), and the NatNet binary is Windows-only with no Linux alternative available.

### Vicon
Download **Vicon Nexus 2** for biomechanics and movement science work. Sessions are organized through the built-in **proEclipse** database. Like Motive, Nexus is a Windows-only application.

### SDKs

**OptiTrack NatNet SDK:** Streams 6DoF rigid body poses, skeleton data, and marker positions over UDP (port 1510 for commands, 1511 for data). Bindings are available for C++, Python, MATLAB, and Unity.

**Vicon DataStream SDK:**
```
pip install vicon_dssdk
```

**ROS 2 (OptiTrack):**
```
mkdir -p ~/mocap_ws/src && cd ~/mocap_ws/src
git clone https://github.com/OptiTrack/mocap_msgs.git
git clone https://github.com/OptiTrack/mocap4ros2_optitrack.git
cd ~/mocap_ws && colcon build && source install/setup.bash
```

> The NatNet binary does not support ARM platforms (Jetson, Raspberry Pi, etc.). A separate x86-64 Ubuntu machine is required as a companion computer between Motive and the robot. WSL2 running on the Motive PC is not a valid substitute.

## Hardware Setup

### Camera Mounting
Mount cameras on rigid rails at 2 to 3 m height, angled 15 to 30 degrees downward. Every point in the capture volume must be visible from at least 3 cameras. Tripods are acceptable provided the pan-and-tilt head is fully tightened, as any camera movement after calibration invalidates the entire session. OptiTrack Prime Series uses **eSync 2** for hardware sync; Flex Series uses **OptiHub 2**. All cameras must share the same clock, otherwise timing jitter will degrade tracking quality.

### Varifocal Lens Tuning (Vicon Vero)
The three dials must be adjusted in the following order. Changing them out of sequence requires starting over from the beginning.

1. **Zoom** (dial closest to camera body): set based on volume size, ranging from 6 mm wide to 12 mm telephoto.
2. **Focus** (outer dial): adjust until markers appear as tight, bright circles with clean edges.
3. **Aperture** (middle dial): open for more light. At 300 Hz and above, increasing strobe intensity or opening the aperture is necessary to avoid underexposure.

### Networking
```
[Cameras] --> [Camera Switch] --> [Motive PC NIC 1]
                                        |
                                 [Motive PC NIC 2]
                                        |
                              [LAN Switch / Router]
                                        |
                           [Ubuntu Companion / Robot]
```
The camera network should be isolated from the general LAN, which is why the Motive PC requires two separate NICs. Assign static IPs to both interfaces (e.g., `192.168.1.100` for Motive, `192.168.1.50` for Ubuntu). Disable Windows Defender Firewall on the Motive PC for the private network, or explicitly open UDP ports 1510 and 1511. Firewall blocking is the most common reason ROS receives no MoCap data.

### Marker Placement
Each rigid body requires a minimum of 3 non-collinear markers, though using 4 to 6 improves robustness against occlusion. Marker placement should be asymmetric, as a symmetric triangle arrangement can cause Motive to misidentify orientation by 180 degrees. Apply matte black tape over reflective robot surfaces such as brushed aluminum or glossy carbon fiber to suppress false markers throughout the volume.

## Software Workflow

**Step 1: Boot and Go Live.** Open Motive or Nexus and click **Go Live**. Allow cameras to warm up for 5 to 10 minutes before beginning calibration, as IR emitters shift geometry as they reach operating temperature and calibration quality improves measurably after the warmup period.

**Step 2: Masking.** In Motive, navigate to the Calibration pane and click **Mask Visible**. Walk the perimeter of the volume so that all static reflections are captured, then apply. Masked reflections will change from white to blue dots in the camera views. In Nexus, open the Camera pane and click **Auto-mask**. Skipping this step is a common mistake, as unmasked reflections are treated as phantom markers during calibration and will silently degrade results.

**Step 3: Calibration.** In Motive, select the appropriate wand type (CW-500 for large volumes; CWM-250 or CWM-125 for precision work) and wave it slowly through the entire volume, covering corners, the floor, and the ceiling. The target result is an **Excellent** rating with residual error below 0.3 mm. In Nexus, set the wand count to approximately 10,000 in Advanced Settings and wave until collection completes; every camera must display a green box in the results.

**Step 4: Set Volume Origin.** In Motive, place the CS-400 calibration square at the desired origin and click **Set Ground Plane**. In Nexus, select floor markers in the 3D viewport and click **Set Volume Origin**. Verify that Z points upward and that axis directions match the robot's expected world frame before collecting any data.

**Step 5: Define Rigid Bodies.** In Motive, place the object in the volume with markers visible, Ctrl+click to select all markers on the object, then right-click and select **Rigid Body > Create From Selected Markers**. Use a descriptive name such as `base_link`. Record the **User Data ID** found under Rigid Bodies > Advanced, as this value is required when configuring the ROS driver.

**Step 6: Record.** Confirm that rigid bodies track cleanly in the viewport with solid outlines and no flickering, then press **Record** (`Ctrl+R`).

## Streaming to ROS 2

**Motive Streaming Settings** (Settings > Streaming):
```
[x] Broadcast Frame Data
Local Interface:    192.168.1.100
Transmission Type:  Unicast
[x] Stream Rigid Bodies
Up Axis:            Z-Up   (change from the Y-Up default; ROS uses Z-up convention)
```

**ROS 2 Driver** (`mocap_optitrack_driver_params.yaml`):
```yaml
mocap_optitrack_driver_node:
  ros__parameters:
    connection_type: "Unicast"
    server_address: "192.168.1.100"
    local_address: "192.168.1.50"
    server_command_port: 1510
    server_data_port: 1511
```

> If multicast is required, the `multicast_address` in this file must match Motive exactly. The driver defaults to `224.0.0.1` while Motive defaults to `239.255.42.99`, and these values do not match out of the box. Unicast is the recommended configuration.

```
ros2 launch mocap_optitrack_driver optitrack2.launch.py
ros2 topic echo /rigid_bodies
```

## Application Notes

### Humanoids
The standard paradigm is reference-motion tracking via reinforcement learning: human MoCap sequences are retargeted to the robot's kinematic structure, a policy is trained in simulation using a weighted reward over joint position, velocity, end-effector, and root pose error (following the DeepMimic formulation, Peng et al., 2018), and the resulting policy is then transferred to hardware. Common public datasets include **AMASS** (45+ hours, SMPL format) and **LAFAN1** (standard for retargeting evaluation).

Human proportions and joint limits do not map directly onto any humanoid platform. Three retargeting methods are in current use. **PHC/ProtoMotions** are SMPL-based retargeters with documented artifact issues on dynamic motions. **GMR** (Araújo et al., ICRA 2026, `arXiv:2510.02252`) is a five-stage pipeline supporting 17+ platforms with direct OptiTrack FBX export, making it the recommended bridge between Motive and RL training. **KDMR** (Georgia Tech, 2025, `arXiv:2603.09956`) enforces rigid-body dynamics and contact constraints, and outperforms GMR on dynamic feasibility.

Tracking policies are typically implemented with PPO and reference-state initialization. **BeyondMimic** (UC Berkeley, 2025, `arXiv:2508.08241`) represents the current state of the art for real-hardware transfer. **AMP** (Peng et al., SIGGRAPH 2021) uses a discriminator-based style reward as a principled alternative to frame-by-frame tracking.

### Exoskeletons
MoCap alone is not sufficient for exoskeleton control research. The standard pipeline combines MoCap with force plates and OpenSim to give the joint torque references that a controller will track:

```
MoCap (TRC) + GRF (MOT) -> OpenSim Scale Tool -> Inverse Kinematics
-> Residual Reduction -> Inverse Dynamics -> joint moments (.sto)
```

Capture the subject with a full-body marker set and synchronized force plate data. Scale OpenSim's **gait2392** model to the subject using a static standing trial; residual marker errors above 2 to 3 cm on major body segments indicate a poor scale. Following IK, **Residual Reduction (RRA)** must be run before computing IK, as computing joint moments on dynamically inconsistent kinematics produces meaningless results. Peak residual forces after RRA should fall below roughly 5 to 10% of body weight.

### UAVs and Drones
Operate at 200 to 360 Hz with high strobe intensity to prevent motion blur from fast platform dynamics. Mount markers on the arms rather than the body, as propellers frequently occlude body-mounted markers during flight. For PX4-based systems, publish MoCap pose data to `/fmu/in/vehicle_visual_odometry` and configure EKF2 to enable vision fusion with appropriate noise parameters.

## Summary
This guide covered the end-to-end process of setting up a motion capture system for robotics: installing Motive or Nexus, configuring cameras and networking, running the calibration workflow, streaming data to ROS 2, and applying the system to humanoid retargeting, exoskeleton control, and UAV flight. Follow the pre-session checklist below before each recording session to avoid common failure modes.

### Pre-Session Checklist
- [ ] Cameras warmed up for at least 10 minutes
- [ ] All cameras showing healthy status in software
- [ ] Room cleared of unnecessary reflective objects
- [ ] Masking applied; static reflections visible as blue dots in camera views
- [ ] Correct wand type selected in calibration software
- [ ] Full calibration completed with result rated **Excellent** or better
- [ ] Ground plane and volume origin set; axis directions verified
- [ ] Rigid bodies created, named, and User Data IDs recorded
- [ ] Motive streaming enabled with correct IP and Z-Up axis selected
- [ ] `ros2 topic echo /rigid_bodies` confirmed returning live data


## Further Reading
- OptiTrack Motive Documentation: <https://docs.optitrack.com>
- OpenSim Documentation: <https://opensimconfluence.atlassian.net>

## References
- Z. Araújo et al., "GMR: A General Motion Retargeting Pipeline for Humanoid Robots," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2026. arXiv:2510.02252.
- G. Liao, T. Truong et al., "BeyondMimic: Whole-Body Humanoid Control with Real-Hardware Transfer," UC Berkeley, 2025. arXiv:2508.08241.
- Z. Luo et al., "Experiment-free Exoskeleton Assistance via Biomechanics-aware Reinforcement Learning," *Nature*, 2024.
- Z. Luo et al., "Perpetual Humanoid Control for Real-time Simulated Avatars," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023.
- X. B. Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control," *ACM Trans. Graph. (SIGGRAPH)*, vol. 40, no. 4, 2021.
- X. B. Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills," *ACM Trans. Graph. (SIGGRAPH)*, vol. 37, no. 4, 2018.
- Y. Zhang et al., "KDMR: Kinodynamic Motion Retargeting for Humanoid Robots," Georgia Tech, 2025. arXiv:2603.09956.
