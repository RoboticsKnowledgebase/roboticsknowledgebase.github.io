---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-05-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Integrating KUKA LBR Manipulators with MoveIt2 
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
This tutorial walks through setting up and programming a KUKA industrial robotic arm using KUKA's proprietary software environment. The goal is to help readers understand how to configure the robot, write and deploy basic motion programs, and address common issues encountered during integration. The tutorial assumes no prior experience with KUKA systems but does expect general familiarity with manipulators, basics of ROS and the general idea of MoveIt. By the end, readers will have the knowledge required to begin controlling a KUKA LBR arm for basic automation tasks.

## Background 
The LBR-Stack is an open-source suite of ROS/ROS2 packages built around KUKA’s Fast Robot Interface (FRI) to support LBR IIWA and Med robot arms in simulation and on real hardware. At its core is the `fri` package (CMake support for KUKA’s FRI client), and on top sit higher-level packages for ROS integration. In ROS2, these include the URDFs for LBR Med/IIWA (`lbr_description`) , the FRI ROS2 interface (`lbr_fri_ros2`), controllers (`lbr_fri_ros2_control`), MoveIt configuration generator (`lbr_moveit_config`), and a set of launch files (`lbr_bringup`) allowing easy terminal access. All of these can be found on the [lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack). Remember to follow the instructions given in the repo readme, these will give you two more packages called `lbr_fri` and `lbr_fri_idl`(Interface definition language messages for FRI). These provide the core functionality needed to bring up an LBR arm in MoveIt: you get the robot description and controllers from the LBR-Stack rather than writing them from scratch. If you plan to add a custom end effector, you can ignore the moveit package completely as this tutorial will guide you on adding a custom end effector to your arm.

**Note:** This tutorial does not cover integrating an active end-effector via KUKA’s native hardware interfaces or IO protocols. If your application requires direct integration through the KUKA flange, this is outside the scope of what’s covered here. Instead, this guide focuses on customizing the arm setup for passive end effectors or active ones that are controlled independently (e.g., via a separate microcontroller interface).

## Step 1: Defining your end-effector 

Assuming you've designed a custom end-effector in your preferred CAD software, your first step is to define it in URDF and incorporate it into the robot's description. Most CAD tools (e.g., SolidWorks, Fusion 360) offer plugins or exporters that generate URDF or Xacro files directly. Use those to export your end-effector geometry. Next, integrate the end-effector model into the `lbr_description` package. Navigate to:`lbr_description/urdf/med7/med7_description.xacro` In this file, you'll need to append a fixed joint that connects your end-effector to the flange of the arm (i.e., link 7). Add the following snippet:
```
<joint name="${robot_name}_joint_ee" type="fixed">
    <parent link="${robot_name}_link_7" />
    <child link="${robot_name}_link_ee" />
    <origin xyz="0 0 0.189" rpy="0 0 0" />
</joint>
```
Then replace or define the following block with the URDF link corresponding to your end-effector:
```
<link name="${robot_name}_link_ee">
    <!-- Your end-effector geometry and inertial parameters go here -->
</link>
```
Make sure to adjust the `origin` tag's `xyz` and `rpy` values to reflect the actual physical offset and orientation of your end-effector relative to the flange. You can verify alignment in RViz after launching the robot model.

## Step 2: MoveIt Configuration

Now that your custom end-effector is integrated into the robot description, it’s time to configure MoveIt to recognize and plan for the modified arm. The easiest way to do this is with the MoveIt Setup Assistant (MSA), which generates a full configuration package based on your URDF/Xacro and automatically sets up kinematics, planning groups, and controllers.

### 1. Launch the MoveIt Setup Assistant

Make sure your Xacro is fully compilable into a URDF. Then run:

```bash
roslaunch moveit_setup_assistant setup_assistant.launch
```

Load your modified robot by pointing to the description file (URDF or Xacro). Ensure all dependencies resolve correctly (no missing meshes, links, or macros).

### 2. Configure Planning Groups

* Define a planning group for the robot arm (e.g., `arm`), including all 7 joints.
* Optionally define an `end_effector` group for the link you just added.
* You can also set up virtual joints if simulating with a fixed base or using Gazebo.

### 3. Add Pose References

Define a default pose for the arm (like `home` or `ready`) which can be helpful during development. You can also import poses from an existing demo or RViz state.

### 4. Configure the End Effector

In the "End Effectors" tab, attach the new link (`*_link_ee`) to the planning group and parent link (`*_link_7`). Even if it’s passive or independently actuated, this is required for MoveIt to reason about it in motion planning.

### 5. Configure Controllers

This is where LBR-Stack starts to differ from more generic robot setups. You won’t define a full controller YAML by hand. Instead:

* Use the `lbr_ros2_control` package to provide compatible `ros2_control` hardware interfaces.
* If using ROS1, you can refer to the equivalent `lbr_control` or bridge it appropriately.

### 6. Save and Generate

Once everything is set up, generate the MoveIt config package (e.g., `my_lbr_moveit_config`). This will include:

* SRDF file with planning groups and links
* Kinematics plugin configuration
* Controllers and launch files
* Move group configuration

You can now use this package to bring up the robot in RViz, plan paths, and visualize trajectories with your custom end-effector.

I highly recommend going through the [MoveIt Setup Assistant guide](https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html#step-2-generate-self-collision-matrix) as this tutorial is too short to cover all details needed to make a complete config package.

```
src/
├── manipulation/
│   ├── fri/
│   ├── lbr_bringup/
│   ├── lbr_description/
│   ├── lbr_fri_idl/
│   ├── lbr_fri_ros2/
│   ├── lbr_ros2_control/
│   └── my_lbr_moveit_config/
└── other_subsystems/
```

From here, you're ready to either script motion or integrate the robot into a more complete application node.

## Next Steps

Now that your basic package is set up, try running sample commands from `lbr_bringup` as given in the documentation. You should be able to load your arm in MoveIt and perform basic GUI-based motion planning. Try experimenting with the OMPL planner to confirm if everything is functioning correctly. A useful way to verify correctness is to load visual axes for all links and inspect the end-effector axis to ensure it aligns as intended. `lbr_bringup` offers both simulation and hardware-based launch files. If you complete the hardware setup as described [here](https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_fri_ros2_stack/doc/hardware_setup.html#), you should also be able to control the actual robot.

If everything works well, congratulations — you’ve successfully integrated your own end effector with the KUKA arm.

If not, feel free to retrace the steps in this tutorial or consult the official documentation linked above. A personal recommendation: don’t waste time searching for YouTube tutorials. I spent hours doing that and found little of value. Your go-to references should be the GitHub repositories and official docs.

**Note:** This tutorial is not meant to be exhaustive. Many minor steps and configuration details have been omitted intentionally to provide a clear starting point without overwhelming the reader. 

## Summary
This tutorial walked through the core steps needed to integrate a custom end effector with a KUKA LBR Med/IIWA arm using the `lbr-stack` and MoveIt. By customizing the robot description, setting up a dedicated MoveIt config package, and using the provided bringup tools, you can get your modified robot running in both simulation and hardware. While many fine details have been abstracted away, you now have a structured foundation to build on for more complex applications like task planning, Cartesian control, or perception-driven manipulation.

## Further Reading
- [KUKA FRI (Fast Robot Interface) Documentation](https://lbr-stack.readthedocs.io/en/latest/fri.html)
- [KUKA LBR IIWA Documentation](https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa)
- [MoveIt Documentation](https://moveit.picknik.ai/)
- [LBR-Stack Documentation](https://lbr-stack.readthedocs.io/en/latest/index.html#)
- [ros2_control Documentation](https://index.ros.org/p/ros2_control/)
- [Explore B.O.N.E.Parte: My Capstone Project That Enhanced My Understanding of KUKA LBR and MoveIt](https://github.com/KNEEpoleon/Boneparte/)