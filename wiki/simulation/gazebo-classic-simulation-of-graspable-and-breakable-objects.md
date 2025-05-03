---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-05-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Gazebo Classic Simulation of Graspable and Breakable Objects
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

# Motivation 

In order to fulfill our SVD demonstration requirements, a simulator was required to test out bimanual manipulation due to the lack of a second physical arm. While an out-of-the-box solution for a Gazebo Classic simulator is already provided by the xarm-ros repository, which additionally provided the option of attaching one of their officially supported end effectors to the end effector of a single simulated arm, several functionalities were required that needed to be addressed:

* The simulation must include a detachable mechanism, which enables the pepper peduncle to be cut, freeing the pepper from the anchor in world position.

* The end effectors should be attached to the last joint of both arms. The control interface of either end effector should not affect the implementation of the motor control module (and should behave as if the real motors are attached). The planner should take the end effectors into account when planning.

* The pepper should be robustly grasped by the gripper end effector and not cause physics engine glitches or randomly fall from the hand. 

## Alternatives 

Our team had several choices to start exploring:

* Using a completely new simulator (such as Mujoco). This is discouraged since a partially working solution was already implemented in Gazebo Classic, and creating a simulation from scratch would result in significantly more work than previously expected, causing major hiccups in progress.

* Migrating the existing simulation stack to Ignition Gazebo, which replaces the Gazebo Classic engine. This was an option to be considered due to the fact that the Ignition Gazebo simulator officially features a world in which objects may be detached when a message is published (https://gazebosim.org/api/sim/9/detachablejoints.html). This is explored briefly, but was later abandoned due to 1) the amount of work potentially required to convert all files currently working with Gazebo Classic into Ignition Gazebo standards, and 2) the additional time and logistics required to create a custom plugin that senses the force being applied to the peduncle/fruit, and publishing a message based on the force reaching a limit. Additionally, due to the fact that we are using ROS 1 (Noetic), Ignition Gazebo is only somewhat supported (see https://gazebosim.org/docs/latest/ros_installation/). 

* Using Gazebo Classic. While the engine is due for deprecation around January 2025, the community support for Gazebo Classic is overwhelmingly more comprehensive than other options, and the xarm_ros repository files already work with Gazebo Classic. Furthermore, with the files associated with their officially supported gripper, development can be significantly improved by referring to existing known solutions for end effectors and grasping. This was the option our team ended up pursuing.

## Development Journey

First of all, the end effectors had to be included in the scene and attached to the arms, and to do this, we leveraged the Solidworks to URDF extension (https://wiki.ros.org/sw_urdf_exporter) to export our CAD assembly into an URDF file. The documentation to do this is available at (https://www.notion.so/Turning-Assemblies-into-URDF-for-RViz-Gazebo-1b6cedb4810780728bdec28add6a1891?pvs=4).

 In short, the assembly was significantly simplified, a base link (where the end effector attaches to the arm) is specified, revolute joints were defined and tested, and the files were exported (URDF and STL).

When the URDF files are exported and work as expected, the file structures in xarm_ros were then mimicked with the newly created gripper/cutter URDFs. This included defining several xacro files, including transmission, gazebo, urdf definitions, as well as an overall xacro file importing other xacro files. Additionally, joint controllers were defined in xarm_controller (PID position controllers) in order to move the joints inside Gazebo using gazebo_ros_control. This was verified by publishing to the joint controller channels with desired joint angles and verifying that the joints move in rviz and gazebo (since this infrastructure was already set up).

## Gazebo Classic plugins

To address the other two points regarding the simulator design, two plugins were found to solve 1) detachable joints and 2) object grasping stability. 

After some research, a native plugin called libBreakableJointPlugin was found, which allows the definition of a breaking force threshold (beyond which the fixed joint breaks). Some manual tuning was done to ensure when the cutter blades collide into the peduncle the force threshold was met, and to ensure that other forces such as small bumps into the plant or gravity did not trigger this threshold.

```xml
<joint name="breakable" type="revolute">
      <parent>peduncle</parent>
      <child>fruit</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0.0</lower>
          <upper>0.0</upper>
        </limit>
        <use_parent_model_frame>true</use_parent_model_frame>
      </axis>
      <physics>
        <ode>
          <erp>1</erp>
          <cfm>1</cfm>
        </ode>
      </physics>
      <sensor name="force_torque" type="force_torque">
        <always_on>true</always_on>
        <update_rate>1000</update_rate>
        <plugin name="breakable" filename="libBreakableJointPlugin.so">
          <breaking_force_N>10</breaking_force_N>
        </plugin>
      </sensor>
    </joint>
```
  
 Secondly, referring to the existing code in xarm_ros, a third-party library called libgazebo_grasp_fix was found (https://github.com/JenniferBuehler/gazebo-pkgs/wiki/The-Gazebo-grasp-fix-plugin), which fixes the object to the gripper given threshold conditions and lets go of the object when the force applied on the object is less than a threshold. This allowed the specification of multiple links (in our case, the gripper fingers and base) to be considered, along with some other options:

```xml
<xacro:macro name="vader_gripper_grasp_fix" params="prefix:='' arm_name:='xarm' palm_link:='link_base'">
    <gazebo>
      <plugin name="gazebo_grasp_fix" filename="libgazebo_grasp_fix.so">
        <arm>
          <arm_name>${arm_name}</arm_name>
          <palm_link>${palm_link}</palm_link>
          <gripper_link>thumb_1</gripper_link>
          <gripper_link>fing_1</gripper_link>
          <gripper_link>fing_2</gripper_link>
        </arm>
        <forces_angle_tolerance>110</forces_angle_tolerance>
        <update_rate>30</update_rate>
        <grip_count_threshold>2</grip_count_threshold>
        <max_grip_count>3</max_grip_count>
        <release_tolerance>0.01</release_tolerance>
        <disable_collisions_on_attach>false</disable_collisions_on_attach>
        <contact_topic>__default_topic__</contact_topic>
      </plugin>
    </gazebo>
  </xacro:macro>
```

## Conclusion 

With these plugins and a large amount of infrastructure work, Gazebo Classic was successfully configured to simulate a breakable pepper, import URDFs of both end effectors, and allow stable grasp of the pepper in the gripper hand. See our teaser (https://youtu.be/Uqwh7hMD_l8?si=i2ckLbr7Mejvx8_g&t=135) for a demonstration of the results!
