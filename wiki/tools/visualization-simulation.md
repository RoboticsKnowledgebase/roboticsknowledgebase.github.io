---
date: 2019-11-12
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. You should set the article's title:
title: Making Field Testing Easier Through Visualization and Simulation
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
If you have an outdoor system, field testing is going to make or break your validation demonstration. No amount of testing in the lab will expose as many issues with your system as 10 minutes of testing outside. Picking up your system and taking it somewhere else is difficult but you have to do it. Therefore, you want to make your time field testing as effective as possible and as painless as possible. You can do that by utilizing simulation before you head to the field and visualization while you are in the field. In this entry, we will explain the importance of visualization and simulation and provide an explanation of how to set up a Gazebo simulation and RVIZ visualization for a Husky as a case study. The setup explanation assumes that you are using ROS for your system, however, the explanation of visualization and simulation hold true regardless of your system setup.

## Visualization
Visualization allows you to know how your system understands its relationship to the world, getting insight into this is one of the best things you can do to make your life easier. Attempting to understand why your system is turning left instead of right or why it insists on driving into that wall over there is a fool's errand if you are only using print statements or by trying to interpret the stream of quaternions coming out of your IMU. It is something that everybody tries, and it is a terrible use of your time. Through visualization, you can instantly compare the orientation position output of your Kalman filter to your IMU GPS so you can see if you are properly fusing in all of the data. 

If you are using ROS, one of your first steps should be to set up an RVIZ configuration specific to your project. Your project is going to require a unique configuration to properly visualize it depending on your specific project details. So the first thing you need to do is determine what will help you monitor the state of your system and debug any issues. A shortlist of topics which I would recommend for a generic mobile robot includes: odometry, IMU, the robot’s base link TF (preferably with a robot model if you have one available), markers to represent any points of interest, and a path. But in theory, any sensor input you have and localization output should be visualized in RVIZ. All of these visualizations are built into RVIZ and can be used by simply connecting each one to the topic your robot is publishing. Additionally, if you have an outdoor robot, I highly recommend changing RVIZ’s background color to white or grey; when working under the sun, it is impossible to see anything on your monitor with RVIZ’s default black background (you can do the same thing with your terminal color). After configuring your environment, you should save the configuration and track it in your repository like any other file. If you don’t save a configuration file, you will have to go through this setup process every time you use RVIZ. You can also create a custom launch file from which to load your RVIZ configuration using the code snippet below:

```
<launch>
  <node type="rviz" name="rviz" pkg="rviz" args="-d $(find my_project)/params/my_project.rviz" />
</launch>
```
The code snippet above is launching an rviz node, from the rviz package. It then passes an argument which includes the absolute path to the file `my_project.rviz`, contained in the `params` directory of the ROS package `my_project`. For this to run properly, the package `my_project` needs to built on your computer, and the `devel/setup.bash` needs to have been sourced in your terminal.

## Simulation
Now that you have configured a visualizer for improved debugging, we will discuss some simple steps you can use to set up a simulator and how to use it to make your life easier. You will need to pick a simulator that matches the needs of your project and unfortunately, we will not be covering that level of simulator discussion this wiki entry. First I want to emphasize why it is important to simulate. As mentioned above, field testing is time-consuming and difficult, it should be reserved for testing your system’s hardware and integration in a non-lab setting. Field testing is not a time to run your software for the first time. If you have not already run your code in simulation and feel some level of confidence that it is going to work in the field, you are wasting your time and your teammates’ time by going on a field test. Beyond that, you are making your life harder than it needs to be.

We will be using a Gazebo simulation of a Husky, which is available via apt install. You can install the simulation yourself by following the instructions at this URL: </https://wiki.ros.org/husky_gazebo/Tutorials/Simulating%20Husky)>. This Husky will be our generic mobile robot acting as a stand in for your robot. Take a look at the topics being published by this robot, odometry (/husky_velocity_controller/odom), GPS (/navsat/fix), IMU (/imu/data), and you can even configure a laser range finder. With this simple install you already have everything you need to set up and begin debugging code for localization, motion control, obstacle avoidance, and motion planning, without needing to have any of your hardware in place. 


Depending on the simulator and hardware you are using, your robot may have different topic names for the same information. Since you will be regularly running your code both in simulation and on your robot you should develop a way of easily switching back and forth between these two setups. Again, assuming you are using ROS, I recommend setting up a single launch file with a “simulation parameter” accessible from the command line. Use this parameter to set up an option that allows you to remap topics appropriately depending on what you are doing. An example is below:

```
<launch>
  <arg name="sim" default="false" /> 
  <node pkg="robot_localization" type="navsat_transform_node" name="navsat_transform" clear_params="true">
    <remap from="gps/fix" to="rtk_gps" unless="$(arg sim)"/>
    <remap from="gps/fix" to="/navsat/fix" if="$(arg sim)"/>
  </node>
</launch>
```
The above code represents a launch file, with parameter `sim` whose default file is false. You can set its value when launching on the command line by appending the option `sim:=true`. The node being launched from the `robot_localization` package which is also a good resource. Finally, we are remapping the topic `gps/fix`, if `sim` is set to false, the topic will be set remapped to `/navsat/fix` and if `sim` is set to true it will be remapped to `rtk_gps`

## Conclusion
Using these simple tools can save you hours during the course of your project. Just remember, these are not a substitution for field testing; they are tools to make field testing more efficient.

## Further Reading
- Robot Localization <https://wiki.ros.org/robot_localization>