---
title: ROS Mapping and Localization
---
## Mapping
To map the environment, there are many ROS packages which can be used:
1. #### [Gmapping](http://wiki.ros.org/gmapping)
  - Gmapping requires odometry data from the mobile robot. So, if one has odometry data coming from the robot, Gmapping can be used.
2. #### [Hector Mapping](http://wiki.ros.org/hector_mapping)
  - The advantage of using Hector Mapping is that it does not need Odometry data and it just requires the LaserScan data. Its disadvantage is that it does not provide loop closing ability but it is still good for real-world scenarios specially when you do not have odometry data.
  - Even if one has odometry data, Hector Mapping is preferred over Gmapping. Hector Mapping also gives good pose estimates of the robot.
  - Using Hector Mapping, one can create a very good map of the environment. The other
  option is to generate a map in softwares like Photoshop. However, one should make sure to have a proper resolution while making a map in Photoshop.

## Localization
#### [AMCL](http://wiki.ros.org/amcl)
For localization of the robot, the ROS package of AMCL (Adaptive Monte Carlo
Localization) works well. It is straightforward to run the AMCL ROS package. AMCL can not handle a laser which moves relative to the base and works only with laser scans and laser maps. The only thing which should be taken care of is that it requires an odometry message. There are different ways to generate this odometry message. The odometry data can be taken from wheel encoders, IMUs, etc.. which can be used to generate the odometry message which can be supplied to AMCL. Another neat trick is to use pose obtained from the Hector mapping to generate an odometry message which can be then supplied to AMCL. If Hector mapping pose
is used to generate odometry message, then no external odometry is required and the result is
pretty accurate.
