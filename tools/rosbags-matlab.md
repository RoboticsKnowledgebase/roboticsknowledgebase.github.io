---
title: Rosbags In MATLAB
---
Sometimes you would rather analyze a rosbag in MATLAB rather than within the ROS environment. This page describes how to set up the Matlab infrastructure and includes a script example.

The following steps provide the parsing setup:
1. Type roboticsAddons and install the [Robotics System Toolbox Interface](https://www.mathworks.com/help/robotics/ref/readmessages.html) for ROS Custom Messages
2. Copy custom package(s) to a standalone folder
3. Check messages for redundant variables
4. Run rosgenmsg(folderpath) where folderpath is the path to the standalone folder above our custom packages
5. Follow directions in Matlab for editing javaclasspath.txt, adding the message to the path, restarting Matlab, and verifying success via rosmsg list

Basic rosbag parsing is as follows:
1. Run `rosbag(filepath)` on the [rosbag](https://www.mathworks.com/help/robotics/ref/rosbag.html) file of interest
2. [Select](](https://www.mathworks.com/help/robotics/ref/select.html)) only the topic of interest with `select(bag,'Topic','<topic name>')`
3. For straightforward values in a message, run `timeseries(bag_select,'<message element>')`
4. For more complicated messages, run `readMessages`

[This script](assets/parseRosbag.m) parses Rosbags and creates a `.mat` and `.csv` file from it given a custom definition. The particular message referenced is an aggregation of multiple messages, hence the recursive structure.
