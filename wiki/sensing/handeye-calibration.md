---
date: 2025-04-26
title: Hand-Eye Calibration
---

**This is a tutorial for estimating the frame transformation between an image frame and an operating frame by using a third reference frame. An application is to estimate the transformation between pixel coordinates to end effector coordinates using an Aruco marker pose as reference. ROS has been chosen as the framework for this process due to its functionality that facilitates synchronized parallel communication. While most existing packages use ROS1, this tutorial uses ROS2. The entire workflow, from scene setup, data capture, computation and integration has been covered in this tutorial.**

## Hand-Eye Calibration

### Different Frames
1. Image Frame (Pixel Space)
2. Target Frame (Operation Space eg, base frame of manipulator or end-effector frame)
3. World Frame (Global Frame: usually set as the operating frame)

### The Algorithm

This package uses the method introduced by Lenz and Tsai in 1989. This is a data-driven method and it was observed that around thirty images are required for this method to work reliably.

- [Documentation](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99)
- [Original Research Paper](https://ieeexplore.ieee.org/document/34770)
- [GitHub Package](https://github.com/SNAAK-CMU/handeye_calibration_ros2)

### This Setup

For this tutorial, ROS2 will be used as the environment for its functionality that makes it easy to define frames and transformations using a transformation tree.

1. Image Frame: Realsense Camera Frame (ROS TF Frame: `camera_color_optical_frame`)
2. Target Frame: Base Frame of manipulator (ROS TF Frames : `base_link: "panda_link_0"`; `ee_link: "panda_hand"`)
3. World Frame: Aruco marker pose (From Aruco marker detection)

### This Process

The GitHub package has detailed instructions on installation and setup. The parameters in the file `handeye_realsense/config.yaml` need to be rewritten with the ROS2 topic and frame names of your system. 

## Summary
1. Keep in mind that the manipulator's poses must be as different as possible when sampling data in order to get a generalized result. If possible, put your manipulator in guide mode and move to the poses yourself, as this repository does not include a random pose generator. 
2. Not moving the Aruco marker's position yields better results. 
3. Configuring a random pose generator would require defining your workspace in a planning framework such as MoveIt! and generate random, collision free poses where the aruco pose is in the field of view of the camera. 
4. Ensure that the `child frame` specified in the config is the frame on which images are published. If not, set the child frame as the camera frame and chain together an intrinsic transformation to the image frame with the extrinsic transform from the target frame you will get from this process. This process has been described in detail on the README of the repository.

## See Also
- [Camera Calibration](/wiki/sensing/camera-calibration/)

## Further Reading
- [Original GitHub Repository](https://github.com/shengyangzhuang/handeye_calibration_ros2)

## References
1. https://github.com/shengyangzhuang/handeye_calibration_ros2
2. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99
3. https://docs.opencv.org/3.4/d0/de3/citelist.html#CITEREF_Tsai89
4. R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration," in IEEE Transactions on Robotics and Automation, vol. 5, no. 3, pp. 345-358, June 1989, doi: 10.1109/70.34770.
