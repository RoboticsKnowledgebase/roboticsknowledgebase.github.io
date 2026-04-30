---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2026-04-29 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: LIO-SAM with Velocity Undistortion & Dynamic Obstacle Filtering
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
LIO-SAM is a real-time LiDAR-inertial SLAM framework that estimates a robot’s pose and builds a 3D map by fusing LiDAR, IMU, and optionally GPS data. It uses factor graph optimization to combine scan matching, IMU preintegration, loop closure, and GPS constraints, producing accurate and consistent localization in large-scale environments.

Links:
- [Classic LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) 
- [Our Improved LIO-SAM](https://github.com/JoshuaTsai0520/LIO-SAM-Velocity/tree/ros2-velocity)

The detailed explanation of the LIO-SAM algorithm and its configuration settings can be found in the original paper and GitHub repository, so they will not be repeated here. The main purpose of this guide is to introduce the core setup steps and key features of our improved version.

After the point cloud map is built, we will also describe a simple approach for removing dynamic obstacles and outliers from the map. Our goal is to generate a static, clean, and high-quality 3D point cloud map.

# Dependencies
Our improved version is tested with ROS2 Humble on Ubuntu 22.04.

Refer to the [Dependencies](https://github.com/TixiaoShan/LIO-SAM/tree/ros2) in LIO-SAM's github ros2 branch.

Install [ROS2](https://docs.ros.org/en/humble/Installation.html):
```
sudo apt install ros-<ros2-version>-perception-pcl \
  	   ros-<ros2-version>-pcl-msgs \
  	   ros-<ros2-version>-vision-opencv \
  	   ros-<ros2-version>-xacro
```

Install [GTSAM](https://gtsam.org/get_started/):
```
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt install libgtsam-dev libgtsam-unstable-dev
```

Use the following script to install [Sophus](https://github.com/strasdat/Sophus.git):

```
bash install_sophus.sh
```

# Installation
Use the following commands to compile the package:

```
git clone https://github.com/JoshuaTsai0520/LIO-SAM-Velocity.git
cd LIO-SAM-Velocity
git checkout ros2-velocity
bash build_lio.sh
```

# Run the package
The source command is already written in the script:
```
bash run_lio.sh
```

Then play bag files:
```
ros2 bag play your-bag
```

# Core Configuration Setup

## Input Topic
- pointCloudTopic (sensor_msgs/msg/PointCloud2)
- imuTopic (sensor_msgs/msg/Imu)
- twistTopic (geometry_msgs/msg/TwistStamped) (Optional)
- gpsTopic (nav_msgs::msg::Odometry) (Optional)

## Distortion Function
- enable_distortion_function: Open if you want to use distortion correction algorithm
- use_imu: Use IMU data in distortion function
- use_velocity: Use velocity data in distortion function

## IMU Settings
- extrinsicTrans: Translation between LiDAR and IMU
- extrinsicRot, extrinsicRPY: Rotation between LiDAR and IMU

## Loop Closure
- loopClosureEnableFlag: Open loop closing
- historyKeyframeFitnessScore: ICP matching threshold, adjust it if loop closing performance is bad

# Distortion Correction Module
LiDAR motion distortion occurs when the sensor or robot moves while a frame is being captured. Since a rotating LiDAR does not measure all points at the same instant, different parts of the point cloud correspond to slightly different sensor poses. This can cause straight walls to appear curved, objects to shift position, and scan matching accuracy to degrade. The rotation distortion video is shown [here](https://drive.google.com/file/d/1KnMkAbJXtDpj2OJxOIoW3D4nRNeQGs_S/view?usp=sharing).

In LIO-SAM, IMU measurements are used to estimate the sensor’s rotational motion during each scan and correct, or “undistort,” the point cloud before mapping. However, translational motion of the sensor is not fully accounted for. When the vehicle moves at high speed, this translation-induced distortion can become significant and may degrade mapping quality.

To address this issue, we integrated vehicle speed information and introduced a 6-DOF motion distortion corrector that accounts for both rotational and translational motion along all axes. A similar approach can be found in [Autoware distortion corrector](https://autowarefoundation.github.io/autoware_universe/main/sensing/autoware_pointcloud_preprocessor/docs/distortion-corrector/) module.

As a result, the quality of the constructed map is significantly improved. Within each LiDAR frame, all points are transformed into a consistent coordinate frame, making the scan more geometrically rigid. The mapping results before and after distortion correction are shown below.

Before distortion correction:
![](/assets/images/state-estimation/Before_Undistortion.png)
After distortion correction:
![](/assets/images/state-estimation/After_Undistortion.png)

# Dynamic Obstacle Filtering
Dynamic obstacle filtering is applied after the initial map is constructed to remove points caused by moving objects, such as vehicles, pedestrians, or cyclists. These dynamic points may appear as ghost artifacts or inconsistent structures in the map because they do not belong to the static environment. By identifying and removing them from the generated point cloud map, the final map becomes cleaner, more reliable, and better suited for localization and navigation.

A simple way to remove these dynamic obstacles is through manual selection. We use CloudCompare to edit and refine point cloud segments in the map. For example, in a map generated by LIO-SAM, many unwanted dynamic obstacles and outliers may remain. In CloudCompare, the Segment tool can be used to select the areas that need to be removed, as shown below:
![](/assets/images/state-estimation/DOF1.png)

After making the selection, click Segment Out and then Confirm to separate that portion of the point cloud. However, the segmented part may still contain both dynamic obstacles and static points, such as the ground. Therefore, you should apply the Segment tool again to further isolate the dynamic objects:
![](/assets/images/state-estimation/DOF2.png)

After the second segmentation, you can obtain a static section with the dynamic points removed:
![](/assets/images/state-estimation/DOF3.png)

After applying multiple segmentation steps across the map, merge all remaining static point cloud segments into a single file. This produces a high-quality static map with dynamic obstacles removed:
![](/assets/images/state-estimation/DOF4.png)
The white points represent the filtered dynamic obstacles, while the colored points represent the remaining static map.

# Future Work
Several improvements can be added in future work. First, GNSS measurements can be integrated as an elevation drift constraint during mapping. This would help reduce vertical drift and improve map consistency, especially in large-scale outdoor environments.

Second, better visualization tools can be developed to make the mapping and filtering results easier to inspect. Clear visualization of trajectory quality, point cloud alignment, removed dynamic objects, and map consistency would help with debugging and evaluation.

Third, scan matching results can be evaluated before each frame is added to the map. Frames with poor matching quality, large residual errors, or unstable pose estimates could be discarded to prevent them from degrading the final map.

Finally, dynamic object removal can be automated using LiDAR-camera sensor fusion and object detection. Camera-based detection can identify dynamic objects such as vehicles and pedestrians, while LiDAR points associated with those objects can be removed from the point cloud map automatically. This would reduce manual editing and improve the scalability of the mapping pipeline.

