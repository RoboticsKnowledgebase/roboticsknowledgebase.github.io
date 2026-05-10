---
date: 2024-12-05
title: Sensing
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

The "Sensing" section serves as an introduction to a broad spectrum of sensory technologies critical for robotic perception, navigation, and interaction. It outlines tools and techniques that empower robots to effectively sense their surroundings and make informed decisions.

This section dives into various sensing modalities such as GPS modules, fiducial markers, radar, thermal cameras, stereo vision, and many more. It provides details on their principles of operation, implementation, and practical use cases, along with helpful troubleshooting tips. Whether you’re working on autonomous navigation, precise localization, object detection, or environmental mapping, this section is designed to equip you with the foundational knowledge and resources to integrate these sensors into your projects.

### Key Subsections and Highlights

- **[Adafruit GPS](/wiki/sensing/adafruit-gps/):**
  Discusses the Adafruit Ultimate GPS module for precise localization, including its features, technical specifications, and usage tips. 

- **[Apple Vision Pro](/wiki/sensing/apple-vision-pro/):**
  Provides an overview on integrating Apple Vision Pro with robotics, covering object tracking, spatial coordinate frames, and ROS communication.

- **[AprilTags](/wiki/sensing/apriltags/):**
  Introduces AprilTags as a fiducial marker system for visual identification and 6D pose estimation. Explains their use in robotics and potential pitfalls.

- **[Azure Block Detection](/wiki/sensing/azure-block-detection/):**
  Outlines the use of the Azure Kinect for object detection, with applications like Jenga block assembly. Explains the step-by-step detection pipeline.

- **[Camera Calibration](/wiki/sensing/camera-calibration/):**
  Emphasizes the importance of calibrating cameras for minimizing errors and improving vision system accuracy. Includes references to key calibration resources.

- **[Camera-IMU Calibration using Kalibr](/wiki/sensing/camera-imu-calibration/):**
  Details the Kalibr library for simultaneous IMU and camera calibration, including example setups and tips for accurate calibration.

- **[Computer Vision Considerations](/wiki/sensing/computer-vision-considerations/):**
  Highlights key considerations when deploying computer vision in robotics, including lighting, frame rates, calibration, and error mitigation.

- **[Delphi ESR Radar](/wiki/sensing/delphi-esr-radar/):**
  Provides an overview of Delphi’s ESR radar for detecting objects and estimating their range, speed, and position.

- **[DWM1001 UltraWideband Positioning System](/wiki/sensing/ultrawideband-beacon-positioning/):**
  Covers the setup and calibration of the DWM1001 UWB system for accurate indoor positioning.

- **[Fiducial Markers](/wiki/sensing/fiducial-markers/):**
  Compares various fiducial marker systems like ArUco, AprilTags, and STag, listing their pros, cons, and ideal use cases.

- **[Hand-Eye Calibration](/wiki/sensing/handeye-calibration/):**
  Provides a tutorial for estimating the frame transformation between an image frame and an operating frame using ROS2 and Aruco markers.

- **[Intel Realsense](/wiki/sensing/realsense/):**
  Introduces Intel RealSense cameras and details SDK installation, ROS integration, calibration, and tuning methods.

- **[OpenCV Stereo Vision Processing](/wiki/sensing/opencv-stereo/):**
  Introduces OpenCV libraries for stereo vision, including camera calibration and 3D triangulation.

- **[Perception via Thermal Imaging](/wiki/sensing/thermal-perception/):**
  Discusses strategies to implement key steps in a robotic perception pipeline using thermal cameras, including depth estimation and metric recovery.

- **[Photometric Calibration](/wiki/sensing/photometric-calibration/):**
  Explains the need for calibrating camera sensors to accurately map light intensity to pixel values, and methods to achieve this.

- **[Point Cloud Library, 3D Sensors and Applications](/wiki/sensing/pcl/):**
  Discusses PCL’s features for processing 3D point clouds and its applications in object detection, segmentation, and mapping.

- **[RTK GPS](/wiki/sensing/gps/):**
  Explains how to achieve centimeter-level accuracy using RTK GPS systems, along with practical lessons and setup guidance.

- **[Reducing Sensor Noise in Thermal or Visual Imaging sensors](/wiki/sensing/sensor-noise/):**
  Explains various techniques for reducing noise in thermal and visual imaging sensors, including filtering and post-processing methods.

- **[Robot-Centric Elevation Mapping](/wiki/sensing/elevation-mapping/):**
  Explains robot-centric elevation mapping using the Grid Map library to create 2.5D maps centered around the robot, accounting for pose uncertainty.

- **[Robotic Total Station (Leica TS16)](/wiki/sensing/robotic-total-stations/):**
  Discusses the use of robotic total stations for high-precision 3D positioning and their applications in surveying and robotics.

- **[Robotics with the Microsoft Hololens2](/wiki/sensing/hololens-101/):**
  Introduces the Microsoft HoloLens 2 as an AR headset for robotics applications, focusing on accessing its onboard sensors (RGB, Greyscale, Depth, and IMU) using a Unity-Python API.

- **[Speech Recognition](/wiki/sensing/speech-recognition/):**
  Explores speech recognition as a robotic interface, including offline and online solutions, wakeword detection, and speech synthesis.

- **[STag](/wiki/sensing/stag/):**
  Presents the STag fiducial marker system for stable pose estimation, with details on implementation and experimental results.

- **[Thermal Cameras](/wiki/sensing/thermal-cameras/):**
  Examines the use of thermal cameras in robotics, including types of thermal cameras, calibration techniques, and debug tips.

- **[Tracking vehicles using a static traffic camera](/wiki/sensing/trajectory-extraction-static-camera/):**
  Describes a system for extracting vehicle trajectories using static traffic cameras, incorporating detection, tracking, and homography estimation.

### Resources

- [Adafruit GPS](https://www.adafruit.com/product/746)  
- [AprilTags](https://april.eecs.umich.edu/apriltag/)  
- [Azure Kinect](https://learn.microsoft.com/en-us/azure/kinect-dk/)  
- [Kalibr Documentation](https://github.com/ethz-asl/kalibr/wiki)  
- [OpenCV Documentation](https://docs.opencv.org/)  
- [PCL Official Documentation](http://pointclouds.org/documentation/)  
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/)  
- [FLIR Boson App](https://www.flir.com/support/products/boson/#Downloads)  
- [DRTLS Android App](https://apkcombo.com/decawave-drtls-manager-r1/com.decawave.argomanager/)  
- [ROS DWM1001 Driver](https://github.com/TIERS/ros-dwm1001-uwb-localization)  
- [ROS2 DWM1001 Driver](https://github.com/John-HarringtonNZ/dwm1001_dev_ros2)  
