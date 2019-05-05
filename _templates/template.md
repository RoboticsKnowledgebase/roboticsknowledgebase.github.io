---
title: IMU-Camera Calibration using Kalibr
---

## Introduction
This tutorial will help you in setting up the Kaliber library developed at ETH Zurich for combined IMU-Camera calibration. This library simultaneously computes the homogeneous transformation (denoted as **<sup>I</sup>H<sub>W</sub>**) between the camera and the world frame, and the homogeneous transformation (denoted as **<sup>C</sup>H<sub>W</sub>**) between IMU and the world frame. A 6x6 AprilTag gridmap is used as the landmarks for the calibration.
Above two transformations (<sup>I</sup>H<sub>W</sub> & <sup>C</sup>H<sub>W</sub>) can be used to compute the transformation between Camera and IMU i.e. <sup>C</sup>H<sub>W</sub> <sup>W</sup>H<sub>I</sub>. In this method the IMU is parameterized as 6x1 spline using the three degrees of freedom for translation & three degrees of freedom for orientation. Based on the raw acceleration & angular velocity readings from the on board IMU, the acceleration and the angular velocity is computed in terms of <sup>C</sup>H<sub>W</sub>.

This library needs an initial guess of <sup>C</sup>H<sub>W</sub>, homogeneous transformation which is first computed for each April tag using the PnP algorithm. Then the error between the predicted positions of the landmark based on the IMU reading and the observed position of the landmark using camera is minimized using Levenberg Marquardt optimizer. So, this library generates the following output: <br/>
(i) the transformation between the camera and the IMU <br/>
(ii) the offset between camera time and IMU time, d <br/>
(iii) the pose of the IMU w.r.t to world frame, I <br/>
(iv) Intrinsic camera calibration matrix, K for the camera <br/>

Going through this [paper's](https://furgalep.github.io/bib/furgale_iros13.pdf) section III B and IV will give deeper insights into the implementation.

## Requirement
1. A printout of 6x6 AprilTag 36h11 markers on A0 size paper. This can be downloaded from [here](https://drive.google.com/file/d/0B0T1sizOvRsUdjFJem9mQXdiMTQ/edit)
2. This [CDE](https://drive.google.com/file/d/0B0T1sizOvRsUVDBVclJCQVJzTFk/edit) package is the easiest and the fastest way to get the library running. All dependencies are packed within this package and no external dependencies are required.
3. ROS drivers for the IMU and camera. Pixhawk, Ardupilot are typically used for UAVs and UM6/7 is popular for ground robots like Husky. ZED, Asus Xtion Pro, Intel Realsense D435 are few popular stereo cameras.

## Installation
1. Download and extract the CDE package.
```
  tar xfvz kalibr.tar.gz
```
  Either you can run the tools directly from the cde-package folder or/and add the package folder to the system path using:
```
  export PATH="/cde/package/path:$PATH"
```
2. Printed AprilTag size is changed into the YAML file and this yaml file is copied inside the downloaded CDE package. <br/> Now make sure that both IMU & Camera publishes raw data and images respectively to ROS. Below command will list all the ROS topics published by the sensor.
```
rostopic list
```
Before creating ROS bag check if the sensor data is published properly.
```
rostopic echo /topic_name
```
Check the publishing rate of a topic. It is recommended that camera should run at 20 Hz and IMU at 200 Hz.
```
rostopic hz /topic_name
```
3. Further move the robot along all its degree of the freedom. For instance, the axes of the UAVs are translated in x,y & z direction & rotated in all the three directions for the proper calibration. Collect the IMU sensor’s raw measurement and camera frame for around 60 seconds. Before recoding camera data, ensure that the RGB images from the camera is converted into grayscale format. 

In this sample example Pixhawk sensor data is subscribed from the MAVROS raw_sensor message & camera frames are subscribed from the ZED sensor camera node. For other IMUs/Cameras only the ROS message name will change.

Start ZED sensor camera
```
roslaunch zed_wrapper zed.launch
```
and similarly start pixhawk
```
roslaunch mavros px4.launch
```
Below command will start the recording.
```
rosbag record -O camera-imu-data.bag /imu/data_raw /zed/left/image_raw /zed/right/image_raw
```
4. Check the recorded data by using following command.
```
rosbag play <your bagfile>
```
5. If all the required data is recorded properly, run following camera calibration command.
```
kalibr_calibrate_cameras --models pinhole-equi pinhole-equi --topics /zed/left/image_raw /zed/right/image_raw --bag camera-imu-data.bag --target aprilgrid_6x6.yaml
```
-models 
This command will generate new yaml file with camera instrinc calibration parameters.

After running above command successful There will be new folder created in the CDE folder with name 
```
kalibr_calibrate_imu_camera --cam chain.yaml --target aprilgrid_6x6.yaml --imu imu0.yaml --bag camera-imu-data.bag
```
6. Both calibrators write reports to the working directory containing the plots shown at the end of the calibration. Further a camchain.yaml has been written by the camera calibrator and is extended by the imu-camera calibrator with imu-camera transformations to the file camchain_cimu.yaml. 

## Few notes and important tips
1. The tool requires a rosbag with more than 1000 images, which leads to a ROS bag of few GBs. So, it is suggested to be run on the development machine instead of the on board SBC.
2. 

## References
Official Kalibr github repository
