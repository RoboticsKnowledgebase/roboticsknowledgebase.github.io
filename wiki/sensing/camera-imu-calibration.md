---
title: IMU-Camera Calibration using Kalibr
---
This tutorial will help you in setting up the Kaliber library developed at ETH Zurich for combined IMU-Camera calibration. 
For better understanding, we have also included an example of Pixhawk (IMU) and ZED camera (stereo camera) calibration procedure. 

# Table of Contents
1. [Introduction](#Introduction)
2. [Requirement](#Requirement)
3. [Calibration Procedure](#Calibration-Procedure)
4. [Few notes and important tips](#Few-notes-and-important-tips)
5. [References](#References)

## Introduction
Kalibr library simultaneously computes the homogeneous transformation (denoted as **<sup>I</sup>H<sub>W</sub>**) between the camera and the world frame, and the homogeneous transformation (denoted as **<sup>C</sup>H<sub>W</sub>**) between IMU and the world frame. A 6x6 AprilTag grid map is used as the landmarks for the calibration procedure.
Above two transformations (<sup>I</sup>H<sub>W</sub> & <sup>C</sup>H<sub>W</sub>) can be used to compute the transformation between Camera and IMU i.e. <sup>C</sup>H<sub>W</sub> <sup>W</sup>H<sub>I</sub>. In this method the IMU is parameterized as 6x1 spline using the three degrees of freedom for translation & three degrees of freedom for orientation. Based on the raw acceleration & angular velocity readings from the on board IMU, the acceleration and the angular velocity is computed in terms of <sup>C</sup>H<sub>W</sub>.

This library needs an initial guess of <sup>C</sup>H<sub>W</sub>, the homogeneous transformation which is first computed for each April tag using the PnP algorithm. Then the error between the predicted positions of the landmark based on the IMU reading and the observed position of the landmark using a camera is minimized using Levenberg Marquardt optimizer. So, this library generates the following output: <br/>
(i) the transformation between the camera and the IMU <br/>
(ii) the offset between camera time and IMU time (synchronization time) <br/>
(iii) the pose of the IMU w.r.t to world frame <br/>
(iv) Intrinsic camera calibration matrix, K for the camera <br/>

Going through this [paper's](https://furgalep.github.io/bib/furgale_iros13.pdf) section III B and IV will give deeper insights into the implementation.

## Requirement
1. A printout of 6x6 AprilTag 36h11 markers on A0 size paper. This can be downloaded from [here](https://drive.google.com/file/d/0B0T1sizOvRsUdjFJem9mQXdiMTQ/edit).
2. This [CDE](https://drive.google.com/file/d/0B0T1sizOvRsUVDBVclJCQVJzTFk/edit) package is the easiest and the fastest way to get the library running. All dependencies are packed within this package and no external dependencies are required.
3. ROS drivers for the IMU and camera. Pixhawk, Ardupilot is typically used for UAVs and UM6/7 is popular for ground robots like Husky. ZED, Asus Xtion Pro, Intel Realsense D435 are few popular stereo cameras.

## Calibration Procedure
Download and extract the CDE package.
```
  tar xfvz kalibr.tar.gz
```

Either you can run the tools directly from the cde-package folder or/and add the package folder to the system path using: 
```
  export PATH="/cde/package/path:$PATH"
```
Change the AprilTag size in the YAML file and copy the same yaml file inside the extracted kalibr folder. <br/> Now make sure that both IMU & Camera publishes raw data and images respectively to ROS. Below command will list all the ROS topics published by the sensor.
```
rostopic list
```
Before creating ROS bag check if the sensor data is published properly.
```
rostopic echo /topic_name
```
Check the publishing rate of a topic. It is recommended that the camera should run at 20 Hz and IMU at 200 Hz.
```
rostopic hz /topic_name
```
Further, move the robot along all its degree of freedom. For instance, the axes of the UAVs are translated in x,y & z direction & rotated in all the three directions (roll,pitch & yaw) for the proper calibration. Collect the IMU sensor’s raw measurement and camera frame for around 60 seconds. Before recoding camera data, ensure that the RGB images from the camera is converted into the grayscale format. 

In this sample example, Pixhawk sensor data is subscribed from the MAVROS raw_sensor message & camera frames are subscribed from the ZED sensor camera node. For other IMUs/Cameras only the ROS message name will change.

Start ZED sensor camera
```
roslaunch zed_wrapper zed.launch
```
and similarly, start pixhawk
```
roslaunch mavros px4.launch
```
Below command will start the recording.
```
rosbag record -O camera-imu-data.bag /imu/data_raw /zed/left/image_raw /zed/right/image_raw
```
Check the recorded data by using the following command.
```
rosbag play <your bagfile>
```
If all the required data is recorded properly, run following camera calibration command.
```
./kalibr_calibrate_cameras --models pinhole-equi pinhole-equi --topics /zed/left/image_raw /zed/right/image_raw --bag camera-imu-data.bag --target aprilgrid_6x6.yaml
```
Arguments:<br/>
- models: Most typically used stereo cameras are of pinhole camera model. For more detail on the supported models, refer to this [link](https://github.com/ethz-asl/kalibr/wiki/supported-models).<br/>
- topics: Name of the recodered camera topics  <br/>
- bag: ROS bag containing the image and IMU data <br/>
- target: yaml configuration file for the used target <br/>

This command will generate a new yaml file with camera intrinsic calibration parameters. Above command can be skipped if the camera calibration yaml is already available. Now run below imu-camera calibration script.
```
./kalibr_calibrate_imu_camera --cam cam_calc.yaml --target aprilgrid_6x6.yaml --imu imu0.yaml --bag camera-imu-data.bag
```
Arguments:<br/>
- cam: Generated camera calibration yaml configuration file<br/>
- imu: yaml configuration file for the IMU<br/>

For more detail on the different yaml format, please check this [link](https://github.com/ethz-asl/kalibr/wiki/yaml-formats)<br/>

After running kalibr_calibrate_imu_camera script, the camera calibration yaml will be extended by the imu-camera calibrator with imu-camera transformations.

## Few notes and important tips
1. During testing, ensure that the robot is moved slowly so that a sufficient amount of data is collected in a single position and try to excite all the IMU axes. 
2. Ensure that most of the camera frame has all the April tags captured in it.
3. Avoid shocks during the testing, especially at the beginning and the end of the data collection. 
4. The tool requires a rosbag with more than 1000 images, which leads to a ROS bag of few GBs. So, it is suggested to be run on the development machine instead of the onboard SBC.
5. This tool needs raw imu sensor data, to ensure that the correct ROS message is recorded from the IMU. 
6. Make sure the calibration target is as flat as possible. 

## References
Official github repository of the Kalibr tool can be found [here](https://github.com/ethz-asl/kalibr/).
