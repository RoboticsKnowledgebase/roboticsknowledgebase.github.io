---
date: 2022-05-04
title: STag
---

STag is a stable fiducial marker system that provides stable pose estimation. STag is designed to be robust against jitter factors. As such it provides better pose stability than existing solutions. This is achieved by utilizing geometric features that can be localized more repeatably. The outer square border of the marker is used for detection and homography estimation. This is followed by a novel homography refinement step using the inner circular border. After refinement, the pose can be estimated stably and robustly across viewing conditions. 

![STag Marker](https://user-images.githubusercontent.com/19530665/57179654-c0c11e00-6e88-11e9-9ca5-0c0153b28c91.png)

![STag Marker Under Occlusion](https://user-images.githubusercontent.com/19530665/57179660-cae31c80-6e88-11e9-8f80-bf8e24e59957.png)


## ROS implementation

[Dartmouth Reality and Robotics Lab](https://github.com/dartmouthrobotics/stag_ros) and [Unmanned Systems & Robotics Lab - University of South Carolina (UofSC)](https://github.com/usrl-uofsc/stag_ros) provide ROS packages that wrap original STag library and provide the marker pose as output in ROS format. We recommend using the library provided by UofSC as it seems to be continously updated. It also has support for ROS kinectic, melodic, and noetic. Moreover it is capable of working with fisheye lens. They also provide ROS nodelet support. All results in this article were generated using the library provided by UofSC. 

## Setup

The [Unmanned Systems & Robotics Lab - University of South Carolina (UofSC)](https://github.com/usrl-uofsc/stag_ros) repository has mentioned the steps necessary to build and how to use the library. In addition they also provide some bag files to verify their implementation. 

We would like to point out that while building on low compute power devices and SBCs like the Raspberry Pi and Odroid, adding the following commands to catkin_make will help prevent freeezing issues while building:

```
catkin_make -DCMAKE_BUILD_TYPE=<BUILD FLAG> -j2 -l2
```

## Configuration 

The library provided uses yaml files which can be found under the `/cfg` folder. 

The `single.yaml` file is used to help configure the library of tags to use and the error correction that should be used (Refer to the paper for exact details, but setting errorCorrection to (libraryHD / 2) - 2 seems to give good results). Set ```camera_info_topic``` and ```raw_image_topic``` to your ros camera topics accordingly.

The `singe_config.yaml` file is used to help configure which marker ids will be used for pose calculation along with the marker size. How to define the marker size, with respect to which corner, frame of reference, etc., can be found in this [issue rasised on github](https://github.com/usrl-uofsc/stag_ros/issues/8). 


## Experimental Results

In informal tests, the STag were able to achieve accuracy within +- 2 centimeters of the actual pose when the camera was within 5 meters of the tag. If the robot was further away, the accuracy decreased proportionally to the distance from the tag. The parameters that can help improve marker detection, speed and accuracy are discussed in the next section.

## Parameters that effect marker detection, speed and accuracy:
1. Marker size
   
   The following results where obtained by using one of the mono global shutter camera in OAK-D at 720p resolution and 60 FPS. 

    | Paper Size   	| Marker Size 	| Distance From Camera<br>at which marker is detected 	|
    |--------------	|-------------	|:---------------------------------------------------:	|
    | A4           	| 16.1 cm     	| 2-2.5 meters                                        	|
    | 24x24 inches 	| 51 cm       	| 12-15 meters                                        	|
    | 36x36 inches 	| 86 cm       	| 22-25 meters                                        	|

2. Detection Speed
   
   This is highly dependent on the image resolution and which flags were used during compilation (Debug/Release).


    | Processor                                	| Flag    	| FPS   	|   	|   	|
    |------------------------------------------	|---------	|-------	|---	|---	|
    | AMD Ryzen™ 7 5800H Processor             	| Debug   	| 20-25 	|   	|   	|
    | AMD Ryzen™ 7 5800H Processor             	| Release 	| 45-50 	|   	|   	|
    | Raspberry Pi 4 Model B 2019<br>(4GB RAM) 	| Debug   	| 5-6   	|   	|   	|
    | Raspberry Pi 4 Model B 2019<br>(4GB RAM) 	| Release 	| 10-15 	|   	|   	|

3. Camera Parameters
    
   **Exposure**, **ISO sensitivity** and **shutter type** are three camera main parameters which can help greatly improve marker detection. It is highly recommend to use ***global shutter cameras*** instead of rolling shutter cameras to avoid aliasing effects. Exposure and ISO sensitivity parameters are dependent on the application. In our case (TeamJ, MRSD 2021-23), the camera was mounted on a VTOL and the marker was to be detected from at least 12 meters away in outdoor environments. In order to do so, we had to use the lowest exposure (= 1 microsecond) and ISO sensitivity (= 100) settings for our camera. The GIF below shows how exposure and ISO sensitivity affects marker detection output.

    ![exposure and iso sensitivity settings affecting marker detection](assets/stag_exposure.gif)

## Tips

- Use your favourite photo editor tool to measure the marker size in meters before printing. Using this value is advised in the config file as it greatly improves the accuracy of marker pose detected.
- Using bundles doesn't seem to effect the accuracy of the marker pose, although please note in this case all markers were assumed to be in a single plane. May be off-setting the markers in Z relative to each other  might lead to better accuracy
- It is recommend to not use the marker pose directly and instead have a processing node that subscribes to the marker pose and performs some kind of filtering on the pose. This [library](https://github.com/dmicha16/ros_sensor_filter_kit) could be used with some changes or the code we have written to do the same can be found [here](https://github.com/Project-DAIR/pose_correction/blob/main/scripts/pose_correction.py).

## References
- STag Paper: https://arxiv.org/abs/1707.06292
- STAG Markers : [Drive Folder](https://drive.google.com/drive/folders/0ByNTNYCAhWbIV1RqdU9vRnd2Vnc?resourcekey=0-9ipvecbezW8EWUva5GBQTQ&usp=sharing)
- STag Github Repository: http://april.eecs.umich.edu/wiki/index.php/AprilTags
- STag ROS Wrapper provided by Dartmouth Reality and Robotics Lab: https://github.com/dartmouthrobotics/stag_ros
- STag ROS Wrapper provided by Unmanned Systems & Robotics Lab - University of South Carolina (UofSC) : https://github.com/usrl-uofsc/stag_ros
- ROS Sensor Filter Kit : https://github.com/dmicha16/ros_sensor_filter_kit
- Pose Correction : https://github.com/Project-DAIR/pose_correction/blob/main/scripts/pose_correction.py