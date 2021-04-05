---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2021-04-06 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: YOLO Integration with ROS and Running with CUDA GPU
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

## Overview
Integrating YOLO(You Only Look Once), a real time object detection algorithm commonly used in the localization task, with ROS might pose a real integration challenge. There are many steps that are not well documented when installing the package in ROS. There is even more difficulty if one tries to switch from using the default CPU computation to using CUDA accelerated GPU computation as a ROS package.

This article serves as a step-by-step tutorial of integrating YOLO algorithm in ROS and setting up GPU to ensure its real-time performance. The tutorial will be detailed into two main parts: integration with ROS, setting up CUDA. The latter one does not depend on the first one. If you are only interested in running GPU with YOLO, then you can simply skip the first part.

## Integration YOLO with ROS
![YOLO Demo](assets/yolo_demo.png)

To do the integration, we will use a YOLO ROS wrapper GitHub repository [darknet_ros](https://github.com/leggedrobotics/darknet_ros). You can simply follow their instructions in the README, or follow the instructions below. 

Before you start the integration, make sure you have prepared your pre-trained YOLO model weights and configurations. Based on different detection tasks, the pre-trained model weights might differ. If your task requires objects that are not included in the default YOLO (which used [VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) or [COCO](https://cocodataset.org/#home) dataset to train), you would need to search for other pre-trained open-source projects and download their model weights and configurations to your local machine. Otherwise, you would need to train the YOLO from scratch with your own dataset. The details will not be included in this article, but you may find helpful from this [tutorial](https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/)


### Requirements:

- Ubuntu: 18.04

- ROS: Melodic

- YOLO: The official YOLO ROS wrapper GitHub repo [darknet_ros](https://github.com/leggedrobotics/darknet_ros) currently only support versions below YOLO v3. If you are using YOLO v4, try this repo instead [yolo_v4](https://github.com/tom13133/darknet_ros/tree/yolov4)


### Steps:
1. #### Download the repo:
   
```cd catkin_workspace/src```

```git clone --recursive git@github.com:leggedrobotics/darknet_ros.git```

Note: make sure you have --recursive tag to download the darknet package

```cd ../```

2. #### Build:

```catkin_make -DCMAKE_BUILD_TYPE=Release```

3. #### Using your own model:
   
Within `/darknet_ros/yolo_network_config`:

   1. add .cfg and .weights (YOLO detection model weights and configs) into /cfg and /weights folder
   
   2. within /cfg, run `dos2unix your_model.cfg` (convert it to Unix format if you have problem with Windows to Unix format transformation)

Within `/darknet_ros/config`:

   1. modify "ros.yaml" with correct camera topic
   
   2. create "your_model.yaml" to configure the model files and detected classes

Within `/darknet_ros/launch`:

   1. modify "darknet_ros.launch" with correct yaml file ("your_model.yaml")

4. #### Run:

```catkin_make```

```source devel/setup.bash```

```roslaunch darknet_ros darknet_ros.launch```

By launch the ROS node, there will be a window poped up automatically with the detections. You can also check it in "rviz".


### ROS Topics:
Published ROS topics:

   * object_detector (std_msgs::Int8) No. of detected objects
   * bounding_boxes (darknet_ros_msgs::BoundingBoxes) Bounding boxes(class, x, y, w, h) Details are shown in the `/darknet_ros_msgs`.
   * detection_image (sensor_msgs::Image) Image with detected bounding boxes.



## Use GPU and Set up CUDA:

You may found that YOLO running in CPU setting is very slow. To further ensure the run-time performance, you can accelerate it by using CUDA GPU. 

Note: The darknet currently only supports versions below CUDA 10.2 with cuDNN 7.6.5. If you are using CUDA 11+ or cuDNN 8.0+, you probably need to downgrade CUDA and cuDNN. 

Here is a detailed instructions on installing CUDA 10.2 + cuDNN 7.6.5:


### Install CUDA 10.2:
Note: If there is any usr/local/cuda directory in your local machine, remove it before re-install

Note: No need to set up driver beforehand. When install CUDA, it will set up driver automatically.

We will follow most of the instructions shown in this [tutorial](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)

1. Remove all of CUDA related files already in the machine:
   
  ```
  sudo rm /etc/apt/sources.list.d/cuda*
  sudo apt remove --autoremove nvidia-cuda-toolkit
  sudo apt remove --autoremove nvidia-*
  ```

2. Install CUDA 10.2:
   
  ```
  sudo apt update
  sudo apt install cuda-10-2
  sudo apt install libcudnn7
  ```

3. Add CUDA into path:
   
  ```sudo vi ~/.profile```

  Add below at the end of .profile:
  ```
  # set PATH for cuda installation
  if [ -d "/usr/local/cuda/bin/" ]; then
      export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  fi
  ```

4. Check CUDA version:
  ```nvcc -V```

### Install cuDNN separately:
1. Go to [page](https://developer.nvidia.com/rdp/cudnn-download), you may need to register an account in NVIDIA.

2. Download all three .deb: runtime/developer/code-sample (make sure the correct version: cuDNN 7.6.5 with CUDA 10.2)
   
3. In Terminal:
   
  Go into the package location:

  ```sudo dpkg -i libcudnn7_7.6.5.32–1+cuda10.2_amd64.deb``` (the runtime library)

  ```sudo dpkg -i libcudnn7-dev_7.6.5.32–1+cuda10.2_amd64.deb``` (the developer library)

  ```sudo dpkg -i libcudnn7-doc_7.6.5.32–1+cuda10.2_amd64.deb``` (the code samples).

4. Check cuDNN version:
   
  ```/sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn```

5. Optional: 
   
  If you cannot locate cudnn.h, or the later compilation fails with not found cudnn.h:

  Copy cudnn.h (in usr/include) to (usr/local/cuda/include)

  Copy libcudnn* (in usr/lib/x86_...) to (/usr/local/cuda/lib64/)



## Run YOLO with GPU Settings:

If you are not using darknet_ros package, the procedure should be similar to below.

1. Modify `/darnet_ros/darknet/Makefile`:
   ```
   GPU = 1
   CUDNN =1
   OPENCV = 1
   ```

   Add your ARCH: you can find your ARCH online

   ```-gencode=arch=compute_75,code=compute_75```

2. Run `make` in /darknet_ros/darknet

3. Modify `/darknet_ros/darknet_ros/ CmakeList.txt`:
   
   ```-gencode=arch=compute_75,code=compute_75```

4. Run `catkin_make` in /catkin_workspace

**GPU is ready!**


## Summary
In this tutorial, we go through the procedures for integrating YOLO model with ROS by deploying a ROS wrapper. Depending on the tasks, the YOLO model weights and configurations files should be added into the ROS wrapper folder. By following modifying the ROS wrapper configuration and launch file, you could run your YOLO model in ROS then. 

We also show the procedures for setting up CUDA and cuDNN to run the YOLO in real-time in this tutorial. By following the detailed instructions, you can reach the best YOLO run-time perfermance.  

## See Also:
- [realsense_camera](https://roboticsknowledgebase.com/wiki/sensing/realsense/)
- [ROS](https://roboticsknowledgebase.com/wiki/common-platforms/ros/ros-intro/)

## Further Reading
- [CUDA_tutorial](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros)
- [darknet_ros_3d](https://github.com/IntelligentRoboticsLabs/gb_visual_detection_3d)

## References
- https://github.com/leggedrobotics/darknet_ros
- https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130
- https://pjreddie.com/projects/pascal-voc-dataset-mirror/
- https://cocodataset.org/#home
- https://github.com/tom13133/darknet_ros/tree/yolov4

