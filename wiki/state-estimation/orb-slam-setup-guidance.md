# ORB SLAM Setup Guidance 
This tutorial will help you in setting up the ORB SLAM2 on a Single Board Computer. We discussed the installation procedure for the stereo mode and explained the changes required in the stereo camera's yaml configuration file. Since the ORB SLAM2 code doesn't publish pose output, we have added a separate section on how to add the ROS publisher support for the stereo mode. 

# Table of Contents
1. [Introduction](#Introduction)
2. [Installation for stereo mode](#Installation-for-stereo-mode)
3. [Setting up yaml configuration file](#Setting-up-yaml-configuration-file)
4. [Adding ROS publisher support for the stereo mode](#Adding-ROS-publisher-support-for-the-stereo-mode)
5. [References](#References)

## Introduction
ORB-SLAM2 is a SLAM library for Monocular and Stereo cameras that computes the camera trajectory and a sparse 3D reconstruction. It is a feature-based SLAM method which has three major components: tracking, local mapping and loop closing. This [link](https://medium.com/@j.zijlmans/lsd-slam-vs-orb-slam2-a-literature-based-comparison-20732df431d) nicely explains all the components of ORB SLAM2 technique in detail. Also, it briefly discusses the different part of ORB SLAM2 code and explains how changing the different parameters in different modules like local mapping, loop-closure, ORBextractor, etc. will affect the performance of ORB SLAM2.

## Installation for stereo mode
ORB-SLAM2 has multiple dependencies on other ROS libraries which includes Pangolin, OpenCV, Eigen3, DBoW2, and g2o. **[Pangolin](https://github.com/stevenlovegrove/Pangolin)** library is used for the visualization and user interface.**[OpenCV](https://docs.opencv.org/3.4.3/d7/d9f/tutorial_linux_install.html)** is used for image manipulation and feature extraction. **[Eigen3](http://eigen.tuxfamily.org)** library for performing mathematical operations on the Matrices. Finally, **[DBoW2](https://github.com/dorian3d/DBoW2)** is an improved version of the DBow library, for indexing and converting images into a bag-of-word representation. It implements a hierarchical tree for approximating nearest neighbors in the image feature space and creating a visual vocabulary. It also implements an image database with inverted and direct files to index images and enabling quick queries and feature comparisons. **[G2o](https://github.com/RainerKuemmerle/g2o)** is C++ library for optimizing graph-based nonlinear error functions. This helps in solving the global BA problem in ORB-SLAM2.<br/>

Clone the repository:
```
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
```
Then execute following set of commands to build the library:
```
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```
Even after installing above dependencies, if you face compilation error related to boost system, you need to install boost libraries and set the path to where you installed it in the makefile. This path should have the include/ and lib/ folders with header files and compiled .so binaries.

```
${PROJECT_SOURCE_DIR}/../../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../lib/libORB_SLAM2.so
-lboost_system
)
```
To build the stereo node, add the path including Examples/ROS/ORB_SLAM2 to the ROS_PACKAGE_PATH environment variable. Replace PATH by the folder where you cloned ORB_SLAM2 and then execute the build_ros script. 
```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
chmod +x build_ros.sh
./build_ros.sh
```
For a stereo input from topic /camera/left/image_raw and /camera/right/image_raw, we need to remap the left and right camera frame output to the ORB-SLAM2 input ROS topic. A sample roslaunch doing the ROS topic remapping is shown below. 
```
<?xml version="1.0"?>
<launch>
    <node name="ORB_SLAM2" pkg="ORB_SLAM2" type="Stereo" output="screen" args="/home/administrator/ORB_SLAM2/Vocabulary/ORBvoc.txt /home/administrator/ORB_SLAM2/Examples/Stereo/zed.yaml true">
    <remap from="/camera/left/image_raw" to="/left/image_raw_color"/>
    <remap from="/camera/right/image_raw" to="/right/image_raw_color"/>
    </node>
</launch>
```
Above launch file also runs the ORB_SLAM2/Stereo node. You will need to provide the vocabulary file and a yaml settings file to run the Stereo node. Just use the same Vocabulary file because it's taken from a huge set of data and works pretty good.  All the popular stereo cameras like ZED, Intel Realsense, Asus Xtion Pro provides the pre-rectified images. So, if you are using one of those cameras, you don't need to provide rectification matrices else you need to specify provide rectification matrices in the yaml configuration file (discussed more in the next section).

## Setting up yaml configuration file
A new yaml file with the stereo sensor calibration parameters for different possible resolutions. We have used ZED camera for running ORB SLAM2 and below is the sample yaml configuration file for our calibrated ZED camera.

```
# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 435.2046959714599
Camera.fy: 435.2046959714599
Camera.cx: 367.4517211914062
Camera.cy: 252.2008514404297

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 752
Camera.height: 480

# Camera frames per second 
Camera.fps: 20.0

# stereo baseline times fx
Camera.bf: 47.90639384423901

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 35
```

```
#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1200

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast			
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize:2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
```

## Adding ROS publisher support for the stereo mode

```

```

```
void ImageGrabber::SetPub(ros::Publisher* pub)
{
    orb_pub = pub;
}
```

Define a 
```
ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("orb_pose", 100);
igb.SetPub(&pose_pub);
```
Initiliaze two matrices for storing the pose of the ORB SLAM
```
cv::Mat T_,R_,t_;
T_ = mpSLAM->TrackStereo(cv_ptrLeft->image,cv_ptrRight->image,cv_ptrLeft->header.stamp.toSec());
        
if (T_.empty())
return;
```

```
if (pub_tf || pub_pose)
{    
    R_ = T_.rowRange(0,3).colRange(0,3).t();
    t_ = -R_*T_.rowRange(0,3).col(3);
    vector<float> q = ORB_SLAM2::Converter::toQuaternion(R_);
    float scale_factor=1.0;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(t_.at<float>(0, 0)*scale_factor, t_.at<float>(0, 1)*scale_factor, t_.at<float>(0, 2)*scale_factor));
    tf::Quaternion tf_quaternion(q[0], q[1], q[2], q[3]);
    transform.setRotation(tf_quaternion);
}
```

```
if (pub_tf)
{
   static tf::TransformBroadcaster br_;
   br_.sendTransform(tf::StampedTransform(transform, ros::Time(cv_ptrLeft->header.stamp.toSec()), "world", "ORB_SLAM2_STEREO"));
}
```
```
if (pub_pose)
{
   geometry_msgs::PoseStamped pose;
   pose.header.stamp = cv_ptrLeft->header.stamp;
   pose.header.frame_id ="ORB_SLAM2_STEREO";
   tf::poseTFToMsg(transform, pose.pose);
   orb_pub->publish(pose);
}
```

## References
This is the official [github repository](https://github.com/raulmur/ORB_SLAM2) of the ORB SLAM2.
