# ORB SLAM Setup Guidance 
This tutorial will help you in understanding and setting up the ORB SLAM on a Single Board Computer. Since the available ORB SLAM github repository doesn't publish pose output, we have added a separate section on how to add the ROS publisher support for the stereo mode.

# Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Setting up yaml configuration file](#Setting-up-yaml-configuration-file)
4. [Adding ROS publisher support for the stereo mode](#Adding-ROS-publisher-support-for-the-stereo-mode)
5. [References](#References)

## Introduction
This [link](https://medium.com/@j.zijlmans/lsd-slam-vs-orb-slam2-a-literature-based-comparison-20732df431d) explains the ORB SLAM2 technique in detail. Also, at the end it briefly discusses the different part of ORB SLAM2 code and how changing the different parameters will affect the performance of ORB SLAM.

## Installation (for stereo mode)
ORB-SLAM2 has multiple dependencies on other ROS libraries which includes Pangolin, OpenCV, Eigen3, DBoW2, and g2o. **[Pangolin](https://github.com/stevenlovegrove/Pangolin)** library is used for the visualization and user interface.**[OpenCV](https://docs.opencv.org/3.4.3/d7/d9f/tutorial_linux_install.html)** is used for image manipulation and feature extraction. **[Eigen3](http://eigen.tuxfamily.org)** library for performing mathematical operations on the Matrices. Finally, **[DBoW2](https://github.com/dorian3d/DBoW2)** is an improved version of the DBow library, for indexing and converting images into a bag-of-word representation. It implements a hierarchical tree for approximating nearest neighbors in the image feature space and creating a visual vocabulary. It also implements an image database with inverted and direct files to index images and enabling quick queries and feature comparisons. **[G2o](https://github.com/RainerKuemmerle/g2o)** is C++ library for optimizing graph-based nonlinear error functions. This helps in solving the global BA problem in ORB-SLAM2.<br/>


Clone the repository:
```
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
```
Then execute following set of commands:
```
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```
Add the path including Examples/ROS/ORB_SLAM2 to the ROS_PACKAGE_PATH environment variable. Open .bashrc file and add at the end the following line. Replace PATH by the folder where you cloned ORB_SLAM2 and execute build_ros script. 
```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
chmod +x build_ros.sh
./build_ros.sh
```
For a stereo input from topic /camera/left/image_raw and /camera/right/image_raw run node ORB_SLAM2/Stereo. You will need to provide the vocabulary file and a settings file. If you provide rectification matrices (see Examples/Stereo/EuRoC.yaml example), the node will recitify the images online, otherwise images must be pre-rectified.

## Setting up yaml configuration file
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
