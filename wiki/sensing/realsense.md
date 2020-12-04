---
date: {}
title: Realsense RGB-D camera
published: true
---
This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).

## Sensor Overview
Intel® RealSense™ D400 Series Depth Cameras are ready-to-use right out of the box and can be easily added into your project. Bring innovative vision awareness to your solution with our best-in-class depth resolution, quality RGB, and high frame rate. Both the Intel® RealSense™ Depth Camera D415 and the Intel® RealSense™ Depth Camera D435 provide highly accurate depth data in a wide variety of indoor and outdoor environments and can be used in unlimited multiple camera configurations.

## SDK 
The RealSense camera package allows access to and provides ROS nodes for Intel 3D cameras and advanced modules. The SDK allows depth and color streaming, and also provides camera calibration information.
The source code can be downloaded and built from this repository:
https://github.com/IntelRealSense/librealsense/

## ROS package
### Installation instructions
Step 1: Install the latest Intel® RealSense™ SDK 2.0
Install from Debian Package - In that case treat yourself as a developer. Make sure you follow the instructions to also install librealsense2-dev and librealsense-dkms packages.
OR
Build from sources by downloading the latest Intel® RealSense™ SDK 2.0 and follow the instructions under Linux Installation

Step 2: Install the ROS distribution
Install ROS Kinetic, on Ubuntu 16.04

Step 3: Install Intel® RealSense™ ROS from Sources
Create a catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src/
Clone the latest Intel® RealSense™ ROS from here into 'catkin_ws/src/'
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^\d+\.\d+\.\d+" | tail -1`
cd ..
Make sure all dependent packages are installed. You can check .travis.yml file for reference.
Specifically, make sure that the ros package ddynamic_reconfigure is installed. If ddynamic_reconfigure cannot be installed using APT, you may clone it into your workspace 'catkin_ws/src/' from here (Version 0.2.0)
catkin_init_workspace
cd ..
catkin_make clean
catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
catkin_make install
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

### Start Camera Node
roslaunch realsense2_camera rs_camera.launch
Will start the camera node in ROS and stream all camera sensors.

### Published ROS Topics
The published topics differ according to the device and parameters. After running the above command with D435i attached, the following list of topics will be available (This is a partial list. For full one type rostopic list):

/camera/color/camera_info  
/camera/color/image_raw  
/camera/depth/camera_info  
/camera/depth/image_rect_raw  
/camera/extrinsics/depth_to_color  
/camera/extrinsics/depth_to_infra1  
/camera/extrinsics/depth_to_infra2  
/camera/infra1/camera_info  
/camera/infra1/image_rect_raw  
/camera/infra2/camera_info  
/camera/infra2/image_rect_raw  
/camera/gyro/imu_info  
/camera/gyro/sample  
/camera/accel/imu_info  
/camera/accel/sample  
/diagnostics  

## Calibration
### Intrinsic Calibration
The intrinsics of the camera module can be found in the /camera/color/camera_info topic when the realsense camera is launched through ROS. The K matrix in the published topic corresponds to the intrinsics matrix.  
The intrinsic of the camera module can be calibrated with the matlab single camera calibration App. www.mathworks.com/help/vision/ug/single-camera-calibrator-app.html  
The general steps are as follows:
1. Prepare images, camera, and calibration pattern.

2. Add images and select standard or fisheye camera model.

3. Calibrate the camera.

4. Evaluate calibration accuracy.

5. Adjust parameters to improve accuracy (if necessary).

6. Export the parameters object.

Note: The checkerboard required can be generated on this site  [https://calib.io/pages/camera-calibration-pattern-generator](https://calib.io/pages/camera-calibration-pattern-generator). After printing out the checkerboard, make sure to measure the box sizes and verify the print accuracy.Printers can be very inaccurate sometimes. Remember to stick the checkerboard on a piece of flat cardboard. When taking pictures of the checkerboard, make sure to take the pictures at different angle and different distant.

### Extrinsic Calibration
The realsense sensors are calibrated out of the box. However, if the sensor was dropped or there is a high amount of vibration in your application it might be worthy to recalirate the extrinc of the sensor for better depth map quality. The extrinsic here refers to the coordinate transformation between the depth modules and the camera on the sensor. For this task, intel has developed a dynamic calibration tool([https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_D400_Dyn_Calib_User_Guide.pdf](https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_D400_Dyn_Calib_User_Guide.pdf)).  
Two types of dynamic calibrations are offered in this tool:

1. Rectification calibration: aligning the epipolar line to enable the depth pipeline to work correctly and reduce the holes in the depth image

2. Depth scale calibration: aligning the depth frame due to changes in position of the optical
elements


## Tuning and Sensor Characteristics 
### Optimal Resolution
The depth image precision is affected by the output resolution. The optimal resolutions of the D430 series are as follow:
- D415: 1280 x 720
- D435: 848 x 480

Note:  

1. Lower resolutions can be used but will degrade the depth precision. Stereo depth sensors
derive their depth ranging performance from the ability to match positions of objects in the
left and right images. The higher the input resolution, the better the input image, the better
the depth precision.  

2. If lower resolution is needed for the application, it is better to publish high resolution image and depth map from the sensor and downsample immediately instead of publishing low resolution image and depth map.

### Image Exposure

1. Check whether auto-exposure works well, or switch to manual exposure to make sure you
have good color or monochrome left and right images. Poor exposure is the number one
reason for bad performance.  

2. From personal experience, it is best to keep auto-exposure on to ensure best quality. Auto exposure could be set using the intel realsense SDK or be set in the realsense viewer GUI.  

3. There are two other options to consider when using the autoexposure feature. When
Autoexposure is turned on, it will average the intensity of all the pixels inside of a predefined
Region-Of-Interest (ROI) and will try to maintain this value at a predefined Setpoint. Both
the ROI and the Setpoint can be set in software. In the RealSense Viewer the setpoint can
be found under the Advanced Controls/AE Control.  

4. The ROI can also be set in the RealSense Viewer, but will only appear after streaming has
been turn on. (Ensure upper right switch is on).


### Range 

1. D400 depth cameras give most precise depth ranging data for objects that are near. The
depth error scales as the square of the distance away.  
2. However, the depth camera can't be too close to the object that it is within the minz distance.
The minZ for the D415 at 1280 x 720 is 43.8cm and the minz for the D435 at 848x480 is 16.8cm

### Post Processing
The realsense SDK offers a range of post processing filters that could drastically improve the quality. However, by default, those filters aren't enabled. You need to manually enable them. To enable the filters, you simply need to add them to your realsense camera launch file under the filters param [https://github.com/IntelRealSense/realsense-ros#launch-parameters](https://github.com/IntelRealSense/realsense-ros#launch-parameters). The intel recommended filters are the following:  

1. **Sub-sampling**: Do intelligent sub-sampling. We usually recommend doing a non-
zero mean for a pixel and its neighbors. All stereo algorithms do involve someconvolution operations, so reducing the (X, Y) resolution after capture is usually
very beneficial for reducing the required compute for higher-level apps. A factor
of 2 reduction in resolution will speed subsequent processing up by 4x, and a scale
factor of 4 will decrease compute by 16x. Moreover, the subsampling can be used
to do some rudimentary hole filling and smoothing of data using either a non-zero
mean or non-zero median function. Finally, sub-sampling actually tends to help
with the visualization of the point-cloud as well.  

2. **Temporal filtering**: Whenever possible, use some amount of time averaging to
improve the depth, making sure not to take “holes” (depth=0) into account. There
is temporal noise in the depth data. We recommend using an IIR filter. In some
cases it may also be beneficial to use “persistence”, where the last valid value is
retained indefinitely or within a certain time frame.  

3. **Edge-preserving filtering**: This will smooth the depth noise, retain edges while
making surfaces flatter. However, again care should be taken to use parameters
that do not over-aggressively remove features. We recommend doing this
processing in the disparity domain (i.e. the depth scale is 1/distance), and
experimenting by gradually increasing the step size threshold until it looks best for
the intended usage. Another successful post-processing technique is to use a
Domain-Transform filter guided by the RGB image or a bilinear filter. This can help
sharpen up edges for example. 

4. **Hole-filling**: Some applications are intolerant of holes in the depth. For example,
for depth-enhanced photography it is important to have depth values for every
pixel, even if it is a guess. For this, it becomes necessary to fill in the holes with
best guesses based on neighboring values, or the RGB image.












### Basic syntax
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.

#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/BKMs_Tuning_RealSense_D4xx_Cam.pdf
- https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/RealSense_D400_Dyn_Calib_User_Guide.pdf
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.