# Setting Up the ZED Camera with ROS

The ZED stereo camera is a powerful perception sensor that provides depth sensing, visual odometry, and spatial mapping capabilities. This tutorial guides you through the complete setup process for integrating ZED cameras with ROS. You'll learn how to install required software, configure the camera, and access sensor data through ROS topics. By following this guide, you'll have a fully functional ZED camera system publishing depth, point cloud, and pose data to ROS.

## Prerequisites
Before beginning this tutorial, ensure you have:

### Required Hardware
- A ZED stereo camera (ZED, ZED Mini, ZED 2, or ZED 2i)
- Computer with NVIDIA GPU (CUDA-capable)
- USB 3.0 port

### Required Software
- Ubuntu 18.04 or 20.04
- ROS Melodic (18.04) or ROS Noetic (20.04)
- CUDA (will be installed with SDK)

## Installation Steps

### Installing the ZED SDK
First, we need to install the ZED SDK which provides the core functionality:

```bash
# Install dependency
sudo apt install zstd

# Download SDK from stereolabs.com
# Add execute permissions
chmod +x ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run

# Run installer
./ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run
```

> Note: Make sure to select 'y' when prompted about installing CUDA if not already installed.

### Installing the ROS Wrapper
The ROS wrapper enables integration with ROS:

```bash
# Setup catkin workspace
cd ~/catkin_ws/src
git clone --recursive https://github.com/stereolabs/zed-ros-wrapper.git

# Install dependencies
cd ..
rosdep install --from-paths src --ignore-src -r -y

# Build packages
catkin_make -DCMAKE_BUILD_TYPE=Release
source ./devel/setup.bash
```

## Using the ZED with ROS

### Starting the Camera Node
Launch the appropriate file for your camera model:

```bash
# For ZED 2i
roslaunch zed_wrapper zed2i.launch

# For ZED 2
roslaunch zed_wrapper zed2.launch

# For ZED Mini
roslaunch zed_wrapper zedm.launch

# For original ZED
roslaunch zed_wrapper zed.launch
```

### Available Topics
The ZED node publishes several useful topics:

- `/zed/rgb/image_rect_color` - Rectified RGB image
- `/zed/depth/depth_registered` - Registered depth image
- `/zed/point_cloud/cloud_registered` - Color point cloud
- `/zed/odom` - Visual odometry
- `/zed/imu/data` - IMU data (ZED 2/Mini only)

### Visualizing Data
Use RViz to view camera output:

```bash
roslaunch zed_display_rviz display_zed2i.launch
```

### Recording and Playback
Record data using the SVO format:

```bash
# Recording
roslaunch zed_wrapper zed2i.launch svo_file:=/path/to/output.svo

# Playback
roslaunch zed_wrapper zed2i.launch svo_file:=/path/to/recording.svo
```

## Common Issues and Troubleshooting

### USB Connection Issues

#### Symptoms
- Camera not detected
- Frequent disconnections
- Poor frame rate
- Error message: "Unable to open camera"

#### Solutions
1. USB Port Problems
```bash
# Check USB port type
lsusb -t

# Check USB bandwidth
sudo apt-get install htop
htop
```
- Ensure using USB 3.0 port (blue connector)
- Connect directly to motherboard, avoid USB hubs
- Try different USB ports
- Test with shorter cable (<3m)

2. Bandwidth Issues
- Close other USB 3.0 devices
- Check system load with `htop`
- Try reducing resolution or FPS in launch file:
```yaml
# In zed_camera.launch
<param name="resolution" value="2"/> <!-- HD720 instead of HD1080 -->
<param name="frame_rate" value="30"/> <!-- Lower from 60 to 30 -->
```

### SDK Installation Problems

#### Symptoms
- Installation fails
- Missing dependencies
- CUDA errors

#### Solutions
1. CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# If CUDA missing, reinstall
sudo apt-get install cuda
```

2. Dependencies
```bash
# Install common missing dependencies
sudo apt-get install build-essential
sudo apt-get install libusb-1.0-0-dev
sudo apt-get install libhidapi-dev
```

### ROS Integration Problems

#### Symptoms
- Node crashes
- Missing topics
- Transform errors

#### Solutions
1. Node Startup Issues
```bash
# Check ROS logs
roscd zed_wrapper
cat ~/.ros/log/latest/zed_wrapper-*.log

# Verify ROS environment
printenv | grep ROS
```

2. Topic Problems
```bash
# List active topics
rostopic list | grep zed

# Check topic publishing rate
rostopic hz /zed/rgb/image_rect_color

# Monitor transform tree
rosrun rqt_tf_tree rqt_tf_tree
```

### Performance Issues

#### Symptoms
- High latency
- Frame drops
- High CPU/GPU usage

#### Solutions
1. System Resources
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CPU temperature
sensors
```

2. Optimization Steps
- Reduce depth computation mode:
```yaml
# In zed_camera.launch
<param name="depth_mode" value="1"/> <!-- PERFORMANCE mode -->
```
- Disable unnecessary features:
```yaml
<param name="pub_frame_rate" value="15.0"/>
<param name="point_cloud_freq" value="1.0"/>
```

### Camera Calibration Issues

#### Symptoms
- Poor depth accuracy
- Misaligned stereo
- Distorted images

#### Solutions
1. Factory Reset
```bash
# Run ZED Explorer
cd /usr/local/zed/tools
./ZED\ Explorer

# Select: Camera > Reset Calibration
```

2. Self Calibration
- Ensure good lighting conditions
- Move camera in figure-8 pattern
- Use `ZED Calibration` tool:
```bash
cd /usr/local/zed/tools
./ZED\ Calibration
```

### Common Error Messages

#### "Failed to open camera"
- Check USB connection
- Verify camera permissions:
```bash
sudo usermod -a -G video $USER
```
- Restart computer

#### "CUDA error: out of memory"
- Reduce resolution/FPS
- Close other GPU applications
- Check available GPU memory:
```bash
nvidia-smi
```

#### "Transform error between camera_link and map"
- Check TF tree:
```bash
rosrun tf tf_echo camera_link map
```
- Verify odometry publication
- Ensure proper initialization time

> Note: Always check the ZED SDK and ROS wrapper versions are compatible. Mixing versions can cause unexpected issues.

## Summary
You should now have a working ZED camera setup integrated with ROS. The camera will publish various sensor data topics that can be used for perception, mapping, and navigation tasks. For advanced usage, explore the dynamic reconfigure parameters and additional features like object detection.

## Further Reading
- [Official ZED Documentation](https://www.stereolabs.com/docs/)
- [ROS Wiki - ZED Wrapper](http://wiki.ros.org/zed-ros-wrapper)

## References
- Stereolabs, "ZED SDK Documentation," 2024.
- M. Quigley et al., "ROS: an open-source Robot Operating System," ICRA Workshop on Open Source Software, 2009.
- P. Fankhauser and M. Hutter, "A Universal Grid Map Library: Implementation and Use Case for Rough Terrain Navigation," in Robot Operating System (ROS) – The Complete Reference (Volume 1), A. Koubaa, Ed. Springer, 2016.