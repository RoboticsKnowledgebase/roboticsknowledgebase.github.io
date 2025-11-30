This article provides a comprehensive guide to the LIVOX Mid-360 LiDAR sensor, a popular choice in both academia and industry for robotic applications. This tutorial covers hardware connections, driver setup procedures, and step-by-step integration into the ROS 2 ecosystem. Advanced topics include Fast-LIO2 SLAM integration that leverages LiDAR-IMU data for robust odometry and mapping, point cloud processing pipelines. Following this guide, readers will be able to set up and integrate the LIVOX Mid-360 LiDAR into their robotic projects with ROS 2 Humble.

![LIVOX Mid-360 LiDAR](assets/livox-mid-360-lidar.png)


## Sensor Overview

The LIVOX Mid-360 is a compact, lightweight solid-state LiDAR sensor designed for low-speed robotics applications. Powered by Livox's unique rotating mirror hybrid-solid technology, the Mid-360 is the first Livox LiDAR to achieve a full 360° horizontal field of view, providing omnidirectional 3D perception capabilities. The sensor is optimized for mobile robot navigation, obstacle avoidance, and SLAM applications, delivering enhanced indoor and outdoor perception performance.

Key specifications:
- **Field of view**: 360° (horizontal) × 59° (vertical)
- **Minimum detection range**: 0.1 m (10 cm)
- **Maximum detection range**: 
  - 40 m @ 10% reflectivity (typical indoor surfaces: concrete floor 15-30%, white wall 90-99%)
  - 70 m @ 80% reflectivity (high-reflectivity surfaces)
- **Point cloud density**: 40-line
- **Dimensions**: 65 × 65 × 60 mm (L × W × H)
- **Weight**: 265 g
- **Interface**: Ethernet (1000BASE-T)
- **Default IP address**: 192.168.1.1XX (where XX is the last two digits of the sensor's serial number)

The Mid-360 features active anti-interference capabilities, allowing reliable operation even with multiple LiDAR signals in the same environment. The sensor performs consistently in both bright and low-light conditions, making it suitable for indoor and outdoor applications. Its compact size and short minimum detection range (10 cm) enable flexible mounting options and help eliminate blind spots in robot designs.

![LIVOX Mid-360 LiDAR](assets/livox-mid-360-lidar-fov.png)

> **Source**: Specifications and technical details are based on the [official LIVOX Mid-360 product page](https://www.livoxtech.com/mid-360).

## Hardware Connections

### Physical Setup

The LIVOX Mid-360 requires the following connections:

1. **Power Supply**: Connect the power adapter to the sensor. The sensor operates at 9-27V DC with a power consumption of 6.5W.
2. **Ethernet Connection**: Connect the sensor directly to your computer or Jetson Orin using an Ethernet cable (1000BASE-T).
3. **Function Connector (Optional)**: The M12 function connector supports GPS time synchronization (PPS and GPS input) and PTPv2 network time synchronization.
4. **Mounting**: Ensure the sensor is securely mounted on your robot platform. The sensor should be positioned to maximize its field of view for your specific application.
   
   **Mounting Tip**: For optimal horizontal FOV coverage, consider mounting the sensor at a slight angle. For example, [Tare Robotics](https://www.tarerobotics.com/) mounts the Mid-360 at a 20-degree tilt angle on their T-Bot platform, which helps balance the horizontal field of view distribution and improves ground-level obstacle detection.

![LIVOX Mid-360 LiDAR Connections](assets/livox-mid-360-lidar-connections.png)

### Network Configuration

The LIVOX Mid-360 supports two IP modes: dynamic IP address mode and static IP address mode. All Mid-360 sensors are set to static IP address mode by default.

#### LiDAR IP Address

The default IP address of each Mid-360 is `192.168.1.1XX`, where `XX` represents the last two digits of the sensor's serial number. For example, if the serial number ends in "54", the IP address will be `192.168.1.154`. The default subnet mask is `255.255.255.0` and the default gateway is `192.168.1.1`.

**Important**: When multiple Mid-360 sensors are connected to one computer, each sensor must have a different static IP address.

#### Computer IP Address Configuration

Your host computer must be configured on the same subnet (`192.168.1.x`) to communicate with the sensor. Livox recommends setting the computer's IP address to `192.168.1.50` with subnet mask `255.255.255.0`.

**Note**: If you have multiple computers connecting to the same Mid-360, each computer needs a different IP address within the `192.168.1.x` subnet (e.g., `192.168.1.50`, `192.168.1.51`, etc.).

#### Temporary Network Configuration (Current Session Only)

For quick testing, you can temporarily add an IP address to your network interface:

```bash
# Replace 'eth0' with your actual network interface name
# Use 192.168.1.50 as recommended by Livox
sudo ip addr add 192.168.1.50/24 dev eth0

# Verify connectivity (replace XX with your sensor's serial number digits)
ping -c 3 192.168.1.1XX
```

#### Permanent Network Configuration (Ubuntu 22.04 with Netplan)

For a permanent configuration that survives reboots, create or edit `/etc/netplan/99-uplink.yaml`:

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:  # Replace with your interface name (e.g., enP8p1s0, enp0s3)
      dhcp4: no
      addresses: 
        - 192.168.1.50/24  # Livox recommended IP address
      # If you need internet access, add gateway and DNS:
      # routes:
      #   - to: default
      #     via: 192.168.1.1
      # nameservers:
      #   addresses: [8.8.8.8, 8.8.8.4]
```

Apply the configuration:

```bash
sudo chmod 600 /etc/netplan/99-uplink.yaml
sudo netplan apply
```

#### Verifying Network Connectivity

After configuration, verify the connection:

```bash
# Check IP assignment
ip addr show eth0 | grep 192.168.1.50

# Test LiDAR connectivity (replace XX with your sensor's serial number digits)
ping -c 3 192.168.1.1XX
```

Expected result: 0-5ms latency, 0% packet loss.

> **Note**: If you need to maintain multiple network connections (e.g., internet access and LiDAR communication), you can assign multiple IP addresses to the same interface. See the network configuration section for advanced setups.

## Driver Setup

### Prerequisites

Before installing the LIVOX driver, ensure you have the following dependencies:

- **OS**: Ubuntu 22.04 (recommended) or Ubuntu 20.04
- **ROS**: ROS 2 Humble 
- **System packages**: CMake 3.10+, Git, build tools
- **ROS packages**: PCL libraries for ROS 2

Install system dependencies:

```bash
sudo apt update
sudo apt install -y git cmake build-essential \
    libpcl-dev libeigen3-dev libopencv-dev \
    ros-humble-pcl-ros ros-humble-pcl-conversions
```

For ARM64 platforms (e.g., NVIDIA Jetson), fix Python setuptools version:

```bash
pip3 install --user 'setuptools<70'
```

### Installing LIVOX SDK2

**Important**: The LIVOX Mid-360 requires **Livox SDK2** (not SDK1). SDK1 will not work with Mid-360.

1. Clone the LIVOX SDK2 repository:

```bash
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
```

2. Build and install the SDK:

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

3. Verify installation:

```bash
ls /usr/local/lib | grep livox
```

You should see Livox SDK libraries listed.

4. (Optional) Test SDK connection:

```bash
cd samples/livox_lidar_quick_start/
./livox_lidar_quick_start ../../../samples/livox_lidar_quick_start/mid360_config.json
```

If successful, you should see IMU and LiDAR streaming messages.

### Installing ROS 2 Driver (livox_ros_driver2)

The ROS 2 driver bridges the Livox SDK2 to ROS 2 topics.

1. Create a ROS 2 workspace (if you don't have one):

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone the ROS 2 driver:

```bash
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
cd livox_ros_driver2
```

3. Link the ROS 2 package.xml:

```bash
ln -s package_ROS2.xml package.xml
```

4. Install dependencies and build:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## ROS 2 Integration

### Launching the Driver

Launch the LIVOX Mid-360 driver:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

### Published Topics

The driver publishes the following topics:

- `/livox/lidar`: Point cloud data in `livox_ros_driver2/CustomMsg` format
- `/livox/imu`: IMU data in `sensor_msgs/Imu` format

### Verifying Data Stream

Check that topics are being published:

```bash
# List all topics
ros2 topic list

# Check topic frequency
ros2 topic hz /livox/lidar
ros2 topic hz /livox/imu

# View point cloud data (once)
ros2 topic echo /livox/lidar --once
```

### Configuration

Edit the driver configuration file to adjust parameters:

**Location**: `ros_ws/src/livox_ros_driver2/config/MID360_config.json`

Key parameters:
- `lidar_bag_ip`: LiDAR IP address (default: 192.168.1.1XX, where XX is the last two digits of the sensor's serial number)
- `host_bag_ip`: Host IP address (default: 192.168.1.50, Livox recommended)
- `imu_bag`: Enable/disable IMU data
- `frame_id`: TF frame name for the LiDAR

## Advanced Topics

### Fast-LIO2 SLAM Integration

Fast-LIO2 is a computationally efficient and robust LiDAR-inertial odometry framework that works well with LIVOX sensors. It provides real-time odometry and mapping capabilities.

#### Installing Dependencies

1. Install third-party libraries:

```bash
sudo apt-get install libpcl-dev libeigen3-dev libopencv-dev

# Install Sophus library
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### Building Fast-LIO2

Fast-LIO2 can be built in a ROS 2 workspace. There are ROS 2 ports available:

```bash
cd ~/ros2_ws/src
git clone https://github.com/hku-mars/FAST_LIO_ROS2.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

#### Running Fast-LIO2

In a separate terminal:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run fast_lio fastlio_mapping --ros-args \
    --params-file src/FAST_LIO_ROS2/config/avia.yaml
```

**Published topics:**
- `/Odometry`: Odometry data (~10 Hz)
- `/path`: Trajectory path visualization
- `/cloud_registered`: Registered point cloud map
- `/tf`: Transform tree

#### Configuration

Edit `src/FAST_LIO_ROS2/config/avia.yaml` to match your setup:
- LiDAR type: 1 (Livox series)
- Scan lines: 6 (for Mid-360)
- Input topics: `/livox/lidar`, `/livox/imu`

## Complete System Workflow

A typical setup involves running multiple components:

### Terminal 1: Livox Driver
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

### Terminal 2: Fast-LIO2 SLAM
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run fast_lio fastlio_mapping --ros-args \
    --params-file src/FAST_LIO_ROS2/config/avia.yaml
```

### Terminal 3: Visualization (Optional)
```bash
ros2 run rviz2 rviz2
```

Add displays for:
- `/livox/lidar` (PointCloud2)
- `/cloud_registered` (PointCloud2)
- `/path` (Path)
- TF tree

## Troubleshooting

### Common Issues

#### LiDAR Not Connecting

**Symptoms**: No data on `/livox/lidar` topic, driver shows connection errors

**Solutions**:
```bash
# Check network configuration
ip addr show eth0 | grep 192.168.1.50
ping 192.168.1.1XX  # Replace XX with your sensor's serial number digits

# Check driver process
ps aux | grep livox_ros_driver2_node

# Verify LiDAR is powered and Ethernet cable is connected
```

#### No ROS Topics Published

**Symptoms**: Driver launches but no topics appear

**Solutions**:
```bash
# List all topics
ros2 topic list

# Check if driver node is running
ros2 node list

# Check topic rate
ros2 topic hz /livox/lidar

# Verify network connectivity (replace XX with your sensor's serial number digits)
ping 192.168.1.1XX
```

#### Fast-LIO2 Errors

**Symptoms**: SLAM node fails to start or shows errors

**Solutions**:
```bash
# Verify driver is running first
ros2 topic list | grep livox

# Check IMU data
ros2 topic echo /livox/imu --once

# Verify configuration file exists
ls src/FAST_LIO_ROS2/config/avia.yaml
```

#### Architecture Mismatch (ARM64/Jetson)

**Symptoms**: "Exec format error" when running binaries

**Solutions**:
- Ensure all packages are built from source on the target platform
- Check binary architecture:
```bash
file ros2_ws/install/livox_ros_driver2/lib/livox_ros_driver2/livox_ros_driver2_node
# Should show: "ELF 64-bit LSB ... ARM aarch64"
```

#### RViz Shows Black Screen

**Symptoms**: RViz launches but displays nothing

**Solutions**:
- Verify TF tree exists: `ros2 run tf2_tools view_frames`
- Check that topics are publishing: `ros2 topic hz /livox/lidar`
- Ensure correct frame IDs in configuration

### Performance Optimization

- Adjust point cloud publishing rate based on computational resources
- Use point cloud filters to reduce data volume
- Consider using compressed point cloud topics for network efficiency
- For embedded platforms (Jetson), monitor CPU/GPU usage and adjust accordingly

## Platform-Specific Notes

### NVIDIA Jetson (ARM64)

- Tested on Jetson Orin Nano and Xavier NX
- Use `setuptools<70` for Python dependencies
- All packages must be built from source
- Monitor thermal throttling during long operations

### x86_64 Systems

- Standard Ubuntu 22.04 installation should work
- All build steps are the same
- Generally better performance than ARM platforms

## Summary

The LIVOX Mid-360 LiDAR provides an excellent solution for robotic perception with its wide field of view and reliable performance. This guide covered hardware setup, network configuration, driver installation with Livox SDK2, ROS 2 Humble integration, and advanced topics including Fast-LIO2 SLAM integration and point cloud processing. With proper configuration, the sensor can serve as a robust foundation for SLAM, navigation, and obstacle avoidance applications in ROS 2 environments.

## See Also:
- [Point Cloud Library, 3D Sensors and Applications](/wiki/sensing/pcl/)
- [ROS Mapping and Localization](/wiki/common-platforms/ros/ros-mapping-localization/)
- [ROS Navigation](/wiki/common-platforms/ros/ros-navigation/)
- [Cartographer SLAM ROS Integration](/wiki/state-estimation/Cartographer-ROS-Integration/)

## Further Reading
- [LIVOX Official Documentation](https://www.livoxtech.com/)
- [Livox SDK2 GitHub Repository](https://github.com/Livox-SDK/Livox-SDK2)
- [livox_ros_driver2 GitHub Repository](https://github.com/Livox-SDK/livox_ros_driver2)
- [Fast-LIO2 GitHub Repository](https://github.com/hku-mars/FAST_LIVO2)

## References
- [1] Xu, W., & Cai, Y., et al. "FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter." IEEE Robotics and Automation Letters, 2021.
