This article provides a comprehensive guide to the LIVOX Mid-360 LiDAR sensor, a popular choice in both academia and industry for robotic applications. This tutorial covers hardware connections, driver setup procedures, and step-by-step integration into the ROS 2 ecosystem. Advanced topics include Fast-LIO2 SLAM integration that leverages LiDAR-IMU data for robust odometry and mapping, point cloud processing pipelines. Following this guide, readers will be able to set up and integrate the LIVOX Mid-360 LiDAR into their robotic projects with ROS 2 Humble.

<img src="assets/livox-mid-360-lidar.png" alt="LIVOX Mid-360 LiDAR" width="30%">

## Table of Contents

This guide is organized into logical sections that build upon each other. Follow the sections sequentially for a complete setup, or jump directly to any section based on your needs:

**Quick Start Path**: For users who want to get the sensor running quickly, follow this sequence:
1. [Hardware Connections](#2-hardware-connections) → 2. [Network Configuration](#network-configuration) → 3. [Driver Setup](#3-driver-setup) → 4. [ROS 2 Integration](#4-ros-2-integration)

**Complete Integration Path**: For full SLAM and navigation capabilities:
1. Quick Start Path (above) → 2. [Fast-LIO2 SLAM Integration](#51-fast-lio2-slam-integration) → 3. [Ego-Planner Path Planning Integration](#52-ego-planner-path-planning-integration) → 4. [Complete System Workflow](#6-complete-system-workflow)

---

### Part I: Getting Started

**1. [Sensor Overview](#1-sensor-overview)**
   - Technical specifications and sensor characteristics
   - Scanning pattern and point cloud characteristics
   - Understanding sensor capabilities and limitations

**2. [Hardware Connections](#2-hardware-connections)**
   - [Physical Setup](#physical-setup) - Power supply, Ethernet connection, mounting considerations
   - [Network Configuration](#network-configuration) - IP address setup, connectivity verification
     - LiDAR IP address configuration
     - Computer IP address setup
     - Temporary and permanent network configuration
     - Advanced multi-network setups

---

### Part II: Software Installation

**3. [Driver Setup](#3-driver-setup)**
   - [Prerequisites](#prerequisites) - System requirements, OS, ROS 2, and dependencies
   - [Installing LIVOX SDK2](#installing-livox-sdk2) - Low-level SDK installation and verification
   - [Installing ROS 2 Driver](#installing-ros-2-driver-livox_ros_driver2) - ROS 2 integration package

**4. [ROS 2 Integration](#4-ros-2-integration)**
   - [Launching the Driver](#launching-the-driver) - Starting the sensor node
   - [Published Topics](#published-topics) - Understanding data formats and message types
   - [Verifying Data Stream](#verifying-data-stream) - Confirming sensor operation
   - [Configuration](#configuration) - Parameter adjustment and optimization

---

### Part III: Advanced Applications

**5. [Advanced Topics](#5-advanced-topics)**

   **5.1 [Fast-LIO2 SLAM Integration](#51-fast-lio2-slam-integration)**
   - Real-time mapping and localization
   - [Installing Dependencies](#installing-dependencies) - Required libraries (Sophus, PCL, Eigen)
   - [Building Fast-LIO2](#building-fast-lio2) - Compilation and workspace setup
   - [Running Fast-LIO2](#running-fast-lio2) - Launching SLAM node
   - [Configuration](#configuration-1) - Parameter tuning for Mid-360

   **5.2 [Ego-Planner Path Planning Integration](#52-ego-planner-path-planning-integration)**
   - Gradient-based trajectory planning
   - [Why Ego-Planner with Mid-360?](#why-ego-planner-with-mid-360) - Integration advantages
   - [Installing Ego-Planner](#installing-ego-planner) - Package installation
   - [Configuring Ego-Planner](#configuring-ego-planner-for-mid-360) - Parameter setup
   - [Running Ego-Planner](#running-ego-planner) - Launching planning node
   - [Integration Workflow](#integration-workflow) - Complete system architecture
   - [Performance Considerations](#performance-considerations) - Optimization guidelines

---

### Part IV: System Integration and Troubleshooting

**6. [Complete System Workflow](#6-complete-system-workflow)**
   - Running all components together
   - Multi-terminal setup procedures
   - Visualization and monitoring

**7. [Troubleshooting](#7-troubleshooting)**
   - [Common Issues](#common-issues) - Connection problems, topic issues, errors
     - LiDAR not connecting
     - No ROS topics published
     - Fast-LIO2 errors
     - Architecture mismatch (ARM64/Jetson)
     - RViz shows black screen
   - [Performance Optimization](#performance-optimization) - System tuning and resource management

**8. [Platform-Specific Notes](#8-platform-specific-notes)**
   - [NVIDIA Jetson (ARM64)](#nvidia-jetson-arm64) - Embedded platform considerations
   - [x86_64 Systems](#x86_64-systems) - Desktop development platforms
   - Multi-sensor setup guidelines

---

### Part V: Reference and Applications

**9. [Application Scenarios](#9-application-scenarios)**
   - Mobile robot navigation and SLAM
   - 3D mapping and surveying (including handheld scanning systems)
   - Infrastructure inspection
   - Drone and aerial applications
   - Research and development use cases

**10. [Summary](#10-summary)**
   - Key takeaways and best practices

**11. Additional Resources**
   - [See Also](#see-also) - Related wiki articles
   - [Further Reading](#further-reading) - External documentation and resources
   - [References](#references) - Academic papers and technical references

---

## 1. Sensor Overview

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

### Technical Specifications Details

The Mid-360 employs a 905 nm laser wavelength, classified as Class 1 eye-safe (IEC60825-1:2014 standard), ensuring safe operation in human environments. The sensor's angular resolution improves over time due to its non-repetitive scanning pattern, which enhances small object detection capabilities. The point cloud output rate reaches 200,000 points per second at a typical frame rate of 10 Hz, providing dense environmental data for high-fidelity mapping and localization.

Range accuracy specifications include:
- **Random range error (1σ)**: ≤ 2 cm at 10 m distance, ≤ 3 cm at 0.2 m distance
- **Angular random error (1σ)**: < 0.15°
- **False alarm rate**: < 0.01% at 100 klx ambient light

The built-in IMU (ICM40609) provides inertial data at high frequency, enabling tight coupling with LiDAR data for robust odometry estimation. The sensor operates in temperatures ranging from -20°C to 55°C with IP67 protection rating, making it suitable for harsh industrial environments.

### Scanning Pattern and Point Cloud Characteristics

Unlike traditional mechanical LiDARs that use repetitive scanning patterns, the Mid-360 employs Livox's proprietary non-repetitive scanning technology. This pattern ensures that over time, the angular resolution improves significantly, with more points accumulating in previously sparse areas. This characteristic is particularly beneficial for SLAM applications, as it provides increasingly detailed maps as the robot moves through the environment.

The vertical field of view distribution is not uniform across the range. The effective detection range varies within the vertical FOV: areas closer to the top have shorter effective ranges, while areas closer to the bottom have longer ranges. This design optimizes the sensor for ground-based mobile robots, where most obstacles and navigation features are located at lower elevations.

Understanding these technical characteristics is essential for proper sensor integration. The combination of wide field of view, high point cloud density, and robust environmental performance makes the Mid-360 particularly well-suited for applications requiring comprehensive spatial awareness. However, to fully leverage these capabilities, proper hardware setup and configuration are critical. The following sections will guide you through the physical installation, network configuration, and software integration necessary to bring the sensor online in your robotic system.

   <img src="assets/livox-mid-360-lidar-fov.png" alt="LIVOX Mid-360 LiDAR" width="70%">


> **Source**: Specifications and technical details are based on the [official LIVOX Mid-360 product page](https://www.livoxtech.com/mid-360).

## 2. Hardware Connections

Before diving into software configuration, establishing proper physical connections is the foundation of a successful integration. The Mid-360's design emphasizes ease of installation while maintaining reliability in various operating conditions. This section covers the essential hardware connections required to power the sensor and establish communication with your computing platform.

### Physical Setup

![LIVOX Mid-360 LiDAR Connections](assets/livox-mid-360-lidar-connections.png)

The LIVOX Mid-360 requires the following connections:

1. **Power Supply**: Connect the power adapter to the sensor. The sensor operates at 9-27V DC with a power consumption of 6.5W. The power connector uses a standard aviation connector (M12), and Livox provides a splitter cable that separates power, Ethernet, and function connections. Ensure the power supply can provide sufficient current (typically 0.5-0.7A at 12V). Pay attention to polarity: the center pin is positive, and the outer shell is ground.

2. **Ethernet Connection**: Connect the sensor directly to your computer or Jetson Orin using an Ethernet cable (1000BASE-T). While the sensor supports 1000BASE-T, it actually uses 100BASE-TX for data transmission. Use a high-quality Ethernet cable (Cat5e or better) with proper shielding to minimize interference. The cable length should not exceed 100 meters for reliable communication. For mobile robot applications, consider using flexible, shielded cables that can withstand repeated bending.

3. **Function Connector (Optional)**: The M12 function connector supports GPS time synchronization (PPS and GPS input) and PTPv2 network time synchronization. The connector pinout includes:
   - Pin 8 (Gray/White): LVTTL_IN for GPS input
   - Pin 10 (Purple/White): LVTTL_IN for Pulse Per Second (PPS)
   - Pin 9 (Gray): LVTTL_OUT (reserved)
   - Pin 11 (Purple): LVTTL_OUT (reserved)
   - Black: Ground

   For GPS synchronization, configure the GPS module to output NMEA messages at 9600 baud rate, 8 data bits, no parity, 1 stop bit. The PPS signal should be a 3.3V TTL pulse with 1 Hz frequency.

4. **Mounting**: Ensure the sensor is securely mounted on your robot platform. The sensor should be positioned to maximize its field of view for your specific application. Use the provided mounting holes (M3 threads) and ensure the mounting surface is flat and rigid. For optimal thermal management, mount the sensor on a metal surface with good thermal conductivity. Maintain at least 5 cm clearance around the sensor for proper heat dissipation. Avoid mounting near heat sources or in areas with restricted airflow.
   
   **Mounting Tip**: For optimal horizontal FOV coverage, consider mounting the sensor at a slight angle. For example, [Tare Robotics](https://www.tarerobotics.com/) mounts the Mid-360 at a 20-degree tilt angle on their T-Bot platform, which helps balance the horizontal field of view distribution and improves ground-level obstacle detection.

Once the physical connections are established and the sensor is properly mounted, the next critical step is configuring the network interface. Unlike USB-connected sensors, the Mid-360 relies entirely on Ethernet communication, making network configuration a prerequisite for any data acquisition. Proper network setup ensures reliable, low-latency data transmission between the sensor and your computing platform, which is essential for real-time robotic applications.

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

With network connectivity established, you can proceed to software installation. However, before moving forward, it's worth understanding advanced network configurations that may be necessary for complex setups. Many robotic systems require simultaneous internet connectivity for software updates, remote monitoring, or cloud services, while maintaining direct communication with the LiDAR sensor. The following section addresses these scenarios, then we'll move on to driver installation.

## 3. Driver Setup

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

**Important**: The LIVOX Mid-360 requires **Livox SDK2** (not SDK1). SDK1 will not work with Mid-360. The SDK2 provides the low-level communication protocol and device management functions necessary for Mid-360 operation.

The SDK2 architecture includes:
- **Device discovery**: Automatic detection of connected Livox sensors on the network
- **Data streaming**: Efficient point cloud and IMU data transmission
- **Device control**: Parameter configuration and status monitoring
- **Time synchronization**: Support for GPS and PTPv2 time sync protocols

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

If successful, you should see IMU and LiDAR streaming messages. The sample program will display connection status, device information, and real-time point cloud statistics including point count and frame rate.

For advanced users, the SDK2 provides C++ APIs for custom applications. Key API functions include:
- `LivoxSdkInit()`: Initialize the SDK
- `LivoxSdkStart()`: Start device discovery and data streaming
- `SetDataCallback()`: Register callback functions for point cloud and IMU data
- `LivoxSdkUninit()`: Clean up resources

Refer to the SDK2 documentation and sample code in the `samples/` directory for implementation examples.

While the SDK2 provides direct access to sensor data, most robotic applications benefit from integration with the Robot Operating System (ROS). The ROS 2 driver serves as a bridge between the Livox SDK2 and the ROS ecosystem, converting raw sensor data into standard ROS message formats that can be easily consumed by navigation, SLAM, and perception algorithms. This abstraction layer simplifies development and enables seamless integration with the broader ROS software ecosystem.

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

Having installed both the SDK2 and the ROS 2 driver, you're now ready to launch the sensor and begin receiving data through ROS topics. The driver handles the complex task of converting Livox's proprietary data format into ROS messages, managing network communication, and publishing sensor data at appropriate rates. This section covers the practical aspects of launching the driver, verifying data streams, and configuring parameters to match your specific application requirements.

## 4. ROS 2 Integration

### Launching the Driver

Launch the LIVOX Mid-360 driver:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

### Published Topics

The driver publishes the following topics:

- `/livox/lidar`: Point cloud data in `livox_ros_driver2/CustomMsg` format. This custom message type contains raw point cloud data with timestamps, point coordinates (x, y, z), and intensity values. The message structure is optimized for Livox's non-repetitive scanning pattern and includes frame information for proper point cloud reconstruction.

- `/livox/imu`: IMU data in `sensor_msgs/Imu` format. The IMU provides linear acceleration and angular velocity measurements at high frequency (typically 200 Hz), which is essential for motion estimation and sensor fusion algorithms. The IMU data includes covariance matrices for uncertainty estimation.

The driver also publishes TF transforms:
- `base_link` → `livox_frame`: Transform from robot base to LiDAR sensor frame
- The transform includes the mounting position and orientation of the sensor

Topic publication rates:
- `/livox/lidar`: Typically 10 Hz (configurable)
- `/livox/imu`: Typically 200 Hz (hardware dependent)

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

The basic ROS 2 integration provides point cloud and IMU data streams, which are sufficient for many applications. However, for advanced robotic systems requiring simultaneous localization and mapping (SLAM), obstacle avoidance, or path planning, additional processing is necessary. The following section introduces Fast-LIO2, a state-of-the-art SLAM algorithm specifically designed to work with Livox sensors, demonstrating how to transform raw sensor data into actionable navigation information.

## 5. Advanced Topics

### 5.1 Fast-LIO2 SLAM Integration

Fast-LIO2 is a computationally efficient and robust LiDAR-inertial odometry framework that works well with LIVOX sensors. It provides real-time odometry and mapping capabilities. The algorithm uses an iterated Kalman filter to tightly couple LiDAR and IMU measurements, achieving high accuracy with low computational cost.

Key advantages of Fast-LIO2 for Mid-360:
- **Non-repetitive scan handling**: Designed to work with Livox's unique scanning patterns
- **Real-time performance**: Typically runs at 10-20 Hz on modern hardware
- **Robust to motion**: Handles aggressive motions and vibrations well
- **Memory efficient**: Incremental map building without storing full point clouds
- **Open source**: Actively maintained with ROS 2 support

The algorithm processes incoming point clouds incrementally, extracting features and matching them with the current map estimate. IMU data provides motion prediction between LiDAR scans, improving accuracy during fast movements.

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

For Mid-360, the default `avia.yaml` configuration works well for most applications. Adjust `filter_size_surf` and `filter_size_map` based on your environment: smaller values (0.2-0.3 m) for indoor environments, larger values (0.5-1.0 m) for outdoor environments.

While Fast-LIO2 provides excellent odometry and mapping capabilities, many robotic applications require not just localization but also path planning and obstacle avoidance. The point cloud maps generated by Fast-LIO2 serve as the foundation for navigation algorithms, but transforming these maps into executable trajectories requires additional planning components. Ego-Planner represents a state-of-the-art solution for this challenge, offering efficient gradient-based path planning that works directly with point cloud data without requiring expensive distance field computations.

### 5.2 Ego-Planner Path Planning Integration

Ego-Planner is an ESDF-free gradient-based local planner designed for efficient and safe trajectory generation. Unlike traditional planning methods that require building Euclidean Signed Distance Fields (ESDF) for gradient optimization, Ego-Planner performs optimization directly on point cloud data, significantly reducing computational overhead while maintaining safety guarantees. This makes it particularly well-suited for real-time applications where computational resources are limited, such as mobile robots and drones.

The algorithm's key innovation lies in its ability to construct effective collision penalty terms by comparing collision-prone trajectories with collision-free reference paths, all without explicitly building distance fields. This approach, combined with an anisotropic curve fitting algorithm, produces smooth, feasible trajectories that respect both dynamic constraints and obstacle boundaries.

#### Why Ego-Planner with Mid-360?

The combination of LIVOX Mid-360 and Ego-Planner offers several advantages for robotic navigation systems. The Mid-360's dense point cloud output provides rich environmental information that Ego-Planner can leverage for accurate obstacle representation. The sensor's 360° horizontal field of view ensures comprehensive coverage, eliminating blind spots that could lead to planning failures. Additionally, the non-repetitive scanning pattern gradually improves point cloud density over time, which enhances the quality of collision checking as the robot operates in an environment.

For mobile robot applications, this integration enables real-time reactive planning in dynamic environments. The planner can quickly adapt to newly detected obstacles, recalculating trajectories within milliseconds to ensure safe navigation. This capability is particularly valuable in environments with moving obstacles, such as warehouses with other robots or public spaces with pedestrians.

#### Installing Ego-Planner

Ego-Planner is available as an open-source ROS 2 package. To integrate it with your Mid-360 and Fast-LIO2 setup:

1. Clone the Ego-Planner repository:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ZJU-FAST-Lab/ego-planner.git
```

2. Install dependencies:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:

```bash
colcon build --packages-select ego_planner
source install/setup.bash
```

#### Configuring Ego-Planner for Mid-360

Ego-Planner requires configuration to work with the point cloud data from Fast-LIO2. The key configuration parameters include:

**Point Cloud Input:**
- `map_topic`: Set to `/cloud_registered` (Fast-LIO2's registered point cloud output)
- `point_cloud_inflation`: Inflation radius for obstacle expansion (typically 0.2-0.5 m for mobile robots)

**Planning Parameters:**
- `planning_horizon`: Maximum planning distance (adjust based on robot speed and sensor range)
- `max_vel`: Maximum velocity constraints
- `max_acc`: Maximum acceleration constraints
- `resolution`: Grid resolution for point cloud processing (balance between accuracy and computation)

**Trajectory Optimization:**
- `optimization_iterations`: Number of optimization iterations (typically 5-10)
- `smoothing_weight`: Weight for trajectory smoothness (higher values produce smoother but potentially longer paths)

Edit the configuration file `src/ego_planner/config/planning.yaml` to match your robot's specifications and operating environment.

#### Running Ego-Planner

Launch Ego-Planner in a separate terminal:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch ego_planner ego_planner.launch.py
```

**Published Topics:**
- `/planning/trajectory`: Generated trajectory waypoints
- `/planning/vis_trajectory`: Visualization markers for RViz
- `/planning/vis_check_trajectory`: Collision checking visualization

**Subscribed Topics:**
- `/cloud_registered`: Point cloud map from Fast-LIO2
- `/Odometry`: Current robot pose from Fast-LIO2
- `/goal`: Goal position (geometry_msgs/PoseStamped)

#### Integration Workflow

The complete integration involves three main components working together:

1. **Fast-LIO2** processes Mid-360 point clouds and IMU data to generate odometry and a registered point cloud map
2. **Ego-Planner** uses the point cloud map and current odometry to generate collision-free trajectories toward the goal
3. **Robot Controller** executes the planned trajectory, sending velocity commands to the robot's actuators

This pipeline enables autonomous navigation in previously unknown environments, with the robot simultaneously mapping its surroundings, localizing itself within the map, and planning safe paths to designated goals. The real-time nature of all three components ensures responsive behavior, allowing the robot to adapt quickly to environmental changes.

#### Performance Considerations

Ego-Planner's efficiency makes it suitable for resource-constrained platforms, but optimal performance requires careful parameter tuning. For indoor environments with dense obstacles, use smaller inflation radii and higher resolution grids. For outdoor environments with sparse obstacles, larger inflation radii and lower resolution can reduce computational load while maintaining safety. The planning horizon should be set based on your robot's maximum speed and the sensor's effective range—too short a horizon may cause frequent replanning, while too long a horizon may include outdated obstacle information.

Monitoring computational performance is important, especially on embedded platforms. Use tools like `htop` or `ros2 topic hz` to verify that planning cycles complete within acceptable time limits (typically 50-100 ms for real-time operation). If planning becomes too slow, consider reducing the optimization iterations or grid resolution, though this may impact trajectory quality.

Now that we've covered the individual components—hardware setup, network configuration, driver installation, SLAM integration, and path planning—it's time to bring everything together into a complete working system. A typical robotic application requires coordinating multiple processes running simultaneously, each handling different aspects of sensor data processing, localization, mapping, and robot control. The following workflow demonstrates how these components interact in a real-world deployment scenario, providing a practical template for your own implementations.

## 6. Complete System Workflow

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

Even with careful setup and configuration, real-world deployments often encounter unexpected issues. These can range from network connectivity problems to software compatibility issues, or environmental factors affecting sensor performance. The troubleshooting section that follows addresses the most common problems encountered during Mid-360 integration, providing systematic approaches to diagnosis and resolution. Understanding these potential pitfalls and their solutions will save significant time during development and deployment.

## 7. Troubleshooting

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

## 8. Platform-Specific Notes

### NVIDIA Jetson (ARM64)

- Tested on Jetson Orin Nano and Xavier NX
- Use `setuptools<70` for Python dependencies
- All packages must be built from source
- Monitor thermal throttling during long operations

The platform-specific considerations discussed above highlight the importance of understanding your target hardware environment. Different platforms have different strengths and limitations, and optimizing your setup for the specific platform can significantly impact system performance. Whether you're deploying on embedded systems for field operation or developing on desktop computers for algorithm testing, these considerations should inform your configuration choices.

### x86_64 Systems

- Standard Ubuntu 22.04 installation should work
- All build steps are the same
- Generally better performance than ARM platforms

## 9. Application Scenarios

The LIVOX Mid-360's compact design, wide field of view, and high point cloud density make it suitable for diverse applications across multiple industries. Here are some notable use cases:

### Mobile Robot Navigation and SLAM

**Autonomous Forklifts**: The Mid-360 is widely used in autonomous forklift systems for warehouse automation. Its 360° horizontal FOV enables comprehensive environment perception, allowing robots to navigate narrow aisles and handle complex loading scenarios. Companies like JingSong Intelligent have integrated Mid-360 into their forklift robots for precise pallet handling and outdoor navigation.

**Service Robots**: Service robots in retail, hospitality, and healthcare environments benefit from Mid-360's ability to detect obstacles at close range (10 cm minimum distance) and provide dense point clouds for accurate localization in dynamic human environments.

**Autonomous Mobile Robots (AMRs)**: The sensor's omnidirectional coverage eliminates blind spots, making it ideal for AMRs operating in manufacturing facilities, logistics centers, and other industrial environments where safety and reliability are critical.

### 3D Mapping and Surveying

**Handheld Scanning Systems**: Companies like Manifold Technology utilize Mid-360 in handheld scanning devices for various mapping applications. These systems enable rapid 3D mapping of indoor and outdoor environments, including:

- **Building Interior Mapping**: High-resolution point cloud generation for architectural documentation, facility management, and renovation planning
- **Urban Planning**: Street-level scanning for smart city applications, capturing detailed 3D models of urban environments
- **Heritage Documentation**: Precise 3D scanning of historical sites and cultural heritage locations for preservation and digital archiving
- **Construction Site Monitoring**: Regular scanning to track construction progress and verify as-built conditions

The Mid-360's compact size and low power consumption make it ideal for portable scanning systems that can be operated by a single person, significantly reducing the time and cost compared to traditional surveying methods.

### Infrastructure Inspection

**Railway Systems**: The Mid-360 is deployed in Train Intelligent Detection Systems (TIDS) for railway infrastructure monitoring. The sensor's ability to operate in various lighting conditions and provide detailed 3D point clouds enables detection of obstacles, track condition assessment, and tunnel clearance verification.

**Tunnel and Underground Mapping**: The sensor's performance in low-light conditions makes it suitable for underground infrastructure mapping, including subway systems, utility tunnels, and mining operations.

### Drone and Aerial Applications

**Autonomous Drones**: The Mid-360's lightweight design (265g) and wide field of view make it suitable for integration into drone platforms for autonomous navigation and obstacle avoidance. Research applications include indoor drone navigation where GPS is unavailable, requiring robust SLAM capabilities.

**Aerial Mapping**: When mounted on aerial platforms, the Mid-360 can provide detailed ground-level point cloud data, complementing traditional aerial LiDAR systems for comprehensive 3D mapping projects.

### Research and Development

**Academic Research**: The Mid-360 is popular in robotics research due to its cost-effectiveness, open-source driver support, and compatibility with ROS 2. Research applications include:

- Multi-robot SLAM systems
- Dynamic obstacle tracking and prediction
- Terrain analysis and traversability assessment
- Sensor fusion with cameras and other sensors

**Prototype Development**: Startups and research institutions use Mid-360 for rapid prototyping of autonomous systems, benefiting from the sensor's ease of integration and comprehensive documentation.

### Key Advantages for Different Applications

- **Indoor Environments**: The 10 cm minimum detection range and high point cloud density make Mid-360 excellent for indoor navigation where close obstacles are common
- **Outdoor Environments**: The 70 m maximum range (at 80% reflectivity) and consistent performance in bright sunlight enable reliable outdoor operation
- **Multi-Sensor Setups**: Active anti-interference capabilities allow multiple Mid-360 sensors to operate simultaneously without signal interference
- **Cost-Effective Solutions**: Compared to traditional mechanical LiDARs, Mid-360 offers similar or better performance at a lower cost point

The diverse application scenarios discussed above illustrate the versatility of the Mid-360 sensor across different industries and use cases. From industrial automation to research and development, the sensor's capabilities enable innovative solutions to complex perception challenges. Understanding these applications provides context for the technical details covered in this guide and helps readers identify how the Mid-360 might fit into their own projects. As we conclude this comprehensive guide, let us summarize the key concepts and provide a foundation for further exploration.

## 10. Summary

The LIVOX Mid-360 LiDAR provides an excellent solution for robotic perception with its wide field of view and reliable performance. This guide covered hardware setup, network configuration, driver installation with Livox SDK2, ROS 2 Humble integration, and advanced topics including Fast-LIO2 SLAM integration. The sensor's compact design, high point cloud density, and robust performance make it ideal for mobile robot applications requiring real-time mapping and localization. With proper configuration and calibration, the Mid-360 can serve as a robust foundation for SLAM, navigation, and obstacle avoidance applications in ROS 2 environments. The non-repetitive scanning pattern and active anti-interference capabilities further enhance its suitability for complex, multi-robot environments. From autonomous forklifts to handheld scanning systems, the Mid-360 demonstrates versatility across diverse application domains, making it a valuable tool for both commercial deployments and research projects.

## See Also:
- [Point Cloud Library, 3D Sensors and Applications](/wiki/sensing/pcl/)
- [ROS Mapping and Localization](/wiki/common-platforms/ros/ros-mapping-localization/)
- [ROS Navigation](/wiki/common-platforms/ros/ros-navigation/)
- [Cartographer SLAM ROS Integration](/wiki/state-estimation/Cartographer-ROS-Integration/)

## Further Reading

### Official Documentation and Resources
- [LIVOX Official Documentation](https://www.livoxtech.com/) - Comprehensive product information, user manuals, and technical specifications
- [Livox SDK2 GitHub Repository](https://github.com/Livox-SDK/Livox-SDK2) - Source code, API documentation, and sample programs
- [livox_ros_driver2 GitHub Repository](https://github.com/Livox-SDK/livox_ros_driver2) - ROS 2 driver source code and configuration examples
- [Fast-LIO2 GitHub Repository](https://github.com/hku-mars/FAST_LIVO2) - Fast-LIO2 SLAM algorithm implementation and documentation

### Application Case Studies
- [JingSong Intelligent Forklift Application](https://www.livoxtech.com/cn/showcase/20) - Case study on autonomous forklift implementation using Mid-360
- [Handheld Scanning Applications](https://www.livoxtech.com/cn/showcase/18) - Examples of 3D mapping and surveying applications
- [Smart City Applications](https://www.livoxtech.com/cn/application/smart-city) - Urban planning and infrastructure monitoring use cases

### Technical Papers and Research
- Fast-LIO: A Fast, Robust LiDAR-inertial Odometry Package - Original research paper on the Fast-LIO algorithm
- Livox Mid-360 Technical Specifications - Detailed sensor specifications and performance characteristics
- Non-repetitive Scanning Pattern Research - Academic papers on Livox's unique scanning technology

### Related Technologies
- Point Cloud Library (PCL) Documentation - Essential for point cloud processing and analysis
- ROS 2 Navigation Stack - Integration with navigation and path planning systems
- Sensor Fusion Techniques - Methods for combining LiDAR data with other sensor modalities

## References
- [1] Xu, W., & Cai, Y., et al. "FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter." IEEE Robotics and Automation Letters, 2021.
