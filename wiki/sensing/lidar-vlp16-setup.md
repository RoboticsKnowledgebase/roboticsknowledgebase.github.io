---
date: 2026-04-29
title: Velodyne VLP-16 LiDAR Setup Guide
---

The Velodyne VLP-16 ("Puck") is a 16-channel 3D LiDAR widely used in robotics for mapping, localization, and obstacle detection. This tutorial covers the full setup on Ubuntu 24.04 with ROS 2 Jazzy — from hardware connection to point cloud visualization.

## Hardware Requirements

Before beginning, confirm you have the following:

**Hardware:**
- Velodyne VLP-16 Lidar
- Velodyne interface cable (provides both power and Ethernet breakout)
- DC power supply (check the voltage and current requirement on the VLP16 user manual)
- A PC with a physical Ethernet port, or a USB-to-Ethernet adapter

**Software (on host PC):**
- Ubuntu 24.04
- ROS 2 Jazzy (installation instructions at [docs.ros.org](https://docs.ros.org/en/jazzy/Installation.html))

> **Note:** This guide is based on Ubuntu 24.04 and ROS 2 Jazzy. The general steps apply to other Ubuntu/ROS 2 combinations, but package names, file paths, and commands may differ slightly.

---

## Step 1: Physical Connection and Power-Up

1. Connect the VLP-16's interface cable to your PC's Ethernet port (or USB-to-Ethernet adapter).
2. Connect the power leads to your DC power supply.
3. Power on the supply.

Within a few seconds of power-up, the sensor should begin spinning. The status LEDs will be turned on once the sensor is operational. If the unit does **not** spin, double-check if the power source provides correct voltage level and supply current.

---

## Step 2: Configure a Static IP Address on Ubuntu
All data is streamed over **UDP/IP Ethernet**. Check the sensor's default IP address in the user manual, and set your PC to a static IP on the same subnet to receive packets.

The commands in the steps below use `192.168.1.201` as the lidar's IP and `192.168.1.100` as the PC's IP.


### 2.1 Identify Your Ethernet Interface

With the Ethernet cable **not yet connected** to the LiDAR, run:

```bash
ip a
```

Look for an interface name such as `enp0s31f6`, `eth0`, `eno1`, or, for USB adapters, something like `enx144fd7c114ac`. 

Note this name, and we will call it `<iface>` below. We will use `<iface>` as a placeholder for the actual network interface name in the below commands.

### 2.2 Assign a Temporary Static IP (Command Line)

```bash
sudo ip addr flush dev <iface>
sudo ip addr add 192.168.1.100/24 dev <iface>
sudo ip link set <iface> up
```

For example, if your interface is `enx144fd7c114ac`:

```bash
sudo ip addr flush dev enx144fd7c114ac
sudo ip addr add 192.168.1.100/24 dev enx144fd7c114ac
sudo ip link set enx144fd7c114ac up
```

Verify the address was applied:

```bash
ip a show <iface>
```

You should see `inet 192.168.1.100/24` in the output.

### 2.3 (Optional) Make the Static IP Permanent

The `ip addr` commands above are temporary — they will be lost after a reboot. If you want to persist the configuration, edit your Netplan configuration:

```bash
sudo nano /etc/netplan/01-netcfg.yaml
```

Add or modify the section for your interface:

```yaml
network:
  version: 2
  ethernets:
    <iface>:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
```

Apply the configuration:

```bash
sudo netplan apply
```

---

## Step 3: Verify the Network Connection

Now **connect the Ethernet cable to the LiDAR** and ping the sensor's default IP:

```bash
ping 192.168.1.201
```

A successful response confirms network connectivity. If ping fails, check:
- The Ethernet cable is seated in both the LiDAR interface box and the PC
- The sensor is powered and spinning
- Your IP was correctly assigned 


### 3.1 Verify UDP Data Packets (No ROS Required)

Even before installing any ROS packages, you can confirm the sensor is transmitting data using `tcpdump`:

```bash
sudo tcpdump -i <iface> udp port 2368
```

If the sensor is working, you will see a rapid flood of UDP packets. This step is highly recommended as it isolates hardware/network issues from ROS configuration issues. If `tcpdump` shows packets but ROS does not receive data, the problem is software-side. If `tcpdump` shows nothing, the problem is hardware or network.

### 3.2 Access the Velodyne Web Interface

The VLP-16 hosts a small configuration web page you can access from any browser:

```
http://192.168.1.201
```

From this page you can:
- Change the sensor's IP address
- Adjust the motor RPM (300–1200 RPM; lower RPM = denser point cloud per rotation)
- Switch return mode: **Strongest**, **Last**, or **Dual** (dual doubles packet rate)
- Update firmware

> If you change the sensor IP here, remember to update your static IP assignment and all ROS launch configurations to match the new address.

---

## Step 4: Install the Velodyne ROS 2 Driver

Install the Velodyne driver packages from the ROS 2 apt repository:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-velodyne*
```

This installs three packages:
- `velodyne_driver` — reads raw UDP packets from the sensor and publishes `VelodyneScan` messages
- `velodyne_pointcloud` — converts raw scans into standard `sensor_msgs/PointCloud2` messages
- `velodyne_msgs` — message type definitions

Source your ROS 2 environment (add to `~/.bashrc` if you haven't already):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

---

## Step 5: Launch the Driver and Verify Topics

Launch all Velodyne nodes with the VLP-16 configuration:

```bash
ros2 launch velodyne velodyne-all-nodes-VLP16-launch.py
```

In a second terminal, list the active topics:

```bash
ros2 topic list
```

You should see at minimum:

```
/velodyne_packets
/velodyne_points
```

Confirm data is flowing:

```bash
ros2 topic echo /velodyne_points --once
```

If you see a large block of point data printed to the terminal, your hardware and driver are fully functional.

---

## Step 6: Visualize Point Clouds in RViz2

RViz2 is the standard 3D visualization tool bundled with ROS 2. It is the fastest way to inspect your point cloud data.

Launch RViz2:

```bash
rviz2
```

In the RViz2 interface, configure the following:

1. **Fixed Frame:** In the "Global Options" panel on the left, set the `Fixed Frame` to `velodyne`. This tells RViz2 to render everything relative to the LiDAR's own coordinate frame.

2. **Add a PointCloud2 display:** Click the **Add** button at the bottom of the Displays panel, select **By topic**, and choose `/velodyne_points` of type `PointCloud2`.

You should now see a live, rotating 360° ring of points around the sensor. Moving objects will update in real time. If you see a static or partial scan, check the rotation rate — at 300 RPM the update is slower but each scan is denser.

---

## Step 7: Visualize Point Clouds in Foxglove Studio

[Foxglove Studio](https://foxglove.dev/) is a modern, web-based (and installable) visualization tool that works with ROS 2 and is particularly useful when you want a richer interface than RViz2, or when working with recorded bag files.

### 7.1 Install Foxglove Studio

Download the desktop application from [foxglove.dev/download](https://foxglove.dev/download), or use the web version at [studio.foxglove.dev](https://studio.foxglove.dev).

Install the Foxglove bridge for ROS 2:

```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```

### 7.2 Launch the Bridge

With your Velodyne driver already running in one terminal, open a second terminal and start the bridge:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

By default, this exposes a WebSocket server on port `8765`.

### 7.3 Connect and Visualize

1. Open Foxglove Studio (desktop or browser).
2. Click **Open connection** → **Rosbridge / Foxglove WebSocket**.
3. Enter the address `ws://localhost:8765` (or replace `localhost` with your PC's IP if connecting from another machine on the network).
4. Once connected, click the **+** button to add a **3D** panel.
5. In the panel settings, add `/velodyne_points` as a topic. The point cloud will render in 3D immediately.

Foxglove offers several advantages over RViz2 for data exploration: you can overlay multiple sensor streams, adjust color maps, and use the timeline scrubber when replaying bag files — all from a clean, browser-style interface.

---

## Step 8: Record and Replay Data with ROS 2 Bags

Recording sensor data is essential for algorithm development and debugging, as it lets you replay exactly what the sensor captured during a real-world run.

### 8.1 Record a Bag File

Navigate to the directory where you want to store the bag, then run:

```bash
ros2 bag record /velodyne_points
```

This records the `/velodyne_points` topic. To record multiple topics simultaneously:

```bash
ros2 bag record /velodyne_points /velodyne_packets /tf
```

Press `Ctrl+C` to stop recording. The output will be a directory named:

```
rosbag2_YYYY_MM_DD-HH_MM_SS/
```

### 8.2 Replay a Bag File

```bash
ros2 bag play rosbag2_YYYY_MM_DD-HH_MM_SS
```

While the bag is playing, you can open RViz2 or Foxglove Studio exactly as described above — they will receive the replayed topic data just as if the physical sensor were connected.

Useful replay options:

```bash
# Play at half speed (useful for slow-motion inspection)
ros2 bag play rosbag2_YYYY_MM_DD-HH_MM_SS --rate 0.5

# Loop the bag continuously
ros2 bag play rosbag2_YYYY_MM_DD-HH_MM_SS --loop
```

### 8.3 Inspect Bag Contents

```bash
ros2 bag info rosbag2_YYYY_MM_DD-HH_MM_SS
```

This prints the recording duration, number of messages per topic, and storage format, which is a useful way to verify if a ROS bag is corrupted.

---


## Summary

This tutorial covered VLP-16 setup: hardware connection, static IP configuration, ROS 2 driver setup, and point cloud visualization. 
- [Point Cloud Library (PCL), 3D Sensors and Applications](/wiki/sensing/pcl/)
  — Next step after getting the sensor running: processing and filtering the point cloud data in your pipeline.
- [Cartographer SLAM ROS Integration](/wiki/state-estimation/cartographer-ros-integration/)
  — A LiDAR-based SLAM algorithm that consumes `/velodyne_points` directly for map building and localization.
- [ROS Mapping and Localization](/wiki/common-platforms/ros/ros-mapping-localization/)
  — Overview of ROS mapping packages (gmapping, Hector Mapping) compatible with LiDAR data.
- [ROS 2 Humble Intra-Process Communication Bag Recorder](/wiki/tools/ros2-humble-ipc-recorder/)
  — Optimized bag recording for high-bandwidth sensors like LiDAR, reducing CPU overhead vs. standard `ros2 bag record`.
- [Stream Rviz Visualizations as Images](/wiki/tools/stream-rviz/)
  — How to stream your RViz2 point cloud view as an image topic, useful for remote monitoring or logging.

## Further Reading
- [Velodyne VLP-16 User Manual](https://ouster.com/downloads/velodyne-downloads)
- [ROS 2 Velodyne Driver Documentation](https://docs.ros.org/en/jazzy/p/velodyne_driver/)
- [Foxglove Studio Documentation](https://docs.foxglove.dev/)