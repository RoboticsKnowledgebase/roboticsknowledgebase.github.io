---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2026-04-29 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Ouster OS1-32 LiDAR Integration Guide
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

The Ouster OS1-32 is a mid-range 32-channel LiDAR available in the MRSD inventory for SLAM, obstacle detection, and terrain mapping. It utilizes a spinning assembly to cover a full 360° horizontal field of view while firing all 32 vertical beams simultaneously, which significantly reduces inter-beam timing errors compared to sequential sensors.

This article serves as a technical guide for integrating the OS1-32 into your robotics stack. It covers critical sensor theory, including photon counting technology, and provides specific instructions for hardware setup, power requirements, and ROS2 Humble implementation. Readers will also find troubleshooting advice for common pitfalls such as networking issues and ARM64-specific software bugs.

## Sensor Specifications and Theory
Understanding the hardware capabilities and underlying detection principles is essential for effective data processing.

### Key Specifications
The OS1-32 provides high-resolution vertical data and reliable range performance.

| Parameter | Value |
| :--- | :--- |
| Channels | 32 (Vertical Resolution) |
| Range | 120m (at 80% reflectivity) |
| Field of View (FoV) | 360° Horizontal / 45° Vertical |
| Precision | ±0.7 cm to ±5 cm (depending on range) |

### Sensor Theory
The OS1-32 uses photon counting rather than measuring reflected light intensity in the conventional way. In practice, this means it handles low-reflectivity surfaces and direct sunlight better than older sensors, resulting in fewer phantom returns and less washout outdoors.



## Hardware and Networking Setup
Initial configuration is best handled through specialized tools before moving to complex software environments.

### First Connection: Ouster Studio
Before touching ROS, get the sensor talking to your machine using Ouster Studio, Ouster's free desktop app for Windows, macOS, and Ubuntu. It auto-detects the sensor on the network via mDNS and provides a live point cloud view with zero configuration. This is the fastest way to confirm the sensor is powered, connected, and healthy.

> **Note**: On Linux, you must set your Ethernet interface to **Link-Local Only** mode in your network manager settings for the sensor to be discovered.

### Power Supply
The datasheet lists an operating range of 12V-24V, but **12V is unreliable**. The sensor draws up to 20W continuously (28W peak at cold startup); at 12V, this requires over 2A, and any voltage sag under load can cause a dropout. Ouster ships the interface box with a 24V supply and assumes this in their guides. **Use 24V**.

### Finding the Sensor IP Address
The sensor self-assigns a link-local address (`169.254.x.x`) when no DHCP server is present. You can find it using `arp-scan`:

```bash
sudo arp-scan --interface=<your_eth_interface> --localnet
```

Replace `<your_eth_interface>` with the name of your Ethernet port (e.g., `eth0` or `enp3s0`). Alternatively, reach the sensor via hostname at `http://os-<serialnumber>.local` in a browser.

## ROS2 Integration (Humble)
The Ouster does not output standard `sensor_msgs/PointCloud2` natively; it sends raw UDP packets that the driver must decode.

### The ouster-ros Driver
The `ouster-ros` driver handles decoding and publishes standard ROS messages, including point clouds and data from the built-in 6-axis IMU.

### Packet Buffering and Phase Lock
The driver reconstructs a 360° scan by buffering packets. For multi-sensor fusion, Phase Lock allows you to pin the start of each scan to a specific azimuth angle to help align timestamps across sensors.

### Launching the Sensor
The sensor requires a Gigabit Ethernet connection. If you see dropped packets, increase the Linux UDP receive buffer:

```bash
sudo sysctl -w net.core.rmem_max=26214400
```

Launch the driver with the following command:

```bash
ros2 launch ouster_ros sensor.launch.py \
sensor_hostname:=169.254.x.x \
viz:=true \
lidar_mode:=1024x10
```

* **1024x10**: 1024 horizontal points at 10Hz (default).
* **2048x10**: Higher horizontal resolution, but doubles the data rate.

## Advanced Configuration and Processing
Once data is flowing, additional steps may be required for specific architectures or multi-sensor setups.

### Coordinate Frames
The OS1-32 exposes three TF frames where the origins do not coincide:
* **os_sensor**: The physical housing.
* **os_lidar**: The actual origin of the laser pulses.
* **os_imu**: The IMU origin.

Ensure your `static_transform_publisher` uses the offsets from the Ouster manual, not zeros.



### ARM64 / Jetson Orin Notes
While the `ouster-ros` driver builds fine on ARM64, downstream packages like FAST-LIO may crash because the sensor emits `NaN` for points below its minimum viable range floor. This is a FAST-LIO bug not gracefully handled on ARM64 due to hardware floating-point compute differences. The fix is to add a filter on the range field to change `NaN` values to a valid number (usually 0) before the data reaches the algorithm.

## Summary
The OS1-32 is a capable sensor, but successful integration relies on handling three main hurdles: stable networking (verify with Ouster Studio first), adequate power (always use 24V), and managing ARM64 architectural nuances.

| Issue | Potential Cause | Solution |
| :--- | :--- | :--- |
| No data in RViz | IP mismatch | Use `arp-scan` or `avahi-browse` to find the sensor IP. |
| "Jagged" walls | Time sync | Sync the sensor via PTP or ensure system clock consistency. |
| Sensor overheating | Poor thermal path | Mount the base to a metal plate for heat dissipation. |
| Dropped packets | UDP buffer too small | Increase Linux UDP receive buffer via `sysctl`. |
| Won't connect | Underpowered supply | Switch to a 24V supply. |
