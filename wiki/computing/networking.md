---
date: 2024-03-01
title: Setup your Network Infrastructure for Robotics Projects
---

This tutorial provides a comprehensive guide for setting up a robust networking infrastructure for robotics projects. It is based on the **MRSD Project Course - Networking Setup Guide**, which focuses on creating reliable, low-latency environments suitable for high-stakes demonstrations and complex multi-machine systems.

## Overview

In robotics, a stable network is critical for communication between onboard computers, ground stations, and remote sensors. Using shared or public networks (like university guest WiFi) often leads to high latency, jitter, and connection drops. This guide focuses on building an **isolated local network** to ensure maximum reliability.

## Core Principles

### 1. Isolated Local Network
Always use a **dedicated router** for your project. This creates a private network segment that is isolated from the traffic and interference of larger organizational networks.
*   **5GHz over 2.4GHz**: Use 5GHz (or Wi-Fi 6) whenever possible to avoid the saturation common in the 2.4GHz band.
*   **Wired Connections**: Use Ethernet cables for high-bandwidth data (e.g., raw camera streams) between fixed components.

### 2. Static IP Configuration
Relying on dynamic IPs (DHCP) can cause systems to fail if a device is assigned a new address. 
*   **DHCP Reservations**: Configure your router to assign specific IP addresses based on a device's MAC address. This is often easier than manual static configuration on every OS.
*   **Manual Static IPs**: If router access is limited, configure static IPs in the OS (e.g., using Netplan on Ubuntu).

### 3. Time Synchronization (Chrony)
For multi-machine systems like ROS, system clocks must be synchronized to ensure message timestamps and sensor data fusion are accurate.
*   Use **Chrony** to synchronize clocks across all devices on your local network.
*   Designate one machine (e.g., the ground station) as the NTP server and configure the robot to sync from it.

### 4. Host Resolution
To make your system more maintainable, map hostnames to IPs in the `/etc/hosts` file on every machine:
```text
192.168.1.10    robot-onboard
192.168.1.11    ground-station
```
This allows you to refer to machines by name (e.g., `ssh robot-onboard`) instead of memorizing IP addresses.

## ROS Multi-Machine Configuration

When running ROS across multiple computers, ensure the following environment variables are set correctly:

1.  **ROS_MASTER_URI**: Set this to the IP of the machine running `roscore`.
    *   Example: `export ROS_MASTER_URI=http://192.168.1.11:11311`
2.  **ROS_IP / ROS_HOSTNAME**: Set this to the local IP of the current machine to ensure other nodes can "call back" to it.
    *   Example: `export ROS_IP=192.168.1.10`

## Common Failure Points to Avoid

*   **Bandwidth Saturation**: Streaming raw 4K video or high-frequency LiDAR data over Wi-Fi will saturate the link. Downsample or compress data before transmission.
*   **Network Interference**: In crowded environments, perform a site survey to identify the least congested Wi-Fi channels.
*   **Background Processes**: Disable auto-updates, cloud syncing, and other background network tasks on your robot's computer.
*   **D-Day Setup Errors**: Use **startup scripts** or `systemd` services to automate your network configuration and ROS environment setup.

## Video Tutorial

The following video tutorial walks through these configuration steps in detail.

[Watch the YouTube tutorial!](https://www.youtube.com/watch?v=Qi5NX4jUSMQ)

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/Qi5NX4jUSMQ?si=XtUfmcpkplp6a4EM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
