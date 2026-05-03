---
date: 2024-03-01
title: Setup your Network Infrastructure for Robotics Projects
---

This tutorial provides a guide for setting up the networking infrastructure required for robotics projects, particularly those involving remote monitoring and multi-machine communication (e.g., ROS).

## Overview

A robust network setup is critical for robotics projects to ensure reliable communication between the robot's onboard computer and external workstations. This guide covers:

*   **Static IP Configuration**: Ensuring your robot and workstation always have the same address.
*   **Remote Login**: Setting up SSH for headless operation and monitoring.
*   **Network Hardware**: Utilizing a dedicated router and LAN connections for low-latency communication.

## Video Tutorial

The following video tutorial walks through the network configuration steps for projects involving remote login and monitoring.

[Watch the YouTube tutorial!](https://www.youtube.com/watch?v=Qi5NX4jUSMQ)

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/Qi5NX4jUSMQ?si=XtUfmcpkplp6a4EM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## General Recommendations

While specific configurations may vary, consider the following best practices:

1.  **Use a Dedicated Router**: Avoid using shared or public networks (like university guest WiFi) which often have isolation policies that prevent peer-to-peer communication.
2.  **DHCP Reservations**: Instead of manually setting static IPs on every device, use your router's DHCP reservation feature to assign specific IPs to MAC addresses.
3.  **Hostname Resolution**: Use `/etc/hosts` or mDNS (Avahi) to refer to machines by name rather than IP address.
4.  **Wired Over Wireless**: Use Ethernet (LAN) whenever possible for critical control loops to minimize jitter and packet loss.
