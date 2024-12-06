---
date: 2024-12-05
title: Networking
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

This section focuses on **networking techniques, tools, and protocols** commonly employed in robotics applications. This section provides practical tools and insights to help you set up robust and efficient networking solutions for robotics systems, from Bluetooth and WiFi to ROS-based multi-robot systems.

## Key Subsections and Highlights

- **[Bluetooth Socket Programming using Python PyBluez](/wiki/networking/bluetooth-sockets/)**
  A comprehensive guide to using PyBluez for Bluetooth socket programming. Covers Bluetooth fundamentals, transport protocols (RFCOMM and L2CAP), and service discovery. Includes step-by-step instructions for setting up Bluetooth connections and error handling, with examples for both Python and Bash scripting.

- **[ROCON Multi-master Framework](/wiki/networking/rocon-multi-master/)**
  Explores ROCON (*Robotics in Concert*), a ROS-based multi-master framework for coordinating robots and devices. Introduces the Appable Robot concept, Rapps for managing multi-master setups, and the gateway model for managing communication between ROS systems.

- **[Running ROS over Multiple Machines](/wiki/networking/ros-distributed/)**
  Explains how to set up a distributed ROS system across multiple computers. Details include configuring network settings, synchronizing clocks, managing namespaces, and addressing potential latency issues. Provides tips for efficient operation and troubleshooting in multi-robot systems.

- **[Setting up WiFi Hotspot at Boot for Linux Devices](/wiki/networking/wifi-hotspot/)**
  A practical guide for creating and configuring WiFi hotspots on Linux-based devices like NVIDIA Jetson and Intel NUC. Covers default setups, editing configuration files, and using the NM Connection Editor for advanced settings. Discusses startup automation for seamless operation during boot.

- **[XBee Pro DigiMesh 900](/wiki/networking/xbee-pro-digimesh-900/)**
  A tutorial for configuring XBee Pro DigiMesh modules to create low-bandwidth mesh networks. Includes instructions for using X-CTU for module setup and a Python-based script for serial communication over a mesh network. Covers advanced configurations like multi-transmit, broadcast radius, and mesh retries.

## Resources

### General Networking
- [Bluetooth Specifications](http://www.bluetooth.org/spec)
- [ROS Multi-Machine Setup Guide](http://wiki.ros.org/ROS/Tutorials/MultipleMachines)
- [Chrony for Time Synchronization](http://www.ntp.org/)

### Protocol-Specific Documentation
- [PyBluez Documentation](https://people.csail.mit.edu/albert/bluez-intro/c33.html)
- [XBee Pro DigiMesh User Guide](http://ftp1.digi.com/support/documentation/90000903_G.pdf)

### Libraries and APIs
- [PySerial Documentation](http://pyserial.readthedocs.org/en/latest/pyserial_api.html)
- [Python Threading API](https://docs.python.org/2/library/threading.html)

### Advanced Topics
- [Comparison of DigiMesh and ZigBee Architectures](http://www.digi.com/pdf/wp_zigbeevsdigimesh.pdf)
- [ROS Network Configuration and Namespaces](http://nootrix.com/2013/08/ros-namespaces/)
