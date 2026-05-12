---
date: 2026-04-30
title: Controller Area Network (CAN)
---

Controller Area Network (CAN) is a communication protocol used for reliable, real-time data exchange between multiple devices in embedded and robotic systems. It allows microcontrollers, sensors, actuators, motor drivers, and power electronics to communicate over a shared two-wire bus without requiring a central controller.

CAN is widely used in robotics because it is robust, fault-tolerant, deterministic, and resistant to electrical noise. These properties make it especially useful in mobile robots, manipulators, autonomous vehicles, drones, and industrial automation systems where many distributed components must communicate reliably.

In robotics, CAN is commonly used for motor control, sensor integration, battery management, actuator feedback, and distributed embedded control. For example, a mobile robot may use CAN to communicate with wheel motor controllers, encoders, an IMU, and a battery management system over the same bus. Instead of running separate signal wires for every device, CAN allows all devices to share a common communication medium.

This article provides an overview of CAN, explains key hardware and software concepts, and gives practical setup and debugging guidance for robotic systems.

## Overview and Key Concepts

CAN is a multi-master, message-based communication protocol. Multi-master means that any node on the bus can transmit when the bus is free. Message-based means that communication is organized around message identifiers rather than direct device addresses.

Unlike protocols such as I2C or SPI, CAN does not require one central controller to manage communication. Each node listens to the bus and decides whether a message is relevant based on its identifier.

Key characteristics of CAN include:

- Message-based communication instead of device addressing
- Multi-master bus access
- Arbitration-based priority system
- Lower message ID means higher priority
- Built-in error detection and retransmission
- Differential signaling using CAN_H and CAN_L
- Strong resistance to electrical noise
- Support for distributed embedded systems

Each CAN message typically contains:

- Identifier: defines message priority and meaning
- Data payload: contains the actual transmitted data
- Data length code: specifies payload length
- CRC and error-checking fields
- Acknowledgment field

In classic CAN, the payload can contain up to 8 bytes of data. In CAN FD, which is a newer extension, the payload can be larger and the data phase can run at a higher bitrate.

## Arbitration and Message Priority

One important feature of CAN is its deterministic arbitration mechanism. When multiple nodes attempt to transmit at the same time, CAN automatically decides which message gets priority based on the message identifier.

A lower numerical CAN ID has higher priority. For example:

```text
CAN ID 0x100 has higher priority than CAN ID 0x300
```

During arbitration, nodes monitor the bus while transmitting. If a node detects that another node is transmitting a higher-priority message, it stops transmitting and waits until the bus becomes free again. This prevents data corruption and allows high-priority messages, such as emergency stop commands or motor safety messages, to be delivered first.

This behavior is very useful in robotics. For example, a motor fault message or emergency stop command can be assigned a low CAN ID so that it always has higher priority than lower-criticality telemetry messages.

## CAN Physical Layer

CAN typically uses two differential signal lines:

```text
CAN_H
CAN_L
```

The voltage difference between these two lines represents the logical state of the bus. Because CAN uses differential signaling, it is more resistant to common-mode electrical noise than single-ended communication protocols.

This is particularly useful in robotics because motors, power electronics, switching regulators, and long cable runs can introduce significant electromagnetic interference.

A typical CAN bus requires:

- A CAN controller
- A CAN transceiver
- Twisted pair wiring for CAN_H and CAN_L
- Two 120 ohm termination resistors
- Common ground reference between nodes

The CAN controller handles message framing, arbitration, and error checking. The CAN transceiver converts controller-level logic signals into differential CAN bus signals.

Many microcontrollers include an internal CAN controller, but an external CAN transceiver is still required to connect to the physical bus.

## Hardware Requirements

Typical hardware components include:

- CAN controller, often integrated into a microcontroller
- CAN transceiver, such as MCP2551, SN65HVD230, or TJA1050
- Twisted pair wiring for CAN_H and CAN_L
- Two 120 ohm termination resistors at both physical ends of the bus
- Proper grounding between devices
- Optional USB-to-CAN adapter for debugging from a computer

Common USB-to-CAN adapters include:

- CANable
- Kvaser CAN adapters
- PEAK-System PCAN adapters
- ValueCAN
- Seeed USB-CAN adapters

For Linux-based robotics systems, USB-to-CAN adapters are often used together with SocketCAN, which exposes CAN interfaces like regular Linux network interfaces.

## CAN Bus Topology

CAN should generally be wired as a linear bus.

Recommended topology:

```text
Node 1 ---- Node 2 ---- Node 3 ---- Node 4
 |                              |
120Ω                          120Ω
```

The two termination resistors should be placed at the physical ends of the bus, not at every node.

Important topology guidelines:

- Use a linear bus configuration
- Avoid star topology
- Keep stub lengths short
- Use twisted pair wiring
- Place one 120 ohm resistor at each end of the bus
- Do not add termination at every device
- Keep cable length appropriate for the selected bitrate

A common mistake is adding termination resistors at every node. This lowers the effective bus resistance too much and can prevent communication.

With the system powered off, the resistance between CAN_H and CAN_L should usually measure around 60 ohms if two 120 ohm termination resistors are installed in parallel.

## Bitrate Selection

All nodes on a CAN bus must use the same bitrate. If one node is configured with a different bitrate, it will not communicate correctly and may generate bus errors.

Common CAN bitrates include:

```text
125000
250000
500000
1000000
```

A common robotics setup uses 500 kbps or 1 Mbps depending on cable length and reliability requirements.

General tradeoff:

- Higher bitrate gives faster communication
- Lower bitrate allows longer cable length and better noise tolerance

For small robots with short cable runs, 1 Mbps may be acceptable. For larger systems or electrically noisy environments, 250 kbps or 500 kbps may be more reliable.

## Software Setup Using Linux and SocketCAN

SocketCAN is the standard Linux interface for CAN communication. It allows CAN devices to be treated similarly to network interfaces.

### Install CAN Utilities

```bash
sudo apt update
sudo apt install can-utils
```

### Bring Up a CAN Interface

```bash
sudo ip link set can0 up type can bitrate 500000
```

To check whether the interface is active:

```bash
ip link show can0
```

To bring the interface down:

```bash
sudo ip link set can0 down
```

To bring it back up with a different bitrate:

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

## Sending and Receiving CAN Messages

### Send a CAN Message

```bash
cansend can0 123#DEADBEEF
```

In this example:

```text
123       = CAN identifier
DEADBEEF  = data payload in hexadecimal
```

### Receive CAN Messages

```bash
candump can0
```

This prints all received CAN frames.

Example output:

```text
can0  123   [4]  DE AD BE EF
can0  201   [8]  01 02 03 04 05 06 07 08
```

### Monitor Changing Signals

```bash
cansniffer can0
```

`cansniffer` is useful when reverse engineering or identifying which bytes in a CAN message are changing over time.

## Useful SocketCAN Tools

Common tools from the `can-utils` package include:

```bash
candump
cansend
cansniffer
cangen
canplayer
isotpdump
isotpsend
isotprecv
```

Useful commands:

Monitor bus traffic:

```bash
candump can0
```

Generate random CAN traffic for testing:

```bash
cangen can0
```

Replay recorded CAN logs:

```bash
canplayer -I log_file.log
```

Record CAN traffic:

```bash
candump -l can0
```

Check CAN interface statistics:

```bash
ip -details -statistics link show can0
```

These tools are very useful when debugging communication between a Linux computer, motor controller, and embedded microcontroller.

## Example Use in a Robotics System

Consider a differential-drive mobile robot with two wheel motor controllers, an IMU, and a battery management system.

A possible CAN message layout could be:

```text
0x100 - Emergency stop command
0x110 - Left motor velocity command
0x111 - Right motor velocity command
0x200 - Left motor encoder feedback
0x201 - Right motor encoder feedback
0x300 - IMU orientation data
0x400 - Battery voltage and current
0x500 - Motor fault status
```

In this setup:

- The robot computer sends velocity commands to the motor controllers.
- The motor controllers send encoder feedback.
- The IMU sends orientation data.
- The battery management system reports voltage, current, and fault state.
- Emergency stop messages are assigned a high-priority low ID.

This allows the robot to maintain reliable communication between distributed components while keeping wiring simple.

## Usage in Robotics Systems

CAN is widely used in robotic systems that require reliable communication between distributed components.

Common robotics applications include:

- Motor controllers such as ODrive, Roboteq, and VESC-based systems
- Servo drives and smart actuators
- Wheel encoder feedback
- IMUs and inertial sensing
- Battery management systems
- Power distribution boards
- Robotic arms with distributed joint controllers
- Autonomous ground vehicles
- Industrial mobile robots
- Drones and UAV subsystems

CAN is useful when a robot has multiple embedded devices spread across the platform. Instead of using separate serial or PWM lines for every subsystem, CAN allows many devices to share a single bus.

This reduces wiring complexity, improves reliability, and makes the system easier to expand.

## CAN Compared with Other Protocols

CAN is often compared with UART, I2C, SPI, and Ethernet.

### CAN vs UART

UART is simple and useful for point-to-point communication. However, it does not naturally support multiple devices on the same bus and does not include the same level of arbitration or error handling as CAN.

CAN is usually better when multiple embedded devices need to communicate reliably.

### CAN vs I2C

I2C is useful for short-distance board-level communication, such as connecting sensors to a microcontroller. However, I2C is not ideal for long cables or electrically noisy robot environments.

CAN is better for distributed systems with longer wiring and higher noise immunity.

### CAN vs SPI

SPI is fast and simple but usually requires more wiring and is mostly used for short-distance communication between chips on the same board.

CAN is better for communication between modules across a robot.

### CAN vs Ethernet

Ethernet provides much higher bandwidth than CAN and is useful for cameras, LiDARs, and high-data-rate sensors. However, CAN is simpler, deterministic, and often more suitable for low-level control messages.

A robot may use both Ethernet and CAN. For example:

- Ethernet for cameras and LiDAR
- CAN for motor control, encoders, and power system feedback

## CANopen and Higher-Level Protocols

CAN defines the lower-level communication mechanism, but it does not define a universal meaning for every message. Higher-level protocols are often built on top of CAN to standardize device behavior.

One common higher-level protocol is CANopen.

CANopen provides standardized communication objects for:

- Device configuration
- Motor control
- Sensor data
- Error reporting
- Network management

CANopen is common in industrial automation and some robotics systems. It can be useful when working with commercial motor drives or industrial controllers.

However, many robotics teams also define their own custom CAN message IDs and payload formats for simpler systems.

## DBC Files and Signal Decoding

A DBC file is a database file that describes how raw CAN frames should be interpreted. Instead of manually remembering that byte 0 and byte 1 represent motor velocity, a DBC file defines the message ID, signal name, bit position, scaling factor, offset, unit, and valid range.

For example, a DBC file can describe signals such as:

- Wheel velocity
- Motor current
- Battery voltage
- Battery temperature
- Fault flags
- Encoder position
- State of charge

DBC files are helpful because they make CAN systems easier to debug, log, visualize, and share across a team. In a robotics project, keeping a DBC file or a shared CAN message table helps prevent confusion during integration.

## Common Issues and Debugging

> Missing termination resistors, incorrect bitrate settings, and wiring mistakes are common causes of CAN communication failure.

Common issues include:

- Missing termination resistors
- Too many termination resistors
- Bitrate mismatch between nodes
- CAN_H and CAN_L swapped
- No common ground between nodes
- Loose connectors
- Electrical noise from motors or power electronics
- Bus overload due to too many high-frequency messages
- Incorrect message ID or payload interpretation
- Faulty CAN transceiver
- Interface not brought up correctly in Linux

## Debugging Checklist

A practical debugging workflow is:

1. Check physical wiring.

Verify that CAN_H connects to CAN_H and CAN_L connects to CAN_L across all devices.

2. Check termination.

With power off, measure resistance between CAN_H and CAN_L. A correctly terminated bus usually measures close to 60 ohms.

3. Check bitrate.

Make sure every node uses the same bitrate.

4. Bring up the Linux CAN interface.

```bash
sudo ip link set can0 up type can bitrate 500000
```

5. Monitor traffic.

```bash
candump can0
```

6. Send a test frame.

```bash
cansend can0 123#DEADBEEF
```

7. Check for bus errors.

```bash
ip -details -statistics link show can0
```

8. Use an oscilloscope if needed.

Check whether CAN_H and CAN_L show differential signaling when messages are transmitted.

9. Reduce the system.

If the full bus does not work, test only two nodes at a time.

10. Check grounding and shielding.

Poor grounding can cause intermittent communication failures, especially near motors.

## Practical Tips for Robotics Teams

Useful practices when designing CAN-based robot systems:

- Assign lower CAN IDs to safety-critical messages.
- Document every message ID and payload format.
- Keep message rates reasonable to avoid bus overload.
- Use twisted pair wiring.
- Keep CAN wiring away from high-current motor wires when possible.
- Use proper connectors for mobile robots.
- Add strain relief for cables.
- Use logging tools during integration testing.
- Test one device at a time before connecting the full bus.
- Keep a shared CAN message spreadsheet or DBC file.
- Use consistent units such as radians per second, meters per second, volts, amps, and degrees Celsius.
- Add versioning to message definitions when the interface changes.
- Log CAN traffic during tests so failures can be reproduced later.

## Example CAN Message Documentation Table

| CAN ID | Message Name | Sender | Receiver | Rate | Payload |
|---|---|---|---|---|---|
| 0x100 | Emergency Stop | Main Computer | All Nodes | On Change | Stop flag |
| 0x110 | Left Wheel Command | Main Computer | Left Motor Controller | 50 Hz | Target velocity |
| 0x111 | Right Wheel Command | Main Computer | Right Motor Controller | 50 Hz | Target velocity |
| 0x200 | Left Encoder Feedback | Left Motor Controller | Main Computer | 100 Hz | Position, velocity |
| 0x201 | Right Encoder Feedback | Right Motor Controller | Main Computer | 100 Hz | Position, velocity |
| 0x300 | IMU Data | IMU | Main Computer | 100 Hz | Angular velocity, acceleration |
| 0x400 | Battery Status | BMS | Main Computer | 10 Hz | Voltage, current, state of charge |
| 0x500 | Motor Fault Status | Motor Controller | Main Computer | On Change | Fault code |

Maintaining this type of table helps with debugging, integration, and onboarding new team members.

## Summary

CAN is a reliable and widely used communication protocol for robotics systems that require real-time, fault-tolerant communication between distributed components. Its message-based design, deterministic arbitration, built-in error handling, and resistance to electrical noise make it especially useful for motor control, sensor feedback, actuator communication, and battery monitoring.

For successful deployment, teams should pay close attention to physical wiring, termination, bitrate configuration, message documentation, and debugging tools. In robotics projects, CAN is most effective when combined with clear message definitions, good electrical design, and systematic testing.

## See Also

- [ROS2 QoS](/wiki/networking/ros2-qos/)
- [Actuation](/wiki/actuation/)
- [Embedded Systems](/wiki/computing/)

## Further Reading

- [Bosch CAN Specification](https://www.bosch-semiconductors.com/media/ip_modules/pdf_2/canliteratur/can2spec.pdf)  
  This is the original CAN 2.0 specification and is useful for understanding the protocol format, arbitration mechanism, frame structure, error handling, and data link layer behavior.

- [ISO 11898 Standard Overview](https://www.iso.org/standard/63648.html)  
  ISO 11898 is the formal international standard for CAN communication. It is useful for understanding the standardized physical and data link layer requirements.

- [Linux Kernel SocketCAN Documentation](https://www.kernel.org/doc/html/latest/networking/can.html)  
  This is the main Linux documentation for SocketCAN. It explains how CAN devices are represented as network interfaces and how CAN sockets work in Linux.

- [Linux can-utils Repository](https://github.com/linux-can/can-utils)  
  This repository contains practical command-line tools such as `candump`, `cansend`, `cangen`, `canplayer`, and `cansniffer`. These tools are very useful for testing and debugging CAN buses in robotics projects.

- [SocketCAN Wikipedia Overview](https://en.wikipedia.org/wiki/SocketCAN)  
  This provides a quick conceptual overview of SocketCAN and is useful for beginners who want a high-level explanation before reading the Linux kernel documentation.

- [CANopen Overview by CAN in Automation](https://www.can-cia.org/can-knowledge/canopen)  
  CANopen is a higher-level protocol built on top of CAN. This link is useful for understanding standardized device profiles, network management, and industrial motor control applications.

- [CAN in Automation Knowledge Base](https://www.can-cia.org/can-knowledge)  
  This is a broader CAN knowledge resource that explains CAN, CAN FD, CANopen, and related standards. It is useful for deeper learning beyond basic robotics usage.

- [Kvaser CAN Protocol Tutorial](https://www.kvaser.com/can-protocol-tutorial/)  
  Kvaser provides a beginner-friendly explanation of CAN frames, arbitration, error detection, and bus behavior. This is helpful for students learning CAN for the first time.

- [Kvaser CAN Bus Troubleshooting Guide](https://www.kvaser.com/developer-blog/technical-associate-qa-can-bus-troubleshooting/)  
  This guide is useful when diagnosing real CAN hardware problems such as missing termination, wiring errors, bus-off states, and bitrate mismatches.

- [PEAK-System PCAN Documentation](https://www.peak-system.com/PCAN-USB.199.0.html?&L=1)  
  PEAK-System adapters are commonly used USB-to-CAN tools. This page is useful for understanding professional CAN adapter hardware used for testing and debugging.

- [CANable USB-to-CAN Adapter](https://canable.io/)  
  CANable is a low-cost USB-to-CAN adapter commonly used in student robotics projects. It is helpful for connecting a laptop to a CAN bus through SocketCAN.

- [ODrive CAN Protocol Documentation](https://docs.odriverobotics.com/v/latest/manual/can-protocol.html)  
  ODrive motor controllers are commonly used in robotics. This documentation is useful as a real example of how a motor controller exposes commands, feedback, and errors over CAN.

- [VESC CAN Bus Documentation](https://vesc-project.com/node/179)  
  VESC motor controllers are used in mobile robots, electric skateboards, and research platforms. This is useful for seeing how CAN is applied in motor control systems.

- [Roboteq CANopen Documentation](https://roboteq.freshdesk.com/support/solutions/articles/70000651430-canopen-introduction)  
  Roboteq motor controllers are used in mobile robots and AGVs. This page is useful for understanding how commercial motor controllers use CANopen for control and feedback.

- [ROS socketcan_bridge Package](https://github.com/ros-industrial/ros_canopen/tree/melodic-devel/socketcan_bridge)  
  This ROS package connects SocketCAN interfaces to ROS topics. It is useful when integrating CAN devices into a ROS-based robotics software stack.

- [ROS2 socketcan Package](https://github.com/autowarefoundation/ros2_socketcan)  
  This package provides SocketCAN support for ROS 2 systems. It is useful for robots that need to send and receive CAN frames inside a ROS 2 architecture.

- [cantools Python Library](https://github.com/cantools/cantools)  
  `cantools` is a Python library for decoding and encoding CAN messages using DBC files. It is useful for logging, testing, and analyzing CAN traffic in robotics experiments.

- [python-can Library](https://python-can.readthedocs.io/en/stable/)  
  `python-can` provides a Python interface for sending and receiving CAN messages. It is useful for writing scripts to test devices, log data, or prototype CAN communication.

- [Vector DBC File Introduction](https://www.csselectronics.com/pages/can-dbc-file-database-intro)  
  This article explains DBC files and how raw CAN bytes are converted into meaningful physical signals. It is useful for teams documenting complex CAN networks.

- [CSS Electronics CAN Bus Explained](https://www.csselectronics.com/pages/can-bus-simple-intro-tutorial)  
  This is a practical beginner-friendly CAN tutorial with diagrams and examples. It is useful for quickly understanding CAN frames, wiring, termination, and real-world applications.

- [CSS Electronics CAN FD Explained](https://www.csselectronics.com/pages/can-fd-flexible-data-rate-intro)  
  CAN FD is an extension of classic CAN with larger payloads and faster data phases. This is useful if a robotics system requires more bandwidth than classic CAN.

- [Texas Instruments CAN Transceiver Basics](https://www.ti.com/lit/an/slla337/slla337.pdf)  
  This application note explains CAN transceiver behavior and physical-layer design. It is helpful for understanding the electrical side of CAN communication.

- [Microchip MCP2551 Datasheet](https://ww1.microchip.com/downloads/aemDocuments/documents/APID/ProductDocuments/DataSheets/MCP2551-High-Speed-CAN-Transceiver-Data-Sheet-20001667G.pdf)  
  The MCP2551 is a common CAN transceiver. Its datasheet is useful when designing embedded boards or understanding how microcontrollers connect to a CAN bus.

- [Texas Instruments SN65HVD230 Datasheet](https://www.ti.com/lit/ds/symlink/sn65hvd230.pdf)  
  The SN65HVD230 is a commonly used 3.3 V CAN transceiver. This datasheet is useful for embedded robotics systems that use 3.3 V microcontrollers.

- [NVIDIA DriveWorks CAN Bus Documentation](https://developer.nvidia.com/docs/drive/driveworks/archives/6.0.4/nvsdk_dw_html/group__can__group.html)  
  This documentation shows how CAN is used in autonomous vehicle software stacks. It is useful as an example of CAN integration in advanced robotic and automotive systems.

## References

- Bosch CAN Specification
- ISO 11898 Standard
- Linux SocketCAN Documentation
- Linux can-utils
- CANopen Standard
- Kvaser CAN Protocol Tutorial
- CAN in Automation Knowledge Base
- ODrive CAN Protocol Documentation
- ROS2 SocketCAN Package
- python-can Documentation
- cantools Documentation
