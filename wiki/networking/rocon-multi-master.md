---
title: ROCON Multi-master Framework
---
ROCON stands for *Robotics in Concert*. It is a multi-master framework provided by ROS. It provides easy solutions for multi-robot/device/tablet platforms.
Multimaster is primarily a means for connecting autonomously capable ROS subsystems. It could of course all be done underneath a single master - but the key point is for subsystems to be autonomously independent. This is important for wirelessly connected robots that can’t always guarantee their connection on the network.

## The Appable Robot
The appable robot is intended to be a complete framework intended to simplify:
- Software Installation
- Launching
- Retasking
- Connectivity (pairing or multimaster modes)
- Writing portable software

It also provides useful means of interacting with the robot over the public interface via two different modes:
1. **Pairing Mode:** 1-1 human-robot configuration and interaction.
2. **Concert Mode:** autonomous control of a robot through a concert solution.

## Rapps
ROS Indigo has introduced the concept of Rapps for setting the multi-master system using ROCON. Rapps are meta packages providing configuration, launching rules and install dependencies. These are essentially launchers with no code.
- The specifications and parameters for rapps can be found at in the [Rapp Specifications](http://cmumrsdproject.wikispaces.com/link)
- [A guide](http://cmumrsdproject.wikispaces.com/Creating+Rapps) to creating rapps and working with them is also available.

## The Gateway Model
The gateway model is the engine of a ROCON Multimaster system. They are used by the Rapp Manager and the concert level components to coordinate the exchange of ROS topics, services and actions between masters. The gateway model is based on the LAN concept where a gateway stands in between your LAN and the rest of the internet, controlling both what communications are allowed through, and what is sent out. Gateways for ROS systems are similar conceptually. They interpose themselves between a ROS system and other ROS systems and are the coordinators of traffic going in and out of a ROS system to remote ROS systems.
A hub acts as a shared key-value store for multiple ROS systems. Further detailed information on how the topics are shared between the multiple masters can be found [here](http://cmumrsdproject.wikispaces.com/The+Gateway+Model).
