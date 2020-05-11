---
title: Design considerations for ROS Architectures
---
# Overview
Over the past few months we have dicussed various ROS architectures, changed it at least two times in a major way and have come to appreciate the importance of designing a good architecture. We have also realized there are several important factors which need to be considered before finalizing a particular architecture. In this post we will detail the design decisions we had to make in the context of our project.

## Background of our application: Use Case of ROS in Driving simulator
To give a brief background of the project we are working on, we are learning specific driving behaviors in simulation. We have several nodes performing various tasks. The nodes are:
- **State extraction node**:  Receives the updated trajectories, updates the simulated state, and for each time step, publishes the required information about the state
- **RL agent node**: Based on the extracted state, the RL agent publishes high level decisions (go straight, turn, merge, stop). 
- **Path planning node:** Based on the current location and high level decisions from CARLA, the path planner node publishes the updated trajectory 

All of this runs in a loop for the simulation to continuously render and reset.


# Design Considerations
Some of the important considerations before designing your ROS architecture are: 
- What is the message drop out tolerance for our system?
- What is the overall latency our system can have?
- Do we need an synchronous form of communication or asynchronous form of communication?
- Which nodes need to necessarily be separated?

### Message Drop Out Tolerance

ROS is a far from ideal system. Although it uses the TCP (Transmission Control Protocol), due to various reasons (e.g. the nodes being too computationally intensive, delays, buffer overflows etc) there is an expected message drop out. This results in some of the messages not being received by the subscriber nodes. Although there is no quantifiable way to estimate how much drop out can occur, it is important to think about this for some critical nodes in your system. A couple of ways to address this issues is: 
- try a bigger queue size
- try implementing smarter logic in your nodes to check for drop outs and handle them properly

### Latency
Depending on your system, latency can cause many issues, ranging from annoying (simulation slowdown) to failures while interacting with dynamic objects in the real world. Some of the reasons for latency can be:
- Large messages
- Nodes running on different systems and communicating over LAN/WLAN
- TCP handshake delays

Some of the ways to address the problems are:

- Large messages naturally take more time to transmit and receive. A lot of delay can be removed by making your messages concise and ensuring only those nodes which need it, subscribe to them.
- Try as not to use WLAN connections for ROS communication, as usually it has the highest latency. Always prefer wired LAN over wireless LAN, and nothing is better than running the nodes on a same system for reducing latency in communication.
- Also weigh the delay of running heavy processes on a weak computer vs delay of communication while running these processes on different systems. This requires analysis of the trade off.
- Another useful trick is to switch on TCP_NODELAY transport hint for your publishers. This reduces latency of publishing.

### Synchronous vs Asynchronous Communication
ROS offers both, a synchronous mode of communication using **services** and, an asynchronous mode of communication with the **publisher/subscriber** system. It is very important to choose the correct communication in your system needs to be synchronous and which does not. Making this decision will save a lot of design effort in the future when you are implementing the logic of these nodes.

Asynchronous communication is more practical and useful in real-time systems. Synchronous communication is useful if you want to enforce a step-by-step execution of some protocol in your system. 

### Separation of tasks
In general, separation of code into modular components is recommended for clarity and the cleanliness of our development process. This rule of thumb is not the best idea when applied to ROS nodes. If nodes, which can be merged are split, for the sake of keeping code modular and clean, we simultaneously pay a price in communication delays. It is better to merge the nodes and keep them modular by maintaining classes and importing them into one node. Separate the tasks only if you feel those tasks can be performed faster and concurrently if it was given another core of the CPU.
