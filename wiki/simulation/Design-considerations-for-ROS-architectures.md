# Overview
Over the past few months me and my team have dicussed various ROS architectures, changed it at least two times in a major way and have come to appreciate the importance of designing a good architecture. We have also realised there are several important factos which need to be considered before nailing down a particular architecture. In this post I will detail the design decisions we had to make in the context of our project.

# Background
To give a brief background of the project we are working on, it is a learning to drive in simulation project. We have several nodes performing various tasks like state extraction from simulator, path planning and a RL agent making decisions. All of this runs in a loop for the simulation to continuously render and reset.

# Design Considerations
Some of the questions we asked ourselves and could be useful discussing amongst your team before framing a ROS architecture are-
- What is the message drop out tolerance for our system?
- What is the overall latency our system can have?
- Do we need an synchronous form of communication or asynchronous form of communication?
- Which nodes need to necessarily be seperated?

# Message Drop Out Tolerance

ROS is not a perfect system. Although it uses TCP, due to various reasons like the nodes being too computationally intensive, delays, buffer overflows etc, there is an expected message drop out. i.e., some of the messages will not be received by the subscriber nodes. Although there is no quantifiable way to estimate how much drop out can occur, it is important to think about this for some critical nodes in your system. When such things happen try bigger queue sizes. Also try implementing smarter logics in your nodes to check for drop outs and handle them properly.

# Latency

Some of the reasons for latency can be:
- Large messages
- Nodes are running on different systems and communicating over LAN/WLAN
- TCP handshake delays

Large messages naturally take more time to transmit and receive. A lot delay can be removed by making your messages concise and ensuring only those nodes which need it, subscribe to them.

Try as much as possible, not to use WLAN connections for ROS communication, as usually it has the highest latency. Always prefer wired LAN over wireless LAN, and nothing is better than running the nodes on a same system for reducing latency in communication.

However, you also need to weigh the delay of running heavy processes on a weak computer vs delay of communication but running these processes on different systems. There is a trade of to be made.

Another useful trick is to switch on TCP_NODELAY transport hint for your publishers. This reduces latency of publishing.

# Synchronous vs Asynchronous Communication

In ROS we can implement a synchronous mode of communication using services and asynchronous mode of communication with the publisher/subscriber system. It is very important to think about which communication in your system needs to be synchronous and which does not. Making this decision will save a lot of design effort in the future when you are implementing the logics of these nodes.

Asynchronous communication is more pratical and useful in real-time systems. Sysnchronous communication is useful if you want to enforce a step-by-step execution of some protocol in your system. 

# Seperation of tasks

In general, we are all taught to seperate code as much as we can, to keep our development process clean. This does not directly apply to ROS nodes. If nodes, which can be merged are split, for the sake of keeping code modular and clean, we are simultaneously paying a price for communication delays. It is better to merge the nodes and keep them modular by maintaining classes and importing them into one node. Seperate the tasks only if you feel those tasks can be performed faster and concurrently if it was given another core of the CPU.


