---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-12-04 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: How to design a robotic state machine
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

# Motivation

One key aspect of designing complex robotic systems is the effective
implementation of state machines. This article will explore the best
practices and design considerations for designing state machines
tailored for robotic applications. We begin by describing an example
state machine, then talk about linear versus asynchronous state
machines, publishers and subscribers versus services in ROS, and finally
discuss strategies for error handling in state machines. To learn more
about state machines, read these articles:

-   [Finite-state_machine](https://en.wikipedia.org/wiki/Finite-state_machine)

-   [State Machine Basics](https://www.freecodecamp.org/news/state-machines-basics-of-computer-science-d42855debc66/)

# Example 

Let us consider a green pepper harvesting robot which consists of a
manipulator arm and a custom end-effector. The major subsystems are
perception, motion planning, and end-effector. This system is structured
in ROS with the following nodes:

-   Perception: takes an image and outputs 6D pose for point-of-interaction (POI)


-   Kalman filtering: filters the noisy 6D poses of the POI to provide filtered 6D pose

-   Motion planning: generates a plan for manipulator arm and actuates it to the POI

-   End-effector: actuates a gripper and cutter mechanism to harvest a pepper

Two other nodes jointly control these subsystems. A state machine node
handles the definitions of different states and transitions between
them. This node accesses all the information published or made available
by other nodes to decide when we transition from one state to another.
On the other hand, a planner node constantly listens to the current
state to execute actions that need to be performed in that state. This
node may utilize any data that it needs.

-   State machine: defines different states and transitions between them

-   Planner: listens to the current state to execute actions that need to be performed

# Linear vs Asynchronous

The easiest way to structure a state machine is to linearly move forward
from one state to another upon completion of tasks in the current state.
Doing so, however, can limit the system's decision-making abilities and
limit scalability to complex architectures. In the case of the pepper
harvesting robot, a linear state machine will move the arm to multiple
positions one by one, capture images to estimate noisy 6D poses for POI,
then obtain a filtered pose, move to the POI, and harvest the pepper
with the end-effector. All these actions are completed one at a time.
The state machine will wait for one state to complete its actions before
moving on.

A direct consequence of this is an increase in harvesting time with an
increased number of positions the arm is moved to. Furthermore, this
system cannot scale to have a mobile base that transports the arm to a
new pepper plant location by detecting peppers. With a linear state
machine, base motion and pepper detection cannot be performed at the
same time.

In the case of an asynchronous state machine, some nodes are structured
to constantly process their inputs to publish outputs at all times. The
perception node can continuously generate POIs from the input images
published by a camera. This would allow the Kalman filtering node to
keep track of existing POIs and update them at the same frequency as the
perception node. One immediate advantage of this structure is that
images are processed in parallel with arm motion. This would drastically
reduce the overall harvesting time. Similarly, a mobile base can easily
be added to the system now. Another advantage of such a state machine is
that it can make decisions asynchronously. Given that some nodes are
always publishing data and some perform actuation based on the state,
the state machine can utilize current data to quickly update the next
action, instead of waiting for a state to be completed.

# Choosing Communication Paradigm for State Machines in ROS

When implementing a state machine in ROS, a crucial decision is whether
to use ROS publishers and subscribers or ROS services for state
communication. Let\'s explore the pros and cons of each approach.

## Publisher & Subscriber
### Pro
1. Decentralized Communication: Enables asynchronous communication, allowing multiple nodes to be informed of the state independently.
2. Real-time Suitability: Non-blocking communication ensures timely updates for decisions requiring real-time computation.
3. Scalability: Easily scalable by adding more nodes as subscribers due to the decentralized system.

### Con
1. No Guarantee of Delivery: Lack of confirmation on message receipt necessitates manual checks for communication.
2. Synchronization Issues: Asynchronous nature may require additional synchronization mechanisms for timely information retrieval.
3. Limited to Point-to-Point Communication: Restriction to one-to-one communication impedes multi-node awareness of the state.
4. Enforces a Linear State Machine: The point-to-point limitation results in a linear state model.
5. Communication Overhead: Frequent state changes with request-response interactions introduce performance overhead.

## Ros service
### Pro

1. Blocks Communication: Code blocking ensures coordination between nodes during communication.

### Con
1. Limited to Point-to-Point Communication: Similar to publishers and subscribers, ROS services only allow communication between two nodes.
2. Enforces a Linear State Machine: Restricts the state machine to a linear model.
3. Overhead in Communication: Frequent state changes with request-response interactions introduce performance overhead.


As many networking errors can arise in a robotic system, it is usually
better to have a decentralized communication system rather than a
centralized, linear system. Using a rosservice often causes the system
to be blocked and slows down the entire system, and it is challenging to
debug where the blockage is happening. Thanks to the increased
computational speed, although publishers and subscribers do not
guarantee delivery, we generally do not have to worry about whether or
not the message gets delivered to the nodes.

# Frequent errors that occur in state machines

As mentioned, there are two ways of implementing the state machine. The
errors that occur vary depending on the implementation.

## Publishing & Subscribing method:

-   The publisher and subscriber frequency may differ. If the publishing node and subscribing node are running at different frequencies, and the publisher is publishing at 500 Hz and the subscriber is listening at 1hz, there might be some messages (states) that are lost on the listener end.

## Ros service method:

-   The system performance is slow: As Ros services block the code from moving on to the next block until it ensures a successful communication, you can easily face your code slowing down to ensure this successful communication. If this communication occurs across multiple nodes, the problem becomes worse. If you go down this route, ensuring no unnecessary blockers is crucial to system performance.

## How to separate scripts

Assuming you have decided to move forward with the publisher and
subscriber, the question of how you will design the script arises. We
suggest having a state_machine.py that just listens to all the other
node's current status and moves from state to state. In this case, all
the subsystem nodes should also publish their status through publishers.

## Handling Errors in State Machine Design:

Consider whether a singular error state requiring human intervention is
acceptable. Alternatively, separate error-handling states can automate
resolution for certain non-critical errors. Design your states
accordingly.

Here are some example cases we encountered.

1.  When the depth image was sparse and could not get a value on that pixel, the system would go into an error state. Instead, you could handle this by getting a new depth image until there is a non-null value

2.  Plt was throwing an error because it was not running in the mainthread. This error can be neglected, and the system can perform without handling this error. Not all errors are crucial to the system and can be ignored.

3.  The end-effector subsystem threw a motor failure error that halted the entire system. You can reset the motors using the motor API instead of intervening when you are in this particular error state.

Some errors make sense for humans to intervene, however, there are many
more cases that make sense to be handled separately through code. When
designing states, especially error states, carefully decide on the
number and nature of error states to optimize system resilience and
efficiency.
