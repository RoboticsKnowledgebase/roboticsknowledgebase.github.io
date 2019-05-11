---
title: ROS Motion Server Framework
---
This tutorial will help you in setting up the ROS motion server framework developed at the Robotics and Automation Society at the University of Pittsburgh. This framework is built around the ROS Simple Action Server. For better understanding, we have also included an example of a high-level task running on Pixhawk with ROS motion server framework.

# Table of Contents
1. [Introduction: Modes of Communication in ROS](#Introduction)
2. [ROS Motion Server Framework](#ROS-Motion-Server-Framework)
3. [Example](#Example)

## Introduction
There are three different modes of communication in ROS: topics, services & actions. Topics and services are the most popular modes of communication in ROS. This section explains when ROS actions are better over topics & services. <br />
ROS topics are used when the flow of data stream is continuous like robot pose, sensor data, etc. Topics implement a publish/subscribe communication mechanism, one of the most common ways to exchange data in a distributed system. So, this type of communication is many to many connections. <br />

On the other hand, ROS services are used when synchronous remote procedure calls can terminate quickly. So, this mechanism is best suited when the remote function needs to be called infrequently and the remote function takes a limited amount of time to complete (like for querying the state of a node or turning the robot on & off). <br />

Now coming to the ROS actions. They are used when you send a request to a node to perform a task which takes a long time to execute and you also expects a response for that request.  This sounds very similar to ROS services but there is a major difference between services and actions. Services are synchronous and actions are asynchronous. Due to the asynchronous mechanism, ROS actions can be preempted and preemption is always implemented by ROS action servers. Similar to services, an action uses a target to initiate a task and sends a result when the task is completed. But the action also uses feedback to provide updates on the task’s progress toward the target and also allows for a task to be canceled or preempted. <br />
 
For example - your task is to send a robot to an x,y position. You send a target waypoint, then move on to other running tasks while the robot is reaching towards the waypoint. Till your robot reaches the target, you will periodically receive the ROS messages (like current x,y position of the robot, the time elapsed, etc). Based on this information, you can check the status of that task and if it is not getting executed properly you can terminate it in between or if something more important comes up (like stop/brake command as some obstacle is detected), you can cancel this task and preempt the new task.<br />

ROS actions architecture is explained in more detail [here](http://wiki.ros.org/actionlib). Going through few [ROS tutorials on action server](http://wiki.ros.org/actionlib_tutorials/Tutorials/Writing%20a%20Simple%20Action%20Server%20using%20the%20Execute%20Callback%20%28Python%29) will help you in understanding upcoming section better.
 
## ROS Motion Server Framework
ROS motion server framework makes writing new task simpler and easy. In this framework, client sends request via ROS Simple Action Client where `action_type` defines which task to run. There is a task manager which manages lifecycle of the task by interfacing with the user defined task config library to check and construct tasks.  It also manages cancelation, abortion, failure, etc. and reports task ending state back to the Action Client.<br />
Now we will discuss task config in detail. User needs to define a task handler with three key methods:
- wait_until_ready : blocks for a specified timeout until all dependencies are ready, or throws an exception if dependencies fail within timeout.<br />
- check_request : checks that the incoming request is valid; if so, returns an instance of the requested task. <br />
- handle : handles the provided (by task) state and command. <br />

Additional implementation details can be found from [here](https://github.com/asaba96/robot_motions_server_ros).

## Example
To setup ROS motion server framework, clone below two repository:
```
git clone https://github.com/shubhamgarg1994/robot_motions_server_ros 
git clone https://github.com/shubhamgarg1994/example_motion_task_config
```
All the required changes to add a new task will be done inside example motion task config folder. Now, we will explain how a new task can be added in this framework.<br /> Here, we are adding a new task “takeoff” command for PX4 flight controller.<br />
High level state machine is written inside the `test_action_client file.py`. Here, we will running a high level command “uavTakeOff” by calling SimpleActionClient Server. Below code snippet will request a new task named `uavTakeOff`. <br />
```
def run(server_name):
   client = actionlib.SimpleActionClient(server_name, TaskRequestAction)

   rospy.loginfo('TaskActionClient: Waiting for server')
   client.wait_for_server()
   rospy.loginfo('TaskActionClient: server ready')
   time.sleep(60)
   request = TaskRequestGoal(action_type='uavTakeOff')

```

