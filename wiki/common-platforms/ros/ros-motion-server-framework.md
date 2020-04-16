---
title: ROS Motion Server Framework
---
This tutorial will help you in setting up the ROS motion server framework developed at the University of Pittsburgh. This framework is built around the ROS Simple Action Server. For better understanding, we have also included an example of a high-level task running on Pixhawk with ROS motion server framework.

# Table of Contents
1. [Introduction: Modes of Communication in ROS](#Introduction)
2. [ROS Motion Server Framework](#ROS-Motion-Server-Framework)
3. [Example](#Example)

## Introduction
There are three different modes of communication in ROS: topics, services & actions. Topics and services are the most popular modes of communication in ROS. This section explains when ROS actions are better over topics & services. <br />
ROS topics are used when the flow of data stream is continuous like robot pose, sensor data, etc. Topics implement a publish/subscribe communication mechanism, one of the most common ways to exchange data in a distributed system. So, this type of communication is many to many connections. <br />

On the other hand, ROS services are used when synchronous remote procedure calls can terminate quickly. So, this mechanism is best suited when the remote function needs to be called infrequently and the remote function takes a limited amount of time to complete (like for querying the state of a node or turning the robot on & off). <br />

Now coming to the ROS actions. They are used when you send a request to a node to perform a task which takes a long time to execute and you also expects a response for that request. This sounds very similar to ROS services but there is a major difference between services and actions. Services are synchronous and actions are asynchronous. Due to the asynchronous mechanism, ROS actions can be preempted and preemption is always implemented by ROS action servers. Similar to services, an action uses a target to initiate a task and sends a result when the task is completed. But the ROS action also uses feedback to provide updates on the task’s progress toward the target and also allows for a task to be canceled or preempted. <br />
 
For example - your task is to send a robot to an x,y position. You send a target waypoint, then move on to other running tasks while the robot is reaching towards the waypoint. Till your robot reaches the target, you will periodically receive the ROS messages (like current x,y position of the robot, the time elapsed, etc). Based on this information, you can check the status of that task and if it is not getting executed properly you can terminate it in between or if something more important comes up (like stop/brake command as some obstacle is detected), you can cancel this task and preempt the new task.<br />

ROS actions architecture is explained in more detail [here](http://wiki.ros.org/actionlib). Going through a few [ROS tutorials on action server](http://wiki.ros.org/actionlib_tutorials/Tutorials/Writing%20a%20Simple%20Action%20Server%20using%20the%20Execute%20Callback%20%28Python%29) will help you in understanding upcoming section better.
 
## ROS Motion Server Framework
ROS motion server framework makes writing new task simpler and easy. In this framework, the client sends a request via ROS Simple Action Client where `action_type` defines which task to run. There is a task manager which manages the lifecycle of the task by interfacing with the user-defined task config library to check and construct tasks.  It also manages cancelation, abortion, failure, etc. and reports task ending state back to the Action Client.<br />
Now we will discuss task config where user defines the task. User needs to define a task handler with three key methods:
- wait_until_ready: blocks for a specified timeout until all dependencies are ready, or throws an exception if dependencies fail within timeout.<br />
- check_request: checks that the incoming request is valid; if so, return an instance of the requested task. <br />
- handle: handle the provided (by task) state and command. <br />

Task config will get more clear when we will look an example in the last section. Additional implementation details can be found from [here](https://github.com/shubhamgarg1994/robot_motions_server_ros).

## Example
To setup ROS motion server framework, clone below two repository in your catkin workspace and build the workspace:
```
git clone https://github.com/shubhamgarg1994/robot_motions_server_ros 
git clone https://github.com/shubhamgarg1994/example_motion_task_config
catkin_make
```
All the required changes to add a new task will be done inside example motion task config folder. Now, we will explain how a new task can be added to this framework. Here, we are adding a new task for the PX4 flight controller. **If you are not familiar with MAVROS, PX4 offboard mode, please look at this** [tutorial](https://akshayk07.weebly.com/offboard-control-of-pixhawk.html). <br />

A high-level state machine is written inside the `test_action_client file.py`. Here, we will be running a high-level command “uavTakeOff” by calling SimpleActionClient Server. Below code snippet will request a new task named `uavTakeOff`. <br />
```
def run(server_name):
   client = actionlib.SimpleActionClient(server_name, TaskRequestAction)

   rospy.loginfo('TaskActionClient: Waiting for server')
   client.wait_for_server()
   rospy.loginfo('TaskActionClient: server ready')
   time.sleep(60)
   request = TaskRequestGoal(action_type='uavTakeOff')

```
Add a new file `highlevelcommands.py` inside the src where we will write the abstract class for the “uavTakeOff” task. This class needs three methods `init`, `get_desired_command` & `cancel`. For simplicity `cancel` method is not implemented here.
In the future, if you want to add a new task, just add a new class of that task with these three methods.
```
import rospy

from task_states import TaskRunning, TaskDone
from task_commands import *

from motions_server.abstract_task import AbstractTask

class uavTakeOff(AbstractTask):
    def __init__(self, request):
        rospy.loginfo('Take off task being constructed')
        self.status = 0
        self.altitude = request.pose.pose.position.z

    def get_desired_command(self):
        if(self.status == 0):
            self.status = 1
            return TaskRunning(), FlightMode(mode='OFFBOARD',arm=1,altitude=self.altitude)
        else:
            return TaskDone(), StringCommand(data='Task done')
            
    def cancel(self):
        return True
```
In the above class, we introduced an additional command of `FlightMode`. All the parameters defined in this class is self explanatory if you are aware of the PX4 offboard mode. This class needs to be defined in the file `task_commands.py`.
```
class FlightMode(object):
    def __init__(self, mode='', arm=0, altitude=0):
        self.mode = mode
        self.arm = arm
        self.altitude = altitude
        self.sp = PositionTarget()
        self.sp.type_mask = int('010111111000', 2)
        # LOCAL_NED
        self.sp.coordinate_frame = 1
```
Task manager is defined in the `task_config.py`. Here we will define the task handler and the required methods `check_request`, `wait_until_ready`, `handle_task_running` which is already explained in the previous section. In the `init` method we will create all the required objects (subscriber/publisher/services). 
```
class CommandHandler(AbstractCommandHandler):
    def __init__(self):

        #initailizing the services and publishers/subscribers
        rospy.wait_for_service('mavros/cmd/arming')
        rospy.wait_for_service('mavros/cmd/takeoff')
        rospy.wait_for_service('mavros/set_mode')

        try:
            self._armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            self._flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s"%e)

        self._sp_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=1)
   
   def check_request(self, request):
        if request.action_type == 'uavTakeOff':
            self._taskname = request.action_type
            return uavTakeOff(request)
   
   def wait_until_ready(self, timeout):
        return
   
   def _handle_task_running(self, command):
        rospy.loginfo("Handle task running: ")
        if(self._taskname == "uavTakeOff"):
            if(command.arm>0 and command.altitude>0):
                
                #Arming the vehicle
                try:
                    self._armService(bool(command.arm))
                except rospy.ServiceException, e:
                    rospy.logerr("Service arming call failed: %s"%e)
                
                #Put the vehicle in offboard mode
                # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect

                k=0
                while k<10:
                    self._sp_pub.publish(command.sp)
                    self._rate.sleep()
                    k = k + 1

                try:
                    self._flightModeService(custom_mode=command.mode)
                except rospy.ServiceException, e:
                    rospy.logger("service set_mode call failed: %s. Offboard Mode could not be set."%e)

                command.sp.position.z = command.altitude

                while(abs(self._local_pos.pose.pose.position.z - command.altitude) > 0.2):
                    self._sp_pub.publish(command.sp)
                    self._rate.sleep()
     
```
If we want to query the current state of the task, you can add that part of code in `task_states.py`. 
You can find above code [here](https://github.com/shubhamgarg1994/ros-motion-server-example) also.
