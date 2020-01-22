---
title: Controlling UR5e arm using ROS
---

Industrial cobots such as UR5 arms have found an increasing interest in academia research nowadays. In order to perfrom complex actuation tasks using a computer, it becomes imperative to know how to control these arms using software. Since ROS is a widely used framework for robotics applications; in this tutorial we will go through the steps needed to be performed in order to control the arm using ROS.

## Prerequisites

In order to begin this tutorial, please ensure:-

1. A computer with an ethernet port and ROS >= Hydro installed. To install ROS you can follow the instructions on this page: http://wiki.ros.org/ROS/Installation

2. ROS-industrial universal\_package installed, which can be installed with the following command `sudo apt-get install ros-<hydro/kinetic/melodic>-universal-robots`
   
   Once we satisfy the above pre-requisites we are now ready to connect to the UR5e arm.
   
   > **_Please Note:_** The tutorial has been verified on a UR5e arm but it  still works for UR5 arm series. We will let you know the changes to be made if you are using a different arm as we move along.

## Steps

The universal_robot metapackage communicates with hardware via Ethernet connection. Upon establishing a connection, ROS-Industrial will upload a program written in *URScript*, Universal Robots' own Python-like scripting language. This program is responsible for listening for messages sent via ROS-Industrial's simple_messages package and interpreting those messages into hardware commands.

> ***Note:*** Please ensure that the UR arm is connected to your computer using a ethernet connection before proceeding ahead.

#### Configure the arm:

The UR arms come with a pendant which is used to set-up the arm for manual/auto control. In order to control the arm we need the IP address of this robot. To enable networking, use the UR’s teach-pendant to navigate to the Setup Robot -> Setup Network Menu which will show you the robots IP address. Please make a note of this IP address. It becomes increasingly annoying if the IP address of the robot changes after each boot-up; hence for robotics application it is recommended that you select the Static IP option on the screen and set the IP to whatever you desire and click Apply. This will commit the changes in the firmware and ensure that whenever the arm is connected to your computer, it will have the same IP address and hence you need-not change the IP address in your code/launch files. To check if your connection with the arm is successful please execute the following command `ping <IP of arm>`

More information on debugging can be found [here](http://wiki.ros.org/universal_robot/Tutorials/Getting%20Started%20with%20a%20Universal%20Robot%20and%20ROS-Industrial)

The below image shows a sample static IP setting which can be used.

![Ur5 static IP page](assets/UR5e_static_ip.JPG)

#### Describing the Arm to ROS:

`roslaunch ur_description ur5e_upload.launch`

This will send a xacro file to the parameter server so that a description of the arm can be read in by the driver at runtime. Please replace ur5e with your appropriate model such as ur5, etc

#### Switching the UR arm on

Press the Power off button at the bottom left and you will be shown a layout as shown in the below figure. Select the ON button to turn the robot arm on and then you will have to press the same button again when prompted to release the breaks and actually control the motors.

![UR5e bootup page](assets/UR5e_bootup.JPG)

#### 

#### Connecting to the arm

The first step is to find out the firmware version being used on the UR arm. To check the version, go to settings menu on the pendant and select the Info menu. The next step is to put the robot into remote control mode which can be done as shown in the images below.

1. Select the menu icon on rop right and then it should show you a menu layout as shown in the below picture.  ![UR5 about help page](assets/UR5e_about_button.jpg)

2. Once you click the About option from the menu, you can see the software version as shown in the below picture. ![UR5e Firmware page](assets/UR5e_about_page.jpg)

If you have a UR arm with firmware version below 3.0 then in order to connect to the arm execute the following command. `roslaunch ur_bringup ur5e_bringup.launch robot_ip:=<IP_OF_THE_ROBOT>` To use any other arm, please replace the prefix ur5e with the appropriate model. If you are using a firmware > 3.0 then run this command `roslaunch ur_modern_driver ur5e_bringup.launch robot_ip:=<IP_OF_THE_ROBOT>`. If you are getting an error such as `ur5e_bringup.launch not found` then you need an active repo of the ur\_modern\_driver which can be found at [this link](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/tree/master/ur_robot_driver). Please clone this repo and build this package.

If you don't get any error messages then you are good to go to the next step. If you get any errors then some troubleshooting like:- pinging the robot to ensure that the IP is correct, checking if the parameters uploaded to the server using `rosparam list` are proper. To check if your connection is successful you would be able to see a output on the console like: ```Trajectory server started```

#### Command joint angles to the arm

Probably the simplest automation we can do is to control the joint angles of the arm, this can be done using the script below where we have a node which subscribes to the target angles and sends the appropriate command to the UR arm.

> Before we start: please ensure that the workspace of the arm is clear and always keep the emergency stop button of the arm handy enough to prevent any damage to anyone or the arm itself.

```
#!/usr/bin/env python
import roslib; roslib.load_manifest('ur_driver')
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from std_msgs.msg import String

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
client = None

def process(msg):
    global curr_angle, count;
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    joint_angles = msg.data[1:-1].split(",")
    joint_angles = [float(i) for i in joint_angles]

    g.trajectory.points = [
        JointTrajectoryPoint(positions=joint_angles, velocities=[0]*6, time_from_start=rospy.Duration(1.0))]
    client.send_goal(g)
    try:
        client.wait_for_result()
    except KeyboardInterrupt:
        client.cancel_goal()
        raise

def main():
    global client
    try:
        rospy.init_node("test_move", anonymous=True, disable_signals=True)
        client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
        print "Waiting for server..."
        client.wait_for_server()
        print "Connected to server"

        subscriber =rospy.Subscriber("target_angles",String, process,queue_size=1)
        rate = rospy.Rate(0.5)

        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise

if __name__ == '__main__': main()
```

And in a terminal window you can just type:

```
rostopic pub target_angles std_msgs/String "[1.57, -2.26893, 2.26893, -1.57, -1.57, -1.57]"
```

The above code and script will move the UR5e arm into the goal position.

## Summary

So now you have a new way to control your arm using ROS. To run the arm autonomously ensure that you follow the below order of steps.

1. Turn on the arm (with breaks released) and put it in remote control mode

2. Connect to the arm using the ur\_driver or ur\_modern\_driver (based on the firmware version)

3. Run your code and have fun!
   
   Always ensure that the workspace is **clear** and have **e-stop handy!**  **Safety first!**

## Further Reading

- You can also perform more complicated operations on the UR5 arm, refer to MoveIt package for more exiciting use-cases https://moveit.ros.org
- Also you can use URScript to control the arm, Universal Robots has detailed documentation on using URScript which can be found [here](https://s3-eu-west-1.amazonaws.com/ur-support-site/18679/scriptmanual_en.pdf)

## References

- http://wiki.ros.org/universal_robot/Tutorials/Getting%20Started%20with%20a%20Universal%20Robot%20and%20ROS-Industrial
