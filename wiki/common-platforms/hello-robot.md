The Stretch RE1 by Hello Robot, is a lightweight and capable mobile manipulator designed to work safely around people in home and office environments. It has 4 degrees of freedom - a telescoping arm which can reach 50cm horizontally, a prismatic lift which can reach 110cm vertically, a differential drive base with a compact footprint of 34x34cm.

The design principle of Hello Robot is making the mobile manipulation robot as simple as possible. To do that, Hello Robot referred to the Roomba robot for its omnidirectional mobile base design, adopted linear joints for simple lift and telescopic arm movement, and obtained enough degrees of freedom by adding pitch, and roll joints to the wrist. All together, the robot becomes a 3-DOF wrist and a 7_DOF robot. The robot’s operational principle is to have two modes that are navigation and manipulation. In the navigation mode, the robot’s telescopic arm retracts, and uses a mobile base as a main actuator. The robot can lower the lift joint and retract the arm joint to lower the robot’s COM and increase stability. In its manipulation mode, the robot uses the mobile base to perform rotations in addition to pure translations of Cartesian motion of the end of the arm. It is also possible to perform curvilinear motion. The Stretch RE1 is mainly designed to perform like a human in an indoor human environment. The configuration of the robot therefore, matches with human dimensions.

This tutorial covers additonal functionality that the documentation of the Hello Robot does not include such as-

(a) The HelloNode Class

(b) The Tool Share Class

## The HelloNode Class
The HelloNode class is defined in "hello_misc.py". This image below shows the location of the file as it would appear when cloning the stretch_ros package off of the github repository into your workspace.

![Fig 1.](https://i.ibb.co/2SM17qS/location.png)

This module has several useful functions along with the class and its methods.  Following is a list of all the functions and methods that can be accessed from the HelloHelpers module:

1. get_wrist_state()
2. get_lift_state()
3. get_left_finger_state()
4. get_p1_to_p2_matrix
5. move_to_pose() - HelloNode Class Method
6. get_robot_floor_pose_xya() - HelloNode Class Method

Each of these functions and how best to use them are explained below in detail.  First we will look at the functions and then the class methods:

### get_wrist_state(joint_states)
The get_wrist_state() function takes in a joint_states object.  The joint states object is essentially an array of joint positions for the arm extension.  The index of a particular joint in the array can be accessed using the key value of that particular joint.  For the wrist extension, these can be names from the list ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3'].  Calling joint_states.name.index(<joint_name>) where "joint_name" is a name from the list returns the index of the joint in that joint array.  The index can then be usedd to access the joint value using the call joint_states.<attribute>[<index>] where attribute can be "position", "velocity" or "effort".  The functions calculates the wrist position based on the individual extensions of each arm and then returns the wrist position values as [wrist_position, wrist_velocity, wrist_effort].
  
### get_lift_state(joint_states)
The get_list_state functions also takes in a joint_states object.  This function then indexs for the "joint_lift" and then returns a list of [lift_position, lift_velocity, lift_effort].  The methods of extracting the values are the same as described earlier.
  
### get_left_finger_state(joint_states)
The get_left_finger_state functions also takes in a joint_states object.  This function then indexs for the "joint_gripper_finger_left" and then returns a list of [left_finger_position, left_finger_velocity, left_finger_effort].  The methods of extracting the values are the same as described earlier.
  
### get_p1_to_p2_matrix(<p1_frame_id>, <p2_frame_id>, tf2_buffer, lookup_time=None, timeout_s=None)
This function while it is a function from the module, requires the ROS node to be running and a tf_buffer object to be passed to it.  It returns a 4x4 affine transform that takes points in the p1 frame to points in the p2 frame.

### move_to_pose(pose)
The move_to_pose is a HelloNode class method that takes in a pose dictionary.  The dictionary has key-value pairs where the keys are joint names and the values are the joint values that are to be commanded.  An example joint-value dictionary that can be sent to the move_to_pose function is "{'joint': 'translate_mobile_base', 'inc': self.square_side}".  One point to be noted is that the main stretch ROS driver has to be shifted into position mode to take the joint values for each joint.  The provided examples controls the base by referring to it's joint name "translate_mobile_base".  This type of control is not possible if the stretch_driver is in manipulation or navigation mode.
  
### get_robot_floor_pose_xya()
The get_robot_floor_pose_xya() is another HelloNode class method that uses the function get_p1_to_p2_matrix to get the robot's base_link position and orientation and projects it onto the floor.  The function returns a list of the robot's x, y position and the orientation angle in radians.
  
**Note:  In order to get ROS running after initializing the class, the main method of the class has to also be called.  This will initialize the ROS node and the tf buffer.  This is essential before the functions that require ROS can be utilized**

## The ToolShare Class
The Hello robot comes with a gripper or a dex wrist which can handle a variety of tasks. However, if users have a custom tool for a specific task, the platform provides the ability to swap the existing tool with the new one. The tutorial below explains how to twitch a custom tool to the robot in detail.

- If the tool has no motors 
The ToolNone interface should be loaded when no tool is attached to the Wrist Yaw joint. To switch to this interface, update the field in your stretch_re1_user_params.yaml  to:
robot:
tool: tool_none

- If the tool has one or more motors:  It is a good idea if dynamixel motors are used for tool actuation. The hello robot already comes with a  dynamixel motor interface which makes controlling the motors easier. These motors can be daisy chained to the current robot structure, making it modular and easy to use. The motors can be daisy chained as shown below.
  ![Fig 2.](https://i.ibb.co/m4qbjBL/ex-106-dual.png)
  
Now to use your tool : 
  
Tool usage with the end_of_arm method can be seen [here](https://docs.hello-robot.com/tool_change_tutorial/). To change the tool without this method - the following steps should be followed. Usually following the end_of_arm method gives you limited capability in terms of using the tool. However, it has advantages like better communication between the stretch driver and the tool. 
  
1. The tool in the user params file has to first be changed to tool_none as shown earlier. After this, the parameters of your tool need to be specified clearly in the user params files. This means specifying its baud rate, motion parameters, torque parameters, range etc. A format for the same can be found by looking at the factory parameters for the gripper or the wrist_yaw joint. These are parameters referring to the motor and  should be done for every motor used.
  
2. The name/ title of the tool refers to the python class made for the tool. For example, here : gripper is a class inheriting from the dynamixel motor class which defines functions like homing, move_to, move_by for the motor. Each motor should have a class like this for separate control. 
  
3. Functions  of the tool 
  
These are some of the functions that have to be included in the tool class. This class inherits from the (DynamixelHelloXL430) class already present on the robot.
  
- Homing:   When using any tool with motors, the tool needs to be homed separately. This allows the robot to calibrate the tool before using it. Homing for motors can be done in a one stop or multi stop process depending on the hardstops present on your tool. One stop homing of motors is easy and effective. Once the first hardstop is hit, it makes that position as the zero of the motor and then reaches its initial calculated position or any position you specify.
- move_to : you can directly command the motor to reach a given position relative to its zero position. The move to command can be given in degrees or radians or ticks. However, this distance should be within the range of the motor ticks. The function looks somewhat like this: 
- move_by. : you can command the mor to move by an incremental distance. This distance can be positive or negative and is given in degrees/ radians/ ticks. The function looks somewhat like: 
- Other functions like startup, init, pose can be inherited from the motor class. These help in smooth functioning of the robot.
  
4. Adding to the URDF 
If you have the design of your custom tool in solidworks or fusion, it is easy to add it to the existing URDF model. 
  
[For solidworks](http://wiki.ros.org/sw_urdf_exporter)
  
[For fusion](https://github.com/syuntoku14/fusion2urdf)

These plugins convery your solidworks/ fusion model to the URDF format. Be careful while naming your joints, joint types and origin. The joints, especially, have to be imported carefully with their constraints and types to the urdf model. The plugins generate mesh files (in STL format), zacro files and a .urdf file. All of these files are needed for the further steps. These files need to be on the stretch robot. After receiving the mesh files, the mesh files need to be copied to the stretch_description folder in stretch_ros. The urdf part for the tool also needs to be added in the actual urdf of the robot.

```  
$ cd ~/<path to folder>
$ cp <folder> ~/catkin_ws/src/stretch_ros/stretch_description/urdf/
$ cp <folder>/meshes/*.STL ~/catkin_ws/src/stretch_ros/stretch_description/meshes/
```
  
Now the tool Xacro for Stretch needs to be edited. This is done by opening ~/catkin_ws/src/stretch_ros/stretch_description/urdf/stretch_description.xacro in an editor, and commenting out the current tool Xacro and including the Xacro for the new tool in the same format. After this, we have to reflect the .urdf and xacro changes in the model.
 
```
$ cd ~/catkin_ws/src/stretch_ros/stretch_description/urdf
$ cp stretch.urdf stretch.urdf.bak
$ rosrun stretch_calibration update_urdf_after_xacro_change.sh
```

Now, we can visualize the tool in RViz. In case the tool is not oriented properly, you can open the .urdf file and change the visual and collision orientations (x,y,z and r,p,y) to reflect the changes in your visualization. 

Running the tool : You are now ready to run your custom tool with the Hello interface! Your python scripts can call either the tool as a separate class or as a part of the end_of_arm class. The script should include homing, startup, your tool operation and end of process. 

## Summary
Using the above tutorials, one can get started easily with the HelloNode and ToolShare class.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- [Hello Robot Documentation](http://docs.hello-robot.com/)
- [Hello Node class](https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py)
- [Tool Share examples](https://github.com/hello-robot/stretch_tool_share)
