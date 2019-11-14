---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. You should set the article's title:
title: Webots
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
[Webots](https://cyberbotics.com/) is a free open source robot simulator used for a variety of purposes and applications. It was previously a paid software developed by Cyberbotics and became open source in December 2018. It is still being developed by Cyberbotics with association from Industry and Academia. It is compatible with ROS 1 and ROS 2 and runs on Linux, Windows and macOS.

## Installation
Webots works out of the box by downloading the right version at this [link](https://cyberbotics.com/download).
### Ubuntu
- Extract the downloaded file to a location of your prefernce.
- Go to that location in your Linux terminal.
- \[Optional: If using ROS with Webots\] Source devel for your catkin workspace `source \path\to\catkin_ws\devel.bash`
- Run the following commands: 
  ```
  cd webots
  ./webots
  ```

## ROS Integration
Here are the two ways you can add ROS integration to webots

1. Standard ROS Controller
    - Webots comes with inbuilt ROS controller which generates ROS topic out of the box based on the elements(robots, sensors, etc) added in the environment. For example, if the stereo sensor is added to the robot and controller selected is “ROS”, then the topic likes “\left_camera_image” and “\right_camera_image” is published by default. 
    - All you have to do to get this working is to change the controller field in robot node to ROS.
    - Some of the examples code can be found at [link] (https://github.com/cyberbotics/webots/tree/master/projects/languages/ros/webots_ros)

2. Custom ROS Controller
    - While using standard ROS controller has plug and play benefits, to gain total control over the data that is published from the webots, using custom ROS controller is preferable
    - Custom ROS controller can be written in both cpp and python.
    - Simply change the controller to `your_controller.py` and `import rospy in <your_controller.py>` and all the rospy commands should work as-is.
    - Make sure you start the webots instance after sourcing the ROS workspace otherwise rospy and other ROS messages won’t be accessible inside webots.
    - All the information regarding elements in the simulation are accessible through custom function which can be further converted to ROS topics and published accordingly

For more information on using ros, you can refer to [link](https://cyberbotics.com/doc/guide/using-ros)
For a sample ROS integration, refer [link](https://cyberbotics.com/doc/guide/tutorial-8-using-ros)


## Controllers
### Robot Controller
- The Robot Controller is the primary controller used to control all the things related to the robot. To start using ros controller you need to import the following library
  ```
  from controller import Robot
  ```
- Robot controller should ideally consist of all the code regarding robot perception, planning, navigation and controls. If you are planning to develop this stacks outside webots, you can use ROS integration to integrate with the webots. For example, we implemented the odometry and controls part in the webots and inputs to this was served through the planner which was developed in ROS. 
- You should activate all the sensors in this file itself.

You can find more information regarding robot controllers at [link](https://cyberbotics.com/doc/guide/controller-programming)

### Supervisor Controller
Supervisor is a special controller type for the robot controller. It is defined in the `wbt` file as a Robot object with the  `supervisor` field set to `TRUE`. It allows you to access and control fields of the simulation that are not or should not be accessible to the robots. For example, it allows you to query the ground truth positions of the objects in the world, change illumination throught time, etc. Note: There can only be ONE supervisor in a world.

#### Tutorial: Changing Illumination with time in a world
Using the supervisor controller, we can change illumination of the world with time. We will see how to change the direction of `DirectionalLight` using a Supervisor controller. 

In the world file of the world you are using, add a supervisor robot by adding the following lines
```
Robot {
  controller "supervisor_controller"
  supervisor TRUE
}
```

We will write the corresponding contoller for the supervisor in python. Create a file with the same name as mentioed in the above lines. In the controller, create a supervisor object, get its root and query the node related to `DirectionalLight`. In a while loop, set the vector in the field `direction` to the appropriate values.
```
# Initialize the supervisor object
supervisor = Supervisor()

# Access the children
root = supervisor.getRoot()
children = root.getField("children")

# Use the children to access the node in charge of handling the DirectionalLight
light_node = children.getMFNode(2) # Usually 2 depends on at what position DirectionalLight was added in the world file.

# Access the field 'direction'
direction_field = light_node.getField("direction")

# initalize dir_sunlight with appropriate sunlight directions across time.

# Step in the world
while(supervisor.step(TIME_STEP)!=-1):
    # Set the direction of the DirectionalLight
    direction_field.setSFVec3f((dir_sunlight[:,num_loops]).tolist())
```

You can also manipulate other fields of the light such as `intensity`, `color`, `ambientIntensity` and `castShadows`. 

## Surface Map
You can add surface for the robot environment from CSV as follows 
Webots provides elevation grid field inside SOLID which can be used to define the heightmap
  - If you are planning to use `w x h` surface map, set xDimension to w and zDimension to h inside the ElevationGrid field.
  - In the height field, you can ideally add the flattened version of the CSV of size `1 x wh`. You can go to the proto file of the solid and add under the height field. 
  - Ideally, it’s recommended to keep xSpacing and zSpacing to 2 or 3 so that height maps are smoothened. Be careful while making these changes as this will convert your map to `xSpacing*w x zSpacing*h` dimension. You might need to make changes accordingly in the other parts of the code. 

## Sensors
### Multisense S21 (Depth camera)
- Webots provides Multisense S21 as depth camera which can be added to the extension slot in robots. Multisense S21 is actually made of left/right camera and range finder to replicate the depth camera. Here are the steps to add the sensors
  - Go to extension slot in the robot node
  - Right-click and click on add new button
  - From Proto Nodes(Webots Project) dropdown, select device, multisense
  - Add the MultisenseS21 sensor
  - You can change a few parameters like FOV, max/min range of the robot. 
- You can use `robot.getCamera(<Camera Name>)` function to get the camera and use `<yourcamera>.enable(timestep)` to activate the sensor. 
- More APIs related to the sensors is available at [link](https://cyberbotics.com/doc/guide/camera-sensors#multisense-s21)
- Sometimes there might be a need for changing sensor parameters which are not publicly accessible in which case you can create your own sensors. You can do this by copy-pasting the proto files and change the parameters in the proto files or make the variables publicly accessible to add it to GUI

### Position sensors (Encoders)
Webots provides Position sensors which can be used both for rotational and translational motion. One of the best use of position sensor is to use it as encoders with motor. Here are the steps to add Position sensors 
- Go to motor node inside `<your_robot>.proto` file and add followings lines inside your devices section
  ```
  PositionSensor {
    name "<custom_name>"
  }
  ```
- Add the following lines to your code to enable the sensor
  ```
  pos_sensor = <your_robot>.getPositionSensor(“<custom_name>”)
  pos_sensor.enable()
  ```

Webots provides a library of sensors which should cover most of your requirements. Here is the link to the library of sensors [link](https://cyberbotics.com/doc/guide/sensors). Each of these sensors has public APIs as well as ROS topic which can be published and used with ROS.

## Summary
This tutorial gives you a brief idea of how to get started with webots and be able to control the basic and some intermediate features for your simulation.

<!-- ## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here. -->
