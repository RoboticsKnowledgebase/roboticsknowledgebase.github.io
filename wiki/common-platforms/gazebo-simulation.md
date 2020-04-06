## Introduction
This is a tutorial for Gazebo. What’s Gazebo anyway?

From Gazebo’s tutorials homepage:
>Gazebo is a 3D dynamic simulator with the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. While similar to game engines, Gazebo offers physics simulation at a much higher degree of fidelity, a suite of sensors, and interfaces for both users and programs.

>Typical uses of Gazebo include:
>* testing robotics algorithms,
>* designing robots,
>* performing regression testing with realistic scenarios
>
>A few key features of Gazebo include:
>* multiple physics engines,
>* a rich library of robot models and environments,
>* a wide variety of sensors,
>* convenient programmatic and graphical interfaces

This is particularly useful when we want to develop systems on robots, but we don’t always have the hardware around, or we don’t want to damage the robot with some potentially incorrect algorithms. Gazebo simulation could be our best friend in a way that we can simulate our robots, sensors, and to test algorithms in it. What’s more, we can launch Gazebo in a ROS launch file and pass in relevant parameters, which makes our life much easier.

## Outline
* Import a Model into Gazebo World
* Customize a Model
* Sensor Plugins for Gazebo-ROS Interface

### Import a Model into Gazebo World
To import a model inside a Gazebo world, one can simply open Gazebo, navigate to the `insert` tab at the top left corner,  and use your mouse to drag models into the scene.
However, if you want to import a customized model that is not under the default gazebo model path, this is what you should do. Add this following line in the launch file that is opening the Gazebo simulation so that Gazebo could recognize the models under a specific directory. (In this example, the model is defined in the package `autovalet_gazebo`)
```
<env name="GAZEBO_MODEL_PATH" value="$(find autovalet_gazebo)/models:$(optenv GAZEBO_MODEL_PATH)" />
```
One could also directly write codes in a .world file to insert a model. It is in URDF format, but it is not as straightforward as dragging an object in the GUI.

### Customize a Model
Sometimes, you might want to change some parts of the gazebo built-in models or simply create a new one from scratch. To create a new one, one can follow this [official tutorial](http://gazebosim.org/tutorials?tut=build_model) of Gazebo. However, if you were to modify some parts of an existing model, what could you do? For example, if we need a ground plane model with a customized ground image, we can follow [this tutorial](http://answers.gazebosim.org/question/4761/how-to-build-a-world-with-real-image-as-ground-plane/). We want to put all associated files under a `/models` directory specifically for Gazebo. Let’s take a look at an example model file structure.
```
~/catkin_ws/src/autovalet/autovalet_gazebo/models/cmu_ground_plane$ tree
.g
├── materials
│   ├── scripts
│   │   └── cmu_ground_plane.material
│   └── textures
│   	├── garage.png
├── model.config
└── model.sdf
```

You might notice that the default model path for Gazebo is
`~/.gazebo/models/cmu_ground_plane`, but instead, we here put our customized model at a different path `autovalet/autovalet_gazebo/models`, which is under our ROS package. This is a good practice since one would not want to mess up default models and self-defined models in the same directory.
The key here is to put the desired image file under `/materials/textures` and modify the image filename correspondingly in `/scripts/cmu_ground_plane.material`. 

### Sensor Plugins for Gazebo-ROS Interface
*Requirement*: Validating algorithms that are in the development stage on actual hardware is risky and time-consuming. Moreover, frequent operation of the robot is limited due to power, logistics and pandemic constraints. For example, in our case (MRSD ‘19-’21 AutoValet), we wanted to test parameters of RTAB-Map (see [this](http://introlab.github.io/rtabmap/)). Running the robot for each parameter would be time consuming and exhausting. Not to mention COVID-19 which prohibits outdoor activity and campus accessibility.

So, in order to ease things, a consistent simulation environment becomes a requirement. [Gazebo](http://gazebosim.org/) is a standard physics simulation environment that is fully supported by ROS. We can spawn robots, create worlds, provide motion commands, and interact with the environment in Gazebo. The [gazebo_ros](http://wiki.ros.org/gazebo_ros) package provides a link between ROS topics and the Gazebo environment.

Gazebo worlds are made of models which are essentially objects that you can pick and place. Models in gazebo require a .sdf file or a .dae file for rendering the object in the environment. This information is enough for passive objects such as a door, wall, dustbin, etc. However, what’s nice about Gazebo is the ability to simulate sensors and publish fake sensor data about its environment. These sensors need a “plugin” which generally defines what topics to publish, rate, topic params and a source file that converts environment information to sensor information.

Few standard sensors such as Microsoft Kinect, 2D laser, IMU have [existing plugin description](http://gazebosim.org/tutorials?tut=ros_gzplugins). However, modern sensors such as Realsense, Velodyne and Asus Xtion Pro, etc. do not have a plugin description. This article deals with acquiring and configuring the plugins for Realsense d435 and Velodyne VLP 16/32 to publish simulated sensor data from Gazebo to ROS which can be visualized in RViz.

*Setup: Velodyne Puck*
_Source: DataSpeed Inc._ <https://bitbucket.org/DataspeedInc/velodyne_simulator/src/master/>
In order to set up the plugin, clone the referred repository into the src of your catkin_ws and build the package using catkin_make.

```
cd path/to/catkin/workspace/src
git clone https://bitbucket.org/DataspeedInc/velodyne_simulator/src/master/
mv master vlp16_simulator
cd .. && catkin_make
```

To include the senor in your robot, add the following line in your URDF inside the robot tag.
`<xacro:include filename="$(find velodyne_description)/urdf/VLP-16.urdf.xacro"/>`

This includes the URDF xacro of the sensor which includes information on physical dimensions and link to the plugin src code.

After, including the sensor, we need to define a static link between the sensor and a previously existing link in the robot. In our case, the Clearpath Husky has a base_footprint, which is the center of the base frame. Most robots would have a base_link to which this link can be made to. For more information on links, joints, and URDF, I suggest [this](http://wiki.ros.org/urdf/Tutorials/Building%20a%20Visual%20Robot%20Model%20with%20URDF%20from%20Scratch). Along with positioning information, as this sensor only publishes on one topic, we can include the name of the topic and publish rate also in the same tag.
```
    <VLP-16 parent="base_footprint" name="velodyne" topic="/velodyne_points" hz="10" samples="440">
   	 <origin xyz="0 -0.015 1.47" rpy="0 0 -1.570796326"/>
      </VLP-16>
```
To check the point cloud being published, run your robot_description.launch ([sample](https://bitbucket.org/DataspeedInc/velodyne_simulator/src/master/velodyne_description/launch/example.launch)) with this updated URDF/Xacro file and run RViZ. Add the point cloud widget on the left pane, and subscribe to /velodyne_points topic to visualize the point cloud. Ensure that the fixed frame is one of the frames that is being published by tf (eg: base_link, Velodyne, velodyne_base_link). 
![Image Of pcl](http://www.contrib.andrew.cmu.edu/~szuyum/mrsd/et.png)
*Fig. 1: Simulated PointCloud of VLP-16*

*Setup: Realsense d435*
_Ref: Pal Robotics_ <https://github.com/pal-robotics/realsense_gazebo_plugin>

There exists no official ros-gazebo plugin for RealSense cameras as of the time this article is written. Pal robotics have an unofficial version that has not been pulled into the main repository yet due to a failed test (refer [this](https://github.com/IntelRealSense/realsense-ros/pull/1008)). However, the output generated by the plug-in is reasonably good for our use case.

In order to use the modified version, you need to pull [their version](https://github.com/pal-robotics-forks/realsense/tree/upstream) of the RealSense ROS package as compared to the [intel version](https://github.com/IntelRealSense/realsense-ros). This is because their version has a modified URDF for the d435 xacro which includes the [plugin parameters](https://github.com/pal-robotics-forks/realsense/blob/upstream/realsense2_description/urdf/_d435.gazebo.xacro). Ensure that you do not have a standard version of the RealSense package. A simple check would be to execute:

`sudo apt-get remove ros-kinetic-realsense-*`
(P.S: It also removes a few extra packages. Ensure it doesn’t remove anything you require. Worst case you can reinstall the removed ones after this command)

Pull and build their package in your catkin workspace. Ensure you have already installed the librealsense SDK from [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages). The librealsense library has build issues for NVIDIA Jetsona and similar ARM processors - stay away from those if you intend to use realsense and vice versa. If there’s no other way and you have to run it on a Jetson, there is another post on rkb that can help.

```
cd path/to/catkin/workspace/src
git clone https://github.com/pal-robotics-forks/realsense.git
cd realsense
git checkout upstream
cd .. && catkin_make
```

Having built the package, we set up the robot’s URDF as below to mount and publish the sensor stream. Inside the robot tag of your robot_description launch file include the modified xacro of the RealSense. Also set up the link w.r.t an existing frame on your robot. In our example, we knew where the camera had to be mounted w.r.t the Velodyne. 

```
<!-- Include the link for the realsense sensors from the local realsense2_package-->
    <xacro:include filename="$(find realsense2_description)/urdf/_d435.urdf.xacro"/>

<!-- Define the back camera link -->
    <sensor_d435 parent="velodyne" name="frontCamera" >
    	<origin xyz="0 0.197 -0.59" rpy="0 0 1.57079"/>
    </sensor_d435>
```

To update the parameters such as topic to publish, rate, and depth limits, you can modify the end of [this](https://github.com/pal-robotics-forks/realsense/blob/upstream/realsense2_description/urdf/_d435.gazebo.xacro) file in the realsense_description package.
![Image of rviz plugin](http://www.contrib.andrew.cmu.edu/~szuyum/mrsd/rviz_realsense_gazebo.png)
*Fig. 2: Camera image and point cloud published by the plug-in.*

## Summary
We have covered several aspects in Gazebo, including importing and customizing a model into a Gazebo world, and to implement plugins for the sensors as well as define the URDF files so that messages would be published accordingly. These two components are crucial when we want to start a ROS simulation using Gazebo.

## See Also:
* [Visualization and Simulation](https://roboticsknowledgebase.com/wiki/tools/visualization-simulation/)

## Further Reading
* Gazebo ROS tutorial: <http://wiki.ros.org/simulator_gazebo/Tutorials>
* RTAB Map: <http://introlab.github.io/rtabmap/>
* Gazebo-ROS documentation: <http://wiki.ros.org/gazebo_ros>
* Existing Gazebo Plugins: <http://gazebosim.org/tutorials?tut=ros_gzplugins>
* librealsense installation: <https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages>
* URDF tutorials: <http://wiki.ros.org/urdf/Tutorials/Building%20a%20Visual%20Robot%20Model%20with%20URDF%20from%20Scratch>

## References
* Customize ground plane with another image: 
<http://answers.gazebosim.org/question/4761/how-to-build-a-world-with-real-image-as-ground-plane/>
* Build a model in Gazebo: <http://gazebosim.org/tutorials?tut=build_model>
