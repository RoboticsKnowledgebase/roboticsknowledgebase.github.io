---
date: {}
title: ROS Mapping and Localization
published: true
---
## Mapping

To map the environment, there are many ROS packages which can be used:

- ### [Gmapping](http://wiki.ros.org/gmapping)

  - Gmapping requires odometry data from the mobile robot. So, if one has odometry data coming from the robot, Gmapping can be used.

- ### [Hector Mapping](http://wiki.ros.org/hector_mapping)

  - The advantage of using Hector Mapping is that it does not need Odometry data and it just requires the LaserScan data. Its disadvantage is that it does not provide loop closing ability but it is still good for real-world scenarios specially when you do not have odometry data.

  - Even if one has odometry data, Hector Mapping is preferred over Gmapping. Hector Mapping also gives good pose estimates of the robot.

  - Using Hector Mapping, one can create a very good map of the environment. The other
    option is to generate a map in softwares like Photoshop. However, one should make sure to have a proper resolution while making a map in Photoshop.

## Localization

### [AMCL](http://wiki.ros.org/amcl)

For localization of the robot, the ROS package of AMCL (Adaptive Monte Carlo
Localization) works well. It is straightforward to run the AMCL ROS package. AMCL can not handle a laser which moves relative to the base and works only with laser scans and laser maps. The only thing which should be taken care of is that it requires an odometry message. There are different ways to generate this odometry message. The odometry data can be taken from wheel encoders, IMUs, etc.. which can be used to generate the odometry message which can be supplied to AMCL. Another neat trick is to use pose obtained from the Hector mapping to generate an odometry message which can be then supplied to AMCL. If Hector mapping pose
is used to generate odometry message, then no external odometry is required and the result is
pretty accurate.

### [Robot Localization](http://docs.ros.org/en/noetic/api/robot_localization/html/index.html)

Robot Localization (```robot_localization```) is a useful package to fuse information from arbitrary number of sensors using the Extended Kalman Filter (EKF) or the Unscented Kalman Filter (UKF). Different from AMCL above, it does not require a map to start working. In projects related to autonomous driving, a map is usually unknown beforehand, in which case the common approach is to localize the robot using its onboard sensors, such as the most prevalent ones, IMU and wheel encoder. Below we will introduce [how to set up](set-up-odometry) the odometry on one's custom robot, [how to simulate](simulate-an-odometry-system-in-gazebo) an odometry system (IMU and wheel encoder), as well as [how to fuse](fusion-using-robot-localization) the odometry sensor inputs from IMU and encoder into a locally accurate smooth odometry information using the handy ```robot_localization``` package. 

#### Intro

The odometry system provides a locally accurate estimate of a robot’s pose and velocity based on its motion. The odometry information can be obtained from various sources such as IMU, LIDAR, RADAR, VIO, and wheel encoders. One thing to note is that IMUs drift over time while wheel encoders drift over distance traveled, thus they are often used together to counter each other’s negative characteristics.

#### Set up odometry

Setting up the odometry system for your physical robot depends a lot on which odometry sensors are available with your robot. Due to the large number of configurations your robot may have, specific setup instructions will not be within the scope of this tutorial. Instead, we will use an example of a robot with wheel encoders as its odometry source. The goal in setting up the odometry is to compute the odometry information and publish the ```nav_msgs/Odometry``` message and ```odom``` => ```base_link``` transform over ROS 2. To calculate this information, you will need to setup some code that will translate wheel encoder information into odometry information. An example of a differential drive robot is as follows:

```
linear = (right_wheel_est_vel + left_wheel_est_vel) / 2
angular = (right_wheel_est_vel - left_wheel_est_vel) / wheel_separation
```

The ```right_wheel_est_vel``` and ```left_wheel_est_vel``` can be obtained by simply getting the changes in the positions of the wheel joints over time. Then one can publish odometry information by following the [tutorial](http://wiki.ros.org/navigation/Tutorials/RobotSetup/Odom/).

For other types of sensors such as IMU, VIO, etc, their respective ROS drivers should have documentation on how publish the odometry information. 

#### Simulate an odometry system in Gazebo

Assuming one has installed Gazebo and is familiar with [Using a URDF in Gazebo](http://wiki.ros.org/urdf/Tutorials/Using%20a%20URDF%20in%20Gazebo), we can add IMU and a differential drive odometry system as Gazebo plugins, which will publish ```sensor_msgs/Imu``` and ```nav_msgs/Odometry``` messages respectively. 

For an overview of all different plugins available in Gazebo, take a look at [Using Gazebo Plugins with ROS](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins). For our robot, we will be using the [GazeboRosImuSensor](http://gazebosim.org/tutorials?tut=ros_gzplugins#IMUsensor(GazeboRosImuSensor)) which is a SensorPlugin. A SensorPlugin must be attached to a link, thus we will create an ```imu_link``` to which the IMU sensor will be attached. This link will be referenced under the ```<gazebo>``` element. Next, we will set ```/demo/imu``` as the topic to which the IMU will be publishing its information. We will also add some noise to the sensor configuration using Gazebo’s [sensor noise model](http://gazebosim.org/tutorials?tut=sensor_noise).

Now we will set up our IMU sensor plugin according to the description above by adding the following lines before the ```</robot>``` line in our URDF:

<details>
  <summary>Click to expand URDF code</summary>
  
```
<link name="imu_link">
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </visual>

  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>

  <xacro:box_inertia m="0.1" w="0.1" d="0.1" h="0.1"/>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.01"/>
</joint>

 <gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
   <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
      <ros>
        <namespace>/demo</namespace>
        <remapping>~/out:=imu</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```
</details>

Now, let us add the differential drive ModelPlugin. We will configure the plugin such that ```nav_msgs/Odometry``` messages are published on the ```/demo/odom``` topic. The joints of the left and right wheels will be set to the corresponding wheel joints of your robot. The wheel separation and wheel diameter are set according to the values of the defined values of ```wheel_ygap``` and ```wheel_radius``` respectively.

To include this plugin in our URDF, add the following lines after the ```</gazebo>``` tag of the IMU plugin:

<details>
	<summary>Click to expand URDF code</summary>
    
```
<gazebo>
  <plugin name='diff_drive' filename='libgazebo_ros_diff_drive.so'>
    <ros>
      <namespace>/demo</namespace>
    </ros>

    <!-- wheels -->
    <left_joint>drivewhl_l_joint</left_joint>
    <right_joint>drivewhl_r_joint</right_joint>

    <!-- kinematics -->
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>

    <!-- limits -->
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>

    <!-- output -->
    <publish_odom>true</publish_odom>
    <publish_odom_tf>false</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>

    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```
</details>

Then you can simply launch your robot in Gazebo as you normally do and verify the ```/demo/imu``` and ```/demo/odom``` topics are active in the system. Observe that the ```/demo/imu``` topic publishes ```sensor_msgs/Imu``` type messages while the ```/demo/odom``` topic publishes ```nav_msgs/Odometry``` type messages. The information being published on these topics come from the gazebo simulation of the IMU sensor and the differential drive respectively. Also note that both topics currently have no subscribers. In the next section, we will create a ```robot_localization``` node that will subscribe to these two topics. It will then use the messages published on both topics to provide a fused, locally accurate and smooth odometry information.

#### Fusion using Robot Localization

A usual robot setup consists of at least the wheel encoders and IMU as its odometry sensor sources. When multiple sources are provided to ```robot_localization```, it is able to fuse the odometry information given by the sensors through the use of state estimation nodes. These nodes make use of either an Extended Kalman filter (```ekf_node```) or an Unscented Kalman Filter (```ukf_node```) to implement this fusion. In addition, the package also implements a ```navsat_transform_node``` which transforms geographic coordinates into the robot’s world frame when working with GPS.

Fused sensor data is published by the ```robot_localization``` package through the ```odometry/filtered``` and the ```accel/filtered``` topics, if enabled in its configuration. In addition, it can also publish the ```odom``` => ```base_link``` transform on the ```/tf``` topic.

If your robot is only able to provide one odometry source, the use of ```robot_localization``` would have minimal effects aside from smoothing. In this case, an alternative approach is to publish transforms through a tf2 broadcaster in your single source of odometry node. Nevertheless, you can still opt to use ```robot_localization``` to publish the transforms and some smoothing properties may still be observed in the output.

Let us now configure the ```robot_localization``` package to use an Extended Kalman Filter (```ekf_node```) to fuse odometry information and publish the ```odom``` => ```base_link``` transform.

First, install it using ```sudo apt install ros-<ros-distro>-robot-localization```. Next, we specify the parameters of the ```ekf_node``` using a YAML file. Create a directory named ```config``` at the root of your project and create a file named ```ekf.yaml```. Copy the following lines of code into your ```ekf.yaml``` file.

<details>
	<summary>Click to expand ekf.yaml file</summary>
    
```
### ekf config file ###
ekf_filter_node:
    ros__parameters:
# The frequency, in Hz, at which the filter will output a position estimate. Note that the filter will not begin
# computation until it receives at least one message from one of theinputs. It will then run continuously at the
# frequency specified here, regardless of whether it receives more measurements. Defaults to 30 if unspecified.
        frequency: 30.0

# ekf_localization_node and ukf_localization_node both use a 3D omnidirectional motion model. If this parameter is
# set to true, no 3D information will be used in your state estimate. Use this if you are operating in a planar
# environment and want to ignore the effect of small variations in the ground plane that might otherwise be detected
# by, for example, an IMU. Defaults to false if unspecified.
        two_d_mode: false

# Whether to publish the acceleration state. Defaults to false if unspecified.
        publish_acceleration: true

# Whether to broadcast the transformation over the /tf topic. Defaultsto true if unspecified.
        publish_tf: true

# 1. Set the map_frame, odom_frame, and base_link frames to the appropriate frame names for your system.
#     1a. If your system does not have a map_frame, just remove it, and make sure "world_frame" is set to the value of odom_frame.
# 2. If you are fusing continuous position data such as wheel encoder odometry, visual odometry, or IMU data, set "world_frame"
#    to your odom_frame value. This is the default behavior for robot_localization's state estimation nodes.
# 3. If you are fusing global absolute position data that is subject to discrete jumps (e.g., GPS or position updates from landmark
#    observations) then:
#     3a. Set your "world_frame" to your map_frame value
#     3b. MAKE SURE something else is generating the odom->base_link transform. Note that this can even be another state estimation node
#         from robot_localization! However, that instance should *not* fuse the global data.
        map_frame: map              # Defaults to "map" if unspecified
        odom_frame: odom            # Defaults to "odom" if unspecified
        base_link_frame: base_link  # Defaults to "base_link" ifunspecified
        world_frame: odom           # Defaults to the value ofodom_frame if unspecified

        odom0: demo/odom
        odom0_config: [true,  true,  true,
                       false, false, false,
                       false, false, false,
                       false, false, true,
                       false, false, false]

        imu0: demo/imu
        imu0_config: [false, false, false,
                      true,  true,  true,
                      false, false, false,
                      false, false, false,
                      false, false, false]
```
</details>

In this configuration, we defined the parameter values of ```frequency```, ```two_d_mode```, ```publish_acceleration```, ```publish_tf```, ```map_frame```, ```odom_frame```, ```base_link_frame```, and ```world_frame```. For more information on the other parameters you can modify, see [Parameters of state estimation nodes](http://docs.ros.org/en/melodic/api/robot_localization/html/state_estimation_nodes.html#parameters), and a sample ```ekf.yaml``` can be found [here](https://github.com/cra-ros-pkg/robot_localization/blob/foxy-devel/params/ekf.yaml).

To add a sensor input to the ```ekf_filter_node```, add the next number in the sequence to its base name (odom, imu, pose, twist). In our case, we have one ```nav_msgs/Odometry``` and one ```sensor_msgs/Imu``` as inputs to the filter, thus we use ```odom0``` and ```imu0```. We set the value of ```odom0``` to ```demo/odom```, which is the topic that publishes the ```nav_msgs/Odometry```. Similarly, we set the value of ```imu0``` to the topic that publishes ```sensor_msgs/Imu```, which is ```demo/imu```.

You can specify which values from a sensor are to be used by the filter using the ```_config``` parameter. The order of the values of this parameter is x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az. In our example, we set everything in ```odom0_config``` to ```false``` except the 1st, 2nd, 3rd, and 12th entries, which means the filter will only use the x, y, z, and the vyaw values of ```odom0```.

In the ```imu0_config``` matrix, you’ll notice that only roll, pitch, and yaw are used. Typical mobile robot-grade IMUs will also provide angular velocities and linear accelerations. For ```robot_localization``` to work properly, you should not fuse in multiple fields that are derivative of each other. Since angular velocity is fused internally to the IMU to provide the roll, pitch and yaw estimates, we should not fuse in the angular velocities used to derive that information. We also do not fuse in angular velocity due to the noisy characteristics it has when not using exceptionally high quality (and expensive) IMUs.

Now you can add the ```ekf_node``` into the launch file. For example, in ROS2, this can be:

```
robot_localization_node = launch_ros.actions.Node(
       package='robot_localization',
       executable='ekf_node',
       name='ekf_filter_node',
       output='screen',
       parameters=[os.path.join(pkg_share, 'config/ekf.yaml'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
)
```

Remember to add the ```robot_localization``` dependency to your package definition. That is, inside ```package.yml```, add one more ```<exec_depend>``` tag: ```<exec_depend>robot_localization</exec_depend>```. Also remember to open ```CMakeLists.txt``` and append the ```config``` directory inside the ```install(DIRECTORY...)``` as shown below:

```
install(
  DIRECTORY src launch rviz config
  DESTINATION share/${PROJECT_NAME}
)
```

After you build and launch, verify that ```odometry/filtered``` and ```accel/filtered``` topics are active in the system. You should see that ```/demo/imu``` and ```/demo/odom``` now both have 1 subscriber each, which is 

```
/ekf_filter_node
Subscribers:
  /demo/imu: sensor_msgs/msg/Imu
  /demo/odom: nav_msgs/msg/Odometry
  /parameter_events: rcl_interfaces/msg/ParameterEvent
  /set_pose: geometry_msgs/msg/PoseWithCovarianceStamped
Publishers:
  /accel/filtered: geometry_msgs/msg/AccelWithCovarianceStamped
  /diagnostics: diagnostic_msgs/msg/DiagnosticArray
  /odometry/filtered: nav_msgs/msg/Odometry
  /parameter_events: rcl_interfaces/msg/ParameterEvent
  /rosout: rcl_interfaces/msg/Log
  /tf: tf2_msgs/msg/TFMessage
Service Servers:
   ...
```

You may also verify that robot_localization is publishing the ```odom``` => ```base_link``` transform by running the command ```rosrun tf_echo odom base_link```. 