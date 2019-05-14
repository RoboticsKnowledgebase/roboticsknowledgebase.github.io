   This excerpt covers the basic methodologies to set up communication channels with the Clearpath Husky and make sure the hardware is receiving commands that are input by the user. The article will cover the hardware used for Husky's onboard processing, additional hardware needed for better Localization and a brief description of Husky's ROS packages for Localization.

## Husky Onboard Controller:
   The main communication channel between the user and the Husky is through an onboard computer. Therefore, based on the project and the algorithms to be run on the Husky, a selection of onboard computer may be made. Nvidia Jetson and Zotac are the two popular onboard computers among Carnegie Mellon University (CMU) Masters in Robotic Systems Development (MRSD) students.  
 
   This article discusses the Nvidia Jetson, its initial integration with the Husky and the advantages of this device in other Husky related tasks.

   The Nvidia Jetson is a good choice for the Husky’s onboard computation. As the Jetson has significant compute power, it can host computation heavy algorithms needed for SLAM(simultaneous localization and mapping) and Path Planning.

   To integrate the Jetson with the Husky, first, the Jetson has to be powered using a 12 volt supply that can be drawn from the Husky's battery-powered electric outputs that are located in the open compartment of the Husky. Also, two Wi-Fi antennas need to be electrically connected to the Jetson. These help in easily accessing the Jetson's folders via Secure Shell(SSH), more on this later. Once the Jetson is powered, it has to be flashed with Linux OS. The detailed steps for flashing the Jetson can be obtained via these YouTube videos:

[*Flashing NVIDIA Jetson TX2*](https://www.youtube.com/watch?v=9uMvXqhjxaQ)

[*JetPack 3.0 - NVIDIA Jetson TX2*](https://www.youtube.com/watch?v=D7lkth34rgM)

   Now the Jetson should be connected to a common WiFi network as the user's laptop. The Jetson's IP address can be obtained by typing "ifconfig" command in the Terminal window\. Then an SSH communication can be set up between the user's laptop and the Jetson using the command: *ssh  "your_jetson_devise_name"@ipaddress*  Ex: ssh husky\_nvidia@192\.168\.0\.102
Once a secure shell connection is established between the user’s laptop and the Jetson, Husky control commands can be sent via this connection.

## Husky Teleoperation
   To test the working of Husky’s internal hardware, a simple Husky movement check would be sufficient. The Husky movement can be achieved by teleoperation commands. The ‘teleop_twist_keyboard’ ROS package can be utilized for this purpose. To install this package to /opt/ros folder, the following command can be used via Terminal : 
*$ sudo apt-get install ros-kinetic-teleop-twist-keyboard*

This package can be run using the command *$ rosrun teleop_twist_keyboard teleop_twist_keyboard.py*

By using the keyboard buttons u; i; o; j;  k; l; m;  , ;  . the Husky’s movement can be tested.

## Husky Localization 
After the initial communication setup of the Husky, the Husky outdoor Localization can be achieved using the following hardware:  
1.	UM7 IMU
2.	GPS receiver
3.	Odometry (*Utilizes the Husky’s internal hardware*)

    First, each of the above hardware's working can be testing using a simple *rostopic echo* command. If all three hardware publish data, the Husky's localization can be tested using the Clearpath Husky's outdoor package. 
        
A detailed description of the navigation package can be found in the following link:
<http://www.clearpathrobotics.com/assets/guides/husky/HuskyGPSWaypointNav.html>

## Summary
Using the above description, the Clearpath Husky can be set up easily and quickly. 
