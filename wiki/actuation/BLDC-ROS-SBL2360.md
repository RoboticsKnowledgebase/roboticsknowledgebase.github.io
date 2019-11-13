# BLDC motor control in ROS using RoboteQ controller
---
__Reference__

- __[reference 1](https://www.roboteq.com/index.php/docman/motor-controllers-documents-and-files/documentation/datasheets/sbl23xx-datasheet/288-sbl23xx-datasheet/file)__ - SBL23xx datasheet: This manual is about how to set up electrical connection of the controller.
- __[reference 2](https://www.roboteq.com/index.php/docman/motor-controllers-documents-and-files/documentation/user-manual/272-roboteq-controllers-user-manual-v17/file)__ - RoboteQ controller user manual: This is a detailed manual about the functions and usage of RoboteQ controllers. I/O configuration, operation mode, and commands reference are the most important sections.
- __[reference 3](https://www.roboteq.com/index.php/docman/motor-controllers-documents-and-files/documentation/user-manual/642-roborun-plus-manual/file)__ - RoboRun user manual: This manual is about the basic usage of the GUI software.
---

## 1. Introduction
RoboteQ primarily manufactures Brushless DC, Brushed DC and AC Induction motor controllers. Apart from controllers, they also make a few IO boards, IMUs and battery management systems. Their motor controllers incorporate a wide set of features, which often makes them a promising choice for most robotics applications. For instance, they typically have:
+ Many interfacing options, such as USB, RS232, CANbus and Ethernet.
+ Many models for different power requirements (even upto 400A)
+ Trapezoidal control or Sinusoidal FOC for Brushless DC motors
+ Multi-channel options (drive multiple motors with one controller)
+ Regenerative Braking
+ Unofficial open-source ROS drivers
+ Well documented and detailed user manuals

Their motor controllers come with a software application called RoboRun+, which is a GUI to configure, test, monitor and run diagnostics for the controller. Although the GUI is fairly intuitive, configuring the controllers can be challenging and can require you to know many parameters of the motor that you are trying to control. For instance, if you intend to use field-oriented control (highly recommended) for a brushless DC motor as opposed to trapezoidal control, there are a few parameters that need to be calculated with the help of the manual. It may help to do the initial setup with the guidance of a technical support representative, who generally are quite responsive, knowledgeable and helpful.

## 2. Electrical Connection
The SBL2360 controllers have two channels and therefore can control two motors.  The controllers can be powered separately from the motors and this is recommended by RoboteQ.  Per the user manual, reference 2, the controllers can be powered with 7V-60V.  In order to turn off the motor controllers the input must be connected to ground.  The controllers will not fully turn off if the power connection is left floating.  

The motors can be powered with up to 60V.  Reference 2 explains recommended connections if batteries are used or if power supplies are used to power the controller and motors.  In both cases an inline fuse can be used to prevent a current surge from reaching the controllers.  If power supplies are used, a power dissipation circuit needs to be connected to the motor supply.  This is due to potential power generation that can occur when the motors are not controlled.  The power generation will cause reverse current to flow which has the potential to damage the power supply.  If batteries are used, the current will charge the battery so it is not an issue.  An example power dissipation circuit is provided in reference 2.  It utilizes a mosfet to redirect the current to ground if the motor controller detects an overvoltage.  

Regardless of how the motors and controllers are powered it is a safety feature to include an emergency stop.  This switch will disconnect power from the motors preventing them from moving in the event they begin to move in an unsafe manner.  Since the motors can use up to 60V it is recommended to use a relay with normally open switches to control power to the motors.  The relay can be connected to the emergency switch that uses less voltage.  When the emergency switch is triggered it triggers the relay which causes the switches that supply power to the motors to open, therefore cutting power to the motors.  

The controllers utilize feedback from sensors such as encoders to control the motors.  The molex connectors on the controllers accept the data lines from the encoder and power the encoder.  There is also a DB25 connector on the controllers that provides pins for digital outputs and inputs, serial communication, and more.  A full list of the functionality can be found in the user manual.  

## 3. Motor Control in ROS
To use the RoboteQ SBL2360 to control brushless DC (BLDC) motors, you need to follow two major steps:
+ controller configuration: configure the motor controller through a GUI software in Windows called RoboRun;
+ ROS interface: look up useful serial command from RoboteQ user manual, and set up c++ serial library to send corresponding commands through ROS.
### 3.1 Controller configuration
The whole configuration part happens in the RoboRun GUI on a Windows laptop. I will only discuss the parameters I modified as following, just keep the default for others.
+ Download the RoboRun software from RoboteQ website and connect the controller to the Windows laptop through USB.
+ Update controller firmware. Just follow “firmware update” in Reference 3.
+ Configure the sensor input. In our case, we use SSI encoders as our feedback sensor. 
    - Go to “Configuration -> Inputs/Outputs -> SSI Sensors -> SSI Sensor 1”. Change “Use” to “Feedback” (“Absolute Feedback” can only deal with one turn, which is 0-360 degrees). 
    - Change “Sensor Resolution” to the resolution of your specific encoder (for example, if your resolution is 4096, the encoder’s feedback value will be 4096 when the motor’s position is 360 degrees). 
    - RoboteQ SBL2360 can control two motor / encoder channels together. If you use two channels, do the same thing to “Sensor 2”.
+ Configure the output. 
    - Go to “Configuration -> General”. Change “Molex Input” to “SSI Sensors” (in our case).
    - Go to “Configuration -> Motor 1 -> Motor Configuration”. Change “Number of Pole Pair” to that of your specific motor. 
    - Change “Switching Mode” to “Sinusoidal”.
    - Go to “Sinusoidal Settings”. Change “Sinusoidal Angle Sensor” to “SSI Sensor” (in our case).
    - Go to “FOC Parameters”. Change “Flux Proportional Gain” and  “Flux Integral Gain”. The equation for calculating these two values are in page 119 of Reference 3.
    - Go to “Configuration -> Motor 1 -> Motor Output”. Change “Operating Mode” to “Closed Loop Position Relative”.
    - Go to “Closed Loop Parameters”. Change the PID gains to 1,0,0 for now.
    - RoboteQ SBL2360 can control two motor / encoder channels together. If you use two channels, do the same thing to “Motor 2”.
+ Test whether the motor can be actuated or not. Go to “Run”, there are two bars you can drag, which correspond to the commands for the two motors. You can monitor “Feedback 1”, “Feedback 2” at the bottom of the “Run” section. If the feedbacks can track the commands, it means the motors is successfully actuated. It probably doesn’t work in your first test. Here are several things you can play with to make it work:
    - It might be good for you to start with “Open Loop” mode instead of “Closed Loop Position” mode. Under “Open Loop”, if there is no load, the motor will spin faster and faster when you increase the command.
    - Modify PID gains in “Configuration -> Motor 1 -> Motor Output -> Closed Loop Parameters”. Bigger gains will reduce steady state errors, as well as increasing the input current amps to the motor. If the current is bigger than the controller’s capacity, it will be detected as “short”.
    - Modify “Configuration -> Motor 1 -> Motor Output -> Speed & Acceleration”. Slower movement will require smaller current.
    - If the zero position of encoder is not desired, move the motor to the desired zero position, go to “Console”, and type “!CSS 1 0”. This command will calibrate the SSI encoder in the channel 1 back to zero.
    - The “Motor / Sensor Setup” button in “Diagnostics” can automatically calibrated some of the configuration parameters for you.
+ Test serial commands. Go to “Console”. Type some serial commands you need and send them. Check whether the performance make sense. For example, “! G 1 500” actuates the motor of channel 1 to the position of 500 (For your own motor, you need to figure out the projection relationship between the unit of the “position” in the command and the actual position in degree). “? CSS 2” will query the SSI encoder feedback of the channel 2 (If the feedback is “CSS=1000”, then the actuate position in degree will be 1000/encoder_resolution*360 degrees).
### 3.2 ROS interface
You can implement whatever interface, as long as it can send the serial commands you want to the controller by ROS, just like what you tested above by GUI console. Our interface is just an example for your reference.
+ Look up commands from the “Commands Reference” section of Reference 2, and pick the required commands for your own use cases. Test these commands to validate they are good on the GUI console. For our case, the most useful commands are “!G x y”, which actuates the motor in channel x to the position y, and “? CSS x”, which queries the encoder feedback of channel x.
+ We used functions developed by Grant Metts, who TAs MSRD project course in Fall 2019. You need these [three files](https://github.com/boshenniu/motor_control_ROS/tree/master/serial_command) to make use of these functions: 
    - These functions are based on C++ boost libraries for serial communication.
    - “Blocking_reader.h” is low level-serial interface we don’t need to modify.
    - You can make use of or create new functions in “RoboteqInterface.cpp” and “RoboteqInterface.h”. The structure of these functions are similar except the specific command they send, which make it easy for you to create your own functions.
+ Create executable C++ file for your ROS package, where you call the functions in “RoboteqInterface.cpp” and do what you want.