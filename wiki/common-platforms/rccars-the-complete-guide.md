---
date: 2023-05-03
title: Making RC Vehicles Autonomous
published: true
---
The goal of this guide is to discuss the trade-offs and choices to be made amongst the options available to set up an RC car and enable low-level feedback control such that the autonomy stack can be built on top of this platform. This can then serve as a test bench for autonomous cars.


## Chassis
In the world of RC vehicles, scale refers to the degree to which the dimensions of a full-sized car are reduced to create the RC model. Commonly available scales include 1/24, 1/18, 1/12, and 1/10. When deciding on a scale for an RC vehicle, there is a trade-off to consider. A larger scale vehicle provides more space for custom sensing and hardware, but may have a larger turning radius. On the other hand, a smaller scale vehicle will have a tighter turning radius, but may present challenges for customizing and mounting additional components due to space constraints.DJI drones have one of the best commercially available PID controllers along with its state of the art hardware, but for research and enterprise development users, the limitations imposed are noteworthy. After a certain depth, the software stack is locked from customization, and the controller performs much like a black box.

An RC car with Ackermann steering is best suited for testing autonomous vehicles being built for city and highway driving. It is the most common steering mechanism found in commercial cars. This type of steering system is typically found in higher-end RC cars. If your goal is to work on the scaled representation of autonomous vehicles for a different environment: 1) Off-road autonomy: consider RC vehicles with bell crank or dual bell-crank steering systems, 2) Warehouse/Industrial settings: consider RC vehicles with differential drives and Mecanum wheels. The following content in this section lists and compares some popular off-the-shelf RC vehicles with an Ackermann steering.

**Latrax Prerunner**: This car is a 1/18 scale 4 wheel drive RC vehicle with 2.4Ghz radio system equipped with 7.2v NiMH battery. It is an Ackerman steered RC vehicle with a drag link. It comes with plastic servos for steering and a plastic spur gear in the drive train.

**Horizon Axial**: The SCX24 Deadbolt is 1/24 Scale 4 wheel drive RC car with Axial AX-4 and 2.4GHz 3-channel radio system. It comes equipped with a 350mAh 7.4V LiPo battery.

**Traxxas XL-5**: The Traxxas XL - 5 is 1/10 scale RC vehicle with 2.4 GHz radio system and 3000mAh, 8.4V, 7-cell NiMH battery. It comes with a metal servo and adjustable suspensions.

**Roboworks ACS**: This 1/10 scale RC vehicle comes with ready to plug slots for ROS controller, LiDAR and Camera. It is equipped with onboard Ubuntu, ROS1 and STM32 drivers. It is the most holistic research RC vehicle platform that is currently available for use.

| Serial No.  | Criteria | Latrax Prerunner | Horizon Axial | Traxxas XL-5 | Roboworks ACS
| ------------- | ------------- |------------- |------------- |------------- |------------- |
| 1 | Inbuilt Sensors  | 1 | 1 | 1 | 8 |
| 2 | Extensibility  | 8 | 1 | 8 | 6 |
| 3 | Cost | 8 | 8 | 5 | 1 |
| 4 | Workability | 5 | 1 | 8 | 10 |
| 5 | Modularity | 6 | 1 | 8 | 10 |
| 6 | Spare Parts | 10 | 10 | 10 | 3 |

_Disclaimer: The scores assigned are the personal opinions of the author and not intended to promote/slander and product/brand._

## Electric Drive System
Brushless DC motors offer advantages over brushed DC motors for autonomous driving due to their electronic commutation, faster response time, and greater efficiency. Brushless motors allow for more precise and accurate control of speed and torque through electronic control. In brushed DC motors the brushes and commutators can create friction, wear and tear, and electromagnetic interference that may limit precision and accuracy over time. However, brushed DC motors are generally less expensive, simpler in design, and relatively simpler to control electronically.

For the electronic speed control of motors, a power electronic circuit is used to modulate the power supplied. It is critical to choose the right ESC based on the power ratings of the motor and the battery. ESCs typically can be controlled via PWM signals - you can use a microcontroller/single-board computer to do so. Using hardware PWM pins offer more accurate control, and it is recommended to use them whenever possible. ESCs typically operate at 50Hz with pulse range 1000 microseconds to 2000 microseconds; here it corresponds to 5% and 10% of duty cycle respectively. Pay close attention to understanding what the beep(s) of your ESC means, be sure to carry out the arm process (safely) upon powering up the ESC, and perform any other required calibration through the UI tool or by flashing a different firmware onto the ESC. This [tutorial](https://www.circuitbasics.com/introduction-to-dc-motors-2/) introduces speed control of DC motors.

Note: ESCs also have BEC (Battery Eliminator Circuit) which provides a 5V output that can be used to power other on-board devices.

## Sensor suite for odometry
There are various options to choose from whilst trying to get odometry feedback from RC vehicles. This simplest way to do so is by using rotary encoders on the wheels - however their output can be misleading when factors like wheel slip come into play. There are various choices in terms of the form factor and sensing technology available here.

Another option to incorporate feedback is to measure the RPM of the motor - multiply this by the final drive ratio and the wheel diameter to estimate the vehicle odometry. This can again be done in multiple ways i.e., via a sensored motor or by integrating a tachometer or by using E-RPM (Electronic RPM: estimated from the back EMF generated by the motor) feedback from the motor.

Finally, an inertial measurement unit can also be used to simulate odometry by integrating the acceleration and angular velocities. However, it is not recommended that this is used as the sole source of odometry information as it is difficult to estimate the bias in IMUs and errors in the odometry estimation can accumulate quickly.

## Microcontrollers and single board controllers
There is a plethora of compact computing devices available for building and deploying algorithms (Arduino boards, ESP series, Intel NUC, Raspberry Pi SBCs, Jetson SBCs, boards from National Instruments, etc.). Consider the number of different pins and their types available on the device, built-in communication protocols, computing power and RAM, microROS vs ROS serial vs a specific distribution of ROS.

## Configuring the VESC 6 EDU and controlling it via ROS Noetic deployed on a Raspberry Pi 4B.
The first step was configuring a light-weight Ubuntu distribution to deploy ROS Noetic on the Raspberry Pi. We found that working with an Ubuntu Server was the best solution - a lot of the development was then done remotely via SSH/updating repos from commits on Github. Some important next steps were to edit the Netplan config files to configure WiFi, set up a GitHub SSH key, enable SSH, install ROS Noetic (base), and install GPIO libraries. If you need to synchronize timestamps across a distributed system - use Chrony which is an implementation of NTP. Using Chrony a server-client architecture can be established, and one single clock can be assumed to be the ground truth - and tracked by different systems on the network (if more precision is required, consider using PTP). Follow the F1Tenth documentation to configure the VESC. Be sure to avoid any ground loops when working with the VESC as it could lead to the VESC getting damaged and spoilt. The ROS Noetic branch for VESC is available [here](https://github.com/VineetTambe/vesc) (access to IMU data was added). Use the VESC tool to configure and tune the low-level control of the motor, servo output and trim, and to calibrate the IMU.
Note: In high speed applications consider communication to the VESC via the CAN bus instead of USB for faster communication and response.

## See Also:
- [F1 Tenth](https://f1tenth.readthedocs.io/en/stable/)
- [Time suynchronization](https://www.robotsforroboticists.com/timing-synchronization/)
- [NTP](https://serverfault.com/questions/806274/how-to-set-up-local-ntp-server-without-internet-access-on-ubuntu)

## References:
- https://robocraze.com/blogs/post/how-to-choose-esc-for-quadcopter
- https://github.com/imbaky/Quadcopter
- https://raspberrytips.com/how-to-power-a-raspberry-pi/
- https://github.com/cst0/gpio_control
- https://www.faschingbauer.me/trainings/material/soup/linux/hardware/brushless-motor/topic.html
- https://howtomechatronics.com/tutorials/arduino/arduino-brushless-motor-control-tutorial-esc-bldc/
