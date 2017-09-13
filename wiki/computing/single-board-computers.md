---
title: Single-Board Computers
---

This article is out of date and should not be used in selecting components for your system. You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning}

To eliminate the use of bulky laptops on mobile robots and prevent tethering otherwise, single board computers may seem like a wise option. There are various affordable single board computers available in the market such as:
1. Hardkernel ODroid X2 – ARM 1.7Ghz quad core, 2gb RAM
2. Hardkernel ODroid XU – ARM 1.7Ghz octa core, 4gb RAM
3. FitPC 2 – Intel Atom 1.6Ghz, 2gb RAM

They are all available under $300 and their specifications and support seem
extremely good for use with Ubuntu and ROS. But, unfortunately, they all fail to
perform as desired. Following are certain problems and challenges that we faced
while trying to get single board computers to run for our project:

The ODroids have an ARM processor and hence, they do not run Ubuntu 12.04 (A
stable version supporting most packages). There is an ARM version of Ubuntu called
Ubuntu Linaro that needs to be installed on the ODroids. This version of Ubuntu has compatibility issues with ROS. The packages required in ROS had to be built from
source and the binary files for the same had to be generated manually. This was a
very troublesome process, as the installation of every package took about 10 times
longer than what it took on a laptop. Further, ODroid did not support certain
hardware such as Kinect and Hokuyo Laser scanner due to the lack of available
drivers.

There were also certain network issues due to which the ODroid did not get
connected to the Secured network of CMU though it was possible to get it connected
to the open CMU network as well as any local wireless networks. If you are ever
ordering the ODroid, make sure that you have the time and patience to set it up for
use like your laptop and that you do not require heavy computation or dependence
on any external hardware. Also, ODroid does not ship with a power adapter, WiFi
dongle, HDMI cable or memory card. Hence, these components need to be ordered
separately.

The FitPC2 runs on an Intel processor and hence, eliminates most of the troubles
that ODroid faces due to the ARM processor. Installation of Ubuntu and ROS is
exactly the same as it is on any other laptop. The installation may take some more
time as the single core processor of FitPC2 is not as good as the quad core
processors provided by Intel in the laptops.
FitPC2 is a great choice if the computation that is required on the robot is not very heavy. For example, we tried
to run Hector SLAM, AMCL, SBPL Lattice planner, Local base planner, Rosserial and
a laser scanner for a task in our project and the latency started getting higher and
the FitPC2 eventually crashed within a couple of minutes of the start of the process.
Another team doing computer vision processing in their project also eliminated
FitPC2 from their project due to its limited computation capabilities. Hence, the
FitPC2 is an option only if the computation required by your robot is not very high.

Conclusion:
The above single board computers did not perform very well and it may not be a
great option to spend much time on either if your project requirements are similar
to ours. There are other options such as the UDoo that comes with an Arduino board
integrated with the single board computer or the Raspberry Pi or the Beaglebone
and they may be great choices for projects that require light computation.
