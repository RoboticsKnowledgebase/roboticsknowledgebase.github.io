
/wiki/networking/bluetooth-sockets/
---
date: 2017-08-21
title: Bluetooth Socket Programming using Python PyBluez
---
This tutorial is intended at providing a primer into Bluetooth programming in general and getting started with PyBluez, a Python Bluetooth module. Bluetooth socket programming is similar to socket programming used by network developers for TCP/IP connections and familiarity with that paradigm will certainly aid understanding although it is not required at all.

## Important Terms and Definitions in Bluetooth Programming
1. Bluetooth address or device address.
  - Every Bluetooth chip that is manufactured is imprinted with a 48-bit address unique, referred to as the Bluetooth address or device address. This is identical in nature to the MAC addresses of Ethernet and is used to identify a particular Bluetooth device
- Device Name
  - It is a human readable name for a bluetooth device and is shown to the user instead of the Bluetooth address of the device. This is usually user configurable and can be the same for multiple devices. It is the responsibility of the client program to translate the name to the corresponding bluetooth addresses. TCP/IP has nameservers for doing so but in the case of Bluetooth the client program has to leverage the proximity of devices with user supplied names.
- Transport Protocol
  - In the layered networking architecture, the transport protocol is the layer actually responsible for creating data packets and ensuring synchronization, robustness and preventing loss of data. In internet programming, transport protocols used are TCP or UDP whereas in the case of Bluetooth the transport protocols most frequently used are RFCOMM and L2CAP.
- Port Numbers
  - Once the numerical address and transport protocol are known, the most important thing in Bluetooth programming is to choose a port number for communication. Without port numbers it would not be possible for multiple applications on the same host to utilize the same transport protocol. This concept is used in Internet Programming as well with different ports for HTTP, TCP etc. In L2CAP transport protocol, ports are known as Protocol Service Multiplexers. RFCOMM has channels numbered from 1-30 for use. In Bluetooth, the number of ports are ery limited as compared to Internet programming. Hence, in Bluetooth ports are assigned at runtime depending on application requirements. Clients know which port description they are looking for using a 128 bit number called Universally Unique Identifier (UUID) at design time using something called a Service Discovery Protocol.
- Bluetooth Profiles
  - Bluetooth is a short-range nature of Bluetooth, Bluetooth profiles are necessary for specific tasks. There are separate profiles for exchanging physical location information, a profile for printing to nearby printers and a profile for nearby modems to make phone calls. There is also a profile for reducing Bluetooth programming to Internet programming. An overview of all profiles available can be found [here.](http://www.bluetooth.org/spec)


## Bluetooth Transport Protocols Explained
There are different transport protocols used in Bluetooth, the most important of which are discussed below
1. RFCOMM
  - RFCOMM protocol is essentially similar to TCP and provides the same service and capabilities. It is quite simple to use in many scenarios and is mostly used in point to point applications. If a part of data is not delivered within a fixed amount of time, then the connection is terminated. On the application end, once the port number has been defined it is similar to serial port programming and all the same practices can be used as with any serial device. This is the most intuitive and easy part of socket programming.
2. L2CAP
  - This transport protocol is mostly used in situations where every packet is not crucial but speed of delivery is required. It is similar to UDP protocol and is used for its simplicity. It sends reliably data packets of fixed maximum length and is fairly customizable for varying levels of reliability. There are three policies an application can use:
    - never retransmit
    - retransmit until total connection failure (the default)
    - drop a packet and move on to queued data.


## Bluez stack and PyBluez
Now coming to the actual application part of the post. PyBluez is a C (known as Bluez) and Python library for Bluetooth socket programming. PyBluez is a Python extension module written in C that provides access to system Bluetooth resources in an object oriented, modular manner. Although it provides the same functionality in both languages, only Python based implementation is discussed here. PyBluez is available for Microsoft Windows (XP onwards) and GNU/Linux. Basic knowledge of Python is assumed in this tutorial and familiarity with Linux operating system as well.

Instructions for installing all the required libraries can be found on [here](htp://www.bluez.org), but installation on Linux is fairly simple through an apt repository. On the terminal one can simply type:

``apt-get install libbluetooth1-dev bluez-utils``

Following this PyBluez can be installed from their [PyBluez website.](http://org.csail.mit.edu/pybluez)
It is usually required to be installed from source.

#### General Steps in Bluetooth Programming using PyBluez
The general steps involved in Bluetooth programming are as follows:
- Choosing a device to communicate with
- Choosing transport protocol to be used
- Establishing an outgoing or incoming connection with encryption
- Sending/Receiving data

Using PyBluez these steps are abstracted to a very high level and application level programs are written almost in the same way as serial communication with a serial device. For most applications RFCOMM is used as the transport protocol and device names are pre-decided to avoid confusion.
In PyBluez, nearby devices for communication can be specified diectly in terms of the device name and hardware address is not required to be known.
Programming is done in terms of socket programming by forming BluetoothSocket objects. In the constructor for theses objects you can specify the transport protocol (RFCOMM/L2CAP) that you want to use. Following which you can use the connect method to simply connect with hte bluetooth device. By default all communication happens on RFCOMM port 1 but you can also dynamically choose ports. Error handling is also fairly straightforward. In case an operation fails a BluetoothError is raised which can be caught in a try catch loop in python.

One of the most attractive features of PyBluez isthe ease with which you can use Service Discovery Protocol. As stated above, service discovery protocol allows us to find at atrun time the available bluetooth ports (RFCOMM or L2CAP) with which we can connect. It does not rely on port numbers and/or device addresses being specified at design time. The advertise_service method advertises a service with a local SDP server and find_service method searches all Bluetooth devices for a specific service and does not require device ids. The get_available_port returns the first available port number for the specified protocol but does not reserve it so you have to bind to the port. Since here is a time delay between hte two, the availability might change and a BletoothException might be raised.
There are also advanced usage options such as Asynchronous device discovery which allows the program to return even when connection to a device has not been established. This function might prove to be very nifty in certain circumstances. Complete code for using PyBluez with RFCOMM sockets can be found at [here.](https://people.csail.mit.edu/albert/bluez-intro/x502.html)

### Bluetooth socket programming using Bash
In some applications, it is important to use Bahs to create nifty scripts for bluetooth communication. In our project we initially used this approach before shifting to Python as it was easily compatible with ROS. This information might be worthwhile for those who want to explore bash more and learn simple scripting.
Bash is an interpreted language similar to Python and Bahs scripts basically execute the commands one by one in the same way as you would enter them in the terminal. The first command to be used for serial communication is sdptool which alows you to add serial ports on different channels. For RFCOMM the channel used is 22. After this, using the rfcomm watch command, you can bind channel 22 to a dev port such as /dev/rfcomm0 22.
Using these simple commands you can listen for serial data on the rfcomm port. Now, when you pair your laptop with a bluetooth device such as an android phone and use a bluetooth emulator app such as BlueTerm, you should be able to serially send data to your laptop and 'echo' it to the screen or to a file stream.



## References
1. Complete Bluetooth specifications. Recommended for serious network programmers: http://www.bluetooth.org/spec
2. A must-read guide to Bluetooth programming by the creators of PyBluez themselves: https://people.csail.mit.edu/albert/bluez-intro/c33.html\
3. Another approach in Windows is to use Windows Socket for Bluetooth programming: https://msdn.microsoft.com/en-us/library/windows/desktop/aa362928%28v=vs.85%29.aspx
4. Java-based alternatives for PyBluez:
  - Rocosoft Impronto: http://www.rocosoft.com
  - Java Bluetooth:http://www.javabluetooth.org
5. Bash programming tutorials uploaded by tutoriallinux on YouTube:
https://www.youtube.com/watch?v=xtS2NiABf54


/wiki/networking/rocon-multi-master/
---
date: 2017-08-21
title: ROCON Multi-master Framework
---
ROCON stands for *Robotics in Concert*. It is a multi-master framework provided by ROS. It provides easy solutions for multi-robot/device/tablet platforms.
Multimaster is primarily a means for connecting autonomously capable ROS subsystems. It could of course all be done underneath a single master - but the key point is for subsystems to be autonomously independent. This is important for wirelessly connected robots that can’t always guarantee their connection on the network.

## The Appable Robot
The appable robot is intended to be a complete framework intended to simplify:
- Software Installation
- Launching
- Retasking
- Connectivity (pairing or multimaster modes)
- Writing portable software

It also provides useful means of interacting with the robot over the public interface via two different modes:
1. **Pairing Mode:** 1-1 human-robot configuration and interaction.
2. **Concert Mode:** autonomous control of a robot through a concert solution.

## Rapps
ROS Indigo has introduced the concept of Rapps for setting the multi-master system using ROCON. Rapps are meta packages providing configuration, launching rules and install dependencies. These are essentially launchers with no code.
- The specifications and parameters for rapps can be found at in the [Rapp Specifications](http://cmumrsdproject.wikispaces.com/link)
- [A guide](http://cmumrsdproject.wikispaces.com/Creating+Rapps) to creating rapps and working with them is also available.

## The Gateway Model
The gateway model is the engine of a ROCON Multimaster system. They are used by the Rapp Manager and the concert level components to coordinate the exchange of ROS topics, services and actions between masters. The gateway model is based on the LAN concept where a gateway stands in between your LAN and the rest of the internet, controlling both what communications are allowed through, and what is sent out. Gateways for ROS systems are similar conceptually. They interpose themselves between a ROS system and other ROS systems and are the coordinators of traffic going in and out of a ROS system to remote ROS systems.
A hub acts as a shared key-value store for multiple ROS systems. Further detailed information on how the topics are shared between the multiple masters can be found [here](http://cmumrsdproject.wikispaces.com/The+Gateway+Model).


/wiki/networking/ros-distributed/
---
date: 2017-08-21
title: Running ROS over Multiple Machines
---
Multi­robot systems require intercommunication between processors running on different
computers. Depending on the data to be exchanged between machines, various means of
communication can be used.
- **Option 1:** Create Python server on one machine and clients on the rest and use python for communication on a locally established network. You can transfer files which contain the desired information.
- **Option 2:** Establish ROS communication between systems with one computer running the ROS master and other computers connecting to the ROS master via the same local network.

Python communication requires the exchange of files and hence, files are created and deleted every time data is communicated. This makes the system slow and inefficient. Hence, communication over ROS is a better option.

## Implementation
Steps to be followed:
1. Connect all computers to the same local network and get the IP addresses of all the
computers.
2. Ensure that all computers have the same version of ROS installed. (Different versions of
ROS using the ROS master of one of the versions may cause problems in running few
packages due to version/ message type/ dependency incompatibilities)
3. Edit your bashrc file and add the following two lines at the end

```
export ROS_MASTER_URI=http://<IP address of the computer running the ROS master (server)>:11311
export ROS_IP= <IP address of the current computer>
```

  - Link: http://wiki.ros.org/ROS/Tutorials/MultipleMachines

4. The computers are now connected and this can be tested by running roscore on the server
laptop and trying to run packages such as ‘rviz’ (rosrun rviz rviz) on other laptops.
5. Though the process may seem complete now, there are certain issues that need to be fixed.
  - The clocks of the computers may not be synchronized and it may cause the system to have
latency while communicating data. Check the latency of a client laptop using the following
command: `sudo ntpdate <IP address of ROS server>`
  - This command will show the latency and if it is not under 0.5 second, follow steps 6­ & 7. Important Link: http://wiki.ros.org/ROS/NetworkSetup
6. To solve this problem, look at Network Time Protocol Link: http://www.ntp.org/.
  - Install ‘chrony’ on one machine and other machine as the server: `sudo apt­get install chrony`
  - Also put this line in the file `/etc/chrony/chrony.conf`:
  ```
  server c1 minpoll 0 maxpoll 5 maxdelay .0005
  ```
  - You should restart your laptop after adding the above line to the file because it tries syncing the clocks during the boot.
7. To synchronize the clocks, run the following commands (run this only if necessary)
```
sudo /etc/init.d/chrony stop
sudo /etc/init.d/chrony start
```
  - Check the latency again by using the command in step 5. The latency should have reduced to under 0.1 seconds
8. ROS over multiple machines is now ready to be used

## Important Tips
1. If clients communicating to the same ROS master are publishing/subscribing to the same
topics, there may be a namespace clash and the computers will not be able to distinguish one topic from the other. Link: http://nootrix.com/2013/08/ros­namespaces/
Hence, make the topics specific to that particular machine. For example, `/cmd_vel` topic for robot 1 should now become `/robot1/cmd_vel` and for robot 2, it should be `/robot2/cmd_vel`. Also, namespaces should be added to the frames also. For three robots, transformation tree should look like this:
![Transofromation Tree for Three Robots](assets/ROSDistributed-1b70c.png)
  - Similarly, there may also be clashing transforms that may cause problems and they need to be fixed in the same way.
2. Try not to run any graphics such as Rviz or RQt on the clients while communicating as they consume a lot of bandwidth and may cause the system to slow down.
All the computers can be controlled from the server laptop for any commands that need to run on them by performing ‘ssh’ into the client laptop from the server laptop.
3. A low quality router might also be a reason of low communication speed. Getting a better
quality router will help solving the latency and timing issues.

## Further Reading
For any other guidance, look at the links given below:
1. http://wiki.ros.org/ROS/NetworkSetup
2. http://wiki.ros.org/ROS/Tutorials/MultipleMachines
3. http://www.ntp.org/
4. http://nootrix.com/2013/08/ros­-namespaces


/wiki/networking/wifi-hotspot/
---
date: 2019-11-11
title: Setting up WiFi Hotspot at the Boot up for Linux Devices 
---

Most of the mobile robot platform uses Linux based Single board computer for onboard computation and these SBCs typically have WiFi or an external WiFi dongle can also be attached. While testing/debugging we need to continuously monitor the performance of the robot which makes it very important to have a remote connection with our robot. So, in this tutorial, we will help you in setting up the WiFi hotspot at the boot for Linux devices like Nvidia Jetson and Intel NUC. We will start with the procedure to set up the WiFi hotspot and then we will discuss how to change Wi-Fi hotspot settings in Ubuntu 18.04 to start it at bootup.

# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Create a WiFi hotspot in Ubuntu 18.04](#create-a-wifi-hotspot-in-ubuntu-1804)
  - [Edit WiFi hotspot settings in Ubuntu 18.04](#edit-wifi-hotspot-settings-in-ubuntu-1804)
      - [Option 1: Edit the hotspot configuration file.](#option-1-edit-the-hotspot-configuration-file)
      - [Option 2: NM Connection Editor.](#option-2-nm-connection-editor)
  - [/wiki/networking/xbee-pro-digimesh-900/](#wikinetworkingxbee-pro-digimesh-900)
  - [title: XBee Pro DigiMesh 900](#title-xbee-pro-digimesh-900)
  - [Configuring the Modules:](#configuring-the-modules)
  - [Using Python for Serial Communication](#using-python-for-serial-communication)
  - [Resources](#resources)

## Create a WiFi hotspot in Ubuntu 18.04
This section will help you in setting up the WiFi hotspot at the boot for Linux devices. 
1. To create a Wi-Fi hotspot, the first turn on the Wi-Fi and select Wi-Fi settings from the system Wi-Fi menu.
2. In the Wi-Fi settings window, click on the menu icon from the window upper right-hand side corner, and select turn On Wi-Fi hotspot.
3. A new Wi-Fi hotspot always uses AP mode in Ubuntu 18.04, and the network SSID and password, as well as the security type (WPA/WPA2 is used by default in Ubuntu 18.04),  are presented on the next screen which is displayed immediately after enabling the Wi-Fi hotspot.

If you are ok with the defaults and don't want to change anything, that's all you have to do to create a Wi-Fi hotspot in Ubuntu 18.04.

## Edit WiFi hotspot settings in Ubuntu 18.04
There are two ways to edit hotspot settings (like the network SSID and password) which will be discussed in this section.
#### Option 1: Edit the hotspot configuration file. 
1. After creating a hotspot for the first time, a file called hotspot is created which holds some settings. So make sure to create a hotspot first or else this file does not exist. 
2. In this file you can configure the network SSID it appears as ```ssid = ``` under ```[wifi]```), the Wi-Fi password is the value of ```psk=``` under ```[wifi-security]```), and other settings as required.
```
sudo nano /etc/NetworkManager/system-connections/Hotspot
```
3. In the same file set ```autoconnect=false``` to set up the hotspot at bootup automatically.
4. After making changes to the hotspot configuration file you'll need to restart Network Manager:
```
sudo systemctl restart NetworkManager
```

#### Option 2: NM Connection Editor.
NM connection editor also allows you to modify the hotspot Wi-Fi mode, band etc. It can be started by pressing ```Alt + F2``` or using this command:
```
nm-connection-editor
```
All changes can be made directly in the nm-connection-editor in its corresponding tab. After making any changes using nm-connection-editor, you'll need to restart Network Manager.
```
sudo systemctl restart NetworkManager
```

To make sure all settings are preserved, start a hotspot by selecting turn On Wi-Fi Hotspot from the Wi-Fi System Settings once. Use the Connect to Hidden Network option for subsequent uses, then select the connection named Hotspot and click Connect.


/wiki/networking/xbee-pro-digimesh-900/
---
date: 2017-08-21
title: XBee Pro DigiMesh 900
---
XBee-PRO 900 DigiMesh is useful for setting up your own low bandwidth mesh network. These adapters are great for Mesh Networks and implementations of networks such as VANETs. This tutorial is intended to help you with configuring these adapters and writing a simple python script to initiate bi-directional communication on a serial device.

## Configuring the Modules:

> IMPORTANT: The latest versions of X-CTU support Linux so don't follow the older guides found online which require 'wine' installation on Linux to run it.

Firstly, download and install X-CTU. [The official guide from DIGI](https://docs.digi.com/display/XCTU/Download+and+install+XCTU) will walk you through this process.

Once you are done with this, plug in your XBee adapter and launch X-CTU. The device should get detected automatically and you'll be presented with a screen similar to the one seen below.
![X-CTU Screen](assets/XbeeProDigiMesh900-1fc56.png)

Now it is important to note that this is not an ordinary XBee adapter which is why you'll see many more options than usual. Firstly, all your devices should be have same Modem VID (ID) and Hopping Channel (HP) for them to communicate. Now, further settings will depend on your individual requirements but just to explain some important parameters:
- **Multi-Transmit (MT):** To set/read number of additional broadcast re-transmissions. All broadcast packets are transmitted MT+1 times to ensure it is received.
- **Broadcast Radius (BR):** To set/read the transmission radius for broadcast data transmissions. Set to 0 for maximum radius. If BR is set greater than Network Hops then the value of Network Hops is used.
- **Mesh Retries (MR):** To set/read maximum number of network packet delivery attempts. If MR is non-zero, packets sent will request a network acknowledgement, and can be resent up to MR times if no acknowledgements are received.

Once you are done configuring your XBee PRO adapter as per your mesh network configurations, you are now ready to start using them.

## Using Python for Serial Communication
To carry out serial communication using Python, first we need to install pySerial.
Skip to Step 2 if you already have pip installed.
1. `sudo apt-get install python-pip`
2. `sudo pip install pyserial`

Now, to have bi-directional or full duplex communication, we'll have to use threads which ensure that read and write can happen simultaneously.
Here is a small python script which does that:
```
import serial
from threading import Thread

ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)

def read_serial():
    while True:
        if ser.inWaiting() > 0:
            print ser.readline()

def write_serial():
    while True:
        x = input("Enter: ")
        ser.write(x)

t1 = Thread(target=read_serial)
t2 = Thread(target=write_serial)
t1.start()
t2.start()
```

To fix the issue of having to set the port manually every time or hard-coding it, you can add the lines below in your python script and then call the function connect, as provided below, to automatically detect the USB port. This will only work for one XBee adapter connected to the computer.
```
import subprocess

def connect():
    global ser, flag
    x = subprocess.check_output("ls /dev/serial/by-id", shell = True).split()
    if x[0].find("Digi") != -1:
        y = subprocess.check_output("dmesg | grep tty", shell = True).split()
        indices = [i for i, x in enumerate(y) if x == "FTDI"]
        address = "/dev/" + y[indices[-1]+8]
        ser = serial.Serial(address, 9600, timeout = 1)
        ser.flushInput()
        ser.flushOutput()
        flag = 0
    else:
        raise ValueError
```

Now you have configured the adapters and written a basic Python Script to carry out serial communication on a Mesh Network.

For advanced usage, look in to:
1. pySerial API: http://pyserial.readthedocs.org/en/latest/pyserial_api.html
 - This library has many more advanced functionalities to configure the port being used. The example code snippets provided on the website serve as a good starting point.
2. Threading API: https://docs.python.org/2/library/threading.html
- This library has many advanced functions to control the operation of a thread. One thing to look in particular is thread lock which is useful while operating on shared data structures between threads.

## Resources
1. To compare the DigiMesh architecture to ZigBee mesh architecture, you can refer to [this guide](http://www.digi.com/pdf/wp_zigbeevsdigimesh.pdf) which points out the pros and cons of both.
- For detailed information, please refer to the [User Guide](http://ftp1.digi.com/support/documentation/90000903_G.pdf). The guide has details regarding technical specifications, internal and overall communication architecture, command reference tables, etc.
