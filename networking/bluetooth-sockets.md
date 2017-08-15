---
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
