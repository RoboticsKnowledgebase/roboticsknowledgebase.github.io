---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-12-03 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Cloud Robotics with ROS - Latency, Network Bandwidth, and Time-Synchronization
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
Cloud robotics is an emerging field that combines robotics with cloud computing and services. While it offers many advantages, it also comes with several key challenges. Some of these are latency, network bandwidth and time-synchronization for which we discuss options in this article and include examples on addressing these in a robotics project setup. As a preclude we provide a concise introduction to cloud robotics, emphasizing the latest research trends and enabling technologies. We will then delve into the specifics of building a cloud robotics testbed, which includes setting up a communication network. We discuss trade-offs and offer guidance on configuring a WLAN network. We present an overview of popular methods and trade-offs in time-synchronization techniques. We will explain on how to use Chrony to synchronize time across distributed systems in the absence of internet connectivity, timestamps from GPS receivers and real-time clocks. Simulating latency is another crucial consideration in cloud robotics testing. We will provide a primer
on common incorrect methods for simulating communication latency and offer a simple implementation of simulating latency in real-time systems
deployed with ROS Noetic. 

## Overview of Cloud Robotics
Cloud robotics represents a paradigm shift in the field of robotics, where the traditional model of integrating sensing and computation within a single robotic system gives way to a distributed architecture with an extended or shared brain. In this approach, robots become compact, consume less power, and also cost-effective by offloading the bulk of processing tasks to the cloud. The core idea is to leverage on-demand computing resources offered by cloud infrastructure. This architecture also enables human operators to remotely teleoperate robots, enhancing flexibility and scalability in various applications. Moreover, the architecture facilitates information sharing across the entire system, fostering collective learning and optimization for multi-agent use cases. Achieving these advantages introduces novel research challenges, such as efficiently allocating parallel computation in the cloud at a reasonable cost, addressing inevitable bottlenecks associated with network connectivity, and devising algorithms capable of distributing computation tasks between the cloud and the robot.

## Configuring a WLAN network
When choosing a WiFi router, you’ll have to decide whether you want a Wi-Fi 6 router, a Wi-Fi 5 (802.11ac) router, or a Wi-Fi mesh system. While there are specific steps to configuring different Wi-Fi routers depending on the manufacturer, the below are some important points to keep in mind.

1. **Frequency:** A 5GHz WLAN network offers several advantages. It provides better performance at short ranges compared to 2.4GHz. The 5GHz band, though somewhat faster, has shorter wavelengths, limiting its range and penetration through objects. However, this limitation becomes an advantage in terms of reduced interference and higher bandwidth. The 2.4GHz band, while traveling further, often faces more congestion and has fewer channel options. Therefore, opting for a 5GHz WLAN network results in higher bandwidth, less cross-network interference, but with the drawback of a relatively smaller range, making it suitable for environments with shorter distances between devices.

2. **DHCP settings to configure static IPs:** Configuring static IPs in a DHCP environment for robotics test beds involves assigning fixed IP addresses to specific devices. This ensures that the devices always have the same IP address, facilitating consistent and predictable communication. Locate the DHCP settings in the router's configuration menu. This is often found under a section like "LAN Settings" or "Network Setup." Look for an option related to reserved IP addresses or static IP assignments. This feature allows you to specify which devices will receive static IPs. Assign a static IP address to each device by associating the device's MAC address with the desired IP address. The MAC address is a unique identifier for each network interface.

3. **QoS settings:** Quality of Service is a set of technologies and policies that prioritize certain types of network traffic over others. We can use QoS settings to assign priority levels to different types of traffic. In the context of robotics applications, it ensures that critical data, such as sensor readings and control commands, receives higher priority over less time-sensitive traffic like file downloads or streaming. For example, give the highest priority to real-time communication data. Navigate to the QoS settings section. This may be labeled as "QoS," "Traffic Control," or a similar term.
Configure rules to prioritize traffic based on application or device. Some routers allow you to set priorities for specific devices or applications.

### Setting up the TP-Link - Archer AX11000 Tri-Band Wi-Fi 6 Router

1. Power on the router  

2. Insert a pin to press the reset button behind the router to restore it to factory settings 

3. Using the TP-link tether mobile app (bluetooth and location must be enbaled on your phone) 

4. From the mobile app 

    4.1 Add a new device → Gaming router → Archer AX10000 

    4.2 Choose static IP (**also used to access the router's web-based interface; usually done by entering the router's IP address in a web browser**)
    
    4.3 Other configurations to be decieded

      Internet Connection Type: This can be various types such as Dynamic IP (DHCP), Static IP, PPPoE, etc. Dynamic IP is the most common, where the router obtains its IP address automatically from the ISP.

      Subnet Mask: This is a 32-bit number that segments an IP address into network and host portions. It is used to divide an IP address into network and host portions. A common subnet mask is 255.255.255.0.

      Default Gateway: This is the IP address of the router that connects your local network to the internet. It is the device that forwards traffic from your local network to the wider internet.

      Primary DNS (Domain Name System): DNS translates domain names into IP addresses. The primary DNS is the first server your router will use to look up domain names. It is typically provided by your ISP, but you can use public DNS servers like those provided by Google (8.8.8.8) or Cloudflare (1.1.1.1).

    4.3 Set-up a 5GHz network; configure the SSID and password


## Time-synchronization across distributed systems
Achieving precise time synchronization is paramount in cloud robotics, where seamless operation relies on coordinated efforts among distributed systems. For instance, drawing correlations from sensor data based on their timestamps requires accurate time alignment. This section delves into various methods and tools designed to ensure effective time synchronization across robotic systems and the cloud.

### RTCs, GPS Time, NTP and PTP
RTC typically stands for Real-Time Clock, and it is a component that keeps track of the current time and date, even when the system is powered off.To add an RTC to a project, you would typically connect the RTC module to the SBC through the appropriate interfaces (e.g., I2C or SPI) and configure the system to use the RTC for timekeeping. 

For a common measure of time across systems GPS time could be an option. GPS time is a global time standard provided by the GPS satellite system. However GPS signals are weak in certain environments and using multiple GPS receivers can be expensive.

Network Time Protocol (NTP) is an internet protocol used to synchronize with computer clock time sources in a network. The NTP client initiates a time-request exchange with the NTP server.The client is then able to calculate the link delay and its local offset and adjust its local clock to match the clock at the server's computer. It can achieve better than one millisecond accuracy in local area networks under ideal conditions. Chrony is a popular implementation of NTP. 

PTP stands for Precision Time Protocol. It is a protocol used to synchronize clocks between devices in a network with high precision. The goal of PTP is to achieve clock synchronization accuracy in the sub-microsecond or even nanosecond range. A PTP network only one GPS receiver is needed, which is less expensive than giving each network node its own GPS receiver. 

### Configuring Chrony to time-synchronize distributed devcies

Computers worldwide use the Network Time Protocol (NTP) to synchronize their times with internet standard reference clocks via a hierarchy of NTP servers. In this section, we will look at configuring Chrony to synchronize devices on the same network in the absence of internet and GPS clocks, optionally using RTCs if available. It is done by configuring one machine as the master clock (primary timeserver) and making all other devices track the master clock. We set the master clock as the device which hosts the ROS master in our [multi-machine ROS](https://wiki.ros.org/ROS/Tutorials/MultipleMachines) setup. 

1. Download Chrony using your package manager (eg. sudo apt-get install chrony)
2. Edit the Chrony configuration file (eg. sudo nano /etc/chrony/chrony.conf)
3. Restart the Chrony deamon (sudo systemctl restart chronyd)

#### Configuring the Chrony server 
```
#/etc/chrony/chrony.conf
local stratum 8
manual 
allow 192.168
driftfile /var/lib/chrony/chrony.drift
logdir /var/log/chrony
#rtcsync
```

The local directive enables a local reference mode, which allows chronyd to appear synchronised even when it is not. The manual directive puts Chrony in manual mode. In this mode, the system clock is not adjusted automatically by Chrony. The allow directive is used to designate a particular subnet from which NTP clients are allowed to access the computer as an NTP server. The use of a driftfile helps Chrony maintain more accurate time synchronization by storing the estimated clock drift. This information is valuable for adjusting the clock over time. Configure Chrony to log tracking information, including measurements and statistics. The log will be stored in the default location or as specified in the configuration. To enable kernel synchronization of the real-time clock include the rtsync directive. The real-time clock synchronization occurs approximately every 11 minutes.

#### Configuring the Chrony clients
```
#/etc/chrony/chrony.conf
server 192.168.108.215 iburst 
driftfile /var/lib/chrony/chrony.drift
log tracking measurements statistics
logdir /var/log/chrony
#rtcsync
makestep 1 3
```
Specify an NTP server with its IP address for time synchronization. The iburst option is used to send a burst of eight packets instead of the usual one when initially querying the server, which can help quickly establish a synchronization.Replace the IP address above to that of your configured Chrony server.

Specify the path to the file where Chrony will store the estimated clock drift. The clock drift is the amount by which the system clock gains or loses time relative to the reference clock.

The final directive specifies that if the adjustment needed to synchronize the clock is larger than one second, the system clock will be stepped instead of being slewed. However, this only happens in the first three clock updates. Having this bound is critical as sudden jumps in time can disrupt processes runnning on the system. 

#### Useful commands 
```
#Restarting the Chrony daemon and service
sudo systemctl restart chronyd
sudo systemctl restart chrony.service

### To execute on client device(s) ###

# Display information about the system clock tracking
chronyc tracking

# Display statistics about the NTP sources
chronyc sourcestats

# Display a list of information about the current NTP sources
chronyc sources

# Step the system clock immediately, bypassing any slewing
# This command is used to make an immediate adjustment to the system clock
sudo chronyc -a makestep


### To execute on the server device ###

# Display a list of clients that have connected to the Chrony server
sudo chronyc clients

```

## Simulating latency
Simulating latency can be tricky. To artificially inject latency in a system by delaying the published ROS messages one must ensure that both the rate at which the message is published and the timestamp are preserved after injecting the latency; and the messages must be available to the subsricriber(s) after the desired delay. This can be done using timer-based callbacks in ROS, and the below implementation preserves the message frequency and timestamp.

```
#!/usr/bin/env python
import rospy
import functools
class LatencySim():
    def __init__(self, inTopic: str, outTopic:str, latencySecs: int, dtype) -> None:
        self.latency = latencySecs
        self.subscriber = rospy.Subscriber(inTopic, dtype, self.callback)
        self.publisher = rospy.Publisher(outTopic, dtype,  queue_size=50)

    def setLatency(self, newLatency): 
        self.latency = newLatency

    def callback(self, msg):
        # rospy.loginfo("  Received data from %f", msg.header.stamp.to_sec())
        if self.latency == 0: 
            self.publisher.publish(msg)
            return
        rospy.Timer(rospy.Duration(self.latency), functools.partial(self.delayedCallback, msg), oneshot=True)
      
    def delayedCallback(self, msg, event):
        # rospy.loginfo("Publishing data from %f", msg.header.stamp.to_sec())
        self.publisher.publish(msg)
```

More features can be added, such as 1) randomly varying latency to simulate varying connectivity strengths, and 2) latency being adjusted using dynamically reconfigurable ROS parameters.
## Summary
Cloud robotics, the integration of robotics with cloud computing and services, presents numerous advantages and challenges. In this article, we explore key challenges such as latency, network bandwidth, and time synchronization in the context of cloud robotics projects. The discussion includes practical examples and solutions for addressing these challenges within a robotics project setup.

The overview of cloud robotics highlights its paradigm shift, emphasizing the distributed architecture's benefits in terms of power efficiency, cost-effectiveness, and remote operability. We delve into building a cloud robotics testbed, focusing on configuring a WLAN network. Key considerations include frequency selection, DHCP settings for static IPs, and Quality of Service (QoS) settings to prioritize critical data.

Configuring a WLAN network is exemplified using the TP-Link Archer AX11000 Tri-Band Wi-Fi 6 Router, detailing steps such as power-on, reset, and mobile app configuration for static IPs and 5GHz network setup.

The importance of time synchronization in distributed systems is explored, covering methods like Real-Time Clocks (RTCs), GPS time, Network Time Protocol (NTP), and Precision Time Protocol (PTP). The configuration of Chrony for time synchronization in a local network is detailed, distinguishing between Chrony server and client configurations. 

To simulate latency in cloud robotics testing, a Python implementation using ROS (Robot Operating System) is provided. The LatencySim class utilizes timer-based callbacks to introduce artificial communication delays while preserving message frequency and timestamps. 

## See Also:
- [ROS Distributed Networking](https://roboticsknowledgebase.com/wiki/networking/ros-distributed/)
- [Rocon Multi-Master](https://roboticsknowledgebase.com/wiki/networking/rocon-multi-master/)
- [WiFi Hotspot](https://roboticsknowledgebase.com/wiki/networking/wifi-hotspot/)
- [ROSlibjs](https://roboticsknowledgebase.com/wiki/tools/roslibjs/)

## Further Reading
- J. Ichnowski, J. Prins, and R. Alterovitz, "The Economic Case for Cloud-Based Computation for Robot Motion Planning," in Robotics Research, N. Amato, G. Hager, S. Thomas, and M. Torres-Torriti, Eds. Springer, vol. 10, pp. 113-126, 2020. doi: 10.1007/978-3-030-28619-4_8.

- J. Ichnowski et al., "FogROS2: An Adaptive Platform for Cloud and Fog Robotics Using ROS 2," arXiv preprint arXiv:2205.09778, May 2023.


- L. Fortuna, J. Abdulsaheb, and D. Kadhim, "Real-Time SLAM Mobile Robot and Navigation Based on Cloud-Based Implementation," *Journal of Robotics*, vol. 2023, Article ID 9967236, pp. 1-11, Mar. 29, 2023. DOI: 10.1155/2023/9967236.


- [Comparison: NTP vs PTP](https://www.masterclock.com/network-timing-technology-ntp-vs-ptp.html?empcnm=NTP&empanm=&empcid=19563161695&empaid=148078419991&emptid=kwd-319140304311&emploc=9005929&empmtp=p&empnet=g&empdev=c&empdmo=&empctv=649545832934&empkwd=ptp%20vs%20ntp&emppos=&gad_source=1&gclid=Cj0KCQiA67CrBhC1ARIsACKAa8RMVMM8T5bTinz0NKh8bjB3fxbaSTrGneVy3fUVVLLVQMSTbaYNau8aAqDaEALw_wcB)
- [Discussion on Hacker News](https://news.ycombinator.com/item?id=28108586)
- [How to Install and Use Chrony in Linux](https://www.geeksforgeeks.org/how-to-install-and-use-chrony-in-linux/)
- [Ubuntu Documentation on Network Time Protocol](https://ubuntu.com/server/docs/network-ntp)
- [Chrony Configuration in Isolated Networks](https://chrony-project.org/doc/4.4/chrony.conf.html#_isolated_networks)

## References

- [How to Set Up and Optimize Your Wireless Router for the Best Wi-Fi Performance](https://www.pcmag.com/how-to/how-to-set-up-and-optimize-your-wireless-router-for-the-best-wi-fi-performance/)
- [What is Cloud Robotics?](https://www.geeksforgeeks.org/what-is-cloud-robotics/)
- [10 Tips to Help Improve Your Wireless Network](https://support.microsoft.com/en-us/topic/10-tips-to-help-improve-your-wireless-network-d28bf4e4-cf8c-e66f-efab-4b098459f298/)
- [PTP Overview](https://www.iol.unh.edu/sites/default/files/knowledgebase/1588/ptp_overview.pdf)
- [Network Time Protocol](https://en.wikipedia.org/wiki/Network_Time_Protocol)
- [Precision Time Protocol](https://en.wikipedia.org/wiki/Precision_Time_Protocol)
- [Chrony Project](https://chrony-project.org/)
- [Delay in Incoming Messages](https://robotics.stackexchange.com/questions/77409/delay-incoming-messages)


