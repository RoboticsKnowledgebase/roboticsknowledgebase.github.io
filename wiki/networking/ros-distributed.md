---
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
