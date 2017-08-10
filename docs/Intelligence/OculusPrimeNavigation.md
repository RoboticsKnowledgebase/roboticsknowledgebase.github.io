# Oculus Prime Navigation

There are multiple techniques to carry out way-point navigation on the Oculus Prime platform using the navigation stack.

1. **"Pure" ROS:** To set a single goal the pure ROS way, bypassing the use of the oculusprime browser UI altogether, first run:
```
$ roslaunch oculusprime globalpath_follow.launch
```
Or try:
```
$ roslaunch oculusprime remote_nav.launch
```
  - This `globalpath_follow` launch file sets up the basic nodes necessary to have Oculus Prime go where ROS Navigation wants it to go, and launches the rest of the navigation stack. Once it’s running, you can [set initial position and goals graphically using Rviz](http://wiki.ros.org/oculusprime_ros/navigation_rviz_tutorial)

2. **Via Command Line:** You can also send coordinates via command line (AFTER you set initial position using the web browser or Rviz map). Enter a command similar to:
```
$ rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \
'{ header: { frame_id: "map" }, pose: { position: { x: 4.797, y: 2.962, z: 0 }, orientation: { x: 0, y: 0, z: 0.999961751128, w: -0.00874621528223 } } }'
```
  - Change the coordinates shown to match where your goal is. An easy way to find goal coordinates is to set the goal on the map with a mouse, and while the robot is driving there, enter:
```
$ rostopic echo /move_base/current_goal
```
  - An example of doing this via a python ROS node, starting with simpler coordinates (x,y,th), can be found [here](https://gist.github.com/xaxxontech/6cbfefd38208b9f8b153).

3. **Waypoints:** If you want to choose from a list of waypoints instead, you can use the functionality built into the oculusprime server and do it via oculusprime commands:
  - First read the waypoints:
```
state rosmapwaypoints
```
  - This should return a long comma-separated string with no spaces using the format "name,x,y,th," for all the waypoints, similar to `<telnet> <state> waypointA, 4.387, -0.858, -0.3218, waypointB, 2.081, 2.739, -1.5103`.
  - Change the coordinates of the particular waypoint you want to change within the string, then send the whole string to the savewaypoints command, eg:
Advanced -> telnet text command -> enter command:
```
savewaypoints waypointA,5.456,-2.345,-0.3218,waypointB,2.081,2.739,-1.5103
```
  - Then drive to the new coordinates by sending:
```
gotowaypoint waypointA
```
