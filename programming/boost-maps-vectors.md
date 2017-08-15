---
title: Iterations in maps and vectors using boost libraries
---
Using Boost libraries for maps and vectors in C++ for ROS.

Maps are hash tables used to store data along with ids. For example if you are working on multiple robots and implement a navigation algorithm in which the controls of robot motion are dependent on robot pose. So one can create a map which maps robot ids to robot pose. This is how you do it:

``std::map<int,Eigen::Affine3d> Poses``

Now suppose you receive the pose of each of the robot, along with their ids from sensor data processing function or (an Apriltags function). Instead of iterating through the map using a for loop or stl iterators once, you can simply use boost library for `std::map`. Below is an example implementation.

```
#include <boost/foreach.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/range/adaptor/map.hpp>

BOOST_FOREACH(const int i, robot_ids | boost::adaptors::map_values)
{
first check if the robot id is valid
if( Poses.find(i)!=Poses.end())
{
update poses
Poses(i)= pose_received_from_function;
}
}
```
[Here is the link](http://cplusplus.bordoon.com/boost_foreach_techniques.html) to tutorial on how to use maps and boost libraries.  Boost libraries are recommended for those who  would use maps.
