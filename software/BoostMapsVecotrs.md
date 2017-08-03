Iterations in maps and vectors using boost libraries
Using boost libraries for maps and vectors in c++ for ros.

Maps are hash tables used to store data along with ids. For example if you are working on multiple robots and implement a navigation algorithm in which the controls of robot motion are dependent on robot pose. So one can create a map which maps robot ids to robot pose.This is how you do it

std::map<int,Eigen::Affine3d> Poses

Now suppose you receive the pose of each of the robot along with their ids from sensor data processing function or Apriltags function. Instead of iterating through the map using for loop or stl iterators once can simply use boost library for std::map
Define the required libraries

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


Here is the link to tutorial on how to use maps and boost libraries
__http://cplusplus.bordoon.com/boost_foreach_techniques.html__
__www.boost.org__

I would recommend people to use boost libraries if they use maps
