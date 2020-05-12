---
date: 2020-04-10
title: ROS Global Planner
---
There are 3 global planners that adhere to `nav_core::BaseGlobalPlanner` interface: `global_planner`, `navfn` and `carrot_planner`. The `nav_core::BaseGlobalPlanner` provides an interface for global used in navigation.

## carrot_planner

The  `carrot_planner` is the simplest global planner, which makes the robot to get as close to a user-specified goal point as possible. The planner checks whether. the user-specified goal is an obstacle. If it is, it moves back along the vector between the robot and the goal. Otherwise, it passes the goal point as a plan to the local planner or controller (internally).

## navfn and global_planner

`navfn` and  `global_planner` are based on the work by Brock and Khatib, 1999[1]. The difference is that `navfn` uses Dijkstra's algorithm, while `global_planner` is computed with more flexible options, including:

1. support A* algorithm
2. toggling quadratic approximation
3. toggling grid path

Therefore, we generally use `global_planner` in our own project. Here, we will talk about some key parameters of it. The most frequently used parameters `allow_unkown`(true), `use_djikstra`(true), `use_quadratic`(true), `use_grid_path`(false), `old_navfn_behavior`(false) could be set to their default values shown in the brackets. If we want to visualize the potential map in RVIZ, it is helpful to change `visualize_potential`(false) to be true. Besides this, there are three parameters that highly impact the quality of the planned global path, since they have an effect on the heuristic of the algorithm. They are `cost_factor`, `neutral_cost`, and `lethal_cost`.

From the source code, we know that the cost values are set to
>cost = COST_NEUTRAL + COST_FACTOR * costmap_cost_value

The comments also mentions:
>With COST NEUTRAL of 50, the COST FACTOR needs to be about 0.8 to ensure the input values are spread evenly over the output range, 50 to 253. If COST FACTOR is higher, cost values will have a plateau around obstacles and the planner will then treat (for example) the whole width of a narrow hallway as equally undesirable and thus will not plan paths down the center.

It takes time to tune the parameters in order to have high quality global path. A suggestion is that you could set `cost_factor = 0.55`, `neutral_cost = 66`, and `lethal_cost = 253`, which results in a desirable global path. After that, you could tune the parameters as you want.

## References

[1] Brock, O. and Khatib, O. (1999). High-speed navigation using the global dynamic window approach. In Proceedings 1999 IEEE Interna- tional Conference on Robotics and Automation (Cat. No. 99CH36288C), volume 1, pages 341–346. IEEE.

[2] Zheng, K., 2020. ROS Navigation Tuning Guide. [online] arXiv.org. Available at: <https://arxiv.org/abs/1706.09068>.

[3] LU, D., 2020. Global_Planner - ROS Wiki. [online] Wiki.ros.org. Available at: <http://wiki.ros.org/global_planner?distro=melodic>.
