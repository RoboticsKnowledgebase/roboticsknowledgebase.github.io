---
date: {}
title: Extensions To A* for Dynamic Planning
published: true
---
## Introduction
This wiki aims to build on the already excellent A* implementation guide by providing information about different planning approaches to counter the pitfalls of standard A* when applied to physical robots. In the scope of this article are two popular extensions - Theta Star and Safe Interval Path Planning (SIPP) that each have their strengths and weaknesses.

Theta Star planning is a pathfinding algorithm that is an extension of the popular A* algorithm. The key difference between Theta Star and A* is that Theta Star uses a line-of-sight check to reduce the number of nodes that need to be explored during pathfinding. The line-of-sight check is a simple test to determine if a path between two points is blocked by any obstacles. If there is a clear line of sight, the path is considered free and the algorithm can move on to the next node. If there is an obstacle in the way, the algorithm needs to generate a new path around it. 

Safe interval path planning is a method of planning robot motion that ensures collision avoidance and safety. It involves dividing the robot's workspace into safe intervals or regions, where the robot can move without colliding with obstacles. The path planning algorithm then generates a safe path by connecting these intervals. This approach ensures that the robot's motion is safe and collision-free. Safe interval path planning is essential for applications where robots are required to work in close proximity to humans or other objects.

## Overview of Theta* Algorithm

Theta* algorithm is a pathfinding algorithm that is an extension of the A* algorithm. It is used to find the shortest path between two points in a 2D grid-based environment.

### Important Terms
Here are the key steps and important terms involved in the Theta* algorithm:
- **Grid Map**: A 2D grid-based environment that represents the space in which the robot moves. Each cell in the grid represents a possible position that the robot can occupy.

- **Heuristics**: A function used to estimate the distance between two points in the grid map. The heuristic function used in Theta* is often the Euclidean distance or Manhattan distance. For even better results, it is recommended to use a pre-computed Dijkstra map since it accounts for static obstacles if the map is known

- **Open List**: A list of nodes that need to be explored during pathfinding. Each node in the list represents a position in the grid map and its associated cost.

- **Closed List**: A list of nodes that have already been explored during pathfinding.

- **Start and Goal Nodes**: The starting and ending points in the grid map between which the shortest path needs to be found.

- **Line of Sight Check**: A test that determines if there is a clear path between two points. If a clear path exists, the algorithm can "shortcut" between two nodes, reducing the number of nodes that need to be explored. In practice, this can be implemented as using Bresenham's line drawing equation borrowed from Computer Graphics to approximate lines on 2D grids, with possible speed ups.

Theta* should always give paths with lower cost than those from standard A*, since the line-of-sight check can only shorten paths. In practice, the paths tend to mimic visibility graphs around obstacles

Node Exploration Behaviour of Theta*           |  Path Comparison 
:-------------------------:|:-------------------------:
![Node Exploration Behaviour of Theta Star]({{site.baseurl}}/assets/images/theta_star_explor.png)  |  ![theta_vs_astar.png]({{site.baseurl}}/assets/images/theta_vs_astar.png)





### Algorithm: Key Steps

The underlying algorithm for Theta* only differs from A* in one key step - the line of sight check to shorten the underlying paths. A simple outline is as follows:

- Initialize the start node and add it to the open list.

- While the open list is not empty, select the node with the lowest cost from the open list.

- If the selected node is the goal node, terminate the algorithm and return the path.

- Otherwise, expand the selected node by generating its neighboring nodes.

- For each neighbor, check if a line of sight exists between the selected node and the neighbor. If it does, compute the cost of the path using the line-of-sight shortcut. Otherwise, compute the cost of the path without the shortcut.

- Update the neighbor's cost and parent if a better path has been found.

- Add the neighbor to the open list if it is not already on the list.

- Move the selected node to the closed list.

Once this process concludes, the path is traced back by recursively looking up the parents starting from the goal node back to the start node. This will carry all the neat theoretical guarantees of A* such as optimality and completeness assuming the hueristic is consistent and admissible.


### Pros and Cons
Pros            |  Cons
:-------------------------:|:-------------------------:
Lower cost of paths  |  Longer compute time due to line of sight check
Paths are easy to navigate as a set of straight line segments | Moves away from neat grid discretization
Easy to extend to higher dimensionalities | Still limited by underlying grid representation

## Overview of Safe Interval Path Planning

Unlike with Theta*, SIPP relies on abstracting the underlying 2D lattice and obstacles as a safety graph. This process requires some knowledge of static and dynamic obstacles, using which safe regions are created and connected by edges if a path exists between them. These 'regions' are actually time intervals over specific map regions, such that if the robot is within the given area inside the time interval, it is guranteed to be safe from collision.

Using this new graph abstraction, we can then apply standard A* on this new graph to produce paths that do not collide, handle dynamic obstacles(given sufficient information from some high level obstacle detection stack) and performs much faster than space-time A* given that intervals are finite compared to discretized time searches.

### Important Terms
Here are the key steps involved in the Safe Interval Path Planning algorithm:
- **Safe Interval Generation**: The first step of the SIPP algorithm is to divide the workspace into safe intervals, which are regions where the robot can move without colliding with obstacles. This is done by considering the robot's shape and size and generating safe intervals around each obstacle.

- **Interval Graph Construction**: Once the safe intervals are generated, the next step is to construct an interval graph that represents the connectivity of the safe intervals. In the interval graph, each safe interval is represented by a node, and there is an edge between two nodes if the corresponding safe intervals are connected and the robot can move between them without colliding with obstacles.

- **Shortest Path Computation**: The third step of the SIPP algorithm is to find the shortest path through the interval graph that connects the start and goal positions while avoiding obstacles. This is done using a path planning algorithm, such as A* or Dijkstra's algorithm, which searches for the path with the lowest cost in the interval graph.

- **Trajectory Computation**: Once the shortest path is found, the next step is to compute the trajectory of the robot along the path. This involves computing the exact position of the robot at each time step along the path, taking into account the robot's size and shape, the safe intervals, and the obstacles in the workspace.

- **Collision Checking**: During trajectory computation, it is important to check for collisions between the robot and obstacles in the workspace. If a collision is detected, the trajectory is adjusted to avoid the obstacle and stay within the safe intervals.

- **Dynamic Obstacle Handling**: The SIPP algorithm also accounts for dynamic obstacles in the workspace, such as moving people or vehicles. To handle dynamic obstacles, the trajectory is continuously updated in real-time to avoid collisions and stay within the safe intervals.

SIPP Schematic           
---
![sipp_timeline.png]({{site.baseurl}}/assets/images/sipp_timeline.png) 

SIPP Behaviour     
---
![sipp_behaviour.png]({{site.baseurl}}/assets/images/sipp_behaviour.png)





## Summary
Theta* and SIPP (Safe Interval Path Planning) are two popular algorithms used in robotic motion planning. Theta* is an improvement over A* algorithm that reduces the number of nodes expanded in the search tree by using a technique called "line-of-sight checking". On the other hand, SIPP is a method that generates safe intervals, constructs an interval graph, finds the shortest path through the graph, computes the trajectory of the robot along the path, and handles dynamic obstacles in real-time. By taking into account the robot's shape and size, SIPP ensures safe and collision-free motion planning for robots in complex and dynamic environments.

Over the course of this article, we explored the key implementation details of each of these planners, their use cases and strengths and weaknesses. Both of them give significant improvements to Space-Time A* in terms of handling dynamic obstacles and path quality, but at the cost of increased computational complexity and time. 

Future work can include plugins for the same that are readily integrated into popular ROS-based packages like MoveIt!, allowing users to simply tune the parameters based on their usecase and use it on a variety of diverse systems

## References
- Daniel, K., Nash, A., Koenig, S., & Felner, A. (2010). Theta*: Any-angle path planning on grids. Journal of Artificial Intelligence Research, 39, 533-579.
- Atzmon, D., Felner, A., Stern, R., Wagner, G., Barták, R., & Zhou, N. F. (2017). k-Robust multi-agent path finding. In Proceedings of the International Symposium on Combinatorial Search (Vol. 8, No. 1, pp. 157-158).
- Phillips, M., & Likhachev, M. (2011, May). Sipp: Safe interval path planning for dynamic environments. In 2011 IEEE international conference on robotics and automation (pp. 5628-5635). IEEE
- Lu, Z., Zhang, K., He, J., & Niu, Y. (2016). Applying k-means clustering and genetic algorithm for solving mtsp. In Bio-inspired Computing–Theories and Applications: 11th International Conference, BIC-TA 2016, Xi'an, China, October 28-30, 2016, Revised Selected Papers, Part II 11 (pp. 278-284). Springer Singapore.
- Nash, A., Koenig, S., & Tovey, C. (2010, July). Lazy Theta*: Any-angle path planning and path length analysis in 3D. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 24, No. 1, pp. 147-154).
