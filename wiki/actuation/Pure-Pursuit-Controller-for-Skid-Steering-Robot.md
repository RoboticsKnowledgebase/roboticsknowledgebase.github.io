---
# Simple Pure Pursuit based Controller for Skid Steering Robot

---
This article will cover the steps for implementing a simple Pure Pursuit based controller for a skid steering robot on a flat surface, limiting the degrees of freedom to x, y and heading(yaw).
The controller will make use of the Pure Pursuit Algorithm to follow a desired trajectory. This is a good method to quickly implement a robust path/trajectory tracking controller for robots travelling at low speeds, where exact trajectory tracking is not required. 
One can make use of optimal control methods like iLQR for better performance in those cases. 
The aim of the controller is to determine the desired velocity of the robot given the current location of the robot and the trajectory to be followed.
The implementation discussed here is based on R. Craig Coulter's  work at the Robotics Institute, Carnegie Mellon University in January, 1992.

## Robot Constrains
A skid steering robot cannot have a velocity in the lateral directon. However, due to wheel slippage, it can have a forward as well as angular velocity at any instant.

## Pure Pursuit
Pure Pursuit is a curvature based trajectory tracking controller. It works by calculating the curvature of the path to follow in order to reach from the current position to some goal position.

This goal position keeps on changing and is a point on the trajectory to be followed at a particular "lookahead distance" from it. 
The following image explains the concept of lookahead distance and the arc to follow.

![Geometry of Pure Pursuit Algorithm [1]](assets/images/pure_pursuit_geometry.png)

In the image above, we see that given a point at a particular location from the robot say at location (x,y) in the robot's frame (the frame fixed on the robot). The point is at a distance l from the current location of the robot. Using a geometrical derivation, we can derive the radius of curvature of this arc as -

\gamma = \frac{2*x}{l^2}\

This is the radius of the path we want the system to follow in order to converge to the trajectory. We can see from the image that the arc is tangential to the current trajectory of the robot. Thus the kinematic constraints of the robot are not violated.

We see that the curvature is only dependent on the cross-track distance between the robot and the point and thus can be intutively thought of as a controller to minimize the cross track error.

In case of a trajectory, the location along the path at any point of time is known and thus the along track error can be used to determine the desired linear velocity utilising a PID controller. In case we want to follow a path, one method is to run a PID controller to reach the lookahead point or the velocity can be set to a constant value.

Given a V from the along-track error controller, since along the arc the linear and angular velocity are related by:

V= wr
\\[V = \omega * r \\]
Thus the desired angular velocity can be determined as -

\\[\omega = \frac{V}{r} \\]
\\[\omega = V * \gamma \\]
\\[ \omega = \frac{2 * V * x}{l^2} \\]

This maps the cross-track error to the desired angular velocity of the robot. One should note that the only parameter that is tuned by the user is the lookahead distance, making it easy to tune.

#Implementation

### 1. Determine current location of the robot
The current location of the robot (x,y,\theta)in the world frame needs to be determined, this could come from the odometry sensor if the world frame is at the start of the path. In case one is using an absolute positioning method like a GPS, the world frame could be the "UTM" frame or any other world fixed frame. 

> One should ensure that the robot and the world frame are provided in the same format i.e. ENU or NED format.

### 2. Find the point closest to the vehicle
The point on the trajectory closest to the robot needs to be determined. This step will depend on how the trajectory is provided, if the waypoints on the trajectory are sparsely provided, one can connect the closest two waypoints through a straight line and project the current location of the robot on it to determine the point on the trajectory closest to the robot.
Alternatively, if the waypoints on the path/trajectory are quite dense, one can just use euclidean distance to compute the closest point.

### 3. Finding the goal point
The goal point is found by moving by one lookahead distance along the path/trajectory from the closest point identified before. This is the point we want to follow. However, this point is in the world frame and not in the robot's frame.

### 4. Transforming the goal point to vehicle's coordinate
Since the goal point is in world frame, it needs to be transformed to the robot's frame. Accurate position and orientation estimate of the robot in the world frame is critical here. Yaw value is of special importance here as it is often noisy and can lead to errors.

### 5. Calculate controller output
The linear velocity is determined using the controller to minimize alongtrack error.
The angular velocity is computed using 
\\[ \omega = \frac{2 * V * x}{l^2} \\]
    
These desired linear and angular velocity is followed by a low level controller.
## References
1. https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

