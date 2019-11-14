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

# INSERT IMAGE HERE

In the image above, we see that given a point at a particular location from the robot say at location (x,y) in the robot's frame (the frame fixed on the robot). The point is at a distance l from the current location of the robot. Using a geometrical derivation, we can derive the radius of curvature of this arc as -

\gamma = \frac{2*x}{l^2}\

This is the radius of the path we want the system to follow in order to converge to the trajectory. We can see from the image that the arc is tangential to the current trajectory of the robot. Thus the kinematic constraints of the robot are not violated.




We see that the curvature is only dependent on the cross-track distance between the robot and the point and thus can be intutively thought of as a controller to minimize the cross track error.

In case of a trajectory, the location along the path at any point of time is known and thus the along track error can be used to determine the desired linear velocity utilising a PID controller. In case we want to follow a path, one method is to run a PID controller to reach the lookahead point or the velocity can be set to a constant value.

Given a V from the along-track error controller, since along the arc the linear and angular velocity are related by:

V= wr
V = \\omega * r
where
V = 
\omega = 

Thus the desired angular velocity can be determined as -

w \omega = 	V/r (\frac{V}{r}
	= 	V*gamma (V * \gamma
    =	2*V*x/l^2 (frac{2 * V * x}{l^2}
    
This maps the cross-track error to the desired angular velocity of the robot. One should note that the only parameter that is tuned by the user is the lookahead distance, making it easy to tune.

#Implementation

### 1. Determine current location of the robot
The current location of the robot (x,y,\theta)in the world frame needs to be determined, this could come from the odometry sensor if the world frame is at the start of the path. In case one is using an absolute positioning method like a GPS, the world frame could be the "UTM" frame or any other world fixed frame. 

**Note:** One should ensure that the robot and the world frame are provided in the same format i.e. ENU or NED format.

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
w \omega = 	V/r (\frac{V}{r}
	= 	V*gamma (V * \gamma
    =	2*V*x/l^2 (frac{2 * V * x}{l^2}
    
These desired linear and angular velocity is followed by a low level controller.









Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Basic syntax
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.


#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering \\( 1/x^{2} \\), and here is a block rendering:
\\[ \frac{1}{n^{2}} \\]

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
-https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your pub
