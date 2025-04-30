<!-- ---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-05-11 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Title goes here
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).

## First subheading
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
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

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
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here. -->


# PILZ Industrial Motion Planner

For industrial robot applications, it is often necessary to have predictable, deterministic motion along well-defined paths like straight lines or circular arcs. PILZ Industrial Motion Planner is a trajectory generator that provides these capabilities within the MoveIt framework, offering a simple and predictable way to plan standard robot motions.

## Background

PILZ Industrial Motion Planner was initially developed as part of the ROS-Industrial project with the concept of bringing interfaces equivalent to conventional industrial robots into the world of ROS. It was designed for situations where "industrial applications often also demand simple things like just moving in a straight line." The planner was incorporated into the MoveIt repository in 2020, making it available as a standard component for ROS-based robotic systems.

Unlike sampling-based planners such as OMPL (Open Motion Planning Library), which focus on finding collision-free paths in complex environments but may produce jerky movements, PILZ was specifically designed to generate trajectories with precise, predictable motion patterns that are common in industrial settings.

## Core Motion Types

PILZ supports three fundamental motion types:

1. **PTP (Point-to-Point)**: PTP commands move the robot so that the end effector reaches the specified coordinates without specifying the path in 3D space. This planner generates fully synchronized point-to-point trajectories with trapezoidal joint velocity profiles.

2. **LIN (Linear)**: In LIN commands, the end effector moves along a straight line connecting the starting point to the endpoint. The motion maintains a straight-line path in Cartesian space, which is essential for many industrial tasks.

3. **CIRC (Circular)**: Allows the end effector to move in a circular arc. To define the arc, additional information beyond start and end points is required, such as a center point or another point on the arc.

## Mathematical Formulation

For PTP motion, PILZ generates trajectories with trapezoidal joint velocity profiles. All joints are assumed to have the same maximal joint velocity/acceleration/deceleration limits, with the strictest limits being adopted if they differ. The joint with the longest time to reach the goal (lead axis) determines the overall motion time, while other axes are decelerated to maintain synchronized motion phases.

For LIN motion, the planner generates a straight-line path in Cartesian space. The rotational motion uses quaternion slerp between start and goal orientation, while translational and rotational motions are synchronized in time.

## Blending Feature

A key feature of PILZ is its ability to blend multiple motion segments together, creating smooth transitions between waypoints without stopping at each point. When the TCP (Tool Center Point) comes closer to a goal than the specified blend radius, it is allowed to begin moving toward the next goal. When leaving a sphere around the current goal, the robot returns to the trajectory it would have taken without blending.

This blending capability is particularly valuable in applications where cycle time is critical, as it allows the robot to move continuously through a sequence of points without the acceleration/deceleration that would otherwise be required at each waypoint.

## Algorithm

The PILZ planning process involves these steps:

1. The planner receives a motion request specifying the start state, goal constraints, and motion type (PTP, LIN, or CIRC).
2. Based on the motion type, the planner calculates a trajectory with appropriate velocity profiles.
3. For PTP, it generates synchronized trapezoidal velocity profiles for all joints.
4. For LIN, it generates a straight-line path in Cartesian space with synchronized translational and rotational motion.
5. For CIRC, it calculates a circular arc meeting the specified constraints.
6. When planning sequences, it applies blending between segments if blend radii are specified.
7. The resulting trajectory includes positions, velocities, and accelerations for each waypoint.

## Implementation in ROS and MoveIt

PILZ is implemented as a plugin for MoveIt and can be accessed through the standard MoveIt interfaces. By loading the corresponding planning pipeline, the trajectory generation functionalities can be accessed through the user interface (C++, Python, or RViz) provided by the move_group node.

The planner uses maximum velocities and accelerations from the parameters of the ROS node. These limits can be specified in the joint_limits.yaml file, which is typically generated using the MoveIt Setup Assistant.

## Applications

PILZ is particularly useful in:

1. Industrial manufacturing where predictable, deterministic motions are required
2. Applications requiring straight-line motion, such as welding or cutting
3. Pick-and-place operations where cycle time is critical
4. Tasks requiring smooth blending between multiple motion segments

## Limitations

While PILZ offers predictable motion patterns, it has some limitations:

1. It is not designed for complex environment navigation where sampling-based planners excel
2. For LIN and CIRC motions, if there are unattainable postures between the start and end points, planning will fail
3. Joint limits may be violated if Cartesian motions are infeasible, requiring adjustment of scaling factors

---

# CHOMP: Covariant Hamiltonian Optimization for Motion Planning

## Background

CHOMP (Covariant Hamiltonian Optimization for Motion Planning) is an optimization-based motion planning algorithm developed at Carnegie Mellon University. CHOMP was introduced as "a novel method for continuous path refinement that uses covariant gradient techniques to improve the quality of sampled trajectories."

Unlike traditional motion planning approaches that separate path finding from trajectory optimization, CHOMP integrates these processes, capitalizing on gradient-based optimization techniques to directly generate smooth, collision-free trajectories.

## Motivation

Traditional motion planning algorithms often produce jerky or inefficient paths that require post-processing. CHOMP was developed to address these limitations by directly optimizing trajectories for both smoothness and collision avoidance simultaneously.

While high-dimensional motion planners can navigate complex environments, they often struggle with "narrow passages" and require additional post-processing to remove jerky motions. CHOMP aims to resolve these issues by providing a standalone motion planner that can converge over a wide range of inputs and optimize higher-order dynamics.

## Mathematical Formulation

CHOMP optimizes trajectories by minimizing a cost functional that combines two primary components:

1. **Smoothness Cost**: Penalizes non-smooth motions, typically represented as the sum of squared derivatives of the trajectory
2. **Collision Cost**: Penalizes proximity to obstacles

The overall cost function can be represented as:

```
C(ξ) = λ_smooth * F_smooth(ξ) + λ_obs * F_obs(ξ)
```

Where:
- `ξ` is the trajectory
- `F_smooth` is the smoothness cost
- `F_obs` is the obstacle cost
- `λ_smooth` and `λ_obs` are weighting factors

CHOMP uses functional gradient techniques to iteratively improve the trajectory, computing the gradient of the cost function and updating the trajectory accordingly.

## Algorithm

CHOMP is a gradient-based trajectory optimization procedure that makes many everyday motion planning problems both simple and trainable. The basic algorithm follows these steps:

1. Start with an initial trajectory (which may be infeasible and colliding with obstacles)
2. Compute the functional gradient of the cost function with respect to the trajectory
3. Take a step in the direction of the negative gradient to update the trajectory
4. Repeat steps 2-3 until convergence or a maximum number of iterations is reached

What makes CHOMP unique is its use of covariant gradient techniques that properly account for the geometry of the trajectory space, leading to more efficient optimization.

## Collision Avoidance

CHOMP represents obstacles using distance fields, which provide a measure of distance to the nearest obstacle at any point in the workspace. This allows for efficient computation of collision costs and their gradients.

In implementations like MATLAB's manipulatorCHOMP, the robot is modeled as a collection of spheres (spherical approximation), and obstacles can be represented either as collections of spheres or as truncated signed distance fields.

## Implementation and Applications

CHOMP has been implemented for various robotic systems, including:

A six degree-of-freedom WAM arm developed by Barrett Technology and a twelve degree-of-freedom quadrupedal robot developed by Boston Dynamics. In testing with the WAM arm, CHOMP successfully found smooth collision-free trajectories for 99 out of 105 planning problems in a household environment.

CHOMP is particularly well-suited for:
1. Path refinement of trajectories generated by other planners
2. Direct motion planning in relatively simple environments
3. Applications requiring smooth, natural motions
4. Scenarios where trajectory quality is important

## Advantages and Limitations

**Advantages**:
- Produces smooth, natural-looking trajectories
- Can optimize trajectories even when initialized with infeasible paths
- Integrates path finding and trajectory optimization
- Can consider dynamics and constraints directly in the optimization

**Limitations**:
- May converge to local minima, especially in complex environments
- Computation time can be high for complex problems
- Performance depends on the quality of the initial trajectory
- Not guaranteed to find a solution in highly constrained environments

## Relation to Other Motion Planning Approaches

CHOMP belongs to a family of optimization-based motion planners that also includes TrajOpt and STOMP (Stochastic Trajectory Optimization for Motion Planning). These approaches differ from sampling-based planners like RRT and PRM, which focus on geometric path finding without considering trajectory smoothness directly.

CHOMP has influenced subsequent work in trajectory optimization and continues to be an important reference in the field of motion planning.

## References

1. Ratliff, N., Zucker, M., Bagnell, J. A., & Srinivasa, S. (2009). CHOMP: Gradient optimization techniques for efficient motion planning. In IEEE International Conference on Robotics and Automation (pp. 489-494).

2. Ratliff, N., Zucker, M., Bagnell, J. A., & Srinivasa, S. (2013). CHOMP: Covariant Hamiltonian optimization for motion planning. The International Journal of Robotics Research, 32(9-10), 1164-1193.

3. Schleicher, J. (2020). Introducing the New Pilz Industrial Motion Planner for MoveIt. MoveIt Blog.

4. MoveIt Documentation: Pilz Industrial Motion Planner. https://moveit.picknik.ai