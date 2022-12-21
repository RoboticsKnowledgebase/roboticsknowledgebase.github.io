---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-12-07 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Task Prioritization Control for Advanced Manipulator Control
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
​	Task Prioritization Control is a concept and control method for robotic systems, mostly applicable to serial manipulators, to allow for execution of multiple simultaneous tasks, for which the tasks have a specific order of priority for the overall system success. In this article, we will discuss what Task Prioritization Control is, its use and applications, how the math behind it is derived, and some other resources available to aid in understanding.

## What is Task Prioritization Control?
​	As stated above, Task Prioritization Control is a concept and control method to control robotic systems when you have to accomplish multiple tasks simultaneously, and you know which of those tasks take precedence over others. More specifically, this control method is extremely beneficial to systems with extra or redundant degrees of freedom, and takes advantage of the redundancy in the system to perform more complementary tasks that would otherwise be harder to execute. If a lower priority task interferes with a higher priority task, the higher priority task will take precedence and its performance will not be affected, and the lower priority task will sacrifice degrees of freedom to the higher priority task to allow for success of the higher priority task.

### What defines a Task?

​	A task is a defined by a simple action that the system has to perform. For manipulators, it can be an action as simple as tracking a frame in space. Tasks can have 6 degrees of freedom or less. For example, for a manipulator doing a welding task, it is tracking a frame in space with a frame on the end-effector. However, this task is only 5 degrees of freedom, as the tool's roll does not matter with respect to the tracked frame in space, assuming the roll axis is aligned with the welding tip axis. When mathematically defining a task, it can be represented as a single vector. For kinematic controls, it is a twist vector, and can have 6 or less elements depending on the task degrees of freedom.

## Why Task Prioritization Control? What are its Use Cases?

​	For a system where you have multiple tasks that are essential to the success of the mission, a task prioritization controller is a great framework to both prioritize those tasks, and also execute them simultaneously. For example, for a 6 degree of freedom manipulator doing a welding task, a welding task is only 5 degrees of freedom. Therefore, there is an extra degree of freedom in the system that could be taken advantage of for another task, such as singularity avoidance or aligning the roll axis to a camera for better localization. For Team C (2022), a Task Prioritization Controller was used to perform multiple tasks simultaneously, such as aligning the reamer to the acetabulum, aligning the end-effector markers to the camera, avoiding singularities, and avoiding joint limits (https://mrsdprojects.ri.cmu.edu/2022teamc/).

## Derivation of Task Prioritization

​	This section focuses on the derivation and math behind task prioritization. It will be put in the context for control of manipulators to aid in understanding.
$$
\dot{x}_i = J_i * \dot{q}
$$
The equation above is the Forward Kinematics equation, translating joint velocities into a task-space velocities through the Task Jacobian. As a refresher, the Jacobian can be thought of as a matrix that maps how the velocity of each joint contributes to a twist of the **Task Frame** in task space. In other words, each column of a Jacobian is a twist vector corresponding to the movement of each joint at unit speed. **i** in this case is the **Task Priority Number**. A Jacobian does not need to be square. The number of columns is depending on the number of joints or controllable degrees of freedom there are. The number of rows is dependent on the number of task degrees of freedom. For the welding task on a 6 degree of freedom manipulator, mentioned in the section above, the Jacobian would be a 5 x 6 element matrix.

However, for our controls, we need to translate the task-space velocities to joint velocities, which is what we have direct control of. Therefore, we need to inverse kinematics, and therefore need to invert the equation above.
$$
\dot{q}_i = J^\# * \dot{x}_i
$$
where
$$
J^\# = (J^T J)^{-1}J^T
$$
for Jacobians that have more rows than columns, and
$$
J^\# = J^T(J J^T)^{-1}
$$
for Jacobians that have more columns than rows.

However, because this controller is only applicable to systems that have redundancy, the second equation only applies. These equations define what we call the Pseudoinverse of the Jacobian.

Using this, inverse kinematics equation, we can now proceed with task prioritization.
$$
\dot{q} = J_1^\#\dot{x}_1 +(I - J_1^\#J_1)\dot{q}_{2...n}
$$
The above equation is the fundamental start to understanding Task Prioritization. Essentially, the equation is saying that the final joint velocities is a sum of the highest priority task multiplied with its Jacobian with the joint velocities contributed from lower priority tasks projected into the null space of the higher priority task. The null space  projector
$$
(I - J_1^\#J_1)
$$
is a matrix that exposes the redundant space of the highest priority task (for i = 1), and by projecting the lower priority tasks into this space, the components of lower priority tasks that interfere with the higher priority task are removed, and only the components that can use the redundant space are left. The lower priority tasks are all summed up in
$$
\dot{q}_{2...n}
$$
A secondary task, i = 2 can be calculated with the following equations
$$
\dot{x}_2 = J_2*\dot{q} = J_2*(J_1^\#\dot{x}_1 +(I - J_1^\#J_1)\dot{q}_{2...n})
$$
Rearranging this equation, we get
$$
J_2(I - J_1^\#J_1)\dot{q}_{2...n} = \dot{x}_2 - J_2J_1^\#\dot{x}_1
$$
To simplify this equation, lets make this equivalency
$$
\tilde{J_2} = J_2(I - J_1^\#J_1)
$$
We can then take the pseudoinverse of this
$$
\tilde{J_2}
$$
and end up with the resulting equation:
$$
\dot{q}_{2} = \tilde{J_2}^\#(\dot{x}_2 - J_2J_1^\#\dot{x}_1)
$$

For simplicity, the following substitution can be made.
$$
\dot{q}_1 = J_1^\#\dot{x}_1
$$
So,
$$
\dot{q}_{2} = \tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1)
$$
However, 
$$
\tilde{J_2}
$$
has its own null space, for which we can project even lower priority tasks. Therefore, we can expand the equation above in the following:
$$
\dot{q}_{2...n} = \tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1) + (I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n}
$$
Plugging this resulting equation into an earlier equation, we get
$$
\dot{q} = J_1^\#\dot{x}_1 +(I - J_1^\#J_1)(\tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1) + (I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n})
$$
For simplification, the following can be proved (but not here):
$$
(I - J_1^\#J_1)*\tilde{J_2}^\# = \tilde{J_2}^\#
$$
So, the final equation is
$$
\dot{q} = J_1^\#\dot{x}_1 +\tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1) + (I - J_1^\#J_1)(I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n}
$$
To really nail this down, we can repeat this for a tertiary task!
$$
\dot{x}_3 = J_3*\dot{q} = J_3*(J_1^\#\dot{x}_1 +\tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1) + (I - J_1^\#J_1)(I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n})
$$

$$
J_3(I - J_1^\#J_1)(I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n} = \dot{x}_3 - J_3(J_1^\#\dot{x}_1 +\tilde{J_2}^\#(\dot{x}_2 - J_2J_1^\#\dot{x}_1))
$$

To simplify this equation, we can make the following substitutions:
$$
\dot{q}_1 = J_1^\#\dot{x}_1
$$

$$
\dot{q}_2 = \tilde{J_2}^\#(\dot{x}_2 - J_2\dot{q}_1)
$$

So,
$$
J_3(I - J_1^\#J_1)(I - \tilde{J_2}^\#\tilde{J_2})\dot{q}_{3...n} = \dot{x}_3 - J_3(\dot{q}_1 + \dot{q}_2)
$$

$$
\tilde{J_3} = J_3(I - J_1^\#J_1)(I - \tilde{J_2}^\#\tilde{J_2})
$$

$$
\dot{q}_3 = \tilde{J_3}^\# (\dot{x}_3 - J_3(\dot{q}_1 + \dot{q}_2))
$$

As shown, there is a pattern emerging here, and we can actually repeat it as much as we want to, with diminishing benefits depending on the number of redundant degrees of freedom.

The algorithm for computing the full task prioritization inverse kinematics can be generalized to the following:
$$
\dot{q} = \sum_{i=1}^{n}\dot{q}_i,
$$
where
$$
\dot{q}_i = (J_i(\prod_{j=1}^{i-1}N_j))^\#(\dot{x}_i-J_i(\sum_{j=1}^{i-1}\dot{q}_j)),
$$

$$
N_j = I - (J_{j-1}N_{j-1})^\#(J_{j-1}N_{j-1})
$$

$$
N_1 = I,
$$

and n is the number of tasks, where highest priority is i = 1.

Once the summed up joint velocities are computed, those can be sent directly to the manipulator to execute the tasks in the prioritized order. If a lower priority task has degrees of freedom that would interfere with the higher priority task, it will sacrifice those degrees of freedom to not compromise the higher priority task.

## Further Reading and Resources

​	As discussed in the sections above, Task Prioritization Control can be used to perform multiple tasks at once, which include not only mission critical tasks, but also tasks that will reduce risk of unstable behaviors which can occur when at singularities or at joint limits. The following papers and resources contain both more information on task prioritization, and more information on other tasks that could be added to the controller to improve robustness of a manipulator system.

### Task Prioritization

The following papers below discuss derivation of task prioritization and application, including obstacle avoidance for manipulators. The first paper is one of the first formulations of task prioritization, and the second is a generalization of the algorithm.

Task-Priority Based Redundancy Control of Robot Manipulators:

https://journals.sagepub.com/doi/10.1177/027836498700600201

A general framework for managing multiple tasks in highly redundant robotic systems:

https://ieeexplore.ieee.org/document/240390

### Pseudoinverse Calculation and Singularity Damping

This paper gives a brief introduction to inverse kinematics in general, but also how to calculate the psuedoinverse through Singular Value Decomposition, which is useful when already using the Singular Value Decomposition to determine if an arm is near singularity. It also includes singularity damping, which can stabilize a kinematic controllers behavior near singularities (which may sacrifice accuracy).

Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods:

http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf

### Measures of Manipulability

This paper discusses an interesting way to decompose a jacobian into separate parts. It has a really good discussion on different metrics for manipulability measurements, which is an inverse to singularity (higher manipulability means farther from singularity).

The joint‐limits and singularity avoidance in robotic welding:

https://www.emerald.com/insight/content/doi/10.1108/01439910810893626/full/html

### Manipulability Gradient Estimation (For Singularity Avoidance)

This paper has a lot of varying content. The main takeaway for this controller specifically is if you want to implement a singularity avoidance algorithm task. This paper has a good method for estimating the gradient of manipulability or singularities, which can be combined with a task to avoid singularities.

Strategies for Increasing the Tracking Region of an Eye-in-Hand System by Singularity and Joint Limit Avoidance:

https://kilthub.cmu.edu/articles/journal_contribution/Strategies_for_Increasing_the_Tracking_Region_of_an_Eye-in-Hand_System_by_Singularity_and_Joint_Limit_Avoidance/6625910

### Joint Limit Avoidance

This paper has a great formulation for a joint limit avoidance task.

A Weighted Gradient Projection Method for Inverse Kinematics of Redundant Manipulators Considering Multiple Performance Criteria:

https://www.sv-jme.eu/article/a-weighted-gradient-projection-method-for-inverse-kinematics-of-redundant-manipulators-considering-multiple-performance-criteria/

## References
- Task-Priority Based Redundancy Control of Robot Manipulators: https://journals.sagepub.com/doi/10.1177/027836498700600201
- A general framework for managing multiple tasks in highly redundant robotic systems: https://ieeexplore.ieee.org/document/240390
- Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods: http://graphics.cs.cmu.edu/nsp/course/15-464/Spring11/handouts/iksurvey.pdf
- The joint‐limits and singularity avoidance in robotic welding: https://www.emerald.com/insight/content/doi/10.1108/01439910810893626/full/html
- Strategies for Increasing the Tracking Region of an Eye-in-Hand System by Singularity and Joint Limit Avoidance: https://kilthub.cmu.edu/articles/journal_contribution/Strategies_for_Increasing_the_Tracking_Region_of_an_Eye-in-Hand_System_by_Singularity_and_Joint_Limit_Avoidance/6625910
- A Weighted Gradient Projection Method for Inverse Kinematics of Redundant Manipulators Considering Multiple Performance Criteria: https://www.sv-jme.eu/article/a-weighted-gradient-projection-method-for-inverse-kinematics-of-redundant-manipulators-considering-multiple-performance-criteria/
