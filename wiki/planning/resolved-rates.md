---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-12-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Resolved-rate Motion Control
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
For several real-world applications, it may be necessary for the robot end-effector to move in a straight line in Cartesian space. However, when one or more joints of a robot arm are moved, the end-effector traces an arc and not a straight line. Resolved-rate[1] is a Jacobian-based control scheme for moving the end-effector of a serial-link manipulator at a specified Cartesian velocity $v$ without having to compute the inverse kinematics at each time step. Instead, the inverse of the Jacobian matrix alone is recomputed at each time step to account for the updated joint angles. The displacement of each joint is given by the product of the current joint velocity and the time step, which is then added to the current joint configuration to update the pose of the robot. The advantage of this incremental approach of updating joint angles is that the robot moves smoothly between waypoints as opposed to exhibiting jerky movements that arise from frequent recomputations of the inverse kinematics. This article presents the mathematical formulation of the resolved-rate motion control scheme and explains its usage in a motion compensation algorithm.

## Derivation
The forward kinematics of a serial-link manipulator provides
a non-linear surjective mapping between the joint space
and Cartesian task space[2]. For an $n$-degree of freedom (DoF) manipulator with $n$ joints, let $\boldsymbol{q}(t) \in \mathbb{R}^{n}$ be the joint coordinates of the robot and $r \in \mathbb{R}^{m}$ be the parameterization of the end-effector pose. The relationship between the robot's joint space and task space is given by:
$$\begin{equation}
    \boldsymbol{r}(t)=f(\boldsymbol{q}(t))
\end{equation}$$
In most real-world applications, the robot has a task space $\mathcal{T} \in \operatorname{SE}(3)$ and therefore $m = 6$. The Jacobian matrix provides the relation between joint velocities $\dot{q}$ and end-effector velocity $\dot{v}$ of a robotic manipulator. A redundant
manipulator has a joint space dimension that exceeds the workspace dimension, i.e. $n > 6$. Taking the derivative of (1) with respect to time:
$$\begin{equation}
    \nu(t)=J(q(t)) \dot{q}(t)
\end{equation}$$
where $J(q(t))=\left.\frac{\partial f(q)}{\partial q}\right|_{q=q_{0}} \in \mathbb{R}^{6 \times n}$ is the manipulator
Jacobian for the robot at configuration $q_0$. Resolved-rate
motion control is an algorithm which maps a Cartesian end-effector
velocity $\dot{v}$ to the robot’s joint velocity $\dot{q}$. By rearranging (2), the required joint velocities are:
$$\begin{equation}
    \dot{\boldsymbol{q}}=\boldsymbol{J}(\boldsymbol{q})^{-1} \nu
\end{equation}$$
It must be noted that (3) can be directly solved only is $J(q)$ is square and non-singular, which is when the robot has 6 DoF. Since most modern robots are several redundant DoFs, it is more common to use the
Moore-Penrose pseudoinverse:
$$\begin{equation}
    \dot{\boldsymbol{q}}=\boldsymbol{J}(\boldsymbol{q})^{+}v
\end{equation}$$
where the $(\cdot)^{+}$ denotes the pseudoinverse operation.


## Algorithm
Since control algorithms are generally run on digital computers, they are often modeled as discrete-time algorithms, i.e., they compute necessary values at discrete time intervals. Consider a time horizon with discrete time steps such that the interval between two consecutive steps is $\Delta_{t}$. The following sequence of steps are repeated for as long as the robot's end-effector is required to move at the specified Cartesian velocity $v$:
1. At each time step $k$, compute the kinematic Jacobian matrix $\mathbf{J}\left(\boldsymbol{q}_{k}\right)$ using the current values of the joint angles $\mathbf{q_k}$.
2. Calculate the joint velocity vector $\mathbf{\dot{q}}$ that must be achieved in the current time step using the equation $\dot{\boldsymbol{q}}=\mathbf{J}\left(\boldsymbol{q}_{k}\right)^{-1} \boldsymbol{v}$.
3. The joint angle displacement is then calculated as $\mathbf{q_{dist}} = \Delta_{t}\mathbf{\dot{q}}$. This quantity signifies how much we want to move each of the joints in the given time step based on the value of joint velocity.
4. The next joint configuration is calculated by $\mathbf{q_{k+1}}=\mathbf{q_{k}}+\mathbf{\Delta_{t} \dot{q}}$
5. The robot hardware is commanded to move the next joint configuration $\mathbf{q_{k+1}}$.

The steps above are repeated for as long as necessary. It is seen that we do not have to compute the inverse kinematics of the robot at each time step; rather, the inverse of the Jacobian matrix alone is sufficient to have the end-effector continue to move at a Cartesian velocity $v$.


## Resolved-rates for motion compensation
The following is an example for how the resolved-rates motion compensation can be used. Specifically, this relates to Team A's Chopsticks Surgical system but is extendable to other systems. Since the Chopsticks Surgical system needs to 3D-scan and palpate a moving liver, the robot arm must incorporate a motion compensation scheme to ensure zero relative motion between the robot and the liver at all times. In order to compensate for the liver's motion, it is first necessary to predict where the liver or some region of interest on the liver will be at any given point in time. The robot arm will then move to this predicted location at the same time the liver does, thereby canceling out the effects of the liver's motion. The liver is represented as a point cloud $\mathbf{P}$ where each point $p_i$ has an associated normal vector. For both 3D-scanning and palpation, the robot arm must go to each point to maximize coverage of the liver's surface. The frequency and amplitude of motion of the liver are estimated within a very small margin of error to the ground truth. The resolved-rate motion controller is incorporated in the motion compensation by iterating over each point of the liver point cloud as follows:
1. Let the current position of the robot's end-effector be $p_i$. The target location for the robot arm is the predicted position of the next point $p_{i+1}$ to visit on the liver's surface. We have a function that outputs the predicted position of the target point in the present time step.
2. However it is generally not possible to get to the desired location in the same time step as the prediction because the robot may need more than one time step to get there and the liver (and hence the current point of interest on the liver) will have moved by then.
3. Therefore, we ``close in" on the target point with each passing time step, i.e., in each time step, we predict where the target point is, and take a step in that direction.
4. This is repeated for a couple of time steps until the robot arm eventually ``catches up" with the current point. When the robot arm gets to the desired location within a very narrow margin of error, we consider that point on the liver's surface to have been visited and move on to the next point.
5. Steps 1 - 4 are repeated for every point on the liver's surface. Since the liver point cloud representation is quite dense, the robot arm can move between points very quickly.

The biggest advantage of the resolved-rate motion controller is that it makes the robot arm move smoothly between waypoints and not exhibit jerky movements that arise from having to compute inverse kinematics at every time step. We simply need to update joint displacements that will eventually converge to the desired point. In the particular case of our surgical project, since it has the robot arm take unit steps along the vector between the current and desired positions at every time step, it ensures zero relative motion between the robot arm and the liver.


## Summary
Resolved-rate motion control allows for direct control of the robot’s end-effector velocity, without expensive path planning. In surgical robots, it is often necessary for the robot arm to move in a straight line at a constant velocity as it lacerates tissue. This controller would be well-suited to the task. In fact, this controller is so versatile that it is known to have been used in space robotics and in general reactive control schemes such as visual servoing[2].

## References

D. E. Whitney, "Resolved Motion Rate Control of Manipulators and Human Prostheses," in IEEE Transactions on Man-Machine Systems, vol. 10, no. 2, pp. 47-53, June 1969, doi: 10.1109/TMMS.1969.299896.

Haviland, Jesse & Corke, Peter. (2020). \Maximising Manipulability During Resolved-Rate Motion Control.

Zevallos, Nicolas & Rangaprasad, Arun Srivatsan & Salman, Hadi & Li, Lu & Saxena, Saumya & Xu, Mengyun \& Qian, Jianing & Patath, Kartik & Choset, Howie. (2018). A Real-time Augmented Reality Surgical System for Overlaying Stiffness Information. 10.13140/RG.2.2.17472.64005. 

https://www.youtube.com/watch?v=rkHs7K0ad14&feature=emb_logo
