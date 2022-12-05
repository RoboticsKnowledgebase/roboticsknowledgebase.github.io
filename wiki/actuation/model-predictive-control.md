---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-05-04 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Model Predictive Control Introduction and Setup
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
​	Model Predictive Control (MPC for short) is a state-of-the-art controller that is used to control a process while satisfying a set of constraints. In this article, we will discuss what MPC is and why one might want to use it instead of simpler but usually robust controllers such as PID control. We will also include some useful resources to get started.

## What is MPC?
​	As stated above, MPC is an advanced state-of-the-art controller that is used to control a process or system while satisfying a set of constraints. It requires a good model of the system or process being controlled, as one of the constraints to satisfy is the dynamics model itself. This dynamics model is then used to make future predictions about the system's behavior to find the optimal control input to minimize the objective function. This minimization is done over a chosen finite time horizon.

​	Because MPC has a control input vector, and also has a state vector, it also has the ability to be multi-input multi-output. For example, for a multi-dof manipulator system, MPC could handle multiple joint torques (or whatever control input) to control the outputs (joint positions, velocities, etc.).  This doesn't just apply to manipulators, however. It can apply to any system ranging from UGVs to quadcopters as well. If one is familiar with LQR control, MPC is a finite time horizon version of LQR with constraints.

## Why MPC?

​	You should choose MPC because it integrates system constraints, something that normal PID control cannot do easily. Furthermore, the dynamics of the environment may change over time, and MPC allows for an easy way to estimate those dynamics and adjust accordingly. PID control is normally not for systems whose dynamics change often (such as the dynamics of drilling differing bone densities as we ream further and further into bone) because the gains for the PID may need to change to produce desired responses. If not, then the changing dynamics may cause overshooting (a very undesired response when drilling into a person). MPC is also multi-input multi-output, allowing for multiple inputs to vary the output of the system. This is possible in PID, but requires either separate axes of control or additional control loops (which is fine, but not simple).

## Formulating the Optimal Control Problem

​	The first step to setting up an MPC is to define your optimal control problem (OCP for short). The Optimal Control Problem contains the following components: the objective function, the dynamics model, and the system constraints. It usually takes the form of the following: 
$$
min_{x,u}(f(x))\\
s.t.\\d(x,u) = 0\\
c(x,u) \leq 0
$$
​	Where the first line is the objective or cost function that we are trying to minimize, the second line d(x, u) = 0, is the dynamics constraint, and the third line, c(x, u) less than or equal to 0,  are additional constraints.

The objective or cost function usually takes similar form to LQR cost functions as a quadratic, convex function with hyperparameters Q and R, which determines what states or inputs are most important to minimize over the time horizon. An example is shown below:
$$
min_{x,u,k\epsilon[1,H]} \frac{1}{2}x^{T}_{H}Q_{f}x_{H} + \sum_{k=1}^{H-1}{\frac{1}{2}x^{T}_{k}Qx_{k} + \frac{1}{2}u^{T}_{k}Ru_{k}}
$$
where H is the time horizon, Q is the matrix determining cost of states, Qf is the matrix determining the cost of the terminal states, and R is the matrix determining the cost of the inputs. Just like in LQR, the goal of the controller is to minimize this objective/cost function. 

​	The dynamics constraint, d(x,u) = 0, is the constraint that takes the following form. Let's say the following function is your dynamics:
$$
\dot x = f(x,u)
$$
Then the dynamics constraint takes the following form:
$$
d(x,u) = f(x,u) - \dot x = 0
$$
​	The other constraints, c(x,u), can be any other constraints one might have on the system. For manipulators, this could be joint position, velocity, or torque limits, for example. For rockets, it could be
$$
p_{z} \geq 0\\
$$
since the rocket can not physically go below the ground in reality (without a catastrophe at least). An example of formulating the Optimal Control Problem can be found in Appendix A below. This example is from a former team, MRSD 2022 Team C for an manipulator that drills/reams the hip socket (acetabulum).

## Getting Started With Implementation

​	There are a lot of different languages and libraries to implement MPC in, but the recommended ones are C++ and Julia. The solvers listed below should list the optimal control inputs. These can be generated both offline or online and used on the system. For online systems, it is best to do one MPC solve every time step , use the first control input for the upcoming time step, and then repeat.

#### Julia

​	Julia is a scripting language, which allows for fast prototyping, while having very similar speeds to C++. There are however some caveats with using Julia. First, it is not as well supported as C++ (although many of the libraries needed to implement MPC are available). Second,  similar to other languages, the runtime performance is very dependent on making sure that there are no global variables, and that allocation of memory is minimized during runtime. The benefits of Julia is that it is simple to code in (very similar syntax to Matlab and Python), it has lots of shortcuts for writing code that would take multiple lines in C++, and it can sometimes outperform C++ if done correctly. This allows for very robust code while also having fast iteration time.

If you are using Julia as a coding language, the following libraries are recommended for use (although there may be others that can be of use):

**Dynamics Models/Simulation**

- RigidBodyDynamics.jl: https://github.com/JuliaRobotics/RigidBodyDynamics.jl
  - This library is a general rigid body dynamics library that allows for fast calculations of rigid body dynamics. The nice thing about this library is that you can import a urdf and perform any dynamics calculations with the imported urdf easily.
- Tora.jl: https://juliarobotics.org/TORA.jl/stable/
  - This library uses the RigidBodyDynamics.jl library for trajectory optimization specifically for robotic arms.
- RobotDynamics.jl: https://github.com/RoboticExplorationLab/RobotDynamics.jl
  - This library was developed as a common interface for calling systems with forced dynamics. For the example in Appendix A, this library was a wrapper for RigidBodyDynamics to allow it to be used with TrajectoryOptimization.jl and ALTRO.jl mentioned below.
- Dojo.jl: https://github.com/dojo-sim/Dojo.jl
  - A relatively new simulator that can simulate dynamics (haven't explored this one too much)
- MuJoCo.jl: https://github.com/Lyceum/MuJoCo.jl
  - Ports over the MuJoCo simulator to Julia for simulation

**Trajectory Optimization**

- TrajectoryOptimization.jl: https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl
  - This library was developed for defining and evaluating optimal control problems, or trajectory optimization problems. In this library, one can define the objective/cost function, the desired trajectory, the system constraints, and initial guesses for trajectories.

**Solvers**

- ALTRO.jl: https://github.com/RoboticExplorationLab/Altro.jl
  - This library was uses iterative LQR (iLQR) with an Augmented Lagrangian framework to solve trajectory optimization problems defined using the TrajectoryOptimization.jl library. It can solve both nonlinear and linear problems and constraints. This library was developed in the Robot Exploration Lab. More explanation of this library and a tutorial can be found in the github link above and in the links below:
    - https://roboticexplorationlab.org/papers/altro-iros.pdf
    - https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf
- OSQP.jl: https://osqp.org/
  - This library is available in multiple languages, with one of them being Julia. This library is a QP (Quadratic Program) Solver, and is generally used to solve linear optimization problems. It is sometimes faster than ALTRO and sometimes slower than ALTRO depending on the system.
- Ipopt.jl: https://github.com/jump-dev/Ipopt.jl
  - This library is a nonlinear solver for trajectory optimization problem. Personally haven't explored much, but should have similar speeds to the other libraries above.

**ROS Support**

- RobotOS.jl: https://github.com/jdlangs/RobotOS.jl
  - This library is a wrapper for the rospy interface. It has been well tested and is stable.
- ROS.jl: https://github.com/gstavrinos/ROS.jl
  - This library is a wrapper for the roscpp interface. It hasn't existed for a long time so it may have some bugs and stability issues. If you need speed, use this, but be aware of potential bugs.

#### C++

​	C++ is also a good choice for implementing MPC as MPC is computationally expensive to run, so fast runtime is necessary. It can outperform Julia if the code is properly optimized, but it may have slower iteration time than Julia since it is not a scripting language. Use C++ if you need faster runtime performance. Personally haven't explored C++ implementation too much, but if you are using C++ here are some libraries that can be used:

**Dynamics Models/Simulation**

- Pinocchio: https://github.com/stack-of-tasks/pinocchio
  - A general rigid body dynamics library that is one of the fastest dynamics solvers out there
- MuJoCo: https://mujoco.org/
  - Good simulator, which has a more accurate physics simulator than Gazebo, and is one of the top choices if using MPC

**Problem Formulation & Solvers**

- AltroCPP: https://github.com/optimusride/altro-cpp
  - Implements the ALTRO.jl library in C++
- OCS2: https://github.com/leggedrobotics/ocs2
  - A toolbox tailored for Optimal Control for Switched Systems (systems that switch modes). Implements SLQ, iLQR, SQP, PISOC algorithms.
- Crocoddyl: https://github.com/loco-3d/crocoddyl
  - Optimal control library for robots under contact sequences, and it directly uses Pinocchio

## Tips/Tricks

- Many systems are nonlinear, which can be tricky when implementing an MPC on those systems since nonlinear systems can be nonconvex and may not always guarantee a solution.
  - Some ways to help with this issue is to linearize the system if writing your own solver, or use some of the solvers listed above.
  - The solvers for nonlinear systems often rely on a good initial guess to the control input or else the solver cannot solve the optimal control problem (since it may not be convex). For instance, for manipulators, a good intial guess is gravity compensation torques. Generally the initial guess needs to be what keeps the system at equilibrium for all times in the time-horizon.
- Since MPC is computationally expensive, sometimes it is hard to keep convergence times on these solvers low. Here are some tips to help:
  - Make sure to optimize code, such as reducing memory allocation or using constants where you can (C++).
  - Use RK3 to rollout the dynamics instead of RK4 since it requires less dynamics calls while still being decently accurate as long as you aren't predicting too far out.
  - Reduce the time horizon. Having a shorter time horizon drastically reduces computation time, while sacrificing optimality for the entire trajectory.
  - Reduce the number of states or constraints. More states or constraints increases computation time by a large factor.
- If your controller isn't accurate enough, then:
  - Your dynamics model may not be accurate. May need to collect data from the real system or even train a reinforcement learning model on it and use that as the dynamics model.
  - Use RK4 instead of RK3, while sacrificing longer computation time.
  - Increase the time-horizon while sacrificing longer computation time.
  - Reduce the dt (increase frequency) of the MPC. This will make it harder to implement since this means that your solver would have to converge even faster.
- You can have several MPCs running in parallel running different parts of your system.
- Last few lectures of Dr. Zachary Manchester's course on Optimal Control and Reinforcement Learning at CMU have some useful hacks.

## Further Reading & Other Resources

A highly recommended resource for understanding MPC and implementing MPC would be CMU Professor Dr. Zachary Manchester who leads the Robot Exploration Lab. He teaches a course on Optimal Control and Reinforcement Learning, which has all of the lectures, notes, and homeworks open source. See the links below:

Lecture Notes and Homeworks: https://github.com/Optimal-Control-16-745

Lecture Videos: https://www.youtube.com/channel/UCTv1-aM_nUJYjfDGDtTU8JA

Another resource for MPC are MathWorks videos for understanding MPC: https://www.mathworks.com/videos/series/understanding-model-predictive-control.html

Other dynamics libraries that haven't been explored by this article's author are listed below (didn't want to recommend them since I haven't looked too much at them). These were compared against RigidBodyDynamics.jl and Pinocchio in the paper here and performed very comparably: https://people.csail.mit.edu/devadas/pubs/iros2019.pdf.

- RBDL (C++)
- RobCoGen (C++)

## Appendix A: Example of Formulating Optimal Control Problem

### Background Dynamics Equations

The equation below defines the effort variable that we are controlling and minimizing in our optimal control problem, which is the torque applied by our manipulator. In other words, the torque applied by our manipulator is our control input, u, into the system.
$$
u = \tau_{Applied}
$$
The equation below converts the force that the environment exerts on the end-effector to the joint torques that the manipulator experiences from the external force. The external forces will be gathered from an external force/torque sensor mounted on the wrist of the manipulator. Here J is the jacobian matrix of the manipulator.
$$
 \tau_{External} = J^T F_{External}
$$
The equation below is the model of the force exerted onto our end-effector from the environment. We modeled it as a mass-damper system based on literature review, which indicates that reaming/drilling bone is proportional to feed rate (velocity), and not position/depth. The damping coefficient may change as depth increases, which would explain the changing force during some studies. The damping coefficient would have to be estimated at each time step. xdot is the task space velocity of the end-effector.
$$
F_{External} = B_{External}\dot{x}
$$

### States

The following variable represents the states, s, that our system will track. q is the joint positions of the manipulator measured from the joint encoders. qdot is the joint velocities of the manipulator, which will also be measured by the joint encoders. x are the cartesian coordinates of the end-effector in task space. This state will be tracked by the Atracsys 300 Camera. xdot are the cartesian velocities of the end-effector in task space. This will be derived from the joint velocities using equation below. The reason for deriving this from the joint velocities instead of the cartesian position of the end-effector from the camera is because the max velocity will be so small that the camera may not detect it with high resolution. Furthermore, the joint encoders are directly measuring velocity with high resolution, while the camera is not. And lastly, F_External will be the forces/moments that the external environment applies to the end-effector, measured from the Force/Torque sensor mounted on the wrist of the manipulator before the end-effector.
$$
\dot{x} = J\dot{q}
$$

$$
s = \begin{bmatrix}q \\ \dot{q} \\ x \\ \dot{x} \\ F_{External}\end{bmatrix}
$$

### Full System Dynamics

In order to formulate the optimal control problem for our Model Predictive Controller (MPC), we need to first define our system dynamics, deriving the state derivatives, sdot.
$$
\dot{s} = \begin{bmatrix}\dot{q} \\ \ddot{q} \\ \dot{x} \\ \ddot{x} \\ \dot{F}_{External} \end{bmatrix} = \begin{bmatrix} \dot{q} \\ M^{-1}(\tau_{Applied} - \tau_{External} - C\dot{q} - G) \\ J\dot{q} \\ \dot{J}\dot{q} + J\ddot{q} \\ B_{External}(\dot{J}\dot{q} + J\ddot{q})\end{bmatrix}
$$
Since T_Applied is our control input, we can rewrite the equation above as
$$
\dot{s} = \begin{bmatrix}\dot{q} \\ \ddot{q} \\ \dot{x} \\ \ddot{x} \\ \dot{F}_{External} \end{bmatrix} = \begin{bmatrix} \dot{q} \\ M^{-1}(u - \tau_{External} - C\dot{q} - G) \\ J\dot{q} \\ \dot{J}\dot{q} + J\ddot{q} \\ B_{External}(\dot{J}\dot{q} + J\ddot{q})\end{bmatrix}
$$

### System Constraints

There are several constraints on the system that are either imposed because of the limitations of the manipulator itself, or the requirements of safety during surgery. The constraints imposed by the manipulator limitations are joint position and velocity limitations, as well as joint torque limitations. The following represents the limitations of the manipulator, with T_Max as the maximum joint torques of each DoF of our arm (specified by manufacturer):
$$
u = \tau_{Applied} \leq \tau_{Max} \\
q \epsilon q_{Limits} \\
\dot{q} \epsilon \dot{q}_{Limits}
$$
 The following represents the limitations imposed by safety requirements:
$$
||F_{External}|| = ||B_{External}J\dot{q}|| \leq F_{Max}\\
    ||\dot{x}|| = ||J\dot{q}|| \leq \dot{x}_{Max}
$$
Since we are reaming a patient's hip, we want to have a limit on force applied during the operation, as well as the velocity to limit an impact forces with the hip as the reamer first makes contact with the bone.

### Objective/Minimization Function

The objective function is the cost function that the MPC must minimize. Within this cost function is the error between desired state trajectory and actual predicted state trajectory, as well as the error between the final desired state and the predicted final state at the end of the time horizon, H, resulting in equation below.
$$
\min_{s_{k}, u_{k}, k\epsilon[1, H]} \quad & \frac{1}{2}(s_{H} - s_{d})^{T} Q_{H} (s_{H} - s_{d}) + \sum_{k=1}^{H-1}{\frac{1}{2}(s_{k} - s_{d})^{T} Q (s_{k} - s_{d}) + \frac{1}{2}u_{k}^{T} R u_{k}}\\
$$

### The Optimal Control Problem

Combining all the equations above result in the complete Optimal Control Problem for our MPC to solve.
$$
\min_{s_{k}, u_{k}, k\epsilon[1, H]} \quad & \frac{1}{2}(s_{H} - s_{d})^{T} Q_{H} (s_{H} - s_{d}) + \sum_{k=1}^{H-1}\frac{1}{2}(s_{k} - s_{d})^{T} Q (s_{k} - s_{d}) + \frac{1}{2}u_{k}^{T} R u_{k}\\
\textrm{s.t.} \quad & \begin{bmatrix}\dot{q} \\ \ddot{q} \\ \dot{x} \\ \ddot{x} \\ \dot{F}_{External} \end{bmatrix} = \begin{bmatrix} \dot{q} \\ M^{-1}(u - \tau_{External} - C\dot{q} - G) \\ J\dot{q} \\ \dot{J}\dot{q} + J\ddot{q} \\ B_{External}(\dot{J}\dot{q} + J\ddot{q})\end{bmatrix}\\
& u = \tau_{Applied} \leq \tau_{Max} \\
&||F_{External}|| = ||B_{External}J\dot{q}|| \leq F_{Max}  \\
& ||\dot{x}|| = ||J\dot{q}|| \leq \dot{x}_{Max}\\ & q\epsilon q_{Limits}\\ & \dot{q}\epsilon \dot{q}_{Limits}
$$

## References
- CMU Lecture Notes on Optimal Control: https://github.com/Optimal-Control-16-745/lecture-notebooks
- Benchmarking and Workload Analysis of Robot Dynamics Algorithms: https://people.csail.mit.edu/devadas/pubs/iros2019.pdf
- High-Frequency Nonlinear Model Predictive Control of a Manipulator: https://hal.archives-ouvertes.fr/hal-02993058v2/document
- ALTRO MPC Paper Code: https://github.com/RoboticExplorationLab/altro-mpc-icra2021
