---  
# layout: single  
title: Choose a Simulator  
mermaid: true  
---

![all_sim](/assets/images/robotics-project-guide/all_sim.png)

Selecting the appropriate simulator is a pivotal step in any robotics project. Simulators allow you to test algorithms, design robots, and visualize environments without the need for physical hardware, saving both time and resources. This section will guide you through the importance of simulators, help you identify the type of simulator that suits your project's needs, and provide comparisons of popular options across various categories.

By the end of this section, you will understand:  
- The significance of simulators in robotics projects and when to use them.
- How physics simulators work. 
- How to determine the type of simulator you require based on your project's specifications.  
- The pros and cons of various simulators tailored for robotics, machine learning/reinforcement learning, and visualization.  

## Should You Simulate? Why Are Simulators Important in Robotics Projects?

Say that you want to deploy our quadruped robot `Tod` in a shopping mall. You already developed the software that will be running on `Tod` and integrated the peripheral hardware such as a camera for vision. But you're not sure whether everything will work nicely together when you actually put it into use. Will the perception work as intended and recognize obstacles? Will the planner output an optimal path based on the perceived obstacles and goal points? Will the controller move the robot as intended along the path given by the planner?

It's hard to tell, and you most certainly do not want to test this out for the first time in a crowded shopping mall! Here, a simulator can help. 
When developing a software component for your robot, you can use readily available open-source simulators to test out how your software might work in the real world. You can simulate the robot's behavior, such as the sensor measurements and controller gains, and also the environment, such as friction, gravity, wind, etc. Once you simulate the robot in this "fake world", you may detect some problems with your current software stack and choose to improve them. You can keep iterating this until you get the desirable performance. 

Simulators play a crucial role in robotics for several reasons:

### 1. Logistics
- **Cost Efficiency**: Developing and testing in a virtual environment reduces the need for expensive hardware prototypes.  
- **Safety**: Allows for the testing of potentially dangerous scenarios without risk to humans or equipment.  
- **Accessibility**: Facilitates rapid prototyping and debugging, enabling quicker iterations. Allows team members to develop and test code in parallel. 

### 2. Algorithmic Validation and Data Generation
- **Reproducibility & Ground Truth**: Ensures deterministic testing conditions. Simulators provide perfect state estimation, which is invaluable for isolating whether an issue is a sensor error or a controller error.
- **Domain Randomization for Robustness**: Can procedurally vary physical constraints (friction, mass), visual parameters (lighting, textures) and locations of items in an environment to force the model to learn a more robust policy.
- **Exploration**: Simulators allow you to safely test specific situations like sensor failures, extreme weather conditions or complex human-robot/multi-robot interactions.
- **Synthetic Data Generation**: For computer vision tasks, simulators can generate thousands of perfectly labeled images (segmentation masks, depth maps, bounding boxes) in seconds, eliminating the massive bottleneck of manual data labeling.

### 3. Performance Scaling and System Design
- **Massive Parallelization & Time Scaling**: Simultaneously run thousands of simulation instances to compress years of robot learning into hours. Additionally, physics can be "overclocked" to run at multiple times real-time speed or slowed down to sub-millisecond intervals to debug high-speed contact dynamics.
- **Environment Design & Planning**: Helps determine the optimal layout of a environment before anything is built, identifying traffic bottlenecks or "blind spots" for static sensors.
- **Hardware-in-the-Loop (HIL) Testing**: Modern simulators allow you to connect actual embedded microcontrollers or flight controllers to the virtual environment. This enables testing of real embedded C++ code and communication latency while the physics remain virtual.


```mermaid
graph TD
    
    subgraph A[Develop Software]
        subgraph ROS
            J[Custom Software]
        end
    end


    subgraph B[Test in Simulation]
        C[Evaluation]
        subgraph D[Computer]
            subgraph E[Simulator]
                F[Sim. Robot]
                G[Sim. Sensors]
            end
        end

        subgraph H[Computer]
            subgraph I[ROS]
                K[Custom Software]
            end
        end

        I --> E
        E --> I

        D --> |results| C

        
    end

    A --> |deploy| B
    B --> |improve| A

```
(Graph based on [Robotics simulation in Unity is as easy as 1, 2, 3](https://unity.com/blog/engine-platform/robotics-simulation-is-easy-as-1-2-3))

## When to Avoid Simulation & Common Pitfalls

While simulators are powerful tools, they are not a silver bullet. Over-committing to simulation can sometimes drain resources and introduce entirely new categories of bugs. 

### 1. The Sim-to-Real Gap & Modeling Limits
- **Inaccurate Contact Dynamics:** Physics engines use rigid bodies and numerical solvers, often struggling with soft-body deformations, complex contact or high-frequency loops.
- **Unrealistic Sensor Noise:** Real-world sensors suffer from environmental interference, specular reflections and dropout. Perception pipelines tested only on synthetic data often fail entirely when fed noisy, real-world inputs.
- **Unstructured Terrains and Fluids:** While modeling flat indoor floors is easier, accurately simulating unstructured environments such as loose gravel, mud, tall grass or fluid dynamics for underwater robotics is computationally prohibitive and often highly inaccurate.
- **Hardware Wear and Degradation:** Simulators default to factory-perfect conditions. They rarely account for gradual mechanical realities over an operational shift, such as gear backlash, thermal throttling of motors under heavy load or nonlinear battery voltage drops.
- **Human Unpredictability:** Simulating authentic human behavior (unpredictable crowd dynamics, sudden interventions, emergencies) is notoriously difficult, often leading to falsely optimistic safety validations.

### 2. Resource Drains & Project Management
- **The Cost of Simulation Accuracy:** For computer vision tasks, building a high-fidelity environment is a massive undertaking. For contact-rich tasks, it requires significant CPU/GPU resources to achieve highly accurate simulation. If a project's goals are heavily hardware-centric, over-investing in virtual modeling may become a distraction from actual deliverables.
- **Runaway Compute Costs:** Spinning up thousands of virtual environments on commercial cloud infrastructure can quietly accumulate costs that exceed the price of simply building and testing a physical prototype.
- **Skillset Requirements:** Constructing high-fidelity virtual worlds requires expertise akin to video game development. This can force a robotics team to divert critical headcount and focus away from core robotics engineering.

### 3. Software Health & Algorithmic Pitfalls
- **Volatility:** The robotics simulation landscape is highly unstable. Platforms frequently follow a cycle of corporate acquisition, industry-wide migrations or architectural resets that force developers to pivot workflows. Repositories often stagnate and reach EOL, quickly losing compatibility with modern operating systems.
- **Network and Latency Abstraction:** High-level simulators frequently abstract away internal robot communication. They often assume instant data transfer between nodes, masking real-world embedded issues like communication congestion, dropped network packets or variable processing lag.
- **Black Box Effects:** Many commercial physics engines utilize proprietary, closed-source algorithms with undocumented shortcuts designed to maintain real-time rendering speeds.

## How do Physics Simulators Work?

Before we get into what simulators are out there for robotics (and ML/RL), we need to talk about how they simulate the world for our use. What makes them comparable (although not nearly identical) to the real world? 

The answer lies in **physics simulation**. Physics simulators like Gazebo and Isaac Lab provide a flexible and general-purpose framework for modeling a wide range of robotic systems. By representing robots as assemblies of rigid bodies connected by joints, applying appropriate constraints, and utilizing numerical integration methods, these engines can simulate complex dynamics without the need for hand-crafted equations of motion for each specific robot configuration.

### Fundamental Concepts of Physics Simulation

At the core of physics simulation are mathematical models that describe how objects move and interact. Key principles include:

**Generic Rigid-Body Framework**

At the core, modern physics engines treat any object—be it a part of a robot, a vehicle, or a drone—as a rigid body characterized by mass, inertia tensors, and geometry. The fundamental equations involved are the [Newton-Euler equations](https://resources.system-analysis.cadence.com/blog/msa2023-describing-rigid-body-dynamics-using-newton-euler-equations), which describe the translational and rotational motion of rigid bodies under applied forces and torques.

**Constraints and Joints Instead of Analytical Equations of Motion**

Rather than starting with a robot-specific analytical model, physics engines define robots as assemblies of multiple rigid bodies connected by joints and constraints. For example, a humanoid might be modeled as a series of rigid links connected by revolute and spherical joints, each with known mass properties. A quadrotor’s propellers, frame, and any jointed mechanisms are similarly represented as rigid bodies, with thrust and aerodynamic forces applied as external inputs.

The engine sets up a large system of equations representing:

- The motion of each body (Newton-Euler)
- The constraints imposed by joints and contacts (e.g., wheels on the ground, limbs connected to a torso)

These constraints ensure the bodies move in a way consistent with the defined mechanical structure (like a robotic arm’s linkage) or with environmental interactions (such as collisions and friction).

**Numerical Integration Methods**

To simulate the continuous motion of robots, physics engines employ numerical integration techniques to solve the differential equations governing dynamics. While classical integrators like Runge-Kutta (RK4) are possible, many physics engines prefer semi-implicit or symplectic integrators, and often use specialized iterative solvers optimized for stability and speed in real-time or real-time-like simulations. Common techniques include:

- Explicit Euler or Semi-implicit Euler: Very simple, often used as a first pass but may lack stability.
- Verlet integration or symplectic methods: Offer better energy conservation for certain problems.
- Constraint solvers (like Projected Gauss-Seidel or Sequential Impulse methods): Used to handle joint and contact constraints efficiently.

Off-the-shelf physics libraries (e.g., ODE, Bullet, DART, MuJoCo) typically blend these methods to achieve stable, real-time simulation. 

*Example: Fourth-Order Runge-Kutta (RK4) Method*
Fourth-Order Runge-Kutta (RK4) provides greater accuracy by evaluating the derivative at multiple points within each time step and combining them to estimate the next state. This method balances computational efficiency with precision and is widely used in simulations.

**RK4 Algorithm**: Given an initial value problem:
$$
\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0
$$

The RK4 method updates the solution as follows:

$$
\begin{align*}
k_1 &= f(t_n, y_n) \\
k_2 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2} \cdot k_1\right) \\
k_3 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2} \cdot k_2\right) \\
k_4 &= f\left(t_n + h, y_n + h \cdot k_3\right) \\
y_{n+1} &= y_n + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)
\end{align*}
$$

Here, $h$ is the time step, $t_n$ and $y_n$ are the current time and state, respectively, and $k_1, k_2, k_3, k_4$ are intermediate slopes used to compute the next state $y_{n+1}$. 

**Defining Robot Models with URDF**
Now, how do we pass in robot-specific information (settings) to simulators? We can do this via URDFs (in most simulators, at least).  
Users define the physical and kinematic properties of robots using formats like the Unified Robot Description Format (URDF), an XML-based language. URDF allows users to specify:

- **Links**: Define the rigid bodies, including their geometry, mass, and inertia.

- **Joints**: Specify the connections between links, including joint type (e.g., revolute, prismatic), axis of rotation or translation, and limits.

- **Sensors and Actuators**: Describe additional components and their properties.

By providing this information in a URDF file, simulators can accurately construct and simulate the robot's behavior within a virtual environment.

![urdf_to_sim](/assets/images/robotics-project-guide/urdf_to_sim.png)
Here, the `"link"` in the red square on left-side is the URDF section that defines the Unitree G1's pelvis link. Based on the information provided here, the physics engine simulates the pelvis on the humanoid robot.   

**Extended Effects Through Plugins or Force/Torque Models**

For complex scenarios, users can enhance simulations by adding custom force models or plugins. For instance, to better simulate a quadrotor, a plugin might model propeller thrust and drag based on propeller speed. However, the underlying approach remains a numerical, constraint-based solution to the general equations of motion for rigid bodies.

### Examples of Physics Simulators

- **Gazebo**: An open-source robotics simulator that integrates with the Robot Operating System (ROS). Gazebo uses physics engines like ODE (Open Dynamics Engine) to simulate complex interactions between robots and their environments, including accurate modeling of sensors and actuators. 

- **Isaac Lab**: Developed by NVIDIA, Isaac Lab is a high-performance simulator designed for robot learning. It leverages GPU acceleration to perform physics simulations and neural network computations directly on the GPU, enabling the parallel simulation of thousands of environments. This capability significantly speeds up reinforcement learning training times for robotics applications. 

## Determining the Right Simulator for Your Project

Before selecting a simulator, consider the following criteria:

### 1. Ecosystem and Compatibility
* **Integration with Middleware:** Does your project utilize the Robot Operating System (ROS)? Some simulators provide native integration (like Gazebo), which reduces latency and setup complexity. Ensure the simulator supports the specific distribution (e.g. Noetic, Humble, etc.) you are using.
* **Operating System Support:** Platform compatibility is essential. Most high-fidelity simulators are optimized strictly for Linux, while others (like Unity or Unreal Engine) may offer better support for Windows or macOS.
* **Language and Library Support:** Ensure the simulator’s API matches your team's expertise and is compatible with your machine learning libraries and CUDA versions. For example, most modern GPUs require newer drivers only available on higher versions of Ubuntu. However, upgrading the OS often locks out older ROS versions, which in turn can render any hardware dependent on those legacy versions unusable.

### 2. Physical and Visual Fidelity
* **Physics Requirements:** Determine the level of contact friction and articulated dynamics required. For reinforcement learning or complex manipulation, high-fidelity engines like MuJoCo or PhysX are preferred. For simple navigation, a low-fidelity engine or a pure visualizer like RViz may suffice, saving significant compute resources.
* **Visualization and Rendering:** If your project involves computer vision, evaluate the rendering engine. While object detection often only requires standard 3D rendering, tasks involving light-sensitive sensors or photorealistic sim-to-real transfer may require ray-tracing capabilities provided by platforms like NVIDIA Isaac Sim or Unity.
* **Customizability:** Evaluate the ease of importing custom robot models (via URDF, MJCF, or USD) and procedurally generating environments. If you are developing unique hardware, the ability to accurately model its mass properties and joint limits is a non-negotiable requirement.

**Pro-Tip**: You can often circumvent these version mismatches by using Docker containers to run legacy ROS environments on a modern host OS, or by manually installing mainline kernels to support new hardware on older Ubuntu builds. However, the downside is added complexity: both of these may introduce networking issues, GUI X11 forwarding, overhead in GPU passthrough, real-time kernel issues, etc.

### 3. Sustainability and Maintenance
* **Software Health and EOL:** For production environments, integrating simulators that lack updates for modern operating systems or hardware architectures may lead to technical dead-ends. Prioritize software that has reached a stable version and has not been flagged for **End-of-Life (EOL)**.
* **Community and Support:** Check the commit history and issue resolution rate on open-source repositories. A simulator with an active community and transparent roadmap is typically easier to use/debug.
* **Budget and Licensing:** For paid software, account for the total cost of ownership. While many tools are open-source, some require proprietary licenses for commercial use or high-performance cloud compute credits to run at scale.

# Simulators for Robotics

We first go over the simulators that are most often used as the backbone of a robotics workflow.

| **Simulator** | **Physics-Based** | **ROS Integration** | **Cost** | **Computation Speed** | **Supported OS** | **Customizability** |
|---------------|------------------|---------------------|----------|------------------------|------------------|----------------------|
| Gazebo        | Yes (High)       | Best                | Free     | Moderate               | Linux, macOS     | Very High            |
| Isaac Sim     | Yes (Very High)  | Good                | Free     | Resource-Intensive     | Linux, Windows   | High                 |
| CoppeliaSim   | Yes (Multi-engine)| Yes               | Free*/Paid | Moderate             | Windows, macOS, Linux | Very High      |
| Unity         | Adjustable       | With Plugins        | Free*/Paid | Variable             | Windows, macOS, Linux | Very High      |
| Unreal Engine | Adjustable       | With Plugins        | Free      | Resource-Intensive     | Windows, Linux   | Very High            |
| MATLAB/Simulink| Yes (Analytical)| Limited             | Paid     | Moderate               | Windows, macOS, Linux | High           |

---

### [Gazebo](https://gazebosim.org/home)
![gazebo_sim](/assets/images/robotics-project-guide/gazebo_sim.png)

Gazebo is a widely-used open-source robotics simulator that offers robust physics simulation and sensor modeling capabilities. It provides a 3D environment where users can test and develop robots in realistic scenarios. Gazebo's integration with the Robot Operating System (ROS) makes it a standard choice for many robotics projects, facilitating seamless communication between simulation and real-world applications. 

**Note**: It is important to distinguish between Gazebo Classic (versions 1-11) and the modern Gazebo (formerly Ignition). Gazebo Classic reached its official End-of-Life (EOL) in January 2025. You can read more about the switch [here](https://gazebosim.org/about).

![gazebo_timeline](/assets/images/robotics-project-guide/gazebo-timeline.svg)

**Pros**:  
- **ROS Integration**: Seamless compatibility with ROS, making it a standard choice for many robotics projects.  
- **Physics-Based**: Offers realistic physics simulations, including gravity, inertia, and collision detection.  
- **Customizability**: Supports custom robot models and environments.  
- **Cost**: Open-source and free to use.  
- **Supported OS**: Compatible with Linux and macOS.

**Cons**:  
- **Computation Speed**: Can be resource-intensive, potentially leading to slower simulations on less powerful hardware.  
- **Learning Curve**: May require time to master its extensive features and functionalities.
- **Version Split**: Legacy tutorials may use Gazebo Classic, while current projects should use modern Gazebo.

**Resources**:
- **Documentation**: [Gazebo Documentation](https://gazebosim.org/docs)
- **Tutorial**: [Gazebo Tutorials](https://gazebosim.org/docs/latest/tutorials)
- **Community**: [Gazebo Community](https://community.gazebosim.org/)
- **GitHub**: [Gazebo Repository](https://github.com/gazebosim)

---

### [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/)

![isaacsim_sim](/assets/images/robotics-project-guide/isaacsim_sim.png)

NVIDIA Isaac Sim is best when you want high-fidelity physics and strong sensor realism, especially for camera- and LiDAR-heavy workflows. The simulator requires GPU access, RTX-capable hardware and provides support for Windows and Linux, which makes the compute profile very different from lighter simulators. Isaac Lab sits on top of Isaac Sim and is the recommended path for highly parallelizable robot learning.

**Note**: Isaac Gym / IsaacGymEnvs are deprecated and should not be the default choice for new work.

**Pros**:
- **Sensor Realism**: Strong for cameras, LiDAR and other perception-heavy workloads.
- **GPU Acceleration**: Designed around GPU-centric simulation and rendering.
- **Sim-to-Real**: Useful when visual fidelity matters for deployment.
- **Isaac Lab Path**: Extendable learning framework for RL and policy training.

**Cons**:
- **Hardware**: Demands a strong NVIDIA GPU and plenty of memory.
- **Difficulty**: The stack is deeper and more opaque than simpler simulators.
- **Scope**: Better for high-end robot learning pipelines than for quick, lightweight experiments.

**Resources**:
- **Documentation**: [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- **Tutorial**: [Isaac Sim Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/learning.html)
- **Community**: [NVIDIA Isaac Sim Forum](https://forums.developer.nvidia.com/c/isaac/isaac-sim/71)
- **GitHub**: [Isaac Sim (Omniverse)](https://github.com/NVIDIA-Omniverse)

---

### [MATLAB / Simulink (Robotics System Toolbox)](https://www.mathworks.com/products/robotics.html)

![matlab_sim](/assets/images/robotics-project-guide/matlab_sim.jpg)

MATLAB and Simulink are most useful when the simulator is only one part of a broader model-based design workflow. Robotics System Toolbox is explicitly aimed at designing, simulating, testing and deploying manipulator and mobile robot applications. It has co-simulation workflows with Gazebo, Unreal Engine and Simulink 3D Animation. That makes it a common choice for control-heavy validation and simpler workflows.

**Pros**:
- **Control Workflow**: Excellent for trajectory generation, kinematics, dynamics and verification.
- **System-Level Design**: Very strong when the full stack includes perception, control and deployment.
- **Co-Simulation**: Plays well with external simulators rather than replacing them.

**Cons**:
- **Commercial Stack**: Not open-source.
- **3D Realism**: Not the best choice if your primary goal is photorealistic simulation.
- **Ecosystem Lock-In**: Best value appears when you are already using MathWorks tools.

**Resources**:
- **Documentation**: [Robotics System Toolbox Documentation](https://www.mathworks.com/help/robotics/)
- **Tutorial**: [Robotics System Toolbox Examples](https://www.mathworks.com/help/robotics/examples.html)
- **Community**: [MATLAB Answers](https://www.mathworks.com/matlabcentral/answers/)
- **GitHub**: [MATLAB Robotics Examples](https://github.com/mathworks-robotics)

---

### [CoppeliaSim (formerly V-REP)](https://www.coppeliarobotics.com/)

![coppeliasim_sim](/assets/images/robotics-project-guide/CoppeliaSim_sim.jpg)

CoppeliaSim is a versatile robotics simulator known for its extensive feature set and modularity. It supports a wide range of robot models and includes several physics engines for accurate simulation. CoppeliaSim’s integrated development environment allows for rapid prototyping and testing of robotic algorithms. It also offers support for multiple programming languages, enhancing its flexibility for various applications.

**Pros**:  
- **ROS Integration**: Supports ROS, facilitating communication with other ROS nodes.  
- **Physics-Based**: Includes several physics engines for accurate simulations.  
- **Customizability**: Highly customizable with an extensive model library.  
- **Cost**: Free for educational purposes; commercial licenses available.  
- **Supported OS**: Cross-platform support for Windows, macOS, and Linux.

**Cons**:  
- **Learning Curve**: The abundance of features can be overwhelming for beginners.  
- **Computation Speed**: Complex simulations may require substantial computational power.

**Resources**:
- **Documentation**: [CoppeliaSim User Manual](https://manual.coppeliarobotics.com/)
- **Tutorial**: [CoppeliaSim Tutorials](https://www.coppeliarobotics.com/helpFiles/index.html)
- **Community**: [CoppeliaSim Forum](https://forum.coppeliarobotics.com/)
- **GitHub**: [CoppeliaSim Repository](https://github.com/CoppeliaRobotics)

---

### [Unity](https://docs.unity3d.com/) / [Unreal Engine](https://docs.unrealengine.com/)

![unity_unreal_sim](/assets/images/robotics-project-guide/unity_unreal_sim.jpg)

Unity and Unreal Engine are best treated as general real-time engines that can be adapted into robotics simulators. Unity emphasizes simulation for design, testing and training, while Unreal’s simulation pages highlight robotics, training and high-end real-time rendering use cases. Both of themare very attractive for synthetic data generation, visual realism and custom simulation experiences, but they usually need extra robotics-specific glue compared with Gazebo or Isaac Sim.

**Pros**:
- **Rendering**: Excellent visual quality.
- **Flexibility**: Very strong when you need fully custom environments.
- **Synthetic Data**: Good choice when perception and appearance matter.

**Cons**:
- **Robotics Plugins**: Usually requires plugins, bridges or custom code.
- **Physics**: Not as turnkey as robotics-first simulators.
- **Difficulty**: More of a game-engine workflow than a robotics-native one.

**Unity Resources**:
- **Documentation**: [Unity Documentation](https://docs.unity3d.com/)
- **Tutorial**: [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- **Community**: [Unity Forum](https://forum.unity.com/)
- **GitHub**: [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)

**Unreal Engine Resources**:
- **Documentation**: [Unreal Engine Documentation](https://docs.unrealengine.com/)
- **Tutorial**: [Unreal Engine Learning](https://dev.epicgames.com/community/learning)
- **Community**: [Unreal Engine Forum](https://forums.unrealengine.com/)
- **GitHub**: [Unreal Engine Repository](https://github.com/EpicGames/UnrealEngine)

---

## Physics Engines and Simulators for Reinforcement Learning

As robotics increasingly overlaps with reinforcement learning, it helps to separate **physics engines and learning simulators** from full robotics simulators.

| **Simulator** | **Speed** | **Cost** | **Language Support** | **Learning Curve*** | **Parallelizability** | **GPU Support** | **CPU Support** | **Physics Accuracy** | **Visualization Quality** |
|---------------|-----------|----------|----------------------|--------------------|-----------------------|------------------|------------------|----------------------|---------------------------|
| MuJoCo        | High      | Free     | Python, C            | Moderate           | Medium (High w/ MJX)  | Yes              | Yes              | High                 | Moderate                  |
| PyBullet      | Moderate  | Free     | Python, C++          | Low                | Moderate              | Limited          | Yes              | Moderate             | Basic                     |
| Isaac Lab     | Very High | Free     | Python               | High               | Very High             | Required         | Yes              | High                 | High                      |
| Genesis       | High      | Free     | Python               | High               | High (emerging)       | Likely           | Yes              | High            | High                      |

---

### [MuJoCo](https://mujoco.readthedocs.io/)

![mujoco_sim](/assets/images/robotics-project-guide/mujoco_sim.jpg)

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for fast and accurate simulation of articulated structures, making it ideal for reinforcement learning tasks that require high-fidelity physics modeling.

**Pros**:
- **High-Fidelity Physics**: Provides precise simulation of complex dynamics, beneficial for RL applications.
- **Computation Speed**: Efficient simulations with real-time performance.
- **Customizability**: Supports the creation of intricate models and environments.
- **Supported OS**: Compatible with Windows, macOS, and Linux.

**Cons**:
- **Learning Curve**: May require time to master its extensive features and functionalities.

**Resources**:
- **Documentation**: [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- **Tutorial**: [MuJoCo Programming Tutorial](https://mujoco.readthedocs.io/en/latest/programming/)
- **Community**: [MuJoCo Github Discussions](https://github.com/google-deepmind/mujoco/discussions)
- **GitHub**: [MuJoCo Repository](https://github.com/deepmind/mujoco)

---

### [PyBullet](https://github.com/bulletphysics/bullet3)

![pybullet_sim](/assets/images/robotics-project-guide/pybullet_sim.png)

PyBullet is an open-source physics engine that offers real-time simulation of rigid body dynamics, making it suitable for reinforcement learning and robotics research.

**Pros**:
- **Real-Time Physics Simulation**: Accurate and efficient physics modeling.
- **Customizability**: Enables the creation of custom robot models and environments.
- **Cost**: Open-source and free to use.
- **Supported OS**: Cross-platform support.

**Cons**:
- **Visualization**: Basic graphics; may not meet requirements for high-quality rendering.
- **Learning Curve**: The abundance of features can be overwhelming for beginners.

**Resources**:
- **Documentation**: [PyBullet Quickstart Guide](https://pybullet.org/wordpress/quickstart-guide/)
- **Tutorial**: [Hello PyBullet (Official Colab)](https://colab.research.google.com/github/bulletphysics/bullet3/blob/master/examples/pybullet/notebooks/HelloPyBullet.ipynb)
- **Community**: [PyBullet Google Group](https://groups.google.com/g/bulletphysics)
- **GitHub**: [PyBullet Repository](https://github.com/bulletphysics/bullet3)

---

### [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

![isaaclab_sim](/assets/images/robotics-project-guide/isaaclab_sim.jpg)

Isaac Lab is an open-source, GPU-accelerated framework for robot learning, built on top of NVIDIA Isaac Sim. It provides high-fidelity physics simulation using NVIDIA PhysX and photorealistic rendering, making it suitable for training robot policies in simulation before deploying them in real-world scenarios.

**Pros**:
- **High-Fidelity Simulation**: Offers realistic physics and sensor simulations, enhancing the quality of trained models.
- **GPU Acceleration**: Utilizes GPU-based parallelization for efficient large-scale training.
- **Modular Architecture**: Flexible framework supporting various robot embodiments and learning workflows.
- **Cost**: Open-source and free to use.
- **Supported OS**: Compatible with Linux.

**Cons**:
- **Hardware Requirements**: Requires high-end NVIDIA GPUs, which may not be accessible to all users.
- **Learning Curve**: Advanced features and dependencies may present a steeper learning curve for beginners.

**Resources**:
- **Documentation**: [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/)
- **Tutorials**: [Isaac Lab Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- **Community**: [Isaac Sim Forum](https://forums.developer.nvidia.com/c/isaac/isaac-sim/71)
- **GitHub**: [Isaac Lab Repository](https://github.com/isaac-sim/IsaacLab)

---

### [Genesis](https://genesis-embodied-ai.github.io/)

![genesis_sim](/assets/images/robotics-project-guide/genesis_sim.png)

Genesis is an emerging simulator worth mentioning because it tries to combine a physics engine, robotics simulator, photorealistic rendering and generative data tooling in one platform. It is a comprehensive physics simulation platform for robotics, embodied AI and physical AI. This makes it an interesting option for future-facing projects, especially if you want a single system for physics plus data generation.

**Pros**:
- **Future Potential**: Interesting for robotics researchers who want to explore newer simulation stacks.
- **Breadth**: Tries to cover physics, rendering and data generation together.
- **Speed**: Lightweight and fast.

**Cons**:
- **Maturity**: Newer than the established tools above.
- **Workflow Stability**: Best treated as a fast-moving option.
- **Learning Curve**: New APIs and assumptions may take time to understand.

**Resources**:
- **Documentation**: [Genesis Documentation](https://genesis-embodied-ai.github.io/)
- **Tutorial**: [Genesis Examples](https://genesis-embodied-ai.github.io/)
- **Community**: [Genesis GitHub Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions)
- **GitHub**: [Genesis Repository](https://github.com/Genesis-Embodied-AI/Genesis)

---

## Specialized Domain Simulators

These tools are best presented by domain, because their value depends more on the application than on general simulator features.

### Autonomous Driving

- **[CARLA](https://carla.org/)** is open-source and built specifically for autonomous driving research and validation. Its strengths are rich urban assets, sensor control and a strong driving benchmark ecosystem. The main downside is compute cost: it is much heavier than lightweight RL driving environments.
- **[MetaDrive](https://metadriverse.github.io/metadrive/)** is also open-source, but it is deliberately lightweight, modular and flexible. It is a good choice when you want fast experimentation and generalization across road layouts, but it is less photorealistic than CARLA.

### Aerial Robotics

- **[Colosseum (formerly AirSim)](https://codexlabsllc.github.io/Colosseum/)** is open-source, Unreal-based, and supports drones and cars, with SITL/HITL support through PX4 and ArduPilot. Its main advantage is realism and sensor-rich experimentation; its main cost is the heavier Unreal-based setup.
- **[Gazebo](https://gazebosim.org/) + [PX4](https://px4.io/) / [MAVROS](https://docs.px4.io/main/en/ros/mavros_installation.html)** is the more ROS-native route for aerial robotics.

### Indoor Navigation

- **[Habitat-Sim](https://aihabitat.org/)** is an open-source embodied-AI simulator built for photorealistic and efficient navigation-style tasks. It is a strong fit for indoor navigation and mobile manipulation research, but it is more specialized toward embodied AI than toward general rigid-body robotics.

### Maritime Robotics

- **[Stonefish](https://stonefish.readthedocs.io/)** is a strong open-source choice for marine robotics, with a physics engine, lightweight rendering pipeline, hydrodynamics and a ROS interface. The main drawback is that it is domain-specific and therefore less general-purpose than Gazebo or Isaac Sim.
- **[HoloOcean](https://holoocean.readthedocs.io/)** is useful when you want underwater robotics with a modern simulator stack and ROS2 bridge support. It is especially attractive for sonar and underwater-agent workflows, but it sits in a more specialized marine niche than the general robotics tools above.
- **[DAVE](https://github.com/Field-Robotics-Lab/dave)** is a Gazebo-based marine plugin rather than a standalone general simulator. The repository shows active migration work toward new Gazebo versions and ROS 2.

### Space Robotics

- **[NASA Astrobee’s Gazebo simulator](https://github.com/nasa/astrobee)** is a set of Gazebo plug-ins that mimic the real hardware and expose the same ROS interfaces.

### Controls and Optimization

- **[Drake](https://drake.mit.edu/)** is the useful when the simulator is directly part of a control and optimization pipeline. It emphasizes model-based design, multibody dynamics and optimization-based analysis, making it especially strong for controls, planning and verification rather than visual realism (uses MeshCat for rendering).

### Legged Robotics

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** is the strongest high-throughput option here when the goal is policy learning on GPU. It is the clean replacement path for Isaac Gym-style workflows.
- **[RaiSim](https://raisim.com/)** is a legged-robot-focused simulator with well-documented examples. It is more niche than the broad robotics platforms above.

### Soft Robotics

- **[SOFA](https://www.sofa-framework.org/)** is one of the most relevant open-source options for soft robotics because it is explicitly aimed at interactive mechanical simulation with emphasis on biomechanics and robotics. Its open-core LGPL structure and cross-platform support make it a strong research tool, especially where deformable bodies matter.
- **[Genesis](https://genesis-embodied-ai.github.io/)** is worth mentioning here as well because it is designed to handle a wide range of materials and physical phenomena, which makes it interesting for soft-robotics work even though it is newer than SOFA.

### Education

- **[Webots](https://cyberbotics.com/)** is open source, cross-platform, well documented and widely used in education and research, which makes it a strong recommendation for teaching and fast prototyping.


## Further Reading and Resources

For a deeper understanding of physics simulation in robotics, consider exploring the following resources. These materials provide comprehensive insights into the mathematical foundations and practical implementations of physics simulation in robotics.

- [URDF XML Specifications](http://wiki.ros.org/urdf/XML)

- [Runge–Kutta Methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)

- [Gazebo Physics Documentation](https://github.com/gazebosim/gz-physics)

- [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning](https://arxiv.org/pdf/2511.04831)

- [Implementing a Fourth Order Runge-Kutta Method for Orbit Simulation](https://spiff.rit.edu/richmond/nbody/OrbitRungeKutta4.pdf)
