---
date: 2026-04-29
title: Creating Custom Robot Models for MuJoCo
---
MuJoCo (Multi-Joint dynamics with Contact) is a high-performance physics engine widely used in robotics research, reinforcement learning, and biomechanics. Creating a custom robot model involves two complementary phases: designing the physical robot and encoding its geometry and dynamics into a simulation-ready model file. This article covers the full pipeline from physical design to a working MuJoCo simulation, including link geometry, joint layout, actuator selection, and authoring both URDF and MJCF model files. Readers will also find guidance on physics parameter tuning, mesh preparation, and testing strategies. Whether you are building a robot arm, legged system, or aerial manipulator, this article provides the foundational knowledge to get your robot simulating correctly in MuJoCo.

## Physical Model Design

Before writing a single line of XML, you must define the robot's physical architecture. Decisions made here directly affect simulation accuracy, computational cost, and control complexity.

### Kinematic Structure

The kinematic chain is the ordered sequence of rigid bodies (links) connected by joints. Three common topologies appear in practice:

- **Serial chains** — one parent per link; used in robot arms and leg segments.
- **Parallel mechanisms** — closed-loop structures such as delta robots and Stewart platforms.
- **Trees** — branching limbs used in humanoids and quadrupeds.

Start with a Degrees of Freedom (DoF) analysis. Count the independent joints needed for your task. Each revolute or prismatic joint costs one DoF and one control signal.

### Link Geometry

Each link must be described by a collision geometry and optionally a separate visual geometry. MuJoCo supports several primitive shapes natively:

| Shape | MJCF Tag | Common Use Case |
|-------|----------|-----------------|
| Box | `<geom type="box"/>` | Chassis, base plates, cuboid links |
| Sphere | `<geom type="sphere"/>` | Ball joints, feet, sensors |
| Capsule | `<geom type="capsule"/>` | Limbs, fingers |
| Cylinder | `<geom type="cylinder"/>` | Wheels, rollers |
| Ellipsoid | `<geom type="ellipsoid"/>` | Organic approximations |
| Mesh | `<geom type="mesh"/>` | CAD-exported STL/OBJ geometry |

> Prefer capsules over cylinders for link geometry. Capsules are computationally cheaper in contact detection and avoid edge-contact singularities that can destabilize the solver.

### Joint Types

MuJoCo uses different joint type names than URDF. The mapping is:

| URDF Type | MJCF Type | Description |
|-----------|-----------|-------------|
| `revolute` | `hinge` | Rotation about a fixed axis |
| `prismatic` | `slide` | Translation along an axis |
| `floating` | `free` | 6-DoF floating base |
| *(not supported)* | `ball` | 3-DoF spherical joint |
| `fixed` | *(merge bodies)* | Rigid attachment, no DoF |

### Mass and Inertia

Accurate inertial parameters are critical for realistic dynamics. For each link you need the mass (kg), center of mass (CoM) position relative to the link frame, and the inertia tensor expressed as six unique values: `ixx`, `iyy`, `izz`, `ixy`, `ixz`, `iyz`.

For uniform primitive shapes, inertia can be computed analytically. For example, for a solid cylinder of mass `m`, radius `r`, and height `h`:

```python
ixx = iyy = m * (3*r**2 + h**2) / 12
izz = m * r**2 / 2
ixy = ixz = iyz = 0.0
```

For mesh-based links, use MeshLab, SolidWorks, or Fusion 360 to export inertial properties from your CAD model. As a starting point, MuJoCo can auto-compute inertia from geometry by setting `inertiafromgeom="true"` in the `<compiler>` tag.

> The inertia tensor must be positive-definite. Verify these triangle inequalities for every link: `ixx + iyy > izz`, `ixx + izz > iyy`, `iyy + izz > ixx`. Violating them will produce `nan` values or simulation instability.

### Actuator Selection

MuJoCo provides several actuator models. The right choice depends on your hardware:

| Actuator | Best For |
|----------|----------|
| `motor` | DC motors and linear actuators (pure torque/force output) |
| `position` | Hobby servos and Dynamixel (PD position servo) |
| `velocity` | Velocity-controlled motors |
| `general` | Maximum flexibility with custom gain, bias, and dynamics chain |
| `cylinder` | Pneumatic or hydraulic actuators |
| `muscle` | Hill-type muscle model for biomechanics |

## Creating the URDF Model

URDF (Unified Robot Description Format) is an XML format originally developed for ROS. It describes a robot as a tree of links and joints and can be converted to MJCF using MuJoCo's built-in converter. URDF is the recommended starting format when ROS compatibility is needed.

### URDF File Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">

  <!-- Links (rigid bodies) -->
  <link name="base_link"> ... </link>
  <link name="forearm"> ... </link>

  <!-- Joints (connections between links) -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="forearm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>

</robot>
```

### Defining a Link

Each link has three sub-elements: `<inertial>`, `<visual>`, and `<collision>`. All three are optional but recommended for simulation:

```xml
<link name="forearm">

  <inertial>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <mass value="0.5"/>
    <inertia ixx="0.002" iyy="0.002" izz="0.0005"
             ixy="0"     ixz="0"     iyz="0"/>
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_robot/meshes/forearm.stl"
            scale="0.001 0.001 0.001"/>
    </geometry>
    <material name="blue">
      <color rgba="0.2 0.4 0.8 1.0"/>
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <geometry>
      <!-- Simplified capsule instead of full mesh -->
      <capsule radius="0.025" length="0.18"/>
    </geometry>
  </collision>

</link>
```

### Defining a Joint

```xml
<!-- Revolute joint -->
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.09" upper="2.09" effort="20.0" velocity="3.14"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>

<!-- Prismatic joint -->
<joint name="linear_actuator" type="prismatic">
  <parent link="base_link"/>
  <child link="slider"/>
  <axis xyz="0 0 1"/>
  <limit lower="0.0" upper="0.3" effort="50" velocity="0.5"/>
</joint>

<!-- Fixed joint (rigid attachment, no DoF) -->
<joint name="sensor_mount" type="fixed">
  <parent link="end_effector"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.02" rpy="0 -0.3 0"/>
</joint>
```

### Validating Your URDF

Always validate the URDF before converting to MJCF:

```bash
# ROS-based validation
check_urdf my_robot.urdf

# Python: parse with urdfpy (no ROS required)
pip install urdfpy
python -c "from urdfpy import URDF; r = URDF.load('my_robot.urdf'); print(r.link_names)"

# Standalone parser
pip install urdf-parser-py
python -c "
from urdf_parser_py.urdf import URDF
robot = URDF.from_xml_file('my_robot.urdf')
print('Links:', [l.name for l in robot.links])
print('Joints:', [j.name for j in robot.joints])
"
```

## MJCF: Native MuJoCo Format

MJCF is the native description language of MuJoCo. It offers significantly richer features than URDF: tendons, actuator dynamics, contact parameters, sensors, keyframes, and more. You can author MJCF directly or convert from URDF and refine.

### Converting URDF to MJCF

```python
import mujoco

# Load from URDF and save as MJCF
model = mujoco.MjModel.from_xml_path('my_robot.urdf')
mujoco.mj_saveModel(model, 'my_robot.xml')
```

> The URDF-to-MJCF conversion may produce suboptimal contact and collision parameters. Always review the converted MJCF manually — especially the `<contact>`, `<option>`, and `<actuator>` sections.

### MJCF File Structure

```xml
<mujoco model="my_robot">

  <compiler angle="radian" meshdir="meshes/" inertiafromgeom="false"/>

  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81">
    <flag sensornoise="enable"/>
  </option>

  <!-- Default parameters inherited by all elements -->
  <default>
    <joint damping="0.1" frictionloss="0.01" armature="0.001"/>
    <geom condim="3" friction="0.8 0.02 0.001"
          solimp="0.9 0.95 0.001" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>

  <asset>
    <mesh name="forearm_mesh" file="forearm.stl" scale="0.001 0.001 0.001"/>
    <material name="metal" rgba="0.6 0.65 0.7 1"/>
  </asset>

  <worldbody>
    <!-- Robot kinematic tree goes here -->
  </worldbody>

  <actuator>
    <!-- Actuator definitions go here -->
  </actuator>

  <sensor>
    <!-- Sensor definitions go here -->
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
  </keyframe>

</mujoco>
```

### Defining Bodies and Geoms

In MJCF, bodies are nested to define the kinematic tree. Each body contains geoms, joints, and optionally sites (named reference frames):

```xml
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" condim="3"/>

  <body name="base_link" pos="0 0 0.05">
    <geom name="base_col" type="box" size="0.05 0.05 0.025"
          mass="1.0" material="metal"/>
    <site name="base_imu" pos="0 0 0"/>

    <body name="link1" pos="0 0 0.025">
      <joint name="shoulder" type="hinge" axis="0 0 1"
             range="-3.14159 3.14159" damping="0.05" frictionloss="0.005"/>
      <geom name="link1_col" type="capsule"
            fromto="0 0 0  0.2 0 0" size="0.02" mass="0.3"/>
      <site name="elbow_site" pos="0.2 0 0"/>

      <body name="link2" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1"
               range="-2.094 2.094" damping="0.03"/>
        <geom name="link2_col" type="capsule"
              fromto="0 0 0  0.15 0 0" size="0.018" mass="0.2"/>

        <body name="end_effector" pos="0.15 0 0">
          <joint name="wrist" type="hinge" axis="0 0 1"
                 range="-3.14 3.14" damping="0.01"/>
          <geom name="ee_col" type="sphere" size="0.02" mass="0.05"/>
          <site name="ee_site" pos="0.02 0 0"/>
        </body>

      </body>
    </body>
  </body>
</worldbody>
```

### Actuator Configuration

```xml
<actuator>
  <!-- Position servo — matches Dynamixel or hobby servo behavior -->
  <position name="shoulder_pos" joint="shoulder"
            kp="100" kv="10"
            ctrllimited="true" ctrlrange="-3.14 3.14"/>

  <!-- Torque motor with gear ratio -->
  <motor name="elbow_motor" joint="elbow"
         gear="100" ctrllimited="true" ctrlrange="-1 1"/>

  <!-- Velocity servo -->
  <velocity name="wrist_vel" joint="wrist"
            kv="20" ctrllimited="true" ctrlrange="-5 5"/>
</actuator>
```

The `gear` parameter scales the control signal to joint torque and models a gearbox ratio. The `armature` joint parameter adds reflected rotor inertia and should always be set for motorized joints — it significantly improves numerical stability at high gear ratios.

### Sensors

```xml
<sensor>
  <!-- Joint state -->
  <jointpos    name="shoulder_qpos"    joint="shoulder"/>
  <jointvel    name="shoulder_qvel"    joint="shoulder"/>
  <actuatorfrc name="shoulder_torque"  actuator="shoulder_pos"/>

  <!-- Force/torque at a site (models an ATI F/T sensor) -->
  <force  name="ee_force"  site="ee_site"/>
  <torque name="ee_torque" site="ee_site"/>

  <!-- IMU-style sensors -->
  <accelerometer name="base_acc"  site="base_imu"/>
  <gyro          name="base_gyro" site="base_imu"/>

  <!-- Contact normal force -->
  <touch name="ee_touch" site="ee_site"/>

  <!-- Cartesian pose of end-effector frame -->
  <framepos  name="ee_pos"  objtype="site" objname="ee_site"/>
  <framequat name="ee_quat" objtype="site" objname="ee_site"/>
</sensor>
```

### Contact and Friction Parameters

Contact behavior is one of MuJoCo's most tunable aspects. The key parameters are set on `<geom>` elements:

| Parameter | Description |
|-----------|-------------|
| `condim` | Contact dimensionality: 1=normal only, 3=+friction, 4=+rolling, 6=full |
| `friction` | Sliding, torsional, and rolling friction coefficients |
| `solimp` | Constraint impedance (softness): `d0 dwidth width` |
| `solref` | Constraint reference: `timeconst dampratio` |
| `margin` | Contact detection margin in meters |

```xml
<!-- Hard floor -->
<geom name="floor" type="plane" solimp="0.999 0.9999 0.001" solref="0.001 1"/>

<!-- Compliant rubber surface -->
<geom name="pad" type="box" size="0.05 0.05 0.005"
      solimp="0.6 0.8 0.01" solref="0.05 0.8" friction="1.5 0.05 0.02"/>

<!-- Exclude self-collision between adjacent links -->
<contact>
  <exclude body1="link1" body2="link2"/>
  <exclude body1="link2" body2="link3"/>
</contact>
```

## Physics Tuning

### Simulation Options

```xml
<option timestep="0.002"
        integrator="RK4"
        solver="Newton"
        iterations="100"
        tolerance="1e-10"
        gravity="0 0 -9.81"/>
```

| Parameter | Recommendation |
|-----------|----------------|
| `timestep` | 0.001–0.005 s depending on system stiffness |
| `integrator` | `RK4` for accuracy; `implicit` for stiff contact-heavy systems |
| `solver` | `Newton` for accuracy; `PGS` for speed |
| `iterations` | Reduce to 50 for speed; increase for many simultaneous contacts |

### Tendon Constraints

MuJoCo supports tendon-based constraints to model cable-driven robots or to couple joints kinematically:

```xml
<tendon>
  <!-- Couples proximal and distal finger joints (underactuated hand) -->
  <fixed name="coupled_fingers" stiffness="500" damping="5">
    <joint joint="finger_prox" coef="1"/>
    <joint joint="finger_dist" coef="1"/>
  </fixed>

  <!-- Cable routed through pulley sites -->
  <spatial name="drive_cable" width="0.003" stiffness="1000" damping="10">
    <site site="motor_pulley"/>
    <site site="redirect_pulley"/>
    <site site="ee_site"/>
  </spatial>
</tendon>
```

## Workflow: CAD to Simulation

The recommended pipeline for building a MuJoCo model from a physical design is:

1. Design robot geometry in CAD (Fusion 360, SolidWorks, OnShape, or FreeCAD).
2. Export visual meshes at full resolution and simplified collision meshes (target < 500 triangles per link).
3. Export inertial properties per body: mass, center of mass, and inertia tensor.
4. Generate a URDF using an exporter such as `sw2urdf` or `onshape-to-robot`.
5. Validate the URDF with `check_urdf` or `urdfpy`.
6. Convert to MJCF using `mujoco.MjModel.from_xml_path()`.
7. Refine the MJCF: contacts, actuators, sensors, defaults, and keyframes.
8. Test in the MuJoCo viewer and adjust physics parameters iteratively.

### Mesh Preparation

Raw CAD meshes are typically too detailed for collision simulation. Use Open3D to generate simplified collision meshes:

```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("forearm_visual.stl")
simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=200)
o3d.io.write_triangle_mesh("forearm_collision.stl", simplified)
```

For concave parts, decompose into convex hulls using V-HACD before importing into MuJoCo. MuJoCo processes convex-vs-convex contacts significantly faster than concave meshes, and convex decomposition avoids internal contact artifacts.

```bash
pip install pyvhacd
python -c "
import pyvhacd
hulls = pyvhacd.compute('concave_part.stl', max_hulls=16)
for i, h in enumerate(hulls): h.export(f'hull_{i:02d}.stl')
"
```

### Automated URDF Generation with onshape-to-robot

For OnShape-based designs, the `onshape-to-robot` tool automates URDF and mesh export:

```bash
pip install onshape-to-robot

# Place a config.json with your OnShape document ID and API credentials,
# then run from the project directory:
onshape-to-robot .

# Convert the resulting URDF to MJCF:
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('robot.urdf')
mujoco.mj_saveModel(m, 'robot.xml')
"
```

## Testing and Visualization

### MuJoCo Viewer

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("my_robot.xml")
data  = mujoco.MjData(model)

# Passive viewer: you drive the simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Programmatic Testing

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("my_robot.xml")
data  = mujoco.MjData(model)

# Reset to named keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

# Apply control and simulate
data.ctrl[:] = [0.5, -0.3, 0.0]
for _ in range(500):
    mujoco.mj_step(model, data)

# Read sensor outputs
ee_pos   = data.sensor("ee_pos").data.copy()
ee_force = data.sensor("ee_force").data.copy()

# Read body Cartesian state
xpos = data.body("end_effector").xpos
xmat = data.body("end_effector").xmat.reshape(3, 3)

# Inspect active contacts
for i in range(data.ncon):
    c  = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"Contact: {g1} <-> {g2}, dist={c.dist:.4f} m")
```

### Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot explodes on start | Penetrating geometries at t=0 | Check initial `qpos`; add clearance between geoms |
| Simulation goes unstable | Timestep too large | Reduce `timestep` or switch to `RK4` |
| Joints drift without input | Insufficient damping | Increase `damping` and `frictionloss` |
| Contacts jitter | `solimp`/`solref` too stiff | Soften with `solref="0.02 1"` |
| Robot sinks into floor | Collision margin overlap | Reduce geom `margin` |
| `nan` values appear | Invalid inertia tensor | Verify positive-definiteness triangle inequalities |
| Actuator saturates immediately | `ctrlrange` too narrow | Widen `ctrlrange` or reduce `gear` ratio |
| Slow simulation | Too many contact pairs | Add `<exclude>` pairs; simplify collision meshes |

## Summary

Building a custom robot model for MuJoCo involves three main phases. First, the physical model design establishes the kinematic structure, link geometry, joint types, inertial parameters, and actuator choices. Second, the robot is encoded as a URDF or MJCF file — URDF for ROS interoperability, MJCF for full access to MuJoCo's feature set including tendons, advanced actuators, and fine-grained contact parameters. Third, the model is validated and iteratively refined through visualization and programmatic testing.

Key practices to carry forward: use simplified collision geometry (capsules and convex hulls), always set `armature` on motorized joints, tune `solimp`/`solref` per contact surface rather than globally, and define a `<keyframe>` for deterministic simulation resets. For complex robots with many links and contacts, profile the simulation with `mujoco.mj_Timer` to identify bottlenecks before optimizing.

## See Also:
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — a curated library of robot models in MJCF, useful as reference implementations.
- [dm_control](https://github.com/google-deepmind/dm_control) — DeepMind's Python library built on MuJoCo, including environment wrappers for RL.
- [ROS URDF Tutorials](https://wiki.ros.org/urdf/Tutorials) — official ROS documentation for URDF authoring and validation.

## Further Reading
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) — complete documentation for every MJCF element and attribute.
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html) — API reference for the `mujoco` Python package.
- [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) — automated URDF and MJCF export from OnShape assemblies.
- [V-HACD](https://github.com/kmammou/v-hacd) — convex hull decomposition library for preparing concave collision meshes.

## References
- Erez, T., Tassa, Y., and Todorov, E. "Simulation Tools for Model-Based Robotics: Comparison of Bullet, Havok, MuJoCo, ODE, and PhysX." *2015 IEEE International Conference on Robotics and Automation (ICRA)*, 2015.
- MuJoCo Documentation. Google DeepMind. Available: https://mujoco.readthedocs.io/
- Todorov, E., Erez, T., and Tassa, Y. "MuJoCo: A Physics Engine for Model-Based Control." *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2012.
