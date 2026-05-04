---
date: 2026-04-30
title: NVIDIA Isaac Sim - File Structure, Tasks, and Core Concepts
---

<style>
.sidebar__right.sticky {
  max-height: calc(100vh - 2em);
  overflow-y: auto;
}
</style>

This article builds on the [NVIDIA Isaac Sim Setup and ROS2 Workflow](/wiki/simulation/simulation-isaacsim-setup/) guide. It covers project structure, Python scripting, URDF-to-USD conversion, joint control, custom Isaac Lab RL tasks, OmniGraph, custom extensions, synthetic data generation with Replicator, MoveIt 2 integration, and parallel environments for RL training.

## How an Isaac Sim project is organized

Understanding the file and folder layout of Isaac Sim will save you a lot of confusion when starting a new project.

### The Isaac Sim installation directory

After installation, your Isaac Sim root folder (e.g., `~/isaacsim`) contains the following key items:

- **`isaac-sim.selector.sh`**: GUI app launcher
- **`runheadless.sh`**: headless launcher for servers without a monitor
- **`python.sh`**: Isaac Sim's own Python interpreter; always use this instead of your system Python
- **`kit/`**: the Omniverse Kit engine that Isaac Sim is built on top of
- **`exts/`**: Isaac Sim-specific extensions (robot importers, ROS 2 bridge, sensors, etc.)
- **`standalone_examples/`**: official Python script examples; this is the best place to start learning
- **`assets/`**: sample robots, environments, and sensors in USD format
- **`apps/omni.isaac.sim.python.kit`**: the `.kit` config file that defines which extensions load at startup, similar to a requirements file

### Recommended project folder structure

A clean layout for a custom Isaac Sim project looks like this:

````
my_robot_project/
├── robots/
│   └── my_robot/
│       ├── my_robot.urdf          # original robot description
│       └── my_robot.usd           # USD version converted from URDF
├── environments/
│   └── warehouse.usd              # scene file
├── tasks/
│   └── pick_and_place_task.py     # custom Isaac Lab task
├── extensions/
│   └── my_extension/
│       ├── config/
│       │   └── extension.toml     # extension metadata
│       └── my_extension/
│           └── extension.py       # entry point
├── scripts/
│   ├── train.py                   # RL training launcher
│   └── standalone_test.py         # standalone simulation script
└── configs/
    └── my_robot_rl.yaml           # Isaac Lab environment config
````

### What is a USD file?

USD stands for **Universal Scene Description**, a format created by Pixar and used throughout NVIDIA Omniverse. Unlike a flat robot description like URDF, a USD file stores a **hierarchy of "prims"** (short for primitives) - meshes, joints, materials, lights, and cameras - in a tree structure called the **stage**.

The stage is structured like a scene graph:

````
World (root prim)
├── Environment/
│   └── Warehouse (mesh)
└── Robot/
    ├── base_link (rigid body)
    ├── left_wheel (rigid body + revolute joint)
    └── right_wheel (rigid body + revolute joint)
````

Each prim has **attributes** (position, mass, friction) and **relationships** (which material is applied, which physics body it belongs to). USD is designed to be layered and composable - you can load a base robot USD and override just the material in a separate layer without touching the original file.

## Writing Python scripts for Isaac Sim

The GUI is useful for exploration, but for reproducible work you will want to control Isaac Sim from Python scripts.

### Always use Isaac Sim's own Python interpreter

Do not use your system Python. Always launch scripts with:

````bash
~/isaacsim/python.sh my_script.py
````

### Standalone scripts

A standalone script launches Isaac Sim programmatically, runs a simulation loop, and exits. This is the simplest way to automate scenes and robot behavior.

````python
# my_script.py

# Step 1: launch the simulator - this MUST come first
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Step 2: import Isaac Sim modules only AFTER SimulationApp is created
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.robots import Robot

# Step 3: create a World - the main container for your simulation
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Step 4: add a robot from a USD file
robot = world.scene.add(
    Robot(
        prim_path="/World/MyRobot",
        name="my_robot",
        usd_path="/path/to/my_robot.usd",
    )
)

# Step 5: reset initializes physics
world.reset()

# Step 6: run the simulation for 500 steps
for i in range(500):
    world.step(render=True)  # set render=False for faster headless training

simulation_app.close()
````

**Why must imports come after `SimulationApp()`?**
Isaac Sim modules like `omni.isaac.core` are loaded as Omniverse extensions at startup. Importing them before `SimulationApp` is created will fail with a cryptic `ModuleNotFoundError` because the extension manager has not run yet.

### Running code inside the GUI

If Isaac Sim is already open, you can run Python snippets in the built-in script editor at **Window > Script Editor**. This is useful for quick tests, but standalone scripts are easier to version control and share with teammates.

### Key World methods

- **`world.reset()`**: resets the scene and initializes all physics objects; call this once before your loop
- **`world.step(render=True)`**: advances the simulation by one physics timestep
- **`world.scene.add(object)`**: adds a robot, sensor, or object to the scene
- **`world.scene.get_object("name")`**: retrieves a previously added object by name

## Converting URDF to USD

Most robots start life as URDF files. Because Isaac Sim is USD-native, you need to convert before importing.

### Option A: GUI importer

1. Go to **Isaac Utils > Workflows > URDF Importer**
2. Select your `.urdf` file
3. Configure the import settings (see below)
4. Click **Import**

### Option B: Python API

Use this approach when you need to automate conversions or run them as part of a pipeline:

````python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.kit.commands
from omni.isaac.urdf import _urdf

import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False   # keep all joints separate
import_config.fix_base = True              # pin root link to world (use for arms)
import_config.import_inertia_tensor = True
import_config.distance_scale = 1.0         # set to 0.01 if URDF uses centimeters
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

result, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path="/path/to/my_robot.urdf",
    import_config=import_config,
    dest_path="/path/to/output/my_robot.usd",
)

print(f"Import success: {result}, prim at: {prim_path}")
simulation_app.close()
````

### Import settings explained

- **`merge_fixed_joints`**: combines links connected by fixed joints into one rigid body; set `True` to simplify a robot with many fixed joints (faster physics), `False` to keep the full structure
- **`fix_base`**: pins the root link to the world; set `True` for arms, `False` for wheeled robots
- **`distance_scale`**: multiplier for all lengths; set to `0.01` if your URDF is in centimeters
- **`default_drive_type`**: sets whether joints default to position or velocity control; use `POSITION` for arms, `VELOCITY` for wheels
- **`convex_decomp`**: decomposes collision meshes into convex shapes; set `True` for complex mesh geometry

### Common URDF import problems

- **Robot explodes or parts fly apart on start**: inertia tensors are missing or unrealistic; add explicit `<inertial>` tags with correct mass and inertia values to your URDF
- **Robot appears tiny (millimeter scale)**: your URDF uses millimeters; set `distance_scale = 0.001`
- **Meshes appear inside-out**: check the mesh winding order in your CAD tool, or add `<material>` tags to your URDF

## Controlling robots with ArticulationController

### Reading joint states

````python
from omni.isaac.core.articulations import Articulation

robot = Articulation(prim_path="/World/MyRobot")
robot.initialize()  # must call this before reading or writing joints

joint_positions = robot.get_joint_positions()    # numpy array, in radians
joint_velocities = robot.get_joint_velocities()  # numpy array, in rad/s
````

### Sending joint commands

````python
from omni.isaac.core.utils.types import ArticulationAction

# position control - for arms
action = ArticulationAction(
    joint_positions=np.array([0.0, -1.0, 0.5, 0.0, 0.5, 0.0])
)
robot.apply_action(action)

# velocity control - for wheels
action = ArticulationAction(
    joint_velocities=np.array([5.0, -5.0])
)
robot.apply_action(action)
````

### Stiffness and damping

Think of each joint as a spring-damper system. **Stiffness (kp)** controls how hard the joint tries to reach its target position. **Damping (kd)** controls how much it resists velocity. The right values depend on the control mode:

- **Position control (arm joint)**: high stiffness (e.g., 10000), low damping (e.g., 100)
- **Velocity control (wheel)**: stiffness must be 0, high damping (e.g., 1000)
- **Torque/effort control**: both stiffness and damping must be 0

````python
from omni.isaac.core.utils.prims import set_prim_attribute_value

joint_path = "/World/MyRobot/base_to_left_wheel"
set_prim_attribute_value(joint_path, "drive:angular:physics:stiffness", 0.0)
set_prim_attribute_value(joint_path, "drive:angular:physics:damping", 1000.0)
````

## Building a custom task for Isaac Lab

Isaac Lab is a reinforcement learning framework built on top of Isaac Sim. A **Task** defines everything the RL algorithm needs: the scene setup, the observations the agent receives, the rewards it gets, and when an episode ends.

### Installing Isaac Lab

````bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
````

### The five required methods

Every custom task inherits from `DirectRLEnv` and must implement five methods. Here is a complete annotated example for a pick-and-place task:

````python
# tasks/my_pick_task.py

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import torch

class MyPickTask(DirectRLEnv):
    """The robot arm must move its end-effector to a target position."""

    def _setup_scene(self):
        """Load robots and objects into the scene."""
        from omni.isaac.lab.assets import Articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

    def _get_observations(self) -> dict:
        """Return what the agent can see - this is the input to the neural network."""
        obs = torch.cat([
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.target_position.unsqueeze(0),
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Return a scalar reward for each environment. Closer to target = higher reward."""
        ee_pos = self.robot.data.body_pos_w[:, -1, :]
        distance = torch.norm(ee_pos - self.target_position, dim=-1)
        return -distance

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, truncated). Episode succeeds when within 2 cm of target."""
        ee_pos = self.robot.data.body_pos_w[:, -1, :]
        distance = torch.norm(ee_pos - self.target_position, dim=-1)
        terminated = distance < 0.02
        return terminated, torch.zeros_like(terminated)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the specified environments. Randomize the target position each episode."""
        super()._reset_idx(env_ids)
        self.target_position[env_ids] = torch.rand(len(env_ids), 3) * 0.2 + 0.4
````

### Registering and running the task

````python
# train.py
import gymnasium as gym

gym.register(
    id="MyPickTask-v0",
    entry_point="tasks.my_pick_task:MyPickTask",
    kwargs={"cfg_entry_point": "tasks.my_pick_task:MyPickTaskCfg"},
)

env = gym.make("MyPickTask-v0", num_envs=64)
obs, _ = env.reset()

for step in range(10000):
    action = env.action_space.sample()  # replace with your trained policy
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
````

## Environment configs and classes in Isaac Lab

Isaac Lab uses Python **dataclasses** decorated with `@configclass` to define every aspect of an environment in one place - the robot, the scene, the observations, the actions, the rewards, and the termination conditions.
### The `@configclass` decorator

Isaac Lab's config system is built on top of Python dataclasses. The `@configclass` decorator adds extra features like merging configs, overriding fields from YAML files, and printing a clean summary. Every config class in Isaac Lab uses it.

````python
from omni.isaac.lab.utils import configclass

@configclass
class MyTaskCfg:
    # fields go here
    num_envs: int = 4096
    episode_length_s: float = 10.0   # episode length in seconds
````

### `DirectRLEnvCfg`: the top-level config

`DirectRLEnvCfg` is the base config class for all Isaac Lab environments. Your task's config inherits from it and fills in the fields that describe your specific problem.

````python
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.utils import configclass
import math

@configclass
class MyPickTaskCfg(DirectRLEnvCfg):

    # --- simulation settings ---
    decimation: int = 2
    # decimation means the RL policy runs every 2 physics steps.
    # physics runs at 120 Hz, so the policy runs at 60 Hz.

    episode_length_s: float = 10.0   # max episode length in seconds

    # --- spaces ---
    # these must match what _get_observations() and _get_rewards() actually return
    num_observations: int = 15       # size of the observation vector
    num_actions: int = 6             # number of joints the policy controls

    # --- scene ---
    # defined separately - see SceneCfg below
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
````

### `InteractiveSceneCfg`: declaring the scene

Instead of loading robots and objects inside `_setup_scene()` imperatively, Isaac Lab encourages declaring them in a `SceneCfg` dataclass. This makes it easy to swap out robots or environments by just changing the config.

````python
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils

@configclass
class MySceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = sim_utils.GroundPlaneCfg()

    # robot declared here - Isaac Lab loads it automatically
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # {ENV_REGEX_NS} is replaced with the correct prim path for each
        # parallel environment automatically - e.g. /World/envs/env_0/Robot
        spawn=sim_utils.UsdFileCfg(
            usd_path="/path/to/my_robot.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.5,
                "joint_3": 0.5,
                # any joints not listed here default to 0
            },
        ),
        actuators={
            # define how each joint group is driven
            "arm_joints": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=["joint_.*"],  # regex matches joint_1, joint_2, etc.
                stiffness=10000.0,
                damping=100.0,
            ),
        },
    )

    # a simple rigid object the robot will pick up
    object: sim_utils.RigidObjectCfg = sim_utils.RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=sim_utils.RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )
````

### `ObservationsCfg`: declaring observations

For the `ManagerBasedRLEnv` style (see below), observations are declared as a config rather than written as Python code. Each term in the config is a function that computes part of the observation vector.

````python
from omni.isaac.lab.managers import ObservationGroupCfg, ObservationTermCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations that go into the policy network."""

        # joint positions - shape (num_joints,)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)

        # joint velocities - shape (num_joints,)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

        # end-effector position relative to target - shape (3,)
        ee_to_target = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
        )

    policy: PolicyCfg = PolicyCfg()
````

### `ActionsCfg`: declaring actions

Actions define how the policy's output numbers map to robot commands. Isaac Lab provides built-in action terms for joint position, joint velocity, and end-effector pose control.

````python
from omni.isaac.lab.managers import ActionTermCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass

@configclass
class ActionsCfg:

    # the policy outputs 6 numbers → they become target joint positions
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],   # regex to match all arm joints
        scale=1.0,                  # multiply policy output by this value
        use_default_offset=True,    # add default joint pos as offset
    )
````

### `RewardsCfg`: declaring rewards

Instead of writing reward logic inside `_get_rewards()`, you can declare reward terms as a config. Each term is a function with a weight. Isaac Lab sums all terms automatically.

````python
from omni.isaac.lab.managers import RewardTermCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass

@configclass
class RewardsCfg:

    # reward for getting the end-effector close to the target
    reaching_target = RewardTermCfg(
        func=mdp.l2_distance_to_target,
        weight=-1.0,        # negative = penalize distance
        params={"target_cfg": ...},
    )

    # small penalty for large joint accelerations (encourages smooth motion)
    joint_acc_penalty = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-0.0001,
    )

    # bonus reward for successfully grasping
    grasp_success = RewardTermCfg(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={"minimal_height": 0.04},
    )
````

### `TerminationsCfg`: declaring done conditions

Similarly, episode termination conditions can be declared as config rather than code:

````python
from omni.isaac.lab.managers import TerminationTermCfg
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass

@configclass
class TerminationsCfg:

    # end episode if max time is reached
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # end episode if robot base tips over
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": ..., "threshold": 1.0},
    )
````

### `DirectRLEnv` vs `ManagerBasedRLEnv`: which to use?

Isaac Lab provides two base environment classes. Choosing the right one depends on how much flexibility you need.

- **`DirectRLEnv`**: you write `_get_observations()`, `_get_rewards()`, and `_get_dones()` directly in Python; simpler to understand and debug; best for custom research environments where the reward function is complex or unusual
- **`ManagerBasedRLEnv`**: observations, actions, rewards, and terminations are all declared as configs using `ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, and `TerminationsCfg`; more modular and reusable; best for standard manipulation and locomotion tasks where you want to mix and match existing Isaac Lab MDP terms

````python
# DirectRLEnv style - everything in Python methods
class MyTask(DirectRLEnv):
    cfg: MyTaskCfg

    def _get_rewards(self) -> torch.Tensor:
        # write any logic you want here
        distance = torch.norm(self.ee_pos - self.target, dim=-1)
        return -distance * self.cfg.reward_scale


# ManagerBasedRLEnv style - rewards declared in config
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

class MyTask(ManagerBasedRLEnv):
    cfg: MyTaskCfg
    # no need to write _get_rewards or _get_observations -
    # the managers handle everything from the config
````

### Overriding config values at runtime

Because configs are Python dataclasses, you can override any field when creating the environment - no need to edit the config file itself:

````python
cfg = MyPickTaskCfg()
cfg.num_envs = 256           # override for a quick test
cfg.episode_length_s = 5.0   # shorter episodes
cfg.scene.env_spacing = 3.0  # more space between envs

env = MyPickTask(cfg=cfg)
````

You can also override from a YAML file, which is useful for running experiments with different hyperparameters without touching code:

````yaml
# experiment_256.yaml
num_envs: 256
episode_length_s: 5.0
scene:
  env_spacing: 3.0
````

````python
from omni.isaac.lab.utils.io import load_cfg_from_registry
cfg = MyPickTaskCfg()
cfg = load_cfg_from_registry(cfg, "experiment_256.yaml")
````

## OmniGraph

### Graph types

Isaac Sim supports two graph types:

- **Action Graph**: general purpose; triggered by events or on every simulation tick; use this for almost all robotics work
- **Push Graph**: data-driven; executes whenever upstream data changes; use for purely reactive data pipelines

### A typical robot control graph

A graph that publishes odometry and accepts velocity commands looks like this:

````
[On Playback Tick]
        |
        |---> [Isaac Compute Odometry] ---> [ROS2 Publish Odometry]
        |
        |---> [ROS2 Subscribe Twist] ---> [Differential Controller] ---> [Articulation Controller]
        |
        +---> [Isaac Read Simulation Time] ---> [ROS2 Publish Clock]
````

### Building a graph in Python

Graphs can be built entirely in code rather than through the GUI, which makes them reproducible across teammates:

````python
import omni.graph.core as og

(graph, _, _, _) = og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick",       "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime",  "omni.isaac.core_nodes.IsaacReadSimulationTime"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick",               "ReadSimTime.inputs:execIn"),
            ("ReadSimTime.outputs:execOut",        "PublishClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("PublishClock.inputs:topicName", "/clock"),
        ],
    },
)
````

### Debugging graphs

- **Nodes are grey and not executing**: check that the simulation is playing (press Space) and that an `On Playback Tick` node is connected upstream of everything
- **ROS 2 topics are not appearing**: make sure `omni.isaac.ros2_bridge` is enabled in **Window > Extensions** and that you sourced your ROS 2 workspace in the terminal before launching Isaac Sim
- **Data type mismatch error on a connection**: check the output type of the upstream node and insert a type conversion node between mismatched connections

## Writing your own Isaac Sim extension

An extension is a Python package that Isaac Sim loads at startup. This lets you add custom menus, UI panels, or background services without modifying Isaac Sim's source code.

### Minimum extension structure

````
my_extension/
├── config/
│   └── extension.toml
└── my_extension/
    ├── __init__.py
    └── extension.py
````

### The `extension.toml` file

````toml
[package]
title = "My Robot Extension"
description = "Custom tools for my robot project"
version = "1.0.0"

[[python.module]]
name = "my_extension"
````

### The `extension.py` file

Every extension must implement `omni.ext.IExt`. The `on_startup` method runs when the extension loads, and `on_shutdown` runs when it unloads.

````python
import omni.ext
import omni.ui as ui

class MyExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        self._window = ui.Window("My Robot Tools", width=300, height=200)
        with self._window.frame:
            with ui.VStack():
                ui.Label("Robot Control Panel")
                ui.Button("Reset Robot", clicked_fn=self._reset_robot)

    def on_shutdown(self):
        if self._window:
            self._window.destroy()
            self._window = None

    def _reset_robot(self):
        from omni.isaac.core import World
        world = World.instance()
        if world:
            world.reset()
````

### Loading your extension

Add the parent directory of your extension to Isaac Sim's search path before launching:

````bash
export ISAAC_USER_APPS=/path/to/my_extension_parent_dir
~/isaacsim/isaac-sim.selector.sh
````

Then in Isaac Sim go to **Window > Extensions**, search for your extension name, and toggle it on.

## Synthetic data generation with Replicator

NVIDIA Replicator is a framework built into Isaac Sim for generating labeled training datasets. You randomize the virtual environment and render thousands of annotated images instead of collecting them by hand.

### What can be randomized

- Object positions, rotations, and scales
- Lighting color, intensity, and position
- Camera pose and field of view
- Material textures and surface colors
- Background environments
- Random distractor objects placed throughout the scene

### A minimal Replicator script

This script generates 1000 images with bounding box labels and saves them to disk:

````python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep

with rep.new_layer():

    camera = rep.create.camera(position=(0, 0, 2), look_at=(0, 0, 0))
    cubes = rep.create.cube(count=5)

    with rep.trigger.on_frame(num_frames=1000):

        with cubes:
            rep.randomizer.scatter_2d(surface_prims=["/World/Floor"])

        with rep.create.light(light_type="Sphere"):
            rep.randomizer.color(
                colors=rep.distribution.uniform((0.5, 0.5, 0.5), (1, 1, 1))
            )
            rep.randomizer.position((-3, 3, 3), (3, 3, 5))

    render_product = rep.create.render_product(camera, (640, 480))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="/tmp/synthetic_data",
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])

rep.orchestrator.run()
simulation_app.close()
````

### Supported output writers

- **`BasicWriter`**: RGB images, depth maps, 2D/3D bounding boxes, segmentation masks
- **`KittiWriter`**: KITTI format for LiDAR and camera datasets
- **`COCOWriter`**: COCO JSON format for object detection training

## MoveIt 2 integration for manipulation

### Architecture

````
MoveIt 2 (ROS 2 node)
    |   <-- /joint_states        (Isaac Sim publishes these)
    |   --> /joint_trajectory    (MoveIt 2 sends planned paths here)
    |
Isaac Sim
    └── Robot arm (ArticulationController receives trajectory commands)
````

### Step 1: publish joint states from Isaac Sim

In your Action Graph, add a **ROS2 Publish Joint State** node connected to your robot's articulation prim. This sends current joint positions and velocities to the `/joint_states` topic that MoveIt 2 reads.

### Step 2: subscribe to trajectory commands

Add a **ROS2 Subscribe Joint Trajectory** node and wire its output into an **Articulation Controller** node. MoveIt 2 can now send planned trajectories directly to the simulated arm.

### Step 3: publish the robot description

````bash
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(cat my_robot.urdf)"
````

### Step 4: launch MoveIt 2

````bash
ros2 launch my_robot_moveit_config move_group.launch.py
````

To generate a MoveIt 2 config package for your robot from scratch, use the Setup Assistant:

````bash
ros2 launch moveit_setup_assistant setup_assistant.launch.py
````

Load your URDF, define planning groups (for example, "arm" and "gripper"), configure self-collision pairs, and export the config package.

## Parallel environments with the Cloner API

Isaac Lab can run thousands of robot environments simultaneously on a single GPU. Instead of waiting days for a single environment to collect data, you run 4096 environments in parallel and finish in hours.

### Setting the number of environments

When you set `num_envs > 1` in your task config, Isaac Lab automatically stamps out N copies of your template environment, each offset in space so they do not overlap. All instances share the same GPU physics simulation.

````python
class MyTaskCfg(DirectRLEnvCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5   # meters between environment centers
````

### Writing vectorized code

Because all environments run in parallel, your task code must operate on tensors rather than looping over individual environments:

````python
# WRONG - never loop over environments in a task method
for env_id in range(self.num_envs):
    reward[env_id] = compute_reward(obs[env_id])

# CORRECT - compute reward for all environments at once
ee_positions = self.robot.data.body_pos_w[:, -1, :]  # shape: (num_envs, 3)
distances = torch.norm(ee_positions - self.targets, dim=-1)  # shape: (num_envs,)
rewards = -distances
````

### Partial reset

When some environments finish their episode, only those need to be reset - the others keep running without interruption:

````python
def _reset_idx(self, env_ids: torch.Tensor):
    """env_ids is a 1D tensor containing the indices of environments to reset."""
    super()._reset_idx(env_ids)
    default_pos = self.robot.data.default_joint_pos[env_ids]
    noise = torch.rand_like(default_pos) * 0.2 - 0.1
    self.robot.set_joint_position_target(default_pos + noise, env_ids=env_ids)
````

## Version compatibility

Isaac Sim and Isaac Lab are released separately and must be matched. Using mismatched versions is the most common source of cryptic import errors.

| Isaac Lab Version | Compatible Isaac Sim Version |
|---|---|
| 2.0.x | 4.5.x |
| 1.4.x | 4.2.x |
| 1.3.x | 4.1.x |
| 1.2.x | 4.0.x |

Check your installed versions:

````bash
# Isaac Sim version
cat ~/isaacsim/VERSION

# Isaac Lab version
cd /path/to/IsaacLab && git describe --tags
````

When following online tutorials, always check which Isaac Sim version they were written for. The Python API has breaking changes between 4.2 and 4.5.

## Troubleshooting

- **Startup takes 10–30 minutes the first time**: this is normal; Isaac Sim compiles and caches shaders on first launch; do not interrupt it; subsequent launches take 10–60 seconds

- **GPU out of memory**
````
CUDA out of memory
````
Reduce `num_envs` in your task config until it fits in your VRAM. Also set `render=False` in `world.step()` during training.

- **ROS 2 topics not appearing**: source your ROS 2 workspace before launching Isaac Sim and confirm `omni.isaac.ros2_bridge` is enabled in **Window > Extensions**:
````bash
source /opt/ros/humble/setup.bash
~/isaacsim/isaac-sim.selector.sh
````

- **USD load failed / missing mesh files**: USD uses absolute paths by default; if you move your project folder, asset paths break; right-click the affected prim in the Stage panel and select **Make Relative Path**

- **Physics instability / robot shaking at rest**: increase solver iterations at **Edit > Physics Settings > Solver Position Iterations** (try 8 or 16); also verify that mass and inertia values in your URDF are physically realistic

- **`ImportError: No module named 'omni'`**: you are using your system Python instead of Isaac Sim's Python:
````bash
# wrong
python3 my_script.py

# correct
~/isaacsim/python.sh my_script.py
````

## References

- [Isaac Sim Documentation 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Standalone Examples](https://github.com/isaac-sim/IsaacSim/tree/main/standalone_examples)
- [NVIDIA Replicator Documentation](https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/)
- [OpenUSD Introduction](https://openusd.org/release/intro.html)
