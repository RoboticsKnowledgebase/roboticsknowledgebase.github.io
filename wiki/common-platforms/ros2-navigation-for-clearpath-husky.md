---
date: 2023-12-07
title: ROS 2 Navigation with the Clearpath Husky
---

ROS 2 has been growing significantly in the robotics community, with its newer features, better security, and improved performance. However, the transition from ROS 1 to ROS 2 is not straightforward. Many packages are still not migrated to ROS 2, and the existing packages are not fully adapted to ROS 2. Particularly, those dedicated to the Clearpath Husky, have not completed the migration to ROS 2. Although there are existing GitHub repositories for Clearpath Robots, specifically the Husky, they are not fully adapted to ROS 2, with some even falling into disrepair.

For instance, in the [Husky](https://github.com/husky/husky/tree/humble-devel) repository, a clear message states that **"For ROS 2 Humble, this repository is no longer used. Please visit [clearpath_common](https://github.com/clearpathrobotics/clearpath_common)."** However, the **clearpath_common** repository is still a work in progress, lacking comprehensive documentation. As a result, the practical solution involves utilizing the ROS 1 packages for the Husky and establishing a bridge to ROS 2.

This tutorial aims to guide you through the process of setting up the ROS 1 navigation stack on the Clearpath Husky and seamlessly connecting it to ROS 2. It assumes a foundational understanding of both ROS 1 and ROS 2.

## ROS 1 - ROS 2 Bridge
To configure the Clearpath Husky hardware, we will be using the [husky_robot](https://github.com/husky/husky_robot) repository. This repository contains the ROS 1 packages for the Husky, including the navigation stack. To connect the ROS 1 packages to ROS 2, we will be using the [ros1_bridge](https://github.com/ros2/ros1_bridge) package. Detailed instruction on how to setup this is provided in [this tutorial](https://roboticsknowledgebase.com/wiki/interfacing/ros1_ros2_bridge/) on the Robotics Knowledgebase. Once the bridge is established, we can proceed to configure the Husky using the following steps.
```
# Install Husky Packages
apt-get update && apt install ros-noetic-husky* -y
mkdir -p /home/ros1_ws/src && cd /home/ros1_ws/src
git clone --single-branch --branch noetic-devel https://github.com/husky/husky_robot.git
apt-get update && cd /home/ros1_ws && source /opt/ros/noetic/setup.bash && rosdep install --from-paths src --ignore-src -r -y --rosdistro=noetic
apt update && apt install ros-noetic-roslint ros-noetic-diagnostics -y
cd /home/ros1_ws && source /opt/ros/noetic/setup.bash && catkin_make
```
NOTE: `ros1_ws` is the name of the ROS 1 workspace. You can name it anything you want.

## Navigation Stack
We will be using the ROS 2 Nav2 stack for navigation. To install the [Nav2](https://navigation.ros.org/index.html) stack, follow the instructions [here](https://navigation.ros.org/getting_started/index.html).

[nav2_bringup](https://github.com/ros-planning/navigation2/tree/main/nav2_bringup) package provides a launch file to launch the navigation stack for the turtlebot as mentioned in the [tutorial](https://navigation.ros.org/getting_started/index.html). In the Nav2 tutorial, they use `amcl` for localization, and have a collision checker to check for collisions. However, in this tutorial, we assume that the robot is localized using any ROS package (e.g., `robot_localization`), and the odometry of the robot is provided in the topic `/odom`. We also assume that the robot is not moving in a dynamic environment, so we will not be using the collision checker or generating any local costmap, and we will be using the static map provided in the topic `/map`.

Moreover, to explore further options other than the default **DWB Controller**, we will be using the **Regulated Pure Pursuit Controller**, and for the path planning we will be using the **SMAC Planner** instead of the default **NavfnPlanner**. The parameters for these plugins are provided in the [Nav2 documentation](https://navigation.ros.org/configuration/index.html).

We will be creating a new package `husky_nav2_bringup`, similar to the `nav2_bringup` package, to launch the navigation stack for the Husky. The package should have the following structure:
```
husky_nav2_bringup
├── params
│   └── nav2_params.yaml
├── launch
│   └── nav_bringup_launch.py
├── package.xml
└── CMakelists.txt
```

The `nav2_params.yaml` file should contain the following parameters:
```
bt_navigator:
  ros__parameters:
    # use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    # 'default_nav_through_poses_bt_xml' and 'default_nav_to_pose_bt_xml' are use defaults:
    # nav2_bt_navigator/navigate_to_pose_w_replanning_and_recovery.xml
    # nav2_bt_navigator/navigate_through_poses_w_replanning_and_recovery.xml
    # They can be set here or via a RewrittenYaml remap from a parent launch file to Nav2.
    # default_nav_to_pose_bt_xml: ""
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_goal_updated_condition_bt_node
      - nav2_globally_updated_goal_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_goal_updated_controller_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_back_up_cancel_bt_node
      - nav2_assisted_teleop_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node
      - nav2_is_battery_charging_condition_bt_node

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    controller_plugins: ["FollowPath"]
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker", "precise_goal_checker"]
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    odom_topic: "odom"

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.1
      movement_time_allowance: 10.0
    general_goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.15 # 15 cm
      yaw_goal_tolerance: 0.25 # 0.25 rad
      stateful: False
    precise_goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.05 # 10 cm
      yaw_goal_tolerance: 0.1 # 0.1 rad
      stateful: False
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.16
      lookahead_dist: 0.4
      use_velocity_scaled_lookahead_dist: false
      # min_lookahead_dist: 0.3                             # reqd only if use_velocity_scaled_lookahead_dist is true
      # max_lookahead_dist: 0.9                             # reqd only if use_velocity_scaled_lookahead_dist is true
      # lookahead_time: 1.5                                 # reqd only if use_velocity_scaled_lookahead_dist is true
      transform_tolerance: 0.1
      min_approach_linear_velocity: 0.05                    # The minimum velocity (m/s) threshold to apply when approaching the goal to ensure progress. Must be > 0.01.
      approach_velocity_scaling_dist: 0.4                   # The distance (m) left on the path at which to start slowing down. Should be less than the half the costmap width.                  
      use_collision_detection: false
      use_regulated_linear_velocity_scaling: true
      regulated_linear_scaling_min_radius: 1.5              # reqd only if use_regulated_linear_velocity_scaling is true
      regulated_linear_scaling_min_speed: 0.15              # The minimum speed (m/s) for which any of the regulated heuristics can send, to ensure process is still achievable even in high cost spaces with high curvature. Must be > 0.1.
      use_cost_regulated_linear_velocity_scaling: false
      use_fixed_curvature_lookahead: false
      # curvature_lookahead_dist: 0.25                      # reqd only if use_fixed_curvature_lookahead is true
      allow_reversing: true
      use_rotate_to_heading: false                          # either allow_reversing or use_rotate_to_heading must be true
      # rotate_to_heading_min_angle: 0.785                  # reqd only if use_rotate_to_heading is true
      # rotate_to_heading_angular_vel: 1.8                  # reqd only if use_rotate_to_heading is true
      # max_angular_accel: 3.2                              # reqd only if use_rotate_to_heading is true
      max_robot_pose_search_dist: 10.0

global_costmap:
  global_costmap:
    ros__parameters:
      # use_sim_time: false
      footprint_padding: 0.01 # default
      footprint: "[[0.5, 0.35], [0.5, -0.35], [-0.5, -0.35], [-0.5, 0.35]]"     # footprint of Husky
      global_frame: map
      lethal_cost_threshold: 100 # default
      robot_base_frame: base_link
      publish_frequency: 1.0
      update_frequency: 1.0
      # robot_radius: 0.22
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
        enabled: true
        subscribe_to_updates: true
        transform_tolerance: 0.1
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        enabled: true
        inflation_radius: 0.55
        cost_scaling_factor: 5.0
        inflate_unknown: false
        inflate_around_unknown: true
      always_send_full_costmap: True

planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]

    GridBased:
      plugin: "nav2_smac_planner/SmacPlannerHybrid"
      downsample_costmap: false               # whether or not to downsample the map
      downsampling_factor: 1                  # multiplier for the resolution of the costmap layer (e.g. 2 on a 5cm costmap would be 10cm)
      tolerance: 0.25                         # dist-to-goal heuristic cost (distance) for valid tolerance endpoints if exact goal cannot be found.
      allow_unknown: true                     # allow traveling in unknown space
      max_iterations: 1000000                 # maximum total iterations to search for before failing (in case unreachable), set to -1 to disable
      max_on_approach_iterations: 1000        # Maximum number of iterations after within tolerances to continue to try to find exact solution
      max_planning_time: 1.0                  # max time in s for planner to plan, smooth
      motion_model_for_search: "REEDS_SHEPP"  # Hybrid-A* Dubin, Redds-Shepp
      angle_quantization_bins: 72             # Number of angle bins for search
      analytic_expansion_ratio: 3.5           # The ratio to attempt analytic expansions during search for final approach.
      analytic_expansion_max_length: 3.0      # For Hybrid/Lattice nodes: The maximum length of the analytic expansion to be considered valid to prevent unsafe shortcutting
      minimum_turning_radius: 2.5             # minimum turning radius in m of path / vehicle
      reverse_penalty: 1.0                    # Penalty to apply if motion is reversing, must be => 1
      change_penalty: 0.0                     # Penalty to apply if motion is changing directions (L to R), must be >= 0
      non_straight_penalty: 1.2               # Penalty to apply if motion is non-straight, must be => 1
      cost_penalty: 2.0                       # Penalty to apply to higher cost areas when adding into the obstacle map dynamic programming distance expansion heuristic. This drives the robot more towards the center of passages. A value between 1.3 - 3.5 is reasonable.
      retrospective_penalty: 0.015
      lookup_table_size: 20.0                 # Size of the dubin/reeds-sheep distance window to cache, in meters.
      cache_obstacle_heuristic: false         # Cache the obstacle map dynamic programming distance expansion heuristic between subsiquent replannings of the same goal location. Dramatically speeds up replanning performance (40x) if costmap is largely static.
      debug_visualizations: false             # For Hybrid nodes: Whether to publish expansions on the /expansions topic as an array of poses (the orientation has no meaning) and the path's footprints on the /planned_footprints topic. WARNING: heavy to compute and to display, for debug only as it degrades the performance.
      use_quadratic_cost_penalty: False
      downsample_obstacle_heuristic: True
      allow_primitive_interpolation: False
      smooth_path: True                       # If true, does a simple and quick smoothing post-processing to the path

      smoother:
        max_iterations: 1000
        w_smooth: 0.3
        w_data: 0.2
        tolerance: 1.0e-10
        do_refinement: true
        refinement_num: 2

behavior_server:
  ros__parameters:
    global_costmap_topic: global_costmap/costmap_raw
    global_footprint_topic: global_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait", "assisted_teleop"]
    spin:
      plugin: "nav2_behaviors/Spin"
    backup:
      plugin: "nav2_behaviors/BackUp"
    drive_on_heading:
      plugin: "nav2_behaviors/DriveOnHeading"
    wait:
      plugin: "nav2_behaviors/Wait"
    assisted_teleop:
      plugin: "nav2_behaviors/AssistedTeleop"
    global_frame: map
    robot_base_frame: base_link
    transform_timeout: 0.1
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2
```

The `nav_bringup_launch.py` file should contain the following:
```
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()

    # Nav2 Planner Server and Controller Server
    configured_params = os.path.join(get_package_share_directory('lx_nav2'), 'config', 'params.yaml')
    tf_remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]
    tf_and_cmdvel_remappings = [('/tf', 'tf'),
                    ('/tf_static', 'tf_static'),
                     ('cmd_vel', 'cmd_vel_nav')]

    lifecycle_nodes = [
                       'controller_server',
                       'planner_server',
                       'behavior_server',
                       'bt_navigator'
                       ]

    nav2_controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[configured_params],
        remappings=tf_and_cmdvel_remappings
    )

    nav2_smoother = Node(
        package='nav2_smoother',
        executable='smoother_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[configured_params],
        remappings=tf_remappings
    )
    
    nav2_planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[configured_params],
        remappings=tf_remappings
    )

    nav2_behaviors = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[configured_params],
        remappings=tf_remappings
    )

    nav2_bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[configured_params],
        remappings=tf_remappings
    )

    nav2_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{'use_sim_time': False}, {'autostart': True}, {'node_names': lifecycle_nodes}],
        remappings=tf_remappings
    )

    ld.add_action(nav2_controller_server)
    ld.add_action(nav2_smoother)
    ld.add_action(nav2_planner_server)
    ld.add_action(nav2_behaviors)
    ld.add_action(nav2_bt_navigator_node)
    ld.add_action(nav2_lifecycle_manager)

    return ld
```

Once the package is built, we can launch the husky_nav2_bringup package using the following command:
```
ros2 launch husky_nav2_bringup nav_bringup_launch.py
```

To navigate the Husky to the desired pose within the map, we will be usingt the `NavigateToPose` action server provided by Nav2. The action definition is as follows:
```
#goal definition
geometry_msgs/PoseStamped pose
string behavior_tree
---
#result definition
std_msgs/Empty result
---
#feedback definition
geometry_msgs/PoseStamped current_pose
builtin_interfaces/Duration navigation_time
builtin_interfaces/Duration estimated_time_remaining
int16 number_of_recoveries
float32 distance_remaining
```

The `behavior_tree` parameter is used to specify the behavior tree to be used for navigation. The default behavior tree is `navigate_to_pose_w_replanning_and_recovery.xml`. The behavior tree can be changed by modifying the `bt_navigator` parameter in the `nav2_params.yaml` file. For example, to use a behavior tree where the path is replanned every 5 seconds, make a file inside the `params` folder named `navigate_to_pose_w_replanning_and_recovery_5s.xml` with the following content:
```
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <RateController hz="0.2">
        <ReactiveSequence>
          <RemovePassedGoals input_goals="{goals}" output_goals="{goals}" radius="0.1"/>
          <ComputePathToPose goals="{goals}" path="{path}" planner_id="GridBased"/>
        </ReactiveSequence>
      </RateController>
      <FollowPath path="{path}" controller_id="FollowPath" goal_checker_id="precise_goal_checker"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

To implement this behavior tree, you can either add the complete path of the XML file to the `default_nav_to_pose_bt_xml` parameter in `bt_navigator` section of the `nav2_params.yaml` file, or you can use the path in the `behavior_tree` string while calling the `NavigateToPose` action server.

## Summary
This tutorial provides a step-by-step guide to configure the Clearpath Husky for navigation using the ROS 1 packages and seamlessly connecting it to ROS 2. The tutorial also provides a brief overview of the Nav2 stack and how to use it for navigation. 

It is recommended to read the [Nav2 documentation](https://navigation.ros.org/index.html) to understand the Nav2 stack in detail. The [Nav2 tutorials](https://navigation.ros.org/getting_started/index.html) are also a good place to start.

## See Also:
- [ROS1 - ROS2 Bridge](https://roboticsknowledgebase.com/wiki/interfacing/ros1_ros2_bridge/)

## Further Readings:
- [Nav2 First Time Setup](https://navigation.ros.org/setup_guides/index.html)
- [Nav2 Behavior Tree](https://navigation.ros.org/behavior_trees/index.html)
- [Configuring Behavior Trees](https://navigation.ros.org/configuration/packages/configuring-bt-xml.html)