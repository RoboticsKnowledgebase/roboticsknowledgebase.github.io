# AirStack Behavior Tree: Complete Engineering Guide

This document provides a comprehensive guide to understanding, implementing, and extending Behavior Trees in AirStack. It is designed as an executable engineering reference with checkpoints for incremental development.

**Repository Reference**: [robot/ros_ws/src/autonomy/5_behavior](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior)

---

## Table of Contents

1. [Part 1: Conceptual Foundation](#part-1-conceptual-foundation)

- [Why Behavior Trees](#why-behavior-trees)
- [Core Concepts](#core-concepts)
- [AirStack Architecture](#airstack-architecture)

2. [Part 2: Codebase Deep Dive](#part-2-codebase-deep-dive)

- [Node Type Implementation](#node-type-implementation)
- [Tree Configuration Syntax](#tree-configuration-syntax)
- [ROS Communication Protocol](#ros-communication-protocol)
- [drone.tree Analysis](#dronetree-complete-analysis)

3. [Part 3: Hands-On Tutorial](#part-3-hands-on-tutorial)

- [Checkpoint 1: Create ROS2 Package](#checkpoint-1-create-ros2-package)
- [Checkpoint 2: Design Decision Tree](#checkpoint-2-design-decision-tree)
- [Checkpoint 3: Implement Conditions](#checkpoint-3-implement-conditions)
- [Checkpoint 4: Implement Actions](#checkpoint-4-implement-actions)
- [Checkpoint 5: Launch and Test](#checkpoint-5-launch-configuration-and-testing)

4. [Part 4: Debugging and Extension](#part-4-debugging-and-extension)

---

# Part 1: Conceptual Foundation

## Why Behavior Trees

Before implementing any autonomy system, engineers must choose a decision-making architecture. The three most common approaches are Finite State Machines (FSM), Hierarchical Finite State Machines (HFSM), and Behavior Trees (BT). Understanding their trade-offs is essential for making informed design decisions.

### Finite State Machines (FSM)

FSMs represent robot behavior as a set of discrete states with explicit transitions between them. For a simple two-state system (Idle and Moving), the implementation is straightforward. However, as system complexity grows, the number of transitions grows quadratically. A system with N states can have up to N*(N-1) transitions.

Consider a drone with states: Idle, Armed, TakingOff, Hovering, Flying, Landing, and Emergency. Each state must define transitions to potentially every other state based on various conditions. The result is a "spaghetti" of transitions that becomes increasingly difficult to maintain, test, and extend.

The fundamental problem with FSMs is tight coupling. Adding a new state requires modifying existing states to add transitions to and from the new state. This violates the Open-Closed Principle: the system is not open for extension without modification.

### Hierarchical Finite State Machines (HFSM)

HFSMs attempt to address FSM complexity by organizing states into nested hierarchies. A "Flying" superstate might contain substates for "Cruising", "Avoiding", and "Approaching". While this reduces visual complexity, it does not eliminate the fundamental coupling problem.

Transitions between states at different hierarchy levels require careful management of entry and exit conditions. The hierarchical structure also makes it difficult to share behavior across different parts of the tree. If both "Flying" and "Landing" need obstacle avoidance behavior, the logic must be duplicated or awkwardly shared through complex state inheritance.

### Behavior Trees: A Modular Alternative

Behavior Trees address these limitations through a fundamentally different design philosophy. Instead of defining explicit state transitions, BTs define task decomposition and execution order through a tree structure.

Key advantages of Behavior Trees include:

**Modularity**: Each subtree is self-contained. The "Takeoff" subtree can be developed, tested, and modified independently of the "Landing" subtree. Adding new behavior means adding new branches, not modifying existing ones.

**Reusability**: Subtrees can be referenced from multiple locations. An "EnsureArmed" subtree that checks if the drone is armed and arms it if not can be reused in Takeoff, EmergencyLand, and ManualControl branches without code duplication.

**Readability**: The tree structure visually represents execution flow. Reading from root to leaves shows the priority of behaviors. Reading from left to right within a branch shows sequential dependencies.

**Testability**: Individual nodes and subtrees can be unit tested in isolation. Mock conditions and actions can verify tree logic without requiring full system integration.

**Decoupling**: In AirStack's implementation, the Behavior Tree communicates with action executors through ROS topics. The tree logic is completely decoupled from the action implementation. This separation allows the tree to be modified without changing any executor code, and vice versa.

### When Not to Use Behavior Trees

Behavior Trees are not universally superior. For simple systems with fewer than five states and well-defined transitions, an FSM may be more straightforward to implement and understand. BTs introduce additional infrastructure (the tree engine, configuration parsing, topic-based communication) that may be unnecessary overhead for simple applications.

Additionally, Behavior Trees are inherently reactive and single-threaded in their basic form. Systems requiring true parallel state maintenance or complex temporal logic may need extensions beyond basic BT semantics.

## Core Concepts

A Behavior Tree is a directed acyclic graph where each node returns one of three statuses: SUCCESS, RUNNING, or FAILURE. The tree is "ticked" at a fixed frequency (20 Hz in AirStack), and each tick propagates from root to leaves, with status propagating back up.

### Node Categories

Nodes are categorized into three types based on their function:

**Execution Nodes** are the leaves of the tree. They interface with the external world, either checking conditions or executing actions. Condition nodes return SUCCESS or FAILURE based on some boolean check. Action nodes return SUCCESS when complete, RUNNING while executing, or FAILURE if the action cannot be performed.

**Control Flow Nodes** determine the execution order of their children. Sequence nodes execute children left-to-right until one fails. Fallback nodes (also called Selector nodes) execute children until one succeeds. Parallel nodes execute all children simultaneously.

**Decorator Nodes** modify the behavior of a single child node. The most common decorator is the Not node, which inverts SUCCESS to FAILURE and vice versa. Other decorators can implement retry logic, timeouts, or rate limiting.

### Tick Propagation

Understanding tick propagation is essential for predicting tree behavior. When the root node is ticked:

1. The root examines its type and determines which children to tick
2. Each ticked child recursively applies the same logic
3. Leaf nodes (Conditions and Actions) perform their checks or operations
4. Status values propagate back up the tree
5. Parent nodes combine child statuses according to their type

For a Sequence node with children A, B, C:

- Tick A. If A returns FAILURE, Sequence returns FAILURE (B and C are not ticked)
- If A returns SUCCESS, tick B
- If B returns RUNNING, Sequence returns RUNNING (C is not ticked)
- If B returns SUCCESS, tick C
- Sequence returns whatever C returns

For a Fallback node with children A, B, C:

- Tick A. If A returns SUCCESS, Fallback returns SUCCESS (B and C are not ticked)
- If A returns FAILURE, tick B
- Continue until one child succeeds or all fail

This short-circuit evaluation is crucial for efficiency and for implementing conditional logic.

## AirStack Architecture

AirStack's Behavior Tree implementation consists of two primary components working in concert:

### Behavior Tree Engine

The Behavior Tree engine is responsible for parsing tree configuration files, maintaining tree state, executing the tick cycle, and publishing visualization data. It is implemented in the `behavior_tree` package and runs as a standalone ROS node.

The engine reads a `.tree` configuration file at startup, constructs the internal tree representation, and then enters a continuous tick loop. During each tick, the engine traverses the tree, publishes activation messages to Action nodes, and subscribes to status updates from both Conditions and Actions.

### Behavior Executive

The Behavior Executive is the implementation layer that gives meaning to Conditions and Actions defined in the tree. It subscribes to Action activation messages, performs the actual robot operations (calling services, publishing commands), and publishes status updates back to the tree engine.

The Executive also monitors robot state through various subscriptions and publishes Condition status updates. This separation allows the tree structure to be modified without changing any executive code, as long as the Condition and Action names remain consistent.

### Communication Flow

The communication between the tree engine and executive follows a well-defined protocol:

For Conditions:

1. Executive monitors robot state (sensors, services, etc.)
2. Executive publishes boolean status to `{condition_name}_success` topic
3. Tree engine subscribes and updates internal Condition node status

For Actions:

1. Tree engine publishes activation to `{action_name}_active` topic with unique ID
2. Executive subscribes, performs action logic
3. Executive publishes status to `{action_name}_status` topic with matching ID
4. Tree engine subscribes and updates internal Action node status

The ID matching ensures that status messages correspond to the correct activation cycle, preventing race conditions when actions are rapidly activated and deactivated.

---

# Part 2: Codebase Deep Dive

## Node Type Implementation

The node type hierarchy is defined in [behavior_tree_implementation.hpp](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree/include/behavior_tree/behavior_tree_implementation.hpp). Understanding this implementation is essential for extending the system.

### Base Node Class

All nodes inherit from the abstract `Node` base class:

```cpp
class Node {
public:
    static double max_wait_time;  // Timeout for execution nodes (default 1.0s)

    std::string label;           // Display label for visualization
    bool is_active;              // Whether node is currently being executed
    uint8_t status;              // Current status: SUCCESS, RUNNING, or FAILURE
    std::vector<Node*> children; // Child nodes

    virtual void add_child(Node*);
    virtual bool tick(bool active, int traversal_count) = 0;  // Pure virtual
};
```

The `tick` method is the core of the execution model. Each node type implements this method to define its specific behavior. The `active` parameter indicates whether this node should actually execute (inactive nodes still update their status but don't perform side effects). The `traversal_count` prevents multiple ticks of the same node within a single tree traversal.

### Control Flow Node Implementation

The Fallback node implementation demonstrates the control flow pattern:

```cpp
bool FallbackNode::tick(bool active, int traversal_count) {
    uint8_t prev_status = status;
    bool child_changed = false;

    if (!active) {
        // Tick all children as inactive to update their state
        for (int i = 0; i < children.size(); i++)
            child_changed |= children[i]->tick(false, traversal_count);
    } else {
        status = behavior_tree_msgs::msg::Status::FAILURE;
        for (int i = 0; i < children.size(); i++) {
            Node* child = children[i];
            if (status == behavior_tree_msgs::msg::Status::FAILURE) {
                // Only tick active if we haven't found a success yet
                child_changed |= child->tick(true, traversal_count);
                status = child->status;
            } else {
                // Already found success, tick remaining as inactive
                child_changed |= child->tick(false, traversal_count);
            }
        }
    }

    bool status_changed = (status != prev_status) || child_changed;
    bool active_changed = (is_active != active);
    is_active = active;

    return status_changed || active_changed;
}
```

Key implementation details:

1. Even inactive nodes tick their children as inactive, ensuring proper state propagation
2. The short-circuit logic stops active ticking once a SUCCESS is found
3. The return value indicates whether any visual state changed, used for efficient re-rendering

The Sequence node follows similar logic but inverts the success/failure handling:

```cpp
bool SequenceNode::tick(bool active, int traversal_count) {
    // ... similar structure ...
    if (active) {
        status = behavior_tree_msgs::msg::Status::SUCCESS;
        for (int i = 0; i < children.size(); i++) {
            Node* child = children[i];
            if (status == behavior_tree_msgs::msg::Status::SUCCESS) {
                child_changed |= child->tick(true, traversal_count);
                status = child->status;
            } else {
                child_changed |= child->tick(false, traversal_count);
            }
        }
    }
    // ...
}
```

### Execution Node Implementation

Execution nodes add ROS communication infrastructure. The `ExecutionNode` base class provides:

```cpp
class ExecutionNode : public Node {
public:
    rclcpp::Time status_modification_time;
    int current_traversal_count;
    rclcpp::Node* node;  // ROS node for pub/sub

    ExecutionNode(rclcpp::Node* node);
    void init_ros();
    virtual std::string get_publisher_name() = 0;
    virtual void init_publisher() = 0;
    virtual std::string get_subscriber_name() = 0;
    virtual void init_subscriber() = 0;
    void set_status(uint8_t status);  // Updates status and timestamp
    double get_status_age();          // Time since last status update
};
```

The status age is critical for the timeout mechanism. If an execution node doesn't receive an update within `max_wait_time` seconds, it automatically transitions to FAILURE. This prevents the tree from getting stuck waiting for dead nodes.

The ConditionNode subscribes to a boolean topic:

```cpp
void ConditionNode::init_subscriber() {
    subscriber = node->create_subscription<std_msgs::msg::Bool>(
        get_subscriber_name(), 1,
        std::bind(&ConditionNode::callback, this, std::placeholders::_1));
}

void ConditionNode::callback(const std_msgs::msg::Bool::SharedPtr msg) {
    uint8_t prev_status = status;
    if (msg->data)
        set_status(behavior_tree_msgs::msg::Status::SUCCESS);
    else
        set_status(behavior_tree_msgs::msg::Status::FAILURE);
    if (status != prev_status) callback_status_updated = true;
}
```

The ActionNode is more complex, handling bidirectional communication with ID matching:

```cpp
void ActionNode::publish_active_msg(int active_id) {
    if (publisher_initialized) {
        behavior_tree_msgs::msg::Active active_msg;
        active_msg.active = is_active;
        active_msg.id = active_id;
        current_id = active_id;
        publisher->publish(active_msg);
    }
}

void ActionNode::callback(const behavior_tree_msgs::msg::Status::SharedPtr msg) {
    uint8_t prev_status = status;
    if (is_active) {
        if (msg->id != current_id) {
            // Ignore status from previous activation cycle
            std::cout << label << " Incorrect ID " << msg->id << " " << current_id << " "
                      << is_active << std::endl;
            return;
        }
        set_status(msg->status);
    }
    if (status != prev_status) callback_status_updated = true;
}
```

### Decorator Node Implementation

The NotNode demonstrates the decorator pattern:

```cpp
void NotNode::add_child(Node* node) {
    ConditionNode* condition_node = dynamic_cast<ConditionNode*>(node);
    if (condition_node != NULL)
        DecoratorNode::add_child(condition_node);
    else {
        std::cout << "A not decorator node can only have a condition node as a child." << std::endl;
        exit(1);
    }
}

bool NotNode::tick(bool active, int traversal_count) {
    // ... 
    if (children.size() > 0) {
        Node* child = children[0];
        child_changed |= child->tick(active, traversal_count);

        if (child->status == behavior_tree_msgs::msg::Status::SUCCESS)
            status = behavior_tree_msgs::msg::Status::FAILURE;
        else if (child->status == behavior_tree_msgs::msg::Status::FAILURE)
            status = behavior_tree_msgs::msg::Status::SUCCESS;
    }
    // ...
}
```

Note that the Not decorator enforces a constraint: its child must be a ConditionNode. This makes semantic sense because inverting RUNNING status is undefined.

## Tree Configuration Syntax

Tree configuration files use an indentation-based syntax. Reference: [behavior_tree/config/drone.tree](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree/config/drone.tree)

### Syntax Rules

Each line represents a node. The node type is determined by the first character(s):

| Symbol | Node Type | Description |

|--------|-----------|-------------|

| `?` | Fallback | Tries children until one succeeds |

| `->` | Sequence | Executes children until one fails |

| `\|\| N` | Parallel | Executes all children, requires N successes |

| `(Name)` | Condition | Checks a boolean state |

| `[Name]` | Action | Executes a behavior |

| `<!>` | Not | Inverts child condition |

Indentation uses TAB characters (not spaces). Each TAB level indicates parent-child relationship:

```

?                    # Root (level 0)

    ->               # Child of root (level 1)

        (Cond A)     # Child of sequence (level 2)

        [Action A]   # Child of sequence (level 2)

    ->               # Another child of root (level 1)

        (Cond B)

        [Action B]

```

### Parser Implementation

The parsing logic in `BehaviorTree::parse_config()` processes line by line:

```cpp
void BehaviorTree::parse_config() {
    std::ifstream in(config_filename);

    if (in.is_open()) {
        std::vector<Node*> nodes_worklist;
        int prev_tabs = 0;

        std::string line;
        while (getline(in, line)) {
            if (line.size() == 0) continue;  // Skip empty lines

            int curr_tabs = count_tabs(line);
            std::string label = strip_space(line);
            Node* node = NULL;

            // Create node based on label prefix
            if (label.compare(0, 2, "->") == 0)
                node = new SequenceNode();
            else if (label.compare(0, 1, "?") == 0)
                node = new FallbackNode();
            else if (label.compare(0, 2, "||") == 0) {
                std::vector<int> arguments = get_arguments(label);
                int child_success_threshold = 0;
                if (arguments.size() > 0)
                    child_success_threshold = arguments[0];
                else {
                    std::cout << "Arguments not provided to parallel node" << std::endl;
                    exit(1);
                }
                node = new ParallelNode(child_success_threshold);
            } else if (label.compare(0, 1, "(") == 0)
                node = new ConditionNode(strip_brackets(label), ros2_node);
            else if (label.compare(0, 1, "[") == 0) {
                node = new ActionNode(strip_brackets(label), ros2_node);
                active_ids[node->label] = 0;
            } else if (label.compare(0, 1, "<") == 0) {
                std::string decorator_label = strip_brackets(label);
                if (decorator_label.compare(0, 1, "!") == 0)
                    node = new NotNode();
            }

            if (node != NULL) {
                nodes.push_back(node);

                if (root == NULL) {
                    root = node;
                    nodes_worklist.push_back(node);
                    continue;
                }

                // Establish parent-child relationship based on indentation
                if (curr_tabs == prev_tabs + 1) {
                    Node* parent = nodes_worklist[nodes_worklist.size() - 1];
                    parent->add_child(node);
                } else {
                    for (int i = 0; i < prev_tabs - curr_tabs + 1; i++)
                        nodes_worklist.pop_back();
                    Node* parent = nodes_worklist[nodes_worklist.size() - 1];
                    parent->add_child(node);
                }

                nodes_worklist.push_back(node);
                prev_tabs = curr_tabs;
            }
        }
        in.close();
    } else {
        std::cout << "Failed to open behavior tree config file: " << config_filename << std::endl;
    }
}
```

### Topic Name Generation

Node names are automatically converted to ROS topic names through a standardization process:

```cpp
std::string ConditionNode::get_subscriber_name() {
    std::string name = label;
    // Convert to lowercase
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    // Replace spaces with underscores
    std::replace(name.begin(), name.end(), ' ', '_');
    // Add suffix
    name = name + "_success";
    return name;
}
```

Examples of name conversion:

| Node Definition | Generated Topic |

|-----------------|-----------------|

| `(In Air)` | `in_air_success` |

| `(Takeoff Complete)` | `takeoff_complete_success` |

| `[Request Control]` | `request_control_active` and `request_control_status` |

| `[Follow Global Plan]` | `follow_global_plan_active` and `follow_global_plan_status` |

This automatic naming convention is crucial for correctly implementing executives. The executive must publish/subscribe to exactly these topic names for communication to work.

## ROS Communication Protocol

The communication between tree engine and executive uses custom message types defined in [common/ros_packages/behavior_tree_msgs](https://github.com/castacks/AirStack/tree/main/common/ros_packages/behavior_tree_msgs).

### Message Definitions

**Status.msg** - Used by Actions to report their execution status:

```

int8 FAILURE=0

int8 RUNNING=1

int8 SUCCESS=2


int8 status

uint64 id

```

**Active.msg** - Used by tree engine to signal Action activation:

```

bool active

int64 id

```

The `id` field is critical for correct operation. When an Action is activated, the tree engine assigns a unique ID and publishes it with the Active message. The executive must include this same ID when publishing status updates. If IDs don't match, the status update is ignored.

This mechanism handles the following edge case:

1. Action A is activated with ID=1
2. Executive starts long operation
3. Tree logic changes, Action A is deactivated
4. Tree logic changes again, Action A is reactivated with ID=2
5. Executive completes operation from step 2, publishes SUCCESS with ID=1
6. Tree engine ignores this because current ID is 2

Without ID matching, the stale SUCCESS would incorrectly affect the current activation cycle.

### Topic Structure

For a tree with namespace `/behavior`, the topic structure is:

```

/behavior/

├── armed_success                  # Condition: Armed

├── in_air_success                 # Condition: In Air

├── takeoff_complete_success       # Condition: Takeoff Complete

├── arm_active                     # Action: Arm (tree -> executive)

├── arm_status                     # Action: Arm (executive -> tree)

├── takeoff_active                 # Action: Takeoff

├── takeoff_status

├── active_actions                 # String listing current active actions

├── behavior_tree_graphviz         # Visualization data

└── behavior_tree_graphviz_compressed

```

## drone.tree Complete Analysis

The production drone tree demonstrates real-world behavior tree design. Let's analyze each major branch.

Reference: [drone.tree](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree/config/drone.tree)

### Root Structure

```

?

    ->  # Auto Takeoff

    ->  # Manual Takeoff

    ->  # Land

    ->  # Pause

    ->  # Rewind

    ->  # Fixed Trajectory

    ->  # Global Plan

    ->  # Offboard

    ->  # Arm

    ->  # Disarm

    ->  # Autonomously Explore

```

The root is a Fallback node with multiple Sequence children. This structure means:

1. Check Auto Takeoff conditions, execute if applicable
2. If not, check Manual Takeoff conditions
3. Continue until one branch succeeds or all fail

This priority ordering is intentional. Auto Takeoff takes precedence over Manual Takeoff, which takes precedence over Land, and so on.

### Auto Takeoff Branch Analysis

```

->

    (Auto Takeoff Commanded)

    ->

        <!>

            (State Estimate Timed Out)

        ?

            (In Air)

            (Armed)

            (Offboard Mode)

            [Request Control]

        ?

            (In Air)

            (Armed)

            [Arm]

        ?

            (Takeoff Complete)

            [Takeoff]

```

Step-by-step execution:

1.**Sequence Entry Condition**: `(Auto Takeoff Commanded)` must be SUCCESS. This condition is set by the executive based on user commands or mission logic.

2.**Safety Check**: `<!>(State Estimate Timed Out)` - The Not decorator inverts the condition. This line returns SUCCESS only if state estimate has NOT timed out. If state estimate has timed out, this returns FAILURE and the entire Auto Takeoff sequence fails immediately.

3.**Ensure Control**: The Fallback `?(In Air)(Armed)(Offboard Mode)[Request Control]` implements a priority check:

- If already in air, SUCCESS (no need for control)
- If already armed, SUCCESS (presumably have control)
- If already in offboard mode, SUCCESS (have control)
- Otherwise, execute [Request Control] action

4.**Ensure Armed**: Similar pattern - check if already in air or armed, otherwise execute [Arm]

5.**Execute Takeoff**: Check if takeoff already complete, otherwise execute [Takeoff]

This pattern of "check-if-already-done-else-do-it" is the canonical BT idiom for ensuring preconditions before actions.

### Autonomously Explore Branch

```

->

    (Autonomously Explore Commanded)

    ->

        <!>

            (State Estimate Timed Out)

        ?

            (In Air)

            (Armed)

            (Offboard Mode)

            [Request Control]

        ?

            (In Air)

            (Armed)

            [Arm]

        ?

            (Takeoff Complete)

            [Takeoff]

    ?

        ->

            (Stuck)

            [Rewind]

        [Follow Global Plan]

```

This branch extends Auto Takeoff with exploration logic:

1. Same precondition sequence (control, arm, takeoff)
2. New final Fallback:

- If stuck, execute Rewind action
- Otherwise, execute Follow Global Plan

The "stuck" detection provides autonomous recovery. If the drone gets stuck (condition set by local planner), it will rewind along its trajectory before continuing exploration.

---

# Part 3: Hands-On Tutorial

This section provides a step-by-step guide to creating a custom Behavior Tree system. Each checkpoint produces a testable artifact.

## Checkpoint 1: Create ROS2 Package

### Objective

Create a properly structured ROS2 package that will contain your custom Behavior Tree executive.

### File Structure

```

robot/ros_ws/src/autonomy/5_behavior/my_robot_behavior/

├── config/

│   └── my_robot.tree

├── include/my_robot_behavior/

│   └── my_robot_executive.hpp

├── src/

│   └── my_robot_executive.cpp

├── launch/

│   └── my_robot_behavior.launch.xml

├── CMakeLists.txt

└── package.xml

```

### package.xml

```xml

<?xml version="1.0"?>

<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>

<packageformat="3">

<name>my_robot_behavior</name>

<version>0.1.0</version>

<description>Custom behavior tree executive for my robot</description>

<maintaineremail="your@email.com">Your Name</maintainer>

<license>MIT</license>


<buildtool_depend>ament_cmake</buildtool_depend>


<depend>rclcpp</depend>

<depend>std_msgs</depend>

<depend>std_srvs</depend>

<depend>behavior_tree</depend>

<depend>behavior_tree_msgs</depend>


<export>

<build_type>ament_cmake</build_type>

</export>

</package>

```

### CMakeLists.txt

```cmake

cmake_minimum_required(VERSION 3.8)

project(my_robot_behavior)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  add_compile_options(-Wall -Wextra -Wpedantic)

endif()


# Find dependencies

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(std_msgs REQUIRED)

find_package(std_srvs REQUIRED)

find_package(behavior_tree REQUIRED)

find_package(behavior_tree_msgs REQUIRED)


# Include directories

include_directories(include)


# Executive executable

add_executable(my_robot_executive src/my_robot_executive.cpp)

ament_target_dependencies(my_robot_executive

  rclcpp

  std_msgs

  std_srvs

  behavior_tree

  behavior_tree_msgs

)


# Install targets

install(TARGETS my_robot_executive

  DESTINATION lib/${PROJECT_NAME}

)


# Install config files

install(DIRECTORY config/

  DESTINATION share/${PROJECT_NAME}/config

)


# Install launch files

install(DIRECTORY launch/

  DESTINATION share/${PROJECT_NAME}/launch

)


ament_package()

```

### Verification Commands

```bash
# From workspace root
cd robot/ros_ws

# Build only your package
colcon build --packages-select my_robot_behavior

# Source the workspace
source install/setup.bash

# Verify package is recognized
ros2 pkg list | grep my_robot_behavior
```

### Expected Output

```

Starting >>> my_robot_behavior

Finished <<< my_robot_behavior [X.XXs]


Summary: 1 package finished

```

If build fails with missing dependencies, ensure behavior_tree and behavior_tree_msgs are built first:

```bash
colcon build --packages-up-to my_robot_behavior
```

---

## Checkpoint 2: Design Decision Tree

### Objective

Create a `.tree` configuration file that defines your robot's decision logic.

### Design Process

Before writing the tree, enumerate:

1. What commands can the robot receive?
2. What conditions determine robot state?
3. What actions can the robot perform?
4. What is the priority order of commands?

For this tutorial, we'll create a simple patrol robot that:

- Responds to Start/Stop commands
- Patrols between waypoints
- Returns home on low battery
- Has emergency stop capability

### config/my_robot.tree

```

?

    ->

        (Emergency Stop Commanded)

        ?

            (Is Stopped)

            [Stop All Systems]

    ->

        (Low Battery)

        ?

            (At Home)

            [Go To Home]

    ->

        (Start Commanded)

        <!>

            (Emergency Stop Commanded)

        ?

            (Systems Ready)

            [Initialize Systems]

        ?

            (At Current Waypoint)

            [Navigate To Waypoint]

        [Advance To Next Waypoint]

    ->

        (Stop Commanded)

        ?

            (Is Stopped)

            [Stop All Systems]

```

### Tree Logic Explanation

**Priority 1 - Emergency Stop**: If emergency stop is commanded, ensure systems are stopped. The Fallback checks if already stopped before executing the stop action.

**Priority 2 - Low Battery**: Regardless of other commands, if battery is low, navigate home. This overrides normal operation.

**Priority 3 - Start/Patrol**: If start is commanded and no emergency stop:

1. Ensure systems are initialized
2. If not at current waypoint, navigate to it
3. Once at waypoint, advance to next waypoint in patrol sequence

**Priority 4 - Stop**: Normal stop command (non-emergency) also stops all systems.

### Condition/Action Summary

| Conditions                 | Description                 |
| -------------------------- | --------------------------- |
| (Emergency Stop Commanded) | User pressed emergency stop |
| (Low Battery)              | Battery below threshold     |
| (Start Commanded)          | User pressed start button   |
| (Stop Commanded)           | User pressed stop button    |
| (Is Stopped)               | All systems are stopped     |
| (At Home)                  | Robot at home position      |
| (Systems Ready)            | Systems initialized         |
| (At Current Waypoint)      | Robot at current target     |

| Actions                    | Description      |
| -------------------------- | ---------------- |
| [Stop All Systems]         | Halt all motion  |
| [Go To Home]               | Navigate to home |
| [Initialize Systems]       | Start up systems |
| [Navigate To Waypoint]     | Move to target   |
| [Advance To Next Waypoint] | Update target    |

### Verification

```bash
# Verify tree syntax by checking parse output
ros2 run behavior_tree behavior_tree_implementation --ros-args -p config:=/path/to/my_robot.tree
```

If the tree has syntax errors, you will see parsing error messages.

---

## Checkpoint 3: Implement Conditions

### Objective

Create the executive node that monitors robot state and publishes Condition status.

### Header File: include/my_robot_behavior/my_robot_executive.hpp

```cpp
#pragma once

#include <behavior_tree/behavior_tree.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class MyRobotExecutive : public rclcpp::Node {
public:
    MyRobotExecutive();

private:
    // Conditions
    bt::Condition* emergency_stop_commanded_condition;
    bt::Condition* low_battery_condition;
    bt::Condition* start_commanded_condition;
    bt::Condition* stop_commanded_condition;
    bt::Condition* is_stopped_condition;
    bt::Condition* at_home_condition;
    bt::Condition* systems_ready_condition;
    bt::Condition* at_current_waypoint_condition;

    // Actions (will be implemented in Checkpoint 4)
    bt::Action* stop_all_systems_action;
    bt::Action* go_to_home_action;
    bt::Action* initialize_systems_action;
    bt::Action* navigate_to_waypoint_action;
    bt::Action* advance_to_next_waypoint_action;

    // State variables
    bool emergency_stop_pressed;
    bool start_pressed;
    bool stop_pressed;
    float battery_level;
    float low_battery_threshold;
    bool systems_initialized;
    geometry_msgs::msg::PoseStamped current_pose;
    geometry_msgs::msg::PoseStamped home_pose;
    geometry_msgs::msg::PoseStamped current_waypoint;
    float waypoint_tolerance;

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_sub;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr battery_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub;

    // Timer
    rclcpp::TimerBase::SharedPtr timer;

    // Callbacks
    void timer_callback();
    void emergency_stop_callback(const std_msgs::msg::Bool::SharedPtr msg);
    void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
    void stop_callback(const std_msgs::msg::Bool::SharedPtr msg);
    void battery_callback(const std_msgs::msg::Float32::SharedPtr msg);
    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    // Helper functions
    double distance(const geometry_msgs::msg::PoseStamped& a, 
                    const geometry_msgs::msg::PoseStamped& b);
};
```

### Source File: src/my_robot_executive.cpp (Part 1 - Conditions)

```cpp
#include "my_robot_behavior/my_robot_executive.hpp"
#include <cmath>

MyRobotExecutive::MyRobotExecutive() : Node("my_robot_executive") {
    // Initialize Conditions - names MUST match .tree file exactly
    emergency_stop_commanded_condition = new bt::Condition("Emergency Stop Commanded", this);
    low_battery_condition = new bt::Condition("Low Battery", this);
    start_commanded_condition = new bt::Condition("Start Commanded", this);
    stop_commanded_condition = new bt::Condition("Stop Commanded", this);
    is_stopped_condition = new bt::Condition("Is Stopped", this);
    at_home_condition = new bt::Condition("At Home", this);
    systems_ready_condition = new bt::Condition("Systems Ready", this);
    at_current_waypoint_condition = new bt::Condition("At Current Waypoint", this);

    // Initialize state
    emergency_stop_pressed = false;
    start_pressed = false;
    stop_pressed = false;
    battery_level = 100.0;
    low_battery_threshold = 20.0;
    systems_initialized = false;
    waypoint_tolerance = 0.5;  // meters

    // Set home position
    home_pose.pose.position.x = 0.0;
    home_pose.pose.position.y = 0.0;
    home_pose.pose.position.z = 0.0;

    // Create subscribers
    emergency_stop_sub = this->create_subscription<std_msgs::msg::Bool>(
        "emergency_stop", 10,
        std::bind(&MyRobotExecutive::emergency_stop_callback, this, std::placeholders::_1));

    start_sub = this->create_subscription<std_msgs::msg::Bool>(
        "start_command", 10,
        std::bind(&MyRobotExecutive::start_callback, this, std::placeholders::_1));

    stop_sub = this->create_subscription<std_msgs::msg::Bool>(
        "stop_command", 10,
        std::bind(&MyRobotExecutive::stop_callback, this, std::placeholders::_1));

    battery_sub = this->create_subscription<std_msgs::msg::Float32>(
        "battery_level", 10,
        std::bind(&MyRobotExecutive::battery_callback, this, std::placeholders::_1));

    pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "robot_pose", 10,
        std::bind(&MyRobotExecutive::pose_callback, this, std::placeholders::_1));

    // Create timer at 20 Hz (matching behavior tree tick rate)
    timer = this->create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&MyRobotExecutive::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "MyRobotExecutive initialized");
}

void MyRobotExecutive::timer_callback() {
    // Update all Conditions based on current state

    // Emergency stop condition
    emergency_stop_commanded_condition->set(emergency_stop_pressed);
    emergency_stop_commanded_condition->publish();

    // Low battery condition
    low_battery_condition->set(battery_level < low_battery_threshold);
    low_battery_condition->publish();

    // Command conditions
    start_commanded_condition->set(start_pressed);
    start_commanded_condition->publish();

    stop_commanded_condition->set(stop_pressed);
    stop_commanded_condition->publish();

    // Is stopped - true if not moving (simplified: if stop pressed or emergency)
    is_stopped_condition->set(stop_pressed || emergency_stop_pressed);
    is_stopped_condition->publish();

    // At home - within tolerance of home position
    at_home_condition->set(distance(current_pose, home_pose) < waypoint_tolerance);
    at_home_condition->publish();

    // Systems ready
    systems_ready_condition->set(systems_initialized);
    systems_ready_condition->publish();

    // At current waypoint
    at_current_waypoint_condition->set(distance(current_pose, current_waypoint) < waypoint_tolerance);
    at_current_waypoint_condition->publish();

    // Action handling will be added in Checkpoint 4
}

// Subscriber callbacks
void MyRobotExecutive::emergency_stop_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    emergency_stop_pressed = msg->data;
    if (emergency_stop_pressed) {
        RCLCPP_WARN(this->get_logger(), "EMERGENCY STOP ACTIVATED");
    }
}

void MyRobotExecutive::start_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    start_pressed = msg->data;
}

void MyRobotExecutive::stop_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    stop_pressed = msg->data;
}

void MyRobotExecutive::battery_callback(const std_msgs::msg::Float32::SharedPtr msg) {
    battery_level = msg->data;
}

void MyRobotExecutive::pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    current_pose = *msg;
}

double MyRobotExecutive::distance(const geometry_msgs::msg::PoseStamped& a,
                                   const geometry_msgs::msg::PoseStamped& b) {
    double dx = a.pose.position.x - b.pose.position.x;
    double dy = a.pose.position.y - b.pose.position.y;
    double dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyRobotExecutive>());
    rclcpp::shutdown();
    return 0;
}
```

### Verification

```bash
# Build
colcon build --packages-select my_robot_behavior

# In terminal 1: Run the behavior tree
ros2 run behavior_tree behavior_tree_implementation --ros-args \
    -p config:=$(ros2 pkg prefix my_robot_behavior)/share/my_robot_behavior/config/my_robot.tree

# In terminal 2: Run your executive
ros2 run my_robot_behavior my_robot_executive

# In terminal 3: Check that condition topics are being published
ros2 topic list | grep _success

# Expected output:
# /behavior/emergency_stop_commanded_success
# /behavior/low_battery_success
# /behavior/start_commanded_success
# ... etc
```

---

## Checkpoint 4: Implement Actions

### Objective

Add Action handling to the executive to respond to behavior tree activation.

### Update src/my_robot_executive.cpp

Add Action initialization in constructor:

```cpp
// In constructor, after Condition initialization:

// Initialize Actions - names MUST match .tree file exactly
stop_all_systems_action = new bt::Action("Stop All Systems", this);
go_to_home_action = new bt::Action("Go To Home", this);
initialize_systems_action = new bt::Action("Initialize Systems", this);
navigate_to_waypoint_action = new bt::Action("Navigate To Waypoint", this);
advance_to_next_waypoint_action = new bt::Action("Advance To Next Waypoint", this);
```

Add Action handling in timer_callback:

```cpp
void MyRobotExecutive::timer_callback() {
    // ... Condition updates from Checkpoint 3 ...

    // Handle Actions

    // Stop All Systems Action
    if (stop_all_systems_action->is_active()) {
        if (stop_all_systems_action->active_has_changed()) {
            RCLCPP_INFO(this->get_logger(), "Stopping all systems...");
            // Send stop command to motors/actuators
            // For this example, we'll just set a flag
        }
        // Check if stopped
        if (is_stopped_condition->get()) {
            stop_all_systems_action->set_success();
            RCLCPP_INFO(this->get_logger(), "All systems stopped");
        } else {
            stop_all_systems_action->set_running();
        }
        stop_all_systems_action->publish();
    }

    // Go To Home Action
    if (go_to_home_action->is_active()) {
        if (go_to_home_action->active_has_changed()) {
            RCLCPP_INFO(this->get_logger(), "Navigating to home...");
            // Set navigation goal to home position
        }
        if (at_home_condition->get()) {
            go_to_home_action->set_success();
            RCLCPP_INFO(this->get_logger(), "Arrived at home");
        } else {
            go_to_home_action->set_running();
            // Continue navigation
        }
        go_to_home_action->publish();
    }

    // Initialize Systems Action
    if (initialize_systems_action->is_active()) {
        if (initialize_systems_action->active_has_changed()) {
            RCLCPP_INFO(this->get_logger(), "Initializing systems...");
            // Perform initialization
            systems_initialized = true;  // Simplified
        }
        if (systems_initialized) {
            initialize_systems_action->set_success();
            RCLCPP_INFO(this->get_logger(), "Systems initialized");
        } else {
            initialize_systems_action->set_running();
        }
        initialize_systems_action->publish();
    }

    // Navigate To Waypoint Action
    if (navigate_to_waypoint_action->is_active()) {
        if (navigate_to_waypoint_action->active_has_changed()) {
            RCLCPP_INFO(this->get_logger(), "Navigating to waypoint...");
            // Start navigation to current_waypoint
        }
        if (at_current_waypoint_condition->get()) {
            navigate_to_waypoint_action->set_success();
            RCLCPP_INFO(this->get_logger(), "Arrived at waypoint");
        } else {
            navigate_to_waypoint_action->set_running();
        }
        navigate_to_waypoint_action->publish();
    }

    // Advance To Next Waypoint Action
    if (advance_to_next_waypoint_action->is_active()) {
        if (advance_to_next_waypoint_action->active_has_changed()) {
            RCLCPP_INFO(this->get_logger(), "Advancing to next waypoint...");
            // Update current_waypoint to next in patrol sequence
            // This is a one-shot action
            advance_to_next_waypoint_action->set_success();
        }
        advance_to_next_waypoint_action->publish();
    }
}
```

### Verification

```bash
# Rebuild
colcon build --packages-select my_robot_behavior
source install/setup.bash

# Run tree and executive as before

# Check action topics
ros2 topic list | grep -E "_active|_status"

# Test by publishing a start command
ros2 topic pub /start_command std_msgs/msg/Bool "data: true" --once

# Watch the tree visualization or active_actions topic
ros2 topic echo /behavior/active_actions
```

---

## Checkpoint 5: Launch Configuration and Testing

### Objective

Create a launch file that starts both the behavior tree and executive with proper configuration.

### launch/my_robot_behavior.launch.xml

```xml
<launch>
    <!-- Arguments -->
    <arg name="robot_name" default="robot1" />
    <arg name="tree_config" default="$(find-pkg-share my_robot_behavior)/config/my_robot.tree" />
    <arg name="timeout" default="1.0" />

    <group>
        <push-ros-namespace namespace="$(var robot_name)/behavior" />

        <!-- Behavior Tree Engine -->
        <node pkg="behavior_tree" exec="behavior_tree_implementation" name="behavior_tree">
            <param name="config" value="$(var tree_config)" />
            <param name="timeout" value="$(var timeout)" />
        </node>

        <!-- Behavior Executive -->
        <node pkg="my_robot_behavior" exec="my_robot_executive" name="executive" output="screen">
            <!-- Remap input topics -->
            <remap from="emergency_stop" to="/$(var robot_name)/emergency_stop" />
            <remap from="start_command" to="/$(var robot_name)/start_command" />
            <remap from="stop_command" to="/$(var robot_name)/stop_command" />
            <remap from="battery_level" to="/$(var robot_name)/battery_level" />
            <remap from="robot_pose" to="/$(var robot_name)/pose" />
        </node>
    </group>
</launch>
```

### Complete System Test

```bash
# Terminal 1: Launch the behavior system
ros2 launch my_robot_behavior my_robot_behavior.launch.xml robot_name:=robot1

# Terminal 2: Simulate robot state
# Publish battery level
ros2 topic pub /robot1/battery_level std_msgs/msg/Float32 "data: 80.0" -r 1 &

# Publish robot pose (at origin)
ros2 topic pub /robot1/pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'world'}, pose: {position: {x: 0, y: 0, z: 0}}}" -r 10 &

# Terminal 3: Monitor behavior tree
ros2 topic echo /robot1/behavior/active_actions

# Terminal 4: Send commands and observe behavior
# Start the robot
ros2 topic pub /robot1/start_command std_msgs/msg/Bool "data: true" --once

# Trigger low battery (should navigate home)
ros2 topic pub /robot1/battery_level std_msgs/msg/Float32 "data: 15.0" --once

# Trigger emergency stop
ros2 topic pub /robot1/emergency_stop std_msgs/msg/Bool "data: true" --once
```

### Expected Behavior

1. **Initial state**: No active actions
2. **After start command**: Initialize Systems → Navigate To Waypoint → Advance To Next Waypoint
3. **After low battery**: Go To Home (overrides patrol)
4. **After emergency stop**: Stop All Systems (highest priority)

### Verification Checklist

- [ ] Package builds without errors
- [ ] All condition topics are published at 20 Hz
- [ ] All action topics respond to tree activation
- [ ] Tree logic executes correct priority order
- [ ] Emergency stop always takes precedence
- [ ] Low battery triggers return home

---

# Part 4: Debugging and Extension

## Common Issues and Solutions

### Issue: Condition Always Shows FAILURE

**Symptoms**: Condition node stays red in visualization

**Causes**:

1. Topic name mismatch between tree and executive
2. Executive not publishing condition
3. Condition timeout (no update within 1 second)

**Diagnosis**:

```bash
# Check topic is being published
ros2 topic hz /behavior/your_condition_success

# Check topic content
ros2 topic echo /behavior/your_condition_success

# Verify topic name matches
ros2 topic list | grep your_condition
```

**Solution**: Ensure condition name in `.tree` exactly matches the string passed to `bt::Condition` constructor.

### Issue: Action Never Activates

**Symptoms**: Action stays white (inactive) even when expected to run

**Causes**:

1. Preceding conditions in sequence are FAILURE
2. Earlier fallback branch is succeeding
3. Action topic namespace mismatch

**Diagnosis**:

```bash
# Check tree visualization to see which branch is active
ros2 topic echo /behavior/behavior_tree_graphviz

# Check what actions are currently active
ros2 topic echo /behavior/active_actions
```

**Solution**: Trace through tree logic. Each Sequence requires all preceding children to SUCCESS before reaching the action.

### Issue: Action Stuck in RUNNING

**Symptoms**: Action stays blue, never completes

**Causes**:

1. Success condition never becomes true
2. Executive not calling `set_success()` or `set_failure()`
3. Executive not calling `publish()`

**Diagnosis**:

```bash
# Monitor action status
ros2 topic echo /behavior/your_action_status
```

**Solution**: Add logging to executive action handling. Ensure `publish()` is called every timer tick when action is active.

### Issue: ID Mismatch Warnings

**Symptoms**: Console shows "Incorrect ID" messages

**Causes**:

1. Action rapidly activates/deactivates
2. Status message from previous activation cycle

**Solution**: This is usually benign. The ID mechanism correctly filters stale messages. However, frequent occurrences may indicate tree logic issues causing rapid toggling.

## Extending the System

### Adding New Node Types

To add a custom control flow node, modify [behavior_tree_implementation.hpp](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree/include/behavior_tree/behavior_tree_implementation.hpp):

```cpp
class RepeatNode : public ControlFlowNode {
private:
    int repeat_count;
    int current_count;
public:
    RepeatNode(int count);
    bool tick(bool active, int traversal_count) override;
};
```

Implement in `behavior_tree_implementation.cpp`:

```cpp
RepeatNode::RepeatNode(int count) : repeat_count(count), current_count(0) {
    label = "R" + std::to_string(count);
    is_active = false;
    status = behavior_tree_msgs::msg::Status::FAILURE;
}

bool RepeatNode::tick(bool active, int traversal_count) {
    uint8_t prev_status = status;
    bool child_changed = false;

    if (!active) {
        current_count = 0;
        // Tick children as inactive to update their state
        if (children.size() > 0)
            child_changed |= children[0]->tick(false, traversal_count);
    } else {
        if (children.size() == 0) {
            status = behavior_tree_msgs::msg::Status::FAILURE;
        } else {
            child_changed |= children[0]->tick(true, traversal_count);

            if (children[0]->status == behavior_tree_msgs::msg::Status::SUCCESS) {
                current_count++;
                if (current_count >= repeat_count) {
                    status = behavior_tree_msgs::msg::Status::SUCCESS;
                    current_count = 0;
                } else {
                    status = behavior_tree_msgs::msg::Status::RUNNING;
                }
            } else {
                status = children[0]->status;
            }
        }
    }

    bool status_changed = (status != prev_status) || child_changed;
    bool active_changed = (is_active != active);

    is_active = active;

    return status_changed || active_changed;
}
```

Add parsing in `parse_config()`:

```cpp
else if (label.compare(0, 1, "R") == 0) {
    std::vector<int> args = get_arguments(label);
    if (args.size() > 0) {
        node = new RepeatNode(args[0]);
    } else {
        std::cout << "Arguments not provided to repeat node" << std::endl;
        exit(1);
    }
}
```

Usage in tree:

```
R 3
    [Do Something]
```

### Multi-Robot Deployment

AirStack supports multi-robot deployment through namespacing. Each robot runs its own behavior tree and executive under a unique namespace:

```xml
<launch>
    <include file="$(find-pkg-share my_robot_behavior)/launch/my_robot_behavior.launch.xml">
        <arg name="robot_name" value="robot1" />
    </include>

    <include file="$(find-pkg-share my_robot_behavior)/launch/my_robot_behavior.launch.xml">
        <arg name="robot_name" value="robot2" />
    </include>
</launch>
```

Each robot will have isolated topic namespaces:

- `/robot1/behavior/...`
- `/robot2/behavior/...`

## Reference Implementation

For a complete production example, study the AirStack drone behavior system:

- Tree configuration: [drone.tree](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree/config/drone.tree)
- Executive implementation: [behavior_executive.cpp](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_executive/src/behavior_executive.cpp)
- Launch configuration: [behavior.launch.xml](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_bringup/launch/behavior.launch.xml)
- Example tutorial: [behavior_tree_example.cpp](https://github.com/castacks/AirStack/tree/main/robot/ros_ws/src/autonomy/5_behavior/behavior_tree_example/src/behavior_tree_example.cpp)

---

## Summary

This guide covered:

1. **Why Behavior Trees**: Modularity, reusability, and decoupling advantages over FSMs
2. **Core Concepts**: Node types, tick propagation, status values
3. **Codebase Analysis**: Implementation details of each node type
4. **Hands-On Tutorial**: Five checkpoints to build a complete behavior system
5. **Debugging**: Common issues and diagnostic approaches
6. **Extension**: Adding custom nodes and multi-robot deployment

The key to successful behavior tree development is maintaining strict consistency between:

- Node names in `.tree` files
- String labels in executive Condition/Action constructors
- ROS topic namespaces in launch configurations

Following the checkpoint-based approach ensures each component works before adding complexity.
