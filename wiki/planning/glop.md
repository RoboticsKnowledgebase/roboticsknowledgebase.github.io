---
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

In the context of robotic task planning, Convex Optimization refers to the process of minimizing a specific cost function over a set of variables, subject to constraints. This property ensures that if a local minimum exists, it is also the global minimum—a critical feature for reliable autonomous decision-making. This entry outlines how to use the Google Linear Optimization Package (GLOP) in C++ to solve such problems. We will cover the theoretical definition, setting up cost functions, integrating the GLOP solver, and converting the abstract mathematical output into actionable robot poses, using the Lunar ROADSTER project as a case study.

In this tutorial, we demonstrate how to implement a Linear Programming solver to optimize the movement of lunar regolith from source piles to target craters. We will walk through the installation of Google OR-Tools, the formulation of the mathematical model, and the C++ code required to solve it. By the end of this guide, you will be able to set up your own optimization pipeline for resource allocation or path planning tasks.

<!-- This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study). -->

## Convex Optimization

Before diving into code, it is vital to understand the mathematical structure GLOP expects. A specific and highly useful subset of convex optimization is Linear Programming (LP). In an LP problem, both the objective function and all constraints are linear equations.

The standard form is represented as:

$$
\begin{aligned}
& \text{Minimize} & & \mathbf{c}^T \mathbf{x} \\
& \text{Subject to} & & A \mathbf{x} \leq \mathbf{b} \\
& \text{and} & & \mathbf{l} \leq \mathbf{x} \leq \mathbf{u}
\end{aligned}
$$

Where:

- $\mathbf{x}$ is the vector of decision variables (e.g., "How much regolith to move").
- $\mathbf{c}$ is the cost vector (e.g., "Distance to travel").
- $A, \mathbf{b}$ represent the constraints (e.g., "Cannot carry more than 5kg").

For the Lunar ROADSTER project, we applied this theory to the "Transportation Problem." Our robot must move regolith from high-elevation areas (Sources) to low-elevation craters (Sinks) to flatten the moon's surface.

### The Cost Function

Our goal is to minimize the total "Work" performed by the system. Mathematically, this is the sum of the volume moved multiplied by the distance traveled:

$$J = \sum_{i \in \text{Sources}} \sum_{j \in \text{Sinks}} \text{Distance}_{ij} \times \text{Flow}_{ij}$$

### The Constraints

We must adhere to physical conservation laws:

- **Source Constraint**: The total volume removed from a specific source node $i$ cannot exceed the amount of regolith actually present at that node.
- **Sink Constraint**: The total volume dumped into a sink node $j$ cannot exceed the hole's available capacity.

Note: In real-world scenarios, the total volume of dirt available to be mined rarely matches the total capacity of the craters exactly. A standard LP solver will return INFEASIBLE if these sums do not match in an equality constraint. To fix this, we used a Big-M relaxation logic to allow for partial filling or partial removal.

## GLOP Introduction

The Google Linear Optimization Package (GLOP) is the primary linear solver provided by Google's open-source OR-Tools suite. It is designed to solve large-scale LP problems efficiently.

We chose GLOP for three main reasons:

- **C++ Integration**: It offers a native C++ API, making it easy to embed directly into ROS/ROS2 nodes.
- **Performance**: It utilizes advanced primal and dual simplex algorithms, capable of solving problems with thousands of variables in milliseconds.
- **Robustness**: It handles numerical instability better than many naive implementations.

### Installation

To use GLOP, you must install the Google OR-Tools libraries. On Ubuntu, you can often use apt:

```
sudo apt-get install -y libgoogle-ortools-dev
```

Alternatively, if building from source or using CMake, ensure you link the library in your CMakeLists.txt:

```
find_package(ortools REQUIRED)
target_link_libraries(transport_planner PRIVATE ortools::ortools)
```

## GLOP Tutorial

Setting up a solver in C++ follows a standard pattern: Initialize, Define Variables, Define Constraints, Solve. We will break this down step-by-step with logging using ROS2.

### Step 1: Include Headers and Initialize Solver

First, include the linear solver header and ROS2 headers. We then create an instance of MPSolver specifying the "GLOP" backend.

```
#include "ortools/linear_solver/linear_solver.h"
#include "rclcpp/rclcpp.hpp"

void SolveExample() {
    // Create the linear solver with the GLOP backend.
    // MPSolver is the main entry point for any OR-Tools linear solver.
    std::unique_ptr<operations_research::MPSolver> solver(
        operations_research::MPSolver::CreateSolver("GLOP"));

    // Check if the solver was created successfully
    if (!solver) {
        RCLCPP_WARN_STREAM(rclcpp::get_logger("glop_solver"), "GLOP solver not available.");
        return;
    }

    const double infinity = solver->infinity();
```

### Step 2: Define Variables

Variables are the unknowns we want to solve for. In Linear Programming, these are usually continuous.

```
    // Create continuous variables 'x' and 'y' with range [0.0, infinity].
    // Syntax: MakeNumVar(min_value, max_value, "variable_name")
    operations_research::MPVariable* const x = solver->MakeNumVar(0.0, infinity, "x");
    operations_research::MPVariable* const y = solver->MakeNumVar(0.0, infinity, "y");

    // Using RCLCPP_INFO_STREAM allows for << syntax similar to std::cout
    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "Number of variables = " << solver->NumVariables());
```

### Step 3: Define Constraints

Constraints limit the possible values of the variables. They are defined as linear inequalities (e.g., $x + 2y \leq 14$).

```
    // Create a linear constraint: 0 <= x + 2y <= 14.
    // MakeRowConstraint(lower_bound, upper_bound, "constraint_name")
    operations_research::MPConstraint* const c0 = solver->MakeRowConstraint(0.0, 14.0, "c0");

    // Set coefficients for the variables in this constraint.
    // This effectively builds the equation: 1*x + 2*y
    c0->SetCoefficient(x, 1);
    c0->SetCoefficient(y, 2);

    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "Number of constraints = " << solver->NumConstraints());
```

### Step 4: Define Objective Function

The objective function is what we are trying to minimize or maximize (e.g., Maximize $3x + 4y$).

```
    // Access the mutable objective function from the solver.
    operations_research::MPObjective* const objective = solver->MutableObjective();

    // Set coefficients for the objective function: 3*x + 4*y
    objective->SetCoefficient(x, 3);
    objective->SetCoefficient(y, 4);

    // Set optimization direction.
    // Use objective->SetMinimization() for cost functions (like distance traveled).
    objective->SetMaximization();
```

### Step 5: Solve and Retrieve Solution

Finally, invoke the solver and check the result status before accessing the values.

```
    // Run the solver
    const operations_research::MPSolver::ResultStatus result_status = solver->Solve();

    // Check that the problem has an optimal solution.
    if (result_status != operations_research::MPSolver::OPTIMAL) {
        RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "The problem does not have an optimal solution!");
        if (result_status == operations_research::MPSolver::FEASIBLE) {
            RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "A potentially suboptimal feasible solution was found.");
        } else {
            RCLCPP_WARN_STREAM(rclcpp::get_logger("glop_solver"), "The solver could not solve the problem.");
            return;
        }
    }

    // Output the solution variables
    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "Solution:");
    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "Objective value = " << objective->Value());
    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "x = " << x->solution_value());
    RCLCPP_INFO_STREAM(rclcpp::get_logger("glop_solver"), "y = " << y->solution_value());
}
```

## Case Study: Lunar ROADSTER Project

To demonstrate GLOP in a real-world scenario, we examine the Lunar ROADSTER project. The objective was to autonomous grade a lunar analog site ("MoonYard") by moving regolith from high mounds to low craters to achieve a target design elevation.

In this specific implementation, the problem is more complex than the simple tutorial above. We do not have just two variables; we have a decision variable for every possible path between a Source node (pile) and a Sink node (crater).

### Step 1: Generating Nodes from the Height Map

Before optimizing, we must discretize the continuous terrain into actionable nodes. We parse the 2.5D height map grid cell by cell.

- **Source Nodes**: Cells where the current height is significantly higher than the design height.
- **Sink Nodes**: Cells where the current height is significantly lower than the design height.

```
// TransportPlanner.cpp
void TransportPlanner::init_nodes(std::vector<TransportNode> &source_nodes,
                                  std::vector<TransportNode> &sink_nodes,
                                  float &vol_source, float &vol_sink,
                                  const Map &current_map, const Map &design_map) {

    // Iterate through every cell in the height map grid
    for (size_t i = 0; i < num_cells; i++) {
        float current_h = current_map.getDataAtIdx(i);
        float design_h = design_map.getDataAtIdx(i);

        // Check if this cell needs excavation (Source)
        // 'threshold' prevents the robot from trying to fix minor imperfections (< 1cm)
        if (current_h > (design_h + threshold)) {
            TransportNode node = {pt.x, pt.y, current_h};
            source_nodes.push_back(node);
            vol_source += node.height; // Track total available dirt
        }
        // Check if this cell needs filling (Sink)
        else if (current_h < (design_h - threshold)) {
            TransportNode node = {pt.x, pt.y, -current_h}; // Store as positive capacity
            sink_nodes.push_back(node);
            vol_sink += node.height; // Track total hole capacity
        }
    }
}
```

### Step 2: Defining Decision Variables

Once the nodes are identified, we define the decision variables. We create a variable for every possible combination of Source and Sink. If we have $N$ sources and $M$ sinks, we generate $N \times M$ variables.

```
float TransportPlanner::solveForTransportAssignments(...) {
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));

    // We iterate through the total number of possible connections (policies)
    std::vector<MPVariable *> transport_plan;

    for (size_t t = 0; t < num_cells_policy; t++) {
        // Create a continuous variable [0.0, infinity] for each path
        // 't' represents the index of a specific Source->Sink connection
        transport_plan.push_back(solver->MakeNumVar(0.0, infinity, "t" + std::to_string(t)));
    }
```

### Step 3: Big-M Relaxation Logic

A key challenge in the Lunar ROADSTER project was that the volume of dirt available ($V_{source}$) rarely matched the volume of the craters ($V_{sink}$). If we used strict equality constraints, the solver would fail (INFEASIBLE). To handle this, we implemented the Big-M method to relax the constraints on the side with excess capacity.

```
    // Determine which volume is larger to decide which constraint to relax.
    // If Sinks > Sources, we relax the Sink constraint (don't need to fill them completely).
    // If Sources > Sinks, we relax the Source constraint (don't need to remove all piles).
    float M = std::max(vol_sink, vol_source);

    // Binary decision variable 'b' controls which side gets the relaxation
    int b = (vol_sink > vol_source) ? 0 : 1;
```

### Step 4: Building Constraints

We iterate through every sink to ensure it receives the correct amount of material. We construct the row constraint using the Big-M factor calculated above.

```
    // Iterate through every sink node 'j'
    for (size_t j = 0; j < m_policy; j++) {

        // Constraint: The total input to this sink must be >= Capacity - Relaxation
        // The term (M * (1-b)) becomes 0 if we enforce the constraint, or M (Large Number) if we relax it.
        auto* constraint = solver->MakeRowConstraint(
            (sink_nodes.at(j).height - (M * (1 - b))), infinity);

        // Sum up contributions from ALL source nodes 'i' to this specific sink 'j'
        for (size_t i = 0; i < n_policy; i++) {
            // 'ij_to_index' maps the 2D source/sink pair to our 1D variable vector
            size_t index = ij_to_index(i, j, m_policy);

            // Set coefficient to 1, meaning 1 unit of flow = 1 unit of volume
            constraint->SetCoefficient(transport_plan.at(index), 1);
        }
    }
```

### Step 5: Solving and Extracting Assignments

After building the matrix, we run the solver. If optimal, we extract the non-zero variables, which represent the actual "missions" the robot must undertake.

```
    solver->Solve();

    std::vector<TransportAssignment> new_assignments;

    // Iterate through the solution matrix to find active paths
    for (size_t i = 0; i < n_policy; i++) {
        for (size_t j = 0; j < m_policy; j++) {
            size_t index = ij_to_index(i, j, m_policy);
            float volume = transport_plan.at(index)->solution_value();

            // Filter out microscopic moves (noise) or tasks that are too far away
            if (volume > 0.01 && distance < max_dist_thresh) {
                TransportAssignment task;
                task.source = source_nodes[i];
                task.sink = sink_nodes[j];
                task.volume = volume;
                new_assignments.push_back(task);
            }
        }
    }
}
```

### Step 6: Generating Robot Poses

The solver outputs abstract "flows." We must convert these into geometric Backblading Maneuvers. Instead of simply dumping dirt, the robot drives past the target crater, drops its dozer, and reverses. This fills the crater and smooths the surface simultaneously.

```
void TransportPlanner::makeGoalsFromAssignment(..., std::vector<cg_msgs::msg::Pose2D> &goalPoses) {
    // 1. Calculate Heading (Yaw)
    // Determine the vector from Source to Sink
    double dy = sink.y - source.y;
    double dx = sink.x - source.x;
    double yaw = atan2(dy, dx);

    // 2. Source Pose (Dig)
    // The point where the robot engages the pile
    cg_msgs::msg::Pose2D source_pose = create_pose2d(source.x, source.y, yaw);

    // 3. Sink Pose (Dump)
    // The center of the crater
    cg_msgs::msg::Pose2D sink_pose = create_pose2d(sink.x, sink.y, yaw);

    // 4. Backblading Pose (Smooth)
    // We project a point 'manipulation_distance' meters PAST the sink.
    // The robot drives here, lowers the blade, and reverses to the sink pose.
    double bb_x = sink.x + manipulation_distance * std::cos(yaw);
    double bb_y = sink.y + manipulation_distance * std::sin(yaw);
    cg_msgs::msg::Pose2D bb_pose = create_pose2d(bb_x, bb_y, yaw);

    // Push poses to the trajectory list
    goalPoses.push_back(source_pose);
    goalPoses.push_back(sink_pose);
    goalPoses.push_back(bb_pose);
}
```

## Summary

In this entry, we demonstrated how to transform a robotic task planning problem into a convex optimization problem using GLOP. We established the cost function (minimizing transport effort), defined the linear constraints (conservation of mass), and implemented the solver in C++. Finally, we showed how to convert the mathematical result into a sequence of robot poses for execution. This workflow allows for highly efficient, globally optimal resource allocation that scales well with complex environments.

## References

- ![Lunar ROADSTER Github](https://github.com/Lunar-ROADSTER/Lunar-ROADSTER/tree/SVD-Encore)
- ![Linear Programming Wikipedia](https://en.wikipedia.org/wiki/Linear_programming)
- ![Google OR-Tools C++ Reference](https://developers.google.com/optimization/reference/linear_solver/linear_solver)

## Further Reading

- "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe (Standard text for optimization theory).
- "Introduction to Linear Programming", MIT OpenCourseWare.
