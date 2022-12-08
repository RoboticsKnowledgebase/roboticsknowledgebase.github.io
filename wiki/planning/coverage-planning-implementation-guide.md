---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-12-07 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Coverage Planner Implementation Guide
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
This wiki details how to implement a basic coverage planner, which can be used for generating paths such that a robot can cover a region with a sensor or end-effector. 

## Introduction to Coverage Planning
 The goal of coverage planning is to generate a path such that a robot sweeps every part of some area with one of its sensors and/or end-effectors. One common example where you would need a coverage planner is for drone monitoring applications, where you need the drone to follow a path such that it's cameras get a view of every part of some larger field/area. Another example would be a robot lawnmower or vacuum, where we want the robot to move such that it mows or vacuums every square meter of some field or room. 
 
 In this guide, we will be focussing on a cellular-decomposition based coverage planner. This type of coverage planner first splits the region to cover into smaller cells such that it is trivial to plan a path to cover the simpler cells. We then plan a cell traversal, and the robot goes from cell-to-cell, covering each simpler cell in order to cover a larger area. There are other broad coverage planner methods. For example, grid-based coverage planning methods would decompose the region into an occupancy grid map, and then use graph algorithms to ensure the robot covers each grid cell. However, these other forms of coverage planners are out of the scope of this simple tutorial.

 In particular, the coverage planner will take in a list of vertices comprising some polygon and the planner will output the vertices for a lawnmower path to cover this polygon. In addition to the outer boundary, the coverage planner will also accept a list of holes, with each hole being a list of vertices, where holes represent regions we do not care about covering. 

## High Level Architecture of Coverage Planner
Before we dive into the small details of this coverage planner, we will first look at the high level architecture of the coverage planner we will be designing. The figure below shows the three main steps. 

![Put a relevant caption here](assets/coverage_planner_steps.png)

The input to the coverage planner is the region we want to generate the coverage plan for. We will represent our region of interest as an outer polygon with a set of polygonal holes. The holes represent areas within the overall outer polygon that we don't want to cover. For instance, in a robotic lawnmowing application, holes could represent gardens in the middle of the lawn that we don't want to mow. Similarly, in a drone wildfire-monitoring application, holes might represent lakes, and so you would not want to spend time flying over the lakes as you wouldn't expect any fire to be there. The outer polygon as well as polygonal holes each are represented as lists of vertices, with each vertex being an x,y coordinate pair.

Once we have our input region, the first step is cellular decomposition, where break up the complicated region of interest into simpler shapes which we call cells. In particular, we will perform trapezoidal decomposition, which forms trapezoidal cells. The intention for performing this decomposition is that we can use a simpler algorithm to generate coverage plans for the individual trapezoids, and we can then combine these coverage plans to form a coverage path for the entire region.

The second step of the algorithm will be generating a cell traversal. This means determining which order to visit the trapezoidal cells. Once we have this order, then in the third step, we form the full coverage path. In particular, we use a simpler coverage planner algorithm that works specifically for trapezoidal cells. Starting at the first cell in the cell traversal, we iterate between covering the current cell with our simpler coverage planner and travelling to the next cell in the cell traversal. This final path should now cover the entire region of interest.

## Step 1: Forming a Trapezoidal Decomposition

As mentioned above, the first step to our coverage planner will be decomposing the region of interest into simpler trapezoids. To perform this trapezoidal decomposition, we will use a vertical sweep line method. This involves "sweeping" a vertical line from left to right across the region. As the sweep line encounters events, which correspond to vertices, it processes them. We maintain a list of trapezoidal cells that are currently open, meaning that their right edge is unknown. Processing an event involves closing some open cells and opening new cells. Once the sweep line has made it past the right-most event, there should be no more open cells and the closed cells represent the full trapezoidal decomposition of the region.

Diving into more detail, the first step of trapezoidal decomposition is to convert the outer boundary and holes into a list of events. To do this, we need to discuss what an event is. Event's correspond to vertices of the region of interest, but they also contain some additional information. In the addition to the current vertex (the vertex that the event corresponds to), an event contains the previous vertex, the next vertex, and the event type. The previous vertex and next vertex refer to the two vertices directly connected to the current vertex via edges. In order to distinguish between the next and previous vertex, we will use the convention that as you traverse the edges of the outer boundary or a hole, the region of interst (area you care about covering), will be to your left. Thus, we will traverse the outer boundary in counter clockwise order and traverse holes in clockwise order.

The 6 different event types we will define are OPEN, CLOSE, IN, OUT, FLOOR, CEILING events. A graphic displaying these 6 event types is shown below. Note that we are currently assuming that all vertices have a unique x-component. We detail some steps to get around this assumption in a section below. This classification into different event types will be useful later as we each type gets processed in a slightly different way by the sweep line. 

To generate the list of events, loop through the vertices of the outer boundary in CCW order and loop through the vertices of each hole in CW order. At each vertex, add a new event with its previous, current, and next vertex. To determine the event type, you need to examine the x and y components of the previous and next vertex. For example, if both the previous and next vertex have an x component to the left of the current vertex, the event is either an OUT or CLOSE event. Comparing the y coordinate of the previous and next vertex can then distinguish between these two event types. Similarly, if the previous and next vertex are both to the right of the current vertex, the event type is either OPEN or IN. Again, you can then compare the previous and next vertex's y coordinate to distinguish them. If the previous vertex is to the left of the current vertex and the next vertex is to the right of the current vertex, the event type is FLOOR. Finally, if the previous vertex is to the right while the next vertex is to the left, the event type is CEILING. 

We can't actually have the computer sweep a vertical line across the region of interest, but we can mimic this behavior by processing each event from left to right. Thus, once you have a list of events, you will have to sort them by their x-component. 

As we then loop through this sorted list, we will maintain a list of open cells, closed cells, and current edges. Thinking back to the analogy of a sweep line being pulled from left to right across the region of interest, the open cells correspond to trapezoidal cells we are forming which the sweep line is currently intersecting. Closed cells would be trapezoidal cells completely to the left of the sweep line. Closed cells have a known right boundary whereas the right boundary for open cells is unknown (the process of closing a cell corresponds to the determination of its right boundary). Finally, current edges represents all edges that the sweep line is intersecting.  

Looping one by one from left to right through the sorted list of events, we individually process each event. The first step to processing an event is identifying the edge immediately above and below the current event. We call these edges the floor and ceiling edge for the event. To find the floor and ceiling edge for an event, we loop through the list of current edges (edges being intersected by the sweep line). We find the vertical intersection point between the sweep line and each edge. The ceiling edge is chosen as the edge who's vertical intersection point is closest to the event while still being above the event. The floor is chosen similarly but must be below the event. 

Now that we have the immediate ceiling and floor edges above and below the event, we can process the event. Each event type is handled differently. At “open” events, we create a new open trapezoid. At “close” events, we close a trapezoid (add the right edge) without opening any new trapezoids. At “floor” and “ceiling” events, we close one trapezoid and open a new one. At “in” events, we close one trapezoid and open two new ones. Finally, at “out” events, we close two trapezoids and open a new one. The floor and ceiling edges are used in the process of creating new trapezoids.

The end result of this step is a set of closed trapezoidal cells.

![Put a relevant caption here](assets/coverage_planner_event_types.png)
## Step 2: Generating a Cell Traversal

Once we have a set of trapezoidal cells, we need to determine a cell traversal. This is an order in which to visit each cell. For our purposes, we will just use a depth first search to generate the cell traversal, although more complicated TSP solvers could be used.

## Step 3: Synthesizing the Full Coverage Plan

Once you have a cell traversal, we form the full coverage plan. Starting at the first cell in the cell traversal, alternate between generating a coverage plan for the given cell, and generating a path to the next cell. The path to the next cell is as simple as a straight line, assuming you do not care if the robot crosses over holes occasionally. To generate the cell coverage plan, we can simply generate a back-and-forth lawnmower pattern over the cell.

## Summary
Overall, coverage planning is useful for tasks that require scanning of an area by a robot. We have seen how we can generate such paths over complicated areas by first splitting the region into simpler trapezoidal cells, planning a traversal across those cells, and then using a simple back-and-forth lawnmower pattern to cover each trapezoid.
