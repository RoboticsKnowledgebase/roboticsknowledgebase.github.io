---
date: {}
title: Title goes here
published: true
---
## Introduction
This wiki aims to build on the already excellent A* implementation guide by providing information about different planning approaches to counter the pitfalls of standard A* when applied to physical robots. In the scope of this article are two popular extensions - Theta Star and Safe Interval Path Planning (SIPP) that each have their strengths and weaknesses.

Theta Star planning is a pathfinding algorithm that is an extension of the popular A* algorithm. The key difference between Theta Star and A* is that Theta Star uses a line-of-sight check to reduce the number of nodes that need to be explored during pathfinding. The line-of-sight check is a simple test to determine if a path between two points is blocked by any obstacles. If there is a clear line of sight, the path is considered free and the algorithm can move on to the next node. If there is an obstacle in the way, the algorithm needs to generate a new path around it. 

Safe interval path planning is a method of planning robot motion that ensures collision avoidance and safety. It involves dividing the robot's workspace into safe intervals or regions, where the robot can move without colliding with obstacles. The path planning algorithm then generates a safe path by connecting these intervals. This approach ensures that the robot's motion is safe and collision-free. Safe interval path planning is essential for applications where robots are required to work in close proximity to humans or other objects.

## Overview of Theta* Algorithm

Theta* algorithm is a pathfinding algorithm that is an extension of the A* algorithm. It is used to find the shortest path between two points in a 2D grid-based environment.

#### Important Terms
Here are the key steps and important terms involved in the Theta* algorithm:
- **Grid Map**: A 2D grid-based environment that represents the space in which the robot moves. Each cell in the grid represents a possible position that the robot can occupy.

- **Heuristics**: A function used to estimate the distance between two points in the grid map. The heuristic function used in Theta* is often the Euclidean distance or Manhattan distance. For even better results, it is recommended to use a pre-computed Dijkstra map since it accounts for static obstacles if the map is known

- **Open List**: A list of nodes that need to be explored during pathfinding. Each node in the list represents a position in the grid map and its associated cost.

- **Closed List**: A list of nodes that have already been explored during pathfinding.

- **Start and Goal Nodes**: The starting and ending points in the grid map between which the shortest path needs to be found.

- **Line of Sight Check**: A test that determines if there is a clear path between two points. If a clear path exists, the algorithm can "shortcut" between two nodes, reducing the number of nodes that need to be explored. In practice, this can be implemented as using Bresenham's line drawing equation borrowed from Computer Graphics to approximate lines on 2D grids, with possible speed ups.

Theta* should always give paths with lower cost than those from standard A*, since the line-of-sight check can only shorten paths. In practice, the paths tend to mimic visibility graphs around obstacles

![Node Exploration Behaviour of Theta Star](assets/images/theta_star_explor.png)


#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
