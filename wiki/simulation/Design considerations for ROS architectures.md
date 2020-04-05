# Overview
Over the past few months me and my team have dicussed various ROS architectures, changed it at least two times in a major way and have come to appreciate the importance of designing a good architecture. We have also realised there are several important factos which need to be considered before nailing down a particular architecture. In this post I will detail the design decisions we had to make in the context of our project.

# Background
To give a brief background of the project we are working on, it is a learning to drive in simulation project. We have several nodes performing various tasks like state extraction from simulator, path planning and a RL agent making decisions. All of this runs in a loop for the simulation to continuously render and reset.

# Design Considerations

