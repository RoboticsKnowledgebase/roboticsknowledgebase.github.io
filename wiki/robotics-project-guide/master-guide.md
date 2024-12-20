---
# layout: single
title: Starting a Robotics Project
mermaid: true
---

Starting a robotics project can be overwhelming. There's too many things to consider and if it's your first time doing robotics, then you may be flooded with all the options available in the internet. Let's say that you are developing a quadrupedal robot named `Tod`. `Tod`'s objective is to carry products in a shopping center for the elders so that they won't have a hard time shopping. We don't want to overwhelm you, but here's a map of possible options for each component of `Tod`. 
```mermaid
mindmap
  root((Tod))
    Quadruped Hardware
      Boston Dynamics Spot
      Unitreee 
        Go1
        Go2
        B2
      ANYmal
    Sensors
      Vision
      IMU
      State Estimation
    Programming Language
      C++
      Python
    Simulations
    Mapping
    Loco Controllers
    Manipulators
    Communications
      ROS
      ROS2

```

# Start here
If you are starting a robotics project for the first time, this is a good place to begin! Here, we will give you some guidance on what to consider in each step of a robotics project in detail.

Below is the overall flow you would need to take in a robotics project. Click on a step that you are interested in and it will take you there!

```mermaid
flowchart TD;
    A[Define your goals and requirements] --> B[Choose a robot];
    A --> I[Make a robot];
    B --> C[Choose your language];
    I --> C
    C --> D[Choose your communication method];
    D --> E[Choose peripherals];
    E --> F[Choose your simulator];
    F --> G[Test and debug your robot];
    G --> H[Demo day!];

    click A href "/wiki/robotics-project-guide/define-your-goals-and-requirements/"
    click B href "/wiki/robotics-project-guide/choose-a-robot/"
    click I href "/wiki/robotics-project-guide/make-a-robot/"
    click C href "/wiki/robotics-project-guide/choose-a-language/"
    click D href "/wiki/robotics-project-guide/choose-comm/"
    click E href "/wiki/robotics-project-guide/choose-peripherals/"
    click F href "/wiki/robotics-project-guide/choose-a-sim/"
```

<!-- click F href "/wiki/robotics-project-guide/test-and-debug/"
click G href "/wiki/robotics-project-guide/demo-day/" -->
