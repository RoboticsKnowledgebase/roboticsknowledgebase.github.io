---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-05-01 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Simulating UGVs in Unity
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
## Overview

Good robotics simulators already exist for popular, common domains like autonomous driving and drone surveying. However, for specialized projects, it's often advantageous to build a simulator yourself. By developing a custom simulator, you can tune the environment, the physics, the lighting, and the communication bridge exactly as you like, and this flexibility will save you development time in the long run.

This tutorial serves as a general walkthrough for developing a custom simulator using the Unity game engine. Unity is free for non-commercial use. It has a highly active, global community. Applications written in Unity can be run in web browsers, on phones, game consoles, and, of course, high-end workstations. The only difference between a video game and a robotics simulator is what we'll call the *bridge*: the connection between the simulated environment and the rest of your robotics software stack.

## Creating a Unity project

First, install the latest version of the Unity Hub. You will need to create a free Unity account. We recommend using your .edu email address for your account. Once you have logged into the Hub, click the blue New Project button. This will show a long list of project templates. Here we will compare the three most relevant.

The **3D (Built-In Render Pipeline)** template is the most common template for all Unity projects. While it has the most extensive user base, and therefore the most extensive documentation, of all templates, its underlying rendering pipeline is rigid and difficult to customize.

The **Universal 3D** template uses Unity's **Universal Rendering Pipline** (URP). URP is a scriptable rendering pipeline (SRP), which means that you may write scripts to alter nearly every aspect of the game engine's rendering process. SRPs are useful to developers who wish to add custom effects to their world, from dust storms to blades of grass bending in the wind to lens flares.

The **High Definition 3D** template uses Unity's **High Definition Rendering Pipeline** (HDRP). The HDRP is another SRP. While the URP is designed to be performant enough to run on most computers and even phones, the HDRP's focus is on realism at the cost of performance. That means that projects using HDRP need to be run on devices with modern GPUs. The visual benefits are profound, as demonstrated in the video below.

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/8VRVWSlVuDQ?si=DjhlyC8fYs5UNmCl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Once you've selected your template, choose a name and open the project in the Unity Editor. We recommend using the latest LTS version of the Editor. Now is a good time to run `git init`.

## Installing Ros2 for Unity

[Ros2 for Unity](https://github.com/RobotecAI/ros2-for-unity) is an unofficial, high-performance ROS common library for C#. Here we assume that you are developing on an Ubuntu system.

1. Clone the repository: `git clone git@github.com:RobotecAI/ros2-for-unity.git ~/ros2-for-unity`
2. If working in Humble, adjust your branch: `git checkout humble`
3. Source ROS2: `. /opt/ros/humble/setup.bash`
4. Pull any message repositories: `./pull_repositories.sh`
5. Build: `./build.sh --standalone --clean-install`
6. Copy the files under `install/asset/` to your Unity project.

More details are available on the [repo wiki](https://github.com/RobotecAI/ros2-for-unity/blob/humble/README-UBUNTU.md).

## Creating a scene

A "scene" is a Unity file that contains a map, game objects, and associated data. It's often called a "level" in video games. Your project will start with an empty scene. You may either build this scene up from scratch or use a demo scene.

To use a demo scene, we recommend downloading and importing the official HDRP (or URP) Demo Scene from the Unity Asset Store.

## Creating an ego vehicle

To create an ego vehicle, follow online guides for creating a simulated car in Unity. Ultimately, you should have a root game object for the robot's chassis, one wheel collider per wheel, a Unity camera, and any additonal meshes and game objects for custom sensors and mechanisms.

## Connecting to ROS2

For each script that needs to publish or subscribe to ROS2 messages, create a "hook object" by adding a `ROS2UnityComponentObject` somewhere in the scene (this was copied from `install/asset/`), then adding this to your script:

```c#
using ROS2;
...
// Example method of getting component, if ROS2UnityComponent lives in different GameObject, just use different get component methods.
ROS2UnityComponent ros2Unity = GetComponent<ROS2UnityComponent>();

private ROS2Node ros2Node;
...
if (ros2Unity.Ok()) {
    ros2Node = ros2Unity.CreateNode("ROS2UnityListenerNode");
}
```

You may then publish or subscribe to topics as outlined in the [Ros2 for Unity README](https://github.com/RobotecAI/ros2-for-unity/tree/humble).

