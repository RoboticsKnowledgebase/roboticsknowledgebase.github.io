---
date: {}
title: An Introduction to Isaac Sim
---
Nvidia's Isaac Sim is quickly becoming a must-know in the world of robotics training through simulation, leveraging powerful GPUs to train numerous agents simultaneously. While our project didn't involve reinforcement learning or parallel simulations, diving into Isaac Sim proved a rewarding detour.

Navigating the complexities of new software can be daunting, and while Nvidia's official documentation provides a solid foundation, it can feel insufficient for beginners. This realization struck me particularly hard as I embarked on my project—there seemed to be a surprising lack of tutorials or comprehensive guides considering Isaac Sim's burgeoning popularity.

Therefore, our took it upon ourself to shed some light on this innovative tool. This blog post aims to simplify your journey with Isaac Sim, providing insights and step-by-step guidance to make your experience as enriching as mine was. So, let’s dive into the world of advanced simulation with Isaac Sim and explore its vast potentials together!

"NVIDIA Isaac Sim is an extensible robotics simulation platform that gives you a faster, better way to design, test, and train AI-based robots. It’s powered by Omniverse to deliver scalable, photorealistic, and physically accurate virtual environments for building high-fidelity simulations".  

## The 1st look
NVIDIA's Isaac Sim is a cutting-edge robotics simulation platform designed to streamline the design, testing, and training of AI-based robots. Harnessing the power of NVIDIA's Omniverse, it offers scalable, photorealistic, and physically accurate virtual environments. This allows for the creation of high-fidelity simulations, significantly enhancing the development process of robotic systems.

But what exactly is NVIDIA Omniverse? It's a comprehensive platform comprising APIs, SDKs, and services that facilitate the integration of Universal Scene Description (OpenUSD) and RTX rendering technologies. This integration is crucial for developers looking to incorporate advanced photorealistic rendering capabilities and GPU-accelerated performance into their existing software tools and simulation workflows, particularly in AI system development.

With a better understanding of the foundational technologies behind Isaac Sim, we can now delve into the practical application of this platform. This involves utilizing Isaac Sim to simulate robotic operations, which I will guide you through step-by-step, ensuring you can leverage this powerful tool to its fullest potential. Let's explore how to effectively simulate robots using NVIDIA Isaac Sim.

## The 1st Delima
As a seasoned roboticist familiar with ROS and Gazebo, I've grown accustomed to working with URDFs (Unified Robot Description Format) when discussing simulation. However, NVIDIA's Isaac Sim introduces a shift by utilizing the USD (Universal Scene Description) format to define robot properties. While this transition might initially raise eyebrows among traditional roboticists, the USD format synergizes with the Omniverse platform to deliver exceptionally photorealistic simulations, a crucial aspect for achieving zero-shot sim-to-real transfer.

NVIDIA supports the process of importing URDF files into Isaac Sim with an [open-sourced extension](https://github.com/NVIDIA-Omniverse/urdf-importer-extension) designed specifically for this purpose. Here’s how you can integrate your existing URDF models into Isaac Sim.

#### Steps to Import URDF 

1. To access this Extension, go to the top menu bar and click Isaac Utils > Workflows > URDF Importer. 

![Step 1 - Access URDF Importer](assets/images/isaac_img_init.png)

2. Adjust the import settings to suit your robot's specifications:
- **Fix base link:** Deselect for mobile robots; select for manipulators.
- **Stage Units per meter:** Setting this to 1 equates one unit in Isaac Sim to 1 meter.
- **Joint Drive Type:** Choose between position or velocity, depending on project needs.
- **Joint Drive Strength and Joint Position Drive Damping:** Recommended values are **10000000.0** and **100000.0** respectively to ensure accurate joint movement. These values are in Isaac units, and emperically we found that, the robot joint doesn't move as they should, if the values are not specified. 
- **Self-Collision:** Typically left unselected as the importer manages collision properties adequately, though enabling it does not impede the process.

 ![Step - 2 Define Import Properties](assets/images/isaac_img_import_settings.png)

3. Click the **Import** button to add your robot to the stage, visualizing it within the simulation.

![Step - 3: Import the URDF](assets/images/isaac_img_import.png)

4. Since Isaac Sim does not automatically create a ground plane, thus we need to create a ground plane.

![Step - 4: Create Ground Plane](assets/images/isaac_img_ground_plane.png)

5. Confirm that the collision properties of your imported robot function correctly.

![Step - 5: Verify the Collision Properties](assets/images/isaac_img_colliders.png)

![Step - 5: Verify the Collision Properties](assets/images/isaac_img_collision_vis.png)


6. Voila, we have successfully imported our URDF to Isaac Sim! Though, the import plugin saves the USD file (check Output Directory option while importing), but that is in **.usd** format which is a file binary format, which obiously can't be read by humans. Thus we will go ahead and save it in **.usda** format. USDA is essentially an ASCII format of USD file that is encoded as UTF-8. 

![Step - 6: Saving as USDA](assets/images/isaac_img_save_as.png)

![Step - 6: Saving as USDA](assets/images/isaac_img_save_as_usda.png)

## Let's Integrate with ROS

After successfully importing the URDF into Isaac Sim and saving the USD file, the next step is to ensure that all expected ROS topics are being published as anticipated. When I opened a terminal, entered the `ros2 topic list` command, and pressed enter, surprisingly, no topics appeared. I was confident that all necessary plugins were defined in the URDF. However, it became clear that the URDF importer does not handle these plugins, and Isaac Sim does not natively support ROS or ROS 2. This means that each component requiring topic publication must have a dedicated workflow defined.

These workflows are defined as Action Graphs which are a part of OmniGraph. Omnigraph is Omniverse’s visual programming framework that seamlessly integrates various systems within Omniverse, enabling the creation of customized nodes and efficient computation for applications like Replicators, ROS bridges, and sensor management in Isaac Sim. Read [this](https://docs.omniverse.nvidia.com/isaacsim/latest/gui_tutorials/tutorial_gui_omnigraph.html#isaac-sim-app-tutorial-gui-omnigraph) article from Nvidia for more details.

The steps to create an Omnigraph node that connects our simulation environment to ROS or ROS 2 can be approached in several ways: entirely through a GUI, scripting within the extension workflow, through standalone Python code, or a mix of both GUI and Python. For this tutorial, we will use GUI to make an action graph and write a script to launch the simulation. While the documentation provides numerous examples for standard sensors and actuaries, I'll briefly discuss them here, directing you to the documentation for more detailed information.

We will explore the workflow for publishing Transforms (TFs), essential for any robotic application, particularly where our project faced significant challenges. Our focus was on establishing a simulation environment for a mobile manipulator, integrating closely linked TFs and Odometry. The ROS 2 extension provides a script node for publishing TFs, requiring specific configurations that we will set up through the GUI.

### Steps to Create Action Graph

- Go to top menu bar and click Window -> Visual Scripting -> Action Graph
- Click **New Action Graph** to open an empty graph
- To start building our action graph, we start adding nodes. The first node we need to add is **'On Playback Tick'**. The On Play Tick node acts as a trigger that executes at every simulation tick, which is a single update cycle of the simulation. 

![Image of Adding On PLayback Tick](assets/images/isaac_img_on_playback_tick.png)

- The next node is **'ROS 2 Context'**. This node acts as a bridge between Isaac Sim and ROS 2, enabling the simulation to communicate and interact with ROS 2-based systems. It sets up the necessary configurations to ensure that the simulation can send and receive messages, services, and actions to and from ROS 2.

![Image of Adding ROS 2 Content](assets/images/isaac_img_ros2_context.png)

- One of the most important node we add is **'Isaac Read Simulation Time'** that is designed to capture and provide access to the current simulation time within the simulation environment. This node is crucial for operations and tasks that depend on the simulation's temporal state. 

![Image of Adding Isaac Read Simulation Time](assets/images/isaac_img_read_sim_time.png)

- Now that all the house-keeping node are added, lets add node to which compute the odometry. Isaac Sim does include a computational node to calculate odometry named **'Isaac Compute Odometry Node'**. <-------------Talk about adding the articulated robot here-----------------> 

![Image of Defining Articulation Root](assets/images/isaac_img_add_articulation_root.png)

![Image of Adding Isaac Compute Odometry Node](assets/images/isaac_img_isaac_compute_odom.png)

- Once we have the node computing odometry, we now will publish the odometry data. The ROS2 plugin has node for publishing the odometry data called the **'ROS2 Publish Odometry Node'**. 

![Image of Adding ROS2 Publish Odometry Node](assets/images/isaac_img_publish_odom.png)

- Similar to node which publishes odometry, the ROS2 plugin has nodes for publishing Transforms called the **'ROS2 Publish Transform Tree'**. Note that, this node will only publish transforms which are dynamics, essentially any prim which is not static. To publish transforms for the static node, we will additonally add **'ROS2 Publish Raw Transform Tree'** node, which, as the name suggest, publishes transform of static prim. 

![Image of Adding ROS2 Publish TF](assets/images/isaac_img_publish_tf_1.png)

We also need to define all the child links for which we need to publish the transforms for. 

![Image of Adding ROS2 Publish TF](assets/images/isaac_img_publish_tf_2.png)

![Image of Adding ROS2 Publish Static TF](assets/images/isaac_img_publish_static_tf.png)

Awesome, now that we have our entire action graph, lets save it (in .usda format). 

![Image of Saving Action Graph](assets/images/isaac_img_save_action_graph_1.png)

![Image of Saving Action Graph](assets/images/isaac_img_save_action_graph_2.png)


## Lets Write a Launch File

Well, launch file here means slightly different from what a ros launch file is, but it does a similar action, load and run everything we need that is needed to run the entire simulation.

```python
# Launchs the simulation
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})
```

This code will essentially launch the barebone simulator. Before proceeding with the import of other Omniverse libraries, it is crucial to complete this step first. Skipping this step will result in errors when attempting to import additional Omniverse libraries. Ensure this initial step is properly executed to maintain a smooth integration process. 

Now we import other libraries which are important for loading the USD and the aciton graph. 

```python
# Importing other libraries
import logging
import argparse
import numpy as np

# Isaac Libraries
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core_nodes.scripts.utils import set_target_prims

# ROS2 Libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
```

-------------------Some text about libraries-------------------

```python
# We make it a ROS2 Node
class simWorld(Node):
    def __init__(self) -> None:
        # Initializing the ROS2 Node and can be used to define any other ROS functionalities
        super().__init__('test_node')

    def scene_setup(self) -> None:
        logging.info("Defining World")
        self.world = World(stage_units_in_meters = 1.0)
        logging.info("World Setup Completed")

        logging.info("Initializing Scene Setup")
        # Adding objects to Simulation
        add_reference_to_stage("env/aims_env.usda", prim_path="/env")
        add_reference_to_stage("robots/fetch.usda", prim_path="/fetch")

        # Updating Simulation
        simulation_app.update()
        logging.info("Scene Setup Completed")
        self.ros2_bridge_setup()
        
```
In the provided code snippet, we begin by defining and logging the initialization of a virtual world within an Omniverse application. The method establishes a `World` object, specifying that one unit in this isaac environment equates to one meter. Then proceeds to populate the simulation with objects, specifically adding environmental and robotic elements from USDA files to the stage. The simulation environment is then updated to reflect these additions. 

```python
def ros2_bridge_setup(self) -> None:
        # enable ROS2 bridge extension
        logging.info("Setting up ROS Bridge")
        extensions.enable_extension("omni.isaac.ros2_bridge")
        logging.info("ROS Bridge Setup done")
        self.action_graph_setup() 

```

This method enables the ROS 2 bridge in Isaac Sim. Additionally, it calls another method which will load the action graphs, as described below. 

```python
def action_graph_setup(self) -> None:
	# Adding TF and Odom Action Graph
	add_reference_to_stage("robots/tf.usda", prim_path="/fetch/tf")
    
    # (Additional, but might come handy)
    # This is used to define/set properties in Action Graph
    set_target_prims(primPath="/fetch/tf/ActionGraph/ros2_publish_transform_tree", 
            inputName="inputs:targetPrims", 
            targetPrimPaths=["/fetch/base_link"]
     )
     self.run_simulation()
```
The method above adds the action graph to our simulation setup. 

> It is crucial to position the action graph responsible for publishing the Transforms (TFs) under the robot's primary path (prim path). This ensures that the TFs correctly identify and relate to the robot's body. Our team learnt it the hard way. 

While the target prim is set during the action graph's definition, it's common to encounter issues where these settings do not persist when loading through a script. To address this, we use the function `set_target_prims` to explicitly set the value of the target prim again, ensuring accurate and consistent referencing within the simulation.

And finally, we define a function to run the entire simulation by taking steps. Remember, before running the simulation its neccesary to initialize the physics setting by calling the `initialize_physics` method from the World object. We also define a ground plane (incase it wasn't defined in our environment USD file). We define an infinite loop and run the simulation until any erorr or keyboard interrupt is encounter. 

```python
def run_simulation(self) -> None:
        # need to initialize physics getting any articulation..etc
        logging.info("Initializing Physics")
        self.world.initialize_physics()
        self.world.add_default_ground_plane()
        self.world.play()

        while simulation_app.is_running():
            # Run with a fixed step size
            self.world.step(render=True)
            # For updating ROS 2 blocks
            rclpy.spin_once(world, timeout_sec = 0)

        simulation_app.close()
```

And thats it!!! We have defined a launch script to bring everything together. 

## Summary
In this article we learnt, how to import a standard URDF file to Isaac Sim, connect it to ROS and how to write a launch file to bring everything together. While this example covers the lessons we learnt the hard way, there is a lot more to explore and we highly recommend checking out the official documentation for a deeper understanding. 


## END
## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.



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
