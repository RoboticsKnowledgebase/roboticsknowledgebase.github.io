---
date: {}
title: An Introduction to Isaac Sim
---
Nvidia's Isaac Sim is quickly becoming a must-know in the world of robotics training through simulation, leveraging powerful GPUs to train numerous agents simultaneously. While our project didn't involve reinforcement learning or parallel simulations, diving into Isaac Sim proved a rewarding detour.

Navigating the complexities of new software can be daunting, and while Nvidia's official documentation provides a solid foundation, it can feel insufficient for beginners. This realization struck me particularly hard as I embarked on my project—there seemed to be a surprising lack of tutorials or comprehensive guides considering Isaac Sim's burgeoning popularity.

Therefore, our took it upon ourself to shed some light on this innovative tool. This blog post aims to simplify your journey with Isaac Sim, providing insights and step-by-step guidance to make your experience as enriching as mine was. So, let’s dive into the world of advanced simulation with Isaac Sim and explore its vast potentials together!

"NVIDIA Isaac Sim™ is an extensible robotics simulation platform that gives you a faster, better way to design, test, and train AI-based robots. It’s powered by Omniverse™ to deliver scalable, photorealistic, and physically accurate virtual environments for building high-fidelity simulations".  

## The 1st look
NVIDIA's Isaac Sim is a cutting-edge robotics simulation platform designed to streamline the design, testing, and training of AI-based robots. Harnessing the power of NVIDIA's Omniverse, it offers scalable, photorealistic, and physically accurate virtual environments. This allows for the creation of high-fidelity simulations, significantly enhancing the development process of robotic systems.

But what exactly is NVIDIA Omniverse? It's a comprehensive platform comprising APIs, SDKs, and services that facilitate the integration of Universal Scene Description (OpenUSD) and RTX rendering technologies. This integration is crucial for developers looking to incorporate advanced photorealistic rendering capabilities and GPU-accelerated performance into their existing software tools and simulation workflows, particularly in AI system development.

With a better understanding of the foundational technologies behind Isaac Sim, we can now delve into the practical application of this platform. This involves utilizing Isaac Sim to simulate robotic operations, which I will guide you through step-by-step, ensuring you can leverage this powerful tool to its fullest potential. Let's explore how to effectively simulate robots using NVIDIA Isaac Sim.

## The 1st Delima 
Being a roboticist who has been using ROS and Gazebo for years, the 1st thing that comes to mind when we talk about simulation is URDFs. But as we discussed above, Isaac Sim doesn’t use URDFs but rather uses USD format to descirbe the properties of the robot. While this choice might not be highly appreicated by roboticist, USD format allows them to use the power of Omniverse to make photorealistic simulation, which is a key factor in zero-shot sim-to-real transfer. 

Now that we know Isaac Sim does not use URDF, lets see how can we convert and import an URDF file to Isaac Sim. 

Nvidia provides an extension to [import URDF to Isaac Sim](https://github.com/NVIDIA-Omniverse/urdf-importer-extension), which they have open-sourced. 

#### Steps to Import URDF 

- To access this Extension, go to the top menu bar and click Isaac Utils > Workflows > URDF Importer. 
![Step 1 - Access URDF Importer](assets/images/isaac_img_init.png)

- Configure the import settings. 
a. **Fix base link:** If you have mobile robot, un-select this option. For a manipulator this option should be selected
b. **Stage Units per meter:** Setthing this to 1, implies that the 1 unit of Isaac sim will be equal to 1m 
c. **Joint Drive Type:** Can be position or velocity based on your project requirements 
d. **Joint Drive Strength and Joint Position Drive Damping:** Set them to **10000000.0** and **100000.0** respectively. These values are in Isaac units, and emperically we found that, the robot joint doesn't move as they should, if the values are not specified. 
e. We do not select the self-collision. Though it sound like a un-intutive choice, but the importer does esstablishes the collision properties without selecting the option. Note, selecting the option doesn't hurt the process. 

 ![Step - 2 Define Import Properties](assets/images/isaac_img_import_settings.png)

- Click the **Import** button to add the robot to the stage, and we see our robot in the simulation.

![Step - 3: Import the URDF](assets/images/isaac_img_import.png)

- Isaac Sim by default does not create a ground plane, thus we need to do that. 

![Step - 4: Create Ground Plane)(assets/images/isaac_img_ground_plane.png)

- Now that we have created the ground plane, we will verify the collision properties of our imported robot.

[Step - 5: Verify the Collision Properties](assets/images/isaac_img_colliders.png)

[Step - 5: Verify the Collision Properties](assets/images/isaac_img_collision_vis.png)


Voila, we have successfully imported our URDF to Isaac Sim! Though, the import plugin saves the USD file (check Output Directory option while importing), but that is in **.usd** format which is a file binary format, which obiously can't be read by humans. Thus we will go ahead and save it in **.usda** format. USDA is essentially an ASCII format of USD file that is encoded as UTF-8. 

[Step - 6: Saving as USDA](assets/images/isaac_img_save_as.png)
[Step - 6: Saving as USDA](assets/images/isaac_img_save_as_usda.png)




A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.


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
