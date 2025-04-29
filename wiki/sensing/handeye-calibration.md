<!--This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).-->

**This is a tutorial for estimating the frame transformation between an image frame and an operating frame by using a third reference frame. An application is to estimate the transformation between pixel coordinates to end effector coordinates using an Aruco marker pose as reference. ROS has been chosen as the framework for this process due to its functionality that facilitates synchronized parallel communication. While most existing packages use ROS1, this tutorial uses ROS2. The entire workflow, from scene setup, data capture, computation and integration has been covered in this tutorial.**

## Hand-Eye Calibration
<!--Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.-->

### Different Frames
	1. Image Frame (Pixel Space)
	2. Target Frame (Operation Space eg, base frame of manipulator or end-effector frame)
	3. World Frame (Global Frame: usually set as the operating frame)

### The Algorithm

This package uses the method introduced by Lenz and Tsai in 1989. This is a data-driven method and it was observed that around thirty images are required for this method to work reliably.

[Documentation](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99)
[Original Research Paper](https://ieeexplore.ieee.org/document/34770)
[GitHub Package](https://github.com/SNAAK-CMU/handeye_calibration_ros2)

### This Setup

For this tutorial, ROS2 will be used as the environment for its functionality that makes it easy to define frames and transformations using a transformation tree

	1. Image Frame: Realsense Camera Frame (ROS TF Frame: "camera_color_optical_frame")
	2. Target Frame: Base Frame of manipulator (ROS TF Frames : 'base_link: "panda_link_0"; ee_link: "panda_hand")
	3. World Frame: Aruco marker pose (From Aruco marker detection)

### This Process

The GitHub package has detailed instructions on installation and setup. The parameters in the file `handeye_realsense/config.yaml` need to be rewritten with the ROS2 topic and frame names of your system. 

## Summary:
	1. Keep in mind that the manipulator's poses must be as different as possible when sampling data in order to get a generalized result. If possible, put your manipulator in guide mode and move to the poses yourself, as this repository does not include a random pose generator. 
    2. Not moving the Aruco marker's position yeilds better results. 
    3. Configuring a random pose generator would require defining your workspace in a planning framework such as MoveIt! and generate random, collision free poses where the aruco pose is in the field of view of the camera. 
    4. Ensure that the `child frame` specified in the config is the frame on which images are published. If not, set the child frame as the camera frame and chain together an intrinsic transformation to the image frame with the extrinsic transform from the target frame you will get from this process. This process has been described in detail on the README of the repository
    

<!--### Basic syntax
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
Use this space to reinforce key points and to suggest next steps for your readers.-->

## See Also:
[This Wiki entry serves as an introduction to calibration](camera-calibration.md) 

## Further Reading
[Original GitHub Repository](https://github.com/shengyangzhuang/handeye_calibration_ros2)

## References
<!--- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.-->

	1. https://github.com/shengyangzhuang/handeye_calibration_ros2
	2. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7
	904a801b99
	3. https://docs.opencv.org/3.4/d0/de3/citelist.html#CITEREF_Tsai89
	4. R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D
	robotics hand/eye calibration," in IEEE Transactions on Robotics and Automation,
	vol. 5, no. 3, pp. 345-358, June 1989, doi: 10.1109/70.34770.