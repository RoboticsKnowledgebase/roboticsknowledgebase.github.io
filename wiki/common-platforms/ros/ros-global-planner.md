---
title: Global Planner in ROS
---
There are 3 global planners that adhere to `nav_core::BaseGlobalPlanner` interface: `global_planner`, `navfn` and `carrot_planner`. The `nav_core::BaseGlobalPlanner` provides an interface for global used in navigation.

## carrot_planner

The  `carrot_planner` is the simplest global planner, which makes the robot to get as close to a user-specified goal point as possible. The planner checks whether. the user-specified goal is an obstacle. If it is, it moves back along the vector between the robot and the goal. Otherwise, it passes the goal point as a plan to the local planner or controller (internally).

## navfn and global_planner

`navfn` and  `global_planner` are based on the work by Brock and Khatib, 1999[1]. The difference is that `navfn` uses Dijkstra's algorithm, while `global_planner` is computed with more flexible options, including:

1. support A* algorithm
2. toggling quadratic approximation
3. toggling grid path

Therefore, we generally use `global_planner` in our own project. Here, we will talk about some key parameters of it. If you are interested in other parameters, you could the run the rqt dynamic reconfigure program to see them, by

```<shell>
rosrun rqt_reconfigure rqt_reconfigure
```

which will show like in Figure 1.

## First subheading

Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Basic syntax

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
Here is an example MathJax inline rendering \\( 1/x^{2} \\), and here is a block rendering:
\\[ \frac{1}{n^{2}} \\]

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

[1] Brock, O. and Khatib, O. (1999). High-speed navigation using the global dynamic window approach. In Proceedings 1999 IEEE Interna- tional Conference on Robotics and Automation (Cat. No. 99CH36288C), volume 1, pages 341–346. IEEE.

- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
