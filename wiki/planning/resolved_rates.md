---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-12-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Resolved-rate Motion Control
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
For several real-world applications, it may be necessary for the robot end-effector to move in a straight line in Cartesian space. However, when one or more joints of a robot arm are moved, the end-effector traces an arc and not a straight line. Resolved-rate[1] is a Jacobian-based control scheme for moving the end-effector of a serial-link manipulator at a specified Cartesian velocity $v$ without having to compute the inverse kinematics at each time step. Instead, the inverse of the Jacobian matrix alone is recomputed at each time step to account for the updated joint angles. The displacement of each joint is given by the product of the current joint velocity and the time step, which is then added to the current joint configuration to update the pose of the robot. The advantage of this incremental approach of updating joint angles is that the robot moves smoothly between waypoints as opposed to exhibiting jerky movements that arise from frequent recomputations of the inverse kinematics. This article presents the mathematical formulation of the resolved-rate motion control scheme and explains its usage in a motion compensation algorithm.

## Derivation
The forward kinematics of a serial-link manipulator provides
a non-linear surjective mapping between the joint space
and Cartesian task space[2]. For an $n$-degree of freedom (DoF) manipulator with $n$ joints, let $\boldsymbol{q}(t) \in \mathbb{R}^{n}$ be the joint coordinates of the robot and $r \in \mathbb{R}^{m}$ be the parameterization of the end-effector pose. The relationship between the robot's joint space and task space is given by:
\begin{equation}
    \boldsymbol{r}(t)=f(\boldsymbol{q}(t))
\end{equation}
In most real-world applications, the robot has a task space $\mathcal{T} \in \operatorname{SE}(3)$ and therefore $m = 6$. The Jacobian matrix provides the relation between joint velocities $\dot{q}$ and end-effector velocity $\dot{v}$ of a robotic manipulator. A redundant
manipulator has a joint space dimension that exceeds the workspace dimension, i.e. $n > 6$. Taking the derivative of (1) with respect to time:
\begin{equation}
    \nu(t)=J(q(t)) \dot{q}(t)
\end{equation}
where $J(q(t))=\left.\frac{\partial f(q)}{\partial q}\right|_{q=q_{0}} \in \mathbb{R}^{6 \times n}$ is the manipulator
Jacobian for the robot at configuration $q_0$. Resolved-rate
motion control is an algorithm which maps a Cartesian end-effector
velocity $\dot{v}$ to the robot’s joint velocity $\dot{q}$. By rearranging (2), the required joint velocities are:
\begin{equation}
    \dot{\boldsymbol{q}}=\boldsymbol{J}(\boldsymbol{q})^{-1} \nu
\end{equation}
It must be noted that (3) can be directly solved only is $J(q)$ is square and non-singular, which is when the robot has 6 DoF. Since most modern robots are several redundant DoFs, it is more common to use the
Moore-Penrose pseudoinverse:
\begin{equation}
    \dot{\boldsymbol{q}}=\boldsymbol{J}(\boldsymbol{q})^{+}v
\end{equation}
where the $(\cdot)^{+}$ denotes the pseudoinverse operation.


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
