title: Registration Techniques in Robotics
Absolute orientation is a fundamental and important task in robotics. It seeks to determine the translational, rotational, and uniform scalar correspondence between two different Cartesian coordinate systems given a set of corresponding point pairs. Dr. Berthold K.P. Horn proposed a closed-form solution of absolute orientation which is often a least-squares problem. Horn validated that his solution holds good for two of the most common representations of rotation: unit quaternions and orthonormal matrices.

In robotics, there are typically several sensors in a scene and it is necessary to register all of them with respect to each other and to the robot. It is quite common to then register these components to a chosen world frame, which can just be the frame of one of these components. Horn's method is frequently used to estimate the rotation (R), translation (t), and scale(s) that registers one frame to another.

### Horn's Method
The advantage of Horn's closed-form solution is that it provides one in a single step with the best possible transformation, given at least 3 pairs of matching points in the two coordinate systems. Furthermore, one does not need a good initial guess, as one does when an iterative method is used. The underlying mathematics of Horn's method is elucidated below.

Let's call the two coordinate systems that we wish to register "left" and "right". Let \\[ x_{l, i} \\] and \\[x_{r, i} \\] be the coordinates of point i in the left-hand and right-hand coordinate systems respectively. The objective of Horn's method is to find the R, t, and s that transforms a point x in the left coordinate system to an equivalent point x' in the right coordinate system according to the equation: 


\\[ x^{\prime}=s R x+t \\]

Usually, the data are not perfect and we will not be able to find the correspondence between the two points perfectly. The residual error is expressed as:

\\[ e_{i}=x_{r, i}-s R x_{l, i}-t \\]

Horn's method minimizes the sum of squares of these errors:

\\[ (t, s, R)=\underset{t, s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2} \\]

Horn's paper [2] offers mathematical rigor into the solution but the most tractable approach [4] to attain the closed-form solution is explained below:

1. Compute the centroids of \\[ $x_l \\] and \\[ x_r \\] as: 

\\[ \bar{x}_{l}=\frac{1}{n} \sum_{i=1}^{n} x_{l, i}\quad ; \quad \bar{x}_{r}=\frac{1}{n} \sum_{i=1}^{n} x_{r, i} \\]

2. Shift points in both coordinate systems so that they are defined with respect to the centroids:

\\[ x_{l, i}^{\prime}=x_{l, i}-\bar{x}_{l}\quad ; \quad x_{r, i}^{\prime}=x_{r, i}-\bar{x}_{r} \\]

3. The residual error term with reference to the centroid of the shifted points is now:

\\[ e_{i}=x_{r, i}^{\prime}-s R x_{l, i}^{\prime}-t^{\prime} \\]
The translational offset between the centroids of the left and right coordinate systems is:
\\[ t^{\prime}=t-\bar{x}_{r}+s R \bar{x}_{l} \\]

In [2], Horn proves that the squared error is minimized when \\[ t' = \mathbf{0} \\]. So the equation that yields \\[ t \\] is:

\\[ t=\bar{x}_{r}-s R \bar{x}_{l} \\]

Thus, the translation is just the difference of the right centroid and the scaled and rotated left centroid.
Once we have solved for \\[ R \\] and \\[ s \\] we get back to this equations to solve for \\[ t \\]. Since the error is minimum when \\[ t' = \mathbf{0} \\], the error term is now:

\\[ \begin{align*} e_{i}=& x_{r, i}^{\prime}-s R x_{l, i}^{\prime}\\ e_{i}=& \frac{1}{\sqrt{s}} x_{r, i}^{\prime}-\sqrt{s} R x_{l, i}^{\prime} \quad \text{(rearranging the terms)} \end{align*} \\]

4. Equation (1) now reduces to:

\\[ \begin{align*} (s, R)=& \underset{s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2} \\ =& \underset{s, R}{\operatorname{argmin}} \frac{1}{s} \sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}+s \sum_{i=1}^{n}\left\|x_{l, i}\right\|^{2} -2 \sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right) \end{align*} \\]

This can be written in the form:

\\[ S_{r}-2 s D+s^{2} S_{l} \\]

where \\[ S_r \\] and \\[ S_l \\] are the sums of squares of the measurement vectors(relative to their centroids), and \\[ D \\] s the sum of the dot products of the corresponding coordinates in the right system with the rotated coordinates in the left system. Completing the square in \\[ s \\] we get:

\\[ (s \sqrt{S_{l}}-D / \sqrt{S_{l}})^{2}+\left(S_{r} S_{l}-D^{2}\right) / S_{l} \\]

This is minimized with respect to scale \\[ s \\] when the first term is zero or \\[ s=D / S_{l} \\]

\\[ s=\sqrt{\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2} / \sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}} \\]

5. Now that \\[ s \\] can be solved for, \\[ R \\] is obtained by substituting the expression for \[[ s \\] in the Equation (3): 

\\[ R=\underset{R}{\operatorname{argmin}} 2\left(\sqrt{\left(\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}\right)\left(\sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}\right)}-\sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right)\right) \\]

6. Once we have \\[ s \\] and \\[ R \\], <\\[ t \\] is solved for using Equation (2).

Alternatively in [2], when there are 3 matching points whose correspondence in both systems have to be obtained, Horn constructs triads in the left and right coordinate systems. Let \\[ x_l, y_l, z_l \\] and \\[ x_r, y_r, z_r \\] be the unit vectors in the left and right coordinate systems respectively. In the left coordinate system, the \\[ x \\] axis is defined as the line that joins points 1 and 2, and the \\[ y \\] axis is a line perpendicular to this axis, and the \\[ z \\] axis is obtained by the cross-product:

\\[ \dot{z}_{l}=\hat{x}_{l} \times \hat{y}_{l} \\]

Similarly, a triad is constructed in the right coordinate system. The rotation matrix \\[ R \\] that we are looking for transforms \\[ \hat{x}_{l} \\] into \\[ \hat{x}_{l} \\], \\[ \hat{y}_{l} \\] into \\[ \hat{y}_{r} \\], and \\[ \hat{z}_{l}\\] into \\[ \bar{z}_{r} \\].

We adjoin column vectors to form the matrices \\[ M_l \\] and \\[ M_r \\] as follows:

\\[ M_l=\left|\hat{x}_l \hat{y}_{l} \hat{z}_{l}\right|, \quad M_{r}=\left|\hat{x}_{r} \hat{y}_{r} \hat{z}_r\right| \\]

Given a vector \\[ \mathbf{r}_l \\] in the left coordinate system, \\[ M_l^T\mathbf{r_l} \\] gives us the components of the vector \\[ r_l \\] along the axes of the constructed triad. Multiplication by \\[ M_r \\] then maps these into the right coordinate system, so

\\[ \mathbf{r}_{r}=M_{r} M_{l}^{T} \mathbf{r}_{l} \\]

The rotation matrix \\[ R \\] is given by:

\\[ R=M_{r} M_{l}^{T} \\]

The result is orthonormal since \\[ M_l \\] and \\[ M_r \\] are orthonormal by construction. Horn admits that while this particular approach to find \\[ R \\] by constructing triads gives a closed-form solution for finding \\[ R \\] given three points, it cannot be extended to deal with more than three points.

Through the steps listed above, Horn's method presents a closed-form solution to the absolute orientation problem.

### Camera-Robot Registration Using Horn's Method
A popular application of Horn's method is to perform camera-robot registration, i.e., to find the \\[ R \\] and \\[ t \\] that relates the camera frame and the robot frame. An elegant technique that uses Horn's method to register a stereo camera and the robot is explained in this section. This procedure is based on [5] in which a stereo camera overlooks the workspace of the da Vinci Research Kit by Intuitive Surgical. The code written by Nicolas Zevallos et al. of [5] can be found on the GitHub repository of Carnegie Mellon University's Biorobotics Lab, linked [here](https://github.com/gnastacast/dvrk_vision/blob/master/src/dvrk_vision/dvrk_registration.py) 
Team A of the MRSD class of 2021, performed camera-robot registration using a slight modification of their technique. The procedure followed by Nicolas Zevallos et al. is explained below:
* The robot is fitted with a colored bead on its end effector; the colored bead can be easily segmented from its background by hue, saturation, and value. It is preferred that the bead be bright in color so as to increase its visibility to the stereo camera.
* The robot is moved to a fixed set of five or six points. The points are chosen to cover a significant amount of the robot's workspace and also stay within the field of view in the camera. 
* The robot is moved to the specified location. The left and right images of the stereo are processed to find the centroid of the colored bead fitted to the robot. This is repeated for as many frames as are received over ROS (which is the framework used in [5]) in one second. The centroid is averaged over all the received frames for a more accurate spatial estimate of the bead.
* The pixel disparity is given as input to a stereo-camera model that ROS provides to calculate a 3D point in the camera-frame.
* Following this, we have six points in both the camera-frame and the robot-frame. The coordinates of the point in the robot-frame are known using the kinematics of the robot.
* Since we now have corresponding points in the camera-frame and the robot-frame, Horn's method is used to calculate the transformation \\[ T_r^c \\] between the robot-frame and the camera-frame.

### Iterative Closest Point Algorithm

Iterative Closest Point (ICP) is an algorithm to register two sets of point clouds iteratively. In each iteration, the algorithm selects the closest points as correspondences and calculates the transformation, i.e., rotation and translation (\\[ R \\] and \\[ t \\]. Although there are a number of techniques to calculate the transformation at each iteration, Horn's method remains a popular choice to do so.

The main steps of ICP [6] are listed below:
1. For each point in the source point cloud, match the closest point in the reference point cloud
2. Calculate a transformation (a combination of rotation, translation, and scaling) which will best align each source point to its match found in the previous step. This is done by using Horn's method or a similar approach.
3. Transform the source points using the obtained transformation
4. Check if the two point clouds align within a desired threshold. If not, iterate through the algorithm again

### Summary
Horn's method is a useful tool to estimate the rotation, translation, and scale that exist between two coordinate systems. It is widely used in robotics to register coordinate frames of different hardware components that comprise a system as well as two sets of point clouds. Thus, Horn's method provides a closed-form solution to the absolute orientation problem.


### References
Micheals, Ross & Boult, Terrance. (1999). A New Closed Form Approach to the Absolute Orientation Problem.

Berthold K. P. Horn, "Closed-form solution of absolute orientation using unit quaternions," in Journal of the Optical Society of America A, vol. 4, no. 2, pp. 629-642, 1987.

Berthold K. P. Horn, "Closed-form solution of absolute orientation using orthonormal matrices," in Journal of the Optical Society of America A, vol. 5, pp. 1127-1135, 1988.

Ivan Stojmenović, Handbook of Sensor Networks: Algorithms and Architectures, First published:2 September 2005, Print ISBN:9780471684725 | Online ISBN:9780471744146 | DOI:10.1002/047174414X

Zevallos, Nicolas & Rangaprasad, Arun Srivatsan & Salman, Hadi & Li, Lu & Saxena, Saumya & Xu, Mengyun & Qian, Jianing & Patath, Kartik & Choset, Howie. (2018). A Real-time Augmented Reality Surgical System for Overlaying Stiffness Information. 10.13140/RG.2.2.17472.64005. 

[ICP Wikipedia](https://en.wikipedia.org/wiki/Iterative_closest_point)

[An Overview of Robot-Sensor Calibration Methods for
Evaluation of Perception Systems](http://faculty.cooper.edu/mili/Calibration/index.html)
