# Registration Techniques in Robotics
### Introduction
Absolute orientation is a fundamental and important task in robotics. It seeks to determine the translational, rotational, and uniform scalar correspondence between two different Cartesian coordinate systems given a set of corresponding point pairs. Dr. Berthold K.P. Horn proposed a closed-form solution of absolute orientation which is often a least-squares problem. Horn validated that his solution holds good for two of the most common representations of rotation: unit quaternions and orthonormal matrices.


In robotics, there are typically several sensors in a scene and it is necessary to register all of them with respect to each other and to the robot. It is quite common to then register these components to a chosen world frame, which can just be the frame of one of these components. Horn's method is frequently used to estimate the rotation (R), translation (t), and scale(s) that registers one frame to another.

### Horn's Method
The advantage of Horn's closed-form solution is that it provides one in a single step with the best possible transformation, given at least 3 pairs of matching points in the two coordinate systems. Furthermore, one does not need a good initial guess, as one does when an iterative method is used. The underlying mathematics of Horn's method is elucidated below.

Let's call the two coordinate systems that we wish to register "left" and "right". Let <a href="https://www.codecogs.com/eqnedit.php?latex=$x_{l,&space;i}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_{l,&space;i}$" title="$x_{l, i}$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$x_{r,&space;i}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_{r,&space;i}$" title="$x_{r, i}$" /></a> be the coordinates of point i in the left-hand and right-hand coordinate systems respectively. The objective of Horn's method is to find the R, t, and s that transforms a point x in the left coordinate system to an equivalent point x' in the right coordinate system according to the equation: 


<a href="https://www.codecogs.com/eqnedit.php?latex=x^{\prime}=s&space;R&space;x&plus;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{\prime}=s&space;R&space;x&plus;t" title="x^{\prime}=s R x+t" /></a>

Usually, the data are not perfect and we will not be able to find the correspondence between the two points perfectly. The residual error is expressed as:

<a href="https://www.codecogs.com/eqnedit.php?latex=e_{i}=x_{r,&space;i}-s&space;R&space;x_{l,&space;i}-t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{i}=x_{r,&space;i}-s&space;R&space;x_{l,&space;i}-t" title="e_{i}=x_{r, i}-s R x_{l, i}-t" /></a>

Horn's method minimizes the sum of squares of these errors:

<a href="https://www.codecogs.com/eqnedit.php?latex=(t,&space;s,&space;R)=\underset{t,&space;s,&space;R}{\operatorname{argmin}}&space;\sum_{i=1}^{n}\left\|e_{i}\right\|^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(t,&space;s,&space;R)=\underset{t,&space;s,&space;R}{\operatorname{argmin}}&space;\sum_{i=1}^{n}\left\|e_{i}\right\|^{2}" title="(t, s, R)=\underset{t, s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2}" /></a> (1)

Horn's paper [2] offers mathematical rigor into the solution but the most tractable approach [4] to attain the closed-form solution is explained below:

1. Compute the centroids of <a href="https://www.codecogs.com/eqnedit.php?latex=$x_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_l$" title="$x_l$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$x_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_r$" title="$x_r$" /></a> as: 


<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\bar{x}_{l}=\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_{l,&space;i}\quad&space;;&space;\quad&space;\bar{x}_{r}=\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_{r,&space;i}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\bar{x}_{l}=\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_{l,&space;i}\quad&space;;&space;\quad&space;\bar{x}_{r}=\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_{r,&space;i}&space;$$" title="$$ \bar{x}_{l}=\frac{1}{n} \sum_{i=1}^{n} x_{l, i}\quad ; \quad \bar{x}_{r}=\frac{1}{n} \sum_{i=1}^{n} x_{r, i} $$" /></a>

2. Shift points in both coordinate systems so that they are defined with respect to the centroids:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$x_{l,&space;i}^{\prime}=x_{l,&space;i}-\bar{x}_{l}\quad&space;;&space;\quad&space;x_{r,&space;i}^{\prime}=x_{r,&space;i}-\bar{x}_{r}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$x_{l,&space;i}^{\prime}=x_{l,&space;i}-\bar{x}_{l}\quad&space;;&space;\quad&space;x_{r,&space;i}^{\prime}=x_{r,&space;i}-\bar{x}_{r}$$" title="$$x_{l, i}^{\prime}=x_{l, i}-\bar{x}_{l}\quad ; \quad x_{r, i}^{\prime}=x_{r, i}-\bar{x}_{r}$$" /></a>

3. The residual error term with reference to the centroid of the shifted points is now:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$e_{i}=x_{r,&space;i}^{\prime}-s&space;R&space;x_{l,&space;i}^{\prime}-t^{\prime}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$e_{i}=x_{r,&space;i}^{\prime}-s&space;R&space;x_{l,&space;i}^{\prime}-t^{\prime}$$" title="$$e_{i}=x_{r, i}^{\prime}-s R x_{l, i}^{\prime}-t^{\prime}$$" /></a>

The translational offset between the centroids of the left and right coordinate systems is:
<a href="https://www.codecogs.com/eqnedit.php?latex=$$t^{\prime}=t-\bar{x}_{r}&plus;s&space;R&space;\bar{x}_{l}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$t^{\prime}=t-\bar{x}_{r}&plus;s&space;R&space;\bar{x}_{l}$$" title="$$t^{\prime}=t-\bar{x}_{r}+s R \bar{x}_{l}$$" /></a>

In [2], Horn proves that the squared error is minimized when <a href="https://www.codecogs.com/eqnedit.php?latex=$t'&space;=&space;\mathbf{0}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$t'&space;=&space;\mathbf{0}$" title="$t' = \mathbf{0}$" /></a>. So the equation that yields <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> is:

<a href="https://www.codecogs.com/eqnedit.php?latex=t=\bar{x}_{r}-s&space;R&space;\bar{x}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=\bar{x}_{r}-s&space;R&space;\bar{x}_{l}" title="t=\bar{x}_{r}-s R \bar{x}_{l}" /></a> (2)

Thus, the translation is just the difference of the right centroid and the scaled and rotated left centroid.
Once we have solved for <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> we get back to this equations to solve for <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>. Since the error is minimum when <a href="https://www.codecogs.com/eqnedit.php?latex=$t'&space;=&space;\mathbf{0}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$t'&space;=&space;\mathbf{0}$" title="$t' = \mathbf{0}$" /></a> , the error term is now:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;e_{i}=&&space;x_{r,&space;i}^{\prime}-s&space;R&space;x_{l,&space;i}^{\prime}\\&space;e_{i}=&&space;\frac{1}{\sqrt{s}}&space;x_{r,&space;i}^{\prime}-\sqrt{s}&space;R&space;x_{l,&space;i}^{\prime}&space;\quad&space;\text{(rearranging&space;the&space;terms)}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;e_{i}=&&space;x_{r,&space;i}^{\prime}-s&space;R&space;x_{l,&space;i}^{\prime}\\&space;e_{i}=&&space;\frac{1}{\sqrt{s}}&space;x_{r,&space;i}^{\prime}-\sqrt{s}&space;R&space;x_{l,&space;i}^{\prime}&space;\quad&space;\text{(rearranging&space;the&space;terms)}&space;\end{align*}" title="\begin{align*} e_{i}=& x_{r, i}^{\prime}-s R x_{l, i}^{\prime}\\ e_{i}=& \frac{1}{\sqrt{s}} x_{r, i}^{\prime}-\sqrt{s} R x_{l, i}^{\prime} \quad \text{(rearranging the terms)} \end{align*}" /></a>

4. Equation (1) now reduces to:

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;(s,&space;R)=&&space;\underset{s,&space;R}{\operatorname{argmin}}&space;\sum_{i=1}^{n}\left\|e_{i}\right\|^{2}&space;\\&space;=&&space;\underset{s,&space;R}{\operatorname{argmin}}&space;\frac{1}{s}&space;\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}&plus;s&space;\sum_{i=1}^{n}\left\|x_{l,&space;i}\right\|^{2}&space;-2&space;\sum_{i=1}^{n}&space;x_{r,&space;i}^{\prime}&space;\cdot\left(R&space;x_{l,&space;i}^{\prime}\right)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;(s,&space;R)=&&space;\underset{s,&space;R}{\operatorname{argmin}}&space;\sum_{i=1}^{n}\left\|e_{i}\right\|^{2}&space;\\&space;=&&space;\underset{s,&space;R}{\operatorname{argmin}}&space;\frac{1}{s}&space;\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}&plus;s&space;\sum_{i=1}^{n}\left\|x_{l,&space;i}\right\|^{2}&space;-2&space;\sum_{i=1}^{n}&space;x_{r,&space;i}^{\prime}&space;\cdot\left(R&space;x_{l,&space;i}^{\prime}\right)&space;\end{align*}" title="\begin{align*} (s, R)=& \underset{s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2} \\ =& \underset{s, R}{\operatorname{argmin}} \frac{1}{s} \sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}+s \sum_{i=1}^{n}\left\|x_{l, i}\right\|^{2} -2 \sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right) \end{align*}" /></a> (3)

This can be written in the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=S_{r}-2&space;s&space;D&plus;s^{2}&space;S_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_{r}-2&space;s&space;D&plus;s^{2}&space;S_{l}" title="S_{r}-2 s D+s^{2} S_{l}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=$S_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$S_r$" title="$S_r$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$S_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$S_l$" title="$S_l$" /></a> are the sums of squares of the measurement vectors(relative to their centroids), and <a href="https://www.codecogs.com/eqnedit.php?latex=$D$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$D$" title="$D$" /></a> s the sum of the dot products of the corresponding coordinates in the right system with the rotated coordinates in the left system. Completing the square in <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> we get:

<a href="https://www.codecogs.com/eqnedit.php?latex=(s&space;\sqrt{S_{l}}-D&space;/&space;\sqrt{S_{l}})^{2}&plus;\left(S_{r}&space;S_{l}-D^{2}\right)&space;/&space;S_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(s&space;\sqrt{S_{l}}-D&space;/&space;\sqrt{S_{l}})^{2}&plus;\left(S_{r}&space;S_{l}-D^{2}\right)&space;/&space;S_{l}" title="(s \sqrt{S_{l}}-D / \sqrt{S_{l}})^{2}+\left(S_{r} S_{l}-D^{2}\right) / S_{l}" /></a>

This is minimized with respect to scale <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> when the first term is zero or <a href="https://www.codecogs.com/eqnedit.php?latex=$s=D&space;/&space;S_{l}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$s=D&space;/&space;S_{l}$" title="$s=D / S_{l}$" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=s=\sqrt{\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}&space;/&space;\sum_{i=1}^{n}\left\|x_{l,&space;i}^{\prime}\right\|^{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s=\sqrt{\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}&space;/&space;\sum_{i=1}^{n}\left\|x_{l,&space;i}^{\prime}\right\|^{2}}" title="s=\sqrt{\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2} / \sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}}" /></a>

5. Now that <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> can be solved for, <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> is obtained by substituting the expression for <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> in the Equation (3): 

<a href="https://www.codecogs.com/eqnedit.php?latex=R=\underset{R}{\operatorname{argmin}}&space;2\left(\sqrt{\left(\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}\right)\left(\sum_{i=1}^{n}\left\|x_{l,&space;i}^{\prime}\right\|^{2}\right)}-\sum_{i=1}^{n}&space;x_{r,&space;i}^{\prime}&space;\cdot\left(R&space;x_{l,&space;i}^{\prime}\right)\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R=\underset{R}{\operatorname{argmin}}&space;2\left(\sqrt{\left(\sum_{i=1}^{n}\left\|x_{r,&space;i}^{\prime}\right\|^{2}\right)\left(\sum_{i=1}^{n}\left\|x_{l,&space;i}^{\prime}\right\|^{2}\right)}-\sum_{i=1}^{n}&space;x_{r,&space;i}^{\prime}&space;\cdot\left(R&space;x_{l,&space;i}^{\prime}\right)\right)" title="R=\underset{R}{\operatorname{argmin}} 2\left(\sqrt{\left(\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}\right)\left(\sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}\right)}-\sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right)\right)" /></a>

6. Once we have <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> is solved for using Equation (2).

Alternatively in [2], when there are 3 matching points whose correspondence in both systems have to be obtained, Horn constructs triads in the left and right coordinate systems. Let <a href="https://www.codecogs.com/eqnedit.php?latex=$x_l,&space;y_l,&space;z_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_l,&space;y_l,&space;z_l$" title="$x_l, y_l, z_l$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$x_r,&space;y_r,&space;z_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_r,&space;y_r,&space;z_r$" title="$x_r, y_r, z_r$" /></a> be the unit vectors in the left and right coordinate systems respectively. In the left coordinate system, the <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> axis is defined as the line that joins points 1 and 2, and the <a href="https://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y" title="y" /></a> axis is a line perpendicular to this axis, and the <a href="https://www.codecogs.com/eqnedit.php?latex=z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z" title="z" /></a> axis is obtained by the cross-product:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{z}_{l}=\hat{x}_{l}&space;\times&space;\hat{y}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{z}_{l}=\hat{x}_{l}&space;\times&space;\hat{y}_{l}" title="\dot{z}_{l}=\hat{x}_{l} \times \hat{y}_{l}" /></a>

Similarly, a triad is constructed in the right coordinate system. The rotation matrix <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> that we are looking for transforms <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_{l}" title="\hat{x}_{l}" /></a> into <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_{l}" title="\hat{x}_{l}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_{l}" title="\hat{y}_{l}" /></a> into <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}_{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}_{r}" title="\hat{y}_{r}" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{z}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{z}_{l}" title="\hat{z}_{l}" /></a> into <a href="https://www.codecogs.com/eqnedit.php?latex=\bar{z}_{r}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar{z}_{r}" title="\bar{z}_{r}" /></a>.

We adjoin column vectors to form the matrices <a href="https://www.codecogs.com/eqnedit.php?latex=$M_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_l$" title="$M_l$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$M_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_r$" title="$M_r$" /></a> as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=M_l=\left|\hat{x}_l&space;\hat{y}_{l}&space;\hat{z}_{l}\right|,&space;\quad&space;M_{r}=\left|\hat{x}_{r}&space;\hat{y}_{r}&space;\hat{z}_r\right|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M_l=\left|\hat{x}_l&space;\hat{y}_{l}&space;\hat{z}_{l}\right|,&space;\quad&space;M_{r}=\left|\hat{x}_{r}&space;\hat{y}_{r}&space;\hat{z}_r\right|" title="M_l=\left|\hat{x}_l \hat{y}_{l} \hat{z}_{l}\right|, \quad M_{r}=\left|\hat{x}_{r} \hat{y}_{r} \hat{z}_r\right|" /></a>

Given a vector <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathbf{r}_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathbf{r}_l$" title="$\mathbf{r}_l$" /></a> in the left coordinate system, <a href="https://www.codecogs.com/eqnedit.php?latex=$M_l^T\mathbf{r_l}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_l^T\mathbf{r_l}$" title="$M_l^T\mathbf{r_l}$" /></a> gives us the components of the vector <a href="https://www.codecogs.com/eqnedit.php?latex=$r_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$r_l$" title="$r_l$" /></a> along the axes of the constructed triad. Multiplication by <a href="https://www.codecogs.com/eqnedit.php?latex=$M_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_r$" title="$M_r$" /></a> then maps these into the right coordinate system, so

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{r}_{r}=M_{r}&space;M_{l}^{T}&space;\mathbf{r}_{l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{r}_{r}=M_{r}&space;M_{l}^{T}&space;\mathbf{r}_{l}" title="\mathbf{r}_{r}=M_{r} M_{l}^{T} \mathbf{r}_{l}" /></a>

The rotation matrix <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> is given by:

<a href="https://www.codecogs.com/eqnedit.php?latex=R=M_{r}&space;M_{l}^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R=M_{r}&space;M_{l}^{T}" title="R=M_{r} M_{l}^{T}" /></a>

The result is orthonormal since <a href="https://www.codecogs.com/eqnedit.php?latex=$M_l$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_l$" title="$M_l$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$M_r$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$M_r$" title="$M_r$" /></a> are orthonormal by construction. Horn admits that while this particular approach to find <a href="https://www.codecogs.com/eqnedit.php?latex=$R$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R$" title="$R$" /></a> by constructing triads gives a closed-form solution for finding <a href="https://www.codecogs.com/eqnedit.php?latex=$R$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$R$" title="$R$" /></a> given three points, it cannot be extended to deal with more than three points.

Through the steps listed above, Horn's method presents a closed-form solution to the absolute orientation problem.

### Camera-Robot Registration Using Horn's Method
A popular application of Horn's method is to perform camera-robot registration, i.e., to find the <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> that relates the camera frame and the robot frame. An elegant technique that uses Horn's method to register a stereo camera and the robot is explained in this section. This procedure is based on [5] in which a stereo camera overlooks the workspace of the da Vinci Research Kit by Intuitive Surgical. The code written by Nicolas Zevallos et al. of [5] can be found on the GitHub repository of Carnegie Mellon University's Biorobotics Lab, linked [here](https://github.com/gnastacast/dvrk_vision/blob/master/src/dvrk_vision/dvrk_registration.py) 
Team A of the MRSD class of 2021, performed camera-robot registration using a slight modification of their technique. The procedure followed by Nicolas Zevallos et al. is explained below:
* The robot is fitted with a colored bead on its end effector; the colored bead can be easily segmented from its background by hue, saturation, and value. It is preferred that the bead be bright in color so as to increase its visibility to the stereo camera.
* The robot is moved to a fixed set of five or six points. The points are chosen to cover a significant amount of the robot's workspace and also stay within the field of view in the camera. 
* The robot is moved to the specified location. The left and right images of the stereo are processed to find the centroid of the colored bead fitted to the robot. This is repeated for as many frames as are received over ROS (which is the framework used in [5]) in one second. The centroid is averaged over all the received frames for a more accurate spatial estimate of the bead.
* The pixel disparity is given as input to a stereo-camera model that ROS provides to calculate a 3D point in the camera-frame.
* Following this, we have six points in both the camera-frame and the robot-frame. The coordinates of the point in the robot-frame are known using the kinematics of the robot.
* Since we now have corresponding points in the camera-frame and the robot-frame, Horn's method is used to calculate the transformation <a href="https://www.codecogs.com/eqnedit.php?latex=$T_r^c$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$T_r^c$" title="$T_r^c$" /></a> between the robot-frame and the camera-frame.

### Iterative Closest Point Algorithm

Iterative Closest Point (ICP) is an algorithm to register two sets of point clouds iteratively. In each iteration, the algorithm selects the closest points as correspondences and calculates the transformation, i.e., rotation and translation (<a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R" title="R" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>). Although there are a number of techniques to calculate the transformation at each iteration, Horn's method remains a popular choice to do so. 
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
