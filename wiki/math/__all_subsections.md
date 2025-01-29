/wiki/math/gaussian-process-gaussian-mixture-model/
---
date: 2019-12-16
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. You should set the article's title:
title: Guassian Process and Gaussian Mixture Model
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

This document acts as a tutorial on Gaussian Process(GP), Gaussian Mixture Model, Expectation Maximization Algorithm. We recently ran into these approaches in our robotics project that having multiple robots to generate environment models with minimum number of samples. 

You will be able to have a basic understanding on Gaussian Process (what is it and why it is very popular), Gaussian Mixture Model and Expectation Maximization Algorithm after this tutorial.

# Table of Contents
1. [Gausssian Process](#Gausssian-Process)
3. [Gaussian Mixture Model](#Gaussian-Mixture-Model)
4. [Expectation Maximization Algorithm](#Expectation-Maximization-Algorithm)
7. [Summary](#Summary)
8. [References](#References)

## Gausssian Process

### What is GP?
A Gaussian Process(GP) is a probability distribution over functions \\( p(\textbf{f}) \\) , where the functions are defined by a set of random function variables \\(\textbf{f} = \{f_1, f_2, . . . , f_N\}\\). For a GP, any finite linear combination of those function variables has a joint (zero mean) Gaussian distribution. It can be used for nonlinear regression, classification, ranking, preference learning, ordinal regression. In robotics, it can be applied to state estimation, motion planning and in our case environment modeling.

GPs are closely related to other models and can be derived using bayesian kernel machines, linear regression with basis functions, infinite multi-layer perceptron neural networks (it becomes a GP if there are infinitely many hidden units and Gaussian priors on the weights), spline models. A simple connections between different methods can be found in the following figure.
![](assets/gpConnection.png)




<!-- Intuitively speaking, in machine learning, Gaussian is a handy tool for Bayesian inference on real valued variables. -->


Before all the equations, lets have a look at why do we want to use GP?
### Why do we want to use GP?
The major reasons why GP is popular includes:
1. It can handle uncertainty in unknown function by averaging, not minimizing as GP is rooted in probability and bayesian inference.![](assets/uncertainty.png)
Classifier comparison(https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) This property is not commonly shared by other methods. For example, in the above classification method comparison. Neural nets and random forests are confident about the points that are far from the training data.
2. We can incorporate prior knowledge by choosing different kernels
3. GP can learn the kernel and regularization parameters automatically during the learning process. (GP is a non-parametric model, but all data are needed to generate the model. In other words, it need finite but unbounded number of parameters that grows with data.)
4. It can interpolate training data.

### Gaussian Process for Regression
In this section, we will show how to utilize GP for solving regression problems. The key idea is to map the independent variables from low-dimensional space to high-dimensional space. The idea is similar to the kernel function in support vector machine, where the linear-inseparable data in the low-dimensional space can be mapped to a high-dimensional where the data can be separated by a hyper-plane.

When using Gaussian process regression, there is no need to specify the specific form of f(x), such as \\( f(x)=ax^2+bx+c \\). The observations of n training labels \\( y_1, y_2, ..., y_n \\) are treated as points sampled from a multidimensional (n-dimensional) Gaussian distribution.

For a given training examples \\( x_1, x_2, ..., x_n \\) and the corresponding observations \\(y_1, y_2, ..., y_n \\), the target \\( y \\) can be modeled by an implicit function \\(f(x)\\). Since the observations are usually noisy, each observation y is added with a Gaussian noise.

\\[ y = f(x) + N(0, \sigma^2_n) \\]

\\( f(x) \\) is assumed to have a prior \\( f(x) \sim GP(0,K_{xx}) \\), where \\(K_{xx}\\) is the covariance matrix. There are multiple method to formulate the covariance here. You can also use a combination of covariance functions. Since the mean is assumed to be zero, the final result depends largely on the choice of covariance function.

Suppose we have a dataset containing \\( n \\) training examples \\( (x_i, y_i) \\), and we want to predict the value \\( y^* \\) given \\( x^* \\). After adding the unknown value, we get a variable of \\( (n+1) \\) dimension.

$$ \left[\begin{array}{c}
y_{1} \\
\vdots \\
y_{n} \\
f\left(x^{*}\right)
\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{c}
0 \\
0
\end{array}\right],\left[\begin{array}{cc}
K_{x x}+\sigma_{n}^{2} I & K_{x x}^{T} \\
K_{x x}^{2} & k\left(x^{*}, x^{*}\right)
\end{array}\right]\right) $$

where,

\\[K_{x x^{* }}=\left[k\left(x_{1}, x^{* }\right) \ldots k\left(x_{n}, x^{* }\right)\right ] \\]

Hence, the problem can be convert into a conditional probability problem, i.e. solving \\( f(x^* )\|\mathbf{y}(\mathbf{x}) \\).

\\[ f\left(x^{* }\right) \| \mathbf{y}(\mathbf{x}) \sim \mathcal{N}\left(K_{x x^{* }} M^{-1} \mathbf{y}(\mathbf{x}), k\left(x^{* }, x^{* }\right)-K_{x x^{* }} M^{-1} K_{x^{* } x}^{T}\right)\\]

where \\( M=K_{x x}+\sigma_{n}^{2} I \\), hence the expectation of \\( y^* \\) is

\\[ E\left(y^{* }\right)=K_{x x^{* }} M^{-1} \mathbf{y}(\mathbf{x})\\]

The variance is
\\[ \operatorname{Var}\left(y^{* }\right)=k\left(x^{* }, x^{* }\right)-K_{x x^{* }} M^{-1} K_{x^{* } x}^{T}\\]



## Gaussian Mixture Model
### Mixture model
The mixture model is a probabilistic model that can be used to represent K sub-distributions in the overall distribution. In other words, the mixture model represents the probability distribution of the observed data in the population, which is a mixed distribution consisting of K sub-distributions. When calculating the probability of the observed data in the overall distribution, the mixture model does not require the information about the sub-distribution in the observation data.

### Gaussian Model
When the sample data X is univariate, the Gaussian distribution follows the probability density function below.
\\[ P(x\|\theta)=\frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2}) \\]

While \\( \mu \\) is the average value of the data (Expectations), \\( \sigma \\) is the standard deriviation.

When the sample data is multivariate, the Gaussian distribution follows the probability density function below.
\\[ P(x|\theta)=\frac{1}{(2\pi)^{\frac{D}{2}} |\Sigma|^{\frac{1}{2}}}exp(-\frac{(x-\mu)^T\Sigma^{-1}(x-\mu)}{2}) \\]

While \\( \mu \\) is the average value of the data (Expectations), \\( \Sigma \\) is the covariance, D is the dimention of the data.

### Gaussian Mixture Model
The Gaussian mixture model can be regarded as a model composed of K single Gaussian models, which are hidden variables of the hybrid model. In general, a mixed model can use any probability distribution. The Gaussian mixture model is used here because the Gaussian distribution has good mathematical properties and good computational performance.

For example, we now have a bunch of samples of dogs. Different types of dogs have different body types, colors, and looks, but they all belong to the dog category. At this time, the single Gaussian model may not describe the distribution very well since the sample data distribution is not a single ellipse. However, a mixed Gaussian distribution can better describe the problem, as shown in the following figure:
![](assets/mixedGaussian.jpg)

Define:
*  \\( x_j \\) is the number \\( j \\) of the observed data, \\( j = 1, 2, ..., N \\)
*  \\( k \\) is the number of Gaussians in the mixture model, \\( k = 1, 2, ..., K \\)
*  \\( \alpha_k \\)is the probability that the observed data is from the \\( k \\)th gaussian, \\( \alpha_k > 0 \\), \\( \Sigma_{k=1}^{K}\alpha_k=1 \\)
*  \\( \phi(x\|\theta_k) \\) is the gaussian density function of the \\( k \\)th sub-model, \\( \theta_k = (\mu, \sigma^2_k) \\)
*  \\( \gamma_{jk} \\) is the probability that the \\( j \\)th observed data is from the \\( k \\)th gaussian. 

The probablity distribution of the gaussian mixture model:
\\[ P(x|\theta)= \Sigma^K_{k=1} \alpha_k\phi(x|\theta_k) \\]

As for this model, \\( \theta = (\mu_k, \sigma_k, \alpha_k) \\), which is the expectation, standard deriviation (or covariance) and the probability of the mixture model.

### Learning the parameters of the model
As for the single, we can use Maximum Likelihood to estimate the value of parameter \\( \theta \\)
\\[ \theta = argmax_\theta L(\theta) \\]

We assume that each data point is independent and the likelihood function is given by the probability density function.
\\[ L(\theta) = \Pi^N_{j=1}P(x_j|\theta) \\]

Since the probability of occurrence of each point is very small, the product becomes extremely small, which is not conducive to calculation and observation. Therefore, we usually use Maximum Log-Likelihood to calculate (because the Log function is monotonous and does not change the position of the extreme value, At the same time, a small change of the input value can cause a relatively large change in the output value):
\\[ logL(\theta) = \Sigma^N_{j=1}logP(x_j\|\theta) \\]


As for the Gaussian Mixture Model, the Log-likelihood function is:
\\[ logL(\theta) = \Sigma^N_{j=1}logP(x_j\|\theta) = 
\Sigma^N_{j=1}log(\Sigma^K_{k=1})\alpha_k \phi(x\|\theta_k) \\]

So how do you calculate the parameters of the Gaussian mixture model? We can't use the maximum likelihood method to find the parameter that maximizes the likelihood like the single Gaussian model, because we don't know which sub-distribution it belongs to in advance for each observed data point. There is also a summation in the log. The sum of the K Gaussian models is not a Gaussian model. For each submodel, there is an unknown \\( (\mu_k, \sigma_k, \alpha_k) \\), and the direct derivation cannot be calculated. It need to be solved by iterative method.

## Expectation Maximization Algorithm
The EM algorithm is an iterative algorithm, summarized by Dempster et al. in 1977. It is used as a maximum likelihood estimation of probabilistic model parameters with Hidden variables.

Each iteration consists of two steps:

* E-step: Finding the expected value \\( E(\gamma_{jk}\|X,\theta) \\) for all \\( j = 1, 2, ..., N \\)
* M-step: Finding the maximum value and calculate the model parameters of the new iteration 

The general EM algorithm is not specifically introduced here (the lower bound of the likelihood function is obtained by Jensen's inequality, and the maximum likelihood is maximized by the lower bound). We will only derive how to apply the model parameters in the Gaussian mixture model.

The method of updating Gaussian mixture model parameters by EM iteration (we have sample data \\( x_1, x_2, ..., x_N \\) and a Gaussian mixture model with \\( K\\) submodels, we want to calculate the optimal parameters of this Gaussian mixture model):

* Initialize the parameters
* E-step: Calculate the possibility that each data \\( j\\) comes from the submodel \\( k\\) based on the current parameters
\\[ \gamma_{jk} = \frac{\alpha_k\phi(x_j\|\theta_k)}{\Sigma_{k=1}^K\alpha_k\phi(x_j\|\theta_k)}, j=1 , 2,...,N; k=1, 2, ..., K \\]

* M-step: Calculate model parameters for a new iteration
\\[ \mu_k = \frac{\Sigma_j^N(\gamma_{jk}x_j)}{\Sigma_j^N\gamma_{jk}}, k=1,2,...,K \\]
\\[ \Sigma_k = \frac{\Sigma_j^N\gamma_{jk}(x_j-\mu_k)(x_j-\mu_k)^T}{\Sigma_j^N\gamma_{jk}}, k=1,2,...,K \\]
\\[ \alpha_k = \frac{\Sigma_j^N\gamma_{jk}}{N}, k=1,2,...,K \\]
* Repeat the calculation of E-step and M-step until convergence (\\( \|\|\theta_{i+1}-\theta_i\|\|<\epsilon \\), \\( \epsilon \\) is a small positive number, indicating that the parameter changes very small after one iteration)

At this point, we have found the parameters of the Gaussian mixture model. It should be noted that the EM algorithm has a convergence, but does not guarantee that the global maximum is found since it is possible to find the local maximum. The solution is to initialize several different parameters to iterate and take the best result.

## Summary
A Gaussian Process(GP) is a probability distribution over functions. It can be used for nonlinear regression, classification, ranking, preference learning, ordinal regression. In robotics, it can be applied to state estimation, motion planning and in our case environment modeling. It has the advantages of learning the kernel and regularization parameters, uncertainty handling, fully probabilistic predictions, interpretability.

The Gaussian mixture model (GMM) can be regarded as a model composed of K single Gaussian models, which are hidden variables of the hybrid model. It utilizes Gaussian's good mathematical properties and good computational performance.

 Expectation Maximization Algorithm (EM) is a iterative algorithm that act as a maximum likelihood estimation of probabilistic model parameters with Hidden variables. 

## Further Reading
Here are some papers on using GP/GMM in robotics:
1.  Learning Wheel Odometry and Imu Errors for Localization: https://hal.archives-ouvertes.fr/hal-01874593/document
2. Gaussian Process Estimation of Odometry Errors for Localization and Mapping: https://www.dfki.de/fileadmin/user_upload/import/9083_20170615_Gaussian_Process_Estimation_of_Odometry_Errors_for_Localization_and_Mapping.pdf
3. Gaussian Process Motion Planning: https://www.cc.gatech.edu/~bboots3/files/GPMP.pdf
4. Adaptive Sampling and Online Learning in Multi-Robot Sensor Coverage with Mixture of Gaussian Processes: https://www.ri.cmu.edu/wp-content/uploads/2018/08/ICRA18_AdaSam_Coverage.pdf

## References
1. C. E. Rasmussen and C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press,2006.
2. An intuitive guide to Gaussian processes: http://asctecbasics.blogspot.com/2013/06/basic-wifi-communication-setup-part-1.html
3. Qi, Y., Minka, T.P., Picard, R.W., and Ghahramani, Z. *Predictive Automatic Relevance Determination by Expectation Propagation*. In Twenty-first International Conference on Machine Learning (ICML-04). Banff, Alberta, Canada, 2004.
4. O’Hagan, A. *Curve Fitting and Optimal Design for Prediction (with discussion)*. Journal of the Royal Statistical Society B, 40(1):1-42, 1978.
5. MacKay, David, J.C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press. p. 540. ISBN 9780521642989.
6. Dudley, R. M. (1975). "The Gaussian process and how to approach it". Proceedings of the International Congress of Mathematicians.
7. Bishop, C.M. (2006). Pattern Recognition and Machine Learning. Springer. ISBN 978-0-387-31073-2.
8. Dempster, A.P.; Laird, N.M.; Rubin, D.B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm". Journal of the Royal Statistical Society, Series B. 39 (1): 1–38. JSTOR 2984875. MR 0501537.
9. "Maximum likelihood theory for incomplete data from an exponential family". Scandinavian Journal of Statistics. 1 (2): 49–58. JSTOR 4615553. MR 0381110.


/wiki/math/registration-techniques/
---
date: 2020-05-11
title: Registration Techniques in Robotics
---
Absolute orientation is a fundamental and important task in robotics. It seeks to determine the translational, rotational, and uniform scalar correspondence between two different Cartesian coordinate systems given a set of corresponding point pairs. Dr. Berthold K.P. Horn proposed a closed-form solution of absolute orientation which is often a least-squares problem. Horn validated that his solution holds good for two of the most common representations of rotation: unit quaternions and orthonormal matrices.

In robotics, there are typically several sensors in a scene and it is necessary to register all of them with respect to each other and to the robot. It is quite common to then register these components to a chosen world frame, which can just be the frame of one of these components. Horn's method is frequently used to estimate the rotation (R), translation (t), and scale(s) that registers one frame to another.

### Horn's Method
The advantage of Horn's closed-form solution is that it provides one in a single step with the best possible transformation, given at least 3 pairs of matching points in the two coordinate systems. Furthermore, one does not need a good initial guess, as one does when an iterative method is used. The underlying mathematics of Horn's method is elucidated below.

Let's call the two coordinate systems that we wish to register "left" and "right". Let \\( x_{l, i} \\) and \\(x_{r, i} \\) be the coordinates of point i in the left-hand and right-hand coordinate systems respectively. The objective of Horn's method is to find the R, t, and s that transforms a point x in the left coordinate system to an equivalent point x' in the right coordinate system according to the equation:

$$ x^{\prime}=s R x+t $$

Usually, the data are not perfect and we will not be able to find the correspondence between the two points perfectly. The residual error is expressed as:

$$ e_{i}=x_{r, i}-s R x_{l, i}-t $$

Horn's method minimizes the sum of squares of these errors:

$$ (t, s, R)=\underset{t, s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2} $$

Horn's paper [2] offers mathematical rigor into the solution but the most tractable approach [4] to attain the closed-form solution is explained below:

1. Compute the centroids of \\( x_l \\) and \\( x_r \\) as:

$$ \bar{x}_{l}=\frac{1}{n} \sum_{i=1}^{n} x_{l, i}\quad \quad \bar{x}_{r}=\frac{1}{n} \sum_{i=1}^{n} x_{r, i} $$

2. Shift points in both coordinate systems so that they are defined with respect to the centroids:

$$ x_{l, i}^{\prime}=x_{l, i}-\bar{x}_{l}\quad \quad x_{r, i}^{\prime}=x_{r, i}-\bar{x}_{r} $$

3. The residual error term with reference to the centroid of the shifted points is now:

$$ e_{i}=x_{r, i}^{\prime}-s R x_{l, i}^{\prime}-t^{\prime} $$
The translational offset between the centroids of the left and right coordinate systems is:
$$ t^{\prime}=t-\bar{x}_{r}+s R \bar{x}_{l} $$

In [2], Horn proves that the squared error is minimized when \\( t' = \mathbf{0} \\). So the equation that yields \\( t \\) is:

$$ t=\bar{x}_{r}-s R \bar{x}_{l} $$

Thus, the translation is just the difference of the right centroid and the scaled and rotated left centroid.
Once we have solved for \\( R \\) and \\( s \\) we get back to this equations to solve for \\( t \\). Since the error is minimum when \\( t' = \mathbf{0} \\), the error term is now:

$$ \begin{align*} e_{i}=& x_{r, i}^{\prime}-s R x_{l, i}^{\prime}\\ e_{i}=& \frac{1}{\sqrt{s}} x_{r, i}^{\prime}-\sqrt{s} R x_{l, i}^{\prime} \quad \text{(rearranging the terms)} \end{align*} $$

4. Equation (1) now reduces to:

$$ \begin{align*} (s, R)=& \underset{s, R}{\operatorname{argmin}} \sum_{i=1}^{n}\left\|e_{i}\right\|^{2} \\ =& \underset{s, R}{\operatorname{argmin}} \frac{1}{s} \sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}+s \sum_{i=1}^{n}\left\|x_{l, i}\right\|^{2} -2 \sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right) \end{align*} $$

This can be written in the form:

$$ S_{r}-2 s D+s^{2} S_{l} $$

where \\( S_r \\) and \\( S_l \\) are the sums of squares of the measurement vectors(relative to their centroids), and \\( D \\) s the sum of the dot products of the corresponding coordinates in the right system with the rotated coordinates in the left system. Completing the square in \\( s \\) we get:

$$ (s \sqrt{S_{l}}-D / \sqrt{S_{l}})^{2}+\left(S_{r} S_{l}-D^{2}\right) / S_{l} $$

This is minimized with respect to scale \\( s \\) when the first term is zero or \\( s=D / S_{l} \\)

$$ s=\sqrt{\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2} / \sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}} $$

5. Now that \\( s \\) can be solved for, \\( R \\) is obtained by substituting the expression for \\( s \\) in the Equation (3):

$$ R=\underset{R}{\operatorname{argmin}} 2\left(\sqrt{\left(\sum_{i=1}^{n}\left\|x_{r, i}^{\prime}\right\|^{2}\right)\left(\sum_{i=1}^{n}\left\|x_{l, i}^{\prime}\right\|^{2}\right)}-\sum_{i=1}^{n} x_{r, i}^{\prime} \cdot\left(R x_{l, i}^{\prime}\right)\right) $$

6. Once we have \\( s \\) and \\( R \\), \\( t \\) is solved for using Equation (2).

Alternatively in [2], when there are 3 matching points whose correspondence in both systems have to be obtained, Horn constructs triads in the left and right coordinate systems. Let \\( x_l, y_l, z_l \\) and \\( x_r, y_r, z_r \\) be the unit vectors in the left and right coordinate systems respectively. In the left coordinate system, the \\( x \\) axis is defined as the line that joins points 1 and 2, and the \\( y \\) axis is a line perpendicular to this axis, and the \\( z \\) axis is obtained by the cross-product:

$$ \dot{z}_{l}=\hat{x}_{l} \times \hat{y}_{l} $$

Similarly, a triad is constructed in the right coordinate system. The rotation matrix \\( R \\) that we are looking for transforms \\( \hat{x}_{l} \\) into \\( \hat{x}_{l} \\), \\( \hat{y}_{l} \\) into \\( \hat{y}_{r} \\), and \\( \hat{z}_{l}\\) into \\( \bar{z}_{r} \\).

We adjoin column vectors to form the matrices \\( M_l \\) and \\( M_r \\) as follows:

$$ M_l=\left|\hat{x}_l \hat{y}_{l} \hat{z}_{l}\right|, \quad M_{r}=\left|\hat{x}_{r} \hat{y}_{r} \hat{z}_r\right| $$

Given a vector \\( \mathbf{r}_l \\) in the left coordinate system, \\( M_l^T\mathbf{r_l} \\) gives us the components of the vector \\( r_l \\) along the axes of the constructed triad. Multiplication by \\( M_r \\) then maps these into the right coordinate system, so

$$ \mathbf{r}_{r}=M_{r} M_{l}^{T} \mathbf{r}_{l} $$

The rotation matrix \\( R \\) is given by:

$$ R=M_{r} M_{l}^{T} $$

The result is orthonormal since \\( M_l \\) and \\( M_r \\) are orthonormal by construction. Horn admits that while this particular approach to find \\( R \\) by constructing triads gives a closed-form solution for finding \\( R \\) given three points, it cannot be extended to deal with more than three points.

Through the steps listed above, Horn's method presents a closed-form solution to the absolute orientation problem.

### Camera-Robot Registration Using Horn's Method
A popular application of Horn's method is to perform camera-robot registration, i.e., to find the \\( R \\) and \\( t \\) that relates the camera frame and the robot frame. An elegant technique that uses Horn's method to register a stereo camera and the robot is explained in this section. This procedure is based on [5] in which a stereo camera overlooks the workspace of the da Vinci Research Kit by Intuitive Surgical. The code written by Nicolas Zevallos et al. of [5] can be found on the GitHub repository of Carnegie Mellon University's Biorobotics Lab, linked [here](https://github.com/gnastacast/dvrk_vision/blob/master/src/dvrk_vision/dvrk_registration.py)
Team A of the MRSD class of 2021, performed camera-robot registration using a slight modification of their technique. The procedure followed by Nicolas Zevallos et al. is explained below:
* The robot is fitted with a colored bead on its end effector; the colored bead can be easily segmented from its background by hue, saturation, and value. It is preferred that the bead be bright in color so as to increase its visibility to the stereo camera.
* The robot is moved to a fixed set of five or six points. The points are chosen to cover a significant amount of the robot's workspace and also stay within the field of view in the camera. 
* The robot is moved to the specified location. The left and right images of the stereo are processed to find the centroid of the colored bead fitted to the robot. This is repeated for as many frames as are received over ROS (which is the framework used in [5]) in one second. The centroid is averaged over all the received frames for a more accurate spatial estimate of the bead.
* The pixel disparity is given as input to a stereo-camera model that ROS provides to calculate a 3D point in the camera-frame.
* Following this, we have six points in both the camera-frame and the robot-frame. The coordinates of the point in the robot-frame are known using the kinematics of the robot.
* Since we now have corresponding points in the camera-frame and the robot-frame, Horn's method is used to calculate the transformation \\( T_r^c \\) between the robot-frame and the camera-frame.

### Iterative Closest Point Algorithm

Iterative Closest Point (ICP) is an algorithm to register two sets of point clouds iteratively. In each iteration, the algorithm selects the closest points as correspondences and calculates the transformation, i.e., rotation and translation (\\( R \\) and \\( t \\). Although there are a number of techniques to calculate the transformation at each iteration, Horn's method remains a popular choice to do so.

The main steps of ICP [6] are listed below:
1. For each point in the source point cloud, match the closest point in the reference point cloud
2. Calculate a transformation (a combination of rotation, translation, and scaling) which will best align each source point to its match found in the previous step. This is done by using Horn's method or a similar approach.
3. Transform the source points using the obtained transformation
4. Check if the two point clouds align within a desired threshold. If not, iterate through the algorithm again

### Summary
Horn's method is a useful tool to estimate the rotation, translation, and scale that exist between two coordinate systems. It is widely used in robotics to register coordinate frames of different hardware components that comprise a system as well as two sets of point clouds. Thus, Horn's method provides a closed-form solution to the absolute orientation problem.


### References
1. Micheals, Ross & Boult, Terrance. (1999). A New Closed Form Approach to the Absolute Orientation Problem.

2. Berthold K. P. Horn, "Closed-form solution of absolute orientation using unit quaternions," in Journal of the Optical Society of America A, vol. 4, no. 2, pp. 629-642, 1987.

3. Berthold K. P. Horn, "Closed-form solution of absolute orientation using orthonormal matrices," in Journal of the Optical Society of America A, vol. 5, pp. 1127-1135, 1988.

4. Ivan Stojmenović, Handbook of Sensor Networks: Algorithms and Architectures, First published:2 September 2005, Print ISBN:9780471684725, Online ISBN:9780471744146, DOI:10.1002/047174414X

5. Zevallos, Nicolas & Rangaprasad, Arun Srivatsan & Salman, Hadi & Li, Lu & Saxena, Saumya & Xu, Mengyun & Qian, Jianing & Patath, Kartik & Choset, Howie. (2018). A Real-time Augmented Reality Surgical System for Overlaying Stiffness Information. 10.13140/RG.2.2.17472.64005.

6. [ICP Wikipedia](https://en.wikipedia.org/wiki/Iterative_closest_point)

7. [An Overview of Robot-Sensor Calibration Methods for
Evaluation of Perception Systems](http://faculty.cooper.edu/mili/Calibration/index.html)
