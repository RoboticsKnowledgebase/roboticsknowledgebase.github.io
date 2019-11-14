---
title: Writing a custom particle filter for localization
---
In this article, we cover the basic steps one should perform to write their custom implementation of a particle filter for localizing a robot. Writing your own particle filter could be beneficiary if, for example, you want to incorporate sensor measurements outside of the commonplace scenarios e.g. you want to measure the bearing and distance to a line feature. A particle filter is a good choice if you have a multimodal state distribution, which cannot be modeled by a Kalman filter.

# Overview
A particle filter consists of three main components: the prediction step; which uses a motion model to propagate the state forward, the measurement step; which uses a sensor model to incorporate your observations; and a resampling step to increase your belief in likely states and decrease your belief in unlikely ones.

# Initialization
First, you need to initialize your particles. If you have a general idea of where your robot is already, you can initialize your particles with a Gaussian distribution. For example, if your state vector is (x, y, theta), then you could use:

`
particles = np.random.multivariate_normal(mean, cov, (num_particles, 3))
`

# Prediction Step
Your prediction step should move your particles forward in time, without incorporating sensor measurements which would fix your robot in a global frame. Usually, in robotics this is done with odometry, e.g. wheel odometry or visual odometry.

# Measurement Step
Your measurement step should weight your particles by approximating P(z|x), the probability of a state given the hypothesis that this particle's state estimate, x, is correct. For example, if you know the location of a landmark in 3D space, and you can measure the distance to that landmark with your sensor, you could use a Gaussian centered on the distance from x to the landmark.

This is arguably most of the work involved in writing your particle filter, and arguably the easiest place to mess things up! One good way to mess things up is to try to do geometry without known good geometric primitives like homogeneous transforms or quaternions. Another tip for debugging is to plot your P(z|x) for some example z and x for which it is obvious to you where the peak in the distribution should be. E.g. if you observe a distance to your landmark of 0, and the sensor model does not have a high probability of for P(z|landmark's position), you know something is wrong!

# Resampling
Your resampling step should eliminate particles proportional to their weight. The choice of resampling method is not cruicial, systematic resampling works well in practice.

# Outputting an Estimate
Now that you have a bunch of weighted particles, you need to output a state estimate! You can average all of the particles, but note that this will fail if your distribution is multimodal (and in theory, having a multimodal state distribution is why you chose a particle filter in the first place!). You can take the particle with the highest weight as your estimate to deal with this, but your output will not be smooth. You can also cluster your particles and output the mean of the biggest cluster.
