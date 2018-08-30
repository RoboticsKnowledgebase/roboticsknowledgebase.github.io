---
title: AprilTags
---
AprilTags is a visual fiducial system, useful for a wide variety of tasks including augmented reality, robotics, and camera calibration. The tags provide a means of identification and 3D positioning, even in low visibility conditions. The tags act like barcodes, storing a small amount of information (tag ID), while also enabling simple and accurate 6D (x, y, z, roll, pitch, yaw) pose estimation of the tag.

![AprilTags placed on multiple mobile robots platforms](assets/apriltags-6719c.png)

The AprilTags project originates from a team at the University of Michigan, that has a [detailed website dedicated to the research](https://april.eecs.umich.edu/apriltag/), which is a good starting off point for learning how to use all of the software. The team has provided implementations in both Java and C to read AprilTags from a camera and there are additional implementations available online for tag reading. In addition to the software available on their website, a student at MIT has released a [C++ implementation](http://people.csail.mit.edu/kaess/apriltags). This website also has printable AprilTags for several tag families available in PDF format.

## ROS implementation

In ROS Groovy and Hydro, there is no built-in AprilTag recognition software. Our team is working on wrapping the code from the MIT C++ implementation into a working ROS package, which outputs AprilTags that are seen in a camera image as tf transforms and messages.

## Experimental Results

In informal tests, the April Tags were able to achieve accuracy within 4 centimeters of the actual pose when the camera was within 2 meters of the tag. If the robot was further away, the accuracy decreased proportionally to the distance from the tag.

A few methods of improving speed and range are available at [AprilTags - Improving Speed and Range](http://cmumrsdproject.wikispaces.com/AprilTags+-+Improving+speed+and+range)

## Potential Problems

There are several key aspects to take into consideration when deciding on whether to use AprilTags.
1. The accuracy of the pose estimation is only as good as the camera being used.
  - In tests conducted by CMU MRSD Team E (2014), the Kinect camera was adequate, since it outputs a fairly high quality RGB image at 640px by 480px. The team had problems when using a low quality, < $10 camera we found online. The tag detections were not able to give reliable detections of tags, and when the camera was being moved, the results were even worse.
  - The same team's final hardware solution was a Logitech c270HD webcam, which we ran at a resolution of 640px x 480px. The quality was good, and the pose estimation results were accurate once the parameters were properly tuned.
- Motion affects the accuracy of the pose estimation
  - When the robot is moving, especially while turning, it was found that the pose estimation could be off by > 20 degrees, and vary in range by up to 0.75 meters. We determined that the best course of action for our robot was to stop when an AprilTag was detected in order to get a more accurate pose estimation.
- Distance to the tag can affect the pose estimate
  - The MRSD team's hardware setup (Kinect), saw that the pose was accurate out to a range of 2.4 meters, but could vary substantially when further than that. They decided that the robot should not localize when it is beyond this threshold range.
- Finding proper camera configuration parameters is essential
  - The initial trials of the robot were inaccurate for several reasons, as mentioned above. Another reason for inaccurate localization and pose estimation was the default values for a camera parameter did not match the specifications of the hardware we were using, specifically focal length. The Kinect sensor has a focal length, in pixels, of 525, but the default value in the software was 600. This caused incorrect size estimation by the software, which meant that the robot thought it was closer than it was to the tag. Once the parameter was adjusted, the accuracy improved by several centimeters at close range, and more at further distances.

## Alternatives
An alternative to AprilTags for localization/pose estimation of an object is the [AR tag,](http://www.artag.net/) which was created for Augmented Reality, but can provide similar pose estimation. The benefit of using AprilTags is the improved accuracy, especially in less optimal lighting conditions.


## Transforming the frame
One of the problems with running the April Tag node is that your frame output is dependent on the orientation of the body moving. Changes in your body's orientation changes the frame and makes it hard to have a consistent coordinate system. Team E 2014 wrote code in order to transform the AprilTag information to a consistent frame. The source code is [april tag frame transform](https://github.com/ColumnRobotics/column/blob/master/src/rectified_april_tag.cpp). The team found this code to be a bit noisy, so we wrote a RANSAC filter, available [here](https://github.com/ColumnRobotics/column/blob/master/src/BodyPoseFilter.cpp). A way to fix these errors would be to rely on the IMU orientation for the transformation and not the orientation from the April Tag.

## References
1. AprilTag homepage: http://april.eecs.umich.edu/wiki/index.php/AprilTags
- AprilTag C implementation: http://april.eecs.umich.edu/wiki/index.php/AprilTags-C
- AprilTag Java implementation: http://april.eecs.umich.edu/wiki/index.php/AprilTags-Java
- AprilTag C++ implementation:http://people.csail.mit.edu/kaess/apriltags/
- AR Tags: http://www.artag.net/
- Augmented Reality Tracking (ALVAR): http://virtual.vtt.fi/virtual/proj2/multimedia/alvar/
- ALVAR ROS Wrapper: http://wiki.ros.org/ar_track_alvar
