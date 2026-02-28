---
date: 2024-05-01
title: Camera Calibration
---

It is quite common for projects to involve vision systems, but despite this many teams working with cameras ignore the basic advantages of calibrating a camera properly. Not only does a good calibration lead to better quality of data in form of images captured but also reduces overall errors in the vision the system, a properly calibrated camera helps avoid unnecessary delays due to radiometric and geometric errors.

## What is Camera Calibration?
Camera Calibration is the process of estimating internal or external parameters of a camera by taking multiple measurements of a known object. Often both are necessary to get a system working well.

#### Types of Camera Calibration
- **Intrinsic** - Estimates the internal parameters of the camera including the focal length, the optical center, and distortion coefficients. This can be useful when trying to transform points between pixel space and 3D space.
- **Extrinsic** - Estimates the external parameters of the camera i.e. the translation and rotation of the camera relative to a reference point. This can be useful when incorporating information from multiple sensors or when trying to act upon information captured by a camera.

## Tools for Camera Calibration
There are multiple methods of camera calibration depending on what you are trying to achieve. Some specific tutorials on camera calibration are listed below:

- [Camera IMU Calibration](/wiki/sensing/camera-imu-calibration.md) - Using the Kalibr library to calibrate a camera and IMU
    - Kalibr can also be used for intrinsic camera calibration as shown in [this video](https://www.youtube.com/watch?app=desktop&v=puNXsnrYWTY)
- [Hand-Eye Calibration with easy_handeye](/wiki/sensing/easy-handeye.md) - Using the easy_handeye library to perform extrinsic camera calibration with a robotic arm
- [Photometric Calibration](/wiki/sensing/photometric-calibration.md) - Calibrating a camera to determine the absolute value of light intensity


## References
This is only a brief overview, for more detailed information about camera calibration, check out the links below
- [Camera Calibration](http://www.cs.rutgers.edu/~elgammal/classes/cs534/lectures/CameraCalibration-book-chapter.pdf) - Zhengyou Zhang
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf) - Richard Szeliski (Chapter 2)
- [Camera Calibration](https://www.mathworks.com/help/vision/camera-calibration.html) - Mathworks
- [What is lens distortion?](https://photographylife.com/what-is-distortion) - Nasim Mansurov 