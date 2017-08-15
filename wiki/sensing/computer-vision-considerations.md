---
title: Computer Vision for Robotics- Practical Considerations
---
Things to take care of when working with vision in robotics:

Its easy to get enchanted by the promises of the camera. However, computer vision can kill you if you don't heed to some important things. (It can kill you even after that though!). Below are some pitfalls one can avoid:

## Lighting
- Your vision algorithms effectiveness is directly dependent on the surrounding lighting, after all camera is just a device to capture light.
- ALWAYS test in your real conditions! The biggest mistake in robotics is telling yourself- it should work!

## Camera
- Know the camera's field of view and maximum resolution.
- Sometimes autofocus can also be a pain to deal with -- mostly you will be better off with a camera without it.
- Same with autoexposure.

## Calibration
- If you are doing pose estimation using CV, your calibration error will affect your pose accuracy/precision.
- When calibrating, make sure your target (chessboard) is as planar as it can get. Also make sure its edges and corners are sharp.
  - The above point is even more relevant if you are using the OpenCV software for calibration because it auto-detects corners
  - Apart from OpenCV, another good calibration toolbox is [Caltech Calib Toolbox](http://www.vision.caltech.edu/bouguetj/calib_doc/).

## Scale
- Any knowledge you recover about the world using Computer Vision is only accurate up to a scale. Estimating this scale is difficult. Possible solutions include:
  - Depth Sensors (Kinect): will consume power and processor though
  - Stereo camera pair: low on power, but inaccurate after a point (depends on baseline)
  - Known measurement in image: simple and effective, however it requires prior knowledge of the robot's operating environment.

## Numerical Errors:
- This is an inherent problem with vision due to the fact that pixels are discrete. One of the things you can do is normalize your values before applying any algorithm to it.
- Another is using as much precision as you can.

## Frame Rates
- CV algorithms can be very slow, especially on the limited hardware capabilities of most robotic systems. This can be bad for estimation algorithms. Try and make your algorithm run faster. Some ways to do that are:
  - Parallelization: Difficult but, if implemented well, can provide huge boosts to speed. Make sure you know what you're doing -- its very easy to go wrong with parallelization.
  - Region of Interest (RoI): if you are not working on the entire image, do not process the entire image. Some intelligent heuristics can be used to select RoIs.
  - Resolution: :ower the resolution. However, this is a tradeoff. You may end up increasing your error.

## Reference Frame
- This is a common problem across robotics and sensors -- not just for vision. Be sure of the frame in which you get values from any software/algorithm. Especially if you are using third party software (say ROS packages).
- ALWAYS perform sanity checks with moving (rotate + translate) camera by hand.

## Summary
If you keep the above points in mind, dealing with Computer Vision will be easier. You will be able to save days of debugging time.
