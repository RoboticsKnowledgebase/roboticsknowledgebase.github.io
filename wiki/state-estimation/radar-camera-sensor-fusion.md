---
title: 'Radar Camera Sensor Fusion'
published: true
---
Fusing data from multiple sensor is an integral part of the perception system of robots and especially Autonomous Vehicles. The fusion becomes specially useful when the data coming from the different sensors gives complementary information. In this tutorial we give an introduction to Radar Camera sensor fusion for tracking oncoming vehicles. A camera is helpful in detection of vehicles in the short range while radar performs really well for long range vehicle detection. 

We will first go through the details regarding the data obtained and the processing required for the individual sensors and then go through the sensor fusion and tracking the part. 

Recovering the 3D velocity of the vehicles solely based on vision is very challenging and inaccurate, especially for long-range detection. RADAR is excellent for determining the speed of oncoming vehicles and operation in adverse weather and lighting conditions, whereas the camera provides rich visual features required for object classification. The position and velocity estimates are improved through the sensor fusion of RADAR and camera. Although sensor fusion is currently not complete, some of the sub-tasks such as Inverse Perspective Mapping and RADAR integration have been completed this semester. The idea behind this is to create a birds-eye-view of the environment around the vehicle by using a perspective transformation. Using this birds-eye-view representation and some known priors such as camera parameters and extrinsic with respect to a calibration checkerboard, all the vehicles can be mapped to their positions in the real-world. Fusion of this data along with RADAR targets can provide us reliable states of all vehicles in the scene. Finally, an occupancy grid will be generated which can be used for prediction and planning.

## Camera
This section will explain how you use the information from a camera to estimate the rough position of a object in the world. The method described below makes an assumption that ego vehicle or a robot with the camera mounted and the other objects that we would like to track are all on the same plane in the world.


### Object Detection
The first step is to detect 2D bounding boxes of the object you want to track. State-of-the-art techniques for object detection that run in real-time are deep learning based single shot detector models such as SSD or YOLOv3. You can read more about single shot detectors [here](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/single-shot-detectors.html). This tutorial uses the YOLOv3 model for object detection. [This](https://github.com/AlexeyAB/darknet) fork of the original implementation is recommended as it has additional optimizations for real-time inference. 

Clone the repository and build the framework on your system.

```
git clone https://github.com/AlexeyAB/darknet
cd darknet
make
```

Refer to `README` in the repository for more instructions on how to build Darknet for different platforms. Make sure you select the correct GPU architecture in the `Makefile`. Additionally, you can set build flags such as `CUDNN_HALF=1` to accelerate training and inference.

The repository also provides a Python API (`darknet_video.py`) to load the YOLOv3 model for inference using an OpenCV image  as the input and return a list of bounding box detections. The detections contain the following data.

```
[class_label confidence x y h w]
```

You can use any other technique for object detection and get a similar list of detections.

### Object Tracking in Images 
Although the goal of camera based perception is to only estimate a position of the object, you can also track the bounding boxes in the image as well to better associate the detection later in sensor fusion and the main tracker. This step is optional and can be skipped.

A popular approach or technique for real-time tracking is [SORT](https://arxiv.org/abs/1602.00763) (Simple Online and Real-Time Tracking). It follows a tracking-by-detection framework for the problem of multiple object tracking (MOT) where objects are detected in each frame and represented as bounding boxes. SORT makes use of Kalman filter to predict the state of the bounding box in the next frame. This helps keep track of the vehicles even if the bounding box detections are missed over few frames.

The official implementation [here](https://github.com/abewley/sort) is recommended. Follow the instructions in the `README` of this repository to setup SORT. Below is the gist of how to instantiate and update SORT.

```
from sort import *

#create instance of SORT
mot_tracker = Sort() 

# get detections
...

# update SORT
track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...
```

### Inverse Perspective Mapping
Inverse Perspective Mapping is basically a perspective transformation or a homography between two planes in the world. The idea here is to project the camera view (image plane) on to the ground plane in the world to obtain a birds-eye-view of the scene. One way to do this is to directly pick a set of points (minimum 4) in the image corresponding to a rectangular region on the ground plane and then estimate the homography matrix.

In order to estimate the homography, you need a set correspondences. You need a reference object lying flat on the world ground plane (any rectangular shaped board would do). Measure the location of the corner points of the board in the world frame and log the projection of these point on the image plane as well. You can use OpenCV's `cv2.getPerspectiveTransform` to feed in the corresponding points and obtain the homography.

Once the homography is known, pick the bottom center of all the bounding boxes, as this point most closely represents the point on the ground plane, and apply the homography to this image point to obtain an estimate of location in the world frame.

TODO: Add image for calibration

There are many pitfalls and assumptions in this technique. As mentioned earlier, the objects to detect must lie on the same ground plane and the relative distance of the camera sensor and orientation with respect to the ground plane must remain constant. If the bounding box detection is inaccurate, a small deviation in the image point might lead to a significant error in the estimated position in the world frame.

You can also model the uncertainty in the position estimate to generate an occupancy grid with the mean and covariance of the position of the object. We will later see how to fuse these estimates with another sensor modality such as a RADAR to refine our estimate and track these detections as well.

TODO: Occupancy grid with Gaussian

#### Camera Output

Following is the result of camera detection and estimated position in the 3D world. The detection was performed on image stream from Carla simulator and the results are visualized in Rviz.

TODO: Add image for final output (with occupancy grid)

## Radar

#### Radar Output


## Camera Radar Tracker

Camera RADAR tracker can be summed up with following sub parts: 
- Data association of camera and radar detections
- Motion compensation of Ego vehicle
- State predicion and update using Extended Kalman Filter
- Data association of predictions and detections
- Handling occlusions and miss detections
- Validation of tracker using MOTP and MOTA metrics

### Data fusion - Camera and RADAR detections
TODO:Fix grammar
You must be getting an array of detections from camera and RADAR for every frame. First of all you need to link the corresponding detections in both (all) the sensors. This is  done using computing a distance cost volume for each detecion og a sensor with each detections from another sensor. scipy library performs good resources for computing such functions in Python. Then you ned to use a minimisation optimization function to associate detections such that overall cost (Euclidian distance) summed up over the entire detections is minimised. For doing that Hungarian data association rule is used. It matches the minimum weight in a bipartite graph. Scipy library provides good functionality for this as well. 

### Motion Compensation of Tracks
This block basically transforms all the track predictions one timestep by the ego vehicle motion. This is an important block because the prediction (based on a vehicle motion model) is computed in the ego vehicle frame at the previous timestep. If the ego vehicle was static, the new sensor measurements could easily be associated with the predictions, but this would fail if the ego vehicle moved from its previous position. This is the reason why we need to compensate all the predictions by ego motion first, before moving on to data association with the new measurements. The equations for ego motion compensation are shown below.

\\[\left[ \begin{array} { c } { X _ { t + 1 } } \\ { Y _ { t + 1 } } \\ { 1 } \end{array} \right] = \left[ \begin{array} { c c c } { \cos ( \omega d t ) } & { \sin ( \omega d t ) } & { - v _ { x } d t } \\ { - \sin ( \omega d t ) } & { \cos ( \omega d t ) } & { - v _ { y } d t } \\ { 0 } & { 0 } & { 1 } \end{array} \right] \left[ \begin{array} { c } { X _ { t } } \\ { Y _ { t } } \\ { 1 } \end{array} \right]\\]
 

Since later we are supposed to associate these detetions with the predictions from EKF (explained in the later section), we need to compensate their state values according to the ego vehicle motion. This is done to compare (associate) the detections from sensors and prediction algorithm on a common ground. You must already be having ego vehicle state information from odometry sensors. Using these two states - Ego vehicles state and oncoming state - oncoming vehicle state is to be output as if the ego vehicle motion was not there. 

### Gaussian state prediction - Extended Kalman Filter
 -- Karmesh
 
### Data association - prediction and detection
TODO:More content
Next once you have the ego-vehicle motion compensated oncoming vehicle state, then you need to follow same algorithm to associate these two sets of state values.

### Occlusion and miss-detections handling
This is the most important section for tuning the tracker. Here you need to handle for how long you will be contnuing the tracks (continue predicting the state of the track) if that detection is not observed from the sensors in the continuous set of frames. Also another tuning parameter is that for how long you want to continuously detect the object through sensors to confirm with a definite solution that the oncoming vehicle is there.You need to use 3 sets of sensor detections as input: 
- Camera only detections
- RADAR only detections
- Above detections that are able to fuse
Here you need to define the misses (age of non-detections) for each detections. The point of this parameter is that you will increment this age if that corresponding state (to that track) is not observed through sensors. Once any of the state from detecions from sensors is able to associate with the prediction produced by the tracks then we again set back that track parameter to 0.

### Validation of tracker using MOTP and MOTA metrics
The most widely used metrics for validation are MOTA (Multi-object tracking accuracy) and MOTP (Multi-object tracking precision). MOTP is the total error in estimated position for matched object-hypothesis pairs over all frames, averaged by the total number of matches made. It shows the ability of the tracker to estimate precise object positions, independent of its skill at recognizing object configurations, keeping consistent trajectories, and so forth. The MOTA accounts for all object configuration errors made by the tracker, false positives, misses, mismatches, over all frames.

To evaluate your tracker, you can use the [`motmetrics`](https://github.com/cheind/py-motmetrics) Python library.

```
pip install motmetrics
```

At every frame instance, you need the ground truth position and track ID along with the predicted detection and track ID. Use the accumulator to record all the events and you can generate a summary and list the metrics and the end of the sequence. Refer to the README [here](https://github.com/cheind/py-motmetrics) for details on how to use this library. Following is an example of the results summary generated.

```
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML FP FN IDs  FM  MOTA  MOTP
seq1 83.3% 83.3% 83.3% 83.3% 83.3%  2  1  1  0  1  1   1   1 50.0% 0.340
seq2 75.0% 75.0% 75.0% 75.0% 75.0%  2  1  1  0  1  1   0   0 50.0% 0.167

```

### Trajectory Smoothing


## Summary


## See Also:
- [Delphi ESR Radar](https://github.com/deltaautonomy/roboticsknowledgebase.github.io/blob/master/wiki/sensing/delphi-esr-radar.md)

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.
- Link to YOLO
- Link to SORT
- [Kalman Filter in Python](https://github.com/balzer82/Kalman)

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
