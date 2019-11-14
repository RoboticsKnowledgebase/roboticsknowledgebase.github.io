---
title: 'Radar Camera Sensor Fusion '
published: true
---
Fusing data from multiple sensor is an integral part of the perception system of robots and especially Autonomous Vehicles. The fusion becomes specially useful when the data coming from the different sensors gives complementary information. In this tutorial we give an introduction to Radar Camera sensor fusion for tracking oncoming vehicles. A camera is helpful in detection of vehicles in the short range, however recovering the 3D velocity of the vehicles solely based on vision is very challenging and inaccurate, especially for long-range detection. This is where the RADAR comes into play. RADAR is excellent for determining the speed of oncoming vehicles and operation in adverse weather and lighting conditions, whereas the camera provides rich visual features required for object classification. 

The position and velocity estimates are improved through the sensor fusion of RADAR and camera. We will first go through the details regarding the data obtained and the processing required for the individual sensors and then go through the sensor fusion and tracking the part. 

TODO: shift this to ipm
Although sensor fusion is currently not complete, some of the sub-tasks such as Inverse Perspective Mapping and RADAR integration have been completed this semester. The idea behind this is to create a birds-eye-view of the environment around the vehicle by using a perspective transformation. Using this birds-eye-view representation and some known priors such as camera parameters and extrinsic with respect to a calibration checkerboard, all the vehicles can be mapped to their positions in the real-world. Fusion of this data along with RADAR targets can provide us reliable states of all vehicles in the scene. Finally, an occupancy grid will be generated which can be used for prediction and planning.


## Camera


### Object Detection 


### Object Tracking in Images 
TODO:Refine
SORT is an approach to multiple object tracking where the focus is to associate objects efficiently for online and real-time applications. It follows a tracking-by-detection framework for the problem of multiple object tracking (MOT) where objects are detected in each frame and represented as bounding boxes. SORT makes use of Kalman filter to predict the state of the bounding box in the next frame. This helps keep track of the vehicles even if the bounding box detections are missed over few frames. Additionally, I incorporated appearance basis into the existing SORT tracker and increased the memory window from one frame to around 100 frames. This helped maintain the same tracker ID for the vehicles over the frames, even if the bounding boxes go unmissed for few seconds. This is important to associate the same trajectory for a vehicle over a period, or else we might end up with multiple short trajectories for the same vehicle.

### Inverse Perspective Mapping
TODO:Refine
Inverse Perspective Mapping is basically a perspective transformation that requires a homography matrix. Since we want to project the camera view on to the ground plane perspective instead, we perform an inverse transformation using this matrix. One way to do this is to directly pick a set of four points corresponding to a rectangular region on the ground plane and then estimate the homography matrix.

TODO:Refine
Since we had made progress in creating our own maps and importing it in CARLA, we were able to place calibration checkerboards on the road in our map as shown in Figure 1. The camera homography was manually calibrated for a given frame with some fixed extrinsic parameters that do not change over the entire simulation sequence. Earlier, I had an issue with setting the output size of the mapped result and finding the mapping between the image frame and real-world cartesian coordinates of the vehicles. By using the checkerboard as reference (8x8 grid, 40cm square cells), placed at some known coordinates in the world frame and using the odometry of the ego vehicle, I was able to find the perspective mapping and the scale factors that map the birds-eye-view image coordinates to world frame coordinates (in meters). Another checkerboard was placed at 10m apart from the first one to validate the mapping and compute the projection error. Errors were less than 10cm for objects mapped within 20-30m from the ego vehicle. However, the projection errors for objects beyond 50m are significant (5m+) and the perspective mapping module needs to be recalibrated more precisely to bring down these errors.

#### Camera Output


## Radar
Radar is becoming an important automotive technology. Automotive radar systems are the primary sensor used in adaptive cruise control and are a critical sensor system in autonomous driving assistance systems (ADAS). In this tutorial it is assumed that the radar internally filters the raw sensor data and gives the final positions and velocities of the vehicles within the field of view. This is a reasonable assumption since most of the automotive grade radars already do such filtering and tracking and return the tracked vehicle estimate in the cartesian coordinates.

#### Radar Output


## Camera Radar Tracker

Camera RADAR tracker can be summed up with following sub parts: 
- Data association of camera and radar detections
- Motion compensation of Ego vehicle
- State predicion and update using Kalman Filter
- Data association of predictions and detections
- Handling occlusions and miss detections
- Validation of tracker using MOTP and MOTA metrics

### Data fusion - Camera and RADAR detections
TODO:Fix grammar
You must be getting an array of detections from camera and RADAR for every frame. First of all you need to link the corresponding detections in both (all) the sensors. This is  done using computing a distance cost volume for each detecion og a sensor with each detections from another sensor. scipy library performs good resources for computing such functions in Python. Then you ned to use a minimisation optimization function to associate detections such that overall cost (Euclidian distance) summed up over the entire detections is minimised. For doing that Hungarian data association rule is used. It matches the minimum weight in a bipartite graph. Scipy library provides good functionality for this as well. 

### Motion compensation of Ego-vehicles
TODO:Refine
This block basically transforms all the track predictions one timestep by the ego vehicle motion. This is an important block because the prediction (based on a vehicle motion model) is computed in the ego vehicle frame at the previous timestep. If the ego vehicle was static, the new sensor measurements could easily be associated with the predictions, but this would fail if the ego vehicle moved from its previous position. This is the reason why we need to compensate all the predictions by ego motion first, before moving on to data association with the new measurements. The equations for ego motion compensation are shown below.

[■(X_(t+1)@Y_(t+1)@1)]=[■(cos⁡(ωdt)&sin⁡(ωdt)&-v_x dt@-sin⁡(ωdt)&cos⁡(ωdt)&-v_y dt@0&0&1)][■(X_t@Y_t@1)]  

Since later we are supposed to associate these detetions with the predictions from KF (explained in the later section), we need to compensate their state values according to the ego vehicle motion. This is done to compare (associate) the detections from sensors and prediction algorithm on a common ground. You must already be having ego vehicle state information from odometry sensors. Using these two states - Ego vehicles state and oncoming state - oncoming vehicle state is to be output as if the ego vehicle motion was not there. 

### Gaussian state prediction - Kalman Filter
Kalman Filter is an optimal filtering and estimation technique which uses a series of measurements (with noise) over time to estimate the unknown variables which tend to be more accurate than the individual estimates. It is widely used concept ina variety of fields ranging from state estimation to optimal controls and motion planning. The algorithm works as a two step process which are as follows:
- Prediction Step
- Measurement Step

In our case we use Kalman Filter to estimate states of the vehicles in the environment using data from Radar and Camera. The states consist of the position and velocity of the vehicle, which are provided by the tracker in the form of measurement.

#### Prediction Step
Prediction step is the step where we will estimate the state of the vehicle in the next timestep using data we have currently and a motion model. We chose to use a constant velocity (CV) motion model for prediction. This model considers the vehicle as a point object which can move at a constant velocity in the x and y direction. This is not the best model to represent the dynamics of a vehicle but since we dont have information about the steering or throttle data of the vehicle, we will have to be satisfied with this. When we predict the future state, we also need to estimate the noise that might have propped up in the system due to a variety of reasons. This noise is called the process noise and we assume that it is dependant on the motion model and the timestep. This [link](https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb?create=1) gives a good idea about the CV motion model and the process noise associated with it.

The sudo code for the prediction step is
```
def predict():
    predicted_state = transition_matrix * old_state
    predicted_covariance = transition_matrix' * old_covariance * transition_matrix + process_noise

```

#### Measurement Step
Our measurement step is actually divided into two parts, one for the camera measurement and the other for the radar measurement. While updating the states of the filter we need to be sure that the timestep of the measurement and the predictions match. The update step is much more complicated and 

The sudo code for the prediction step is
```
def update():
    residual = sensor_measurement - measurement_function * predicted_state
    projected_covariance = measurement_function * predicted_covariance * measurement_function + sensor_noise
    kalman_gain = measurement_function' * projected_covariance * 
    new_state = predicted_state + kalman_gain * residual
    new_covariance = (identity - kalman_gain * measurement_function) * projected_covariance

```


 
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
