---
title: 'Radar Camera Sensor Fusion '
published: true
---
Fusing data from multiple sensor is an integral part of the perception system of robots and especially Autonomous Vehicles. The fusion becomes specially useful when the data coming from the different sensors gives complementary information. In this tutorial we give an introduction to Radar Camera sensor fusion for tracking oncoming vehicles. A camera is helpful in detection of vehicles in the short range while radar performs really well for long range vehicle detection. 

We will first go through the details regarding the data obtained and the processing required for the individual sensors and then go through the sensor fusion and tracking the part. 

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).

## Camera
Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Object Detection 
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.
![This is an object detection image](assets/images/Hk47portrait-298x300.jpg)
> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.

### Object Tracking in images 
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.

### Inverse Perspective Mapping 
![This is IPM input and output](assets/images/Hk47portrait-298x300.jpg)

#### Camera Output
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

## Radar

#### Radar Output
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

## Camera Radar Tracker

Camera RADAR tracker can be summed up with following sub parts: 
- Data association of camera and radar detections
- Motion compensation of Ego vehicle
- State predicion and update using Extended Kalman Filter
- Data association of predictions and detections
- Handling occlusions and miss detections
- Validation of tracker using MOTP and MOTA metrics

### Data fusion - Camera and RADAR detections
You must be getting an array of detections from camera and RADAR for every frame. First of all you need to link the corresponding detections in both (all) the sensors. This is  done using computing a distance cost volume for each detecion og a sensor with each detections from another sensor. scipy library performs good resources for computing such functions in Python. Then you ned to use a minimisation optimization function to associate detections such that overall cost (Euclidian distance) summed up over the entire detections is minimised. For doing that Hungarian data association rule is used. It matches the minimum weight in a bipartite graph. Scipy library provides good functionality for this as well. 

### Motion compensation of Ego-vehicles
Since later we are supposed to associate these detetions with the predictions from EKF (explained in the later section), we need to compensate their state values according to the ego vehicle motion. This is done to compare (associate) the detections from sensors and prediction algorithm on a common ground. You must already be having ego vehicle state information from odometry sensors. Using these two states - Ego vehicles state and oncoming state - oncoming vehicle state is to be output as if the ego vehicle motion was not there. 

### Gaussian state prediction - Extended Kalman Filter
 -- Karmesh
 
### Data association - prediction and detection 
Next once you have the ego-vehicle motion compensated oncoming vehicle state, then you need to follow same algorithm to associate these two sets of state values.

### Occlusion and miss-detections handling
This is the most important section for tuning the tracker. Here you need to handle for how long you will be contnuing the tracks (continue predicting the state of the track) if that detection is not observed from the sensors in the continuous set of frames. Also another tuning parameter is that for how long you want to continuously detect the object through sensors to confirm with a definite solution that the oncoming vehicle is there.You need to use 3 sets of sensor detections as input: 
- Camera only detections
- RADAR only detections
- Above detections that are able to fuse
Here you need to define the misses (age of non-detections) for each detections. The point of this parameter is that you will increment this age if that corresponding state (to that track) is not observed through sensors. Once any of the state from detecions from sensors is able to associate with the prediction produced by the tracks then we again set back that track parameter to 0.

### Validation of tracker using MOTP and MOTA metrics

-- Apoorv

### Trajectory Smoothing

-- Heethesh

## Summary

-- Apoorv

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
