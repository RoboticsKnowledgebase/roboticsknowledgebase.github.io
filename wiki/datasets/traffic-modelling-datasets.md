---
date: 2020-04-24
title: Traffic Modelling Datasets
---

Traffic modelling is a hot topic in the field of autonomous cars currently. Here you will find a list of open datasets which can be used as source for building a traffic model. The list will include data captured from a variety of sources as listed below:
- UAV/Drones
- Traffic Camera 
- Autonomous Cars

> This is not a list of datasets for learning-to-drive. This list is more focused towards dataset which provide global perspective of the traffic scenarios, rather than ego-vehicle perspective. Though few ego-vehicle datasets can be used for traffic modelling as well.

# Datasets
1. ### [Argoverse](https://www.argoverse.org/data.html#download-link)
    - 3D tracking data: 113 segments of 15-30secs each.
    - Motion Forecasting: 324,557 segments of 5secs each (Total 320hrs) 
    - Python APIs available for visualization on HD maps
    - Data not split into signalized and non-signalized intersections.

2. ### [Interaction](https://interaction-dataset.com/)
    - Contains roundabout, lane merging - which do not require Traffic Light info.
    - "DR_" recorded via drones. "TC_" are track files recorded via fixed cameras.
    - HD map has drive-able area and lane markings
    - No video or image available
    - Position, velocity, orientation, bbox dimension in csv
    - Python APIs available for visualization on HD maps

3. ### [In-D](https://www.ind-dataset.com/)
    - Python script to visualize data available.
    - bbox dimension, center (x,y), velocity (x,y and lat-long), acceleration (x,y and lat-long), heading in csv.
    - xx_background.png contains image of the road. No other images/ videos.
    - Lane markings, drive-able area part of HD maps (.osm file).
    - Entire data at signalized intersection.

4. ### [High-D](https://www.highd-dataset.com/)
    - Python script to visualize data available.
    - bbox dimension, center (x,y), velocity (x,y and lat-long), acceleration (x,y and lat-long), heading in csv.
    - xx_background.png contains image of the road. No other images/ videos.
    - Lane markings, drive-able area part of HD maps (.osm file).
    - Entire data at highway.

5. ### [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
    - Vehicle ID, local x,y and global x,y, vehicle width and length.
    - Video available for road and lane data. Video quality very bad!
    - CAD files of area available.

6. ### [Stanford](https://cvgl.stanford.edu/projects/uav_data/)
    - Contains data at 8 (~ * 5) scenes.
    - 100% view not available for all intersection. In-campus roads not a good representation of normal traffic scenarios. More pedestrian and bike data.
    - More info available [here](https://mrsd-teamh.atlassian.net/wiki/spaces/M/pages/193363984/Dataset+Details).

7. ### [NuScenes](https://www.nuscenes.org/)
    - 1000 scenes.
    - All location data is given with respect to the global coordinate system.
        - Global x, y, z
        - Bbox l, b, h
        - Rotation in Quaternion
        - Contains Pedestrian Data
        - [Publication](https://arxiv.org/pdf/1903.11027.pdf)

8. ###  [Apollo](http://apolloscape.auto/trajectory.html)
    - The trajectory dataset consists of 53min training sequences and 50min testing sequences captured at 2 FPS
        - Global x, y, z positions
        - object length, width, height, heading
        - frame_id, object_id, object type
    - Data not split into signalized and non-signalized intersections.

9. ### [Round-D](https://www.round-dataset.com/)
    - Python script to visualize data available.
    - bbox dimension, center (x,y), velocity (x,y and lat-long), acceleration (x,y and lat-long), heading in csv.
    - xx_background.png contains image of the road. No other images/ videos.
    - Lane markings, drive-able area part of HD maps (.osm file).
    - Full data on roundabouts.

Below we mention several parameters crucial for learning behavior and interactions between vehicles in a recorded scenario against each dataset.

| S.No. | Data Source |              Road             |         Lane Boundary         | Vehicle | Pedestrian |              Traffic Light             |
|-------|:-----------:|:-----------------------------:|:-----------------------------:|:-------:|:----------:|:--------------------------------------:|
|     1 |  [Argoverse](https://www.argoverse.org/data.html#download-link)  | ✔️                          | ✔️                           | ✔️     | ✔️        | ❌  |
|       |             |                               |                               |         |            |                                        |
|     2 | [Interaction](https://interaction-dataset.com/) | ✔️                           | ✔️                           | ✔️     | ❌         | ❌                                     |
|       |             |                               |                               |         |            |                                        |
|     3 |     [In-D](https://www.ind-dataset.com/)    | ✔️                           | ✔️                           | ✔️     | ✔️        | ❌                                     |
|       |             |                               |                               |         |            |                                        |
|     4 |    [High-D](https://www.highd-dataset.com/)   | ✔️                           | ✔️                           | ✔️     | ✔️        | ❌                                     |
|       |             |                               |                               |         |            |                                        |
|     5 |    [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)    | ✔️                           | ✔️                           | ✔️     | ❌         | ✔️                                    |
|       |             |                               |                               |         |            |                                        |
|     6 |   [Stanford](https://cvgl.stanford.edu/projects/uav_data/)  | ❌  | ❌  | ✔️     | ✔️        | ❌                                     |
|       |             |                               |                               |         |            |                                        |
|     7 |   [NuScenes](https://www.nuscenes.org/)  | ✔️                           | ✔️                           | ✔️     | ✔️        | ❌  |
|       |             |                               |                               |         |            |                                        |
|     8 |    [Apollo](http://apolloscape.auto/trajectory.html)   | ❌                            | ❌                           | ✔️     | ✔️        | ❌                                     |
|       |             |                               |                               |         |            |                                        |
|     9 |    [Round-D](https://www.round-dataset.com/)   | ✔️                           | ✔️                           | ✔️     | ✔️        | ❌                                     |
|       |             |                               |                               |         |            |                                        |
