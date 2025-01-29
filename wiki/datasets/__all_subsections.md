wiki/datasets/open-source-datasets/
---
# Date the article was last updated like this:
date: 2021-04-27 # YYYY-MM-DD
# Article's title:
title: Open Source Datasets
---
This is an article to teach you how to make your own dataset or where to find open-source datasets that are free to use and download.

## Creating a Custom Dataset
Capture your own images with a camera then create labels for each image that indicates the bounding boxes and IDs of the object class captured.

*Option 1:*
Create labels for all of the images using Yolo_mark [1]. The repo and instructions for use can be found [here](https://github.com/AlexeyAB/Yolo_mark). These labels will be made in the darknet format. 

*Option 2:*
Use Innotescus, a Pittsburgh startup working on high-performance image annotation. They offer free academic accounts to CMU students. You can upload datasets and have multiple people working on annotations. There are task metrics that track how many of each class of image are annotated and show heat maps of their relative locations within an image so you can ensure proper data distributions.

Create a free beta account [here](https://innotescus.io/demo/)


## Open-Source Datasets:
### General Datasets
[OpenImages](https://storage.googleapis.com/openimages/web/index.html)

[MS COCO](https://cocodataset.org/#home)

[Labelme](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php)

[ImageNet](http://image-net.org/)

[COIL100](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)

Image to Language:   
[Visual Genome](http://visualgenome.org/)  
[Visual Qa](http://www.visualqa.org/)  

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


### Specific Application Datasets:

[Chess Pieces](https://public.roboflow.com/object-detection/chess-full)

[BCCD](https://public.roboflow.com/object-detection/bccd)

[Mountain Dew](https://public.roboflow.com/object-detection/mountain-dew-commercial)

[Pistols](https://public.roboflow.com/object-detection/pistols)

[Packages](https://public.roboflow.com/object-detection/packages-dataset)

[6-sided dice](https://public.roboflow.com/object-detection/dice)

[Boggle board](https://public.roboflow.com/object-detection/boggle-boards)

[Uno Cards](https://public.roboflow.com/object-detection/uno-cards)

[Lego Bricks](https://www.kaggle.com/joosthazelzet/lego-brick-images)

[YouTube](https://research.google.com/youtube8m/index.html)

[Synthetic Fruit](https://public.roboflow.com/object-detection/synthetic-fruit)

[Fruit](https://public.roboflow.com/classification/fruits-dataset)

Flowers:  
[Flower Classification 1](https://public.roboflow.com/classification/flowers_classification)  
[Flower Classification 2](https://public.roboflow.com/classification/flowers)  
[Flower Classification 3](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  

Plants:   
[Plant Doc](https://public.roboflow.com/object-detection/plantdoc)  
[Plant Analysis](https://www.plant-image-analysis.org/dataset)  

[Wildfire smoke](https://public.roboflow.com/object-detection/wildfire-smoke)

[Aerial Maritime Drone](https://public.roboflow.com/object-detection/aerial-maritime)

[Anki Vector Robot](https://public.roboflow.com/object-detection/robot)

[Home Objects](http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/)

Indoor Room Scenes:   
[Princeton lsun](http://lsun.cs.princeton.edu/2016/)  
[MIT toralba](http://web.mit.edu/torralba/www/indoor.html)  

[Places](http://places.csail.mit.edu/index.html)

[Parking Lot](https://public.roboflow.com/object-detection/pklot)

[Car Models](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)

[Improved Udacity Self Driving Car](https://public.roboflow.com/object-detection/self-driving-car)

[Pothole](https://public.roboflow.com/object-detection/pothole)

[Hard Hat](https://public.roboflow.com/object-detection/hard-hat-workers)

[Masks](https://public.roboflow.com/object-detection/mask-wearing)

#### People and Animals:
[Aquarium](https://public.roboflow.com/object-detection/aquarium)

[Brackish Underwater](https://public.roboflow.com/object-detection/brackish-underwater)

[Racoon](https://public.roboflow.com/object-detection/raccoon)

[Thermal Cheetah](https://public.roboflow.com/object-detection/thermal-cheetah)

[ASL](https://public.roboflow.com/object-detection/american-sign-language-letters)

[RPS](https://public.roboflow.com/classification/rock-paper-scissors)

[Human Hands](https://public.roboflow.com/object-detection/hands)

[Human Faces](http://vis-www.cs.umass.edu/lfw/)

[Celebrity Faces](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

[Thermal Dogs and People](https://public.roboflow.com/object-detection/thermal-dogs-and-people)

[Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

[Dogs and Cats](https://public.roboflow.com/object-detection/oxford-pets)


## Summary
We reviewed how to create labels for custom images to build a dataset. We also reviewed where to access specific and general open-source datasets depending on your application.

## See Also:
- Using your [custom dataset to train YOLO on darknet for object detection](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io.git/wiki/machine-learning/train-darknet-on-custom-dataset)

## References
[1] AlexeyAB (2019) Yolo_mark (Version ea049f3). <https://github.com/AlexeyAB/Yolo_mark>.  




/wiki/datasets/traffic-modelling-datasets/
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
