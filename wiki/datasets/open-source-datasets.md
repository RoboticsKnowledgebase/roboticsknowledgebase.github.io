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


