---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-05-02 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Mediapipe - Live ML Anywhere
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

## Introduction - What is Mediapipe?

MediaPipe offers cross-platform, customizable ML solutions for live and streaming media. With common hardware, Mediapipe allows fast ML inference and processing. With Mediapipe, you can deploy the solutions anywhere, on Android, iOS, desktop/cloud, web and IoT platforms. The advantage of Mediapipe is that you get cutting-edge ML solutions that are free and open source. 

## Solutions offered

![Figure 1. Mediapipe Solutions](../assets/mediapipe_solutions.png)
The image above summarizes the solutions offered by mediapipe. 
The solutions below have been classified into 2 categories based on the use cases:

Following are the solutions offered for the detection of humans and their body parts: 
1. Face detection
MediaPipe Face Detection is an ultra-fast face detection solution that comes with 6 landmarks and multi-face support. 

2. FaceMesh
MediaPipe Face Mesh is a solution that estimates 468 3D face landmarks in real-time even on mobile devices. 

3. Mediapipe Hands
MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.

4. MediaPipe Pose
MediaPipe Pose is an ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks and background segmentation mask on the whole body from RGB video frames. 

5. MediaPipe Holistic
The MediaPipe Holistic pipeline integrates separate models for the pose, face and hand components, each of which is optimized for its particular domain.

6. MediaPipe Hair Segmentation
MediaPipe Hair Segmentation segments the hairs on the human face. 

7. MediaPipe Selfie Segmentation
MediaPipe Selfie Segmentation segments the prominent humans in the scene. It can run in real-time on both smartphones and laptops. 

Following are the solutions offered for the detection and tracking of everyday objects
1. Box tracking 
The box tracking solution consumes image frames from a video or camera stream, and starts box positions with timestamps, indicating 2D regions of interest to track, and computes the tracked box positions for each frame.

2. Instant Motion tracking
MediaPipe Instant Motion Tracking provides AR tracking across devices and platforms without initialization or calibration. It is built upon the MediaPipe Box Tracking solution. With Instant Motion Tracking, you can easily place virtual 2D and 3D content on static or moving surfaces, allowing them to seamlessly interact with the real-world environment.

3. Objectron
MediaPipe Objectron is a mobile real-time 3D object detection solution for everyday objects. It detects objects in 2D images, and estimates their poses through a machine learning (ML) model, trained on the Objectron dataset.

4. KNIFT  
MediaPipe KNIFT is a template-based feature matching solution using KNIFT (Keypoint Neural Invariant Feature Transform). KNIFT is a strong feature descriptor robust not only to affine distortions, but to some degree of perspective distortions as well. This can be a crucial building block to establish reliable correspondences between different views of an object or scene, forming the foundation for approaches like template matching, image retrieval and structure from motion.

The table below describes the support of the above models for currently available platforms:
![Figure 1. Mediapipe supported platforms](../assets/mediapipe_platforms.png)

## Quickstart Guide
Mediapipe solutions are available for various platforms viz. Android, iOS, Python, JavaScript, C++. The guide at [Getting Started](https://google.github.io/mediapipe/getting_started/getting_started.html) comprises instructions for various platforms. 

For this section of the quick-start guide, we will introduce you to getting started using Python.
MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. MediaPipe Python package is available on [PyPI](https://pypi.org/project/mediapipe/) for Linux, macOS and Windows.

1. Step 1 - Activate the virtual environment:

```
python3 -m venv mp_env && source mp_env/bin/activate
```

The above code snippet will create a virtual environment `mp_env` and start the virtual environment. 

2. Step 2 - Install MediaPipe Python package using the following command:

```
(mp_env)$ pip3 install mediapipe
```

You are all set! You can now start using mediapipe. A quickstart script for Mediapipe hands is present in the Example section. 

## Example

The example code below is the example for Media-pipe hands pose estimation. Ensure that you have OpenCV installed. If not you can use the terminal command below to install OpenCV. 
```
pip3 install opencv-python
```

The code below shows the quick start example for Media-pipe hands. Appropriate comments have been added to the code which can be referred to understand the code. 
```
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

## Summary
In Summary, Mediapipe is an amazing tool for running ML algorithms online. For common applications like human pose detection, hands pose estimation, etc this package eliminates the need to go over the tedious process of data collection, data labeling and training a deep learning model. However, the downside is that if object detection is needed for custom objects, users still need to go through the process of labeling and training a deep learning model. Nevertheless, using the APIs in the projects, users can focus more on using the output to create impactful applications.  


## See Also:
- Gesture Control of your FireTV with Python [here](https://medium.com/analytics-vidhya/gesture-control-of-your-firetv-with-python-7d3d6c9a503b).
- MediaPipe Object Detection & Box Tracking [here](https://medium.com/analytics-vidhya/mediapipe-object-detection-and-box-tracking-82926abc50c2)
- Deep Learning based Human Pose Estimation using OpenCV and MediaPipe [here](https://medium.com/nerd-for-tech/deep-learning-based-human-pose-estimation-using-opencv-and-mediapipe-d0be7a834076)

## References
- Mediapipe Documentation. [Online]. Available: https://google.github.io/mediapipe/.
- Getting Started Documentation. [Online]. Available: https://google.github.io/mediapipe/getting_started/getting_started.html
- Mediapipe Hands Architecture. [Online]. Available: https://arxiv.org/abs/2006.10214
- MediaPipe: A Framework for Building Perception Pipelines [Online]. Available: https://arxiv.org/abs/1906.08172