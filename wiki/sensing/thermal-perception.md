---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-04-29 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Perception via Thermal Imaging
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

In this article, we discuss strategies to implement key steps in a robotic perception pipeline using thermal cameras.
Specifically, we discuss the conditions under which a thermal camera provides more utility than an RGB camera, followed
by implementation details to perform camera calibration, dense depth estimation and odometry using thermal cameras.

## Why Thermal Cameras?

Thermal cameras are useful in key situations where normal RGB cameras fail - notably, in perceptual degradation like
smoke and darkness.
Furthermore, unlike LiDAR and RADAR, thermal cameras do not emit any detectable radiation.
If your robot is expected to operate in darkness and smoke-filled areas, thermal cameras are a means for your robot to
perceive the environment in nearly the same way as visual cameras would in ideal conditions.

## Why Depth is Hard in Thermal

Depth perception — inferring the 3D structure of a scene — generally relies on texture-rich, high-contrast inputs. Thermal imagery tends to violate these assumptions:

- **Low Texture**: Stereo matching algorithms depend on local patches with distinctive features. Thermal scenes often lack these.
- **High Noise**: Infrared sensors may introduce non-Gaussian noise, which confuses pixel-level correspondence.
- **Limited Resolution**: Consumer-grade thermal cameras are often <640×480, constraining disparity accuracy.
- **Spectral Domain Shift**: Models trained on RGB datasets fail to generalize directly to the thermal domain.
_________________________

## Calibration
Calibration is the process by which we can estimate the internal and external parameters of a camera. Usually, the camera intrinsics matrix has the following numbers
- fx, fy - This the focal length of the camera in the x and y directions **in the camera's frame**. px/distance_unit
- cx, cy OR px, py - The principal point or the optical center of the image
- distortion coeffs (2 - 6 numbers depending on distortion model used)

Additionally, we must also estimate camera extrinsics which is the pose of the camera relative to another sensor - the body frame of a robot is defined to be the same as the IMU, or another camera in the case of multi-camera system
- This will be in the form of series of 12 numbers - 9 for the rotation matrix and 3 for the translation
- *NOTE*: BE VERY CAREFUL OF COORDINATE FRAMES
- If using more than one sensor, timesync will help you.

- Calibrating thermal cameras is quite similar to calibrating any other RGB sensor. To accomplish this you must have a checkerboard pattern, Aruco grid or some other calibration target. 
  - A square checkerboard is not ideal because it is symmetrical and it is hard for the algorithm to estimate if the orientation of the board has changed. 
  - An aruco grid gives precise orientation and is the most reliable option but is not necesasary. 

General tips
- For a thermal camera you will need to use something with distinct hot and cold edges, eg: a thermal checkerboard
- Ensure that the edges on the checkerboard are visible and are not fuzzy. If they are adjust the focus, wipe the lens and check if there is any blurring being applied
- Ensure the hot parts of the checkerboard are the hottest things in the picture. This will make it easier to detect the checkboard
- Thermal cameras by default give 16bit output. You will need to convert this to an 8bit grayscale image.
- Other than the checkboard, the lesser things that are visible in the image, the better your calibration will be
- If possible, preprocess your image so that other distracting features will be ignored

### Camera Intrinsics
- Calibrating thermal camera instrinsics will give you fx, fy, cx, cy, and the respective distortion coefficients
1. Heat up the checkerboard
2. Record a rosbag with the necessary topics
3. Preprocess your images
4. Run them through OpenCV or Kalibr. There are plenty of good resources online.

Example output from Kalibr:
```text
  cam0:
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [-0.3418843277284295, 0.09554844659447544, 0.0006766728551819399, 0.00013250437150091342]
  distortion_model: radtan
  intrinsics: [404.9842534577856, 405.0992911907136, 313.1521147858522, 237.73982476898445]
  resolution: [640, 512]
  rostopic: /thermal_left/image
```

### Thermal Camera peculiarities
- Thermal Cameras are extremely noisy. There are ways you can reduce this noise
- **Camera gain calibration:** The gain values on the camera are used to reduce or increase the intensity of the noise in the image. 
  - The noise is increased if you are trying to estimate the static noise and remove it from the image (FFC)

- **Flat Field Correction**: FFC is used to remove any lens effects in the image such as vignetting and thermal patterns in the images
  - FFC is carried out by placing a uniform object in front of the camera and taking a picture
  - Then the noise patterns and then vigneting effects are estimated and then removed from the cameras
  - The FLIR thermal cameras constantly "click" which them placing a shutter in front of the sensor and taking picture and correcting for any noise
  - The FLIR documentation describes Supplemental FFC (SFFC) which is the user performing FFC manually. It is recommended that this is performed when the cameras are in their operating conditions


### Camera Extrinsics
- Relative camera pose is necessary to perform depth estimation. Kalibr calls this a camchain
- Camera-IMU calibration is necessary to perform sensor fusion and integrate both sensor together. This can be estimated using CAD as well.
- Time-sync is extremely important for this because the sensor readings need to be at the exact same time for the algorithm to effectively estimate poses.
- While performing extrinsics calibration, ensure that all axes are excited (up-down, left-right, fwd-back, roll, pitch, yaw) sufficiently. ENSURE that you move slow enough that there is no motion blur with the calibration target but fast enough to excite the axes enough.
________

## Our Depth Estimation Pipeline Evolution

### 1. **Stereo Block Matching**

We started with classical stereo techniques. Given left and right images $I_L, I_R$, stereo block matching computes disparity $d(x, y)$ using a sliding window that minimizes a similarity cost (e.g., sum of absolute differences):

$d(x, y) = argmin_d \space Cost(x, y, d)$

In broad strokes, this brute force approach compares blocks from $I_L$ and $I_R$. For each block it computes a cost based on the pixel to pixel similarity (using a loss between feature descriptors generally). Finally, once a block match is found, the disparity is found by checking how much each pixel has moved in the x direction.

As you can imagine, this approach is simplea nd lightweight. However, it is dependent on many things such as the noise in your images, the contrast separation, and will struggle to find accurate matches when looking at textureless and colorless inputs (like a wall in a thermal image). The algorithm performed better than expected, but we chose not to go ahead with it.

---

### 2. **Monocular Relative Depth with MoGe**

If you are using a single camera setup, this is called a monocular approach. One issue is that this problem is ill posed. For example, I can move back the objects twice in distance and scale them to twice their size and you will end up with the same image as you did earlier. This means there are multiple solutions to the problem and when an image is captured in a single camera, this information is lost. Therefore, learning based models are employed to hallucinate the right depth (most likely based on data driven priors like the standard height of chairs). One such model is MoGe (Monocular Geometry) which estimates *relative* depth $z'$ from a single image. These estimates are affine-invariant, meaning they suffer from unknown global scale and shift:

$z = s \cdot z' + t$

This means they look visually coherent (look at the image below on the right), but the ambiguity limits 3D metric use (SLAM based applications).

![Relative Depth on Thermal Images](/assets/images/Moge_relative_thermal.png)

---

### 3. **MADPose Solver for Metric Recovery**

To address MoGe’s ambiguity, we incorporated MADPose — a solver that optimizes scale and shift across time by integrating motion estimates. This optimizer also estimates other properties such as extrinsics between the cameras solving for more unknowns than that were necessary. Additionally, there is no temporal constraint imposed (you are looking at mostly the same things between $T$ and $T+1$ timesteps). This meant that the metric depth that we recovered would keep changing significantly across frames, resulting in pointclouds of different sizes and distances across timesteps.

---

### 4. **Monocular Metric Depth Predictors**

We also tested monocular models trained to output metric depth directly. This problem would be the most ill-posed problem as you would definitely overfit to the baseline of your training data and the approach would fail to generalize to other baselines. These treat depth as a regression problem from single input $I$:

$z(x, y) = f(I(x, y))$

Thermal's lack of depth cues and color made the problem even harder, and the models performed poorly.

---


### 5. **Stereo Networks Trained on RGB (e.g., MS2, KITTI)**

Alternatively, when a dual camera setup is used, we call it a stereo approach. This inherently is a much simpler problem to solve as you have two rays that intersect at the point of capture. I encourage looking at the following set of videos to understand epipolar geometry and the formualtion behind the stereo camera setup [Link](https://www.youtube.com/watch?v=6kpBqfgSPRc). 

We evaluated multiple pretrained stereo disparity networks. However, there were a lot of differences between the datasets used for pretraining and our data distribution. These models failed to generalize due to:

- Domain mismatch (RGB → thermal)
- Texture reliance
- Exposure to only outdoor content
- Reduced exposure

---


## Final Approach: FoundationStereo

Our final and most successful solution was [FoundationStereo](https://github.com/NVlabs/FoundationStereo), a foundation model for depth estimation that generalizes to unseen domains without retraining. It is trained on large-scale synthetic stereo data and supports robust zero-shot inference.

### Why It Works:

- **Zero-shot Generalization**: No need for thermal-specific fine-tuning.
- **Strong Priors**: Learned over large datasets of scenes with varied geometry and lighting. (These variations helped overcome RGB to thermal domain shifts and textureless cues)
- **Robust Matching**: Confidence estimation allows the model to ignore uncertain matches rathern than hallucinate.
- **Formulation**: Formulating the problem as dense depth matching problem also served well. This allowed generalization to any baseline by constraining the output to the pixel space.

Stereo rectified thermal image pairs are given to FoundationStereo and we receive clean disparity maps (image space). We recover metric depth using the intrinsics of the camera and the baseline. Finally we can reproject this into the 3D space to get consistent point clouds:

$
z = \frac{f \cdot B}{d}
$

Where:
- $f$ = focal length,
- $B$ = baseline between cameras,
- $d$ = disparity at pixel.

An example output is given below (thermal preprocessed on the top left, disparity is middle left, and the metric pointcloud is on the right).

![Metric Depth using Foundation Models](/assets/images/foundation_stereo.png)
## Lessons Learned

1. **Texture matters**: Thermal's low detail forces the need for models that use global context.
2. **Don’t trust pretrained RGB models**: They often don’t generalize without retraining.
3. **Stereo > Monocular for thermal**: Even noisy stereo is better than ill-posed monocular predictions.
4. **Foundation models are promising**: Large-scale pretrained vision backbones like FoundationStereo are surprisingly effective out-of-the-box.

## Conclusion

Recovering depth from thermal imagery is hard — but not impossible. While classical and RGB-trained methods struggled, modern foundation stereo models overcame the domain gap with minimal effort. Our experience suggests that for any team facing depth recovery in non-traditional modalities, foundation models are a compelling place to start.

## See Also
- The [Thermal Cameras wiki page](https://roboticsknowledgebase.com/wiki/sensing/thermal-cameras/) goes into more depth about how thermal cameras function.
