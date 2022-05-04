

---
date: 2022-04-26
title: Comparison of Fiducial Markers
published: true
---
## Why is this comparison important?
Different fiducial markers are best suited for different applications, so this article elucidates a brief list of pros and cons to evaluate different fiducial markers, enabling readers to choose a suitable one for one’s application. 

Many important considerations are required to make a decision on which fiducial marker is suitable for a particular application, such as accuracy, OpenCV/ROS support, available repositories/usage in robotics community, resistance to occlusion, marker size/camera resolution, CPU usage for detection, suitability for a particular sensor, etc. There are many available options starting from barcodes and QR codes which are the most widely used in the general population to ArUco markers which are commonly used in the robotics community due to their reliability and existing OpenCV implementation. However, there is no silver bullet. Even some of the most popular fiducial markers, ArUco markers are not detected under occlusion hence it is important to pick alternatives like STag if occlusion resistance is an important criteria. In addition to this, the camera used impacts the choice as well. For example, regular web cameras may suffer from perspective distortion, which makes perspective support an important criteria. In summary, one must balance all the desired criteria and weigh them as per one’s use-case.

## WhyCon
WhyCon is a simple circular fiducial marker with a black disk and white circular interior that does not feature any ID encoding. It uses circular encoding similar to encoders to decode. The marker identities are determined in the algorithm using the diameter ratios with relaxed tolerances for the inner/outer segment area ratio to prevent false detections. It is designed for use on low-cost devices for tasks that require fast detection rates. Note: It requires camera_info topic to calibrate.

The WhyCode marker is an extension to WhyCon that features traditional topological encoding, offering a six bit central component instead of a plain white circular interior. This addition of the bit information provides the orientation of the tag which is useful in various applications.

CPU usage with image resolution 840x480: 8% @ 6fps , 9% @ 15fps, 19% @ 90fps 
Average frame detection time: 7ms

### Pros
- Very fast (20-30x ArUco)

### Cons
- No proper library support for pose estimation, pose estimation will require very precise marker placements.
- Occlusion sensitive (90% visibility required)
- Prone to detecting false positive simple circles. A workaround can be to use complex markers and not simple numbers to remove the false positives.
- Can be confused by concentric circles. A workaround can be to use multiple markers.

## CALTag
[CALTag](https://www.cs.ubc.ca/labs/imager/tr/2010/Atcheson_VMV2010_CALTag/)  is a self-identifying marker pattern that can be accurately and automatically detected in images. This marker was developed for camera calibration. It is a square, monochromatic tag made up of multiple black and white squares, each with individual identification bits in their own center. These small identification bits are extremely beneficial for calibration tasks which require a high level of precision. However, this is less beneficial in autonomous systems applications since the precision comes at the cost of the identification rate. Nevertheless, CALTag marker detection is robust to occlusions, uneven illumination and moderate lens distortion. It has about the same speed and compute load as ArUco marker detection.

### Pros
- Very resilient to occlusion
- Very high (mm level) accuracy

### Cons
- Runs only on MATLAB
- Larger size marker needed for detection when marker needs to be detected at not very close distance (~4m)

## STag
STag is a fiducial marker that uses ellipse fitting algorithms to provide stable detections. The STag has a square border similar to other markers, but it also includes a circular pattern in their center. After the line segmentation and quad detection of the square border, an initial homography is calculated for the detected marker. Then, by detecting the circular pattern in the center of the marker, the initial homography is refined using elliptical fitting. Elliptical fitting is shown to provide a better localization compared to the quad detection and localization. This refinement step provides increased stability to the measurements. [STag repository](https://github.com/bbenligiray/stag), [STag ROS repository](https://github.com/usrl-uofsc/stag_ros/)
It has a similar compute load to arUco markers.
CPU usage with image resolution 848x640 : 33% @ 5fps, 70% @ 30fps, 150% @ 90fps
Avg frame detection time: 35ms

Pros:
- High resilience to occlusion
- Much greater viewing angle
- Larger marker size needed
- Higher compute cost
Cons:
- No library is provided to generate tags (yet!), however a drive link for 57K tags has been provided which can be used.

## CCTag
[CCTag](https://github.com/alicevision/CCTag) is a round, monochromatic marker that consists of concentric black and white rings. It is designed to achieve reliable detection and identification of planar features in images, in particular for  unidirectional motion blur. CCTag lacks any ID encoding but the authors propose that it can potentially be used as a circular barcode with varying ring sizes. It also boasts low rates of false positives and misidentifications to provide a tag with high detection accuracy and recognition rate even under adverse conditions.

### Pros:
- Extreme resilience to occlusion and blur
### Cons:
- High CPU usage (200%)
- Recommended to run on GPU

## Fourier Tag
The Fourier Tag encodes binary data into a grayscale, radially symmetric structure instead of using a topological method and uses the Fourier Transform to detect the variation in the intensity of the rings. This method optimizes partial data extraction under adverse image conditions and improves the detection range. The detection can be optimized for different tasks to achieve the necessary balance for robustness and encoding capacity depending on the task. The main advantage is the slow degradation of accuracy compared to other fiducial markers that may be either entirely detected or entirely lost, this tag is able to degrade smoothly. 
### Pros:
Can be viewed from much greater distance or tag size can be extremely small
### Cons:
Cannot handle occlusion but code can be modified to include some level of resilience 

## AprilTag
AprilTag is based on the framework established by ARTag while offering a number of  improvements. Firstly, a graph-based image segmentation algorithm was introduced that analyzes gradient patterns on the image to precisely estimate lines. Secondly, a quad extraction method is added that allows edges that do not intersect to be identified as possible candidates. Thirdly, a new coding system is also implemented to address issues stemming from the 2D barcode system. These issues include incompatibility with rotation and false positives in outdoor conditions. As a result, AprilTag has an increased robustness against occlusions and warping as well as a decreased amount of misdetections. It is still supported by the creators with regular updates. The primary source of error in AprilTag pose estimatation is the angular rotation of the camera. To address the challenges caused by an unaligned yaw, a trigonometric correction method is introduced to realign the frame along with a pose-indexed probabilistic sensor error model to filter the tracking data. 

### Pros:
- Allow the user to specify a list of markers to detect
- Existing ROS support
### Cons:
- Less accurate than ArUco, especially with varying viewing angle

## ArUco
ArUco is another package based on ARTag and ARToolkit. The most notable contribution of ArUco marker is that it allows the user to create configurable libraries. Instead of including all possible markers in a standard library, users can generate a custom library based on their specific needs. This library will only contain specific markers with the greatest Hamming distance possible, which reduces the potential for false detections. Another advantage of the smaller size of the custom libraries is the reduced computing time. 
CPU usage with image resolution 840x480 : 3% @ 1fps, 20% @ 6fps , 40% @ 15fps, 100% @ 90fps tnet
Average frame detection time: 17 ms

### Pros: 
- Pre defined/optimized OpenCV library
- Existing ROS support
- Markers can be generated using OpenCV
### Cons:
- Very high occlusion sensitivity. ( 100% visibility required, specially corners )
- For companies: Older version licensed for commercial use. ( Missing out continuous camera tracking feature increasing the pose estimate and detection accuracy
