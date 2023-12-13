---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-12-13 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Azure Block Detection 
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

This article presents an overview of object detection using the Azure camera without relying on learning-based methods. It is utilized within our Robot Autonomy project, specifically for detecting Jenga blocks and attempting to assemble them.

### Detection Pipeline
To identify individual blocks and their respective grasping points, the perception subsystem undergoes a series of five steps. Initially, it crops the Azure Kinect camera image to center on the workspace. Following this, it applies color thresholding to filter out irrelevant objects and discern the blocks. Subsequently, it identifies the contours of these blocks and filters them based on their area and shape characteristics. Once the blocks are recognized, the perception subsystem computes the grasping points for each block. Collectively, these steps facilitate the accurate detection of block locations and their corresponding grasping points on the workstation.


![Pipeline of Block Detection](assets/pipeline.png)

### Image Cropping
The initial stage of the perception subsystem involves cropping the raw image. Raw images often contain extraneous details, such as the workspace's supporting platform or the presence of individuals' feet near the robot. By cropping the image to focus solely on the workspace, we eliminate a significant amount of unnecessary information, thereby enhancing the system's efficiency and robustness.

Currently, this approach employs hard-coded cropping parameters, requiring manual specification of the rows and columns to retain within the image.

![Cropped Image](assets/cropped.png)

### Color Segmentation
Color segmentation can pose challenges in images with prominent shadows. Shadows cause a decrease in RGB pixel values, while light causes an increase, making it challenging to distinguish between different colors. To address this, we employ HSV (Hue, Saturation, Value) thresholding on the image.

For reliable detection of the brown color of the Jenga blocks under varying lighting conditions, we utilize the HSV color space, consisting of three channels: hue, saturation, and value. By thresholding these channels, we filter out the desired colors. However, using a fixed RGB threshold for detecting brown is challenging due to its variable RGB values under different lighting.

To tackle this issue, we employed color meter software to establish the brown color range for the Jenga blocks. This range, encompassing lower and upper brown values, was applied to our HSV thresholding function. The resulting HSV thresholded image is depicted in Figure 10b.

To further refine Jenga block detection and eliminate background noise, we apply a mask to the HSV thresholded image. Initially, we create a mask by contour area thresholding and then fill any holes within the contour to obtain a solid mask. The resulting masked image is shown in Figure 6a. This process ensures the reliable detection of Jenga blocks by removing remaining noise or unwanted objects.

![RGB Vector](assets/rgb_vector.png)




