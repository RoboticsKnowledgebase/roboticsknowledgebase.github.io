# Lane Detection in the Simulated Environment

This post will walk you through the different lane detection techniques with the code and results in the a simulated environment (For example Carla, NVIDIA drive simulators). 
Simulator used: Carla
System specifications: 
Nvidia Titan V GPU, 32 GB RAM and intel i7 processor - 3.5 GHz 

Firstly, I’ll be talking about the OpenCV based method (classical computer vision).This approach has mostly been derived from Udacity nanodegree course on autonomous vehicles - https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013


## OpenCV Based Lane Detection:

First step will be to convert the RGB image to Gray-scale image. This means that a 3 channel image is converted into a single channel image. This is done, as canny edge detector (whch is used afterwards) takes in an input only from a gray-scale image. Code used for this is: 
'''
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
'''
canny() function parses the pixel values according to their gradient. What’s left over are the edges — or where there is a steep derivative in at least one direction. We will need to supply thresholds for canny() as it computes the gradient. Recommended values of low to high threshold ratio of 1:2 or 1:3.

Code for doing this is:

low_threshold = 50
high_threshold = 150
canny_edges = canny(gauss_gray,low_threshold,high_threshold)

Please go through following link if you want to know in details of how OpenCV performs Canny Edge detection - https://docs.opencv.org/master/da/d22/tutorial_py_canny.html

Now we will be using Hough transform to which will convert a line in the cartesian space to a point in the Hough space and vice versa. 



## Algorithm for Hough transform is as follows:

1. Edge detection, e.g. using the Canny edge detector. 
2. Mapping of edge points to the Hough space and storage in an accumulator. 
3. Interpretation of the accumulator to yield lines of infinite length. The interpretation is done by thresholding and possibly other constraints. 
4. Conversion of infinite lines to finite lines.

If you want to go in details on it, there are some awesome YouTube videos which does the explanation with the appropriate visualization. This is could be a good read as well - http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf

Lastly to overlay the original image with lane marking detection mask image can be done by - a OpenCV function → cv2.addWeighted().
cv2.addWeighted(initial_img, alpha, line_image, beta, lambda)

You can refer my repository for more details, variable names and comments should be self-explanatory about how the code works - https://github.com/singaporv/lane_detect_openCV

This technique if not very robust to different kinds of noises. Since it depends on hand crafted feature, this algorithm works on a straight lanes, and as per the development of the code it can only detect ego vehicle (self vehicle) lanes. 
One of the major problem with this technique is that it doesn’t handle any occlusions at all. If any vehicle comes in the view of lanes then this algorithm is simply going to ignore the lanes.

Now, we will be discussing about a better, but slower technique - LaneNet. 
This is a deep learning based method, and like any other field this LaneNet has beaten the classical computer vision techniques by big margins.

This lanenet algorithm works at 90 FPS on above defined system specs. 

Following is the opensource code for this - https://github.com/MaybeShewill-CV/lanenet-lane-detection 
Also, refer to the arxiv paper - https://arxiv.org/pdf/1802.05591.pdf

This technique replaces the handcrafter techniques that was previously shown in OpenCV approach hence making it more robust to different kinds of environments. 

This algorithm is an end-to-end lane detection algorithm, i.e. you give in an image and it will give out the detected lanes. 
Another advantage is that it can handle occlusions very well. This is because that is how this model has been trained, cityscapes dataset. 
Another big plus point is that it doesn’t just detects the lanes, but it does instance segmentation of the lanes. This way we will have the specific IDs for the lanes. Hence, the subsystem which comes after this i.e. prediction/planning can make a good use of those IDs in doing their maneuvers and also doing predictions. 
 LaneNet is a branched, multi-task architecture to cast the lane detection problem as an instance segmentation task, that handles lane changes and allows the inference of an arbitrary number of lanes. In particular, the lane segmentation branch outputs dense, per-pixel lane segments, while the lane embedding branch further disentangles the segmented lane pixels into different lane instances. A network that given the input image estimates the parameters of a perspective transformation that allows for lane fitting robust against road plane changes, e.g. up/downhill slope.

Below is the image showing its architecture:

## Working of LaneNet:
combines the benefits of binary lane segmentation with a clustering loss function designed for one-shot instance segmentation. In the output of LaneNet, each lane pixel is assigned the id of their corresponding lane.

We still have to fit a curve through these pixels to get the lane parametrization. Typically, the lane pixels are first projected into a ”bird’s-eye view” representation, using a fixed transformation matrix. However, due to the fact that the transformation parameters are fixed for all images, this raises issues when non-flat ground-planes are encountered, e.g. in slopes. To alleviate this problem, we train a network, referred to as H-Net, that estimates the parameters of an ”ideal” perspective transformation, conditioned on the input image. This transformation is not necessarily the typical ”bird’s eye view”. Instead, it is the transformation in which the lane can be optimally fitted with a low-order polynomial.
For more information on H-Net following link is a good resource - https://www.ijcai.org/proceedings/2018/119

As seen in the architecture there are 2 branches - Binary segmentation and Instance segmentation. 
binary segmentation The segmentation branch of LaneNetis trained to output a binary segmentation map, indicating which pixels belong to a lane and which not. To construct the ground-truth segmentation map, all ground-truth lane points1 are connected together, forming a connected line per lane. These ground-truth lanes are drawn even through objects like occluding cars, or also in the absence of explicit visual lane segments, like dashed or faded lanes. This way, the network will learn to predict lane location even when they are occluded or in adverse circumstances. The segmentation network is trained with the standard cross-entropy loss function. Since the two classes (lane/background) are highly unbalanced, we apply bounded inverse class weighting, as described in. instance segmentation To disentangle the lane pixels identified by the segmentation branch, we train the second branch of LaneNet for lane instance embedding. Most popular detect-and-segment approaches are not ideal for lane instance segmentation, since bounding box detection is more suited for compact objects, which lanes are not. Therefore we use a one-shot method based on distance metric learning, proposed by De Brabandere et al., which can easily be integrated with standard feed-forward networks and which is specifically designed for real-time applications.

For clustering we used DBScan cluster algorithm. It clusters the points only if there are specific minimum points are given as an input. Hence it prevents the misdetections from taking place. For more information on it please refer - https://en.wikipedia.org/wiki/DBSCAN

One thing that was missing in this algorithm was the filtering, i.e. it was having lot of noises, hence lanes were moving all over the place. I made my own filter for doing this. 
You can refer my code from following repository - https://github.com/deltaautonomy/delta_perception/tree/master/lanenet


This filtering techniques penalizes too much deviation in the parameters of slopes by giving more weights to the one prior frame. If there are more than specific deviations then it completely ignores the detection on that particular frame and uses prior knowledge of the lanes to do the detection in the current frame. This filtering has been inspired from this post - http://petermoran.org/robust-lane-tracking/?fbclid=IwAR2yEK0cnJUxIUOtiLVfC1KFYUPgtndojkB8a_SVZFJy1TlsRxLinIJ_UL4 

