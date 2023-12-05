---
date: 2023-12-04
title: OpenCV Stereo Vision Processing + OpenCV Methods with ROS 2
---
For a stereo vision camera, there are multiple operations that can be done like tracking, detection, position estimation, etc. OpenCV has lot of libraries which are much faster than MATLAB's computer vision toolbox.

## Resources
Following are some links to helpful resources:
1. Video on installing OpenCV and opencv_contrib modules: https://www.youtube.com/watch?v=vp0AbhXXTrw
2. Camera Calibration resource: http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html#gsc.tab=0
3. Camera calibration and 3D triangulation resources:
  - http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  - http://docs.opencv.org/3.1.0/d0/dbd/group__triangulation.html#gsc.tab=0



## OpenCV with ROS 2 and Image Processing Guide
This part of the page serves as a guide for using some of OpenCV’s most useful functions with ROS 2. The versions of both libraries that will be used for this guide are OpenCV 4.8.0-dev (for C++) and ROS 2 Foxy Fitzroy. This page assumes you have already installed both libraries and other necessary libraries in a Linux distribution and have already configured g++ and CMake to organize your software, link dependencies, execute the code, etc. The focus of this page will be on the functions and setup. 
OpenCV with ROS 2
An issue some encounter with using OpenCV libraries with ROS 2 and camera attachments/sensors is that data comes in as one or more streams of ROS images on a ROS topic, but connecting the ROS 2 node to the data stream(s) is not as simple as subscribing to the topic. Moreover, before any OpenCV methods can be applied the objects being streamed from the camera must be converted into cv::Mat objects. A CvBridge function needs to be used to translate sensor messages to OpenCV Mat objects. 
There are two methods to accomplish this. The simpler method to connect the two is:

```
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

const std::string topic_name = “/camera/image_rect_raw”; // CHECK YOUR CAMERA’S GUIDEBOOK OR CONNECT TO IT AND RUN “ros2 topic list” TO GET THE CORRECT TOPIC NAME
const std::string image_encoding = “16UC1”; // CHECK YOUR CAMERA’S SETTINGS TO GET THE CORRECT IMAGE ENCODING (examples: “bgr8”, “mono8”, “mono16”, “16UC1”, “CV_16S”, etc.) 
// SEE http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages FOR MORE DETAILS
const int _pub_queue = 10;
const inst _sub_queue = 10;
class OpenCVClass : public rclcpp::Node
{
  OpenCVClass() : Node (“opencv_test_node”)
  {
    subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(topic_name, _sub_queue, std::bind(&OpenCVClass::image_callback, this, std::placeholders::_1));
  }

  cv::namedWindow(“Display Window”);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat msg_image_matrix = cv_bridge::toCvCopy(msg, image_encoding)->image;
    // OpenCV Functions Go Here
    cv::imshow(“Display Window”, msg_image_matrix);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto _node= std::make_shared<OpenCVClass>();
  rclcpp::spin(_node);
  rclcpp::shutdown();
  return 0;
}

```

However, it has been noticed the above method may not work on occasion due to linking issues between the distro of ROS and OpenCV, or dependency issues with OpenCV. This may be an artifact of bugs that have since been patched, but there is a second method that can be used in case it reoccurs. This method uses an image_transport object to pass images from the topic to the ROS 2 node:

```
// SAME DEPENDENCIES AS ABOVE, THEN ADD:

#include <image_transport/image_transport.hpp>
class OpenCVClass : public rclcpp::Node
{
  OpenCVClass() : Node (“opencv_test_node”)
  {
    subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(topic_name, _sub_queue, std::bind(&OpenCVClass::image_callback, this, std::placeholders::_1));
  }

  cv::namedWindow(“Display Window”);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat msg_image_matrix = cv_bridge::toCvCopy(msg, image_encoding)->image;
    // OpenCV Functions Go Here
  }
};

class ImageTransportClass : public rclcpp::Node
{
  ImageTransportClass(ros::Node* _nodeptr) : Node (“image_transport_node”)
  {
    it = image_transport::ImageTransport it(_nodeptr);
    sub = it.subscribe(topic_name, 1, _node->image_callback);
  }
  image_transport::Subscriber sub;
  image_transport::ImageTransport it;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto _node= std::make_shared<OpenCVClass>();
  ImageTransportClass itc = ImageTransportClass(_node);
  rclcpp::spin(_node);
  rclcpp::shutdown();
  return 0;
}

```

## OpenCV Image Processing
With OpenCV image processing you can identify features in an image, such as edges and objects, or transform the image as necessary. Here are a few relevant functions and quick code snippets of how to use them:

#### Resize
*void cv::resize( InputArray src, OutputArray dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR)*
The resize function resizes an image stored in an input Mat object and saves the newly sized image into the output Mat. See the following page for more details: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

```
cv::Mat src = image; // the src matrix should be nonempty
cv::Mat dst; // Destination matrix should be empty
int x_size = 900; // Desired horizontal size of the dst matrix in pixels
int y_size = 600; // Desired horizontal size of the dst matrix in pixels
cv::Size size = cv::Size(x_size, y_size);
cv::resize(src, dst, size);

```

#### Gaussian Blur
*void cv::GaussianBlur( InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT)*
The GaussianBlur function applies a Gaussian blur convolution across a matrix. (https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)

```
cv::Mat src, gauss;
src = image; // the src matrix should be nonempty
const int convolution_size = 5; // Size of the Gaussian kernel – must be positive and odd
const int sigma = 1; // Gaussian kernel standard deviation
cv::Size size = cv::Size(convolution_size, convolution_size);
cv::GaussianBlur(src, gauss, size, sigma);

```

#### Sobel Derivatives
*void cv::Sobel( InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)*
The Sobel function calculates the x- or y-axis gradient of a matrix to evaluate a directional derivative of an image. (https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)

```
cv::Mat src, gauss, gray, grad_x, grad_y;
src = image; // the src matrix should be nonempty
cv::GaussianBlur(src, gauss, cv::Size(3, 3), 0, 0); // Gaussian Blur
cv::cvtColor(gauss, gray, cv::COLOR_BGR2GRAY); // Grayscale
cv::Sobel(gray, grad_x, cv::CV_16S, 1, 0); // X-axis first-order derivative gradient
cv::Sobel(gray, grad_y, cv::CV_16S, 0, 1); // Y-axis first-order derivative gradient

```

## OpenCV Edge Detection

Edge Detection in OpenCV can be done via the Gaussian Laplace function or the Canny Edge Detector:

```
cv::Mat src, gauss, gray, laplace, edge_values;

const int kernel_size = 3;
const int scale = 1;
const int delta = 0;
const int depth = cv::CV_16S;
const int convolution_size = 3; // Size of the Gaussian kernel – must be positive and odd

src = image; // the src matrix should be nonempty

cv::Size size = cv::Size(convolution_size, convolution_size);
cv::GaussianBlur(src, gauss, size, 0, 0);
cv::cvtColor( gauss, gray, cv::COLOR_BGR2GRAY ); // Convert the image to grayscale
cv::Laplacian( gray, laplace, depth, kernel_size, scale, delta); // apply Gaussian Laplace kernel
cv::convertScaleAbs( laplace, edge_values); // convert to 8-bit – edge_values now contains the edge values in the mono8 data encoding

```

```
cv::Mat src, gray, gauss, edge_detection, edge_values;
cv::Mat src = image; // the src matrix should be nonempty
cv::cvtColor( src, gray, cv::COLOR_BGR2GRAY ); // convert to grayscale
const int convolution_size = 3; // Size of the Gaussian kernel – must be positive and odd
cv::Size size = cv::Size(convolution_size, convolution_size);
cv::GaussianBlur(gray, gauss, size, 0, 0);

int threshold_min = 100;
const int ratio = 2; // Recommended ratio is between 2 and 3
const int kernel_size = 3;
cv::Canny(edge_detections, edge_detections, threshold_min, threshold_min*ratio, kernel_size );	
// Canny Edge Detector
edge_values = cv::Scalar::all(0); // Initialize edge_values
src.copyTo( edge_values, edge_detection); // Copy values from src to edge_values, only where edge_detection is nonzero. The edge_values matrix matches the original bgr8 data encoding

```

## References 
https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
