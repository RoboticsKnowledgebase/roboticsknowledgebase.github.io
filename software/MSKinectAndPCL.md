Getting Started with MS Kinect and PCL

MS Kinect Short Overview:
Kinect for Xbox 360 is a low-cost vision device equipped with one IR camera, one color camera and one IR projector to produce RGB images as well as voxel (depth-pixel) images. The RGB video stream gives a 8-bit VGA resolution (640 x 480 pixels) with a Bayer color filter, while th monochrome depth sensin vidoe stream is in VGA resolution. The sensor has an angular field of view of 57 degrees horizontally and 43 degrees vertically. Kinect has been reverse engineered to a great extent by the open soure community which has revealed many facts on how depth is measured. Kinect uses a structured light approach form which we can extract time of return. They use a standard off-the shellf CMOS sensor for the same.

Libraries for MS Kinect Interfacing in Ubuntu
There are many different open source libraries that can be chosen from for interfacing with Kinect in Ubuntu. Kinect for windows provides direct interfacing for a windows based computers. The libraries for Ubuntu are:
1. OpenNI and OpenNI2
The older version of Kinect supports openNI whereas the newer version of Kinect uses openNI2. The installation and usaage of openNI2 as as a standalone library can be found here:
http://structure.io/openni
2. libfreenect and libfreenect2

libfreenect is also a reliable librarys that can be used and in my experience have proven to be more reliable than openNI. The only drawback is that while openNI and openNI2 can be used for other sensors such as Asus Xtion Pro or Orbecc Astra depth camera, libfreenect is mostly suited for Kinct only.
This github link provides information on getting started with libfreenect
https://github.com/OpenKinect/libfreenect
The full distribution of ROS also includes these libraries by default and the documentation for that can be found here
General libfreenect:
http://wiki.ros.org/libfreenect
freenect_launch to access depth data stream which can be visualized on RVIZ
http://wiki.ros.org/freenect_launch
openni_launch: Simiar to libfreenect_launch
http://wiki.ros.org/openni_launch


PCL Short Overview:
PCL is a large scale open source library for processing 2D and 3D images and point cloud processing. It is a state of the art library used in most perception related projects. PCL has an extensive documentation and ready to use examples for segmentation, recognition and filtering. It has data structures for kdtree, octree and pointcloud arrays which allows for very fast processing and efficient implementation. A complete documentation for PCL can be found on their official website
http://pointclouds.org/documentation/
Using VocelGrid filter pointcloud can be initially downsampled and further sparsification of Kinect data can be ddone using PassThrough filters. After these basic filtering, you can further performing clustering and cylinderical or planar segmentation in real time applications. The PCL tutorials are helpfule even if you are an expert and can be found here.
http://pointclouds.org/documentation/tutorials/index.php#filtering-tutorial


Installing PCL from source and ros_pcl:
Although PCL comes installed with ROS full installation by default, a complete installation of PCL from source mught be required in some circumstances alongwith CUDA for GPU processing. The following link provides an exhaustive tutorial for the same and also shows how to install openNI.
http://robotica.unileon.es/mediawiki/index.php/PCL/OpenNI_tutorial_1:_Installing_and_testing


References
A complete guide for using MS Kinect for navigation can be found in this Masters thesis
https://www.nada.kth.se/utbildning/grukth/exjobb/rapportlistor/2011/rapporter11/mojtahedzadeh_rasoul_11107.pdf
