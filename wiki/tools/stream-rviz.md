---
title: Stream Rviz Visualizations as Images
---

This tutorial will help you stream live Rviz visualizations as `Image` topics in ROS. The tutorial will cover how one can setup multiple virtual cameras and virtual lighting within the Rviz environment. Streaming Rviz visualizations has several applications such as, one can record multiple views simultaneously for off-line analysis or these image streams can be incorporated into Python or web-based GUIs.

## Contents
1. [Setting up Virtual Cameras in Rviz](http://roboticsknowledgebase.com/wiki/tools/stream-rviz/setting-up-virtual-cameras-in-rviz)
2. [Setting up Virtual Lighting in Rviz](http://roboticsknowledgebase.com/wiki/tools/stream-rviz/setting-up-virtual-lighting-in-rviz)
3. [Compressing Image Streams](http://roboticsknowledgebase.com/wiki/tools/stream-rviz/compressing-image-streams)

## Setting up Virtual Cameras in Rviz

First, we begin with setting up virtual cameras in Rviz. As per convention, for any given camera frame, the positive Z-axis points in the direction of camera view.

1. Download the Rviz camera stream plug-in ROS package into the `src` directory of your workspace.

```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/lucasw/rviz_camera_stream
cd ..
catkin_make
```

2. Now, we first need to setup a camera with its extrinsics and intrinsics properties. We can use `static_transform_publisher` to setup the camera extrinsics with respect to some world frame (`map` here) and publish a the `camera_info` topic which will have the intrinsic properties of the camera along with the width and the height of the image we want to publish. Refer to the documentation of `camera_info` message [here](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html) for more details on setting up the parameters. The following example is for a camera with image size `640x480` with focal length `500.0`, principal point at `(320, 240)`, equal aspect ration, and rectified image without distortions.

You can add the following block of code in your launch file.

```
<group ns="camera1">
  <node pkg="tf" type="static_transform_publisher" name="camera_broadcaster"
      args="-15 0 15 1.57 3.14 1.1 map camera1 10" />
  <node name="camera_info" pkg="rostopic" type="rostopic"
      args="pub camera_info sensor_msgs/CameraInfo
     '{header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: 'camera1'},
      height: 480, width: 640, distortion_model: 'plumb_bob',
      D: [0],
      K: [500.0, 0.0, 320, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
      R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      P: [500.0, 0.0, 320, 0.0, 0.0, 500, 240, 0.0, 0.0, 0.0, 1.0, 0.0],
      binning_x: 0, binning_y: 0,
      roi: {x_offset: 0, y_offset: 0, height: 480, width: 640, do_rectify: false}}' -r 2"
      output="screen"/>
</group>
```

To add multiple cameras, just replicate the code above (the `group` block), replacing the camera name everytime.

3. Next, open up Rviz by running `rviz` is your terminal and click on `Add` to add a new display type. Select the `CameraPub` plug-in from the list. Now in the display tab, under the plug-in properties, select the camera name (`camera1` in our case) and enter a new topic name for the image to be published.

4. That's it! Now you can use `image_view` node or add a new `Image` in Rviz itself to visualize the image stream. 

```
rosrun image_view image_view image:=/your/topic/name
```

It might be tricky to setup the extrinsics of the camera and position it properly at first. I would first recommend to comment out the `static_transform_publisher` block in your launch file and manually set up the transform on the fly to position the camera. The `tf_publisher_gui` node from `uos_tools` package come in handy.

Launch the TF publisher GUI as follows.

```
rosrun tf_publisher_gui tf_publisher_gui _parent_frame:=/map _child_frame:=/camera1
```

This should open a GUI with sliders which help you set the translation and rotations for you camera. Once you are satisfied with the setup, plug these values back into your original launch file and publish the static transform.

## Setting up Virtual Lighting in Rviz

By default, Rviz has a single spot light focused on whatever the user is currently viewing in the Rviz GUI. For example if you are viewing a robot model from the front view, the URDF mesh would be well light. However, if we want to simultaneously view the same robot model from the top or the side view, the URDF mesh would have shadows on it and will not be well lit. This is important in our case, because we aim to simultaneously view the Rviz environment from multiple views and would like to visualize our environment without shadows.

`rviz_lighting` is an RViz plug-in that provides ambient, directional, point, and spot lights. These lights can be used to give an RViz scene a more interesting lighting environment. They can be attached to tf frames, mimicking lights attached to moving robot components.

1. Download the Rviz lighting ROS package into the `src` directory of your workspace.

```
cd catkin_ws/src
git clone https://github.com/mogumbo/rviz_lighting
cd ..
catkin_make
```

2. To setup this up in Rviz, follow the similar step as earlier to add a new display type in Rviz by selecting `AmbientLight`. For our purpose, we will just use the `Directional` light and set the frame the same as on of our camera frames (`camera1`) under the properties of the plug-in.

## Compressing Image Streams

The image streams published from the camera publisher plug-in is in raw format, or in other words, these are bitmap images streams, which consume more memory. This section covers how to compress these image streams to reduce memory usage and make streaming more efficient. This is an optional step and usually this maybe beneficial only if you have multiple camera publishers from Rviz.

If you want to integrate these image stream with `roslibjs`, the ROS socket bridge expects images in compressed format and this step would be mandatory (refer [See Also](http://roboticsknowledgebase.com/wiki/tools/stream-rviz/see-also)).

To compress images, just add the following block into your launch file.

```
<node
    pkg="image_transport"
    type="republish"
    name="republish_image"
    output="screen"
    args="raw in:=/camera1/image
          compressed out:=/camera1/compressed/image">
</node>
```

You can now subscribed to the compressed image topic in your ROS nodes. Note that the message type will now be [`sensor_msgs/CompressedImage`](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CompressedImage.html).

## See Also
- A [tutorial](http://roboticsknowledgebase.com/wiki/tools/roslibjs) on ROS JavaScript library used to develop web-based GUIs which integrates image streams from Rviz.

## Further Reading
- Refer to the [`delta_viz`](https://github.com/deltaautonomy/delta_viz) repository developed by Delta Autonomy, which is a web-based dashboard GUI that implements all of what's covered in this tutorial.

## References
- https://github.com/lucasw/rviz_camera_stream
- https://github.com/mogumbo/rviz_lighting
- http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html
- http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CompressedImage.html
- https://github.com/deltaautonomy/delta_viz
