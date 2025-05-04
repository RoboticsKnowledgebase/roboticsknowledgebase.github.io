---
date: 2025-05-04
title: ROS2 Humble Intra-Process Communication Bag Recorder
---

High-bandwidth sensors like cameras and LiDAR can easily overwhelm a ROS 2 system when recording data. Intra-process communication (IPC) in ROS 2 offers a solution by allowing zero-copy message passing between nodes in the same process, dramatically reducing CPU and memory overhead. This wiki entry introduces the [**ROS 2 Humble Intra-Process Communication Bag Recorder**](https://github.com/MRSD-Wheelchair/humble_ipc_rosbag) – a tool that enables efficient recording of data-heavy topics (e.g. high-resolution images and point clouds) on ROS 2 Humble. We’ll explain how ROS 2’s intra-process mechanism and composable nodes work, why the default `ros2 bag record` in Humble doesn’t use IPC, and how this custom recorder backports “zero-copy” recording to Humble. By the end, you’ll understand when and how to use this tool to reliably capture multiple camera streams or other large-data topics without dropping messages.

**Note:** This tool is specifically for **ROS 2 Humble Hawksbill (2022)**. Newer ROS 2 releases (starting with Jazzy) already include intra-process support in rosbag2, so Humble users benefit most from this solution.

## Table of Contents

* [Introduction](#introduction)
* [Background: Intra-Process Communication and Composable Nodes](#background-intra-process-communication-and-composable-nodes)
* [The Problem: Recording High-Bandwidth Data in ROS 2 Humble](#the-problem-recording-high-bandwidth-data-in-ros-2-humble)
* [Humble IPC Bag Recorder: Overview and Benefits](#humble-ipc-bag-recorder-overview-and-benefits)
* [Installation and Setup](#installation-and-setup)
* [Usage Guide: Intra-Process Rosbag Recording](#usage-guide-intra-process-rosbag-recording)

  * [Launching the Recorder via Launch Files](#launching-the-recorder-via-launch-files)
  * [Configuring Topics and Parameters](#configuring-topics-and-parameters)
  * [Starting and Stopping the Recording](#starting-and-stopping-the-recording)
  * [Validating the Recorded Bag](#validating-the-recorded-bag)
* [Example Use Case: Recording Multiple Cameras](#example-use-case-recording-multiple-cameras)
* [Summary](#summary)
* [See Also](#see-also)
* [Further Reading](#further-reading)
* [References](#references)

## Introduction

Robotic systems often need to log data from high-resolution cameras, 3D LiDARs, or depth sensors for debugging and analysis. However, recording these high-bandwidth topics in ROS 2 can strain the system. By default, ROS 2 nodes run in separate processes, meaning messages are passed via the middleware (DDS) between publisher and subscriber. For large messages (images, point clouds, etc.), this inter-process communication incurs significant serialization and copying overhead. As data rates climb into the hundreds of MB/s, a standard rosbag2 recorder may start dropping messages or consuming excessive CPU.

**Intra-Process Communication (IPC)** is a ROS 2 feature that bypasses the middleware when publisher and subscriber are in the same process. Instead of serializing data, ROS 2 can transfer a pointer to the message, enabling *zero-copy* or near zero-copy delivery. This drastically reduces CPU load and memory usage during message passing. In fact, node composition has been shown to save \~28% CPU and 33% RAM by avoiding duplicate message copies. For very large messages (on the order of 1 MB or more), intra-process messaging is essentially a necessity – such data flows are *“virtually impossible without node composition.”*

ROS 2 provides the concept of **Composable Nodes** to leverage IPC. By combining multiple nodes into a single process (a *composition*), messages exchanged between those nodes use the efficient intra-process mechanism. In ROS 1, a similar idea existed as nodelets; ROS 2 improves on this with a more robust design. Using composable nodes, we can place a data-producing node (e.g. a camera driver) and a data-consuming node (e.g. a recorder) in the same process, eliminating the expensive inter-process message copy. This wiki entry focuses on applying these techniques to rosbag recording on ROS 2 Humble.

We will first cover the background of ROS 2 IPC and composable nodes. Next, we’ll discuss why the default `ros2 bag record` in Humble does *not* utilize IPC (and the limitations that imposes). We then introduce the **Humble IPC Bag Recorder** tool which backports intra-process recording to Humble. Detailed usage instructions – from installation to runtime operation – are provided, including example launch files and commands. An example scenario of recording multiple camera streams demonstrates the benefits. Finally, we include a summary, related links, further reading, and references for deeper insight.

## Background: Intra-Process Communication and Composable Nodes

To understand the solution, it’s important to grasp how ROS 2 implements intra-process communication and what composable nodes are:

* **ROS 2 Intra-Process Communication (IPC):** In ROS 2, when a publisher and subscriber are in the same process and use compatible QoS settings, the middleware can be bypassed. The message data is stored in an in-memory buffer and a pointer or reference is shared with the subscriber, rather than serializing the data through DDS. This “zero-copy” transport means the data isn’t duplicated for the subscriber; both publisher and subscriber access the same memory (until it’s no longer needed). The result is much lower latency and CPU usage for message passing, since we avoid copy and serialization costs. In effect, IPC treats message exchange like a direct function call or shared memory queue inside one process, rather than sending over network interfaces. ROS 2’s IPC was significantly improved starting in Dashing and Eloquent (after early versions had limitations). With modern ROS 2, IPC supports `std::unique_ptr` message types to achieve zero-copy transfer in most cases. The trade-off is that publisher and subscriber must run in one process – which leads to the need for composed nodes.

* **Composable Nodes (Node Composition):** ROS 2 allows multiple nodes to be launched into a single OS process using the rclcpp composition API. A *Composable Node* is a node that can be loaded at runtime into a container process (an `rclcpp_components` container) instead of running as a standalone process. By composing nodes together, intra-process communication becomes possible among them. Composition is ROS 2’s answer to ROS 1’s nodelets, but it’s more flexible – any node that is written as a component (basically, exports itself as a plugin) can be dynamically loaded into a container. The ROS 2 launch system supports this via the `ComposableNodeContainer` and `ComposableNode` descriptions in Python launch files. When launching a component, you can pass an argument `extra_arguments=[{'use_intra_process_comms': True}]` to ensure intra-process communication is enabled for that node’s subscriptions. Composed nodes still appear as separate logical nodes (with their own names, topics, etc.), but physically they share the process. This eliminates the need to send messages via DDS for intra-process delivery. Composable nodes are especially beneficial for high-frequency or high-bandwidth data pipelines – a recent study showed composition **dramatically improves performance** for large messages like images and point clouds. It reduces latency and prevents redundant copies, which is crucial on resource-constrained systems or when running dozens of nodes.

In summary, intra-process communication in ROS 2 provides a way to avoid costly data copying by delivering messages within the same process, and composable nodes are the mechanism to colocate publishers and subscribers so that IPC can happen. By using these features, one can create a “zero-copy” pipeline for data-intensive tasks. Next, we look at how this applies to rosbag recording and why ROS 2 Humble’s default bagging tool doesn’t take advantage of it by default.

## The Problem: Recording High-Bandwidth Data in ROS 2 Humble

Recording topics in ROS 2 is done with the rosbag2 tool (`ros2 bag`). In ROS 2 Humble (and earlier releases), `ros2 bag record` runs as a standalone process that subscribes to the chosen topics and writes messages to storage (SQLite3 or MCAP). Because the recorder runs in its own process separate from the publishers, all data is transferred via *inter-process* communication. For each message published by, say, a camera driver, a copy must be sent through DDS to the recorder process. This introduces significant overhead for large messages:

* **CPU Overhead:** Each message may be serialized and copied in the publisher, then deserialized in the recorder, consuming CPU time. At high publish rates (e.g. 30 FPS video at 1080p), the serialization workload can saturate a core. The recorder itself must keep up with decoding and writing the data.
* **Memory Bandwidth:** Instead of sharing memory, the data travels through the loopback network or shared memory transport of DDS, effectively duplicating the data in RAM. Large images (several MB each) or point cloud scans will quickly chew through memory bandwidth when duplicated for recording.
* **Dropped Messages:** If the recorder cannot keep up with the incoming data stream, it will start to drop messages. Users often observe that after some seconds, frames begin to drop – the recorder falls behind as buffers overflow. In ROS 1, `rosbag` had tuning options like queue size, buffer size, and chunk size; ROS 2 rosbag2 has a `--max-cache-size` parameter to buffer more data, but ultimately if the throughput exceeds what inter-process transport+disk can handle, data will be lost.
* **Latency:** The additional hops through the middleware add latency. For live monitoring purposes (like visualizing a camera while recording), inter-process communication can introduce noticeable lag or jitter.

The net effect is that on ROS 2 Humble, recording high-bandwidth topics can be inefficient. As an example, consider two 640×480 cameras at 30 Hz: one user reported smooth viewing in RViz but as soon as they started a rosbag2 recording, frames began dropping after \~20 seconds, even though CPU and disk usage appeared under limits. The default recorder simply couldn’t sustain the required throughput with inter-process message handling.

**Why doesn’t ROS 2 Humble use IPC for rosbag2?** The capability to use intra-process communication in rosbag2 was not yet integrated into Humble’s stable release. The rosbag2 recorder and player in Humble run only as standalone processes. While ROS 2 did have the IPC mechanism available, the rosbag2 tool hadn’t been refactored to take advantage of it. In late 2021 and 2022, ROS 2 developers started addressing this via feature requests to make rosbag2’s Recorder and Player into *components* that could be launched in-process. This work landed in ROS 2’s development branch after Humble’s release. By the time of **ROS 2 Iron** (mid-2023) and the **ROS 2 Jazzy (2024)** release, rosbag2 gained the ability to run as a composable node. In Jazzy’s changelog, it’s noted that *“Player and Recorder are now exposed as rclcpp components”*, enabling zero-copy recording and reducing CPU load, and helping avoid data loss for high-bandwidth streams. In other words, the feature we need is available in newer ROS 2 distributions (and in nightly Rolling builds), but it’s absent from Humble by default.

For users sticking with the Humble Hawksbill LTS, this posed a challenge: how to efficiently record high-volume data without upgrading the entire ROS 2 distribution. This is where the **Humble IPC Bag Recorder** comes in – it fills that gap by bringing intra-process recording to Humble.

## Humble IPC Bag Recorder: Overview and Benefits

The **ROS2 Humble Intra-Process Communication Bag Recorder** (or *Humble IPC Bag Recorder* for short) is a custom tool that enables rosbag2 recording with intra-process communication on ROS 2 Humble. In essence, this tool provides a **composable recorder node** that you can run in the same process as your publishers, achieving the zero-copy data transfer that the stock Humble recorder lacks.

**Purpose:** This package was created as a backport of the rosbag2 composable recorder functionality. It allows Humble users to record topics without the inter-process overhead, by manually composing the recorder with the target publishing nodes. The motivation came from real-world needs to log camera and LiDAR data at full rate on Humble systems without losing data. Rather than waiting for the next ROS 2 release or attempting a risky partial upgrade, developers can integrate this recorder into their existing Humble workflows.

**Key Features and Advantages:**

* **Intra-Process (Zero-Copy) Recording:** The recorder node subscribes to topics via intra-process communication when co-located with the publishers. This minimizes CPU usage and virtually eliminates the extra memory copy for each message. High-frequency image streams can be recorded with far less overhead, reducing the chance of drops.
* **Backported Capability:** Provides in Humble what ROS 2 Iron/Jazzy offer out-of-the-box. Under the hood, it uses ROS 2’s rclcpp composition API similarly to how it’s done in newer releases, but packaged for Humble compatibility. (One early implementation of this idea was released as a standalone package `rosbag2_composable_recorder` for Galactic/Humble – this tool builds on that concept, tailored for Humble Hawksbill).
* **Same Bag Format:** The tool still writes standard rosbag2 files (SQLite3 by default, or MCAP if configured). The output is a normal `.db3` (or `.mcap`) bag that can be played back with `ros2 bag play` or analyzed with existing tools. The difference lies in how data arrives to the recorder, not in the file format.
* **Flexible Topic Selection:** It supports recording all topics or a specified list of topics, just like `ros2 bag`. You can configure which topics to record via parameters or launch arguments. This means you can target only the high-bandwidth topics with the IPC recorder if desired, and let the rest use standard methods.
* **Composability:** The recorder is designed to be launched as a component in a container. You have control over how to integrate it with your system. For example, you might launch multiple camera drivers and the recorder together in one process, or start your drivers first and then load the recorder into that process on-demand. Launch files are provided (and can be customized) to simplify these use cases.
* **Reduced Dropped Messages:** By avoiding the DDS hop, the recorder can keep up with the publishers more easily. Disk write speed remains a factor (you can still saturate the disk if you try to record *too much* data), but the elimination of serialization delays buys significant headroom. In tests with this approach, recording stability is greatly improved – e.g., users have reported the recorder keeping up with dual 30 Hz camera streams that previously showed periodic frame drops.
* **When to Use vs. Standard Recorder:** Use the Humble IPC recorder **when you have high-bandwidth or high-frequency topics** that the standard `ros2 bag` struggles with. If you’re recording video streams, depth images, large point clouds, or any scenario where CPU usage for bagging is high, this tool is likely beneficial. On the other hand, for low-bandwidth topics (odometry, small sensor readings, etc.) or quick debugging sessions, the default recorder might be sufficient and simpler (since it doesn’t require setting up a special launch). In summary, for serious data logging on Humble – especially multi-camera rigs or multi-sensor fusion – this tool can make the difference between a successful bag and a bag with gaps in data.

It’s worth noting that this solution is a **stopgap specific to Humble**. If you upgrade to ROS 2 Jazzy or later, you can achieve the same intra-process recording by using the built-in rosbag2 components (without this external package). In fact, the package’s maintainers suggest not using it on ROS 2 distributions where the feature is already native. But for those on Humble, the IPC bag recorder is a game-changer for data collection tasks.

## Installation and Setup

To get started with the Humble IPC Bag Recorder, you will need to build it from source in your Humble workspace. It’s assumed you have an existing ROS 2 Humble installation and a workspace for building packages (e.g. `~/ros2_ws` with a `src/` directory).

**Requirements:**

* ROS 2 Humble Hawksbill (Ubuntu 22.04 or equivalent platform with ROS 2 Humble).
* Developer tools to build ROS 2 packages (C++ compiler, colcon, etc.).
* The `rosbag2` core packages and `rclcpp_components` should be installed (if you have ROS 2 binaries installed, you have these). The recorder will link against rosbag2 libraries.
* This package’s source code, available via git.

**Installation Steps:**

1. **Clone the Repository:** Obtain the source code for the humble IPC recorder. (Replace `<your-repo>` with the actual repository if needed):

   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone https://github.com/<your-repo>/humble_ipc_rosbag.git
   ```

   This will create a package (e.g. named `humble_ipc_rosbag` or similar) in your workspace. It contains the recorder node implementation and example launch files.

2. **Install Dependencies:** Ensure all necessary ROS dependencies are available. You can use rosdep to install any missing ones:

   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

   This will check the package.xml for dependencies like `rclcpp`, `rosbag2`, etc. On a standard Humble installation, you likely already have them. If not, rosdep will use apt to install what's needed.

3. **Build the Package:** Use colcon to build your workspace:

   ```bash
   colcon build --symlink-install
   ```

   (Add `--parallel-workers N` if you want to limit parallel jobs, or `--cmake-args -DCMAKE_BUILD_TYPE=Release` for an optimized build.) If the build succeeds, you should have the recorder node plugin available in the workspace.

4. **Source the Setup:** Before using the recorder, source the new workspace overlay:

   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

   This makes ROS 2 aware of the new package and its executables/launch files.

After these steps, the Humble IPC recorder is ready to use. The package provides launch files to help you run the recorder node in various ways, as described below.

## Usage Guide: Intra-Process Rosbag Recording

Using the IPC bag recorder involves launching the recorder node in the same process as the target publisher nodes. There are a couple of ways to do this, ranging from one-shot launch files that start everything together, to loading the recorder into an already running container. We’ll go through the provided methods and general tips for usage.

### Launching the Recorder via Launch Files

The package comes with two example launch files – let’s call them `simple.launch.py` and `load.launch.py` – to facilitate common use cases.

* **`simple.launch.py`:** This launch file is designed to start up both your sensor nodes *and* the recorder in one go, within a single process. It’s particularly geared towards launching one or more ZED camera nodes alongside the recorder. You can adapt it to other camera types or nodes if they support composition. The launch will create a container (an instance of `rclcpp_components` container) and then load each camera driver component and the recorder component into it.

  For example, if you have a ZED camera, you might use:

  ```bash
  ros2 launch humble_ipc_rosbag simple.launch.py \
      cam_names:=[zed_front] cam_models:=[zed2] cam_serials:=[<your_serial>]
  ```

  In this example, `cam_names`, `cam_models`, and `cam_serials` are arguments the launch file expects (specifically for ZED cameras) to identify and configure the camera nodes. Under the hood, `simple.launch.py` will:

  1. Start a composable container (if not already running).
  2. Load the ZED camera driver component (with `use_intra_process_comms=True`) into that container.
  3. Load the IPC recorder component into the same container, configured to immediately start recording all topics (or specific topics you set).

  The result is that the ZED camera’s topics (e.g. `/zed_front/zed_node/left/image_raw`, `/.../point_cloud/cloud_registered`) are being published in the same process where the recorder is subscribed. All image frames and point cloud messages are delivered via intra-process to the recorder, greatly reducing the load. This is a convenient way to launch if you want to begin recording as soon as the sensors start. Once you Ctrl-C this launch, recording stops and the bag file will be finalized.

* **`load.launch.py`:** This launch is used to **add** the recorder into an *already running* container process. It’s useful in scenarios where you might want to start the sensors first (perhaps letting them warm up or stabilize) and then begin recording at a specific moment. With `load.launch.py`, the assumption is you have a container running elsewhere (for example, maybe you launched a multi-camera container using another launch file or command). You will pass the name of that container into `load.launch.py`, which will then load the recorder node into that existing process.

  Typical usage might look like:

  ```bash
  # In one terminal, start your sensor container (e.g., using the vendor's launch file for multiple cameras)
  ros2 launch zed_multi_camera zed_multi_camera.launch.py

  # In another terminal, list running component containers to get the container name
  ros2 component list
  # Suppose it outputs something like "/zed_multi_camera/zed_container"

  # Now launch the recorder into that container
  ros2 launch humble_ipc_rosbag load.launch.py container_name:=/zed_multi_camera/zed_container
  ```

  The `load.launch.py` will find the container by name and use the ROS 2 composition service to load the recorder component into it. Once loaded, the recorder (if configured to start immediately) will begin subscribing and recording the topics from the sensors in that process.

  One advantage of this method is control: you can start and stop the recorder without shutting down the sensors. For instance, to stop recording, you could unload the recorder component:

  ```bash
  # Find the component ID of the recorder in the container
  ros2 component list 
  # (this will show something like "/zed_multi_camera/zed_container : <ID> : humble_ipc_rosbag/Recorder")
  ros2 component unload /zed_multi_camera/zed_container <recorder_component_id>
  ```

  Unloading the component stops the recording (and closes the bag file properly). You could later load it again to start a new bag. *(Alternatively, you could leave the recorder running but use its service interface to start/stop recording; more on that shortly.)*

Both provided launch files can be customized. If you are using a different camera or sensor, you can edit them or write a new launch file. The key is that you include:

* A `ComposableNodeContainer` (if one isn’t already running).
* Your publisher nodes as `ComposableNode` entries (with intra-process enabled).
* The recorder as a `ComposableNode` (with appropriate parameters).

**Example (conceptual):** If we wanted to compose an Intel RealSense camera node (if it supports composition) with the recorder, a launch might look like:

```python
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        output='screen'
    )
    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        name='realsense',
        namespace='camera',
        parameters=[{'enable_pointcloud': True}],
        extra_arguments=[{'use_intra_process_comms': True}]
    )
    recorder_node = ComposableNode(
        package='humble_ipc_rosbag',
        plugin='humble_ipc_rosbag::RecorderNode',  # hypothetical class name
        name='ipc_recorder',
        parameters=[{'record_all_topics': True}],
        extra_arguments=[{'use_intra_process_comms': True}]
    )
    load_components = LoadComposableNodes(
        target_container=container,
        composable_node_descriptions=[realsense_node, recorder_node]
    )
    return LaunchDescription([container, load_components])
```

In this hypothetical snippet, we launch a container, then load a RealSense camera component and the IPC recorder component into it. All topics published by the RealSense driver (e.g. `/camera/color/image_raw`, `/camera/depth/points`) would be recorded via intra-process. This illustrates how one could set up their own launch if needed.

### Configuring Topics and Parameters

By default, the recorder can be set to record all topics or a specified list. The configuration is done through ROS 2 parameters on the recorder node. Some important parameters include:

* **`record_all` / `all_topics`:** Boolean to indicate whether to record all available topics. If `True`, the recorder will subscribe to every topic it discovers (except perhaps some excluded internal topics). If `False`, you should provide a list of specific topics.
* **`topics`:** An explicit list of topic names to record. Use this if you only want to record certain topics. For example, you might list `/camera1/image_raw`, `/camera2/image_raw` and ignore everything else.
* **`bag_file` / `bag_name`:** The output file name or prefix for the bag. If not set, a default name (like `rosbag2_<timestamp>`) will be used. You can set this to organize your bag files (e.g. `experiment1.bag`).
* **`storage_id`:** Storage plugin to use (`sqlite3` by default, or `mcap` for the newer format). Humble defaults to SQLite3; you can use MCAP for better performance on large recordings.
* **`max_cache_size`:** Size in bytes of the cache before writing to disk. By default, rosbag2 may cache 100 MB or so. Increasing this can help smooth bursts of data (at the cost of memory). Setting it to 0 means no caching (write every message immediately). For high-speed recording, a cache is usually beneficial, so leaving this as default or even raising it might help.
* **`start_paused` / `start_recording_immediately`:** Some implementations allow starting in a paused state, requiring a service call or user input to actually begin recording. In our Humble recorder, the launch files by default set the recorder to start right away (to mimic `ros2 bag record` behavior). If you prefer to manually trigger start, you could configure this parameter and then call a ROS service or use a provided script to start.
* **`topics_ignore` / `exclude`:** If available, you could specify topics to exclude (this is analogous to `ros2 bag record --exclude` option in newer versions). For example, you might exclude `/tf` or other high-frequency but unneeded topics.

These parameters can be set via a YAML config file passed to the launch, or directly in the launch file as shown in the example. The package likely includes a default `params.yaml` with documentation on each setting. For instance, you might find in `config/params.yaml`:

```yaml
humble_ipc_rosbag:
  ros__parameters:
    record_all_topics: true
    bag_file: ""
    storage_id: "sqlite3"
    max_cache_size: 100000000  # 100 MB
    # topics: ["/camera1/image_raw", "/camera2/image_raw"]  # example if not record_all
```

**Selecting Topics to Record:** If you only care about certain topics (to save space or bandwidth), set `record_all_topics: false` and list those topics. Just ensure those topics will be published in the same process. If you accidentally omit a needed topic, the recorder won’t subscribe to it. Conversely, if `record_all_topics` is true, the recorder will latch onto every topic it sees – which could include some you don’t need. In a multi-camera container that might be fine (you likely want all camera topics). But if you have other nodes in that container (e.g. some intermediate processing nodes), you might record internal topics unnecessarily. Tailor the config to your use case.

### Starting and Stopping the Recording

When using the provided launch files, recording begins as described (immediately on launch for `simple.launch.py`, or immediately on loading via `load.launch.py` unless configured otherwise). Stopping the recording can be done in a few ways:

* **Graceful Shutdown:** If you launched everything together, hitting Ctrl-C will stop the launch, which in turn will shutdown the recorder node. Rosbag2 recorder is designed to close the bag file properly on shutdown (writing metadata, etc.), so the resulting bag should be usable. Always ensure the process actually terminates so that the bag is finalized (if a node hangs for some reason, you might need to Ctrl-C twice or kill, which could leave an incomplete bag).

* **Unloading the Component:** As shown earlier, if you loaded the recorder into an existing container, you can unload it without killing the whole process. The `ros2 component unload` command is the clean way to remove the recorder. This triggers its shutdown routines just like Ctrl-C would. After unloading, the container and other nodes keep running, so you could later load the recorder again to start a new recording session. When unloading, you refer to the container and the component ID (which you can get via `ros2 component list`). For example:

  ```bash
  ros2 component unload /camera_container 1
  ```

  (if the recorder was listed with ID 1 in that container).

* **Service Calls (if supported):** The IPC recorder node may provide services like `/start_recording` and `/stop_recording` (this was the case in earlier implementations). If available, you could start the recorder in a paused state (parameter `start_paused: true` or similar) and then call:

  ```bash
  ros2 service call /start_recording std_srvs/srv/Trigger "{}"
  ```

  to begin recording, and later call `/stop_recording` to stop without unloading. This approach requires that the node advertise those services. Check the package documentation or running `ros2 interface list | grep record` to see if such services exist. If they do, this can be a very flexible way to control recording programmatically (for instance, start recording only when a certain event happens, via a ROS service call from another node).

* **Keyboard Controls:** Newer rosbag2 versions have keyboard controls (space to pause, etc.) – in this Humble tool, since it’s running as a component, you typically won’t have an interactive console in the same way. So rely on the above methods for control.

Regardless of how you stop, once the recorder node stops, the bag file is closed. It will typically be created in the directory from where you launched the command (unless you specified an output path).

### Validating the Recorded Bag

After you have recorded data using the IPC recorder, you should verify that everything was captured as expected. Since the output is a standard rosbag2, you can use the usual ROS 2 bag utilities for validation:

* **Inspect with ros2 bag info:** This will show you what topics were recorded, the number of messages, frequency, duration, etc. For example:

  ```bash
  ros2 bag info experiment1
  ```

  You should see all the topics you intended to record. Pay special attention to the message count and frequency. If you recorded a camera at 30 Hz for 60 seconds, you expect around 1800 messages. If the count is significantly lower, it could indicate dropped frames. Ideally, with the IPC recorder, the count will match the publish count (aside from perhaps the first few frames during startup or last few at shutdown).

* **Play back or visualize:** You can do a quick playback test:

  ```bash
  ros2 bag play experiment1
  ```

  Perhaps run rqt or RViz to visualize the images/points from the bag to ensure they look okay. This is more of a sanity check that the data is not corrupted and that timestamps are correct. Because intra-process affects only how data gets to the recorder, the recorded data itself should be identical to what a normal recorder would have stored (only with fewer drops). The bag file structure is the same, so playback should work normally.

* **Check metadata and performance:** The `ros2 bag info` will also tell you the storage format and any splitting of bag files. If your recording was long or large, rosbag2 might split the bag into multiple files (e.g., `experiment1_0.db3`, `experiment1_1.db3`, etc.). Ensure all expected splits are present. You can also open the metadata YAML (`experiment1/metadata.yaml`) to see if end\_time matches your recording end, etc.

If you encounter an empty bag or missing data:

* Ensure that the recorder actually started (if using services to start, maybe it never got triggered).
* Double-check that the topics you wanted were in the same container. If a topic was published in a different process than the recorder, the recorder would not see it at all (since in our usage we likely disabled discovery from outside to focus on intra-process, or simply never loaded a subscription for it). The symptom would be 0 messages recorded for that topic or it not appearing at all.
* Look at the console output during recording. The recorder might log warnings if it cannot subscribe to a topic or if something goes wrong. For instance, if you accidentally had two recorder nodes or a leftover one, you might see errors about failing to write.
* If using the `topics` parameter list, verify the names exactly match (including namespaces) of the publishers.

In practice, once set up, the Humble IPC recorder tends to “just work” and you’ll immediately notice the lower CPU usage. It’s good to confirm that by observing system metrics: CPU should be much lower compared to the standard recorder scenario, since no expensive serialization is happening in the recording path. This improved efficiency directly translates to more reliable data capture.

## Example Use Case: Recording Multiple Cameras

To illustrate the value of the Humble IPC Bag Recorder, let’s consider a concrete scenario: **recording multiple high-resolution cameras simultaneously on ROS 2 Humble**. Suppose a robot is equipped with two 1080p RGB cameras (say, Intel RealSense or webcams) publishing at 30 frames per second, and also a depth camera or LiDAR producing large point clouds. Recording all of these streams is challenging:

In a traditional setup (using `ros2 bag record` in a separate process), each image (which could be \~1920×1080×3 bytes ≈ 6 MB per frame uncompressed) has to be sent to the recorder process. At 30 FPS, that’s \~180 MB/s per camera of raw data. Two cameras double that to 360 MB/s, not even counting point clouds. Even if images are compressed, the data rate is still significant. The CPU would spend a lot of time just handling these images – packaging them into DDS messages and copying them across processes. It’s no surprise that frames would drop: the recorder simply can’t keep up, especially on modest hardware (e.g., a single-board computer on a robot).

Now, using the IPC recorder approach:

* We launch both camera drivers and the recorder in one process using composition. Each camera’s driver publishes images, which the recorder subscribes to via a shared memory pointer. There’s **no serialization** of the image data for transport. Instead, the image memory is handed off to the recorder directly inside the process.
* The CPU usage plummets compared to before. Instead of two processes each handling 6 MB × 30 FPS of copy, those copies are eliminated. The system mainly needs to handle the disk writing. As long as the disk (or SSD) can sustain the write throughput (which 360 MB/s might still be high but perhaps using compression or downsampling can mitigate), the recorder will not be the bottleneck.
* The benefit scales with number of cameras: if you had 4 cameras, the savings are even more dramatic. In fact, a ROS 2 developer mentioned being able to record data from **8 HD cameras simultaneously (over 2.6 GB/s)** by splitting loads and presumably using techniques like this in a distributed fashion. While 2.6 GB/s is extreme, the principle stands – removing inter-process overhead is essential to hit those levels.

With our two-camera example, after switching to the Humble IPC recorder, you’d likely observe that the frame drops disappear or are vastly reduced. The recorded bag will have close to 30 Hz for each camera consistently. The **latency** from capture to disk is also reduced, meaning if you were timestamping or triggering events based on frame capture, the recorded timestamps more accurately reflect the real timing (no big delays due to queuing in DDS). In mission-critical logging (e.g. recording all sensor data during a test run of an autonomous vehicle or robot), this fidelity is crucial.

Another example: the **Stereolabs ZED cameras** are stereo + depth cameras that produce multiple high-bandwidth topics (left/right images, depth map or point cloud, etc.). Stereolabs provides a composition-based launch for multiple ZEDs specifically to leverage IPC. By adding the IPC recorder into that same process (as our provided launch does), you can record all ZED outputs with minimal overhead. This approach *“achieves zero-copy data transfer for efficient data exchange”* and *“enhances performance and reduces resource consumption by minimizing data copying”*. Essentially, the heavy data published by each camera does not bounce between processes, making multi-camera recording on one machine feasible.

In summary, the IPC recorder shines in use cases with **multiple high-data-rate sources**. Whether it’s an array of cameras, a combination of cameras and LiDAR, or any sensor setup pushing lots of bytes, running the recorder in-process ensures that recording no longer becomes the weakest link. You can fully exercise your sensors at max resolution and frame rate, confident that the logger can keep up (within the limits of your disk write speed). This allows robotics developers to gather richer datasets on ROS 2 Humble, which otherwise might have required upgrading to a newer ROS 2 or using custom hacky solutions.

## Summary

Recording large volumes of data in ROS 2 Humble is greatly improved by leveraging intra-process communication. **ROS 2’s IPC and node composition** allow publishers and subscribers to exchange messages without going through the middleware, avoiding redundant copies and serialization. The default rosbag2 in Humble, however, doesn’t utilize this, which can lead to dropped messages and high overhead for high-bandwidth topics.

The **Humble IPC Bag Recorder** addresses this gap by providing a composable recorder node for ROS 2 Humble. By running the recorder in the same process as the data producers, it achieves near zero-copy recording. This results in significantly lower CPU usage and more reliable logging of topics like HD images and point clouds. We discussed how to set up and use this tool: you clone and build the package in a Humble workspace, then either launch your sensors and recorder together in one container or load the recorder into an existing container when needed. You can configure it to record all or specific topics, and control the start/stop as needed. The recorded bag files are in the standard format and can be used like any other.

**When to use this tool:** If you’re running ROS 2 Humble and need to record high-bandwidth data streams, the IPC recorder is likely the right choice. It’s particularly useful for multi-sensor systems (multiple cameras, etc.) where the standard recorder can’t keep up. It essentially brings Humble up to feature-parity (for recording) with newer ROS 2 versions, without requiring an upgrade. On the other hand, if you have already moved to ROS 2 Iron or beyond, you can use the built-in rosbag2 component capability instead (and this tool isn’t necessary). For low-data topics, the default recorder might suffice, but using the IPC recorder has little downside besides the extra setup.

By adopting intra-process communication for rosbag recording, ROS 2 users can capture datasets that were previously very difficult to obtain on Humble. This empowers better testing, debugging, and data analysis for robots with rich sensor suites. In robotics, data is king – and with the right tools, you won’t have to compromise on collecting it. The Humble IPC Bag Recorder is one such tool, enabling robust data recording on ROS 2 Humble Hawksbill.

## See Also

- [Building ROS2 Custom Packages](/wiki/common-platforms/ros/ros2-custom-package)
- [ROS Introduction](/wiki/common-platforms/ros/ros-intro)
- [Building an iOS App for ROS2 Integration – A Step-by-Step Guide](/wiki/common-platforms/ros2_ios_app_with_swift)

## Further Reading

- [ROS 2 composition and ZED ROS 2 Wrapper](https://www.stereolabs.com/docs/ros2/ros2-composition)
- [rosbg2_composable_recorder](https://github.com/berndpfrommer/rosbag2\_composable\_recorder)

## References

\[1] Yu-Hsin Chan and Haoyang He, “*humble_ipc_rosbag*,” GitHub repository, 2025. Accessed: May 4, 2025. [Online]. Available: [https://github.com/MRSD-Wheelchair/humble_ipc_rosbag](https://github.com/MRSD-Wheelchair/humble_ipc_rosbag)

\[2] Open Robotics, “*Intra-Process Communications in ROS 2*,” *ROS 2 Design Articles*, 2020. Accessed: May 4, 2025. \[Online]. Available: [https://design.ros2.org/articles/intraprocess_communications.html](https://design.ros2.org/articles/intraprocess_communications.html)

\[3] Open Robotics, “*ROS 2 Jazzy Jalisco Release Highlights: rosbag2 Components*,” *ROS 2 Documentation*, 2024. Accessed: May 4, 2025. \[Online]. Available: [https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html#rosbag2](https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html#rosbag2)

\[4] ROS 2 Tutorial, “*Recording and Playing Back Data (Humble)*,” *docs.ros.org*, 2022. Accessed: May 4, 2025. \[Online]. Available: [https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)
