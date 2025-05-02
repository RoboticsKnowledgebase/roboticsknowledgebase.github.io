---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-04-27 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Robotics with the Microsoft Hololens2
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
This robot wiki entry revolves around Augmented Reality (AR) headset development, specifically the Microsoft HoloLens2. For most robotics related applications, one might want to capitalize on the perception capabilities of the device by accessing data from the [onboard sensors](https://learn.microsoft.com/en-us/hololens/hololens2-hardware), namely the RGB camera, 4 Greyscale cameras, 1 Time-of-Flight depth sensor and 1 IMU sensor. 

Moreover, as the headset has a limited amount of onboard compute, it’s useful to have an Application Programming Interface (API) to get this information from the HoloLens2 with a Python backend that extracts this data onto the offboard system with communication between the HoloLens and offboard system being completely wireless. This article in the Robot wiki is about how to use a Unity-Python API to extract information from the HoloLens2 to access sensor data, crucial to any downstream computer vision application. The contents present a basic overview of the headset, including details about its hardware, and a script that allows for sensor access and measures latency of transmission over a private wifi network. 

## Sensor Access and Deep Learning model deployment on the Microsoft HoloLens2

In most robotic applications, it helps to have perception modules on top of the dynamic actor in an environment in order to prevent line-of-sight occlusions between cameras mounted at fixed locations and the object of interest. An under-explored area of study in past MRSD Projects has been in using **Augmented Reality** (AR) headsets as dynamic sensors in order to sense the environment or object tracking while eliminating line-of-slight occlusions. The Microsoft HoloLens2 is a sophisticated piece of hardware that (datasheet) has been around since 2017 and uses the Microsoft Azure-Kinect-Sensor SDK (reference) that is wrapped around by the HoloLens2forCV (reference) GitHub repository that is maintained by Microsoft. This goes to show the merit in having this product out for so long is that there’s a lot of open-source support and mature user development forums that aid in debugging and in obtaining boilerplate code.


Briefly, the HoloLens2 can transmit the following data offboard at mentioned resolution and frequency. Knowing this ahead of time allows one to know if the HoloLens2 will be able to adhere to the sensing and latency requirements of the task at hand.

### HoloLens 2 Onboard Sensor Data Specifications:

The Microsoft HoloLens 2 provides access to a rich set of onboard sensor data, crucial for various robotics and perception tasks. Here's a breakdown of the available data streams and their specifications:

* **RGB Camera Feed:** Color images with a resolution of **640 x 360 pixels** captured at a rate of **40 FPS**.
* **Depth Map Data:** Depth information represented as images with a resolution of **360 x 288 pixels**, acquired at **40 FPS**.
* **Inertial Measurement Unit (IMU) Data:** High-frequency readings at **93 Hz**, providing detailed information about the device's motion and orientation.
* **Greyscale Camera Feeds (from four separate cameras):** Images in grayscale format with a resolution of **640 x 480 pixels**, captured at **40 FPS** from each of the four cameras.


## Experiments and Environment Setup

In our initial exploration with the Microsoft HoloLens 2, we utilized the [HL2SS](https://github.com/jdibenes/hl2ss) wrapper, which builds upon the [HoloLens2forCV](https://github.com/microsoft/HoloLens2ForCV) repository. This wrapper provided a set of fundamental scripts for extracting data from individual sensors. The subsequent sections of this guide will detail the process of developing custom scripts to achieve time-synchronized data acquisition from multiple sensors within a single script. Furthermore, we will outline the necessary steps to establish a complete development environment, including all required dependencies, to facilitate the execution of deep learning models. As a practical demonstration, we will showcase an example of accessing sensor data and meticulously measuring the latency involved in its transmission to an offboard system.

### Setting Up the Development Environment

Establishing a well-configured development environment with the precise versions of all required libraries is of paramount importance. Failure to adhere to these specifications can often lead to intricate and challenging dependency-related issues, potentially hindering the development process significantly. To mitigate these risks effectively, we strongly recommend installing the following specific versions of the necessary software libraries:

* **mmcv-full:** Ensure version `1.6.1` is installed for robust computer vision functionalities.
* **mmdet:** Utilize version `2.25.1` for object detection and instance segmentation tasks.
* **mmengine:** Install a version of `mmengine` that maintains compatibility with `mmdet` version `2.25.1` to ensure seamless integration.
* **numpy:** Employ version `1.26.4` for efficient numerical computations.
* **open3d:** Use version `0.18.0` for 3D data processing and visualization.
* **opencv-python:** Install version `4.10.0.84` for a wide range of computer vision algorithms.
* **torch:** Opt for version `1.12.1+cu116`, specifically compiled to leverage CUDA `11.6` for accelerated tensor computations on compatible NVIDIA GPUs.
* **torchaudio:** Install version `0.12.1+cu116` for audio processing tasks, also compiled with CUDA `11.6`.
* **torchvision:** Use version `0.13.1+cu116`, the vision library companion to PyTorch, compiled for CUDA `11.6`.
* **Python:** Ensure that Python version `3.9.12` is installed as the core programming language.
* **CUDA Version:** The underlying CUDA (Compute Unified Device Architecture) version on your system should be `12.3` to ensure optimal compatibility with the compiled PyTorch and related libraries.

### Important mmdet Library Modification

In addition to installing the specified library versions, it's crucial to make a minor adjustment within the `mmdet` library to ensure compatibility and proper functionality. Locate the following import statement:

```python
from mmdet.registry import DATASETS
```

This line needs to be replaced with the following corrected import statement:

```python
from mmdet.datasets import DATASETS
```


### Testing HoloLens Latency Metrics
With the environment correctly set up, you can now utilize the following Python script to evaluate the latency characteristics of data transmission from the HoloLens 2. This script establishes connections to the Personal Video (RGB) and [Research Mode](arXiv:2008.11239) Depth Long Throw cameras, captures synchronized frames, and calculates various performance metrics, including end-to-end latency, frame delivery time, and processing time.

```python
import multiprocessing as mp
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_utilities
import numpy as np
import hl2ss_3dcv
import time

# HoloLens address
host = "172.26.62.226"

# Camera parameters for Personal Video
pv_width = 640
pv_height = 360
pv_framerate = 30

# Buffer length in seconds
buffer_length = 10

if __name__ == '__main__':
    # Start the Personal Video subsystem on the HoloLens
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    producer = hl2ss_mp.producer()
    print(f"Personal Video stream configuration - width: {pv_width}, height: {pv_height}, framerate: {pv_framerate}")
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='bgr24'))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)

    # Configure and start the Research Mode Depth Long Throw stream
    port = hl2ss.StreamPort.RM_DEPTH_LONGTHROW
    fps = hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS if (port == hl2ss.StreamPort.RM_DEPTH_LONGTHROW) else hl2ss.Parameters_RM_DEPTH_AHAT.FPS
    print(f"\nConfiguring streams - Personal Video Port: {hl2ss.StreamPort.PERSONAL_VIDEO}, Depth Port: {port}, Expected FPS: {fps}")
    producer_depth = hl2ss_mp.producer()
    ht_profile_z = hl2ss.DepthProfile.SAME
    ht_profile_ab = hl2ss.VideoProfile.H265_MAIN
    calibration = hl2ss_3dcv.get_calibration_rm(host, port, '../calibration')
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(calibration.uv2xy, calibration.scale)
    max_depth = 8.0

    producer_depth.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, profile_z=ht_profile_z, profile_ab=ht_profile_ab))
    producer_depth.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer_depth.initialize(port, buffer_length * fps)
    producer_depth.start(port)

    # Create consumers and sinks for both video and depth streams
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    frame_stamp = sink_pv.get_attach_response()

    consumer_depth = hl2ss_mp.consumer()
    manager_depth = mp.Manager()
    sink_depth = consumer.create_sink(producer_depth, port, manager_depth, ...)
    sink_depth.get_attach_response()

    fps_counter = None
    prev_ts = None
    delta_ts = 0
    sample_time = 1

    latencies = []
    frame_delivery_times = []
    processing_times = []
    last_frame_time = None
    start_process_time = time.time()

    # Main loop to capture and process frames
    while True:
        # Get the most recent Personal Video frame
        stamp, data_pv = sink_pv.get_most_recent_frame()
        if data_pv is not None and stamp != frame_stamp:
            frame_stamp = stamp
        else:
            continue
        rgb_image = data_pv.payload.image

        # Get the most recent Research Mode Depth frame
        _, data_depth = sink_depth.get_most_recent_frame()
        depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, scale)
        ab = hl2ss_3dcv.slice_to_block(data_depth.payload.ab) / 65536

        # Combine depth and amplitude data for visualization (depth scaled for visibility)
        image = np.hstack((depth / max_depth, ab))

        # Calculate end-to-end latency
        current_time = time.time()
        frame_timestamp = data_pv.timestamp / hl2ss.TimeBase.HUNDREDS_OF_NANOSECONDS
        latency = current_time - frame_timestamp
        latencies.append(latency)

        # Calculate frame delivery time
        if last_frame_time is not None:
            frame_delivery_time = current_time - last_frame_time
            frame_delivery_times.append(frame_delivery_time)
        last_frame_time = current_time

        # Calculate and print the frame rate
        if fps_counter is None:
            fps_counter = hl2ss_utilities.framerate_counter()
            fps_counter.reset()
        else:
            fps_counter.increment()
            if fps_counter.delta() > sample_time:
                print(f'FPS: {fps_counter.get()}')
                delta_ts = 0
                fps_counter.reset()

        # Calculate processing time
        end_process_time = time.time()
        process_time = end_process_time - start_process_time
        processing_times.append(process_time)

        # Report latency and frame delivery statistics periodically
        if len(latencies) > 0 and time.time() - start_process_time > 5:
            latency_range = max(latencies) - min(latencies)
            print(f"Latency Range: {latency_range:.3f} s")
            print(f"Frame Delivery Time - Min: {min(frame_delivery_times):.3f} s, Max: {max(frame_delivery_times):.3f} s, Average: {np.mean(frame_delivery_times):.3f} s")
            print(f"Shape of RGB Image: {rgb_image.shape}, Shape of Depth/AB Image: {image.shape}")
            latencies.clear()
            frame_delivery_times.clear()
            processing_times.clear()
            start_process_time = time.time()

    # Clean up resources
    sink_pv.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

if __name__ == '__main__':
    main()
```

## Understanding the producer-consumer software design principle employed in this codebase

To write effective code using the hl2ss API for interfacing with the HoloLens 2, it is essential to understand the core software design principles that underlie this library.

Take the following code snippet, for example. It independently creates clients to receive data from the HoloLens’ RGB and depth sensor streams:


```python
# # Start subsystems
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.EXTENDED_VIDEO, shared=True, global_opacity=group_index, output_width=source_index, output_height=profile_index)
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.EXTENDED_DEPTH, shared=True, global_opacity=group_index_depth, output_width=source_index_depth, output_height=profile_index_depth)

# create clients
depth_client = hl2ss_lnm.rx_extended_depth(host, hl2ss.StreamPort.EXTENDED_DEPTH, profile_z=profile_z, media_index=media_index_depth, mode=hl2ss.StreamMode.MODE_0)
rgb_client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.EXTENDED_VIDEO, width=width, height=height, framerate=framerate, divisor=divisor, decoded_format=decoded_format) #mode=hl2ss.StreamMode.MODE_1, 

depth_client.open()
rgb_client.open()

while:
    depth_data = depth_client.get_next_packet()
    rgb_data = rgb_client.get_next_packet()

# stop subsystems
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.EXTENDED_DEPTH)
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.EXTENDED_VIDEO)
```

Unfortunately, the above approach doesn’t work. This is because the hl2ss `receivers rx_[...]` work in blocking mode. When `rx_[...].get_next_packet` is called, the function does not return until a whole packet is received. To understand why the software is written this way, we need to analyse the pros and cons of adopting the [producer-consumer]((https://jenkov.com/tutorials/java-concurrency/producer-consumer.html)) design pattern in this case.

The producer-consumer design pattern is a classic concurrency paradigm where producer threads generate data and place it into a shared buffer, while consumer threads retrieve and process it asynchronously. This decouples the rate of data production from data consumption, which is particularly advantageous in scenarios involving variable processing times or heterogeneous data sources. In the context of the HoloLens 2, each ```rx_[...]``` receiver acts as a producer, streaming data from a specific sensor (e.g., RGB, depth, IMU), while downstream components—such as visualization, fusion, or ML inference modules—function as consumers. One key benefit is that it allows sensor data to be buffered in memory, smoothing out jitter or delays caused by consumers processing at different rates. Additionally, this pattern supports modularity and scalability by allowing multiple independent consumers to access the same stream. However, it requires careful management of synchronization and buffer capacity to avoid overflows or stale data. Given the asynchronous nature of HL2SS’s multi-modal streaming and the need for frame-aligned processing across modalities, the producer-consumer pattern is an ideal choice—it enables parallelism without blocking and mitigates the latency and resource contention issues inherent in a naïve multi-receiver setup.

 When using only one receiver, this is not a problem, but when using multiple receivers, all the blocking stalls transfers, increases latency, and can cause frame drops due to send/recv buffers getting full. We tried to work around this using threads but it was too slow due to the Global Interpreter Lock [(GIL)](https://realpython.com/python-gil/) which limits true parallelism, so we used multiprocessing.

The producer wraps the receivers so each one can run on its own process (producer.configure). The received packets from ```rx_[...].get_next_packet``` are put into a circular buffer with a fixed capacity (capacity set by ```producer.initialize```) after streaming is started (producer.start). It is possible to have multiple producers (e.g., one producer per port), but the ports they access must not overlap.

This design has been employed by the need to support multiple readers per stream. Each reader is a consumer, and the sinks provide access to each of the producer streams. It is possible to have multiple consumers reading the same producer stream (port), but a consumer can have at most one sink of a given stream (port).

The sinks allow reading the nth most recent frame by using negative indices (-1 is most recent), or reading frames sequentially by using absolute non-negative indices (sink.get_buffered_frame). The absolute index of the most recent frame is given by ```sink.get_frame_stamp```. The most recent frame and its absolute index can also be obtained using ```sink.get_most_recent_frame```. When a sink is created, the producer gives it the index of the most recent frame, which can be obtained with ```sink_ev.get_attach_response``` (must always be called after creating the sink, even if the index is not used), then it is possible to read frames sequentially like this:

```python
index = sink_x.get_attach_response()

while (some_condition):
    status, frame_stamp, data = sink_x.get_buffered_frame(index)
    # status: 
    #      0 = ok
    #     -1 = requested frame is too old and has been discarded (internal circular buffer has limited capacity)
    #      1 = requested frame has not been received yet (index of future frame was given)
    # frame_stamp: absolute index, same as index if index >= 0, used for converting negative (relative) indices to absolute indices, invalid if status != 0
    # data: None if status != 0
    if (data is None):
        # optional wait
        continue
    index += 1
    # process data
    …

```

The frame closest in time to a given timestamp (and its absolute index) can be obtained using ```sink.get_nearest```. However, since all hl2ss server streams run independently and depending on network conditions, the optimal corresponding frame may not have yet arrived when ```sink.get_nearest``` is called. To keep it simple, I used a delay in the script in the above script to increase the chance that the optimal frame has been received, but it would be better to store the received color/depth frames, keep track of their timestamps, and then return optimal pairs. When a sink is not needed anymore, call ```sink.detach```. All sinks should be detached before stopping their corresponding producer stream (producer.stop). 
Finally, sinks support waiting via semaphores, in hopes of reducing cpu power by not busy waiting and not messing the scheduler by using sleep. For the last parameter of consumer.create_sink, None means no semaphore and no support for waiting, ```...``` means create a new semaphore for the sink, and a member of ```hl2ss.StreamPort``` means reuse the semaphore created for the sink of that port (enables using a single wait for all sinks sharing the semaphore in an OR fashion). Waiting for new frames can be done using ```sink.acquire```. Peeking can be done using ```sink.acquire``` \ ```sink.release``` pairs. Then, sequential access can be written as:

```python
sink_x = consumer.create_sink(producer, port, manager, ...)
index = sink_x.get_attach_response()

while (some_condition):
    sink_x.acquire()
    index += 1
    status, frame_stamp, data = sink_x.get_buffered_frame(index)
    # status is always 0
    # process data
    …	

```



## Summary
In this article, we explored the basics of the Microsoft Hololens2, including a detailed breakdown of its hardware and capabilities. We looked into using the Azure SDK to access onboard sensor information from the HoloLens and wrote boilerplate code that takes the sensor readings from the headset and puts it onto the offboard Personal Computer. However, this iteration of the wiki does not speak about the design pattern followed by this library/toolkit. Understanding this design pattern is essential to build good, reliable, robust and efficient software using this platform.


## References
- [HoloLens2github](https://github.com/jdibenes/hl2ss)
- [HL2forCV](https://github.com/microsoft/HoloLens2ForCV)
- [hl2_rm](arXiv:2008.11239)
- [PythonGlobalInterpreterLock](https://realpython.com/python-gil/)
- [Producer-Consumer-Design-Pattern](https://jenkov.com/tutorials/java-concurrency/producer-consumer.html)
- [HL2_hardware](https://learn.microsoft.com/en-us/hololens/hololens2-hardware)
- [WMixedReality](https://learn.microsoft.com/en-us/windows/mixed-reality/)
