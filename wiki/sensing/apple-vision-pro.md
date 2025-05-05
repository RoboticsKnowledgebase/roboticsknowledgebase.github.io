# Apple Vision Pro for Robotics

## Introduction

This guide provides an overview on integrating robotics projects with Apple Vision Pro. Broadly covering setting up the Vision Pro for development, use it for object tracking, understand spatial coordinate frames, and send data to a ROS (Robot Operating System) environment over TCP.

## Setting Up Apple Vision Pro for Development

### Prerequisites:
- A Mac with macOS 14+ (ideally) and Xcode 15.x or higher installed (which includes the visionOS SDK).
- An Apple Developer account (needed to deploy apps to the device - costs $100 for a year as of 2025).
- Ensure both Mac and Vision Pro are on the same network.

### Steps:
1. **Pair the Vision Pro with Xcode**:  
   Make sure your Vision Pro and Mac are on the same Wi-Fi network. On the Vision Pro, go to *Settings → General → Remote Devices* and enter pairing mode. Then, on your Mac, open *Xcode → Window → Devices and Simulators*. Select Vision Pro and click “Pair”. This establishes a wireless connection for deployment.

2. **Enable Developer Mode on Vision Pro**:  
   After pairing, on the headset navigate to *Settings → Privacy & Security → Developer Mode* and turn it on.

3. **Deploy apps from Xcode**:  
   In Xcode (version 15+ with the visionOS SDK), create a new visionOS project (you can start with a simple “Window” scene for a 2D app in space). Choose the headset as the run destination and build/run your app.

### Key Tools:
visionOS development uses Xcode 15 or later (running on macOS Ventura/Monterey and Apple silicon Macs). The visionOS SDK is included in Xcode and provides the frameworks for spatial computing. Apps are built primarily with SwiftUI for UI and interface layouts, combined with RealityKit/ARKit for 3D content and AR interaction. You can also design 3D content using Reality Composer Pro, a visual scene editor, useful for scene manipulation if you have multiple objects in the scene, it bypasses a lot of the low-level coding to develop simple apps.

### Learning Resources:
If you’re new to Apple development or SwiftUI, start with Apple’s official tutorials and documentation. Apple’s “Creating your first visionOS app” guide walks you through a basic project setup [1]. SwiftUI itself has excellent tutorials and documentation on the Apple Developer site [2].

## Object Tracking with Apple Vision Pro

Vision Pro’s outward cameras and ARKit framework allow you to detect and track real-world objects, which can be useful in robotics (e.g. tracking tools, parts, or fiducial markers). The process involves creating a reference of the object and then detecting that object in the headset’s view using ARKit’s object tracking. You have two primary approaches to get a reference:

1. Scan the object to create an ARKit reference object.
2. Train a machine learning model (via Create ML) for detection.

**1. Example workflow of scanning a real object to create an ARReferenceObject**. The process involves preparing the object, defining a bounding box around it, scanning from multiple angles, adjusting the reference coordinate origin (shown by axis gizmos), and then testing and exporting the reference object file.

### Scanning and Reference Objects:
The most direct method is to scan the object using ARKit’s scanning configuration. Apple’s ARKit provides an `ARObjectScanningConfiguration` that lets you move the Vision Pro (or an iPhone/iPad) around the object to capture its shape. This produces an `ARReferenceObject` – essentially a file containing a 3D point cloud of the object’s features [3].

The reference object doesn’t store a full 3D model, just enough spatial data for ARKit to recognize the object later. After scanning, you’d save this reference (a `.arobject` file). For example, Apple’s sample app for object scanning guides you through defining the object’s bounding box, scanning all sides, and adjusting the coordinate origin for the object before exporting the reference file. Once you have a good ARReferenceObject, add it to your Xcode project (by importing it into an AR Resource Group in Assets).

### Detecting and Tracking Objects in visionOS:
To recognize the object at runtime, your visionOS app must load the ARReferenceObject and tell ARKit to look for it. In code, this means adding the reference object to an AR session configuration (e.g. `ARWorldTrackingConfiguration`) via its `detectionObjects` property [4].

When the app runs, ARKit will scan the environment through Vision Pro’s cameras and attempt to match the stored reference. If the object is in view, ARKit will detect it and create an `ARObjectAnchor`, which represents the real object’s position and orientation in the world. You can retrieve this anchor via ARKit’s callbacks – for instance, ARKit’s delegate method will provide an `ARObjectAnchor` when the object is first detected.

The `ARObjectAnchor` contains useful info like the anchor’s transform (i.e. the object’s pose) and the reference object’s name, extent, etc.

Once you have the `ARObjectAnchor`, you can attach virtual content to it. In RealityKit, you would create an `AnchorEntity` from that anchor so that any entities you add will move with the real object [5].

### 2. Using Create ML for Object Tracking:
In cases where scanning the object is not feasible or you want to recognize it from visuals, Apple’s visionOS supports an object tracking workflow using machine learning. Apple introduced the ability to create reference objects using Create ML in 2024. This involves training a custom Vision model (using Create ML with many images of your object from different angles) to recognize the object.

The result is an `.arobject` or `.mlmodel` that ARKit can use to detect the item via its cameras, similar to how ARKit would use a scanned reference. The workflow is: feed images of the object into Create ML’s Object Detection template, train a model, and integrate that model into your visionOS app.

At runtime, ARKit will utilize the ML model to identify the object and still provide you with an `ARObjectAnchor`. Apple’s WWDC24 session “Explore object tracking for visionOS” showcases in detail how to create a reference object with ML and attach content relative to it in RealityKit or Reality Composer Pro [6].

### Relevant Classes and APIs:
- `ARReferenceObject` – holds the scanned/trained object reference data  
- `ARObjectAnchor` – anchor ARKit produces when the object is found  
- `AnchorEntity` – RealityKit anchor tied to the ARObjectAnchor  
- `ARSession` / `ARView` – manages the session with object detection configuration  

## Understanding Coordinate Frames in visionOS

Working with Vision Pro in robotics requires understanding the coordinate systems that ARKit and RealityKit use. In spatial computing (and robotics), keeping track of coordinate frames is crucial so you know where an object or anchor is in space relative to the user or robot. Vision Pro’s coordinate system conventions are consistent with ARKit’s on other devices: a right-handed 3D coordinate system with a gravity-aligned up-axis [7].

### World Coordinate System:
By default, visionOS apps use ARKit’s world coordinate frame. This is a right-handed system where the Y-axis points upward (aligned with gravity), the Z-axis points toward the viewer, and the X-axis points to the viewer’s right.

The origin (0,0,0) is typically where the AR session starts. All object detections are given relative to this frame.

### Anchors and Local Coordinate Frames:
Each ARAnchor (including `ARObjectAnchor`) represents a fixed world point with its own coordinate frame. RealityKit uses a parent-child transform hierarchy where `Entity` positions are interpreted in their parent anchor's local frame.

### Default Orientation and Units:
1 unit = 1 meter. Conventionally, Y is up, X is right, and Z is toward the viewer (meaning forward is -Z). 

### Reality Composer Pro Coordinates:
Reality Composer Pro uses the same coordinate system. When you build scenes in RC Pro and anchor them to objects or space, they retain scale and orientation when used in your app. This makes it intuitive for designing spatially-accurate visual overlays [8].


## Setting Up ROS Communication Using TCP

After using Vision Pro to detect or track objects, you’ll often want to send that information to a robot or a ROS-based system. Although you can’t run ROS natively on visionOS, you can communicate over a network. Vision Pro applications can use standard networking (TCP/UDP sockets, etc.) to send data to a ROS computer on the same local network. Below is a simple way to stream data (like an object’s pose) from a visionOS app to ROS using a TCP connection:

1.  **Network Setup:** Ensure the Vision Pro and the ROS machine (PC or robot) are on the same local network. You might need to allow local network access for your app (the first time the app tries to connect, visionOS will prompt the user to grant permission for local network communication).

2.  **Open a TCP Socket:** In your visionOS app, use Apple’s networking APIs to create a TCP client that connects to the ROS system. Use `Network.framework` (in Swift) which provides the `NWConnection` class for TCP connections. For instance, you can initiate an `NWConnection` to the robot’s IP (say `192.168.10.10`) on a specific port (you choose a port number that the robot will listen on). `Network.framework` allows you to send and receive data asynchronously.

3.  **Structure the Message:** Decide on a simple message format for the data you want to send. A common, human-readable choice is JSON. For example, if you want to send the pose of a detected object, you could send a JSON string like: `{"object": "toolbox", "position": [0.2, 0.0, -1.5], "orientation": [0.0, 0.0, 0.0, 1.0]}`.

4.  **Send Data from visionOS App:** Use the network connection to send the serialized message. With `Network.framework`, you would convert the JSON string to `Data` (UTF8) and send it over the `NWConnection`. This framework ensures the data is delivered reliably over TCP. Developers have reported that sending JSON-encoded updates over `Network.framework` is straightforward and performs well in real-time use cases. You can send messages periodically (e.g., every time you get a new ARFrame or when an object moves) or on events (e.g., when a new object is detected).

5.  **ROS Side – Receive and Handle Data:** On the ROS machine, run a small server to accept the TCP connection. This could be a simple Python script or C++ node using sockets that listens on the chosen port. When the Vision Pro app connects, the ROS side accepts the socket and reads incoming data. It will receive the JSON strings (or whatever format you chose). The ROS node can then parse the JSON (using a JSON library) to extract the numbers. With the data in hand, you can publish it onto a ROS topic and use it wiht your robotic system. A great example of this functionality being used in robotics projects is the OpenTelivision App[9].


**Tips:** Keep the message format simple – since both ends are under your control, you could even use a very lightweight format (like a comma-separated string `"toolbox,0.2,0.0,-1.5,0.0,0.0,0.0,1.0"` to save space). JSON is fine for most cases and easier to extend (you might include multiple objects or other sensor data in one message). Ensure your app doesn’t send data too fast for the network; throttling to, say, 20 Hz or less is usually plenty for object pose updates. Also, be mindful of coordinate frames: as a simple starting point, you can assume the Vision Pro is a sensor and treat its world coordinate system as a separate frame in ROS (publishing TF transforms between `“VisionPro_frame”` and the robot’s frame after calibration).

The TCP socket approach is just one method; for more complex integrations, you might explore higher-level protocols (for instance, ROS2 could communicate over DDS or you could use a WebSocket with ROSbridge). But a simple client-server TCP link is easy to set up and sufficient for many robotics experiments.

## Relevant Resources
[1] [Getting Started with visionOS](https://developer.apple.com/documentation/visionos/)

[2] [SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui)

[3] [ARKit Object Scanning Guide](https://developer.apple.com/documentation/arkit/scanning-and-detecting-3d-objects)

[4] [ARWorldTrackingConfiguration Documentation](https://developer.apple.com/documentation/arkit/arworldtrackingconfiguration)

[5] [RealityKit AnchorEntity API](https://developer.apple.com/documentation/realitykit/anchorentity)

[6] [Object Tracking for visionOS](https://developer.apple.com/videos/play/wwdc2024/10091/)

[7] [ARKit Coordinate Space](https://developer.apple.com/documentation/arkit/arsession/2941103-currentframe)

[8] [Apple Network.framework Documentation](https://developer.apple.com/documentation/network)

[9] [OpenTelevision App](https://github.com/OpenTeleVision/TeleVision)