---
date: 2025-11-30
title: Apple Vision Pro for Robotics Applications
---

This entry provides a systems-focused guide to integrating Apple Vision Pro with robotic systems for teleoperation, spatial perception, and on-device AI. It focuses on ARKit and RealityKit capabilities on visionOS, and Apple’s Foundation Models framework on visionOS. We provide architecture diagrams, key API references, and example workflows, and discuss practical limitations including latency, environmental constraints, and resource considerations.

This article targets roboticists who want to use Vision Pro as an interaction and perception interface. After reading, you will know how to stream head and hand pose data to robots, how to use object tracking and world anchors to augment robot perception and task execution, and how to leverage the Foundation Models framework on-device for low-latency semantic understanding and decision support.

## Teleoperation: Use Head and Hand Tracking for Robot Control

### Scenarios and Goals
- An operator wears Vision Pro and remotely controls a mobile robot or manipulator using head pose and hand joint data.
- Simultaneously record high-quality human demonstrations for downstream policy learning or imitation learning.

### System Architecture (schematic)
```
┌──────────────────────────────────────────────────────────────┐
│ Apple Vision Pro (visionOS)                                  │
│  • ARKitSession + HandTrackingProvider                       │
│  • RealityKit AnchorEntity (.hand / .head)                    │
│  • SpatialTrackingSession for authorization & data access     │
│  • Packetization: JSON / Protobuf                             │
│  • Streaming: WebSocket / UDP / TCP                           │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Robot side (base station / on-robot)                          │
│  • Receive pose stream (head/hand joints, device pose)        │
│  • Extrinsic calibration (HMD ↔ Robot Base)                    │
│  • Control mapping (joints / velocities / trajectories)        │
│  • ROS 2 / custom control stack                               │
└──────────────────────────────────────────────────────────────┘
```

### Key Apple APIs and References
- Overview of ARKit capabilities on visionOS: world tracking, hand tracking, object tracking, and more: https://developer.apple.com/documentation/arkit/arkit-in-visionos
- Track and visualize hand joints with `HandTrackingProvider` and `ARKitSession` (official sample): https://developer.apple.com/documentation/visionos/tracking-and-visualizing-hand-movement
- Configure cross-platform RealityKit anchors and request tracking authorization via `SpatialTrackingSession` (visionOS 2.0): https://developer.apple.com/videos/play/wwdc2024/10104/
- RealityKit cross-platform APIs and hand tracking input (WWDC24): https://developer.apple.com/videos/play/wwdc2024/10103/

### Capture and Stream Hand and Device Pose (visionOS app example)
The snippet uses RealityKit’s `SpatialTrackingSession` for authorization and ARKit’s `HandTrackingProvider` with `ARKitSession` to collect asynchronous hand anchor updates, package them as JSON, and stream them to the robot.

```swift
import ARKit
import RealityKit
import Foundation

final class TeleopTracker {
    private let arSession = ARKitSession()
    private let handProvider = HandTrackingProvider()
    private var worldProvider: WorldTrackingProvider?

    struct JointPose: Codable {
        let name: String
        let transform: simd_float4x4
        let tracked: Bool
    }

    struct TeleopPacket: Codable {
        let timestamp: TimeInterval
        let deviceTransform: simd_float4x4?
        let leftHand: [JointPose]
        let rightHand: [JointPose]
    }

    func authorize() async throws {
        let session = SpatialTrackingSession()
        let config = SpatialTrackingSession.Configuration(tracking: [.hand, .world])
        _ = try await session.run(config)
    }

    func start() async {
        do {
            try await arSession.run([handProvider])
        } catch {}

        Task {
            for await update in handProvider.anchorUpdates {
                let anchor = update.anchor
                let joints = anchor.skeleton.jointNames.map { name -> JointPose in
                    let pose = anchor.transform(for: name)
                    return JointPose(name: String(describing: name),
                                     transform: pose?.originFromAnchorTransform ?? matrix_identity_float4x4,
                                     tracked: pose != nil)
                }
                let packet = TeleopPacket(timestamp: Date().timeIntervalSince1970,
                                          deviceTransform: nil,
                                          leftHand: anchor.chirality == .left ? joints : [],
                                          rightHand: anchor.chirality == .right ? joints : [])
                send(packet)
            }
        }
    }

    private func send(_ packet: TeleopPacket) {
        // Serialize and send to robot control stack
    }
}
```

When using world tracking and device/head pose, include `WorldTrackingProvider` or query device pose to perform frame transforms and kinematic constraints. See the WWDC24 RealityKit drawing app for the `SpatialTrackingSession` workflow: https://developer.apple.com/videos/play/wwdc2024/10104/

### Frames and Calibration
- Define frames: `HMD` (headset), `World` (ARKit world), `Robot` (robot base).
- Estimate extrinsics:

\[ T_{Robot \leftarrow World},\; T_{World \leftarrow HMD} \]

Hand joint to robot end-effector mapping:

\[ p_{ee} = T_{Robot \leftarrow World} \cdot T_{World \leftarrow HMD} \cdot p_{hand\_joint} \]

Map poses/velocities to robot topics (for example, ROS 2 `geometry_msgs/Twist` or `sensor_msgs/JointState`) under rate limiting and collision constraints.

### Latency and Jitter
- Prefer local Wi‑Fi 6 or wired links; stream binary frames over WebSocket or UDP to reduce overhead.
- Use predicted tracking mode in RealityKit where appropriate to improve responsiveness.
- Apply smoothing and damping on the robot side (exponential smoothing, low‑pass filters).
- Implement safety policies that clamp speeds/accelerations and trigger failsafe stops.

### Authorization and Privacy
- RealityKit hand `AnchorEntity` enables visual anchoring but doesn’t expose precise transforms. For cross‑joint transforms, use ARKit `HandTrackingProvider` with user authorization. WWDC24: https://developer.apple.com/videos/play/wwdc2024/10104/

## Spatial Perception: Object Tracking and Anchors for Robot Tasks

### Capabilities and Use Cases
- Object tracking: recognize and track specific real‑world items (static placement) and attach content or extract object pose for robot manipulation. See WWDC24 sessions: https://developer.apple.com/videos/play/wwdc2024/10100/ and https://developer.apple.com/videos/play/wwdc2024/10101/
- World/plane anchors: environment understanding for tables, floors, rooms; use as constraints for navigation or collision reasoning. Overview: https://developer.apple.com/documentation/arkit/arkit-in-visionos

### Reference Object Workflow
1. Obtain a USDZ 3D model of the target object.
2. Train a reference object in Create ML’s spatial object tracking and export a `.referenceobject`.
3. Load reference objects and start `ObjectTrackingProvider`.

Example (start object tracking):
```swift
import ARKit
import RealityKit

let session = ARKitSession()

func startObjectTracking(referenceObjects: [ReferenceObject]) async throws {
    let provider = ObjectTrackingProvider(referenceObjects: referenceObjects)
    try await session.run([provider])
    for await update in provider.anchorUpdates {
        let anchor = update.anchor
        // Use anchor.referenceObject and anchor.transform in perception/controls
    }
}
```
Full sample and workflow guides: https://developer.apple.com/documentation/visionos/exploring_object_tracking_with_arkit and https://developer.apple.com/documentation/visionos/using-a-reference-object-with-arkit

### RealityKit Anchors and UI
- Use `AnchorEntity` to affix content to world/plane/hand/object anchors for coaching UIs and overlays: https://developer.apple.com/documentation/realitykit/anchorentity
- In visionOS, body‑related anchor transforms may be restricted; for precise transforms in robot compute, use ARKit providers with authorization: https://developer.apple.com/videos/play/wwdc2024/10104/

### Robot Integration Examples
- Manipulation: obtain object pose from anchors, apply extrinsics, generate grasp pose/trajectory, send to end effector.
- Mobile navigation: use room/plane anchors for occupancy/boundary estimation, generate local constraints and spatial markers.
- Coaching UIs: overlay transparent models or callouts near objects to guide assembly/repair; author content with Reality Composer Pro.

### Calibration and Robustness
- Perform camera extrinsics and hand‑eye calibration to stabilize HMD ↔ robot transforms.
- Use robust estimation and error models to handle anchor loss/drift; fuse depth/IMU when needed.

## On‑Device Intelligence: Foundation Models on visionOS

### Overview
- Apple’s Foundation Models framework provides on‑device LLM capabilities across iOS, iPadOS, macOS, and visionOS, including prompting, guided generation, streaming, and tool calling. Intro: https://developer.apple.com/videos/play/wwdc2025/286/ Deep dive: https://developer.apple.com/videos/play/wwdc2025/301/
- Use on‑device semantics for command parsing, plan drafting, scene descriptions, or task planning prototypes.

### Example (guided generation)
```swift
import FoundationModels

@Generable
struct CommandPlan {
    var intent: String
    var steps: [String]
}

func plan(from userInput: String) async throws -> CommandPlan {
    let session = LanguageModelSession()
    let prompt = "Generate a robot task plan based on input: \(userInput)"
    let response = try await session.respond(to: prompt, generating: CommandPlan.self)
    return response.content
}
```

### Tools and Multimodal Fusion
- Use tool calling to access device sensors or external services: query robot status or object poses, then produce semantic summaries or step plans. Code‑along demo: https://developer.apple.com/videos/play/wwdc2025/259/

### Adapter Training (advanced)
- Train `.fmadapter` to specialize the system LLM for your domain; entitlement and deployment details: https://developer.apple.com/apple-intelligence/foundation-models-adapter/

### Safety and HIG
- Apply layered safety: model suggestions feed deterministic safety controllers. Prompt design and safety guidance: https://developer.apple.com/videos/play/wwdc2025/248/

## End‑to‑End Integration Example

### Goal
- Operator uses hand gestures to control a manipulator to grasp an object on a table. The system interprets a spoken/text instruction on device, identifies the target, plans a grasp, and sends commands to the robot.

### Steps
1. Authorization and tracking: `SpatialTrackingSession` requests hand and world capabilities: https://developer.apple.com/videos/play/wwdc2024/10104/
2. Collect hand joints and device pose: `HandTrackingProvider` and `WorldTrackingProvider` streams: https://developer.apple.com/documentation/arkit/arkit-in-visionos
3. Object tracking: load `.referenceobject`, run `ObjectTrackingProvider`, maintain target pose: https://developer.apple.com/documentation/visionos/exploring_object_tracking_with_arkit
4. Frame fusion and grasp planning: combine object pose with robot extrinsics to compute grasp pose and trajectory.
5. On‑device semantics: Foundation Models parses the instruction and emits structured steps and safety notes: https://developer.apple.com/videos/play/wwdc2025/286/
6. Control and safety: execute, monitor force/vision feedback, and apply safety policies.

### Data Structures
- Hand joints: joint name, tracked flag, local/world transforms.
- Device pose: homogeneous transform, timestamp, tracking state.
- Object anchors: reference object ID, pose, confidence, bounding box.
- Task semantics: intent, steps, constraints, fallback policies.

## Performance, Resources, and Constraints

### Latency and Update Rates
- Tracking updates arrive as async streams; avoid blocking, use producer–consumer queues and structured concurrency.
- On‑device LLM latency scales with prompt length; use LLMs for low‑frequency intent parsing, not high‑rate control loops.

### Resource Use
- Foundation Models run on device without increasing app size, but inference incurs power/thermal costs; apply throttling/timeouts for long sessions.
- Object tracking has limits on detection rate and instance counts; configure `TrackingConfiguration` and related entitlements when needed: https://developer.apple.com/documentation/visionos/using-a-reference-object-with-arkit

### Privacy and Permissions
- Access to hand/world/object anchors follows visionOS authorization; if denied, RealityKit `AnchorEntity` may still visually anchor content but won’t update transform data flows—degrade UX and control accordingly: https://developer.apple.com/videos/play/wwdc2024/10104/

### Environment and Robustness
- Lighting, texture, occlusions, and reflections impact tracking; prefer textured targets and place visual aids where possible.
- Periodically re‑localize and correct drift for long tasks.

## Example: Hand–Eye Fusion to End‑Effector Pose

```swift
import simd

func endEffectorPose(robotToWorld: simd_float4x4,
                     worldToHMD: simd_float4x4,
                     handJointInHMD: simd_float4x4) -> simd_float4x4 {
    let worldToRobot = robotToWorld.inverse
    let handInWorld = worldToHMD * handJointInHMD
    return worldToRobot * handInWorld
}
```

Map `endEffectorPose` to the robot control interface (position or velocity control) and apply safety clamping and collision checks.

## Summary
- Vision Pro plus ARKit/RealityKit offers high‑quality spatial tracking and scene understanding for teleoperation and perception.
- visionOS 2.0 adds `SpatialTrackingSession` for a simpler authorization/data‑access flow; RealityKit provides cross‑platform anchors and interaction APIs.
- On‑device Foundation Models enable privacy‑friendly semantics and planning components that pair well with deterministic robot control.
- Address latency, resources, privacy, and environment robustness with layered safety and concurrent data pipelines.

## See Also
- ARKit overview (visionOS): https://developer.apple.com/documentation/arkit/arkit-in-visionos
- Hand tracking sample: https://developer.apple.com/documentation/visionos/tracking-and-visualizing-hand-movement
- RealityKit cross‑platform APIs (WWDC24): https://developer.apple.com/videos/play/wwdc2024/10103/
- RealityKit drawing app and `SpatialTrackingSession` (WWDC24): https://developer.apple.com/videos/play/wwdc2024/10104/
- Object tracking workflow (WWDC24): https://developer.apple.com/videos/play/wwdc2024/10101/
- Foundation Models framework: https://developer.apple.com/videos/play/wwdc2025/286/
- Foundation Models deep dive: https://developer.apple.com/videos/play/wwdc2025/301/

## Further Reading
- Using reference objects with ARKit: https://developer.apple.com/documentation/visionos/using-a-reference-object-with-arkit
- Exploring object tracking with ARKit: https://developer.apple.com/documentation/visionos/exploring_object_tracking_with_arkit
- RealityKit AnchorEntity documentation: https://developer.apple.com/documentation/realitykit/anchorentity
- Code‑along with Foundation Models (WWDC25): https://developer.apple.com/videos/play/wwdc2025/259/
- Prompt design and safety (WWDC25): https://developer.apple.com/videos/play/wwdc2025/248/

## References
- Apple, “ARKit in visionOS,” Apple Developer Documentation. Available: https://developer.apple.com/documentation/arkit/arkit-in-visionos
- Apple, “Tracking and visualizing hand movement,” Apple Developer Documentation. Available: https://developer.apple.com/documentation/visionos/tracking-and-visualizing-hand-movement
- Apple, “Discover RealityKit APIs for iOS, macOS, and visionOS,” WWDC24. Available: https://developer.apple.com/videos/play/wwdc2024/10103/
- Apple, “Build a spatial drawing app with RealityKit,” WWDC24. Available: https://developer.apple.com/videos/play/wwdc2024/10104/
- Apple, “Create enhanced spatial computing experiences with ARKit,” WWDC24. Available: https://developer.apple.com/videos/play/wwdc2024/10100/
- Apple, “Explore object tracking for visionOS,” WWDC24. Available: https://developer.apple.com/videos/play/wwdc2024/10101/
- Apple, “AnchorEntity,” Apple Developer Documentation. Available: https://developer.apple.com/documentation/realitykit/anchorentity
- Apple, “Meet the Foundation Models framework,” WWDC25 Session. Available: https://developer.apple.com/videos/play/wwdc2025/286/
- Apple, “Deep dive into the Foundation Models framework,” WWDC25 Session. Available: https://developer.apple.com/videos/play/wwdc2025/301/
- Apple, “Foundation Models adapter training,” Apple Intelligence. Available: https://developer.apple.com/apple-intelligence/foundation-models-adapter/