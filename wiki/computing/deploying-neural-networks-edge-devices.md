---
date: 2026-04-30
title: Deploying Neural Networks on Edge Devices for Robotics
---
Deploying neural networks directly on edge devices is essential for modern robotic systems that must perceive and act in real time. Cloud-based inference introduces network latency, bandwidth constraints, and single points of failure that are unacceptable for safety-critical applications such as autonomous navigation, manipulation, and human-robot interaction. This article provides a comprehensive, practical guide to the full edge deployment pipeline: exporting trained models to portable formats like ONNX, optimizing them with inference engines such as TensorRT and OpenVINO, applying quantization techniques for maximum throughput, and integrating optimized models into ROS 2 robotic software stacks on platforms like NVIDIA Jetson, Google Coral, and Raspberry Pi with AI accelerators.

## Why Edge Inference Matters for Robotics

Robotic perception and control loops impose strict requirements that make on-device inference the preferred deployment strategy in most real-world systems.

### Latency

A robot navigating at 1 m/s with a 100 ms perception latency travels 10 cm blind between frames. Reducing inference latency to 10 ms shrinks that gap to 1 cm. Edge deployment eliminates the round-trip network delay inherent in cloud inference, which typically adds 20–200 ms depending on connectivity.

### Bandwidth

A single 720p RGB camera at 30 FPS generates roughly 50 MB/s of raw data. Stereo cameras, LiDAR, and depth sensors multiply this several-fold. Streaming all sensor data to a remote server is impractical over typical wireless links, especially when multiple robots share the same network.

### Reliability

Wireless connectivity is unreliable in many robotics environments — warehouses with metal shelving, underground tunnels, outdoor fields. Edge inference guarantees that perception continues operating regardless of network conditions.

### Power and Form Factor

Mobile robots carry limited battery capacity. Modern edge accelerators like the NVIDIA Jetson Orin NX deliver up to 100 TOPS of INT8 performance at 15–25 W, enabling sophisticated deep learning models within tight power budgets.

### Common Edge Platforms

| Platform | Compute | Power | Typical Use Case |
|---|---|---|---|
| NVIDIA Jetson Orin AGX | 275 TOPS (INT8) | 15–60 W | High-performance mobile robots, AV |
| NVIDIA Jetson Orin NX | 100 TOPS (INT8) | 10–25 W | Mid-range robots, drones |
| NVIDIA Jetson Orin Nano | 40 TOPS (INT8) | 7–15 W | Cost-sensitive edge inference |
| Google Coral Edge TPU | 4 TOPS (INT8) | 2 W | Low-power classification, detection |
| Raspberry Pi 5 + Hailo-8L | 13 TOPS (INT8) | 5–10 W | Lightweight perception, education |
| Intel NUC + Neural Compute Stick | ~4 TOPS (FP16) | 10–25 W | OpenVINO-optimized workloads |

For platform-specific setup guidance on the Jetson Orin, see the [Jetson Orin AGX](/wiki/computing/jetson-orin-agx/) article.

## The Edge Deployment Pipeline

The deployment pipeline transforms a trained model into an optimized inference engine tailored to the target hardware. The general workflow follows four stages:

```
Train (GPU Workstation/Cloud)
  → Export (PyTorch/TF → ONNX)
    → Optimize (ONNX → TensorRT / OpenVINO / TFLite)
      → Deploy (Edge Device + ROS 2)
```

### Model Interchange Formats

**ONNX (Open Neural Network Exchange)** is the most widely supported intermediate representation. It defines a common set of operators and a standard file format that bridges training frameworks and inference runtimes.

| Source Framework | Target Runtime | Path |
|---|---|---|
| PyTorch | TensorRT (Jetson) | PyTorch → ONNX → TensorRT |
| PyTorch | OpenVINO (Intel) | PyTorch → ONNX → OpenVINO IR |
| TensorFlow | TFLite (Coral/RPi) | TensorFlow → SavedModel → TFLite |
| TensorFlow | TensorRT (Jetson) | TensorFlow → ONNX → TensorRT |

### Why Not Deploy the Training Framework Directly?

Training frameworks like PyTorch are designed for flexibility and automatic differentiation, not inference speed. They carry substantial overhead: dynamic graphs, gradient bookkeeping, and unoptimized operator scheduling. Dedicated inference engines strip this overhead and apply hardware-specific optimizations that can yield 2–10x speedups.

## ONNX Export

### Exporting a PyTorch Model

PyTorch provides built-in ONNX export through `torch.onnx.export()`. The exporter traces the model with sample input and records the operations into an ONNX graph.

```python
import torch
import torchvision

# Load a pretrained detection model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights="DEFAULT"
)
model.eval()

# Create a dummy input matching expected input dimensions
# Batch size 1, 3 channels, 640x640 resolution
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "detector.onnx",
    opset_version=17,            # Use a recent opset for broad operator support
    input_names=["images"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={               # Allow variable batch size at inference time
        "images": {0: "batch"},
        "boxes": {0: "batch"},
        "labels": {0: "batch"},
        "scores": {0: "batch"},
    },
)
print("ONNX export complete: detector.onnx")
```

> When using `dynamic_axes`, ensure the model architecture supports variable dimensions. Some operations (e.g., hard-coded reshapes) may break with dynamic shapes.

### Exporting a TensorFlow/Keras Model

For TensorFlow models, use the `tf2onnx` converter:

```bash
pip install tf2onnx
python -m tf2onnx.convert \
    --saved-model ./saved_model_dir \
    --output model.onnx \
    --opset 17
```

### Validating the ONNX Model

Always validate the exported model before optimization to catch export errors early.

```python
import onnx
import onnxruntime as ort
import numpy as np

# Structural validation: checks graph consistency and operator support
model = onnx.load("detector.onnx")
onnx.checker.check_model(model)
print("ONNX model structure is valid.")

# Runtime validation: run inference and verify output shapes
session = ort.InferenceSession("detector.onnx")
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {"images": dummy})

for i, out in enumerate(outputs):
    print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
```

If the checker raises errors, common fixes include:
- Updating the `opset_version` to support newer operators
- Replacing unsupported custom operators with standard ONNX equivalents
- Simplifying the model with `onnx-simplifier`: `python -m onnxsim model.onnx model_simplified.onnx`

## TensorRT Optimization

NVIDIA TensorRT is the highest-performance inference engine for NVIDIA GPUs, including all Jetson platforms. It applies a suite of optimizations that are impossible at the framework level.

### What TensorRT Does

1. **Layer Fusion**: Combines sequences of operations (e.g., Conv → BatchNorm → ReLU) into single GPU kernels, reducing memory traffic and kernel launch overhead.
2. **Kernel Auto-Tuning**: Benchmarks multiple CUDA kernel implementations for each layer on the specific target GPU and selects the fastest.
3. **Precision Calibration**: Converts FP32 weights to FP16 or INT8 with minimal accuracy loss, doubling or quadrupling throughput.
4. **Memory Optimization**: Reuses memory buffers across layers with non-overlapping lifetimes, reducing peak memory consumption.
5. **Dynamic Tensor Memory**: Allocates only the memory needed for the actual input dimensions when using dynamic shapes.

### Building a TensorRT Engine from ONNX

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True, int8=False, calibrator=None):
    """Build and serialize a TensorRT engine from an ONNX model."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return None

    # Configure builder settings
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 30  # 1 GB workspace
    )

    # Enable FP16 precision (2x speedup on most Jetson platforms)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Enable INT8 precision (requires calibration dataset)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    # Build the engine (this takes minutes — auto-tunes kernels)
    serialized_engine = builder.build_serialized_network(network, config)

    # Save to disk for later loading
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")
    return serialized_engine

# Build an FP16 engine
build_engine("detector.onnx", "detector_fp16.engine", fp16=True)
```

> TensorRT engines are **not portable** across GPU architectures. An engine built on a Jetson Orin will not run on a Jetson Xavier or a desktop GPU. Always build on the target device or a device with the same GPU architecture.

### Running Inference with a TensorRT Engine

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def load_engine(engine_path):
    """Deserialize a TensorRT engine from disk."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_data):
    """Run inference on a single input tensor."""
    context = engine.create_execution_context()

    # Allocate device memory for input and output tensors
    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)

    # Query output tensor shape and allocate memory
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
    output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
    d_output = cuda.mem_alloc(output_size)

    # Set tensor addresses and execute
    context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(d_output))
    context.execute_async_v3(stream_handle=cuda.Stream().handle)

    # Copy result back to host
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    return output
```

For a complete YOLOv5 deployment example with TensorRT on Jetson, see the [YOLOv5 Training and Deployment on NVIDIA Jetson](/wiki/machine-learning/yolov5-tensorrt/) article.

## Quantization Deep Dive

Quantization reduces the numerical precision of model weights and activations from 32-bit floating point to lower-bitwidth representations, trading a small accuracy reduction for significant gains in speed, memory, and power efficiency.

### The Quantization Formula

Uniform affine quantization maps a floating-point value $x$ to an integer $q$ using a scale factor $s$ and zero-point $z$:

$$q = \text{round}\left(\frac{x}{s}\right) + z$$

The inverse (dequantization) recovers an approximation of the original value:

$$\hat{x} = s \cdot (q - z)$$

The scale and zero-point are computed from the observed range $[x_{\min}, x_{\max}]$ of the tensor:

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)$$

For INT8 quantization, $q_{\min} = -128$ and $q_{\max} = 127$, giving 256 discrete levels. The quantization error is bounded by $\frac{s}{2}$.

### FP16 vs INT8

| Property | FP32 (Baseline) | FP16 | INT8 |
|---|---|---|---|
| Bits per value | 32 | 16 | 8 |
| Typical speedup | 1x | 1.5–2x | 2–4x |
| Memory reduction | 1x | 2x | 4x |
| Accuracy impact | Baseline | Negligible | 0.5–2% drop typical |
| Calibration needed | No | No | Yes |

**Use FP16** as the default for edge deployment — it provides a significant speedup with virtually no accuracy loss on modern hardware. **Use INT8** when latency or throughput requirements cannot be met with FP16, and when you can afford the calibration and validation effort.

### Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained FP32 model without retraining. It requires a small calibration dataset (typically 500–1000 representative samples) to determine optimal scale factors for each tensor.

```python
import tensorrt as trt
import numpy as np
import os

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator that feeds representative images for INT8 quantization."""

    def __init__(self, calibration_dir, batch_size=8, input_shape=(3, 640, 640)):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape

        # Load calibration image paths
        self.image_paths = [
            os.path.join(calibration_dir, f)
            for f in sorted(os.listdir(calibration_dir))
            if f.endswith((".jpg", ".png"))
        ]
        self.current_index = 0
        self.cache_file = "calibration.cache"

        # Pre-allocate device buffer for one batch
        self.device_input = cuda.mem_alloc(
            batch_size * int(np.prod(input_shape)) * 4
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Load and preprocess the next batch of calibration images."""
        if self.current_index >= len(self.image_paths):
            return None

        batch = []
        for i in range(self.batch_size):
            idx = self.current_index + i
            if idx >= len(self.image_paths):
                break
            # Load and preprocess image (resize, normalize to [0, 1])
            img = load_and_preprocess(self.image_paths[idx], self.input_shape)
            batch.append(img)

        self.current_index += self.batch_size
        batch_array = np.array(batch, dtype=np.float32)
        cuda.memcpy_htod(self.device_input, batch_array)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Read cached calibration data to avoid recalibrating."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration data to disk for future reuse."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Build INT8 engine with calibrator
calibrator = ImageCalibrator("./calibration_images/", batch_size=8)
build_engine("detector.onnx", "detector_int8.engine", fp16=True, int8=True, calibrator=calibrator)
```

> Select calibration images that represent the actual distribution your robot will encounter. For a warehouse robot, use images from the warehouse — not ImageNet validation images.

### Quantization-Aware Training (QAT)

QAT simulates quantization effects during training by inserting fake quantization nodes that round weights and activations to their quantized equivalents on the forward pass while maintaining full-precision gradients on the backward pass. This allows the model to learn to be robust to quantization noise.

QAT typically recovers 0.5–1% accuracy compared to PTQ. Use it when:
- PTQ accuracy is unacceptable for your application
- The model is small and retraining is inexpensive
- You need INT8 for latency-critical control loops

PyTorch supports QAT through the `torch.ao.quantization` API:

```python
import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

model.train()

# Attach quantization configuration to the model
model.qconfig = get_default_qat_qconfig("x86")  # or "qnnpack" for ARM

# Insert fake quantization observers
model_prepared = prepare_qat(model)

# Fine-tune for a few epochs with quantization simulation
for epoch in range(fine_tune_epochs):
    train_one_epoch(model_prepared, train_loader)

# Convert to fully quantized model
model_quantized = convert(model_prepared)
```

## Deployment Patterns for Robotics

### ROS 2 Inference Node

Wrapping an optimized inference engine in a ROS 2 node allows seamless integration with the broader robotic perception and planning stack.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np

class DetectorNode(Node):
    """ROS 2 node that runs TensorRT object detection on camera images."""

    def __init__(self):
        super().__init__("detector_node")

        # Load TensorRT engine at startup
        self.engine = load_engine("detector_fp16.engine")
        self.context = self.engine.create_execution_context()
        self.bridge = CvBridge()

        # Subscribe to camera images
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # Publish detection results
        self.pub = self.create_publisher(
            Detection2DArray, "/detections", 10
        )
        self.get_logger().info("Detector node initialized with TensorRT engine")

    def image_callback(self, msg):
        # Convert ROS Image to numpy array
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        # Preprocess: resize, normalize, transpose to CHW, add batch dim
        input_tensor = self.preprocess(cv_image)

        # Run TensorRT inference
        boxes, labels, scores = self.infer(input_tensor)

        # Publish detections above confidence threshold
        det_array = Detection2DArray()
        det_array.header = msg.header
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                det = Detection2D()
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(label))
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                det_array.detections.append(det)

        self.pub.publish(det_array)

def main():
    rclpy.init()
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Multi-Model Pipelines

Many robotic perception systems chain multiple models. A common pattern is a fast detector followed by a more expensive classifier or pose estimator that runs only on detected regions of interest (ROIs):

```
Camera Frame → Detector (YOLOv8-nano, ~5ms)
  → Crop ROIs
    → Classifier (ResNet-18, ~2ms per ROI)
      → Pose Estimator (only for target class, ~8ms)
```

Run the detector on every frame and the downstream models only on relevant ROIs. This keeps the total pipeline latency low while reserving computational headroom for the most informative processing.

### Batching Strategies

- **Latency-optimized (batch=1)**: Process each frame immediately as it arrives. Best for real-time control loops where freshness matters more than throughput.
- **Throughput-optimized (batch=N)**: Accumulate N frames and process them together. TensorRT and GPUs achieve higher utilization with larger batches. Use this for offline processing, mapping, or when multiple cameras feed the same model.
- **Adaptive batching**: Adjust batch size based on the current processing queue depth. Process immediately when idle; batch when backlogged.

### Memory Management on Constrained Devices

Jetson devices share memory between CPU and GPU. A few practices help avoid out-of-memory crashes:

- **Pre-allocate all CUDA buffers at node startup**, not per-frame. Dynamic allocation on the GPU path causes fragmentation and latency spikes.
- **Use CUDA unified memory** (`cudaMallocManaged`) on Jetson, which avoids explicit host-device copies since CPU and GPU share physical memory.
- **Limit TensorRT workspace size** to leave headroom for other processes: `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)` for 512 MB.
- **Monitor memory** with `tegrastats` or `jetson_stats` to track actual usage under load.
- **Containerize inference nodes** with [Docker for Robotics](/wiki/tools/docker-for-robotics/) to isolate memory usage and manage dependencies cleanly.

## Benchmarking and Profiling

Thorough benchmarking is essential to validate that the optimized model meets the robotic system's real-time requirements.

### Key Metrics

| Metric | What It Measures | Target (Typical) |
|---|---|---|
| Latency (p50) | Median inference time | < 30 ms for perception |
| Latency (p99) | Worst-case inference time | < 50 ms for safety-critical |
| Throughput (FPS) | Frames processed per second | ≥ 30 FPS for real-time |
| GPU Memory | Peak GPU memory usage | < 50% of device total |
| Power Draw | Average power consumption | Within platform TDP |

### trtexec: TensorRT's Built-In Benchmarking Tool

`trtexec` is the fastest way to benchmark an ONNX model or TensorRT engine without writing code:

```bash
# Benchmark an ONNX model with FP16 precision
trtexec --onnx=detector.onnx --fp16 --iterations=1000 --warmUp=500

# Benchmark an existing TensorRT engine
trtexec --loadEngine=detector_fp16.engine --iterations=1000

# Benchmark with specific input shape
trtexec --onnx=detector.onnx --fp16 \
    --shapes=images:1x3x640x640 \
    --iterations=1000 --percentile=99
```

`trtexec` reports mean, median, and percentile latencies, along with throughput and GPU compute utilization.

### jetson_stats (jtop)

`jetson_stats` provides real-time monitoring of Jetson platform metrics:

```bash
# Install jetson_stats
pip install jetson-stats

# Launch the interactive dashboard
jtop
```

`jtop` displays CPU/GPU utilization, memory usage, temperature, power draw, and clock frequencies — all critical for understanding whether the system is thermally throttling or memory-starved during inference.

### NVIDIA Nsight Systems

For deep profiling of the inference pipeline, Nsight Systems captures GPU kernel timelines, CUDA API calls, and CPU-GPU synchronization:

```bash
# Profile a Python inference script
nsys profile --trace=cuda,osrt --output=profile_report python inference.py

# Open the trace in the Nsight Systems GUI for visual analysis
nsys-ui profile_report.nsys-rep
```

Look for:
- **Gaps between kernels** indicating CPU bottlenecks or unnecessary synchronization
- **Long memory copies** suggesting suboptimal host-device transfers
- **Kernel occupancy** below 50% suggesting the model is too small to saturate the GPU

## Best Practices and Common Pitfalls

### Best Practices

1. **Always benchmark on the target device.** Desktop GPU performance does not predict Jetson performance. Build and test on the actual hardware.
2. **Start with FP16, try INT8 only if needed.** FP16 provides most of the speedup with no accuracy work. INT8 requires calibration and validation effort.
3. **Use static input shapes when possible.** Dynamic shapes prevent some TensorRT optimizations. If your input resolution is fixed, hardcode it.
4. **Pin your TensorRT version.** Engines are not compatible across TensorRT versions. Lock the version in your Docker container or JetPack release.
5. **Profile the full pipeline, not just inference.** Pre-processing (resize, normalize) and post-processing (NMS, decoding) often dominate end-to-end latency, especially for small models. Use OpenCV's CUDA backend or GPU-accelerated pre-processing.
6. **Set the Jetson power mode appropriately.** Use `sudo nvpmodel -m 0` for maximum performance or tune to balance power and speed. See the [GPU System Setup](/wiki/computing/setup-gpus-for-computer-vision/) guide.

### Common Pitfalls

1. **Building TensorRT engines on the wrong platform.** Engines are tied to the specific GPU architecture and TensorRT version. An engine built on an RTX 4090 will not run on a Jetson Orin.
2. **Ignoring pre/post-processing overhead.** A 5 ms model inference is meaningless if preprocessing takes 20 ms on the CPU. Move preprocessing to the GPU.
3. **Using dynamic shapes unnecessarily.** Dynamic shapes add runtime overhead. Only use them when truly needed (e.g., variable batch size).
4. **Calibrating INT8 with unrepresentative data.** Calibration images must match deployment distribution. Using ImageNet images for a warehouse robot will produce poor quantization scales.
5. **Not validating accuracy after quantization.** Always run the quantized model through your evaluation pipeline. A 2% drop in mAP on a benchmark may translate to critical missed detections in your specific scenario.
6. **Forgetting to warm up the engine.** The first few inferences are slower due to CUDA context initialization and memory allocation. Run 10–50 warm-up inferences before measuring latency.
7. **Blocking the ROS 2 callback thread.** If inference takes longer than the camera frame period, frames will queue up and latency will grow unboundedly. Use separate threads or process the latest frame only.

## Summary

Deploying neural networks on edge devices is a core competency for modern robotics engineers. The pipeline — export to ONNX, optimize with TensorRT or OpenVINO, quantize to FP16 or INT8, and deploy within a ROS 2 node — transforms research models into production-ready perception systems that meet real-time, power, and reliability constraints. Start with FP16 quantization for immediate gains, profile the full pipeline to identify bottlenecks beyond the model itself, and always validate accuracy on representative data from the target deployment environment. As edge hardware continues to advance, these techniques will enable increasingly sophisticated on-device AI for autonomous robotic systems.

## See Also:
- [Jetson Orin AGX](/wiki/computing/jetson-orin-agx/) - Platform setup and optimization for NVIDIA Jetson
- [Setup Your GPU System for Computer Vision](/wiki/computing/setup-gpus-for-computer-vision/) - CUDA, cuDNN, and driver configuration
- [YOLOv5 Training and Deployment on NVIDIA Jetson](/wiki/machine-learning/yolov5-tensorrt/) - End-to-end YOLO deployment with TensorRT
- [Docker for Robotics](/wiki/tools/docker-for-robotics/) - Containerizing robotic applications for reproducible deployments
- [Mediapipe: Live ML Anywhere](/wiki/machine-learning/mediapipe-live-ml-anywhere/) - Lightweight on-device ML pipelines

## Further Reading
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) — Comprehensive documentation covering all TensorRT features, optimization strategies, and API references.
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) — Official documentation for the ONNX Runtime inference engine, including execution providers for different hardware.
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) — Tutorials and pre-built containers for deploying AI models on Jetson platforms.
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) — Guide for deploying models on extremely resource-constrained devices.
- [OpenVINO Toolkit Documentation](https://docs.openvino.ai/) — Intel's toolkit for optimizing and deploying models on Intel hardware including CPUs, GPUs, and VPUs.

## References
- A. Jacob, B. Kligys, B. Chen et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 2704–2713.
- NVIDIA Corporation, "TensorRT: Programmable Inference Accelerator," NVIDIA Developer Documentation, 2024. [Online]. Available: https://developer.nvidia.com/tensorrt
- ONNX Project Contributors, "Open Neural Network Exchange (ONNX)," GitHub Repository, 2024. [Online]. Available: https://github.com/onnx/onnx
- R. Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper," arXiv preprint arXiv:1806.08342, 2018.
- S. Han, H. Mao, and W. J. Dally, "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2016.
