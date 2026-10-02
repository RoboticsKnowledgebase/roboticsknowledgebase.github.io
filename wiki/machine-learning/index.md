---
date: 2024-12-05
title: Machine Learning
---
<!-- **This page is a stub.** You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning} -->

This section is **not** an overview of machine learning (ML) techniques nor an introduction to the field of machine learning. Instead, it focuses on the **use cases of ML in robotics** and specific implementations where ML is applied to enhance robotic systems. From perception to decision-making, this section explores practical guides and real-world applications.

This section demonstrates how machine learning enhances robotic systems by enabling functionalities like object detection, natural language understanding, and decision-making. By focusing on specific implementations and integration techniques, it bridges the gap between theoretical ML concepts and practice.

## Key Subsections and Highlights

- **[Creating Custom Semantic Segmentation Data](/wiki/machine-learning/custom-semantic-data/)**
  Explains how to prepare and annotate custom datasets for semantic segmentation tasks. Includes guidelines for outsourcing labeling tasks and examples using GIMP.

- **[Introduction to Reinforcement Learning](/wiki/machine-learning/intro-to-rl/)**
  Covers reinforcement learning concepts and Bellman equations. Discusses methods like dynamic programming, Monte Carlo, and temporal difference learning, with an emphasis on robotic applications.

- **[Generative Modeling](/wiki/machine-learning/generative-modeling/)**
  High-level overview of major generative modeling families including VAEs, GANs, flow-based models, diffusion models, and autoregressive approaches.

- **[Offline Reinforcement Learning](/wiki/machine-learning/offline-rl/)**
  Introduces offline RL, why fixed-dataset learning matters in robotics, and commonly used methods such as CQL, behavior regularization, and IQL.

- **[Introduction to Diffusion Models and Diffusion Policy](/wiki/machine-learning/intro-to-diffusion/)**
  Comprehensive introduction to diffusion models and their application in robotics through diffusion policies. Covers ODE and SDE formulations, their practical implications, and how diffusion policies enable multi-modal action learning for complex robotic tasks.

- **[GRPO for Diffusion Policies in Robotics](/wiki/machine-learning/grpo-diffusion-policies/)**
  Introduces Group Relative Policy Optimization (GRPO) and its application to diffusion policies using SDE formulation for stochastic sampling. Covers GRPO's origins in LLMs, the mathematical framework, and practical implementation strategies for optimizing robot policies with reward-based learning.

- **[Mediapipe: Real-Time ML for Robotics](/wiki/machine-learning/mediapipe-live-ml-anywhere/)**
  Introduces MediaPipe for live ML inference on various platforms, including Android, iOS, and IoT. Highlights body pose tracking, hand tracking, and object detection pipelines.

- **[NLP for Robotics](/wiki/machine-learning/nlp-for-robotics/)**
  Explores how natural language processing (NLP) enables robots to understand and respond to human language. Includes an overview of transformer models and HuggingFace library usage.

- **[Python Libraries for Reinforcement Learning](/wiki/machine-learning/python-libraries-for-reinforcement-learning/)**
  A comparison of popular RL libraries like OpenAI’s Spinning Up, Stable Baselines, and RLlib. Provides guidance on choosing the right library based on scalability and ease of use.

- **[YOLO Integration with ROS and GPU Acceleration](/wiki/machine-learning/ros-yolo-gpu/)**
  Step-by-step tutorial for integrating the YOLO object detection framework with ROS. Includes GPU acceleration setup with CUDA for real-time performance.

- **[Training Darknet on Custom Datasets](/wiki/machine-learning/train-darknet-on-custom-dataset/)**
  Explains how to train YOLO (Darknet) models on custom datasets. Covers data preparation, configuration, and tips to improve detection performance.

- **[YOLOv5 on NVIDIA Jetson Platforms](/wiki/machine-learning/yolov5-tensorrt/)**
  Guides training YOLOv5 and deploying it on Jetson edge devices using TensorRT. Includes environment setup, ONNX export, and TensorRT integration with ROS.

- **[Distributed Training With PyTorch](/wiki/machine-learning/distributed-training-with-pytorch/)**
  A tutorial on scaling deep learning models across multiple GPUs and nodes using PyTorch's `DistributedDataParallel` (DDP).

- **[Imitation Learning With a Focus on Humanoids](/wiki/machine-learning/imitation-learning/)**
  Covers the foundations of imitation learning and provides a practical guide for data collection and policy deployment on humanoid robots.

- **[6D Pose Estimation with YOLO and ZED](/wiki/machine-learning/6d-pose-tracking-yolo-zed/)**
  Practical implementation notes for fusing YOLO detections with ZED depth data for 6D pose and velocity tracking workflows.

- **[Practical Guide to Model Quantization and TensorRT Optimization](/wiki/machine-learning/practical-guide-to-model-quantization-and-tensorrt-optimization/)**
  Techniques for reducing model footprint and accelerating inference with quantization strategies and TensorRT deployment.

- **[Installing YOLO on ARM Architecture Devices](/wiki/machine-learning/installing-yolo-on-arm-architecture-devices/)**
  Deployment-focused guide for getting YOLO running efficiently on ARM-based platforms.

- **[Comprehensive Guide to Albumentations](/wiki/machine-learning/comprehensive-guide-to-albumentations/)**
  Overview of augmentation pipelines and practical transform choices for robust vision model training.

- **[Kornia Technical Guide](/wiki/machine-learning/kornia-technical-guide/)**
  Technical overview of using Kornia for differentiable computer vision operations inside deep learning pipelines.

- **[Integrating OLLAMA LLMs with Franka Arm](/wiki/machine-learning/integrating-ollama-llms-with-franka-arm/)**
  Integration approach for connecting local LLM inference through OLLAMA to Franka robot control workflows.

- **[Multi-task Learning: A Starter Guide](/wiki/machine-learning/multitask-learning-starter/)**
  Introductory guide to shared-representation multi-task model design and training considerations.

- **[Understanding Kalman Filters and Visual Tracking](/wiki/machine-learning/understanding-kalman-filters-and-visual-tracking/)**
  Explains Kalman filtering fundamentals and their practical role in visual tracking pipelines.

- **[Knowledge Distillation Practical Implementation Guide](/wiki/machine-learning/knowledge-distillation-practical-implementation-guide/)**
  Applied walkthrough for transferring performance from larger teacher models into smaller deployable students.

- **[Neural Network Optimization Using Model Pruning](/wiki/machine-learning/neural-network-optimization-using-model-pruning/)**
  Practical pruning methods to reduce model complexity while preserving task performance.

- **[Deep Learning Techniques for 3D Datasets](/wiki/machine-learning/deep-learning-techniques-for-3d-datasets/)**
  Survey of methods and implementation patterns for deep learning over point clouds and other 3D data modalities.

- **[Optical Flow: Classical to Deep Learning Implementation](/wiki/machine-learning/optical-flow-classical-to-deep-learning-implementation/)**
  Comparative guide from traditional optical-flow methods to modern deep-learning implementations.

## Resources

- [GIMP Installation Guide](https://www.gimp.org/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Mediapipe Documentation](https://google.github.io/mediapipe/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable Baselines Documentation](https://stable-baselines3.readthedocs.io/)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib.html)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [TensorRT Tutorial](https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/)
