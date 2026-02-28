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

- **[Introduction to Diffusion Models and Diffusion Policy](/wiki/machine-learning/intro-to-diffusion/)**
  Comprehensive introduction to diffusion models and their application in robotics through diffusion policies. Covers ODE and SDE formulations, their practical implications, and how diffusion policies enable multi-modal action learning for complex robotic tasks.

- **[GRPO for Diffusion Policies in Robotics](/wiki/machine-learning/grpo-diffusion-policies/)**
  Introduces Group Relative Policy Optimization (GRPO) and its application to diffusion policies using SDE formulation for stochastic sampling. Covers GRPO's origins in LLMs, the mathematical framework, and practical implementation strategies for optimizing robot policies with reward-based learning.

- **[Mediapipe: Real-Time ML for Robotics](/wiki/machine-learning/mediapipe-live-ml-anywhere/)**
  Introduces MediaPipe for live ML inference on various platforms, including Android, iOS, and IoT. Highlights body pose tracking, hand tracking, and object detection pipelines.

- **[NLP for Robotics](/wiki/machine-learning/nlp_for_robotics/)**
  Explores how natural language processing (NLP) enables robots to understand and respond to human language. Includes an overview of transformer models and HuggingFace library usage.

- **[Python Libraries for Reinforcement Learning](/wiki/machine-learning/python-libraries-for-reinforcement-learning/)**
  A comparison of popular RL libraries like OpenAI’s Spinning Up, Stable Baselines, and RLlib. Provides guidance on choosing the right library based on scalability and ease of use.

- **[YOLO Integration with ROS and GPU Acceleration](/wiki/machine-learning/ros-yolo-gpu/)**
  Step-by-step tutorial for integrating the YOLO object detection framework with ROS. Includes GPU acceleration setup with CUDA for real-time performance.

- **[Training Darknet on Custom Datasets](/wiki/machine-learning/train-darknet-on-custom-dataset/)**
  Explains how to train YOLO (Darknet) models on custom datasets. Covers data preparation, configuration, and tips to improve detection performance.

- **[YOLOv5 on NVIDIA Jetson Platforms](/wiki/machine-learning/yolov5-tensorrt/)**
  Guides training YOLOv5 and deploying it on Jetson edge devices using TensorRT. Includes environment setup, ONNX export, and TensorRT integration with ROS.

## Resources

- [GIMP Installation Guide](https://www.gimp.org/)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Mediapipe Documentation](https://google.github.io/mediapipe/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable Baselines Documentation](https://stable-baselines3.readthedocs.io/)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib.html)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [TensorRT Tutorial](https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/)


