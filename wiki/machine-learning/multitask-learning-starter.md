---
date: 2024-12-22
title: Multi-task learning - A starter guide
---

# Multi-task learning: A starter guide

## Introduction to Multi-Task Learning in Computer Vision

Multi-task learning represents a powerful paradigm in deep learning where a single neural network learns to perform multiple related tasks simultaneously. In computer vision, this approach is particularly valuable because many vision tasks share common low-level features. For instance, both depth estimation and semantic segmentation benefit from understanding edges, textures, and object boundaries in an image.

In this guide, we'll explore how to build a HydraNet architecture that performs two complementary tasks:

1. Monocular depth estimation: Predicting the depth of each pixel from a single RGB image
2. Semantic segmentation: Classifying each pixel into predefined semantic categories

The power of this approach lies in the shared learning of features that are useful for both tasks, leading to more efficient and often more accurate predictions than training separate models for each task.

## Understanding the System Architecture

The HydraNet architecture consists of three main components working in harmony:

### 1. MobileNetV2 Encoder

The encoder serves as the backbone of our network, converting RGB images into rich feature representations. We choose MobileNetV2 for several reasons:

- Efficient design with inverted residual blocks
- Strong feature extraction capabilities
- Lower computational requirements compared to heavier architectures
- Good balance of speed and accuracy

### 2. Lightweight RefineNet Decoder

The decoder takes the encoded features and processes them through refinement stages. Its key characteristics include:

- Chained Residual Pooling (CRP) blocks for effective feature refinement
- Skip connections to preserve spatial information
- Gradual upsampling to restore resolution

### 3. Task-Specific Heads

Two separate heads branch out from the decoder:

- Depth head: Outputs continuous depth values
- Segmentation head: Outputs class probabilities for each pixel

## Detailed Implementation Guide

### 1. Environment Setup and Prerequisites

First, let's understand the constants we'll be using for image processing:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Image normalization constants
IMG_SCALE = 1./255  # Scale pixel values to [0,1]
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))  # ImageNet means
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))   # ImageNet stds
```

These constants are crucial for ensuring our input images match the distribution that our pre-trained MobileNetV2 encoder expects. The normalization process transforms our images to have similar statistical properties to the ImageNet dataset, which helps with transfer learning.

### 2. HydraNet Core Architecture

The HydraNet class serves as our model's foundation. Let's examine its structure in detail:

```python
class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tasks = 2  # Depth estimation and segmentation
        self.num_classes = 6  # Number of segmentation classes
        
        # Initialize network components
        self.define_mobilenet()  # Encoder
        self.define_lightweight_refinenet()  # Decoder
```

This initialization sets up our multi-task framework. The `num_tasks` parameter defines how many outputs our network will produce, while `num_classes` specifies the number of semantic categories for segmentation.

### 3. Understanding the MobileNetV2 Encoder

The encoder uses inverted residual blocks, a key innovation of MobileNetV2. Here's how they work:

```python
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        
        hidden_dim = in_channels * expansion_factor
        
        self.output = nn.Sequential(
            # Step 1: Channel Expansion - Increases the number of channels
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Step 2: Depthwise Convolution - Spatial filtering
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Step 3: Channel Reduction - Projects back to a smaller dimension
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
```

Each inverted residual block performs three key operations:

1. Channel expansion: Increases the feature dimensions to allow for more expressive transformations
2. Depthwise convolution: Applies spatial filtering efficiently by processing each channel separately
3. Channel reduction: Compresses the features back to a manageable size

The name "inverted residual" comes from the fact that the block expands channels before the depthwise convolution, unlike traditional residual blocks that reduce dimensions first.

### 4. Lightweight RefineNet Decoder Deep Dive

The decoder's CRP blocks are crucial for effective feature refinement:

```python
def _make_crp(self, in_planes, out_planes, stages):
    layers = [
        # Initial projection to desired number of channels
        nn.Conv2d(in_planes, out_planes, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    ]
    
    # Create chain of pooling operations
    for i in range(stages):
        layers.extend([
            nn.MaxPool2d(5, stride=1, padding=2),  # Maintains spatial size
            nn.Conv2d(out_planes, out_planes, 1, 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        ])
    
    return nn.Sequential(*layers)
```

The CRP blocks serve several important purposes:

- They capture multi-scale context through repeated pooling operations
- The chain structure allows for refinement of features at different receptive fields
- The 1x1 convolutions after each pooling operation help in feature adaptation
- The residual connections help maintain gradient flow

### 5. Task-Specific Heads in Detail

The heads are designed to transform shared features into task-specific predictions:

```python
def define_heads(self):
    # Segmentation head: Transforms features into class probabilities
    self.segm_head = nn.Sequential(
        nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
        nn.BatchNorm2d(self.feature_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.feature_dim, self.num_classes, 1)
    )
    
    # Depth head: Transforms features into depth values
    self.depth_head = nn.Sequential(
        nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
        nn.BatchNorm2d(self.feature_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.feature_dim, 1, 1)
    )
```

Each head follows a similar structure but serves different purposes:

- The segmentation head outputs logits for each class at each pixel
- The depth head outputs a single continuous value per pixel
- The 3x3 convolution captures local spatial context
- The final 1x1 convolution projects to the required output dimensions

### 6. Forward Pass and Loss Functions

The forward pass coordinates the flow of information through the network:

```python
def compute_loss(depth_pred, depth_gt, segm_pred, segm_gt, weights):
    # Depth loss: L1 loss for continuous values
    depth_loss = F.l1_loss(depth_pred, depth_gt)
    
    # Segmentation loss: Cross-entropy for classification
    segm_loss = F.cross_entropy(segm_pred, segm_gt)
    
    # Weighted combination of losses
    total_loss = weights['depth'] * depth_loss + weights['segm'] * segm_loss
    return total_loss
```

The loss function balancing is crucial for successful multi-task learning:

- The depth loss measures absolute differences in depth predictions
- The segmentation loss measures classification accuracy
- The weights help balance the contribution of each task
- These weights can be fixed or learned during training

## Training Considerations and Best Practices

When training a multi-task model like HydraNet, several factors require careful attention:

### 1. Data Balancing

- Ensure both tasks have sufficient and balanced training data
- Consider the relative difficulty of each task
- Use appropriate data augmentation for each task

### 2. Loss Balancing

- Monitor individual task losses during training
- Adjust task weights if one task dominates
- Consider uncertainty-based loss weighting

### 3. Optimization Strategy

- Start with lower learning rates
- Use appropriate learning rate scheduling
- Monitor task-specific metrics separately
- Implement early stopping based on validation performance

## Conclusion

The HydraNet architecture demonstrates the power of multi-task learning in computer vision. By sharing features between depth estimation and segmentation tasks, we achieve:

- More efficient use of model parameters
- Better generalization through shared representations
- Fast inference times suitable for real-world applications

Success with this architecture requires careful attention to implementation details, particularly in the areas of loss balancing, training dynamics, and architecture design. The code provided here serves as a foundation that can be adapted and extended based on specific requirements and constraints.