---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-04-23 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Distributed Training With PyTorch Tutorial
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

Deep learning applications, being fundamentally data-driven, require substantial datasets for effective training. In 2025, the robotics field has widely adopted deep learning technologies, particularly for vision systems that enable semantic understanding of the environment in which it operates. Vision Transformers (ViT) have become the industry standard, effectively replacing Convolutional Neural Networks (CNNs) in most applications. However, organizations/teams frequently face resource constraints regarding high-performance GPUs with sufficient Video RAM (VRAM).
### The Multi-GPU Solution
When multiple smaller-VRAM GPUs are configured in a daisy-chain arrangement on a single motherboard or across a network (such as in a cluster), they can be utilized collectively within a single training pipeline. PyTorch provides built-in functionality for training models, accumulating gradients, and performing backpropagation across multiple GPU units.
### Purpose of This Tutorial
While PyTorch offers multi-GPU capabilities, implementing these features effectively on specific hardware configurations presents significant challenges. This tutorial provides step-by-step guidance for configuring and optimizing distributed GPU training for deep learning applications in robotics. For guidance on data curation methodologies, please refer to our "Machine Learning" section within this wiki.

## Introduction:

In this tutorial, we will cover:

1. Why distributed training is essential for modern deep learning workloads.
   
2. Few Things to Consider Before Opting To Parallelise your trianing.

3. How to use `DistributedDataParallel` (DDP) for efficient training on multiple GPUs and multiple nodes.

4. Code and explanation

5. Key best practices, caveats, and debugging strategies to ensure efficient and correct training.
  
6. The fundamentals of model parallelism—useful when your model doesn’t fit on a single GPU.

7. Summary



This tutorial is intended for those already comfortable with PyTorch who want to scale their models to use multiple GPUs or machines.


### Why Distributed Training?

What is the problem?

* Deep learning models have grown so large they can (seldom) no longer fit on a single GPU.

* Training time increases dramatically with data and model size.

* You may need to train across multiple GPUs or machines to make training feasible in hours or days instead of weeks.

How does PyTorch help?

PyTorch offers two main ways to parallelize training:

* Data Parallelism: Replicate the model across devices, and feed each copy a different data slice.

* Model Parallelism: Split the model architecture itself across devices.

## Few Things to Consider Before Opting to Parallelise Your Training

### Memory Optimization Strategies for Deep Learning Training
### Understanding Batch Size Optimization
When facing memory constraints during model training, your first approach should be systematically reducing the batch size. However, this reduction should be executed with careful consideration of model characteristics and training dynamics.
For large-scale models exceeding 50 million learnable parameters, it is strongly recommended to maintain a minimum batch size of 8 samples per iteration. This threshold is not arbitrary but grounded in statistical learning principles.
### The Critical Importance of Adequate Batch Sizing
The rationale behind this recommendation stems from the behavior of normalization layers, particularly BatchNormalization. When batch sizes become excessively small (such as 1 or 2 samples), the following training inefficiencies emerge:

* Gradient Direction Instability: With minimal samples per batch, gradients become disproportionately influenced by individual training examples, creating erratic update patterns.
* Statistical Inconsistency: BatchNormalization depends on reliable batch statistics to function effectively. Extremely small batches generate unreliable statistical estimates, compromising the normalization process.
* Optimization Landscape Challenges: Limited batch diversity significantly increases the probability of convergence toward suboptimal local minima, potentially undermining model performance.

### Alternative Memory Optimization Approaches
If reducing batch size to 8 still exceeds your GPU memory capacity, consider implementing these secondary optimization strategies:
* Input Dimensionality Reduction
Modify your input preprocessing pipeline to decrease memory requirements per sample. For image-based models, implement resolution reduction through PyTorch's transformation modules (specifically Resize operations). This approach trades off some spatial information for memory efficiency.
* Performance Validation Requirements
It is imperative to benchmark your memory-optimized model against established performance metrics. Thoroughly compare your results with those reported in the original repository documentation, particularly focusing on:

* Evaluation dataset performance metrics
* Convergence rate characteristics
* Generalization capabilities

### Decision Point
Should you observe a substantial degradation in model performance or continue to encounter memory limitations despite these optimization efforts, proceed with the distributed training implementation detailed in the remainder of this tutorial.

The distributed GPU approach provides a sophisticated solution that maintains training integrity while effectively circumventing hardware constraints through parallelized computation.

## DistributedDataParallel (DDP) — The Right Way to Scale Training

`DistributedDataParallel` (DDP) is PyTorch’s preferred way to scale training across multiple GPUs and nodes. Each GPU runs in its own process and maintains its own model replica. Gradients are synchronized after the backward pass.

### Key Distributed Computing Concepts
### What is a Node?
A "node" represents a physical machine or server in your computing infrastructure. Each node may contain multiple GPUs. For example, a computing cluster might have 4 nodes, where each node is a separate physical server with its own processors, memory, and GPUs.
### What is World Size?
"World size" refers to the total number of parallel processes participating in your distributed training job. In most common configurations, this equals the total number of GPUs being utilized across all nodes. For instance, if you're using 4 GPUs on a single machine, your world size would be 4.
### What is Rank?
"Rank" is a unique identifier assigned to each process in your distributed training setup. Ranks start at 0 and extend to (world_size - 1). Each process typically manages one GPU. The process with rank 0 often serves as the "master" process that coordinates certain activities like initialization and saving checkpoints.
### What is a Backend?
The "backend" specifies the communication protocol used for inter-process message passing. PyTorch supports several backend options:

* NCCL (NVIDIA Collective Communications Library): Optimized for NVIDIA GPUs, offering the best performance for GPU-to-GPU communication.
* Gloo: A general-purpose backend that works on both CPU and GPU, with good compatibility across platforms.
* MPI (Message Passing Interface): A standardized communication protocol used in high-performance computing.

For GPU-based training, NCCL is almost always the preferred choice due to its superior performance characteristics.

Setting Up a DDP Training Loop (Single Node):

```import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost" # Sets the address of the coordinating process. "localhost" indicates all processes are on the same machine.
    os.environ["MASTER_PORT"] = "12355" # Designates a network port for inter-process communication.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            ...
            loss = compute_loss(ddp_model(batch))
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
```

Launching with torchrun:
```
torchrun --nproc_per_node=4 train.py
```
### Process Group Initialization Function
```
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

This function configures the distributed environment:

* os.environ["MASTER_ADDR"] = "localhost": Sets the address of the coordinating process. "localhost" indicates all processes are on the same machine.
* os.environ["MASTER_PORT"] = "12355": Designates a network port for inter-process communication.
* dist.init_process_group("nccl", rank=rank, world_size=world_size): Initializes the process group with the NCCL backend, assigning the current process its rank and informing it of the total world size.
* torch.cuda.set_device(rank): Maps this process to a specific GPU. When running on a multi-GPU system, each process is typically assigned to a different GPU, and the rank value conveniently serves as the GPU index.

### Training Function
```
def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            ...
            loss = compute_loss(ddp_model(batch))
            loss.backward()
            optimizer.step()

    cleanup()
```

This is the core training function that each process executes:

* setup(rank, world_size): Initializes the distributed environment as described above.
* model = MyModel().to(rank): Creates a model instance and moves it to the appropriate GPU (identified by rank).
* ddp_model = DDP(model, device_ids=[rank]): Wraps the model with DDP, which handles the synchronization of gradients across processes during backward passes.
* sampler = torch.utils.data.distributed.DistributedSampler(...): Creates a special sampler that ensures each process works on a different subset of the data, preventing redundant computation:

  * num_replicas=world_size: Tells the sampler how many parallel processes exist.
  * rank=rank: Informs the sampler which specific subset of data this process should handle.


* dataloader = torch.utils.data.DataLoader(...): Creates a DataLoader with the distributed sampler.
* sampler.set_epoch(epoch): Critical for shuffling data differently across epochs while maintaining process data separation.

Training loop:

  * Forward pass through the DDP-wrapped model.
  * Loss computation and backward pass (loss.backward()), during which DDP automatically synchronizes gradients across all processes.
  * Optimizer step to update model parameters.

## Best Practices for Distributed Training

* Always prefer DDP over DataParallel.

* Use DistributedSampler to avoid duplicated data across processes.

* Set the epoch on the sampler each iteration to enable proper data shuffling.

* BatchNorm layers can cause issues—use SyncBatchNorm for consistency. (replace BatchNorm layers in your model with SynBatchNorm)

* Save checkpoints using if rank == 0: to avoid duplicate files.

* Monitor training with per-rank logging or only on rank 0.

## Model Parallelism — Splitting the Model Across Devices

GPUs to utilize hardware efficiently.
Example: Manual Layer Partitioning

```
class ModelParallelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Linear(1024, 2048).to('cuda:0')
        self.relu = nn.ReLU().to('cuda:0')
        self.part2 = nn.Linear(2048, 1024).to('cuda:1')

    def forward(self, x):
        x = self.part1(x.to('cuda:0'))
        x = self.relu(x)
        x = x.to('cuda:1')
        x = self.part2(x)
        return x
```
Here, the input is first processed by part1 and a ReLU on cuda:0, then the tensor is moved to cuda:1 before part2 processes it.

Practical Considerations:

* Communication Overhead: Manually moving tensors between devices (.to('cuda:x')) can cause latency. For performance, it's important to minimize these transfers and overlap communication with computation where possible.

* Asynchronous Execution: PyTorch's CUDA operations are asynchronous. Use torch.cuda.synchronize() carefully when benchmarking.

* Memory Bottlenecks: While model parallelism solves GPU memory limits, it may introduce underutilization due to idle GPUs waiting on other GPUs to finish processing.

Usually this is not a preferred optimization method as it is tough to set up, decide which layers go to which GPU and accumulate gradients manually. At this point it is better to choose a lighter model for your task. But, if you want to learn more please check the link in the Further Reading section. 

## Summary 
You have been introduced to the concept of distributed training and when to use it. You have also read about the importance of distributed training. Different types of distributed training, best practices, caveats and some code to help you get started with the implementation. 

### See Also
Dataset curation for semantic segmentation:
- [Custom data-set for segmentation](https://roboticsknowledgebase.com/wiki/machine-learning/custom-semantic-data/)

## Further Reading
You may come across `DataParallel`, we have covered only `DistributedDataParallel`. 
- [Quick excerpt on DataParallel vs DistributedDataParallel](https://discuss.pytorch.org/t/dataparallel-vs-distributeddataparallel/77891)
- [Useful article with visualizations on DataParallel vs DistributedDataParallel](https://medium.com/@mlshark/dataparallel-vs-distributeddataparallel-in-pytorch-whats-the-difference-0af10bb43bc7)


- A more comprehensive guide on distributed training with PyTorch
[Getting Started with PyTorch Distributed](https://medium.com/red-buffer/getting-started-with-pytorch-distributed-54ae933bb9f0) 
- Official documentation for the RPC framework, which is now the default way to help "model-parallelise" your training.
[RPC Framework](https://pytorch.org/docs/main/rpc.html)
- Tutorial on how to combine RPC framework with DDP to train on large clusters, think in the order of 100s of GPUs
[Combining Distributed DataParallel with Distributed RPC Framework](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html#)
