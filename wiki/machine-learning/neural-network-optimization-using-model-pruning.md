# Neural Network optimization using model pruning

## Understanding Model Pruning

Model pruning is a fundamental technique in deep learning model optimization where we systematically remove weights or neurons from a neural network while maintaining its performance. This process is analogous to biological neural pruning, where the brain eliminates less important neural connections to improve efficiency.

## Theoretical Foundation

Neural networks typically contain redundant parameters that contribute minimally to the model's outputs. Pruning identifies and removes these parameters by:

1. Evaluating parameter importance using specific criteria
2. Removing parameters deemed less important
3. Fine-tuning the remaining parameters to maintain performance

## Implementation with PyTorch

Let's explore different pruning techniques using PyTorch's pruning utilities:

```python
import torch.nn.utils.prune as prune
import torch.nn as nn
from copy import deepcopy
import numpy as np
```

### Basic Pruning Setup

First, let's create a simple linear layer to demonstrate pruning concepts:

```python
# Create a test module
fc_test = nn.Linear(10, 10)
module = deepcopy(fc_test)

# Examine initial parameters
print('Before pruning:')
print(list(module.named_parameters()))
print(list(module.named_buffers()))
```

## Unstructured Pruning

### L1 Unstructured Pruning

L1 unstructured pruning removes individual weights based on their absolute magnitude. This is the most flexible form of pruning but results in sparse matrices that may not provide practical speed benefits without specialized hardware.

```python
def apply_l1_unstructured_pruning(module, amount=0.3):
    """
    Apply L1 unstructured pruning to a module
    
    Args:
        module: PyTorch module to prune
        amount: Fraction of weights to prune (0.3 = 30%)
    """
    prune.l1_unstructured(module, name='weight', amount=amount)
    
    # Examine the pruned weights
    weight = module.weight.cpu().detach().numpy()
    mask = module.get_buffer('weight_mask').cpu().numpy()
    
    return weight, mask
```

The process works by:
1. Computing the L1 norm (absolute values) of all weights
2. Sorting weights by magnitude
3. Setting the smallest weights to zero based on the specified amount

### Visualizing Unstructured Pruning

```python
def visualize_pruning_pattern(weight, mask, title):
    """
    Visualize weight matrix before and after pruning
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original weights
    im1 = ax1.imshow(weight, cmap='viridis')
    ax1.set_title('Original Weights')
    
    # Pruned weights
    pruned_weight = weight * mask
    im2 = ax2.imshow(pruned_weight, cmap='viridis')
    ax2.set_title(f'After {title}')
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    return fig
```

## Structured Pruning

### L1 Structured Pruning

Structured pruning removes entire groups of weights (e.g., neurons or channels) based on their collective importance. This approach results in dense but smaller matrices that can provide immediate speed benefits.

```python
def apply_l1_structured_pruning(module, amount=0.3, dim=0):
    """
    Apply L1 structured pruning to a module
    
    Args:
        module: PyTorch module to prune
        amount: Fraction of structures to prune
        dim: Dimension along which to prune (0=rows, 1=columns)
    """
    prune.ln_structured(
        module, 
        name='weight', 
        amount=amount, 
        n=1,  # L1 norm
        dim=dim
    )
    
    return module.weight.cpu().detach().numpy()
```

The process works by:
1. Computing the L1 norm of each structure (row/column)
2. Sorting structures by their total magnitude
3. Removing entire structures with lowest magnitude

## Advanced Pruning Techniques

### Iterative Pruning

Iterative pruning gradually removes weights over multiple rounds, allowing the network to adapt:

```python
def iterative_pruning(model, pruning_schedule, fine_tune_steps=1000):
    """
    Iteratively prune a model according to a schedule
    
    Args:
        model: PyTorch model to prune
        pruning_schedule: List of (epoch, amount) tuples
        fine_tune_steps: Number of steps to fine-tune after each pruning
    """
    for epoch, amount in pruning_schedule:
        # Apply pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, 'weight', amount=amount)
        
        # Fine-tune
        fine_tune_model(model, steps=fine_tune_steps)
```

### Global Pruning

Instead of pruning each layer independently, global pruning considers the importance of weights across the entire network:

```python
def global_magnitude_pruning(model, amount):
    """
    Prune weights globally across the model based on magnitude
    
    Args:
        model: PyTorch model to prune
        amount: Fraction of weights to prune globally
    """
    # Collect all weights
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            all_weights.extend(module.weight.data.abs().cpu().numpy().flatten())
    
    # Compute global threshold
    threshold = np.percentile(all_weights, amount * 100)
    
    # Apply pruning
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            mask = module.weight.data.abs() > threshold
            module.weight.data *= mask
```

## Best Practices for Model Pruning

### 1. Pruning Strategy Selection

Choose your pruning strategy based on your requirements:

```python
def select_pruning_strategy(model_type, hardware_target):
    """
    Select appropriate pruning strategy based on model and hardware
    """
    if hardware_target == 'gpu':
        return 'structured'  # Better for parallel processing
    elif hardware_target == 'sparse_accelerator':
        return 'unstructured'  # Better for specialized hardware
    else:
        return 'structured'  # Default to structured for general purpose
```

### 2. Performance Monitoring

Monitor key metrics during pruning:

```python
def evaluate_pruning(model, test_loader, original_accuracy):
    """
    Evaluate the impact of pruning
    """
    metrics = {
        'accuracy': compute_accuracy(model, test_loader),
        'model_size': get_model_size(model),
        'inference_time': measure_inference_time(model),
        'compression_ratio': compute_compression_ratio(model)
    }
    
    return metrics
```

## Conclusion

Effective model pruning requires:

1. Understanding different pruning techniques and their trade-offs
2. Careful selection of pruning parameters and schedules
3. Proper monitoring of model performance during pruning
4. Consideration of hardware constraints and deployment targets

When implemented correctly, pruning can significantly reduce model size and improve inference speed while maintaining most of the original model's accuracy.