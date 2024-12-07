# Optical Flow: Classical to Deep Learning Implementation

## Introduction

Optical flow represents one of the foundational challenges in computer vision: how do we track the motion of objects between frames? When you watch a video, your brain effortlessly tracks the movement of objects across frames. Implementing this computationally requires sophisticated algorithms that can detect and quantify motion at the pixel level.

## Classical Methods and Their Mathematics

### The Lucas-Kanade Method

The Lucas-Kanade algorithm approaches optical flow through a fundamental equation that relates pixel intensity changes to motion. The algorithm is built on two key assumptions:

1. **Brightness Constancy**: A pixel maintains its intensity as it moves
2. **Spatial Coherence**: Nearby pixels move similarly

These assumptions lead to the optical flow equation:
```
Ix * u + Iy * v + It = 0
```
where (u,v) represents the flow vector we want to compute.

Here's the implementation with detailed breakdown:

```python
def lucas_kanade_flow(I1, I2, window_size=15):
    # Compute spatial and temporal gradients
    Ix = cv2.Sobel(I1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I1, cv2.CV_64F, 0, 1, ksize=3)
    It = I2.astype(np.float32) - I1.astype(np.float32)
    
    # Solve for each pixel in window
    u = np.zeros_like(I1, dtype=np.float32)
    v = np.zeros_like(I1, dtype=np.float32)
    
    for i in range(window_size//2, I1.shape[0]-window_size//2):
        for j in range(window_size//2, I1.shape[1]-window_size//2):
            # Extract window gradients
            ix = Ix[i-window_size//2:i+window_size//2+1,
                   j-window_size//2:j+window_size//2+1].flatten()
            iy = Iy[i-window_size//2:i+window_size//2+1,
                   j-window_size//2:j+window_size//2+1].flatten()
            it = It[i-window_size//2:i+window_size//2+1,
                   j-window_size//2:j+window_size//2+1].flatten()
            
            # Construct system of equations
            A = np.vstack([ix, iy]).T
            b = -it
            
            # Solve least squares
            if np.min(np.linalg.eigvals(A.T @ A)) >= 1e-6:
                nu = np.linalg.solve(A.T @ A, A.T @ b)
                u[i,j], v[i,j] = nu
    
    return u, v
```

This implementation:
1. Computes image gradients using Sobel operators (Ix, Iy) and frame difference (It)
2. For each pixel, considers a window of surrounding pixels
3. Solves a least squares problem to find the motion vector
4. Checks eigenvalues to ensure the solution is well-conditioned

### The Farnebäck Method

Farnebäck's algorithm represents a more sophisticated classical approach that can handle larger motions by using polynomial expansion to approximate pixel neighborhoods:

```python
def farneback_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr,
        None,
        pyr_scale=0.5,  # Pyramid scale
        levels=3,       # Pyramid levels
        winsize=15,     # Window size
        iterations=3,   # Iterations per level
        poly_n=5,       # Polynomial expansion neighborhood
        poly_sigma=1.2, # Gaussian sigma
        flags=0
    )
    return flow
```

The key parameters control:

1. **Multi-scale Analysis**:
   - `pyr_scale`: Controls pyramid scale reduction (0.5 means each level is half the size)
   - `levels`: Number of pyramid levels (more levels handle larger motions)

2. **Local Approximation**:
   - `winsize`: Size of neighborhood for polynomial expansion
   - `poly_n`: Size of neighborhood used for polynomial approximation
   - `poly_sigma`: Gaussian smoothing for polynomial coefficients

3. **Refinement**:
   - `iterations`: Number of iterations at each pyramid level

## Deep Learning Approaches

### FlowNet: End-to-End Flow Estimation

FlowNet revolutionized optical flow by showing that deep networks could learn to estimate flow directly from data. The architecture processes concatenated frames through an encoder-decoder structure:

```python
class FlowNetS(nn.Module):
    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()
        
        # Encoder
        self.conv1 = conv(batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(batchNorm, 128, 256, kernel_size=5, stride=2)
        
        # Decoder with skip connections
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        
        # Flow prediction
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
```

The architecture consists of:

1. **Encoder Path**:
   - Takes 6-channel input (concatenated RGB frames)
   - Progressive downsampling with increasing feature channels
   - Large initial kernels capture substantial motions
   - Batch normalization stabilizes training

2. **Decoder Path**:
   - Upsampling through deconvolution layers
   - Skip connections preserve fine details
   - Channel counts include flow predictions (e.g., 1026 = 1024 + 2)

3. **Multi-scale Prediction**:
   - Flow predicted at multiple resolutions
   - Coarse predictions handle large motions
   - Fine predictions refine details
   - Loss computed at all scales

### RAFT Architecture

RAFT (Recurrent All-Pairs Field Transforms) represents the current state-of-the-art through iterative refinement:

```python
class RAFTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.conv2 = nn.Conv2d(256, 256, 1)
        
    def forward(self, x):
        # Extract features at 1/8 resolution
        x = self.backbone(x)
        # Split into feature and context networks
        feat = self.conv1(x)
        ctx = self.conv2(x)
        return feat, ctx
```

RAFT innovates through:

1. **Feature Extraction**:
   - Shared backbone network (ResNet18) processes both frames
   - Separate feature and context pathways
   - Features optimized for correlation computation
   - Context provides additional motion information

2. **All-Pairs Correlation**:
```python
def compute_correlation_volume(feat1, feat2, num_levels=4):
    """Compute 4D correlation volume"""
    b, c, h, w = feat1.shape
    feat2 = feat2.view(b, c, h*w)
    
    # Compute correlation for all pairs
    corr = torch.matmul(feat1.view(b, c, h*w).transpose(1, 2), feat2)
    corr = corr.view(b, h, w, h, w)
    
    # Create correlation pyramid
    corr_pyramid = []
    for i in range(num_levels):
        corr_pyramid.append(F.avg_pool2d(
            corr.view(b*h*w, 1, h, w), 
            2**i+1, 
            stride=1, 
            padding=2**i//2
        ))
    
    return corr_pyramid
```

This creates a 4D correlation volume that:
- Captures all possible matches between frames
- Enables large displacement handling
- Provides multi-scale correlation information

3. **Iterative Updates**:
```python
class RAFTUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = ConvGRU(hidden_dim=128)
        self.flow_head = FlowHead(hidden_dim=128)
        
    def forward(self, net, inp, corr, flow):
        # Update hidden state using correlation and context
        net = self.gru(net, inp, corr)
        # Predict flow update
        delta_flow = self.flow_head(net)
        return net, flow + delta_flow
```

The updater:
- Maintains flow estimate in hidden state
- Refines estimate through multiple iterations
- Uses GRU for temporal coherence
- Predicts incremental updates

## Training and Evaluation

### Loss Functions

The standard metric for optical flow is the EndPoint Error (EPE):

```python
def endpoint_error(pred_flow, gt_flow):
    """
    Calculate average end-point error
    pred_flow, gt_flow: Bx2xHxW tensors
    """
    # Compute per-pixel euclidean distance
    epe = torch.norm(pred_flow - gt_flow, p=2, dim=1)
    # Return mean error
    return epe.mean()
```

For multi-scale training, we use a weighted combination:

```python
def multiscale_loss(flow_preds, flow_gt, weights):
    """
    Compute weighted loss across multiple scales
    """
    loss = 0
    for flow, weight in zip(flow_preds, weights):
        # Downsample ground truth to match prediction
        scaled_gt = F.interpolate(
            flow_gt, 
            size=flow.shape[-2:], 
            mode='bilinear'
        )
        # Compute EPE at this scale
        loss += weight * endpoint_error(flow, scaled_gt)
    return loss
```

## Conclusion

The evolution of optical flow algorithms shows a clear progression:
1. Classical methods built on mathematical principles and assumptions
2. Early deep learning replaced hand-crafted features with learned ones
3. Modern architectures like RAFT combine learning with sophisticated architectural designs

Each approach offers different trade-offs between:
- Accuracy vs. computational cost
- Large vs. small motion handling
- Training data requirements
- Real-time performance capabilities

Choose your method based on your specific requirements for these factors.