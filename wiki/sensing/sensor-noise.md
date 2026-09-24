---
date: 2025-10-01
title: Reducing Sensor Noise in Thermal or Visual Imaging sensors
published: true
---


# Dealing with Visual/Thermal Sensor Noise

Imaging sensors are very powerful and allow for a wide variety of use cases using the same hardware. An imaging sensor could range from a thermal camera that operates in the Long Wavelength Infrared (LWIR) to simple RGB cameras. Cameras can be cheap sensors (such as [this \$11 RGB camera](https://www.waveshare.com/ov9655-camera-board.htm)) or the [\$7000 FLiR Boson LWIR Thermal Cameras](https://oem.flir.com/products/boson-plus/?vertical=lwir&segment=oem). These imaging sensors can be used to provide depth, optical flow, object detection, object recognition, surface reconstruction, and many other tasks. Imaging systems form a backbone of many robotics systems but frequently struggle with noise and ill effects that are not representative of the real world. Take for example your phone's camera trying to take a picture on a no moon night, a thermal camera operating in an environment with little temperature difference, or a cracked lens. While some are easier to deal with, others are difficult to remedy.

This guide covers common noise types encountered in both visual and thermal imaging sensors and practical techniques to mitigate them. While the examples apply broadly, thermal cameras often experience unique challenges due to their sensitivity to temperature variations and calibration drift.

## Different types of noise

### Salt and Pepper Noise

Salt and pepper noise manifests as randomly scattered bright (salt) and dark (pepper) pixels throughout an image. This type of noise is particularly common in thermal sensors experiencing electronic interference or faulty pixel readouts, and in RGB cameras with sensor defects or transmission errors.

**Characteristics:**
- Appears as isolated white and black pixels randomly distributed across the image
- More prevalent in low-cost sensors or those operating at extreme temperatures
- Can be caused by dead or hot pixels on the sensor array
- Thermal cameras may exhibit this during rapid temperature transitions

**Impact on applications:** This noise can confuse edge detection algorithms, create false features in object detection pipelines, and interfere with visual odometry or optical flow calculations. This kind of noise will cause issues in feature matching and gradient calculations.

### Graininess (Gaussian Noise)

Graininess appears as a uniform speckled texture across the entire image, resembling film grain. This is caused by random variations in sensor readout and is typically modeled as Gaussian (normal) distribution noise around the true pixel intensity.

**Characteristics:**
- Increases with higher ISO settings (increased sensor gain) in RGB cameras
- More pronounced in low-light conditions where signal-to-noise ratio is poor
- Thermal cameras exhibit this when observing scenes with minimal temperature gradients
- Affects all pixels but with varying intensity

**Impact on applications:** Graininess reduces feature matching accuracy, degrades template matching performance, and can cause drift in SLAM systems. In thermal imaging, it makes it difficult to distinguish subtle temperature differences.

### Vignetting

Vignetting is the gradual darkening of image corners and edges relative to the center. It's a problem for robotic vision systems that assume uniform illumination and causes poor performance when objects are close to the edges. Vignetting arises because light spreads out more as we go away from the principle axis of the camera.

**Characteristics:**
- Radial intensity falloff from image center to corners
- Can be caused by lens design limitations, lens hood interference, or sensor geometry
- More pronounced at wider apertures
- Thermal cameras can exhibit vignetting due to optical design constraints

**Impact on applications:** Vignetting interferes with photometric calibration, creates artifacts in image stitching and panoramas, and biases brightness-based feature detection toward the image center.

### Lens Aberrations

Lens aberrations are optical imperfections that cause distortion or reduced image quality. The most common types affecting robotic systems are chromatic aberration (color fringing), spherical aberration (blurriness), and geometric distortion (barrel/pincushion effects).

**Characteristics:**
- **Chromatic aberration:** Color fringing at high-contrast edges, particularly toward image corners
- **Spherical aberration:** Loss of sharpness, especially at wider apertures
- **Barrel distortion:** Straight lines bow outward (common in wide-angle lenses)
- **Pincushion distortion:** Straight lines bow inward (common in telephoto lenses)
- Thermal lenses typically show less chromatic aberration but can exhibit significant geometric distortion

**Impact on applications:** Geometric distortion severely affects measurement accuracy and 3D reconstruction. Chromatic and spherical aberrations reduce feature detection reliability and degrade image-based localization.

### Blurriness

Blurriness results from several causes including motion blur (camera or subject movement during exposure), defocus blur (incorrect focus distance), and optical blur (inherent lens limitations).

**Characteristics:**
- **Motion blur:** Directional smearing in the direction of movement
- **Defocus blur:** Uniform softening with characteristic "bokeh" patterns at highlights
- **Thermal drift blur:** Unique to thermal cameras experiencing rapid temperature changes
- More problematic at slower shutter speeds or with long-range lenses

**Impact on applications:** Blurriness destroys fine details needed for feature matching, causes inaccurate optical flow estimation, and creates ghosting artifacts in multi-frame processing. For thermal cameras, it can blend temperature gradients and lose critical thermal boundaries.

---

## Different types of image processing to reduce noise

### Improving Image Contrast

Contrast enhancement stretches the dynamic range of pixel intensities to better utilize the available bit depth. This is particularly valuable for thermal cameras operating in environments with small temperature variations and for RGB cameras in hazy or low-contrast conditions. However, contrast enhancement can also amplify noise if not applied carefully.

**Common techniques:**
- **Linear contrast stretching:** Maps the minimum pixel value to 0 and maximum to 255 (for 8-bit images)
- **Percentile clipping:** Uses 1st and 99th percentiles as bounds to avoid outlier influence
- **Adaptive contrast:** Applies local contrast enhancement to different image regions

**Implementation considerations:** Simple contrast stretching can amplify noise, so it's often combined with denoising. For thermal images, ensure your dynamic range adjustment doesn't clip important temperature information at the extremes.

**Code example concept:** Map `[min_value, max_value]` → `[0, 255]` using `output = (input - min) * 255 / (max - min)`, but consider using percentile values instead of absolute min/max to handle outliers.

### Histogram Equalization

Histogram equalization redistributes pixel intensities to achieve a more uniform histogram, maximizing contrast globally. Adaptive variants (CLAHE - Contrast Limited Adaptive Histogram Equalization) divide the image into tiles and apply equalization locally, preventing over-amplification.

**When to use:**
- Images with poor contrast or narrow intensity distributions
- Thermal images where features are barely visible due to small temperature differences
- Preprocessing for feature detection algorithms that rely on intensity gradients

**Advantages and limitations:** Global histogram equalization is simple and fast but can over-enhance noise and create artificial contrast. CLAHE provides better results by limiting contrast amplification and adapting to local statistics, but requires careful parameter tuning (tile size, clip limit).

### Median Filtering and Other Blurring Methods

Spatial filtering methods reduce noise by replacing each pixel with a function of its neighbors. Different filters have different characteristics suited to specific noise types.

**Median filtering:**
- Replaces each pixel with the median value in its neighborhood (typically 3×3 or 5×5)
- Extremely effective against salt and pepper noise
- Preserves edges better than averaging filters
- Nonlinear operation, so more computationally expensive than simple convolution

**Gaussian blur:**
- Weighted averaging with weights following a Gaussian distribution
- Effective for reducing Gaussian noise while maintaining some edge information
- Kernel size (sigma) controls the amount of smoothing
- Can be applied separably for computational efficiency

**Bilateral filtering:**
- Edge-preserving smoothing that considers both spatial proximity and intensity similarity
- Excellent for reducing noise while maintaining important boundaries
- Particularly valuable for thermal imaging where temperature boundaries are critical
- More computationally expensive than simple Gaussian blur

**Box/Average filtering:**
- Simple uniform averaging of neighborhood pixels
- Fast but produces more blurring than Gaussian methods
- Can be useful as a quick preprocessing step

**Practical recommendations:** Use median filters (3×3 or 5×5 kernel) for salt and pepper noise since the assumption is that the high and low intensity noise will be on either side of the median. Bilateral filtering can be used for general noise reduction in thermal images where edge preservation is critical but can be slow. Gaussian blur is the standard blurring mechanism for general graininess reduction in images.

### Lens Calibration

Lens calibration characterizes the intrinsic parameters and distortion coefficients of your camera system, allowing you to correct geometric distortions and accurately map 2D image coordinates to 3D rays.

**Calibration parameters:**
- **Intrinsic matrix:** Focal length (fx, fy) and principal point (cx, cy)
- **Distortion coefficients:** Radial distortion (k1, k2, k3...) and tangential distortion (p1, p2)
- **Vignetting correction:** Optional polynomial or lookup table for intensity correction

**Calibration process:**
The standard approach uses a checkerboard or calibration target viewed from multiple angles. Computer vision libraries (OpenCV, MATLAB) provide automated calibration routines that solve for parameters. For thermal cameras, specialized high-emissivity calibration targets may be required. For estimating vignetting, capture images of a uniform surface (e.g., a white wall) under consistent lighting and use a LookUp Table (LUT) or polynomial fit to model intensity falloff.

**When to calibrate:**
- Before deployment for any application requiring accurate measurements
- After any mechanical shock or lens adjustment
- Periodically for thermal cameras (every few months) as their optics can drift with temperature cycling

**Applying corrections:**
Once calibrated, undistort images using the distortion coefficients before any geometric operations (stereo matching, structure from motion, object measurement). Most vision libraries provide efficient undistortion functions. For real-time applications, precompute undistortion maps.

**Practical tip:** Store calibration parameters with timestamp and sensor serial number. For thermal cameras, consider calibrating at your expected operating temperature range, as thermal expansion can affect lens geometry.

---

## Recommended Processing Pipeline

For a robust imaging pipeline applicable to both thermal and visual sensors:

1. **Capture and validate raw images** - Check for sensor communication errors and obvious artifacts
2. **Apply lens calibration corrections** - Undistort geometry and optionally correct vignetting
3. **Temporal filtering (if applicable)** - For video streams, consider frame averaging or temporal median filtering to reduce temporal noise
4. **Spatial noise reduction** - Apply appropriate filtering (median for salt/pepper, bilateral for general noise)
5. **Contrast enhancement** - Use CLAHE or adaptive methods if needed for your downstream algorithms
6. **Task-specific processing** - Feature detection, object recognition, etc.

The exact pipeline should be tuned based on your specific sensor characteristics, computational budget, and application requirements. For real-time systems, consider GPU acceleration for filtering operations and undistortion mapping.

## Additional Considerations

**Thermal-specific challenges:** Thermal cameras require periodic non-uniformity correction (NUC) cycles, typically performed by shuttering the sensor to recalibrate pixel responses. Plan for brief interruptions in your imaging pipeline. Additionally, thermal cameras are sensitive to their own temperature, so consider active cooling or temperature stabilization for critical applications.

**Multi-spectral fusion:** When using both RGB and thermal sensors, proper temporal and spatial alignment is critical. Calibrate the stereo baseline between sensors and account for different fields of view and resolution.

**Validation:** Always validate your noise reduction pipeline with ground truth data or controlled test scenes. Overly aggressive filtering can destroy genuine features that your application depends on.