# Complete Guide to Installing ORB SLAM3

## 1. Introduction
ORB SLAM3 (Oriented FAST and Rotated BRIEF Simultaneous Localization and Mapping, Version 3) is a versatile SLAM system that performs real-time mapping using various camera setups. This guide explains each step of the installation process, ensuring you understand not just what commands to run, but why they're necessary.

## 2. System Preparation
First, we need to add required repositories and update the system. Ubuntu Xenial's security repository contains some legacy libraries that ORB SLAM3 depends on:

```bash
# Add the Xenial security repository for legacy dependencies
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"

# Update package lists to include the new repository
sudo apt update
```

## 3. Installing Core Dependencies

### Basic Development Tools
These tools provide the fundamental build environment:

```bash
# Install build-essential which provides gcc, g++, and make
sudo apt-get install build-essential

# Install cmake for building C++ projects and git for version control
# Install GTK for GUI applications and codec libraries for video processing
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

### Image Processing Libraries
These libraries handle various image formats and processing tasks:

```bash
# Install Python development files and numpy for numerical computations
# Install TBB for parallel programming support
# Install various image format support libraries
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libjasper-dev

# OpenGL and Boost libraries for visualization and advanced C++ features
sudo apt-get install libglew-dev libboost-all-dev libssl-dev

# Eigen library for linear algebra and matrix operations
sudo apt install libeigen3-dev
```

## 4. Installing OpenCV 3.2.0
ORB SLAM3 requires specifically OpenCV 3.2.0 for compatibility. Here's how to install it:

```bash
# Create development directory and navigate to it
cd ~
mkdir Dev && cd Dev

# Clone OpenCV repository and checkout version 3.2.0
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.2.0
```

We need to fix a compatibility issue with modern FFmpeg versions:

```bash
# Add necessary definitions for FFmpeg compatibility
echo '#define AV_CODEC_FLAG_GLOBAL_HEADER (1 << 22)
#define CODEC_FLAG_GLOBAL_HEADER AV_CODEC_FLAG_GLOBAL_HEADER
#define AVFMT_RAWPICTURE 0x0020' > ./modules/videoio/src/cap_ffmpeg_impl.hpp
```

Now build OpenCV:

```bash
# Create and enter build directory
mkdir build && cd build

# Configure build with CMake - we disable CUDA for better compatibility
cmake -D CMAKE_BUILD_TYPE=Release -D WITH_CUDA=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local ..

# Build using 3 CPU threads (adjust based on your CPU)
make -j3

# Install to system directories
sudo make install
```

## 5. Installing Pangolin
Pangolin provides visualization capabilities for ORB SLAM3. We use a specific commit known to work well:

```bash
# Move to development directory and clone Pangolin
cd ~/Dev
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin 

# Checkout specific working commit
git checkout 86eb4975fc4fc8b5d92148c2e370045ae9bf9f5d

# Create build directory and configure
mkdir build && cd build 
cmake .. -D CMAKE_BUILD_TYPE=Release 

# Build and install
make -j3 
sudo make install
```

## 6. Installing ORB SLAM3
Now we'll install ORB SLAM3 itself:

```bash
# Clone ORB SLAM3 repository
cd ~/Dev
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
```

Before building, we need to make several modifications to fix common issues:

### Issue 1: C++ Standard Compatibility
Open `CMakeLists.txt` and update the C++ standard settings to use C++14:

```cmake
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
```

This change is necessary because some features used in the code require C++14.

### Issue 2: Eigen Include Paths
Modern Eigen installations use a different include path structure. You'll need to update all Eigen includes in the codebase. For example:

```bash
// Find all files containing Eigen includes
find . -type f -exec grep -l "#include <Eigen/" {} \;

// In each file, change includes from:
#include <Eigen/Core>

// to:
#include <eigen3/Eigen/Core>
```

Issue 3: Loop Closing Fix  
In include/LoopClosing.h, modify line 51 to fix a type compatibility issue:

```cpp
// Change from:
Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

// To:
Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > > KeyFrameAndPose;
```

Finally, build ORB SLAM3:

```bash
# Make build script executable
chmod +x build.sh

# Run build script (may need multiple attempts)
./build.sh
```

## 7. Testing the Installation
Test the installation with one of the example datasets:

```bash
# Test with EuRoC dataset
./Examples/Stereo/stereo_euroc \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Stereo/EuRoC.yaml \
    ~/Datasets/EuRoc/MH01 \
    ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt
```

This command:

- Loads the ORB vocabulary file  
- Uses the EuRoC camera calibration settings  
- Processes the MH01 sequence  
- Uses timestamp information for synchronization  


This completes the installation process. Each step is crucial for the proper functioning of ORB SLAM3, building from basic system libraries through specialized components to the final system.