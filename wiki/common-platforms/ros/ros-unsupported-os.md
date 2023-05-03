---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-05-03 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: How to run a ROS distribution on an operating system that doesn't natively support it?
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
Each ROS distribution supports only a select number of platforms, and for all supported platforms, readily available Debian packages are available which can be easily installed using package managers such as apt. However, there often arise situations where you need to install and use a ROS distribution on a platform that is not supported for the distribution. This issue is a common occurrence when attempting to run specific ROS distributions on Single Board Computers (SBCs). The problem often arises due to the SBC's published images being either too outdated or too recent to support the required ROS distribution. This article will describe how to install and run ROS distributions in such cases. 

## Approaches
### 1. Running ROS2 Humble in a docker container on Ubuntu 20.04
- Pros:
  - Easy Setup
  - Isolated Environment
- Cons:
  - Limited control
  - Performance Overheads
### 2. Installing ROS2 Humble from source on Ubuntu 20.04
- Pros:
  - More Control
  - Better Performance
- Cons:
  - Time-consuming
  - Difficult to maintain


## Setups
### 1. Running ROS2 Humble via a docker container on Ubuntu 20.04

#### Installing Docker
Follow instructions given here to install docker - https://roboticsknowledgebase.com/wiki/tools/docker/
#### Running ROS2 Humble docker container
Running the command below will pull the latest ROS2 Humble Docker Image, start a container and attach a shell to it
```
docker run -it ros:humble bash
```

By running the following commands in two separate terminals, you should see the talker saying that it’s Publishing messages and the listener saying I heard those messages. This verifies both the C++ and Python APIs are working properly.
```
ros2 run demo_nodes_cpp talker
```
```
ros2 run demo_nodes_py listener
```

#### Creating a custom docker image using ROS2 Humble
We can also create a custom docker image from ROS2 Humble base image by adding the following snippet to the Dockerfile:
```
FROM ros:humble
# Add your code here
```
After modifying the Dockerfile, run the command below to build the image from the Dockerfile
```
docker build -t my/ros:app .
```
#### Useful docker flags for running ROS in docker
```
docker run -it --rm \
    --privileged \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/home/admin/.Xauthority:rw \ 
    -e DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \ 
    -v $ROS2_WS:/workspaces/ros_ws \
    -v /dev/:/dev/ \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --user="admin" \
    --workdir /workspaces/ros2_ws \
    ros:humble \
    /bin/bash
```
- `--privileged` flag gives the container full access to the host system's devices and capabilities including all devices on host, USB devices, cameras, etc.
- The following flags are related to running graphical applications inside a Docker container and they are used to allow the container to connect to the host system's X server, which is responsible for displaying graphical user interfaces.
```
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/home/admin/.Xauthority:rw \ 
-e DISPLAY \
```
- The following flags are required to enable the container to use the Nvidia GPUs for computation tasks.
```
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \ 
```
- `-v /dev/:/dev/` helps the container access hardware devices, such as USB devices, cameras, or serial ports.
- `-v $ROS2_WS:/workspaces/ros_ws` helps mount ROS2 workspace on to the container
- `--runtime nvidia` is used to enable Nvidia container runtime which gives the container access to Nvidia runtime components and libraries
### 2. Installing ROS2 Humble from source on Ubuntu 20.04
#### Set locale
```
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings
```
#### Add the ROS 2 apt repository
```
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
#### Install development tools and ROS tools
```
sudo apt update && sudo apt install -y \
  python3-flake8-docstrings \
  python3-pip \
  python3-pytest-cov \
  ros-dev-tools
```
```
python3 -m pip install -U \
   flake8-blind-except \
   flake8-builtins \
   flake8-class-newline \
   flake8-comprehensions \
   flake8-deprecated \
   flake8-import-order \
   flake8-quotes \
   "pytest>=5.3" \
   pytest-repeat \
   pytest-rerunfailures
```

#### Create a workspace and clone all repos
```
mkdir -p ~/ros2_humble/src
cd ~/ros2_humble
vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src
```

#### Install dependencies using rosdep
```
sudo apt upgrade
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers"
```

#### Build the code in the workspace
```
cd ~/ros2_humble/
colcon build --symlink-install
```

#### Set up your environment by sourcing the following file
```
. ~/ros2_humble/install/local_setup.bash
```

#### Try examples
By running the following commands in separate terminals, you should see the talker saying that it’s Publishing messages and the listener saying I heard those messages. This verifies both the C++ and Python APIs are working properly.
```
. ~/ros2_humble/install/local_setup.bash
ros2 run demo_nodes_cpp talker
```
```
. ~/ros2_humble/install/local_setup.bash
ros2 run demo_nodes_py listener
```
## Summary
In this article, we discussed how to run a ROS distribution on an unsupported OS. Specifically we looked at the process of running ROS2 Humble on Ubuntu 20.04. However, the methods presented in the article can be easily adapted to accommodate other scenarios.

## See Also:
- https://roboticsknowledgebase.com/wiki/tools/docker/

## Further Reading
- https://docs.docker.com/engine/reference/run/

## References
- https://docs.ros.org/en/humble/Installation/Alternatives.html
- https://github.com/osrf/docker_images
- https://hub.docker.com/_/ros/