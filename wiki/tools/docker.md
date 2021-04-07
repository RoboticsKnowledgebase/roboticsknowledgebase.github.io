---
date: 2019-05-16
title: Docker 
---

# Docker Setup

Docker is a platform for developers and sysadmins to develop, deploy, and run applications with containers. The use of Linux containers to deploy applications is called containerization. Containers are not new, but their use for easily deploying applications is.

Containerization is increasingly popular because containers are:

  - Flexible: Even the most complex applications can be containerized.
  - Lightweight: Containers leverage and share the host kernel.
  - Interchangeable: You can deploy updates and upgrades on-the-fly.
  - Portable: You can build locally, deploy to the cloud, and run anywhere.
  - Scalable: You can increase and automatically distribute container replicas.
  - Stackable: You can stack services vertically and on-the-fly

## Install Docker on Ubuntu 16.04:

Now let us download Docker into a Ubuntu Xenial (16.04). Firstly, let's get started with updating previous repositories

```sh
$ sudo apt-get update
```

In order to ensure the downloads are valid, add the GPG key for the official Docker repository to your system:

```sh
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 
```
If the GPG key is added properly, the terminal should output 'ok' message.

Add the Docker repository to APT sources:

```sh
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

Next, update the package database with the Docker packages from the newly added repo:

```sh
$ sudo apt-get update
```

Make sure you are about to install from the Docker repo instead of the default Ubuntu 16.04 repo:

```sh
$ apt-cache policy docker-ce
```

Finally, install the Docker

```sh
$ sudo apt-get install -y docker-ce
```

Docker should now be installed, the daemon started, and the process enabled to start on boot. Check that it's running:

```sh
$ sudo systemctl status docker
```

If the Docker is properly installed, the above command will output something similar to the following:

```sh
● docker.service - Docker Application Container Engine
   Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
   Active: active (running) since Tue 2019-05-07 14:01:38 EDT; 25min ago
     Docs: https://docs.docker.com
 Main PID: 2112 (dockerd)
    Tasks: 42
   Memory: 107.3M
      CPU: 1.460s
   CGroup: /system.slice/docker.service
           └─2112 /usr/bin/dockerd -H fd://
```

## Setup Nvidia Docker:

For Projects having different versions of software packages like tensorflow, Docker helps to keep a uniform version across various machines so incompatibility issues wouldn't arise. This section will highlight how to use Nvidia docker for your project.

Ensure that your system is able to access the GPU using the following command:

```sh
$ nvidia-smi
```

The above command should display the system's GPU information. If the above doesn't display the system's GPU information, run the following command to detect the presence of GPU: 

```sh
$ lspci | grep -i nvidia
```

Failure of any of the above command indicates that the NVIDIA GPU is not installed into the system. You may want to follow this tutorial to install NVIDIA drivers [install_nvidia_driver](<https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup>).

Now, we need to install package repositories.

```
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \ sudo apt-key add -distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
```

Install NVIDIA docker-2 and reload daemon configuration

```
$ sudo apt-get install -y nvidia-docker2
$ sudo pkill -SIGHUP dockerd
```

Test Installation with CUDA Docker image:

```
$ docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

## Running Your First Docker Container:

Now, let's dive into using Docker for your project

Docker containers are run from Docker images. By default, it pulls these images from Docker Hub. Anybody can build and host their Docker images on Docker Hub, so most applications and Linux distributions you'll need to run Docker containers have images that are hosted on Docker Hub.

We will begin with a simple 'Hello Docker' program. Run the following command:

```sh
$ docker run hello-world
```

You should see the following output:

```sh
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.
```

## Basic Docker Usage:
Docker has three main types of objects that you need to be familiar with - Dockerfiles, Images, and Containers.
A Dockerfile is a file that describes how an image should be built. An image is a binary that can be used to create a container. A container is an isolated runtime environment with its own file system. Lets walk though a simple workflow.

### Make A Dockerfile
First, you need to make a Dockerfile. It describes a series of commands that should be executed in order to build docker image. Ussually a Dockerfile will install software dependencies, install development tools, setup environment variables, etc. Whatever software you need for your project should be installed in your Dockerfile.

### Build A Docker Image
After you have made a Dockerfile, it can be built into a docker image, which is simply a compiled version of the dockerfile. Execute the following command from the same folder that the Dockerfile is in.
```sh
sudo docker build -t {IMAGE_NAME}:{IMAGE_TAG} .
```
Here {IMAGE_NAME} is the name of your image, and {IMAGE_TAG} specifies a version. If you are not interested in keeping track of version you can simply set {IMAGE_TAG} to be "latest". It is important that you remember the {IMAGE_NAME} and {IMAGE_TAG} you use because you will need it to run a container.

### Run A Docker Container
To make a container from you image, run
```sh
sudo docker run -it --name {CONTAINER_NAME} {IMAGE_NAME}:{IMAGE_TAG}
```
This will create a docker container named {CONTAINER_NAME} using the image and tag that was just created. You should now be in your new docker and be able to execute shell commands in it.

### Exit A Docker Container
To leave the container, terminate the shell process.
```sh
exit
```

### Re-enter A Docker Container
After you have exited a docker container, you may want to launch it again. However, if you use ```docker run``` you will get an error saying "The container name "{CONTAINER_NAME} is already in use by container...". This is because docker stops a container when all of its processes have exited, but it does not remove container. Each container must have a unique name. To re-enter the container, you have two options.

Option 1: You can re-enter the container by starting it and the ```attach``` command. 
```sh
sudo docker start {CONTAINER_NAME}
sudo docker attach {CONTAINER_NAME}
```
Option 2: Remove the container and launch a new one.
```sh
sudo docker rm {CONTAINER_NAME}
sudo docker run -it --name {CONTAINER_NAME} {IMAGE_NAME}:{IMAGE_TAG}
```

## Other Useful Docker Features
### Running Multiple Processes
If you have a container that is running and you want to run another process in that container you can use ```docker exec```. Note that this must be done from the operating system and NOT from within the docker container. For example, to launch another shell you would use
```sh
sudo docker exec {CONTAINER_NAME} /bin/bash
```
### Persistent Storage Across Container Cycles
Chances are you want to have persistent access to data in a docker container. One the easiest ways to do this using a docker volume. This will add a folder to your docker container that will persist after the container is deleted. To do this, add the ```-v``` argument to ```docker run```
```sh
sudo docker run -it --name {CONTAINER_NAME} -v {LOCAL_DIR}:{CONTAINER_DIR} {IMAGE_NAME}:{IMAGE_TAG}
```
This will create a folder called {CONTAINER_DIR} inside the container that will also exist at {LOCAL_DIR} on your operating system. The data in this folder will persist after a container is deleted and can be used again when another container is started.
### See All Running Docker Containers
To see all running containers, use
```sh
sudo docker ps
```
### See All Images On Your Machine
```sh
sudo docker images
```
### Delete Unnecessary Containers and Images.
When you are first creating your docker file you may end up with many unused images. You can get rid of them using the following command
```sh
sudo docker prune
```

## Common Docker Issues On Ubuntu and Their Fixes
### Build fails because software cannot properly use debconf
Debconf is something that helps software configure itself while it is being installed. However, when a dockerfile is being built the software cannot interact with debconf. To fix this, add this line to your Dockerfile before you the line that causes the debconf error
```
ARG DEBIAN_FRONTEND=noninteractive
```
### QT does not work for applications in docker
Add this to your dockerfile
```
ARG QT_GRAPHICSSYSTEM="native"
```
Run command before you run the docker container to give UI permissions to docker
```sh
xhost + local:docker
```
Add the following arguments when you use docker run
```
-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged
```
### The terminal prompt is not properly highlighted
The terminal prompt, which is the PS1 environment variable, is set by the bashrc file. The default file may not properly enable or has logic built in which disables it. To get around it, add this to your dockerfile
```
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
```


To Create docker files for your project, you can follow the tutorial [here](<https://www.mirantis.com/blog/how-do-i-create-a-new-docker-image-for-my-application/>)

## Further Reading:
1. Create your very own Docker image: https://www.scalyr.com/blog/create-docker-image

2. Create Docker containers, services,  swarms, stacks for your application: https://docs.docker.com/get-started

3. Deploy Docker containers in AWS: https://aws.amazon.com/getting-started/tutorials/deploy-docker-containers

4. Docker Images inventory: https://hub.docker.com/search/?type=image

## References:
1. About Docker: https://docs.docker.com

2. NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker

3. Creating your own Dockerfile: https://www.youtube.com/watch?v=hnxI-K10auY

4. Docker file reference: https://docs.docker.com/engine/reference/builder

5. Docker CLI reference: https://docs.docker.com/engine/reference/commandline/cli/

6. Docker Volumes: https://docs.docker.com/storage/volumes/
