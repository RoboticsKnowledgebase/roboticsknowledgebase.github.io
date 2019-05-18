---
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

## Working With Docker:

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
