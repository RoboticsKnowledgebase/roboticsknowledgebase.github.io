---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-12-05 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Docker for Pytorch
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
In this article we will go over the following:
   1. Creating a custom docker image for deep learning workflows
   2. Docker run command for starting the container

## Docker Image for PyTorch based workflows
This article is going to be about how to set up a docker container for PyTorch training environment. Setting up docker for deep learning workflows is useful because configuring a GPU and installing all the necessary software packages is time consuming and has the risk of breaking the system. Also, configuring the GPU varies from system to system. Hence if we can create a docker image that has access to GPUs and all the software packages pre installed within it then that would save a lot of time. If in future you decide to switch to a different machine you don’t have to go through the hassle of configuring the new system again. You can just export the docker image to the new machine and continue your work. Having a docker image for complex workflows also allows us to experiment without worrying about breaking the system. If anything goes wrong, it is contained within the docker container and you can delete the container and create a new one. Hence having your workflow inside docker can be very useful.

## 1. Docker Image Setup
When a docker container is created it starts a new instance of an OS from scratch. The basic CPU drivers are pre-configured but the container does not have access to the GPUs of the system. Luckily for Nvidia GPUs we have a solution: NVIDIA Container Toolkit. In short, the NVIDIA Container Toolkit (formerly known as NVIDIA Docker) is a library and accompanying set of tools for exposing NVIDIA graphics devices to Docker containers. The NVIDIA Container Toolkit is a docker image that automatically recognizes GPU drivers on our base machine and passes those same drivers to our Docker container when it runs.
 
But before we go ahead, make sure that you have NVIDIA GPU drivers installed. You can check if the drivers are installed by running the following command. If this fails, it means that the drivers are not installed and you first have to do that before proceeding ahead.

```properties
nvidia-smi
```
 
For running the NVIDIA Container Toolkit, we can simply pull the NVIDIA Container Toolkit image at the top of our Dockerfile like so,
 
<em><strong>
FROM nvidia/cuda:10.2-base
</strong></em>
 
 
This is all the code we need to expose GPU drivers to Docker. In that Dockerfile we have imported the NVIDIA Container Toolkit image for 10.2 drivers. You should check your Nvidia driver versions and pull the appropriate image. The above command will only get us nvidia-smi. For deep learning tasks we also need cuDNN, for which we should pull the following image:
 
<em><strong>
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
</strong></em>
  
For a list of all the available NVIDIA Container Toolkit images check the following webpage. You can choose from various Ubuntu versions depending on your Nvidia CUDA driver version[1]. The selected version is [2].
 
Using this Nvidia base image now we can create our own docker image and configure it to have all the packages pre installed.
 
Create a new folder
```properties
mkdir nvidia_docker
cd nvidia_docker
```
Paste the following lines in a text file inside the folder:
 
```properties
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
 
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
 
RUN wget \
   https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh \
   && bash Anaconda3-2021.11-Linux-x86_64.sh -b \
   && rm -f Anaconda3-2021.11-Linux-x86_64.sh
 
ENV PATH=/root/anaconda3/bin:${PATH}
 
RUN conda install \
   pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch \
   && conda install -c conda-forge tensorboard \
   && conda install -c anaconda scikit-learn
```
 
This is the dockerfile for our custom Deep Learning Training image. Explanation of the dockerfile:
 
The first line pulls the Nvidia base image on top of which we build our image.
```properties
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
```
 
Following lines install wget package.
```properties
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
```
 
For DL tasks we should install Conda because it comes with all the necessary Python packages. Having a Conda environment lets us manage version conflicts for various DL workflows. For example, for some projects you need the latest Pytorch version but a different  project requires an old version. The Conda environment handles such instances smoothly. Hence we are going to pre-install the Conda environment in our image. Following lines download the Conda installer bash script from the website, run the bash script and delete it after completion. The URL in the command is for the latest version at the time of writing this article. Please change it to the current version when you install.
 
```properties
RUN wget \
   https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh \
   && bash Anaconda3-2021.11-Linux-x86_64.sh -b \
   && rm -f Anaconda3-2021.11-Linux-x86_64.sh
```
 
We want our docker container to activate the Conda environment at creation. The following line adds the path to Conda installation to the environment variable PATH.
```properties
ENV PATH=/root/anaconda3/bin:${PATH}
```
 
The following lines will install all the necessary packages for our DL workflow such as PyTorch, Torchvision, cudatoolkit, Tensorboard and scikit-learn. Change the version number for the cudatoolkit based on your specs. Feel free to add the lines to any other software that you like.
```properties
RUN conda install \
   pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch \
   && conda install -c conda-forge tensorboard \
   && conda install -c anaconda scikit-learn
```
 
Now we can build the image using the dockerfile we created above with the following command. This will create our custom DL docker image called nvidia-test.
```properties
sudo docker build . -t nvidia-test
```
 
## 2. Running the container
 
The command to create a container instance from the image is as follows:
 
```properties
sudo docker run -it --privileged --shm-size 8G --net=host --name test -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/user/docker_share/:/home/docker_share --gpus all nvidia-test bash
```
 
Breakdown of the command:
 
**-it**: This flag creates an interactive instance of the container
 
**--privileged:** This flag gives permission to the docker container access over all the I/O ports of the host computer.
 
**--net=host:** This flag sets the docker container to share the same IP address as the host computer.
 
**-shm-size 8G:** This flag sets the shared memory between the host computer and the docker container as 8GB. This flag is important because by default the shared memory between docker container and host computer is very less and during DL training this creates low memory issues. Hence set it to 8GB or more.
 
**--name test:** This flag gives a name to the container. Change according to your needs.
 
 
**-v /home/user/docker_share/:/home/docker_share:** This flag is used to map a directory in the host file system to a directory inside the docker container. This is useful because docker and host pc do not share the filesystem. So there is no way to transfer files between the docker container and the host PC. By mapping the volume we create a common folder between the docker container and the host PC which allows us to easily transfer files and save our work from inside the docker container. Change the paths according to your needs.
 
**--gpus all:** This is the most important flag of all. This gives the container access to the GPUs.
 
## Summary
With this we come to the end of our article. If you were able to successfully follow the above commands and were able to run the container, you now have a docker environment for training your PyTorch model. Hope this makes your life easier.

## See Also
- Docker https://roboticsknowledgebase.com/wiki/tools/docker/
- Setup GPU https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/
- Python construct https://roboticsknowledgebase.com/wiki/programming/python-construct/

## Further Reading
- PyTorch https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- Setting Up TensorFlow And PyTorch Using GPU On Docker https://wandb.ai/wandb_fc/tips/reports/Setting-Up-TensorFlow-And-PyTorch-Using-GPU-On-Docker--VmlldzoxNjU5Mzky
- Develop like a Pro with NVIDIA + Docker + VS Code + PyTorch https://blog.roboflow.com/nvidia-docker-vscode-pytorch/

## References
- CUDA Driver versions https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
- Selected version https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
