---
date: 2025-04-25 # YYYY-MM-DD
title: Docker for Robotics
---

# Docker Overview

Docker is a tool that helps you create, share, and run applications in containers. Containers are small, lightweight packages that include everything your application needs, like the code, libraries, and settings. They are faster and use fewer resources compared to virtual machines (VMs).

**Difference Between Containers and Virtual Machines:**

- Virtual Machines (VMs): Full operating systems running on top of a hypervisor. They are resource-heavy and slow to start.
- Docker Containers: Share the host OS kernel, making them faster, more lightweight, and more resource-efficient.

**Key components:**

- Images: Think of an image like a recipe. It tells Docker what to include in the container (software, libraries, etc.).
- Containers: These are the actual "live" versions of the image. It’s like cooking from the recipe — your container is the meal ready to eat.
- Dockerfile: A script that defines how an image is built, including base images, commands, and configurations.
- Volumes: Persistent storage that containers can use to save data.
- Docker Hub: A cloud-based repository where pre-built Docker images are shared.

# Installation

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

[Install docker compose plugin](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually)

# Docker Commands

## Images

- Build an image from a Dockerfile:
    
    ```bash
    $ docker build -t <image_name>
    ```
    
- Pull an image from a Docker Hub:
    
    ```bash
    $ docker image pull <image_name>:<tag>
    ```
    
- Search Hub for an image:
    
    ```bash
    $ docker image search <image_name>
    ```
    
- List local images:
    
    ```bash
    $ docker image ls
    $ docker images
    ```
    
- Delete an image:
    
    ```bash
    $ docker image rm <image_name>
    $ docker rmi <image_name>
    ```
    
- Remove all unused images:
    
    ```bash
    $ docker image prune
    ```
    

## Containers

- Create and run a container from an image, with a custom name:
    
    ```bash
    $ docker run --name <container_name> <image_name>
    ```
    
- Run a container with terminal:
    
    ```bash
    $ docker run -it <image_name>
    ```
    
- Start or stop an existing container:
    
    ```bash
    $ docker start|stop <container_name> (or <container-id>)
    ```
    
- Start an existing container with terminal:
    
    ```bash
    $ docker start -i <container_name>
    ```
    
- List running containers:
    
    ```bash
    $ docker container ls
    $ docker ps
    ```
    
- List all containers (even the stopped ones):
    
    ```bash
    $ docker container ls -a
    $ docker ps -a
    ```
    
- Remove a stopped container:
    
    ```bash
    $ docker rm <container_name>
    ```
    
- Remove all available containers:
    
    ```bash
    $ docker container prune
    ```
    
- Open terminal inside a running container:
    
    ```bash
    $ docker container exec -it <container_name> /bin/bash
    ```
    
    For any commands within a running container:
    
    ```bash
    $ docker container exec -it <container_name> <command>
    ```
    

### Working with Volumes

- Mount Host Directory to Container: This is how we can make a directory on host available inside the container

```bash
$ docker run -it -v <absol_path_on_host>:<absol_path_in_container> <image_name>
```

```bash
$ docker run -it --network=host --ipc=host -v <absol_path_on_host>:<absol_path_in_container> <image_name>
```

Any files created in a container in a shared volume will be locked — can be accessed only by the root.

**Note:** `docker run` always creates a new container. We lose any changes we make to the environment every time we `run` the container.

## Setting up a Dockerfile

The `Dockerfile` contains the steps for creating an image. It typically starts with a base image and includes commands for installing software, setting environment variables, and defining the container’s entrypoint.

```docker
# FROM <base_image_name>
FROM osrf/ros:humble-desktop-full

# Commands to perform on base image
RUN apt-get -y update \
 && apt-get -y install some_package \
 && git clone https://github.com/some_user/some_repository some_repo \
 && cd some_repo \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j$(nproc) \
 && make install \
 && rm -rf /var/lib/apt/lists/*

# Install additional packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-turtlebot3-simulations

# Set up workspace
ENV WS_DIR="/root/ros2_ws"
WORKDIR ${WS_DIR}
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Default command
CMD ["/bin/bash"]

# COPY <configuration_file to copy> <direction in image to be copied into>
COPY config/ site_config/

# Define the script that should be launched upon start of the container
ENTRYPOINT ["/root/ros2_ws/src/my_script.sh"]
```

- All commands run in the docker container will run as `root`.
- The `COPY` command assumes paths are relative to the build context specified in the `docker-compose.yml` or the `docker build` command.
- To build and run the Docker image, go into the directory which will be the new image

```bash
$ docker image build -t <new_image_name> <directory>
$ docker run -it <new_image_name>
```

## Entrypoint Scripts

Entrypoint scripts automate container setup at runtime. It runs every time the container is brought up.

- Create a new file called `entrypoint.sh` inside the directory.

```bash
#!/bin/bash
source /opt/ros/humble/setup.bash
exec "$@"
```

- Add it to the `Dockerfile`

```docker
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

## GUI in Docker

```bash
$ docker run -it --network=host --ipc=host -v <absol_path_on_host>:<absol_path_in_container> -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY <image_name>
```

# Docker Compose

Docker Compose simplifies multi-container applications by allowing you to define and manage them through a YAML file (`docker-compose.yml`).

For newer versions (Docker v2.0 and later), use `docker compose` instead of `docker-compose`.

## Basic Commands

- Start and run all services defined in the `docker-compose.yml` file:

```bash
$ docker compose up
```

- Start services in detached mode (background):

```bash
$ docker compose up -d
```

- Stop all running services:

```bash
$ docker compose stop
```

- Stop and remove all services, networks, and volumes:

```bash
$ docker compose down
```

- Restart all services:

```bash
$ docker compose restart
```

## Configuration Management

- Validate the `docker-compose.yml` file:

```bash
$ docker compose config
```

- View the service logs (real-time streaming):

```bash
$ docker compose logs
```

- View logs of a specific service:

```bash
$ docker compose logs <service_name>
```

- Build or rebuild services

```bash
$ docker compose build
```

- Build a specific service:

```bash
$ docker compose build <service_name>
```

- Pull service images defined in the `docker-compose.yml` file:

```bash
$ docker compose pull
```

## Service Management

A service represents a single containerized application or component in a multi-container setup. Each service corresponds to a container, and the docker-compose.yml file is used to define the configuration for these services.

- Start a specific service

```bash
$ docker compose up <service_name>
```

- Stop a specific service:

```bash
$ docker compose stop <service_name>
```

- Remove stopped service containers:

```bash
$ docker compose rm
```

- Remove a specific service container:

```bash
$ docker compose rm <service_name>
```

## Network and Volume Management

- View networks created by Docker Compose:

```bash
$ docker network ls
```

- View volumes created by Docker Compose:

```bash
$ docker volume ls
```

- Remove unused networks:

```bash
$ docker network prune
```

- Remove unused volumes:

```bash
$ docker volume prune
```

## **Writing and launching a Docker-Compose file**

Example `docker-compose.yml` for a ROS 2 project:

```docker
version: '3.8'
services:
  ros-master:
    image: osrf/ros:humble-ros-core
    container_name: ros-master
    networks:
      - ros-network

  turtlebot-sim:
    image: osrf/ros:humble-desktop
    container_name: turtlebot-sim
    depends_on:
      - ros-master
    networks:
      - ros-network

networks:
  ros-network:
    driver: bridge
```

After having created both a `Dockerfile` as well as a `docker-compose.yml` you can launch them with:

```bash
$ docker compose -f docker-compose.yml build
$ docker compose -f docker-compose.yml up
```

where with the option `-f` a Docker-Compose file with a different filename can be provided. If not given it will default to `docker-compose.yml`.

More general `docker-compose.yml`:

```docker
version: "3.9"
services:
  some_service: # Name of the particular service (Equivalent to the Docker --name option)
    build: # Use Dockerfile to build image
      context: . # The folder that should be used as a reference for the Dockerfile and mounting volumes
      dockerfile: Dockerfile # The name of the Dockerfile
    container_name: some_container
    stdin_open: true # Equivalent to the Docker -i option
    tty: true # Equivalent to the Docker docker run -t option
    volumes:
      - /a_folder_on_the_host:/a_folder_inside_the_container # Source folder on host : Destination folder inside the container
  another_service:
    image: ubuntu/20.04 # Use a Docker image from Dockerhub
    container_name: another_container
    volumes:
      - /another_folder_on_the_host:/another_folder_inside_the_container
volumes:
  - ../yet_another_folder_on_host:/a_folder_inside_both_containers # Another folder to be accessed by both images
```

If instead you wanted only to run a particular service you could do so with:

```bash
$ docker compose -f docker-compose.yml run my_service
```

Then similar to the previous section, we can connect to the container from another console with

```bash
$ docker compose exec <docker_name> sh
```

where `<docker_name>` is given by the name specified in the `docker-compose.yml` file and `sh` stands for the type of comand to be execute, in this case we open a `shell`.

# Docker Registry

- Build image locally:

```bash
$ docker compose build <servie_name>
```

- Tag the resulting image for Docker Hub:

```bash
$ docker tag <service_name> <your_dockerhub_username>/<name>:<tag>
```

- Push the image to Docker Hub:

```bash
$ docker push <your_dockerhub_username>/<name>:<tag>
```

In case you're not logged in, use

```bash
$ docker login -u <username>
```

And then enter the password.

It is necessary to include your Docker Hub username in the tag.


# Building Docker Images for Multiple Architectures

- Ensure `qemu` emulation is enabled: You need to have `qemu-user-static` installed and properly configured for cross-platform builds.

```bash
$ sudo apt-get install -y qemu-user-static
$ docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

This ensures that the `qemu` emulator is registered for the required architectures.

When building or running Docker images for a different architecture:

1. Build Process: QEMU emulates the target architecture (e.g., `arm64`) on the host (e.g., `amd64`), enabling you to compile binaries and packages for the target system.
2. Run Process: QEMU interprets `arm64` instructions so that the container can run on an `amd64` host without errors.

- Setup `buildx`

```bash
$ docker buildx create --name multiarch --use
$ docker buildx inspect --bootstrap
```

- Tag the image that you want to push

```bash
$ docker tag <service_name> <your_dockerhub_username>/<name>:<tag>
```

- Build the multi-arch image

```bash
$ docker buildx build --platform linux/amd64,linux/arm64/v8 \
  -t your-dockerhub-username/your-image-name:tag \
  --push \
  -f /path/to/Dockerfile /path/to/context
```

- The image will be pushed to Docker Hub



# Additional Resources

- [Docker for Development](https://docs.nav2.org/tutorials/docs/docker_dev.html)
- [Docker for Robotics](https://github.com/2b-t/docker-for-robotics/tree/main)
- [YouTube](https://youtube.com/playlist?list=PLunhqkrRNRhaqt0UfFxxC_oj7jscss2qe&si=j5NCJxazjTFhSNZ3)