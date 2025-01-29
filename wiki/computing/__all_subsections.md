/wiki/computing/arduino/
---
date: 2017-08-21
title: Arduino
---
This tutorial covers the basics of different Arduinos, and how to implement common functions with them.

## The Board
The main Arduino boards witnessed being used in these applications are the Arduino Uno and Arduino Mega.

## The Uno
![Arduino Uno R3 Front](assets/Arduino-d9b3f.png)

Good for smaller projects
- Has 14 digital input/output pins, 6 analog inputs (which can also be used as digital input/output pins), and a 5v as well as a 3.3v regulator
    - 6 of the digital input/output pins can be used for PWM
    - Pins 2 and 3 are usable for interrupts
    - Pins 0 and 1 cannot be used as normal digital inputs/output pins.
- If you are going to power it externally, you have to use between 7 and 12 volts on the Vin pin, and the ground of your power source has to go to a GND pin. A 9 volt battery works well for this. Make sure you connect the hot to Vin and the ground/negative terminal to ground of the power supply, or else you can fry the board.

## The Mega
![Arduino Mega R3](assets/Arduino-c30e6.png)

Good for bigger projects
- Has 54 digital input/output pins, 16 analog inputs (which can also be used as digital input/output pins), and a 5v as well as a 3.3v regulator
  - 15 of the digital input/output pins can be used for PWM
  - Pins 2, 3, 18, 19, 20, 21 are useable for interrupts.
  - Pins 0 and 1 cannot be used as normal digital inputs/output pins.
- If you are going to power it externally, you have to use between 7 and 12 volts on the Vin pin, and the ground of your power source has to go to a GND pin. A 9 volt battery works well for this. Make sure you connect the hot to Vin and the ground/negative terminal to ground of the power supply, or else you can fry the board.


## Wiring:
### Limit Switch:
#### Example:
![Limit Switch Wiring](assets/Arduino-2369d.png)

Whatever pin that is connected to the Normally Open pin of the limit switch, needs to be setup by using ``pinMode(pin#, INPUT_PULLUP);``

In this example, setup the pinMode as:

``pinMode(2, INPUT_PULLUP);``

In this setup, by using ``digitalRead(pin#)``, if the switch is open, it will read as ``HIGH (1)``, and when the switch is closed, digitalRead will return ``LOW (0)``. This happens because with the ``INPUT_PULLUP`` activated on the pin, it activates a pullup resistor for that pin, which, when the switch is open, the pin gets pulled HIGH by internal circuitry, but when closed, it gets pulled LOW since it is now directly connected to ground.

So to use this intuitively, use ``!digitalRead(pin#);`` this will return HIGH when pressed, and LOW when not pressed.


## Motor Driver:
Example with L298 Compact Motor Driver available in Mechatronics Lab:

![Motor Driver Wiring](assets/Arduino-de522.png)

With this example, the yellow lines connected to pins 10 and 11 (which are PWM) are the enables for the motors. When the enable is HIGH, the motor is turned on. For PWM lines, you use ``analogWrite(pin#, pwmValue);``, where ``pwmValue`` is an integer between 0-255, with 0 being off, and 255 being always HIGH, with inbetween values able to control speed if your motor is capable of that.

The blue and green lines can be connected to any digital pin, but in this example I kept them grouped. These are the direction pins for the motors. If both are LOW or both are HIGH, the motor will not move. But if direction pin 1 is HIGH and direction pin 2 is LOW, then the motor will move. When direction pin 1 is LOW and direction pin 2 is HIGH, the motor will move in the other direction.

> WARNING: This will send current through the positive node in one orientation, but when the direction is reversed the current will go through the negative node of the motor, make sure you check wiring accordingly

Another thing with this motor driver is that you can either turn off the motor, or you can stall it. Stalling the motor holds the motor taught and doesn't let it move while just turning it off will give play to the motor which is usually undesirable. To stall the motor, set the enable HIGH, and the two direction pins to either LOW or HIGH. To just turn it off, just set the enable to LOW.

Example code:
```
//declare pins
en1 = 11;
dir11 = 12;
dir21 = 13;
en2 = 10;
dir21 = 8
dir22 = 9;

//setup all pins as outputs
setup(){
  pinMode(en1, OUTPUT);
  pinMode(dir11, OUTPUT);
  pinMode(dir12, OUTPUT);
  pinMode(en2, OUTPUT);
  pinMode(dir21, OUTPUT);
  pinMode(dir22, OUTPUT);
}

loop(){
 //spin motors one way at half speed
  analogWrite(en1, 128);
  digitalWrite(dir11, HIGH);
  digitalWrite(dir12, LOW);
  analogWrite(en2, 128);
  digitalWrite(dir21, HIGH);
  digitalWrite(dir22, LOW);
  delay(1000);
 //stall motors
  digitalWrite(en1, HIGH);
  digitalWrite(dir11, HIGH);
  digitalWrite(dir12, HIGH);
  digitalWrite(en2, HIGH);
  digitalWrite(dir21, LOW);
  digitalWrite(dir22, LOW);
  delay(1000);
 //turn motors the other way at 3/4 speed
  analogWrite(en1, 192);
  digitalWrite(dir11, LOW);
  digitalWrite(dir12, HIGH);
  analogWrite(en2, 192);
  digitalWrite(dir21, LOW);
  digitalWrite(dir22, HIGH);
  delay(1000);
 //turn motors off
  digitalWrite(en1, LOW);
  digitalWrite(en2, LOW);
  delay(1000);
}
```


/wiki/computing/aws-quickstart/
---
date: 2020-05-11
title: Amazon Web Services Quickstart
---

This article will cover the basics of remote login on an Ubuntu machine. More specifically this will help you set up your AWS machine and serve as a tutorial to launch, access and manage your AWS Instance.
## Launching an EC2 Instance
First, we will have to sign up on AWS.
After logging into your account you will have to Choose a region.
The instances you make are linked to specific regions. After you have selected your region, click on Services in the top left. Then select EC2 under Compute.

1.  Launch an EC2 instance (Click on Launch Instance on the Dashboard)
2.  Select your required AMI. This will create a virtual machine for you with some pre-installed packages/applications. We will use the Deep Learning Base AMI for our tutorial.
    An Amazon Machine Image (AMI) provides the information required to launch an instance. You must specify an AMI when you launch an instance. You can launch multiple instances from a single AMI when you need multiple instances with the same configuration. You can use different AMIs to launch instances when you need instances with different configurations. -source AWS Website
3.  Select the version that matches your operating system (Ubuntu 16.04 / Ubuntu 18.04)
4.  Select the instance type. If you created a new account on AWS, you will be eligible for free usage of machines in the free tier range. T2.micro which falls under this category can be used to get familiar with AWS. To know more about the free-tier visit <https://aws.amazon.com/free/free-tier-faqs/>
5.  After selecting your instance, click configure Instance and dd torage.
6.  Depending on your requirement, select the amount of storage required. The first 30GB of storage is free under the one year eligibility.
7.  Continue pressing next until you see ‘Security Groups’.
    Here you will define the open ports for your machine. By default, port 22 will be open for you to ssh into your machine. However, you will have to define certain rules if you want to host different applications.
8.  Continue pressing ext until you see the review page.
9.  Launch Instance and select keypair. If you have previously generated a keypair, you can use the same file to access different machines.
10. To check your instances, click on Instances in the left sidebar.


**Now, to login to your instance -**
1.  Go to your terminal and login to your instance with the following command:\
    `ssh -i keyPair.pem -L 8000:localhost:8888 ubuntu@instance`
2.  You may need to set permissions for your key file. This can be done using chmod 400 keyPair.pem

Instead of having to write this huge command in your terminal, you can edit your ssh config file and use an alias to log in to your remote machine. For mac users the config file can be found in **/Users/user_name/.ssh/config**

The file should include\
*Host alias*\
*HostName remote_machine_ip*\
*IdentityFile path_to_keyfile/keyPair.pem*\
*User ubuntu*

## Using TMUX / SCREEN
While using ssh for remote access, your connections may be terminated if there is no activity for a long period of time. While training large models, this may be a problem. As your connection is piped through ssh, once the machine is left idle, the connection breaks causing all running applications / processes to terminate. To avoid this we can use Tmux or Screen.

TMUX​ is a ​terminal multiplexer​ for ​Unix-like​ ​operating systems​. It allows multiple ​terminal sessions to be accessed simultaneously in a single window. It is useful for running more than one ​command-line​ program at the same time. It can also be used to detach ​processes from their controlling terminals, allowing ​SSH​ sessions to remain active without being visible.
*- Wikipedia*

Here are a few important references in this regard
1.  [TMUX Quickstart](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
2.  [TMUX-Cheatsheet](https://tmuxcheatsheet.com/)

## Stopping Instances
To stop working with your instance for the night (or an extended period of time),
1. Go to your running instances
2. Select your active instance
3. Select Actions --> Instance State --> Stop

When your instance is not active, you will only be charged for storage (which is fairly cheap, but could add up.) To start the instance back up, follow the same steps but select start.
I you are done with an instance, follow the same steps, but terminate instead of stopping the instance.


## Spot Instances

A Spot Instance is an unused EC2 instance that is available for less than the On-Demand price. Because Spot Instances enable you to request unused EC2 instances at steep discounts, you can lower your Amazon EC2 costs significantly. The hourly price for a Spot Instance is called a Spot price. The Spot price of each instance type in each Availability Zone is set by Amazon EC2, and adjusted gradually based on the long-term supply of and demand for Spot Instances. Your Spot Instance runs whenever capacity is available and the maximum price per hour for your request exceeds the Spot price.

Spot Instances are a cost-effective choice if you can be flexible about when your applications run and if your applications can be interrupted. For example, Spot Instances are well-suited for data analysis, batch jobs, background processing, and optional tasks. For more information, see Amazon EC2 Spot Instances - AWS Official

Check how to use spot instances over here:
[Using Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html#spot-get-started)

## A word on Visual Studio Code:
VsCode comes with a plugin for remotely logging into your machine. This way you can develop and edit your code using VS Code-
The Visual Studio Code Remote - SSH extension allows you to open a remote folder on any remote machine, virtual machine, or container with a running SSH server and take full advantage of VS Code's feature set. Once connected to a server, you can interact with files and folders anywhere on the remote filesystem.
No source code needs to be on your local machine to gain these benefits since the extension runs commands and other extensions directly on the remote machine.

The following link provides details on the same.\
[Visual Studio - Remote SSH](https://code.visualstudio.com/blogs/2019/07/25/remote-ssh)


/wiki/computing/setup-gpus-for-computer-vision/
---
date: 2020-02-03
title: Setup your GPU Enabled System for Computer Vision and Deep Learning
---

This tutorial will help you setup your Ubuntu (16/17/18) system with a NVIDIA GPU including installing the Drivers, CUDA, cuDNN, and TensorRT libraries. Tutorial also covers on how to build OpenCV from source and installing Deep Learning Frameworks such as TensorFlow (Source Build), PyTorch, Darknet for YOLO, Theano, and Keras. The setup has been tested on Ubuntu x86 platform and should also hold good for other Debian based (x86/ARM64) platforms.

## Contents
1. [Install Prerequisites](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#1-install-prerequisites)
2. [Setup NVIDIA Driver for your GPU](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#2-install-nvidia-driver-for-your-gpu)
3. [Install CUDA](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#3-install-cuda)
4. [Install cuDNN](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#4-install-cudnn)
5. [Install TensorRT](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#5-install-tensorrt)
6. [Python and Other Dependencies](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#6-python-and-other-dependencies)
7. [OpenCV and Contrib Modules](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#7-install-opencv-and-contrib-modules)
8. [Deep Learning Frameworks](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#8-install-deep-learning-frameworks)
    - [PyTorch](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#pytorch)
    - [TensorFlow](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#tensorflow)
    - [Keras](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#keras)
    - [Theano](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#theano)
    - [Darknet for YOLO](https://roboticsknowledgebase.com/wiki/computing/setup-gpus-for-computer-vision/#darknet-for-yolo)

## 1. Install Prerequisites
Before installing anything, let us first update the information about the packages stored on the computer and upgrade the already installed packages to their latest versions.

    sudo apt-get update
    sudo apt-get upgrade

Next, we will install some basic packages which we might need during the installation process as well in future.

    sudo apt-get install -y build-essential cmake gfortran git pkg-config 

**NOTE: The following instructions are only for Ubuntu 17 and 18. Skip to the next section if you have Ubuntu 16.04**
    
The defualt *gcc* vesrion on Ubuntu 17 and 18.04 is *gcc-7*. However, when we build OpenCV from source with CUDA support, it requires *gcc-5*. 

    sudo apt-get install gcc-5 g++-5
    
Verify the *gcc* version:

    gcc --version
    
You may stil see version 7 detected. We have to set higher priority for *gcc-5* as follows (assuming your *gcc* installation is located at */usr/bin/gcc-5*, and *gcc-7*'s priority is less than 60.

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 60
    
Now, to fix the CXX and CC environment variable system-wide, you need to put the lines in your .bashrc file:

    echo 'export CXX=/usr/bin/g++-5.4' >> ~/.bashrc
    echo 'export CC=/usr/bin/gcc-5.4' >> ~/.bashrc
    source ~/.bashrc


## 2. Install NVIDIA Driver for your GPU
Before installing the NVIDIA driver, make sure **Secure Boot** is **Disabled** in BIOS and **Legacy Boot** is selected and **UEFI Boot** is disabled. 

**NOTE (FOR UEFI BOOT ONLY)**: If you still intend to install the driver along with UEFI boot enabled, follow the steps below to enroll the MOK keys and only then proceed with the next driver installation section. If these steps are not followed, it is likely that you might face the login loop issue.

```
sudo openssl req -new -x509 -newkey rsa:2048 -keyout UEFI.key -outform DER -out UEFI.der -nodes -days 36500 -subj "/CN=rambou_nvidia/"
sudo mokutil --import UEFI.der
```

At this step after reboot you will be prompted to select your certificate to import in in key database. If you have inserted a password at certificate creation you'll be prompted to insert it. If you are not prompted, you may have to enter the BIOS by using function keys at boot time.

### Driver Installation
The NVIDIA drivers will be automatically detected by Ubuntu in *Software and Updates* under *Additional drivers*. Select the driver for your GPU and click apply changes and reboot your system. *You may also select and apply Intel Microcode drivers in this window.* If they are not displayed, run the following commands from your terminal and refresh the window.

```
sudo add-apt-repository -y ppa:graphics-drivers
sudo apt-get update
```

*At the time of writing this document, the latest stable driver version is 418*.

Run the following command to check whether the driver has installed successfully by running NVIDIA’s System Management Interface (*nvidia-smi*). It is a tool used for monitoring the state of the GPU.

    nvidia-smi
    
In case the above mentioned steps fail or you run into any other issues and have access only to a shell, run the following set of commands to reinstall the driver. 

```
sudo apt-get purge -y nvidia*
sudo add-apt-repository -y ppa:graphics-drivers
sudo apt-get update
sudo apt-get install -y nvidia-418
```

## 3. Install CUDA
CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by NVIDIA which utilizes the parallel computing capabilities of the GPUs. In order to use the graphics card, we need to have CUDA libraries installed on our system.

Download the CUDA driver from the [official nvidia website here](https://developer.nvidia.com/cuda-downloads?target_os=Linux). We recommend you download the *deb (local)* version from installer type as shown in the screen-shot below.

*At the time of writing this document, the latest stable version is CUDA 10.0*.

![](https://roboticsknowledgebase.com/wiki/computing/assets/nvidia-cuda.png)

After downloading the file, go to the folder where you have downloaded the file and run the following commands from the terminal to install the CUDA drivers. Please make sure that the filename used in the command below is the same as the downloaded file and replace the `<version>` number.

    sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda

Next, update the paths for CUDA library and executables.

    echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"' >> ~/.bashrc
    echo 'export CUDA_HOME=/usr/local/cuda-10.0' >> ~/.bashrc
    echo 'export PATH="/usr/local/cuda-10.0/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
You can verify the installation of CUDA version by running:

    nvcc -V
        
## 4. Install cuDNN
CUDA Deep Neural Network (cuDNN) is a library used for further optimizing neural network computations. It is written using the CUDA API.

Go to official cuDNN website [official cuDNN website](https://developer.nvidia.com/cudnn) and fill out the form for downloading the cuDNN library. 

*At the time of writing this document, the latest stable version is cuDNN 7.4*.

**Make sure you download the correct cuDNN version which matches with you CUDA version.**

![](https://roboticsknowledgebase.com/wiki/computing/assets/nvidia-cudnn.png)

### Installing from TAR file (Recommended Method)
For cuDNN downloaded using _cuDNN Library for Linux_ method, go to the folder where you have downloaded the “.tgz” file and from the command line execute the following (update the filename).

    tar -xzvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

### Installing from Debian Package
Install the downloaded packages (Runtime Library, Developer Library and Code Samples) as follows.

    sudo dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb
    sudo dpkg -i libcudnn7-doc_7.4.2.24-1+cuda10.0_amd64.deb

To check installation of cuDNN, run this in your terminal:
    
    dpkg -l | grep cudnn

### Fixing Broken Symbolic Links
If you have issues with broken symbolic links when you run `sudo ldconfig`, follow the steps below to fix them. **Note the minor version number, which may differ on your system (shown for 7.4.2 here)**

    cd /usr/local/cuda/lib64
    sudo rm libcudnn.so
    sudo rm libcudnn.so.7
    sudo ln libcudnn.so.7.4.2 libcudnn.so.7
    sudo ln libcudnn.so.7 libcudnn.so
    sudo ldconfig

## 5. Install TensorRT
The core of NVIDIA TensorRT facilitates high performance inference on NVIDIA graphics processing units (GPUs). TensorRT takes a trained network, which consists of a network definition and a set of trained parameters, and produces a highly optimized runtime engine which performs inference for that network.

*At the time of writing this document, the latest stable version is TensorRT 5.0.4*.

Download the TensorRT local repo file [from here](https://developer.nvidia.com/tensorrt) and run the following commands. You'll need to replace ubuntu1x04, cudax.x, trt4.x.x.x and yyyymmdd with your specific OS version, CUDA version, TensorRT version and package date (refer the downloaded filename).

    sudo dpkg -i nv-tensorrt-repo-ubuntu1x04-cudax.x-trt5.x.x.x-ga-yyyymmdd_1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-cudax.x-trt5.x.x.x-ga-yyyymmdd/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install tensorrt
    
For Python and TensorFlow support, run the following commands.

    sudo apt-get install libnvinfer5 python-libnvinfer-dev python3-libnvinfer-dev
    sudo apt-get install uff-converter-tf
    
To check installation of TensorRT, run this in your terminal:
    
    dpkg -l | grep TensorRT
    
## 6. Python and Other Dependencies

Install dependencies of deep learning frameworks:

    sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev

Next, we install Python 2 and 3 along with other important packages like boost, lmdb, glog, blas etc.

    sudo apt-get install -y --no-install-recommends libboost-all-dev doxygen
    sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev libblas-dev 
    sudo apt-get install -y libatlas-base-dev libopenblas-dev libgphoto2-dev libeigen3-dev libhdf5-dev 
     
    sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy python-wheel python-six
    sudo apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy python3-wheel python3-six
    
**NOTE: If you want to use Python2, replace the following pip commands with pip2.**

Before we use pip, make sure you have the latest version of pip.

    sudo pip3 install --upgrade pip

Now, we can install all the required python packages for deep learning frameworks:

    sudo pip3 install numpy matplotlib ipython protobuf jupyter mock
    sudo pip3 install scipy scikit-image scikit-learn
    sudo pip3 install keras_applications==1.0.6 --no-deps
    sudo pip3 install keras_preprocessing==1.0.5 --no-deps
    
Upgrade numpy to the latest version:

    sudo pip3 install --upgrade numpy
     
## 7. Install OpenCV and Contrib Modules
First we will install the dependencies:

    sudo apt-get remove -y x264 libx264-dev
    sudo apt-get install -y checkinstall yasm
    sudo apt-get install -y libjpeg8-dev libjasper-dev libpng12-dev
    
    sudo apt-get install -y libtiff5-dev
    sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
     
    sudo apt-get install -y libxine2-dev libv4l-dev
    sudo apt-get install -y libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
    sudo apt-get install -y libqt4-dev libgtk2.0-dev libtbb-dev
    sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
    sudo apt-get install -y libvorbis-dev libxvidcore-dev
    sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
    sudo apt-get install -y x264 v4l-utils

#### NOTE: Checkout to the latest version of OpenCV. 3.4.5 is used here

Download OpenCV 3.4.5:

    git clone https://github.com/opencv/opencv.git
    cd opencv
    git checkout 3.4.5
    cd ..

Download OpenCV-contrib 3.4.5:

    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv_contrib
    git checkout 3.4.5
    cd ..
    
#### NOTE: Keep the build folder in the same location as it may be required in future to upgrade or uninstall OpenCV

Configure and generate the MakeFile in */opencv/build* folder (make sure to specify paths to downloaded OpenCV-contrib modules correctly):

    cd opencv
    mkdir build
    cd build
    
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=ON \
          -D WITH_OPENGL=ON \
          -D WITH_CUDA=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=ON ..

#### NOTE: If you are using Python3, you must add the following flag as well

          -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
          
#### NOTE: If you are using Ubuntu 17 or 18, you must add the following flags as well

          -D CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-5 \
          -D CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-5 \
    
Compile and install:

    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
Retain the build folder in the same location. This will be required if you want to uninstall OpenCV or upgrade in the future or else the uninstall process might become very tedious.   
    
Check installation of OpenCV:

    python
    >>> import cv2
    >>> cv2.__version__
    
#### NOTE: If you get any errors with `import cv2`, make sure the `PYTHONPATH` points to the location of `cv2.so` file correctly in your `~/.bashrc` file as follows.

    export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
    export PYTHONPATH=/usr/local/lib/python3.5/site-packages:$PYTHONPATH

To uninstall OpenCV:

    cd /opencv/build
    sudo make uninstall

## 8. Install Deep Learning Frameworks

### PyTorch  

You can run the commands for installing pip packages `torch` and `torchvision` from [the Quick Start section here](https://pytorch.org/).

### TensorFlow

#### Quick Install (Not Recommended)

A quick way to install TensorFlow using pip without building is as follows. However this is not recommended as we have several specific versions of GPU libraries to improve performance, which may not be available with the pip builds.

    sudo pip3 install tensorflow-gpu
     
#### Building TensorFlow from Source

Now we will download the TensorFlow repository from GitHub in the */home* folder. Checkout to the latest version of TensorFlow (`r1.13` is used here).

    cd ~
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git checkout r1.13
    
Next we need to install Bazel along with its dependencies
    
    sudo apt-get install pkg-config zip zlib1g-dev unzip 
    wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
    chmod +x bazel-0.21.0-installer-linux-x86_64.sh
    ./bazel-0.21.0-installer-linux-x86_64.sh --user
    
    export PATH="$PATH:$HOME/bin"
    source ~/.bashrc

To verify installation of Bazel run:
    
    bazel version

Now install brew on your system:

    sudo apt-get install linuxbrew-wrapper
    brew doctor
    brew install coreutils
    
The root of the *tensorflow* folder contains a bash script named configure. This script asks you to identify the pathname of all relevant TensorFlow dependencies and specify other build configuration options such as compiler flags. You must run this script prior to creating the pip package and installing TensorFlow.

    cd ~/tensorflow
    ./configure

**NOTE: Enter the your correct CUDA and CuDNN version below. CUDA 10.0 and CuDNN 7.4 is used here**

>Select Python 3, no to all additional packages, gcc as compiler (GCC 5.4).
>
>For CUDA, enter 10.0
>
>For cuDNN, enter 7.4
>
> Select yes for TensorRT support
>
>Enter your GPU Compute Capability (Eg: 3.0 or 6.1). Find yout GPU Compute Capability from [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
>
>Use nvcc as the CUDA compiler.

Finally, build the pip package:
    
    bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package 

The build might take upto an hour. If it fails to build, you must clean your build using the following command and configure the build once again.

    bazel clean --expunge
    ./configure

The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the /tmp/tensorflow_pkg directory:

    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

Once the build is complete, invoke pip install to install that pip package. The filename of the .whl file depends on your platform. Use tab completion to find your package. If you get an error saying package is not supported for the current platform, run pip explicity (as pip2 for Python 2.7).

    sudo pip3 install /tmp/tensorflow_pkg/tensorflow <TAB> (*.whl)
    
You can make a backup of this built .whl file.

    cp /tmp/tensorflow_pkg/tensorflow <TAB> (*.whl) <BACKUP_LOCATION>
    
Verify that TensorFlow is using the GPU for computation by running the following python script.

**NOTE: Running a script from the */tensorflow* root directory might show some errors. Change to any other directory and run the following python script.**

    import tensorflow as tf
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    
    with tf.Session() as sess:
        print (sess.run(c))

Here,

- "*/cpu:0*": The CPU of your machine.
- "*/gpu:0*": The GPU of your machine, if you have one.

If you have a gpu and can use it, you will see the result. Otherwise you will see an error with a long stacktrace. 

### Keras

    sudo pip3 install keras
    
### Theano

    sudo pip3 install Theano
    
### Darknet for YOLO

First clone the Darknet git repository.

    git clone https://github.com/pjreddie/darknet.git

Now, to compile Darknet with CUDA, CuDNN and OpenCV support, open the `Makefile` from the `darknet` folder and make the changes as following in the beginning of this file. Also make sure to select the right architecture based on your GPU's compute capibility. For Pascal architecture you may want to use [this version of Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) and compile with the `CUDNN_HALF=1` flag for 3x speed improvement.

    GPU=1
    CUDNN=1
    OPENCV=1
    
Once done, just run make from the darknet folder.

    cd darknet
    make

Refer [here](https://pjreddie.com/darknet/yolo/) for more details on running YOLO and training the network.

## References
The tutorial is actively maintained at [https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/](https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/).

[Issues Page](https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/issues) | [Contributions Page (Pull Requests)](https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/pulls)


/wiki/computing/single-board-computers/
---
date: 2017-09-13
title: Single-Board Computers
---

This article is out of date and should not be used in selecting components for your system. You can help us improve it by [editing it](https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io).
{: .notice--warning}

To eliminate the use of bulky laptops on mobile robots and prevent tethering otherwise, single board computers may seem like a wise option. There are various affordable single board computers available in the market such as:
1. Hardkernel ODroid X2 – ARM 1.7Ghz quad core, 2gb RAM
2. Hardkernel ODroid XU – ARM 1.7Ghz octa core, 4gb RAM
3. FitPC 2 – Intel Atom 1.6Ghz, 2gb RAM

They are all available under $300 and their specifications and support seem
extremely good for use with Ubuntu and ROS. But, unfortunately, they all fail to
perform as desired. Following are certain problems and challenges that we faced
while trying to get single board computers to run for our project:

The ODroids have an ARM processor and hence, they do not run Ubuntu 12.04 (A
stable version supporting most packages). There is an ARM version of Ubuntu called
Ubuntu Linaro that needs to be installed on the ODroids. This version of Ubuntu has compatibility issues with ROS. The packages required in ROS had to be built from
source and the binary files for the same had to be generated manually. This was a
very troublesome process, as the installation of every package took about 10 times
longer than what it took on a laptop. Further, ODroid did not support certain
hardware such as Kinect and Hokuyo Laser scanner due to the lack of available
drivers.

There were also certain network issues due to which the ODroid did not get
connected to the Secured network of CMU though it was possible to get it connected
to the open CMU network as well as any local wireless networks. If you are ever
ordering the ODroid, make sure that you have the time and patience to set it up for
use like your laptop and that you do not require heavy computation or dependence
on any external hardware. Also, ODroid does not ship with a power adapter, WiFi
dongle, HDMI cable or memory card. Hence, these components need to be ordered
separately.

The FitPC2 runs on an Intel processor and hence, eliminates most of the troubles
that ODroid faces due to the ARM processor. Installation of Ubuntu and ROS is
exactly the same as it is on any other laptop. The installation may take some more
time as the single core processor of FitPC2 is not as good as the quad core
processors provided by Intel in the laptops.
FitPC2 is a great choice if the computation that is required on the robot is not very heavy. For example, we tried
to run Hector SLAM, AMCL, SBPL Lattice planner, Local base planner, Rosserial and
a laser scanner for a task in our project and the latency started getting higher and
the FitPC2 eventually crashed within a couple of minutes of the start of the process.
Another team doing computer vision processing in their project also eliminated
FitPC2 from their project due to its limited computation capabilities. Hence, the
FitPC2 is an option only if the computation required by your robot is not very high.

Conclusion:
The above single board computers did not perform very well and it may not be a
great option to spend much time on either if your project requirements are similar
to ours. There are other options such as the UDoo that comes with an Arduino board
integrated with the single board computer or the Raspberry Pi or the Beaglebone
and they may be great choices for projects that require light computation.


/wiki/computing/troubleshooting-ubuntu-dual-boot/
---
date: 2023-05-12
title: Ubuntu Dual Boot and Troubleshooting Guide
---
This page serves as a tutorial and troubleshooting guide for dual booting Ubuntu alongside Windows for the uninitiated. This page assumes you have a PC or laptop with Windows 10 or 11 as the sole OS. Several difficulties can be encountered during setup if not aware of the process in advance. Read the following sections as needed and be aware of the potential issues brought up.

> It is recommended to begin the dual boot process as soon as possible in case things go wrong, or so that difficulties particular to the user’s hardware or desired setup are discovered as soon as possible.

## Create a bootable USB drive for Ubuntu
First, acquire an empty USB pen drive that is 8 GB or larger and insert it into the computer of choice. Have at least 64 GB of unused hard disk space on the computer (only 25-30 GB is needed for the Ubuntu installation but it is recommended to reserve at least another 35 GB for packages and project files). 

Go to the [Ubuntu Releases page](https://releases.ubuntu.com/) and select an LTS Release. Check with someone currently working with the software tools needed for the current project to know which version to install, as the latest release may not work with all software needed. Download the .iso file for that release. Download and use balenaEtcher from the [Balena webpage](https://www.balena.io/etcher) with that .iso and the inserted USB drive to create a bootable Ubuntu drive.

## Create partitions safely
Creating the Ubuntu partition on the hard drive for the dual boot can be done while installing Ubuntu from the USB drive but it is better to do it while in Windows Go to the Disk Management page in Windows by right clicking the Start menu. From there, right click on a partition in the middle labeled “NTFS” and click “Shrink Volume”. Do not shrink the EFI System Partition or the Recovery Partition on either end of the large partition. Type in the amount of MB to free up - this should be at least 65536 for anticipated future work. This amount in GB should appear to the right of the Windows partition shrank with the label “Unallocated”.

### Troubleshooting: Wrong partition shrunk / Wrong amount shrunk
If the wrong volume was shrinked or more space needs to be shrinked, right click on the partition that was just reduced and click Extend Volume. Extend it by the amount reduced to recover the unallocated space.

### Troubleshooting: Not allowed to shrink partition
If the Disk Management page is saying there are 0 MB available to shrink the volume, then this likely is because there are permanent files at the end of the partition, like the hibernation file and the system volume information folder. Disable Hibernate mode by opening the command prompt by typing “cmd” into the Start search bar, right-clicking to run as administrator, then running the command “powercfg /hibernate off”. Disable the System Restore feature by opening the System Properties window and under the System Protection header find Restore Settings and click the “Disable system protection” button. Finally, click the Advanced tab of System Properties, then click Settings under “Performance”, then click the Advanced tab of the Performance Options window, then under “Virtual memory” click “Change”, then select “No paging file” and “Set”. After these steps, restart the computer and the partition should be able to get shrinked now.

## Enter BIOS Setup to launch Ubuntu from the inserted USB drive
Restart the computer. While it is booting up, press the button for the computer that opens BIOS Setup. This is either F2, F8, F10, or F12, but check the computer’s manual pages. When the partition options show up, move down to the name of the USB drive inserted and select it. When booting the USB, pick the option that says “Ubuntu (safe graphics)” to prevent display issues caused by the graphics card.

## Move/combine partitions to make use of inaccessible partitions
Partitions of unallocated data can only be incorporated into another partition using Extend Volume if the unallocated partition is to the right of an existing partition with an OS. If there are multiple partitions of unallocated data or it is in a place where it is not able to get extended, use gparted in Ubuntu. (If it is not installed by default on Ubuntu, then install gparted following [this guide](https://linuxways.net/centos/how-to-install-gparted-on-ubuntu-20-04/)). This can be done before installing Ubuntu by selecting “Try Ubuntu” when loading Ubuntu from the bootable USB drive. Open a terminal by pressing at the same time Ctrl+Alt+T and run the command “gparted”. Once the window opens, select the partition of unallocated space and click the “Resize/Move” option to move the partition to where it can be used by Extend Volume on the partition, or click a partition used for an OS and move the sides of the partition to occupy the desired amount of unallocated memory. After each desired operation, click the Resize/Move button to queue the operation.

### Troubleshooting: gparted does not allow actions
If gparted is not allowing a certain action or is preventing it from ocurring, undo all previous steps and make sure each step is done individually by clicking Resive/Move after the step to prevent operations from conflicting.

## Ubuntu Installation
When installing Ubuntu, follow the prompts after clicking “Install Ubuntu”. Pay closer attention to the following steps:

1. Updates and other software
	- Choose “Normal Installation” and check the boxes that say “Download updates” and “Install third-party software”.
2. Installation Type
	- Choose “Something Else” and select the newly created partition with the intended space for Ubuntu for installation.

## Safely add additional memory 
If additional RAM memory sticks or an SSD are needed to improve the computer’s performance, be sure to make sure the specs are correct so resources are not wasted. 
- For RAM, check that the size of the RAM sticks already in the computer have the same memory size, support speed, and type of card as the ones purchased. 
- For SSDs, the internal memory size does not have to match but the transfer speeds still do.

## Why the WiFi adapter may not work in some installations 
After following all these steps, the WiFi option may not appear for some laptops after installation and a reboot. In Ubuntu, search for the Wi-Fi page in Settings and check if it says “No Wi-Fi Adapter Found”. If so, return to Windows and check what the WiFi card is under the Device Manager window. If it is a RealTek WiFi RTL8852, then the issue is that (as of 2022/2023) RealTek WiFi cards are not adapted to work with Linux distributions. To remedy the situation, choose one of the following options:
1. Purchase an external WiFi adapter from Amazon or other retailers.
	- Check that the product says it will work with Linux. 
    - The adapter may require drivers to be installed for the adapter to work as well, which are available from a CD or online.
2. Install a driver from a git repository. 
	- The correct repo will depend on the exact type of WiFi card. 
    	- For the 8852be there is this git repo [this git repo](https://github.com/HRex39/rtl8852be/tree/main). Follow the instructions on [this page](https://askubuntu.com/questions/1412219/how-to-solve-no-wi-fi-adapter-found-error-with-realtek-rtl8852be-wifi-6-802-11).

In either case, however, usage will require essential packages like build-essential, which are normally installed during Ubuntu installation but can be missed due to the lack of WiFi card support during installation. As a result, the Catch-22 of connecting to WiFi to install the packages and drivers needed to permanently connect to WiFi needs to be resolved. If using an external driver, see if the drivers can be installed via CD or from the adapter itself instead of from online. Otherwise, find a smartphone that has the function to pass Internet connection via tethering and use this connection temporarily to run apt commands and install all necessary packages for the desired drivers. Once all instructions for the chosen method are finished, it may take a few minutes, but then the WiFi adapter will be functional. 

## Summary
There are a few ways Ubuntu installation can go wrong or be delayed but this page hopefully will help a few people avoid major mistakes that held the writers of this page back a few weeks. After this guide the computer should be ready for installing browsers (such as Firefox), IDEs (such as VSCode or PyCharm), and libraries (such as mujoco or realsense-ros) as desired.

## See Also:
- [Ubuntu 14.04 on Chromebook](https://roboticsknowledgebase.com/wiki/computing/ubuntu-chromebook)
- [Upgrading Ubuntu Kernels](https://roboticsknowledgebase.com/wiki/computing/upgrading-ubuntu-kenel)

## Further Reading
- [Git repositories for drivers for different types of RealTek cards](https://www.github.com/lwfinger)
- [Instructions for how to install the driver listed in this page](https://www.askubuntu.com/questions/1412219/how-to-solve-no-wi-fi-adapter-found-error-with-realtek-rtl8852be-wifi-6-802-11)

## References
- [How to Dual Boot Ubuntu 22.04 LTS and Windows 10 | Step by Step Tutorial - UEFI Linux](https://www.youtube.com/watch?v=GXxTxBPKecQ)
- [The Best Way to Dual Boot Windows and Ubuntu](https://www.youtube.com/watch?v=CWQMYN12QD0)
- [Moving Space Between Partitions](https://gparted.org/display-doc.php?name=moving-space-between-partitions)


/wiki/computing/ubuntu-chromebook/
---
date: 2017-08-21
title: Installing stable version of Ubuntu 14.04 on Chromebook
---
Installing Linux on Chromebook Acer C-720 (stable version)

There are various installation variants for installing linux on Chromebooks. This is the one known stable version:
1. Get the custom Ubuntu 14.04 (64-bit) image available [here](https://www.distroshare.com/distros/get/12/).
2. Create a USB drive with image.
3. Boot Chromebook into Developer Mode.
4. Enable booting from USB device: `$ sudo crossystem dev_boot_usb=1 dev_boot_legacy=1`
5. Insert USB drive.
6. Reboot. Press Ctrl+L to boot from USB drive.
7. Install Ubuntu 14.04 LTS as usual.
  - Clear all partitions on `/dev/sda`
  - Make a new `swap` partition.
  - Make a new `ext4` partition with mount point: `/`
  - Continue to create a user ‘username'.
8. Once the installation is complete, reboot.
9. Press `Ctrl+L` to boot into Ubuntu.
10. Make sure you can connect to wireless network and have internet access.
```
$ sudo apt-get update; sudo apt-get -y dist-upgrade
$ sudo apt-get install git openssh-server
```
11. Whenever you restart, it will always say “OS is missing”. Do not fret. Just press `Ctrl+L` and you will boot to Ubuntu.
12. If `Ctrl+L` enables blip sounds then follow the above installation steps again. (This happens rarely).


/wiki/computing/upgrading-ubuntu-kernel/
---
date: 2017-08-21
title: Upgrading the Ubuntu Kernel
---
Following are the steps to be followed for upgrading the Ubuntu kernel:

For 64 bit processors(download the following amd64 files for your desired kernel)
Here the desired kernel is `3.19.0-generic`. The following commands should be run in a terminal:
1. `wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v3.19-vivid/linux-headers-3.19.0-031900-generic_3.19.0-031900.201504091832_amd64.deb`
2. `wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v3.19-vivid/linux-headers-3.19.0-031900_3.19.0-031900.201504091832_all.deb`
3. `wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v3.19-vivid/linux-image-3.19.0-031900-generic_3.19.0-031900.201504091832_amd64.deb`
4. `sudo dpkg -i linux-headers-3.19.0*.deb linux-image-3.19.0*.deb`
5. `sudo update-grub`
6. `sudo reboot`

> For 32 bit processors(download the i386 files instead of the above and follow the same steps).

After rebooting, check whether the kernel has upgraded using: `uname -r`. If on booting a screen appears saying kernel panic, then:
1. Restart the computer
2. Switch to the Grub menu
3. Go to Ubuntu Advanced Options
4. Select the older kernel that was working properly
5. This will take you to your older Ubuntu kernel
