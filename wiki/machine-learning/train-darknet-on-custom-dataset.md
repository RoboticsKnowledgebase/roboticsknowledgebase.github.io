This serves as a tutorial for how to use YOLO and Darknet to train your system to detect classes of objects from a custom dataset. We go over installing darknet dependencies, accessing the darknet repository, configuring your dataset images and labels to work with darknet, editing config files to work with your dataset, training on darknet, and strategies to improve the mAP between training sessions.

## Install Darknet Dependencies
### Step 1:
Install Ubuntu 18.04  
Make sure you have GPU with CC >= 3.0: <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>  
  

### Step 2:
CMake >= 3.18: <https://cmake.org/download/>  
Download Unix/Linux Source   

### Step 3:
CUDA 10.2: <https://developer.nvidia.com/cuda-10.2-download-archive>  

*Option 1:* 
Make a NVIDIA account  
Select Linux -> x86_64 -> Ubuntu -> 18.04 -> deb (local)  
Follow instructions & do Post-installation Actions  

*Option 2:*
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin  
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600  
$ wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb  
$ sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb  
$ sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub  
$ sudo apt-get update  
$ sudo apt-get -y install cuda  
$ nano /home/$USER/.bashrc  
```
Add the following to the bottom of the file  
```  
export PATH="/usr/local/cuda/bin:$PATH"  
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"  
```       
Save the file    
Close and reopen terminal  
Test for success with:    
```$ nvcc --version```  

**If it fails:**  
Restart computer    
Close and reopen terminal 
```
$ sudo apt-get autoremove    
$ sudo apt-get update    
$ sudo apt-get -y install cuda  
```

### Step 4:  
**OpenCV == 3.3.1 download from OpenCV official site:** 
```
$ git clone https://github.com/opencv/opencv  
$ git checkout 3.3.1      
``` 

### Step 5:  
**cuDNN v8.0.5 for CUDA 10.2:**
https://developer.nvidia.com/rdp/cudnn-archive   

**Download cuDNN Library for Linux (x86_64):**
https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar

**Extract it**  
```
$ tar -xzvf cudnn-10.2-linux-x64-v8.1.0.77.tgz  
```
**Copy files to CUDA Toolkit directory**     
```
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include   
$ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64   
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*  
```
**If it fails:**  
Download cuDNN Runtime Library for Ubuntu18.04 x86_64 (Deb)  
Download cuDNN Developer Library for Ubuntu18.04 x86_64 (Deb)  

if there is still an issue please visit the reference site.

## Setting Up a Custom Dataset for Darknet
### Step 1: Get the images
Collect the images for your dataset (either download them from open source datasets or capture images of your own). The images must be .jpg format.

Put all the images for the dataset into a folder called “images”

### Step 2: Get the labels
#### If you already have labels:

**Check to see if the labels are in the darknet format.**
If they are, put all of the labels for the images into a folder called “labels”. 

Darknet labels are accepted as:
`<object-class> <x_center> <y_center> <width> <height>` 
Where:
`<object-class>` - integer object number from 0 to (classes-1)
`<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
`<x_center> <y_center>` - are center of rectangle (not top-left corner)

**If you have labels for the images, but they are not in the darknet format:**
*Option 1:* Use Roboflow
Roboflow is an online tool that can convert many standard label formats between one and another. The first 1,000 images are free, but it costs $0.004 per image above that. Visit Roboflow here: <https://roboflow.com/formats>

*Option 2:* Write a script and convert the labels to the darknet format yourself.

Put all of the newly converted labels for the images into a folder called “labels”.

#### If you need to make labels for the images:
Create labels for all of the images using Yolo_mark [2]. The repo and instructions for use can be found here: <https://github.com/AlexeyAB/Yolo_mark>. These labels will automatically be made in the darknet format. Put all of the labels for the images into a folder called “labels”.

### Step 3: Create the text files that differentiate the test, train, and validation datasets.

  - Make a text file with the names of the image files for all of the images in the train dataset separated by a new line. Call this file “train.txt”. 

  - Make a text file with the names of the image files for all of the images in the validation dataset separated by a new line. Call this file “valid.txt”. 

  - Make a text file with the names of the image files for all of the images in the test dataset separated by a new line. Call this file “test.txt”. 

## Running Darknet  
### Step 1: Get the Darknet Repo locally and set up the data folders 
**If you do not already have the darknet github repo [1]:**
```$ git clone https://github.com/AlexeyAB/darknet  ```

**If you already have the github repo:**
```$ git pull```


### Step 2: Make Darknet 
```$ cd ./darknet  ```

**Check the Makefile and make sure the following as set as such:**  
```
GPU=1  
CUDNN=1  
OPENCV=1  
```
Save any changes and close the file.  

**Compile darknet**
```$ make  ```

### Step 3: Setup the darknet/data folder
Move the “images” and” labels” folders as well as the test.txt,  train.txt, and  valid.txt into the darknet/data folder

### Step 4: Setup the cfg folder
#### Create a new cfg folder in darknet:
```
$ mkdir custom_cfg
$ cd custom_cfg
```
#### Create the file that names the classes:
```
$ touch custom.names
```
Populate it with the names of the classes in the order of the integer values assigned to them in the darknet label format separated by new lines.

For example:
```
Light switch
Door handle
Table
```
Then in the labels, a light switch bounding box would be labeled with `0` and a table labeled with `2`.

#### Create the data file that points to the correct datasets:
```
$ touch custom.data
```
In custom.data, copy the following
```
classes= <num_classes>
train  = ./data/train.txt
valid = ./data/valid.txt
names =./custom_cfg/custom.names
backup = ./backup
eval=coco
```
Where `<num_classes>` is equal to an integer value corresponding to the distinct number of classes to train on in the dataset.

#### Create the cfg files
**Copy the cfg files to the custom cfg directory:**
```
$ cp cfg/yolov4-custom.cfg custom_cfg/
$ cp cfg/yolov4-tiny-custom.cfg custom_cfg/
```
**Edit the variables in the cfg files that are directly related to the dataset.**
> This information is taken from the darknet README but listed here for your convenience.  

If you are training YOLOv4, make these changes in ```custom_cfg/yolov4-custom.cfg```.  
If you are training YOLOv4-tiny make these changes in ```custom_cfg/yolov4-tiny-custom.cfg```.  

  - change line batch to: `batch=64`
  - change line subdivisions to: `subdivisions=16`
  - change line max_batches to:  `max_batches=<num_classes*2000>`
  > this number should not be less than number of training images, so raise it if necessary for your dataset 
  - change line steps to: `steps=<80% max_batches>, <90% max_batches>`
  - set network size `width=416 height=416` or any value multiple of 32: 
  - change classes to: `classes=<num_classes>`
  - change `filters=255` to `filters=<(num_classes + 5)x3>` in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers.


### Step 5: Download the weights files
For Yolov4, download this file and put it in darknet/custom_cfg/
<https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137>

For Yolov4-tiny, download this file and put it in darknet/custom_cfg/
<https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29>

### Step 6: Modify the config files for mAP improvement
  - Edits will be in yolov4-tiny-custom.cfg or yolov4-custom.cfg depending on if you are running YOLOv4-tiny or YOLOv4, respectively   
  - Make sure you aren't repeating a trial already tested
  - Document your training configurations and save the config file, best weights, and the mAP graph for each iteration of training
  - See the Tips & Tricks section for recommendations to improve mAP


### Step 6: Run Darknet 
**Compile darknet again after making changes**
```$ make  ```

#### Options for how to run darknet
**To run YOLOv4 on darknet in the foreground:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-custom.cfg custom_cfg/yolov4.conv.137 -map```

**To run YOLOv4-tiny on darknet in the foreground:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-tiny-custom.cfg custom_cfg/yolov4-tiny.conv.29 -map ``` 

**To run YOLOv4 on darknet in the background and pass output to a log:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-custom.cfg custom_cfg/yolov4.conv.137 -map  >  ./logs/darknet_logs_<date/time/test>.log 2>&1 & ``` 

**To run YOLOv4-tiny on darknet in the background and pass output to a log:**  
```$ ./darknet detector train custom_cfg/custom.data custom_cfg/yolov4-tiny-custom.cfg custom_cfg/yolov4-tiny.conv.29 -map   >  ./logs/darknet_logs_<date/time/test>.log 2>&1 &  ```


#### Check jobs to show command running:  
```$ jobs  ```

#### Show log:  
```$ tail -f ./logs/darknet_logs_<date/time/test>.log  ```

**Note**: if running in the background, Ctrl+C will not terminate darknet, but closing the terminal will  

At the end of training, find the weights in the backup folder. Weights will be saved every 1,000 iterations. Choose the weights file that corresponds with the highest mAP to save.  

**Repeat Steps 5 & 6 until a desired mAP is achieved.**


## Tips and Tricks for Training
### Train with mAP Graph
```./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -map```

### Change Network Image Size
Set network size width=416 height=416 or any value multiple of 32  

### Optimize Memory Allocation During Network Resizing  
Set random=1 in cfg   
This will increase precision by training Yolo for different resolutions.  

### Add Data Augmentation
[net] mixup=1 cutmix=1 mosaic=1 blur=1 in cfg  

### For Training with Small Objects
  - Set layers = 23 instead of <https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L895>  
  - set stride=4 instead of <https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L892>  
  - set stride=4 instead of <https://github.com/AlexeyAB/darknet/blob/6f718c257815a984253346bba8fb7aa756c55090/cfg/yolov4.cfg#L989>  

### For Training with Both Large and Small Objects  
Use modified models:  
  - Full-model: 5 yolo layers: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3_5l.cfg>  
  - Tiny-model: 3 yolo layers: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny_3l.cfg>  
  - YOLOv4: 3 yolo layers: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-custom.cfg>  

### Calculate Anchors for Custom Data Set
  - ./darknet detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416  
  - Set the same 9 anchors in each of 3 [yolo]-layers in your cfg-file  
  - Change indexes of anchors masks= for each [yolo]-layer, so for YOLOv4 the 1st-[yolo]-layer has anchors smaller than 30x30, 2nd smaller than 60x60, 3rd remaining  
  - Change the filters=(classes + 5)*<number of mask> before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.  

## Summary
We reviewed the start to finish process of using YOLO and darknet to detect objects from a custom dataset. This included going over the darknet dependencies, dataset engineering for format compatibilities, setting up and running darknet, and improving mAP across training iterations.

## See Also:
- Integrating darknet with ROS: <https://github.com/RoboticsKnowledgebase/roboticsknowledgebase.github.io.git/wiki/common-platforms/ros/ros-yolo-gpu.md>

## Further Reading
- Learn more about YOLO and the various versions of it here: <https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109>

## References
[1] AlexeyAB (2021) darknet (Version e83d652). <https://github.com/AlexeyAB/darknet>.  
[2] AlexeyAB (2019) Yolo_mark (Version ea049f3). <https://github.com/AlexeyAB/Yolo_mark>.  
