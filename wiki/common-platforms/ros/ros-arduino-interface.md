---
date: 2020-04-06
title: ROS Arduino Interface
published: true
---
The Arduino Microcontroller is a versatile piece of hardware that can be used in various applications to integrate hardware and software components in a short span of time. Fortunately, It can also be used with Robot Operating System (ROS) without much effort. The following tutorial will guide you through the process.

## Setup
The connection is based on serial communication and there is one big advantage:
you don’t need to install the Arduino IDE in the host computer when running it. In other words, you don’t even need a GUI to enable it. However, there are quite a few items to install to import the ros library onto arduino to code.

Type the following code in the terminal : (Replace Indigo with your current version of ROS)
```
sudo apt-get install ros-indigo-rosserial-arduino
sudo apt-get install ros-indigo-rosserial
```

The preceding installation steps created the necessary libraries, now the following will create the ros_lib folder that the Arduino build environment needs to enable Arduino program	s to interact with ROS.

`ros_lib` has to be added in the arduino libraries directory. By default, it will be typically located in 
`~/sketchbook/libraries` or in `my documents`.
If there is a `ros_lib` in that location already, delete it and run the following commands.
```
cd ~/sketchbook/libraries
rm -rf ros_lib
rosrun rosserial_arduino make_libraries.py .     
```
(DO NOT FORGET THE PERIOD AT THE END)

## Example Code - Publisher
Now you have everything set up to start writing your publisher and subscriber in Arduino IDE.
After doing the above steps, you should be able to see `ros_lib` under File -> Examples.

I’ll step you through an example program and help you run it as well at the end as it might be tricky at first.

Let’s talk about a basic publisher subscriber program. If you are confused or want to refresh your memory on Ros Publishers and Subscribers, checkout [Ros Pub-Sub](http://wiki.ros.org/action/fullsearch/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29?action=fullsearch&context=180&value=linkto%3A%22ROS%2FTutorials%2FWritingPublisherSubscriber%28c%2B%2B%29%22). Once you understand that, you can head back and continue from here.

Here is an example from the official repository and also an example of my own which uses a Boolean data type rather than a String.

Basic Hello world program:
```
#include <ros.h>
#include <std_msgs/String.h>

ros::NodeHandle nh; //Node Handler ()

std_msgs::String str_msg;	//initialise variable 
ros::Publisher chatter("chatter", &str_msg);

char hello[13] = "Hi, This is my first Arduino ROS program!";

void setup()
{
 nh.initNode();
 nh.advertise(chatter);
}

void loop()
{
 str_msg.data = hello;
 chatter.publish( &str_msg );
 nh.spinOnce();
 delay(1000);
}
```

Explanation:    
- [Node Handler](https://answers.ros.org/question/68182/what-is-nodehandle/)
- [Initialise Node (InitNode)](http://wiki.ros.org/rospy/Overview/Initialization%20and%20Shutdown)
- [spinOnce](https://answers.ros.org/question/11887/significance-of-rosspinonce/)

The above program can be used to send some messages to ros through serial communication. But if you are working on a real project, sending characters or String has very minimum use as you may know. So, let’s also go through the same example, but with some modifications to use a different data type, like a Boolean. Boolean, in my opinion, is the most used data type to flag different aspects of a program and to make decisions based on that to enable or disable various items.
```
#include <ros.h>
#include <std_msgs/Bool.h>

ros::NodeHandle nh; //Node Handler ()

std_msgs::Bool bool_flag;	//initialise variable 
ros::Publisher chatter("chatter", &bool_flag);  # 

Bool flag = true;

void setup()
{
 nh.initNode();
 nh.advertise(chatter);
}

void loop()
{
 bool_flag.data = flag;
 chatter.publish( &bool_flag );
 nh.spinOnce();
 delay(1000);
}
```
The only difference between the code above and the one before is the change in datatypes and the initialized values. You could use this to process information and send true or false information based on the condition. An example would be to monitor the temperature in a room with a sensor and when it's greater than a threshold, send out a warning message or turn on the Air conditioner.

This can be achieved directly by connecting the sensors to Arduino itself, but in more complex applications it is often desirable for ROS to handle all communications between devices. In that case, this can be achieved by subscribing to a topic which is published on ROS by something else. So, let’s see how to write a subscriber in Arduino.

## Example Code - Subscriber
```
#include <ros.h>
#include <std_msgs/Empty.h>

ros::NodeHandle nh;
 
void messageCb(const std_msgs::Empty& toggle_msg){
 digitalWrite(13, HIGH-digitalRead(13));   // blink the led
}

ros::Subscriber<std_msgs::Empty> sub("toggle_led", &messageCb );

void setup()
{
 pinMode(13, OUTPUT);
 nh.initNode();
 nh.subscribe(sub);
}

void loop()
{
 nh.spinOnce();
 delay(1);
}
```
Explanation for the above code:

It looks pretty similar to a generic non-arduino ROS subscriber. The topic being subscribed to `toggle_led` doesn’t have any data being published through it but the availability of the topic itself triggers a call back which would turn on the LED on pin 13 of the arduino. This is different from a usual subscriber wherein the subscriber subscribes for some data from the topic, say a boolean or integer, but it is `std_msgs::Empty` in this case. This is probably not known by many of us and if that’s the case, you learned a new thing today! You can subscribe to an Empty Data type and have a call back based on that.

As an addition, let’s also have a look at an example which actually subscribes to a topic with some data unlike the previous case. Specifically, let’s look at subscribing to a topic which has boolean data.
The following example is to toggle the magnet on and off based on the bool value subscribed under the Ros Topic - `magnet_state`
```
#include <ros.h>
#include <std_msgs/Empty.h>

ros::NodeHandle nh;

int flag;
bool magnet_state;

void call_back( const std_msgs::Bool& msg){
magnet_state = msg.data;
flag = 1;
}

ros::Subscriber<std_msgs::Bool> sub(“magnet_state", call_back );
void setup() {
nh.initNode();
nh.subscribe(sub);
}

void loop() {
if ((magnet_state== false) && flag)
{
        //Turn magnet on;
}
else if ((magnet_state == true) && flag)
{
        //Turn magnet off; 
}

nh.spinOnce();
}
```

## Running the code
Now, since we are done with all of the basic ways and examples to write an Arduino Pub Sub code, let’s see how to run it.

First and foremost, Upload the code to Arduino and make sure all the relevant connections are established.

Now, launch the roscore in a new terminal window by typing:

`roscore`

Next, run the rosserial client application that forwards your Arduino messages to the rest of ROS. Make sure to use the correct serial port:

`rosrun rosserial_python serial_node.py /dev/ttyUSB0`

In the above line, `/dev/ttyUSB0` is the default port. It might be `/dev/ttyACM0` depending on your local machine. The number at the end may vary depending on the COM port the arduino is connected to.

Once the publisher and the subscriber are turned on, and the above code is executed, check with the rostopic list command to see if it is working. You should see the names of the topic that is being published and the topic that is being subscribed to.

Note: Once the Arduino has the code, it can be used in any other machine with ROS. The arduino IDE and the `ros_lib` library are not necessary to run rosserial. (No GUI as such is needed.)

Run the following after connecting the required items.
```
sudo apt-get install ros-indigo-rosserial-arduino
sudo apt-get install ros-indigo-rosserial

rosrun rosserial_python serial_node.py /dev/ttyUSB0
```
Note in many cases user permission can be an issue. If the connection is denied when trying to establish the rosserial connection, type the following command.
```
sudo chmod +x /dev/ttyUSB0
```
for example should open the port.

Alternatively, try 
```
sudo chmod a+rw /dev/ttyUSB0
```
if the above command doesn’t work.

## See Also:
- [ROS Introduction](https://roboticsknowledgebase.com/wiki/common-platforms/ros/ros-intro/)
