---
title: Blink(1) LED
---
## Introduction
Using an LED indicator is an extremely useful troubleshooting technique. The [Blink(1) USB RGB LED](https://blink1.thingm.com/) makes this very easy and straightforward. Using this product can help determine the state of the robot without needing to have a direct data link. This can be useful on mobile robots, especially drones where cables are not an option. It can run on any operating system, including anything running on your single board computer.

## Purchasing and Setup
To purchase the Blink(1) LED, go to [their home page](https://blink1.thingm.com/) and click "Buy". At this time of writing the cost was around $30.

Their software needs to be downloaded and installed from their web site as well. It is a very small set of files, so it does not take up much storage space and will not take long to download. Follow the link to the home page and click "downloads" and follow the instructions for the blink1-tool command-line. In Linux, the easiest way to install this is by entering the following commands in a terminal from your home directory.
```
$ git clone https://github.com/todbot/blink1.git
$ cd blink1/commandline
$ make
```
## Getting Started
Once you have your product and the software is installed, try controlling the LED from the terminal command line. In Linux, navigate to the `/blink1/commandline` folder.
```
$ cd ~/blink1/commandline
$ ls
```
There should be an executable file called blink1-tool shown in green. This is the program that controls the LED. While in this folder, try some of the following commands.
```
$ ./blink1-tool --on
$ ./blink1-tool --off
$ ./blink1-tool --green
$ ./blink1-tool --red --blink 5
```
These command can be run even if you are not navigated to the `~/blink1/commandline` folder by adding the entire path to the command as follows.
```
$ ~/blink1/commandlin/blink1-tool --on
```
For a full list of commands and options, see the [Blink1 Tool Tutorial](https://github.com/todbot/blink1/blob/master/docs/blink1-tool-tips.md).

## Integration with ROS
In any ROS node, you can write text out to a command line. Here is a guide on how to do this in c++.

First, make sure you import this package at the top of your file.
```
#include <stdlib.h>
```
With this package included, you can simply output a string using the system() command and it will execute that string as if you typed it into a command line. For example:
```
std::string output_string;
output_string = "~/blink1/commandline/blink1-tool --on";

char* output_char_string = new char [output_string.length()+1];
std::strcpy(output_char_string, output_string.c_str());

system(output_char_string);
```
Running this inside your node will turn the LED on. Note that `system()` accepts a C-type string, as apposed to a C++ string. It is likely easier to manipulate a C++ string in your code (using the "+" operator to concatenate, for example), so I suggest converting it to a C-type char string just before calling the `system()` command.

In your code, I would suggest making a node that subscribes to any topics that include information that you would like to check. For our project, we subscribed to our state estimator, vision node, and the flight mode flag. Create a function that contains a series of `if()` statements that checks all of the conditions you would like to visualize. Then assign a color to each condition and create a string based on that color code. At the end of this function, send that string to `system()`.
