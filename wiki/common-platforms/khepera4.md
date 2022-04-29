---
date: {}
title: Title goes here
published: true
---
This article with serve as an introduction, overview, and quick-start guide to K-Team's Khepera 4 robots. 

The Khepera 4 is a robot platform designed for indoor applications, meant to operate on tables and floor of labs. They are meant to provide a way to test algorithms developed in simulations or on paper in the real world. They're  most often used to test navigation, swarm, and artificial intelligence applications, providing researchers an easy way to see how their code runs on an actual robot. 

## Overview
Website about the Khepera 4: [link](https://www.k-team.com/khepera-iv)

The Khepera 4 is a circular, differential drive robot. It is 140.8 mm in in diameter and can carry up to 2 kg. It has a suite of sensors built-in: 3-axis gyroscope & accelerometer, 5 ultrasonic sensors, 12 infrared sensors, wheel encoders, and a front-facing camera. It also supports various add-ons through connectors at the top of the robot, allowing you to get [upward-facing camera](https://www.k-team.com/extensions-khepera-iv#stargazer) or install a [LIDAR sensor](https://www.k-team.com/extensions-khepera-iv#laserrangefinder). More detailed information can be found in the user manual: [link](https://ftp.k-team.com/KheperaIV/software/Gumstix%20COM%20Y/UserManual/Khepera%20IV%20User%20Manual%204.x.pdf)

## Quick-start guide
The quick-start guide will discuss how to program and compile for the Khepera 4, and how to get it onto the robot.

### Programming
To program for the Khepera 4, you will need a cross-compiler. A compiler turns high-level code into machine code that can be executed by a processor. A cross-compiler does the same thing, but for a processor other than the one compiling the code. While you will be developing and compiling the code on your computer, your computer won't run the code; the Khepera 4 will; therefore, you will need a cross-compiler.

The instructions for installing a cross-compiler can be found in the [user manual ch 5](https://ftp.k-team.com/KheperaIV/software/Gumstix%20COM%20Y/UserManual/Khepera%20IV%20User%20Manual%204.x.pdf#page=36). The light toolchain will most likely be sufficient for your use, unless you want to modify the kernel running on the Khepera 4. Follow the instructions to get the cross-compiler; when you get to the [FTP](https://ftp.k-team.com/KheperaIV/software/), I picked the *Gumstix COM Y* folder. Continue on with the installation, per the instructions. 

Once installation is complete, you can run the *make* command to use the cross-compiler, assuming the Makefile is set up properly. At the installed location, the *template* directory will provide example code and Makefile that you can play around with. Using this as an example for the rest of the article, when you run *make* it will use *prog-template.c* to generate *prog-template.o* and *template*; the latter is the binary that the Khepera 4 will actually use. Now, you have a binary you can execute on the Khepera 4!

### WiFi
Before the Khepera 4 can execute your code, you have to get it on their first. The easiest way is SCP, but you need a WiFi connection to do that. Straight out of the box, you probably won't be able to access it through WiFi, so here's how you set that up.

First, you need to connect to the Khepera 4 over WiFi:

1. Connect to the Khepera 4 using a USB cable; connect it to the robot's USB mini port. On Ubuntu, the robot should show up as */dev/ttyACM0*
2. Use a serial port software to communicate with it; I used minicom. To run minicom, run *minicom -s*
3. After running the command above, you can configure your settings; here's what you need to communicate properly: **TODO**

Once you have serial communication with the Khepera 4, you can access its files. You'll need to modify two files: */etc/systemd/network/wifi.network* and */etc/wpa_supplicant/wpa_supplicant-wlan0.conf*. *wifi.network* configures what the robot's IP address should be (or if it should be assigned dynamically by the router). *wpa_supplicant-wlan0.conf* configures what network it should connect to, and what username or password it should use if that's the case. Here's an example for a TPLINK router:

> Note: backslash below should actually be forward slashes
```
wifi.network:
[Match]
Name=wlan0

[Network]
DNS=192.168.0.1
Address=192.168.0.108\24
```
```
ctrl_interface=\var\run\wpa_supplicant
ctrl_interface_group=0
update_config=1

network={
    ssid="TP-Link_ASDF"
    proto=WPA2
    key_mgmt=WPA-PSK
    pairwise=CCMP TKIP
    group=CCMP TKIP
    psk="pwd"
    priority=10
    scan_ssid=1
}
```
Example above will configure the Khepera 4 to connect to WiFi named *TP-Link_ASDF* using password *pwd*, and it will have the IP address of *192.168.0.108* with subnet mask *255.255.255.0*. Make sure the DNS is valid for your specific WiFi!

Power cycling the robot after changing the WiFi settings is usually a good idea.

### ssh & scp
Now that you have the Khepera 4 connected to the internet, you can SSH into it, allowing you to communicate without relying on a USB cable. Make sure your computer is connected to the same WiFi as the one you configured the Khepera 4 to connect to. Then, use the following command to SSH into the khepera:
```
ssh root@192.168.0.108
```
If you're asked if you trust this IP address, say yes. Now, you can navigate the Khepera 4's computer like you would a terminal: *ls*, *cp*, *mv*, etc.

If you're still in the *template* directory, and you have the binary that you cross-compiled called *template*, then you can transfer it over to the Khepera 4 using SCP:

> Note: backslash below should actually be forward slashes
```
cp template root@192.168.0.108:/home/root
```
This will put the *template* binary file on the robot with IP address 192.168.0.108 at the location */home/root*.

## Example

## First subheading
Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Basic syntax
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.


#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
