NETWORKING in UBUNTU
This page is to share some of the knowledge about networking our team has amassed using multiple computers networked together with ROS for our project.

OVERVIEW
First off, let's get some basic information clear. Go read some more authoritative work on this subject, like Networking Basics, and remember that this page could very well have many errors. But the basic idea is that you have two computers trying to talk to each other, and they have to agree on some basics - in human terms, something like what language to speak in and what to call each other. For the most part, the modern network tools that come with your operating system and Ubuntu or whatever linux you have on the robot will take care of this for you.

Essentially the way most students will use networking in their projects is to SSH into a small computer like a BeagleBone Black (BBB). For this, the setup is very straightforward. You need to make sure that the interface is initialized on your BBB. To test this, log into your BBB (if you can't ssh into it, you will need to use a monitor and keyboard) and then run the command 'ifconfig'. You should see 'eth#', where # is usually 0 or 1. If you don't, you need to add some information to your /etc/network/interfaces file. Running 'ifconfig -a' should show you the eth#, but you still need to add to your interfaces file.

Remember that this is going to override the settings of the network manager typically installed on ubuntu. You should be only seeing a command line on your BBB, loading xserver is a waste of resources and not necessary for most things. If you need to figure out how to disable xserver at boot, look here.

Once you are editing your interfaces file, you should see something like
auto lo
iface lo inet loopback

After this should be your ethernet settings. If you don't see anything, write something very similar to the above:
auto eth#
iface eth# inet dhcp

Remember that eth# is the number you saw in ifconfig. This will allow you to ssh into your BBB. You should generally try to set it up so that it shares your laptop's internet connection, so you can install packages and update the software.

You might need to get more advanced, however. If you want your robot to connect to another computer/robot, one of the two will have to decide on addresses to use. Your laptop will normally do this for the robot if you are sharing internet. The way this is done is using Dynamic Host Configuration Protocol, or DHCP. The recommended way to do this on Ubuntu (as of 14.04) is dnsmasq. This program allows for a lot more than just DHCP serving, but that is the only part we will need.

Install dnsmasq and then edit the /etc/dnsmasq.conf file. You will have to add your interface and tell it the range of addresses you want to give from that interface. The dnsmasq.conf file has examples of all of these, but for brevity the two lines you will want are:
interface=eth#

dhcp-range=eth#,10.0.0.80,10.0.0.150,255.255.255.0,12h

This will give a dynamic IP address to whatever you connect to eth# of between 10.0.0.80 and 10.0.0.150. The other settings should be fine, and can be left alone.

Once you have dnsmasq running in this fashion, you need to change how you describe your interface in your /etc/network/interfaces file. Instead of this line:
iface eth# inet dhcp
you want:
iface eth# inet static
address 10.0.0.1
netmask 255.255.255.0

This will put you on the same subnet (last octet of the IP address) as your connecting computer, and is outside the range you are dynamically giving so you can be sure there won't be overlaps.

If you need to connect multiple robots over multiple ethernet ports (we used a bunch of usb->ethernet dongles, for example) and want them all to connect you will need to also break IP forwarding. This will allow computers on different subnets to talk to each other. In /etc/sysctl.conf remove the '#', which acts like a comment, from this line:
net.ipv4.ip_forward=1

Finally, if you are using ROS make sure you are changing ROS_MASTER_URI and ROS_HOSTNAME on both computers.
