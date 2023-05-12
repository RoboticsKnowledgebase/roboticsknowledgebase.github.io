---
date: 2023-05-12
title: Ubuntu Dual Boot and Troubleshooting Guide
published: true
---
This page serves as a tutorial and troubleshooting guide for dual booting Ubuntu alongside Windows for the uninitiated. This page assumes you have a PC or laptop with Windows 10 or 11 as the sole OS. Several difficulties can be encountered during setup if not aware of the process in advance. Read the following sections as needed and be aware of the potential issues brought up.

> It is recommended to begin the dual boot process as soon as possible in case things go wrong, or so that difficulties particular to the user’s hardware or desired setup are discovered as soon as possible.

## Create a bootable USB drive for Ubuntu
First, acquire an empty USB pen drive that is 8 GB or larger and insert it into the computer of choice. Have at least 64 GB of unused hard disk space on the computer (only 25-30 GB is needed for installation but the rest is needed for packages and coursework). 

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
