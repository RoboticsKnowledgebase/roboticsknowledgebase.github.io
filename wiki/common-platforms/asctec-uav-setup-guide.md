
# Asctec Pelican UAV Setup Guidance

This document acts as a tutorial on how to set up a [Asctec Pelican UAV](http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-pelican/) for autonomous waypoint navigation using ros pakage [asctec_mav_framework](http://wiki.ros.org/asctec_mav_framework). We are setting up an Asctec Quadrotor Pelican running with Ubuntu 14.04 and pre-installed ROS-jade.
<div style="text-align:center" markdown="1">

![](https://i.imgur.com/1RTxLVk.jpg)

</div>


# Table of Contents
1. [Useful Sites](#Useful-Sites)
2. [System Overview](#System-Overview)
3. [Setup Network and SSH](#Setup-Network-and-SSH)
4. [Install ROS Packages](#Install-ROS-Packages)
5. [Flash Programs in the HLP](#Flash-Programs-in-the-HLP)
6. [Run the ROS package on Atomboard](#Run-the-ROS-package-on-Atomboard)
7. [Summary](#Summary)
8. [References](#References)

## Useful Sites
Before we start, here are some useful sites for Asctec Pelican Information:
- [Asctec Pelican Official Documentation](http://wiki.asctec.de/display/AR/AscTec+Research+Home): General Description of the system, user manual, SDK,etc. (some contents are outdated)
- [Asctec User Discussion Forum ](http://asctec-users.986163.n3.nabble.com/): Great for troubleshooting
- [asctec_mav_framework ROS wiki](http://wiki.ros.org/asctec_mav_framework): Interface with HLP
- [asctec_drivers](http://wiki.ros.org/Robots/AscTec): Interface with LLP
> **Note**: `asctec_drivers` is develped to enable on-board computer communicate with Low Level Processor (LLP), while`asctec_mav_framework` enable on-boad computer and High Level Processor (HLP) communication. You need to decide which package to use based on your hardware configuration.

## System Overview
The structure of control subsystem of Asctec Pelican is shown below
![](https://i.imgur.com/XYMKcWM.png)
While the connection ports of AutoPilot Board is
![](https://i.imgur.com/frdwMy5.jpg)

>**Warning**: The [documentation](http://wiki.asctec.de/display/AR/AscTec+Atomboard) of Atomboard on-board computer on AscTec Wiki is based on **Atomboard 2** instead of **Atomboard 3**.

##  Setup Network and SSH
As we have an easy access to a monitor, a keyboard, and a mouse, we configed the network and ssh using a LVDS screen for the first time. [This blog](http://asctecbasics.blogspot.com/2013/06/basic-wifi-communication-setup-part-1.html) has detailed instruction if you need to set up the network without a screen. 

* After pluging in the monitor, keyboard and mouse, turn on the on-board computer. Follow the instruction on the screen to start the GUI. Setup WiFi connection and set up static IP (If you are using hotspot or standby WiFi router, the IP should be static usually). You can check the IP address using `ifconfig` command.

* Check if SSH is enabled on atomboard using `systemctl is-enabled ssh` and active `systemctl is-active ssh`. The output `enbaled` and `active`, but if not, go through [this link](http://ubuntuhandbook.org/index.php/2014/09/enable-ssh-in-ubuntu-14-10-server-desktop/) to enable ssh.
 
* Set up ssh on your **Master Computer** (your PC) use the above link as well.

* Test ssh: Connect your **Master Computer** to the same WiFi network with Atomboard and try `ssh asctec@IP_OF_ATOMBOARD` and enter the password (should be `asctec`).
* Configure `ROS_MASTER_URI` on **Atomboard**. Here we select our master computer to run the ros master, so use your favorite editor to open `~/.bashrc` and add 
    ```
     export ROS_MASTER_URI=YOUR_MASTER_COMPUTER_IP:11311
     export ROS_HOSTNAME=ATOMBOARD_IP
    ```
* Configure `ROS_MASTER_URI` on **Master Computer**. Again, use your favorite editor to open `~/.bashrc` and add 
    ```
    export ROS_MASTER_URI=YOUR_MASTER_COMPUTER_IP:11311
    export ROS_HOSTNAME=MASTER_COMPUTER_IP
    ```
* Now your Asctec Atomboard should be able to communicate with the master computer via ROS and you can run the talker/listener test to test described in [ROS Multimachine Tutorial](http://wiki.ros.org/ROS/Tutorials/MultipleMachines)to test the communication.

## Install ROS Packages
- Create a new catkin workspace
- git clone `asctec_mav_framework`
- Install all the required dependencies
    - bullet
- catkin_make
- Calibrate cameras

## Flash Programs in the HLP
- Download [AscTec SDK](http://wiki.asctec.de/display/AR/SDK+Downloads)
- Setup OpenOCD
    - [SDK Setup for Linux](http://wiki.asctec.de/display/AR/SDK+Setup+for+Linux)
    - [SDK Setup for Windows](http://wiki.asctec.de/display/AR/SDK+Setup+for+Windows)
        >**Note**: If the driver of JTAG is not properly installed, try to update the drivers in the device manager manually and point it to the `JTAG/oocd_link_treiber` subfolder in the AscTec_ARM_SDK installation folder. Windows tends to refuse this driver due to a missing signature. Please search for "Disable Windows Driver Signature Enforcement" for a tutorial on how to temporarily disable this check. Then you should be able to install the driver.
- If you are using Ubuntu 16.04 and openocd 0.8.0 or above, if you run the `sudo openocd -f lpc2xxx_asctecusbjtag05.cfg` when following the official instruction, you will run into this error:
    ![](https://i.imgur.com/nG55vEy.png)

    This is because openocd package update changes the syntax and no longer support ft2232. You can choose to either switch back to openocd 0.7.0 (which is not easy) or modify config file `lpc2xxx_asctecusbjtag05.cfg` to this.
    ```
    interface ftdi 

    #ftdi_layout_signal oocdlink 
    #ftdi_vid_pid 0x0403 0xbaf8 

    # Added by Onion 
    #ftdi_device_desc "OOCDLink" 
    ftdi_vid_pid 0x0403 0xbaf8 

    ftdi_layout_init 0x0508 0x0f1b 
    ftdi_layout_signal nTRST -data 0x0200 -noe 0x0100 
    ftdi_layout_signal nSRST -data 0x0800 -noe 0x0400 

    adapter_khz 5 
    telnet_port 4444 
    gdb_port 3333 

    # Use RCLK. If RCLK is not available fall back to 500kHz. 
    # 
    # Depending on cabling you might be able to eek this up to 2000kHz. 
    jtag_rclk 500 

    if { [info exists CHIPNAME] } { 
       set _CHIPNAME $CHIPNAME 
    } else { 
       set _CHIPNAME lpc2148 
    } 

    if { [info exists ENDIAN] } { 
       set _ENDIAN $ENDIAN 
    } else { 
       set _ENDIAN little 
    } 

    if { [info exists CPUTAPID ] } { 
       set _CPUTAPID $CPUTAPID 
    } else { 
       set _CPUTAPID 0x4f1f0f0f 
    } 

    adapter_nsrst_delay 200 
    adapter_nsrst_delay 200 

    # NOTE!!! LPCs need reset pulled while RTCK is low. 0 to activate 
    # JTAG, power-on reset is not enough, i.e. you need to perform a 
    # reset before being able to talk to the LPC2148, attach is not possible. 

    reset_config trst_and_srst srst_pulls_trst 

    jtag newtap $_CHIPNAME cpu -irlen 4 -ircapture 0x1 -irmask 0xf -expected-id $_CPUTAPID 

    set _TARGETNAME $_CHIPNAME.cpu 
    target create $_TARGETNAME arm7tdmi -endian $_ENDIAN -chain-position $_TARGETNAME 

    #-variant arm7tdmi-s_r4 

    $_TARGETNAME configure -work-area-phys 0x40000000 -work-area-size 0x4000 -work-area-backup 0 

    $_TARGETNAME configure -event reset-init { 
            # Force target into ARM state 
            arm core_state arm 

            # Do not remap 0x0000-0x0020 to anything but the flash (i.e. select 
            # "User Flash Mode" where interrupt vectors are _not_ remapped, 
            # and reside in flash instead). 
            # 
            # See section 7.1 on page 32 ("Memory Mapping control register") in 
            # "UM10139: Volume 1: LPC214x User Manual", Rev. 02 -- 25 July 2006. 
            # http://www.standardics.nxp.com/support/documents/microcontrollers/pdf/user.manual.lpc2141.lpc2142.lpc2144.lpc2146.lpc2148.pdf
            mwb 0xE01FC040 0x01 
    } 

    # flash bank lpc2000 <base> <size> 0 0 <target#> <variant> <clock> [calc_checksum] 
    set _FLASHNAME $_CHIPNAME.flash 
    flash bank $_FLASHNAME lpc2000 0x0 0x7d000 0 0 $_TARGETNAME lpc2000_v2 14745 calc_checksum 

    arm7_9 fast_memory_access enable 
    arm7_9 dcc_downloads enable 
    ```
    Refer to [this link](http://asctec-users.986163.n3.nabble.com/Cannot-connect-asctec-pelican-with-the-computer-using-JTAG-td4025048.html) for the detailed discussion.
- Connect your computer with HLP via JTAG
- Test connection
    - Check if the device is plugged in and recognized
        ```
        wayne@asctec:~/AutoPilot_HL_SDK$ ls /dev/ttyUSB*
        /dev/ttyUSB0
        ```
    - Connect to the device via OpenOCD 
        ```
        sudo openocd -f lpc2xxx_asctecusbjtag05.cfg
        ```
    - Open a telnet connection to OpenOCD 
        ```
        telnet localhost 4444
        ```
- Flash `main.hex` from `astec_hl_firmware` into HLP
    ```
    reset halt
    flash write_image erase main.bin
    reset run
    ```

## Run the ROS package on Atomboard
- change baud rate paramter in `asctec_mav_framework/asctec_hl_interface/launch/fcu_parameters.yaml` from 
`fcu/baudrate:57600` to `fcu/baudrate:460800`
- run `roslaunch asctec_hl_interface fcu.launch`
- check ROS topics by running `rostopic list`



Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.


## Summary
This tutorial gives an overview on how to set up the Asctec Pelican UAV. In summary, you could first setup the WiFi connection and enable SSH, followed by setting up ROS environment and communication, and then flash the `asctec_hl_firmware` into HLP. Now you should be able to ssh to the AscTec Atomboard and read all the sensor data from the autopilot board by running `roslaunch asctec_hl_interface fcu.launch`. The next step is to test motors and start your first flight!

## References
[1] AscTec Wiki: http://wiki.asctec.de/display/AR/AscTec+Research+Home
[2] AscTec Pelican Network Setup Blog: http://asctecbasics.blogspot.com/2013/06/basic-wifi-communication-setup-part-1.html
[3] AscTec JTAG Driver Setup Blog: 
http://asctec-users.986163.n3.nabble.com/Can-t-connect-with-JTAG-td4024671.html
[4] OpenOCD Syntax Error Discussion: http://asctec-users.986163.n3.nabble.com/Cannot-connect-asctec-pelican-with-the-computer-using-JTAG-td4025048.html
