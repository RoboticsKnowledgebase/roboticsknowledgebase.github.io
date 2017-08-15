# udev Rules
[udev](https://en.wikipedia.org/wiki/Udev) is the system on Linux operating systems that manages USB devices. By default, when you plug in a USB device, it will be assigned a name like `ttyACM0`. If you are writing code and need to communicate with that device, you will need to know the name of the device. However, if you plug the devices into different ports or in a different order, the automatically assigned names might change. To solve this, you can create custom udev rules to always assign specific devices to a particular name.

> Note that you can put more than one rule in a file -- just make sure that they are on separate lines.

## Arduino
With Arduinos, the process is fairly simple:

1. Determine the device's automatic serial port assignment. This can be done by doing the following:
  - Leave the Arduino unplugged.
  - Run `ls /dev` from the command line
  - Plug in the Arduino.
  - Run `ls /dev` from the command line and see which new item appeared. It will probably have a name like `ttyACM*`.
2. Find the device's USB parameters using the device name from the prior step:
    ```
    udevadm info -a -n /dev/ttyACM*
    ```
3. Create a file in `/etc/udev/rules.d` called `99-usb-serial.rules`
Enter the following in the file, replacing the parameters with those found in the prior step:
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042", ATTRS{serial}=="85334343638351804042", SYMLINK+="arduino"
```
4. Unplug and reconnect the Arduino. It should now show up as a link in the `/dev` folder called `arduino`. You can check this with `ls -l /dev`

## Other Devices
With devices other than Arduino, you may need to use different parameters to identify the device, or you may need to perform an extra step to get the device's serial ID. Generally, you will have to figure this out on a device-by-device basis. For instance, the udev rule for a Hokuyo looks like this:
```
ATTRS{manufacturer}=="Hokuyo Data Flex for USB", ATTRS{idVendor}=="15d1", MODE="0666", GROUP="dialout", PROGRAM="/opt/ros/indigo/env.sh rosrun hokuyo_node getID %N q", SYMLINK+="sensors/hokuyo_%c"
```
The Hokuyo requires a special program (`getID`) to get the serial ID. It can be downloaded as part of the `hokuyo_node` ROS package.
