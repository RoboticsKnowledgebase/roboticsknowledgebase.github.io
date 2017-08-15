---
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
