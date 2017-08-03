Installing stable version of Ubuntu 14.04 on Chromebook

Installing Linux on chromebook acer C-720 (stable version)

There are various installation variants for installing linux on chromebooks. I tried a couple of them but none of them were stable. This is the one stable version I came across and wish to share it with people who may face

1. Get the custom Ubuntu 14.04 (64-bit) image available here: https://www.distroshare.com/distros/get/12/
2. Create a USB drive with image.
3. Boot Chromebook into Developer Mode.
4. Enable booting from USB device.
$ sudo crossystem dev_boot_usb=1 dev_boot_legacy=1
5. Insert USB drive.
6. Reboot. Press Ctrl+L to boot from USB drive.
7. Install Ubuntu 14.04 LTS as usual.
7a. Clear all partitions on /dev/sda
7b. Make a new swap partition.
7c. Make a new ext4 partition with mount point: /
7d. Continue to create a user ‘username'.
8. Once the installation is complete, reboot.
9. Press Ctrl+L to boot into Ubuntu.
10. Make sure you can connect to wireless network and have internet access. .
$ sudo apt-get update; sudo apt-get -y dist-upgrade
$ sudo apt-get install git openssh-server
11. Whenever you restart, it will always say “OS is missing”. Do not fret. Just press ctrl+L and you will boot to uBuntu.
12. If ctrl+L enables blip sounds then follow the above installation steps again. (This happens rarely).
