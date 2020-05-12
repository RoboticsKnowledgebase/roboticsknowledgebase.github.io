---
date: 2019-11-11
Most of the mobile robot platform uses Linux based Single board computer for onboard computation and these SBCs typically have WiFi or an external WiFi dongle can also be attached. While testing/debugging we need to continuously monitor the performance of the robot which makes it very important to have a remote connection with our robot. So, in this tutorial, we will help you in setting up the WiFi hotspot at the boot for Linux devices like Nvidia Jetson and Intel NUC. We will start with the procedure to set up the WiFi hotspot and then we will discuss how to change Wi-Fi hotspot settings in Ubuntu 18.04 to start it at bootup.

# Table of Contents
1. [Create a Wi-Fi hotspot in Ubuntu 18.04](#Create-a-WiFi-hotspot-in-Ubuntu-18.04)
2. [Edit WiFi hotspot settings in Ubuntu 18.04](#Edit-WiFi-hotspot-settings-in-Ubuntu-18.04 )

## Create a WiFi hotspot in Ubuntu 18.04
This section will help you in setting up the WiFi hotspot at the boot for Linux devices. 
1. To create a Wi-Fi hotspot, the first turn on the Wi-Fi and select Wi-Fi settings from the system Wi-Fi menu.
2. In the Wi-Fi settings window, click on the menu icon from the window upper right-hand side corner, and select turn On Wi-Fi hotspot.
3. A new Wi-Fi hotspot always uses AP mode in Ubuntu 18.04, and the network SSID and password, as well as the security type (WPA/WPA2 is used by default in Ubuntu 18.04),  are presented on the next screen which is displayed immediately after enabling the Wi-Fi hotspot.

If you are ok with the defaults and don't want to change anything, that's all you have to do to create a Wi-Fi hotspot in Ubuntu 18.04.

## Edit WiFi hotspot settings in Ubuntu 18.04
There are two ways to edit hotspot settings (like the network SSID and password) which will be discussed in this section.
#### Option 1: Edit the hotspot configuration file. 
1. After creating a hotspot for the first time, a file called hotspot is created which holds some settings. So make sure to create a hotspot first or else this file does not exist. 
2. In this file you can configure the network SSID it appears as ```ssid = ``` under ```[wifi]```), the Wi-Fi password is the value of ```psk=``` under ```[wifi-security]```), and other settings as required.
```
sudo nano /etc/NetworkManager/system-connections/Hotspot
```
3. In the same file set ```autoconnect=false``` to set up the hotspot at bootup automatically.
4. After making changes to the hotspot configuration file you'll need to restart Network Manager:
```
sudo systemctl restart NetworkManager
```

#### Option 2: NM Connection Editor.
NM connection editor also allows you to modify the hotspot Wi-Fi mode, band etc. It can be started by pressing ```Alt + F2``` or using this command:
```
nm-connection-editor
```
All changes can be made directly in the nm-connection-editor in its corresponding tab. After making any changes using nm-connection-editor, you'll need to restart Network Manager.
```
sudo systemctl restart NetworkManager
```

To make sure all settings are preserved, start a hotspot by selecting turn On Wi-Fi Hotspot from the Wi-Fi System Settings once. Use the Connect to Hidden Network option for subsequent uses, then select the connection named Hotspot and click Connect.