# Setting up WiFi hotspot at the boot for linux devices 
This tutorial will help you in setting up the WiFi hotspot at the boot for linux devices like Nvidia Jetson and Intel NUC. We will start with the procedure to setup the WiFi hotspot and then we will discuss how to change Wi-Fi hotspot settings in Ubuntu 18.04 to start it at bootup.

# Table of Contents
1. [Create a Wi-Fi hotspot in Ubuntu 18.04](#Create-a-WiFi-hotspot-in-Ubuntu-18.04)
2. [Edit WiFi hotspot settings in Ubuntu 18.04](#Edit-WiFi-hotspot-settings-in-Ubuntu-18.04 )

## Create-a-WiFi-hotspot-in-Ubuntu-18.04
1. To create a Wi-Fi hotspot, the first turn on the Wi-Fi and select Wi-Fi settings from the system Wi-Fi menu.
2. In the Wi-Fi settings window, click on the menu icon from the window upper right-hand side corner, and select turn On Wi-Fi hotspot.
3. New Wi-Fi hotspot always uses AP mode in Ubuntu 18.04, and the network SSID and password, as well as the security type (WPA/WPA2 is used by default in Ubuntu 18.04) are presented on the next screen which is displayed immediately after enabling the Wi-Fi hotspot.

There are two ways to edit hotspot settings (like the network SSID and password) which will be explained in next section.

## Edit-WiFi-hotspot-settings-in-Ubuntu-18.04
## Option 1:
1. After creating a hotspot for the first time, a file called hotspot is created which holds some settings. So make sure to create a hotspot first or else this file does not exist. 
2. In this file you can configure the network SSID it appears as ```ssid = ``` under ```[wifi]```), the Wi-Fi password is the value of ```psk=``` under ```[wifi-security]```), and other settings as required.
```
sudo nano /etc/NetworkManager/system-connections/Hotspot
```
3. In the same file set ```autoconnect=false``` to setup the hotspot at bootup automatically.
4. After making changes to the hotspot configuration file you'll need to restart Network Manager:
```
sudo systemctl restart NetworkManager
```

## Option 2:
