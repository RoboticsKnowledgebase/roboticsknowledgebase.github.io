---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-12-05 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Externally Referenced State Estimation for GPS Lacking Environments
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---
Robotic systems will often encounter environments where GPS access is either lacking or denied. In these environments, it's important to still maintain some accurate global representation of the robot's state, both for localization and mapping purposes. This is a relatively common problem in robotics, and is commonly dealt with through a combination of proprioceptive sensors and an estimation of the local environment in order to perform [SLAM](https://www.mathworks.com/discovery/slam.html). This is effective in environments where there are unique features to localize off of, but can struggle in uniform or dynamic environments. For that reason, there is benefit in considering sensor suites that take advantage of external infrastructure to provide a fixed positional reference. This fixed positional reference can mitigate drift or catastrophic localization error for robotic systems where signifacant state estimation errors are disallowed or not preferable. 


## Common Types of External Referencing Positioning Systems

### UltraWideband Beacons (UWB)

UltraWideband is a short-range wireless communication protocol, allowing connected devices to communicate across short ranges, similar to bluetooth. Unlike blueooth, UltraWideband transmits over a wider frequency, allowing for a more precise and wider bandwith of communication, albeit over a shorter range. 

Ultrawideband positioning takes advantage of the communication pulses to sense distances between transmitters through a two-way ranging protocol described in detail [here](https://forum.qorvo.com/uploads/short-url/5yIaZ3A99NNf2uPHsUPjoBLr2Ua.pdf). As a simplification, it is able to measure the time period that messages take to transmit between devices, while accouting for potential clock and frequency shift between devices. 

By using multiple stationary devices, a single or multiple mobile beacons can be tracked by combining ranges through trilateration. 

![Example usage of a DWM1001 setup](assets/decawave_example_multi_anchor.png)
[Source](https://www.researchgate.net/profile/Teijo-Lehtonen/publication/281346001/figure/fig4/AS:284460038803456@1444831966619/DecaWave-UWB-localization-system-SDK-5.png)

At the time of writing, one of the most common modules for UWB is the DWM1001. Information of how to get started with that can be seen [here](LINK TO WIKI PAGE ON DWM1001). Since these modules are mass manufactured, they can be purchased very inexpensively and should be considered one of the most affordable options for positioning systems.

### Ultrasonic Positioning

Ultrasonic positioning works in a similar way to UWB, but rather than transitting frequencies at a very high frequency, the products instead rely on a combination of lower frquency communication pulses and beamforming. By using a sensor array on each device, they are able to claim a 2D positioning accuracy of +-2cm.


![Example usage of a DWM1001 setup](assets/marvelmind_example.jpg)
[Source](https://marvelmind.com/)

As an important note, ultrasonic pulses are harmful to human hearing over an extended period of time and should not deployed around humans without ear protection.

### Robotic Total Stations

Total stations have an extended heritage in civil engineering, where they have been used to precisely survey worksites since the 1970s. The total station sends beams of light directly to a glass reflective prism, and uses the time-of-flight properties of the beam to measure distances. The robotic total station tracks it's calibration orientaiton to high precision, such that the measured distance can be converted into a high-precision 3D position mesaurement. Total stations, depending on the prism type and other factors, can accurate track with in millimeter range at up to 3.5km [Leica-Geosystems](file:///home/john/Downloads/Leica_Viva_TS16_DS-2.pdf).

![Example usage of a Total Station in the Field](assets/leica_field_image.jpg)
[Source](https://leica-geosystems.com/)

## Key Factors to Consider

When considering external referencing positioning systems, it's important to consider the following:

- What is the environmental conditions? Is the robot running indoors or outdoors? interference)
- Will there be humans in environment
- Will the robot have line-of-sight to the fixed infrastructure?
- What are the accuracy and precision requirements?


## Helpful Links
- Links to relevant material within the Robotics Knowledgebase go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.
