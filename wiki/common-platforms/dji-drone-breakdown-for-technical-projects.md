---
date: 2020-04-06
title: DJI Drone Breakdown for Technical Projects
published: true
---
DJI is one of the best selling drone companies in the world. Considering also the fact that a DJI Matrice M100 costs \$3500 and a M210 costs \$12000, compared to a couple hundred dollars for most other manufacturers, it is obvious that they have some of the best drones in the market. DJI’s hardware is one of the best if not the best at the writing of this article (2020). 

It’s a good choice for many projects where one would prefer not to build their own drone from scratch; however, it comes with important caveats especially for professional and research projects. Below we will look at its main drawback, and then give an introduction to its flight modes as well as general tips for drone projects.


## Controller Drawbacks
DJI drones have one of the best commercially available PID controllers along with its state of the art hardware, but for research and enterprise development users, the limitations imposed are noteworthy. After a certain depth, the software stack is locked from customization, and the controller performs much like a black box.

To be more specific, below we will look at the components involved in controlling a DJI drone.

![DJI Control Scheme](../../assets/images/DJI.png)

The keypoint is:
> DJI drones *need to read from its own GPS* to satisfy its EKF needs in order to produce proper state estimations for Position and Velocity Controls.

DJI drones in general rely heavily on the GPS for controls, and mostly doesn’t work if GPS isn’t connected or enabled. The EKF (Extended Kalman Filter) running inside the DJI drone needs the GPS to be connected and working in order to produce the necessary state estimation which in turn is responsible for producing the ultimate RC commands (Attitude commands).

This won't be an issue for regular users as mentioned before, but if the project relies on precise control and localisation of the drone beyond 2 meters accuracy, GPS becomes unreliable even when its health is at the max. To tackle this we can take the help of other better localisation sensors like radio beacons or even RTK GPS which usually provides centimeter level accuracy. 
(Read more about RTK here) However, the EKF and position/velocity controllers will need to be replaced.

That’s why for advanced users and researchers, there’s a hacky workaround by swapping out the DJI controller with an open source controller like the PX4. Imagine the customizability under your disposal by merging the best hardware with one of the best customizable controllers.

This is a growing trend in the research community, with robust and plentiful community support  behind the open source PX4 as well as the switch and integration with DJI drones.


## DJI flight modes
Below are the flight controller modes for DJI drones. You can also read more [here](https://www.heliguy.com/blog/2017/11/08/dji-intelligent-flight-modes/).

### Positioning Mode (P-Mode)
P-Mode is the standard flight mode for the majority of pilots. In this mode, all the sensors on the aircraft are active, GPS and any available vision or infrared sensors. This results in precise hovering and automatic breaking of the aircraft when no signals are given by the remote controller.

P-Mode requires a strong GPS signal to function and will disconnect if lost.

### Attitude Mode (ATTI Mode)
ATTI mode will only maintain the altitude of the aircraft and does not use any GPS or visual system data. The aircraft will therefore drift and move with wind, and needs to be manually controlled. Some pilots prefer this mode as it gives near-complete control of the aircraft without interference, but is more dangerous than the P-Mode.

Besides manual selection, the aircraft will also enter ATTI Mode if GPS signal is lost or if compass interference exists. All pilots should learn to fly in ATTI Mode as it’s likely to happen at some point during a flight, and is critical in avoiding crashes. Try it out in a large open space area with no obstacles first and get used to operating your aircraft in this mode.

ATTI Mode is available on all aircraft however, it cannot be manually selected on the Mavic Pro range.

### F-mode
This mode is used for running custom software stack to control the drone. It opens the controller to external interference.


## Drone project Tips
- Have a designated pilot for your team.
- Carry more than just one battery.
- Always have more than 2 batteries charged.
- If you can buy an extra drone, do it. Otherwise, buy 3 sets of essential parts.
- Always carry a battery warmer during winter (Can’t stress this enough)
- Add simulation to augment your project and increase productivity.
- Obtain a drone license. 
- Register your drone.
- Before purchasing a drone, consider the Payload as well if you’re going to place sensors and processors on top of it.
- Calculate the battery capacity based on the other items placed on the drone.
- Charge the battery only using the original charger.
- Consider the wind factor before attempting to fly. 
- Drones won’t operate if the battery temperature is less than a particular temperature (For DJI it’s 15 degree Celsius)
- Don’t operate it when there’s an animal nearby. Things can go bad in a very short period.
- Fly in an area where you’re licensed to fly.
- Calibrate the drone compass every time you fly
- If in P mode, make sure the GPS is fully functional. Else the drone will drift out of control.
- Make sure you place the drone in the desired orientation to make sure pitching forward actually pitches forward from the pilot’s POV.
- If you’re adding weight to the drone, make sure it’s balanced properly.
- Make sure it’s waterproof before trying out in rain or areas close to the water body.
- Make sure the right propellers are attached to the appropriate ones (CW and CCW)
- Make sure to use the offset Function (Trim) if you find the controller is not perfect.
- Change from P to ATTI mode when taking manual control during emergencies.

## Safety Tips
Quadcopters are basically flying lawnmowers.They can be dangerous if not operated carefully.
Here are some quadcopter safety precautions to keep in mind:

- If you’re about to crash into something, turn the throttle down to zero, so you don’t potentially destroy your quadcopter, injure somebody, or injure yourself.
- Keep your fingers away from the propellers when they’re moving.
- Unplug/take out the battery of the quad before doing any work on it. If it turns on accidentally and the propellers start spinning, you might have a tough time doing future flights with missing fingers.
- If you’re a beginner learning to fly indoors, tie the quadcopter down or surround it by a cage.


## Summary
If you are thinking of using DJI drones for your project, either be sure to stick with their GPS, implement your own EKF and Controller, or augment their drone with a PX4 controller to take advantage of the DJI hardware. Be sure to know the three flight control modes well, and follow general and safety tips for a successful project.

## See Also:
- [DJI SDK Introduction](https://roboticsknowledgebase.com/wiki/common-platforms/dji-sdk/)
