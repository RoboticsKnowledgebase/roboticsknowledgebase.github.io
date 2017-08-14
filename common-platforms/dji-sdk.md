---
 title: Outdoor UAV navigation (focus on DJI SDK)
---
While DJI has been developing great hardware and software to make relatively safe UAV technology commercially available and has made available an amazing SDK to help research and development, sometimes it becomes difficult to get details on their implementation as their documentation is pretty limited. This article will help a beginner quickly understand what kind of flight information they can get using DJI SDK and how they can use it for their specific use case.

It's critical to understand coordinate systems that DJI aircrafts use. [This link](https://developer.dji.com/mobile-sdk/documentation/introduction/flightController_concepts.html) is a good guide.

Following are a few important points which follow from the information given on the link above but are not directly stated, which I found to be really important:

1. Since the aircraft uses a NED (North-East-Down) coordinate system which aligns the positive X, Y, and Z axes with the North, East, and Down directions, respectively, the attitude information (roll, pitch, and yaw) we get for the drone at any point is independent of the roll, pitch, yaw of the aircraft at the take-off location.
- It follows from the above that the yaw information we get from the data gives us the magnetic heading of the aircraft. This heading is measured with respect to the magnetic North and is positive towards East. The value of yaw (and thus, heading) ranges from -180 degrees to +180 degrees.
- Since the origin of the world coordinate system is usually the take-off location, altitude reported in the flight data is with respect to the starting location. So, even if you are flying on a hilly terrain with altitude (used by the DJI SDK) set to a constant number, the aircraft’s altitude with respect to the ground points over which it flies will keep changing because the aircraft controls try to make the aircraft altitude constant with respect to the take-off location.

## Importance of using a Compass
If you got the second point above, then you probably understand outdoor navigation well enough. If not, [this link](http://diydrones.com/profiles/blogs/the-difference-between-heading) gives a really concise overview of the terms used in navigation. One important point to note is that heading does not give the direction in which the drone is flying, it gives the direction to which the drone points while flying. And, the heading we get using the DJI SDK is with respect to the magnetic North, not true North. [A forum discussion](http://forum.dji.com/thread-14103-1-1.html) makes discussion makes clear the importance and need of compass even though the aircraft might have a good GPS receiver. Additionally, it discusses the compass's use in various modes, like ‘Course-lock’ that DJI offers. It also sheds light on why compass calibration is important and what kind of changes in magnetic fields it compensates for.

## Waypoint Navigation
Want to create a waypoint navigation app of your own using DJI Mobile SDK? You may find the following links useful:
- DJI Quick Start: https://developer.dji.com/mobile-sdk/documentation/quick-start/index.html
- DJI Android Tutorial: https://developer.dji.com/mobile-sdk/documentation/android-tutorials/GSDemo-Google-Map.html
- The following Github repositories give you demos to create DJI Waypoint Mission apps:
  - For Android platform: https://github.com/DJI-Mobile-SDK-Tutorials/Android-GSDemo-GoogleMap
  - For iOS: https://github.com/DJI-Mobile-SDK-Tutorials/iOS-GSDemo
