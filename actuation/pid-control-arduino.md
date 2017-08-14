---
title: PID Control on Arduino
---
The main idea behind PID control is fairly simple, but actually implementing it on an Arduino can be tricky. Luckily, the Arduino IDE comes with [a standard library for PID](http://playground.arduino.cc/Code/PIDLibrary).

On top of that library you can also install [a package that auto-tunes your PID gains](http://playground.arduino.cc/Code/PIDAutotuneLibrary). Because it's not a standard library, you have to download the code and place in your Arduino/libraries folder. The Arduino is usually in your documents folder or wherever you installed the Arduino program.

If your sensor readings are very noise you might want to consider adding a Kalman filter before your PID control loop. For example, for a gyro or an accelerometer generally have zero mean gaussian noise, perfect for a Kalman filter. Check out [this guide](http://forum.arduino.cc/index.php?topic=58048.0) to implement a filter on an Arduino.

Finally, you can write your own PID software using [this guide](http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/). You might want to do this if you want to add custom features or just want to learn more about controls. Only recommended for advanced users.
