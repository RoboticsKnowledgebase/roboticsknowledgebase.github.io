---
title: Motor Controller with Feedback
---
In most of the applications for a DC geared motor speed control or position control might be important. This is achieved by implementing feedback control.

In order to use feedback control, we need information about the state of the motor. This is achieved through the use of encoders mounted on the motor shaft. Typically, data from this encoder is fed into a microcontroller, such as an Arduino. The microcontroller would need to have a code for PID control. Another easier option would be to use a motor controller which has the ability to read data from the encoder. One such controller is Pololu Jrk 21v3 USB Motor Controller with Feedback. More details about this component can be found [here](https://www.pololu.com/product/1392.).
