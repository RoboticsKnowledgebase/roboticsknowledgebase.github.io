---
title: Vedder Open-Source Electronic Speed Controller
---

This page provides resources for the open-source electronic speed controller designed by Benjamin Vedder.

## Advantages
The main webpage of the Vedder Esc project is [found here](http://vedder.se/2015/01/vesc-open-source-esc/). The speed controller is designed for brushless DC motors with each controller driving a single motor. A summary of features is listed below:

- Voltage: 8V – 60V (Safe for 3S to 12S LiPo).
- Current: Up to 240A for a couple of seconds or about 50A continuous depending on the temperature and air circulation around the PCB.
- Sensored and sensorless FOC wich auto-detection of all motor parameters is implemented since FW 2.3.
- Firmware based on ChibiOS/RT.
- PCB size: slightly less than 40mm x 60mm.
- Current and voltage measurement on all phases.
- Regenerative braking.
- DC motors are also supported.
- Sensored or sensorless operation.
- A GUI with lots of configuration parameters
- Good start-up torque in the sensorless mode (and obviously in the sensored mode as well).
- Duty-cycle control, speed control or current control.
- Seamless 4-quadrant operation.
- Interface to control the motor: PPM signal (RC servo), analog, UART, I2C, USB or CAN-bus.
- Adjustable protection against:
  - Low input voltage
  - High input voltage
  - High motor current
  - High input current
  - High regenerative braking current (separate limits for the motor and the input)
  - Rapid duty cycle changes (ramping), High RPM (separate limits for each direction)

## Acquisition
The speed controller can be built from scratch, but it is far more time-effective to purchase a model from a supplier. These can be purchased for between $80 - $150 with various design modifications. A proven model is the VESC-X from Enertion. Despite being an Australian company, the controllers are produced and distributed in the US with a 3-5 day lead time. There is a quantity discount for 4+ controllers and if you place the controllers in your cart after registering and wait a bit, you'll get a 10% off coupon code by e-mail. The current code as of 3/5/2017 was *"pushingsucks".*

## Sample Arduino Code
The [code in this zip file](assets/DriveCharacterization.zip) interfaces an Arduino Mega 2560 to a VESC-X over serial. The sketch runs an open loop characterization routine that drives the motor in each direction. Motor braking is applied when a duty cycle command is set to 0. Motor freewheeling occurs when a current command is set to 0.
