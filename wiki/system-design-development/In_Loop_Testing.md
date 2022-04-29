---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-04-29 # YYYY-MM-DD
title: Hardware-in-Loop and Model-in-Loop Testing
---
Hardware-in-the-loop (HIL) testing is a test methodology that can be used throughout the development of real-time embedded controllers to reduce development time and improve the effectiveness of testing. As the complexity of electronic control units (ECUs) increases, the number of combinations of tests required to ensure correct functionality and response increases exponentially.

Older testing methodology tended to wait for the controller design to be completed and integrated into the final system before system issues could be identified.

Hardware-in-the-loop testing provides a way of simulating sensors, actuators and mechanical components in a way that connects all the I/O of the ECU being tested, long before the final system is integrated. It does this by using representative real-time responses, electrical stimuli and functional use cases 

The term ‘software-in-the-loop testing’, or SIL testing, is used to describe a test methodology where executable code such as algorithms (or even an entire controller strategy), usually written for a particular mechatronic system, is tested within a modelling environment that can help prove or test the so.

## Need Hardware in Loop and Software in Loop Testing : ADAS usecase

Primitive techniques for collision and lane change avoidance are being replaced with advanced drive assistance systems (ADAS). These new systems introduce new design and test challenges. Modern ADAS architectures combine complex sensing, processing, and algorithmic technologies into the what will ultimately become the guts of autonomous vehicles. As ADASs evolve from simple collision-avoidance systems to fully autonomous vehicles, they demand sensing and computing technologies that are complex. For example, consider the sensing technology on the Tesla Model S, which combines information from eight cameras and 12 ultrasonic sensors as part of its autopilot technology. Many experts are claiming that the 2018 Audi A8 is the first car to hit Level 3 autonomous operation. At speeds up to 37 mph, the A8 will start, accelerate, steer and brake on roads with a central barrier without help from the driver. The car contains 12 ultrasonic sensors on the front, sides, and rear, four 360° cameras on the front, rear and side mirrors, a long-range radar and laser scanner at the front, a front camera at the top of the windscreen and a mid-range radar at each corner. 
As a result, autonomous vehicles employ significantly more complex processing technologies and generate more data than ever before. As an example, the Tesla Model S contains 62 microprocessors – more the three times the number of moving parts in the vehicle. In addition, Intel recently estimated that tomorrow’s autonomous vehicles will produce four terabytes of data every second. Making sense of all this data is a significant challenge – and engineers have experimented with everything from simple PID loops to deep neural networks to improve autonomous navigation. 
This is the reason why we need Hardware-in-Loop and Software-in-Loop Testing and the importance of this has increased its importance more


Increasingly complex ADAS technology makes a lot of demands on test regimes. In particular, hardware-in-the-loop test methods, long used in developing engine and vehicle dynamics controllers, are being adapted to ADAS setups. 
Hardware-in-the-loop testing offers a few important advantages over live road testing. Most notably, a HIL setup can help you understand how hardware will perform in the real world, without having to take it outdoors.
•	By testing in the lab rather than on public highways, AV manufacturers can:
•	Accelerate testing and reduce costs - millions of miles of testing can be done in a shorter space of time in a lab
•	Avoid complex laws and regulations around public autonomous vehicle testing
•	Assess different hardware and sensor configurations earlier in the development process
•	Build a repeatable test process
•	Test weather or time-dependent edge cases at any time


## Hardware-in-Loop
HIL test systems use mathematical representations of dynamic systems which react with the embedded systems being tested. An HIL simulation may emulate the electrical behavior of sensors and actuators and send these signals to the vehicle electronic control module (ECM). Likewise, an ADAS HIL simulation may use real sensors to stimulate an emulation of the ECM and generate actuator control signals.
For example, a HIL simulation platform for the development of automotive anti-lock braking systems may have mathematical representations for subsystems that include the vehicle dynamics such as suspension, wheels, tires, roll, pitch and yaw; the dynamics of the brake system’s hydraulic components; and road qualities.
A point to note is that the in-the-loop notation means the reaction of the unit being tested influences the simulation. HIL methods verify and validate the software in the physical target ECUs. So HIL test benches must provide the ECU with realistic real-time stimuli and simulated loads. Multiple ECUs get tested in network HIL setups that wring out the bus systems, sensors, and actuators.
More recently, vehicle-in-the-loop methods use a physical car to replace vehicle simulations and most of the virtual ECUs. This approach comes in handy for testing safety critical ADAS functions by, say, crashing into virtual obstacles while operating the real vehicle.
The way an ADAS typically works is that it first processes the raw sensor data via feature and object recognition algorithms. The result is a data set that resembles a grid-based map of the environment or a list of recognized objects (such as trees, pedestrians, vehicles, and so forth). A situation analysis algorithm combines this processed data and estimates the current traffic situation. This situational analysis gets forwarded to the ADAS application. The application finally decides on actions to take such as slowing down or triggering an emergency brake.
Many HIL tests of ADAS functions boil down to sending simulated object lists to the UUT (unit under test). Depending on the sensor systems involved, the object lists may include information about object kinematics as well as whether the object is a pedestrian, vehicle, or something else.

Besides actual sensor data, the ADAS application needs supplementary vehicle data from other ECUs. This information usually passes to the UUT via the CAN bus and generally includes details like gear rate, acceleration, engine speed, steering angle, or GPS data. All ADAS require at least some of this information to check the plausibility of the situation analysis. The ADAS makes safety-critical interventions only if all this information is consistent.
When testing ADAS functions within HIL regimes, there can be a variety of UUTs and interfaces. For example, a single ECU might be tested while it runs core emergency functions such as a radar-based emergency brake-assist. Or a dedicated ADAS ECU might receive sensor information from other control units. Depending on the setup, multiple real ECUs may be part of the test bench and connected via automotive network systems like CAN, FlexRay, or Automotive Ethernet.
In an HIL test bench there can be several points at which virtual data is added. One option is feeding simulated data to physical sensors. Of course, real ADAS have multiple sensors so this strategy entails simultaneous generation of multiple signals. In the case of cameras, for example, engineers might show each camera a sequence of images of actual scenery via a screen or projector. For radar returns, the HIL system needn’t generate the radar output, just simulated echoes coming back.
One advantage of using physical sensors in HIL testing is that there’s no need to modify the UUT for testing purposes – the simulated signals come via the same physical interfaces as found in real vehicles. Among the biggest challenges is assuring the quality of the injected signals. For example, images projected on a screen might not represent the dynamic range a camera would see in real life. The classic example is that of a car driving straight into a blazing sunset, then descending a hill and plunged into dusk. All in all, the injection of data into a physical sensor involves few modifications of the UUT, but the accurate representation of scenarios can be quite demanding and not currently possible for all sensors. 
A typical test configuration for an ADAS HIL simulation depicts the different methods of simulating sensor data. Test equipment can inject digital data into the sensor channel to simulate sensor inputs. Alternatively, test equipment can develop sensor inputs in the form of images for cameras or echo patterns for radars to simulate actual sensed data. There are advantages and drawbacks to each approach.

There is also need to look into the latency issue of the HIL and Data communication. Having low latency HIL system is important especially in ADAS as car safety is very critical and speed testing is important.   More details of this particular topic can be found at [here]( https://www.spirent.com/blogs/how-to-create-a-realistic-low-latency-hil-environment-for-autonomous-vehicle-testing/)

## Platform based Hardware-in-Loop

Manufacturers such as National Instruments offer a platform-based approach to HIL testing. Key features of NI’s HIL test approach include a tight synchronization among numerous PXI instruments to enable simulations of driving scenarios and sensors. Particularly important in ADAS and autonomous driving applications is NI’s FPGA technology. FPGA technology enables engineers to design HIL test systems with extremely fast loop rates for quick decision making.
One recent example of an HIL test system using the platform-based approach to sensor fusion testing was demonstrated by a consortium called [ADAS Innovations] (https://www.spirent.com/blogs/how-to-create-a-realistic-low-latency-hil-environment-for-autonomous-vehicle-testing/) in Test. This group is a collaboration between NI alliance partners S.E.T., Konrad Technologies, measX, and S.E.A. At the recent NIWeek conference, the group demonstrated an ADAS test setup which can synchronously simulate radar, lidar, communications, and camera signals for an ADAS sensor. In one case, the setup was able to simulate a virtual test drive using IPG CarMaker and NI VeriStand software.
Modular systems such as PXI simulate many vehicular physical signals – effectively recreating the physical environment of the ADAS sensor or ECU. Synchronization is a critical requirement of the PXI modules – because all these signals must be precisely simulated in parallel. The ECU needs the radar, V2X and camera signal to arrive simultaneously if it is to process and understand the scenario and act accordingly.


HIL test techniques based on a platform let engineers simulate a virtually unlimited duration of “driving time.” The lengthy test time provides more opportunities for finding problems and lets engineers better understand how the embedded software performs in a wide range of situations.
As a result of simulating driving conditions in the lab, engineers can identify critical design flaws much earlier in the design process.



Other than Hardware-in-Loop Testing, there are few other types of testing techniques as well. I will brief them below: 

### Model-in-the-Loop (MIL) simulation or Model-Based Testing: 

Hardware-in-the-loop (HIL) simulations have been terms of motion, force, strain, etc. used successfully for a number of years. These have evolved from pure computer simulations of a con in which the test specimen is part real and part troller and plant, such that (usually) the controller is run in real-time and used to control the real plant, making use of the correct actuators and sensors. The model-in-the-loop concept in that the real-time simulation is part of the mechanical system, not of a controller. Only the key component with unknown dynamics real-time simulation in a MiL system interfaces with need be physically tested, reducing the cost and the rest of the system using actuators and sensors complexity of the physical test apparatus. that would not otherwise be present;

Model-in-Loop testing, also called MiL testing or model testing, is the testing of single or integrated modules in a [model-based development environment] (https://piketec.com/tpt/test-environments/simulink-targetlink-testing/), such as MATLAB Simulink from Mathworks or [ASCET from ETAS] (https://www.etas.com/en/company/ascet-model-based-development-of-application-software-mastering-in-real-time.php/).

When developing a system,  a model-based design (MBD) allows early identification and correction of errors and bugs because the models can be simulated easily in a very early design phase. Due to its early development stage, testing iterations (“loops”) can be finalized much faster than in longer (later) development stages. This approach is therefore a cost-efficient way to reduce development time significantly. Additionally, the graphical approach of programming is intuitive and loved by engineers.
Model-in-Loop testing is also related to unit testing, while the units under test are “model units”.
The MBD approach, and thus MiL testing, is commonly used in the automotive industry. Development processes used in MBD can be fully compliant with safety norms such as e.g. ISO26262 or IEC61508. These norms usually require MiL testing in early design phases.
The test level following MiL testing is Software-in-Loop testing (SiL testing) using software that is generated, oftentimes automatically, directly from the models, or written manually in C-code. For this test level, auto code generation is used. Common C-code generators for Simulink models are dSpace TargetLink or Mathworks Embedded Coder. 

See how Mathworks products facilitate model-based testing: https://www.mathworks.com/discovery/model-based-testing.html

### Software-in-the-Loop (SIL) simulation

Once your model has been verified in MIL simulation, the next stage is Software-in-Loop(SIL), where you generate code only from the controller model and replace the controller block with this code. Then run the simulation with the Controller block (which contains the C code) and the Plant, which is still the software model (similar to the first step). This step will give you an idea of whether your control logic i.e., the Controller model can be converted to code and if it is hardware implementable. You should log the input-output here and match it with what you have achieved in the previous step. If you experience a huge difference in them, you may have to go back to MIL and make necessary changes and then repeat steps 1 and 2. If you have a model which has been tested for SIL and the performance is acceptable you can move forward to the next step. Once both MiL and SiL testing phases are done, the individual test results can be tested “back-to-back”. Back-to-back testing is used for the comparison of test results between MiL and SiL testing.
See how to run SIL simulations using Embedded Coder: <https://www.mathworks.com/help/ecoder/software-in-the-loop-sil-simulation.html>

### Processor-in-the-Loop (PIL) or FPGA-in-the-Loop (FIL) simulation

The next step is Processor-in-the-Loop (PIL) testing. In this step, we will put the Controller model onto an embedded processor and run a closed-loop simulation with the simulated Plant. So, we will replace the Controller Subsystem with a PIL block which will have the Controller code running on the hardware. This step will help you identify if the processor is capable of running the developed Control logic. If there are glitches, then go back to your code, SIL or MIL, and rectify them.
See how to run PIL simulations using Embedded Coder: 

https://www.mathworks.com/help/ecoder/processor-in-the-loop.html

In the case of running the simulation on an FPGA instead of an embedded processor, the simulation is called FPGA-in-the-Loop (FIL). See how to run FIL simulations using HDL Verifier

https://www.mathworks.com/help/hdlverifier/ug/fpga-in-the-loop-fil-simulation.html



## In-Loop Testing and V-Loop Model


The V model is frequently used in the automotive industry to depict the relationships of vehicle-in-the-loop, hardware-in-the-loop, software-in-the-loop, and model-in-the-loop methods. Stages depicted on the left have a corresponding testing counterpart on the right. MiL settings test a model of the functions to be developed. MiL is applied in early stages to verify basic decisions about architecture and design. SiL setups test the functions of the program code complied for the target ECU but without including real hardware. HiL methods verify and validate the software in the physical target ECUs. ViL tests replace the vehicle simulation and most of the virtual ECUs with a real vehicle. The simulated parts of the environment are injected into the vehicle sensors or ECUs.

Advanced systems like autonomous vehicles are quickly rewriting the rules for how test and measurement equipment vendors must design instrumentation. In the past, test software was merely a mechanism to communicate a measurement result or measure a voltage. Going forward, test software is the technology that allows engineers to construct increasingly complex measurement systems capable of characterizing everything from the simplest RF component to comprehensive autonomous vehicle simulation. As a result, software remains a key investment area for test equipment vendors – and the ability to differentiate products with software will ultimately define the winners and losers in the industry.






















## Summary
The given article describes in detail what is in-loop testing with specific focus on ADAS as usecase. The importance and relevance of in-loop testing with V-model in systems engineering is also highlighted in the given model. 


## See Also:
< https://github.com/shahrathin/roboticsknowledgebase.github.io/blob/master/wiki/system-design-development/subsystem-interface-modeling.md>


## Further Reading

< https://www.guru99.com/loop-testing.html>



## References

< https://en.wikipedia.org/wiki/Hardware-in-the-loop_simulation>

