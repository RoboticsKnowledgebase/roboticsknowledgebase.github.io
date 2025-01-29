/wiki/system-design-development/cable-management/
---
date: 2017-08-15
title: Cable Management
---
Effective cable management is essential for servicing, debugging and creating reliable robotic systems. Cabling is often regarded as one of the less glamorous parts of electrical engineering, but it is an essential and an extremely important part of creating a complete system.

This page highlights the important steps for cable management including **Planning, Cable Creation and Installation.** If you decide to only read one part of this page, read the **Planning** section. This is the most important part of cable management that is often skipped. It's hard to skip the other two steps.

To illustrate its importance imagine the following scenario:

You are at the final integration steps of your project and something isn't working. You start by checking out the hardware and see Figure 1.

![Cable Management Gone Wrong](assets/CableManagement-c5b3a.png)

**Figure 1: Cable management gone wrong**

You spend all night figuring out what wires go where and eventually find that one of your subsystems was unplugged. Unfortunately, that was all of the time you had allocated to working on the project. This simple problem could have been solved in minutes with proper cable management

## Planning

### Understanding the Electronics Used in Your System
The first step to planning is to understand how all of your hardware connects in your system. One way to do this effectively is to create a wiring block diagram. The main purpose of this diagram is to understand all of the interfaces on your electronics and how they connect to different subsystems. Don't worry about the details of the signals of the wires just yet. Just try and get a high level picture of your system. Figure 2 shows a typical wiring block diagram. Notice how much of the detail is left out and cables are simply represented by one line. This abstraction is meant to highlight the main purpose of representing system connectivity. This diagram can also be used later as a guide for someone who is less familiar with the electrical system.

If this is your first time creating a wiring diagram for your system you will probably already notice problems with your system outside of cabling! For example, you might notice that you have 5 auxiliary devices that you plan to connect to your main computer, but it only supports 3. These are huge benefits that only come from effective planning. You may also find that some of the connections in your system are undefined. That's okay! Define them if possible and if not, make sure that you have a plan for when they do become defined.

Note: in systems engineering you will be required to create a cyberphysical architecture for your system. This diagram may be similar, but they are still fundamentally different. Notice how here we are explicitly differentiating cables, and boards. We get more information from a cable management perspective this way. For example notice how, in **Figure 2**, cable 3 only has one plug. This tells us that Sensor 1 has an unremovable cable attached to it.

![Wiring Managment](assets/CableManagement-b3a03.png)

**Figure 2: Example wiring block diagram of 2 circuit card assembly boards and 1 sensor connected by three cables. Here the JX labels stand for the jacks on the different circuit boards and the PX stand for the plugs of the cables.**

### Understanding the Connections Used in Your System
Now that you have outlined all of the different boards and connections it is time to go one level deeper and start defining each cable. If you are using mostly commercial off-the-shelf (COTS) boards and sensors it is possible that these cables are already defined. If that is the case simply make sure that you understand all of the wiring and signals from each board.

If you are defining your own connections it is time to make some cable wiring diagrams. You can imagine these diagrams as a zoomed in detailed picture of each of the cables in your wiring block diagram. Figure 3 shows an example cable wiring diagram. It is not necessary to get as detailed as Figure 3. In fact, I've used excel to create very effective wiring diagrams before. The most important objective of these diagrams is to understand the signals and wires on each cable. One more crucial detail highlighted in these diagrams is the length of each cable. Make sure you have a rough idea of how long each cable will be, keyword: rough. You may not be able to determine the exact length of your cable, but going through this exercise will make you start thinking about cable routing. At this step it is necessary to sit down with the mechanical designer of your system and discuss how your cables are physically getting from A to B.

![Example Cable Wiring Diagram for a USB Cable](assets/CableManagement-654ae.png)

**Figure 3: Example Cable wiring diagram for a USB cable**

### Final Notes on Planning
While this is the most important step of cable management, it is also necessary to mention that there is a balance between planning and pragmatism. I guarantee that you will not have time to make detailed diagrams like Figure 3 or even diagrams for every cable that you will be using in your system. Instead, balance planning with common sense. Don't make diagrams for USB cables, or other cables that are already formally defined. At the bare minimum, create a wiring block diagram and basic excel cabling diagram for the custom cables in you system.

## Cable Creation
After following through with the planning method defined above you should know all of the cables in your system: what subsystems they go to, what signals they carry, and roughly how they will be routed. Next, it is time to explicitly define the components used in your cables as well as the wires used. This step can be seen as an extension to planning and some of it will be implicitly done when you are creating cable wiring diagrams However, there is enough detail to for cable creation to have an entire section.

## Wiring
### Color Coding
An essential part of making cables is making sure that the wire colors are easily identifiable as certain signals. This will make checking signals and power connections easy. You can use any color code that you want, but a color code for common signals that is given in **Table 1.**

Signal | 	Color
----|------
24V | Red
12V | Green
5V  | Brown
RX  | Orange
TX  | Blue
GND | Black
**Table 1: Example table of wiring colors**

### Choosing a Wire Gauge
In addition to color coding, it is essential that the correct wire gauge be used so that your wires don't melt. I usually follow the American Wire Gauge Chart [here](http://www.powerstream.com/Wire_Size.htm)

However, it is important to keep in mind that these numbers are a ballpark estimate when considering the use case for your project.

### Wire Length
By now it is probably apparent that your initial estimates of cable length were off. If you haven't done this yet, now is the time to do a mock cable routing with the wires you have chosen to figure out the exact cable length to use. Cut to length, measure and update your cabling diagram to reflect the correct length

### Connector Selection
When picking connectors it is essential to consider the signals that the cables as well the role that the cable will be playing in the system. Some good questions to ask are:

> How often will I need to unplug and plug this cable?
>
> How much current can this connector carry?
>
> Are these connectors already available to me in the lab?
>
> What crimps and crimper correspond to my chosen connector?


Depending on the answer you may choose a certain connector series or you may decide to directly solder to the circuit board if you don't foresee yourself unplugging the cable very often (be careful with this decision). For convenience, pack a connector that is commonly available in the lab and sticking to it for all of the cabling during the project.

### A Note on Modularity
A very underrated topic in connector selection is modularity and cable re-usability. This is the reason why USB, and RJ45 cables are so great to use. They are symmetrical and easily swappable. This idea can be extended to any cables that you build. There are times where it may make sense to use over-sized connectors (more pins than wires) to allow for easy cable replacements with other similar but different cables.

#### Crimping
If you do decide to use connectors that require crimps it is important that you know what crimps to use and that they are compatible with some crimper in the lab. Crimps come in all shapes and sizes so for a guide on crimping see your manufacturer's web page.

## Cable Assembly
### Grouping Wires
Once you know what connector, wires, and crimps to use it is time to assemble the cable. For the most part this is straightforward, but there are some tricks that will make cable management later on easier. First, if you have any twisted pairs in your cable or want to group wires together without additional hardware, you can use the drill trick to make nice looking cables like **Figure 4**. A video of the Twisted Pair Drill Trick can be viewed [here](https://www.youtube.com/watch?v=uTJhrTTl-EE)

![Twisted Wires Done Using a Drill](assets/CableManagement-6f4d5.png)

**Figure 4: Twisted wires done using a drill**

Another way to group wires together is to simply use heatshrink tubing ever 6-12 inches in your cable. Finally, If you have expando sleeves available this is another option for grouping.

### Labeling
By this point you should know how to create a cable so that it is electrically sound for installation. However, there is still one important step left. You should always label your cable, at the bare minimum with the cable name and preferably also with the designators that you outline in your wiring block diagram.

### Installation
The final step is installing the cables into your system. This step is extremely easy if you were diligent with the previous two steps. If you weren't, then you may find yourself having to recreate cables or rewire some parts.

If you have a lot of cables in your system, and even if you did a good job at grouping the wires in each cable you may still find yourself with messy cabling. Zip-ties are your friend here. Once you have routed all of the cables in your system, zip-tie the ones that are routed along the same path together.

### References
1. https://s-media-cache-ak0.pinimg.com/564x/b2/db/43/b2db4385f4faa3fdb542d2d277a3278a.jpg
2. https://www.usb3.com/images/567861.jpg
3. http://inlowsound.weebly.com/uploads/1/9/8/5/1985965/5192126_orig.jpgE


/wiki/system-design-development/In-Loop-Testing/
---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2022-04-29 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Hardware-in-Loop and Software-in-Loop Testing
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---


Hardware-in-the-loop (HIL) testing is a test methodology that can be used throughout the development of real-time embedded controllers to reduce development time and improve the effectiveness of testing. As the complexity of electronic control units (ECUs) increases, the number of combinations of tests required to ensure correct functionality and response increases exponentially.

Older testing methodology tended to wait for the controller design to be completed and integrated into the final system before system issues could be identified.

Hardware-in-the-loop testing provides a way of simulating sensors, actuators and mechanical components in a way that connects all the I/O of the ECU being tested, long before the final system is integrated. It does this by using representative real-time responses, electrical stimuli and functional use cases 

The term "software-in-the-loop testing", or SIL testing, is used to describe a test methodology where executable code such as algorithms (or even an entire controller strategy), usually written for a particular mechatronic system, is tested within a modelling environment that can help prove or test the so.

## Need Hardware in Loop and Software in Loop Testing : ADAS usecase

Primitive techniques for collision and lane change avoidance are being replaced with advanced drive assistance systems (ADAS). These new systems introduce new design and test challenges. Modern ADAS architectures combine complex sensing, processing, and algorithmic technologies into the what will ultimately become the guts of autonomous vehicles. As ADASs evolve from simple collision-avoidance systems to fully autonomous vehicles, they demand sensing and computing technologies that are complex. For example, consider the sensing technology on the Tesla Model S, which combines information from eight cameras and 12 ultrasonic sensors as part of its autopilot technology. Many experts are claiming that the 2018 Audi A8 is the first car to hit Level 3 autonomous operation. At speeds up to 37 mph, the A8 will start, accelerate, steer and brake on roads with a central barrier without help from the driver. The car contains 12 ultrasonic sensors on the front, sides, and rear, four 360� cameras on the front, rear and side mirrors, a long-range radar and laser scanner at the front, a front camera at the top of the windscreen and a mid-range radar at each corner. 
As a result, autonomous vehicles employ significantly more complex processing technologies and generate more data than ever before. As an example, the Tesla Model S contains 62 microprocessors � more the three times the number of moving parts in the vehicle. In addition, Intel recently estimated that tomorrow�s autonomous vehicles will produce four terabytes of data every second. Making sense of all this data is a significant challenge � and engineers have experimented with everything from simple PID loops to deep neural networks to improve autonomous navigation. 
This is the reason why we need Hardware-in-Loop and Software-in-Loop Testing and the importance of this has increased its importance more


Increasingly complex ADAS technology makes a lot of demands on test regimes. In particular, hardware-in-the-loop test methods, long used in developing engine and vehicle dynamics controllers, are being adapted to ADAS setups. 
Hardware-in-the-loop testing offers a few important advantages over live road testing. Most notably, a HIL setup can help you understand how hardware will perform in the real world, without having to take it outdoors.
- By testing in the lab rather than on public highways, AV manufacturers can:
- Accelerate testing and reduce costs - millions of miles of testing can be done in a shorter space of time in a lab
- Avoid complex laws and regulations around public autonomous vehicle testing
- Assess different hardware and sensor configurations earlier in the development process
- Build a repeatable test process
- Test weather or time-dependent edge cases at any time


## Hardware-in-Loop

HIL test systems use mathematical representations of dynamic systems which react with the embedded systems being tested. An HIL simulation may emulate the electrical behavior of sensors and actuators and send these signals to the vehicle electronic control module (ECM). Likewise, an ADAS HIL simulation may use real sensors to stimulate an emulation of the ECM and generate actuator control signals.
For example, a HIL simulation platform for the development of automotive anti-lock braking systems may have mathematical representations for subsystems that include the vehicle dynamics such as suspension, wheels, tires, roll, pitch and yaw; the dynamics of the brake system�s hydraulic components; and road qualities.
A point to note is that the in-the-loop notation means the reaction of the unit being tested influences the simulation. HIL methods verify and validate the software in the physical target ECUs. So HIL test benches must provide the ECU with realistic real-time stimuli and simulated loads. Multiple ECUs get tested in network HIL setups that wring out the bus systems, sensors, and actuators.
More recently, vehicle-in-the-loop methods use a physical car to replace vehicle simulations and most of the virtual ECUs. This approach comes in handy for testing safety critical ADAS functions by, say, crashing into virtual obstacles while operating the real vehicle.
The way an ADAS typically works is that it first processes the raw sensor data via feature and object recognition algorithms. The result is a data set that resembles a grid-based map of the environment or a list of recognized objects (such as trees, pedestrians, vehicles, and so forth). A situation analysis algorithm combines this processed data and estimates the current traffic situation. This situational analysis gets forwarded to the ADAS application. The application finally decides on actions to take such as slowing down or triggering an emergency brake.
Many HIL tests of ADAS functions boil down to sending simulated object lists to the UUT (unit under test). Depending on the sensor systems involved, the object lists may include information about object kinematics as well as whether the object is a pedestrian, vehicle, or something else.

Besides actual sensor data, the ADAS application needs supplementary vehicle data from other ECUs. This information usually passes to the UUT via the CAN bus and generally includes details like gear rate, acceleration, engine speed, steering angle, or GPS data. All ADAS require at least some of this information to check the plausibility of the situation analysis. The ADAS makes safety-critical interventions only if all this information is consistent.
When testing ADAS functions within HIL regimes, there can be a variety of UUTs and interfaces. For example, a single ECU might be tested while it runs core emergency functions such as a radar-based emergency brake-assist. Or a dedicated ADAS ECU might receive sensor information from other control units. Depending on the setup, multiple real ECUs may be part of the test bench and connected via automotive network systems like CAN, FlexRay, or Automotive Ethernet.
In an HIL test bench there can be several points at which virtual data is added. One option is feeding simulated data to physical sensors. Of course, real ADAS have multiple sensors so this strategy entails simultaneous generation of multiple signals. In the case of cameras, for example, engineers might show each camera a sequence of images of actual scenery via a screen or projector. For radar returns, the HIL system needn�t generate the radar output, just simulated echoes coming back.
One advantage of using physical sensors in HIL testing is that there�s no need to modify the UUT for testing purposes � the simulated signals come via the same physical interfaces as found in real vehicles. Among the biggest challenges is assuring the quality of the injected signals. For example, images projected on a screen might not represent the dynamic range a camera would see in real life. The classic example is that of a car driving straight into a blazing sunset, then descending a hill and plunged into dusk. All in all, the injection of data into a physical sensor involves few modifications of the UUT, but the accurate representation of scenarios can be quite demanding and not currently possible for all sensors. 
A typical test configuration for an ADAS HIL simulation depicts the different methods of simulating sensor data. Test equipment can inject digital data into the sensor channel to simulate sensor inputs. Alternatively, test equipment can develop sensor inputs in the form of images for cameras or echo patterns for radars to simulate actual sensed data. There are advantages and drawbacks to each approach.

There is also need to look into the latency issue of the HIL and Data communication. Having low latency HIL system is important especially in ADAS as car safety is very critical and speed testing is important.   More details of this particular topic can be found at [here](https://www.spirent.com/blogs/how-to-create-a-realistic-low-latency-hil-environment-for-autonomous-vehicle-testing/)

## Platform based Hardware-in-Loop

Manufacturers such as National Instruments offer a platform-based approach to HIL testing. Key features of NI�s HIL test approach include a tight synchronization among numerous PXI instruments to enable simulations of driving scenarios and sensors. Particularly important in ADAS and autonomous driving applications is NI�s FPGA technology. FPGA technology enables engineers to design HIL test systems with extremely fast loop rates for quick decision making.
One recent example of an HIL test system using the platform-based approach to sensor fusion testing was demonstrated by a consortium called [ADAS Innovations] (https://www.spirent.com/blogs/how-to-create-a-realistic-low-latency-hil-environment-for-autonomous-vehicle-testing/) in Test. This group is a collaboration between NI alliance partners S.E.T., Konrad Technologies, measX, and S.E.A. At the recent NIWeek conference, the group demonstrated an ADAS test setup which can synchronously simulate radar, lidar, communications, and camera signals for an ADAS sensor. In one case, the setup was able to simulate a virtual test drive using IPG CarMaker and NI VeriStand software.
Modular systems such as PXI simulate many vehicular physical signals � effectively recreating the physical environment of the ADAS sensor or ECU. Synchronization is a critical requirement of the PXI modules � because all these signals must be precisely simulated in parallel. The ECU needs the radar, V2X and camera signal to arrive simultaneously if it is to process and understand the scenario and act accordingly.

HIL test techniques based on a platform let engineers simulate a virtually unlimited duration of �driving time.� The lengthy test time provides more opportunities for finding problems and lets engineers better understand how the embedded software performs in a wide range of situations.
As a result of simulating driving conditions in the lab, engineers can identify critical design flaws much earlier in the design process.

Other than Hardware-in-Loop Testing, there are few other types of testing techniques as well. I will brief them below: 

### Model-in-the-Loop (MIL) simulation or Model-Based Testing: 

Hardware-in-the-loop (HIL) simulations have been terms of motion, force, strain, etc. used successfully for a number of years. These have evolved from pure computer simulations of a con in which the test specimen is part real and part troller and plant, such that (usually) the controller is run in real-time and used to control the real plant, making use of the correct actuators and sensors. The model-in-the-loop concept in that the real-time simulation is part of the mechanical system, not of a controller. Only the key component with unknown dynamics real-time simulation in a MiL system interfaces with need be physically tested, reducing the cost and the rest of the system using actuators and sensors complexity of the physical test apparatus. that would not otherwise be present;

Model-in-Loop testing, also called MiL testing or model testing, is the testing of single or integrated modules in a [model-based development environment] (https://piketec.com/tpt/test-environments/simulink-targetlink-testing/), such as MATLAB Simulink from Mathworks or [ASCET from ETAS] (https://www.etas.com/en/company/ascet-model-based-development-of-application-software-mastering-in-real-time.php/).

When developing a system,  a model-based design (MBD) allows early identification and correction of errors and bugs because the models can be simulated easily in a very early design phase. Due to its early development stage, testing iterations (�loops�) can be finalized much faster than in longer (later) development stages. This approach is therefore a cost-efficient way to reduce development time significantly. Additionally, the graphical approach of programming is intuitive and loved by engineers.
Model-in-Loop testing is also related to unit testing, while the units under test are �model units�.
The MBD approach, and thus MiL testing, is commonly used in the automotive industry. Development processes used in MBD can be fully compliant with safety norms such as e.g. ISO26262 or IEC61508. These norms usually require MiL testing in early design phases.
The test level following MiL testing is Software-in-Loop testing (SiL testing) using software that is generated, oftentimes automatically, directly from the models, or written manually in C-code. For this test level, auto code generation is used. Common C-code generators for Simulink models are dSpace TargetLink or Mathworks Embedded Coder. 

See how Mathworks products facilitate model-based testing: https://www.mathworks.com/discovery/model-based-testing.html

### Software-in-the-Loop (SIL) simulation

Once your model has been verified in MIL simulation, the next stage is Software-in-Loop(SIL), where you generate code only from the controller model and replace the controller block with this code. Then run the simulation with the Controller block (which contains the C code) and the Plant, which is still the software model (similar to the first step). This step will give you an idea of whether your control logic i.e., the Controller model can be converted to code and if it is hardware implementable. You should log the input-output here and match it with what you have achieved in the previous step. If you experience a huge difference in them, you may have to go back to MIL and make necessary changes and then repeat steps 1 and 2. If you have a model which has been tested for SIL and the performance is acceptable you can move forward to the next step. Once both MiL and SiL testing phases are done, the individual test results can be tested �back-to-back�. Back-to-back testing is used for the comparison of test results between MiL and SiL testing.
See how to run SIL simulations using Embedded Coder: <https://www.mathworks.com/help/ecoder/software-in-the-loop-sil-simulation.html>

### Processor-in-the-Loop (PIL) or FPGA-in-the-Loop (FIL) simulation

The next step is Processor-in-the-Loop (PIL) testing. In this step, we will put the Controller model onto an embedded processor and run a closed-loop simulation with the simulated Plant. So, we will replace the Controller Subsystem with a PIL block which will have the Controller code running on the hardware. This step will help you identify if the processor is capable of running the developed Control logic. If there are glitches, then go back to your code, SIL or MIL, and rectify them.
See how to run PIL simulations using Embedded Coder: 

https://www.mathworks.com/help/ecoder/processor-in-the-loop.html

In the case of running the simulation on an FPGA instead of an embedded processor, the simulation is called FPGA-in-the-Loop (FIL). See how to run FIL simulations using HDL Verifier

https://www.mathworks.com/help/hdlverifier/ug/fpga-in-the-loop-fil-simulation.html



## In-Loop Testing and V-Loop Model


The V model is frequently used in the automotive industry to depict the relationships of vehicle-in-the-loop, hardware-in-the-loop, software-in-the-loop, and model-in-the-loop methods. Stages depicted on the left have a corresponding testing counterpart on the right. MiL settings test a model of the functions to be developed. MiL is applied in early stages to verify basic decisions about architecture and design. SiL setups test the functions of the program code complied for the target ECU but without including real hardware. HiL methods verify and validate the software in the physical target ECUs. ViL tests replace the vehicle simulation and most of the virtual ECUs with a real vehicle. The simulated parts of the environment are injected into the vehicle sensors or ECUs.

Advanced systems like autonomous vehicles are quickly rewriting the rules for how test and measurement equipment vendors must design instrumentation. In the past, test software was merely a mechanism to communicate a measurement result or measure a voltage. Going forward, test software is the technology that allows engineers to construct increasingly complex measurement systems capable of characterizing everything from the simplest RF component to comprehensive autonomous vehicle simulation. As a result, software remains a key investment area for test equipment vendors � and the ability to differentiate products with software will ultimately define the winners and losers in the industry.

## Summary

The given article describes in detail what is in-loop testing with specific focus on ADAS as usecase. The importance and relevance of in-loop testing with V-model in systems engineering is also highlighted in the given model. 


## See Also:

[https://github.com/shahrathin/roboticsknowledgebase.github.io/blob/master/wiki/system-design-development/subsystem-interface-modeling.md](Subsystem Interface Modeling)


## Further Reading

[https://www.guru99.com/loop-testing.html](Loop Testing)



## References

[https://en.wikipedia.org/wiki/Hardware-in-the-loop_simulation](Hardware-in-the-loop simulation)


/wiki/system-design-development/mechanical-design/
---
date: 2017-08-15
title: Mechanical Design
---
This section will provide you some of the design criteria’s and consideration you should keep in mind when developing a mechanical system:
- If possible, develop a mathematical model before ordering parts and prototyping. This will help you determine whether the values you have in mind are suitable enough or not.
- Develop a detailed CAD model of your product early. A visual representation helps in determining shortcomings of the model and allows optimization.
- If your system will experience load, it is always helpful to do a quick simulation to determine the stresses in the system. This helps you determine whether your system will able to sustain the loads applied on it and help you optimize the design. It also helps you to remove redundancies which are taking up space or increasing the weight of the system.
- Some of the mechanism that are commonly used are Lead Screw, Rack and Pinion, One Way Clutch, Belt drive, pulley drive, Gear Reduction for higher torque, solenoid actuation for push-pull, Crank and Slider mechanism and extendable linkages.
- Base your design as much as possible on already available parts instead of engineering yourself. This reduces the human error possible and allows you to concentrate on other aspects of design.
- Always keep a high factor of safety in you design, be it mechanical components, motor selection or sensor accuracy. This helps in reusability of material if you make changes in your design.
- Modular design helps a lot when you encounter failure. If you have a failure in a small part of your system, instead of replacing the whole system and rebuilding it, you can just remove that section. This saves time and money.
- Be ready to reiterate the design. It happens that you build a product and it fails in some aspects. You then try to make that design work by making modification to it over and over again. At one point, the design will get saturated and you will never achieve the expected output. It is important to decide early in the development stage that the design will perform as expected or not. If not, reiterate the design and develop the system from a different perspective. It is a lot of work in the beginning but will prove helpful in the long run.
- Try developing simple systems and mechanical designs for your product. The more complex the design gets the chances of failure increases. Number of components and inter-dependencies between them plays an important role in making the design robust. Minimum number of components and as many independent systems as possible help in improving the reliability and robustness of the system.


/wiki/system-design-development/pcb-design/
---
date: 2017-08-15
title: PCB Design Notes
---
## Software Tools
- [Eagle CAD](http://www.cadsoftusa.com/download-eagle/freeware/?language=en) - Freemium CAD software package
  - Free version limited to 4" X 3.2" board sizes w/ two layers
  - [CAD Soft Tutorials](https://www.cadsoftusa.com/training/tutorials/)
  - [Spark Fun Tutorials](https://www.sparkfun.com/tutorials/108)
- [GerbMerge](http://ruggedcircuits.com/gerbmerge) - Free Python program for panellizing generated PCBs
  - [Ubuntu install instructions](http://eliaselectronics.com/installing-gerbmerge-on-ubuntu/)
  - [GerbMerge Tutorial](http://pauljmac.com/projects/index.php/Gerbmerge)
- [Atlium Circuitmaker](https://circuitmaker.com/) - Alternative to Eagle CAD currently in Open Beta

## Vendors
### Search Engines
- [FindChips](http://www.findchips.com/)
- [Octopart](http://octopart.com/)

### Discrete components Suppliers
- [Mouser](http://www.mouser.com/)
- [Digi-Key](http://www.digikey.com/)

### DIY breakouts and other components
- [Adafruit Industries](http://www.adafruit.com/)
- [Sparkfun](https://www.sparkfun.com/)
- [Pololu](http://www.pololu.com/)

### PCB Fabrication Houses
- [OSH Park](http://oshpark.com/): FOSS Hardware community service
- [Advanced Circuits](http://www.4pcb.com/): Recommended by CMU Gadgetry
  - [33 Each Service](http://www.4pcb.com/33-each-pcbs/): Great for small batches of panellized boards


## Design Guidelines
### General Guidelines
#### Schematics
- Before designing a schematic have a clear idea of what are your system requirements. Discuss it thoroughly with your team members.
- You should be aware of the power ratings, current ratings each component requires. This will help in selecting the components.
- Label the schematic such that it makes sense as per your requirement. For example:
  - A connector for 5V relay that will be used for paint system can be named as "5relayPaint."" This will help later on to debug.
- Select the right package that you are going to use for all the components (TH,SMD).
- Ensure proper protection circuitry for all the critical paths and components (as shown in **Figure 2**). For example:
  - Fuses for overcurrent protection, zener­diodes for over voltage protection, capacitors
(electrolytic and non­electrolytic) for noise­reduction.
- It is important to have proper LED indication  on the PCB (as shown in **Figure 1** and **Figure 2**) for testing and accountability. Use LED  indicators to ensure proper functioning of the board.

![LED indication in schematic with noise reduction cap](assets/PCBDesignNotes-8fedc.png)

**Figure 1: LED indication in schematic with noise reduction cap**

![Labeled Fuse and Zener­diode for Protection](assets/PCBDesignNotes-09baf.png)

**Figure 2: Labeled Fuse and Zener­diode for Protection**

- You can add additional comments on schematic as shown in **Figure 3**, this helps to put information related to board  which is otherwise not explicit from the schematic.

![Schematic with comments](assets/PCBDesignNotes-d5ba2.png)

**Figure 3: Schematic with comments**

#### Track Width
The heat generated by the IC and various other components is transferred from the device to the copper layers of the PCB. The ideal thermal design will result in the entire board being the same temperature. The copper thickness, number of layers, continuity of thermal paths, and board area will have a direct impact on the operating temperature of components. Track width is an important factor to maintain thermal stability.
Track width for both and POWER and GROUND lines should be sufficiently thick to be able to carry the estimated current to prevent heat damage.  A web tool to calculate the appropriate width can be found [here](http://circuitcalculator.com/wordpress/2006/01/31/pcb­trace­width­calculator/). Use a 1 oz/ft^2 thickness. This is the thickness value most PCB manufacturer use.
Your board can look like what has been shown below in **Figure 4.** It can have variable track widths depending on the requirements. Also, avoid 90 degree turns while laying down traces.

#### Drill Size
As you get your board design ready, please keep in mind that you want to have some clearance between the hole and the component's terminal lead for the solder to fill up. If your component has a 20mil round lead, don't specify a 20mil hole diameter. In addition to needing a gap, keep in mind that as part of the PCB
manufacturing process, some metal is going to be deposited inside each hole. This reduces the diameter of the hole, making it smaller than it was when it was drilled. As a rule of thumb, add 8­12mils to the nominal round lead diameter, and round up if you need to match the diameter to a particular drill size list. Keep in mind that a bit bigger is better than too small; you don't want to find yourself filing a lead that is too big (you
cannot make the hole bigger, because it would remove the through­hole metal). Double­check and make sure that none of the holes is smaller than 0.015” diameter, and that neither tracks nor clearances between tracks/pads/planes are smaller than 0.006”.

![Variable Track Width](assets/PCBDesignNotes-05e85.png)

**Figure 4: Variable Track Width**

#### Structural Stability
For proper structural support place proper mounting holes on all the sides.
#### Package
Custom package designing can be the most critical part. If there are components for which custom library needs to be generated, few things need to be kept in mind. The package layout as per the supplier datasheet can be in BOTTOM view and not TOP view. It is important to take care of the top/bottom view when you create a layout in PCB designing software. If the package was created in bottom view, and if you place it on the TOP layer of PCB, make sure it is mirrored.  If this issue was not considered, you might
render the PCB board useless if the pins of the component are asymmetrical. A few things to try if you're running out of space
- Use both sides of the board to place components. Some (or all) of the SMD components can go in the bottom layer, while the through­hole ones can be placed in the top layer. Some of the SMD can be mounted on the opposite side of bulky things, like DC­DC converters, to save space (unless these converters need a ground plane directly underneath it).
- Components like through­hole diodes and resistors can be placed vertically instead of horizontally. Though not the best choice, this helps if space is limited. Just make sure to cover the long terminal with some tubing or extra protection when you solder it.
- Try to find a different type of connector that is more compact, like a terminal block, to concentrate all the connections in a single connector (but not a terminal block per se, but something that saves the need for individual fasteners per connector).

#### PCB Manufacturing
The PCB’s documents should include the hardware dimensional drawings, schematic, BOM, layout file, component placement file, assembly drawings and instructions, and Gerber file set. User guides also are useful but aren’t required. The Gerber file set is PCB jargon for the output files of the layout that are used by PCB manufacturers to create the PCB. A complete set of Gerber files includes output files generated from the board layout file:
- Silkscreen top and bottom
- Solder mask top and bottom
- All metal layers
- Paste mask top and bottom
- Assembly drawing top and bottom
- Drill file
- Drill legend
- FAB outline (dimensions, special features)
- Netlist file
- [Video](http://www.youtube.com/watch?v=B_SbQeF83XU) covering how to create gerber files for manufacturing.

#### Minimize manufacturing cost
In general once you create all your gerber files, you can print them on paper before sending them for
manufacturing. This will help to:
- Check whether all your drill holes look alright
- Whether track widths are perfect
- Text is readable
- All your components and their packages are on appropriate layer

#### Other Recommendations
- Check trace widths with [PCB Trace-Width Calculator](http://www.circuitcalculator.com/wordpress/2006/01/31/pcb-trace-width-calculator/)
- Design with 1oz Copper and +25 C max in mind
- Use straight & 45 degree runs, no curvy stuff, no right angles!
- 50mil (.05in, ~1.27mm) pin spacing is reasonable for hand soldering (20mils is way to small)

### Surface Mount Package Selection Guide
#### Discrete Components (RCL)
- 1206 is easiest to solder and takes most board space
- 1210 is slightly wider (usually used for capacitors), and so on
- 0603 is half as big in each dimension (very difficult to hand solder anything this size or smaller)

#### Semiconductors
- Diodes – SMA, SMB, SMC
- SOT223 is a good-sized SMT regulator
- TQFP, SOIC are usually good for ICs
- NO QFN, LC*, or BGA PACKAGES! (very bad for hand assembly)

## EAGLE Board Export Process
### Gerber files
1. Click the CAM Processor button
2. Export Layers
  1. Go to file->open job
  2. Select gerb274x
  3. Hit “Process Job”
3. Export Drill files
  1. Go to file->open job
  2. Select excellon
  3. Hit “Process Job”
4. Verify generated files
  1. [FreeDFM](http://www.freedfm.com/) - Associated with 4pcb
  2. [OSH Park](http://oshpark.com/) - FOSS Hardware community service
###Bill of Materials
1. In Eagle: File -> Export ->
2. Choose 'CSV' for the format, rename and save the file
3. Open the spreadsheet, add Supplier Part #s and links.


PCB Panelization

    Methods of PCB Panelization


/wiki/system-design-development/subsystem-interface-modeling/
---
date: 2017-08-15
title: System-Subsystem-Interface Modeling
---
A useful tool in system engineering is the system model, which can drive development, requirements generation, design, and just about anything else to do with your project. While the WBS, schedule, and other tools are good for their particular area, the system model has the potential to be useful in replacing all your other planning methods, becoming the one-stop source for information about your project.

The system model language SysML was first developed by No Magic - though the framework is somewhat older - for project management and was quickly adapted for use in the Department of Defense, NASA, and other government bodies. The full framework is quite extensive, covering every possible level of system engineering from the highest goal oriented concept to the most basic layers like what data types are used and what interfaces carry them. The system model is a key part of the functionality of Rational Rhapsody, which uses the system model as built to generate code for programs the system runs. The a good set of resources can be found [here](https://www.nomagic.com/support/quick-reference-guides.html). A quick glance makes it obvious that most roboticists doe not need most of the views and charts, but some are useful and we will go over them in the following sections.

## Hierarchy
The most important aspect of system modeling tools is the hierarchy. The highest level is usually the requirement - or even the rationale behind the requirement - and each of these should have a separate diagram which itself is made up of blocks, capabilities, activities, information flows, etc which all expand into further diagrams until you reach the lowest operating level of a system. This way you can go up and down the conceptual flow to find out how every part of your development supports a requirement on the system.

### Operational View
One of the most useful views is the operational. In practice, you will rarely go below level 2, it being the meat of the view type for our purposes. It can generally be combined with the concept of the activity diagram which you can find in the overview of SysML.

#### Level 1
In this level, one can usually put the Use Case. While technically it belongs in the Capability View or All View, since we're otherwise not using CVs (unless you start modeling at the very beginning and use the CV for requirements). In the Use Case, put the Actors (the user, obviously, and the highest level subsystems you'd normally see in a WBS), and then what information flows between them. This would be the highest level information packet containing all other information flow we see on the level 2 Activity Diagrams.
![SIDD colorless OV 1](assets/SubsystemInterfaceModeling-8748c.png)

#### Level 2
This is the meat of the Interface Design Description, where you can flow out all the things the system might do. Rather than go into a lot of operational or capability viewpoints, go straight into activity diagrams since the system is relatively small. Each box at the first indenture of level 2.

>Use "indentures" of activity diagrams since the other "levels" of OV are technically something else, signified by letters like OV-2b or OV-2c) is a high level activity (see below for the highest level view of MRSD 2015 "Dock In Piece" project).

![SIDD colorless OV 2 Dock Quadcopter](assets/SubsystemInterfaceModeling-fb765.png)

As you can see from this chart, the highest level activities of docking the quadcopter are things like 'take off' and 'rendezvous with docking face', which have lines between to signify the temporal flow, and labels to signify what information needs to pass in order to make it happen. Most important are the labels which cross swimlanes, because that's information that will have to be packaged and sent from one subsystem to another. At this level the swimlanes are the actors and high level subsystems, but if we go a level down, we see subsystems within subsystems (the system-of-systems concept made manifest).

![SIDD-colorless-OV-2b Rendezvous with Docking Face](assets/SubsystemInterfaceModeling-4e816.png)

The first three swimlanes from the left are subsystems of the quadcopter while the final is crossing to/from the Palantir.

![SIDD-colorless-OV-2a Determine Docking Possibility](assets/SubsystemInterfaceModeling-8e94f.png)

Here all four swimlanes are different subsystems (though the Palantir is technically part of the dock, it is quite separate physically and informationally), but support the higher level activity of docking)


#### Level 3
OV-2c is where intrasystem activities show up in the architecture. This indenture shows information flowing and activities that occur entirely on one system, sometimes with a single information flow leaving the internal swimlanes to some external box or all flowing to a single box which has the same name as a box in an indenture above, signifying what information and actions flow inside a system to make that one action occur - similar how how an OV-2b is often the expanded version of a single box in the OV-2a indenture. We cancelled all of our OV-2c diagrams early on and by the time the new system was finalized, we didn't have time to model, but this is what it looked like when we had the quadcopter doing localization with an onboard camera.

![SIDD-colorless-CANCELED OV-2c Localize Quadcopter](assets/SubsystemInterfaceModeling-73d8e.png)

All swimlanes are of systems internal to the quadcopter subsystem, and eventually flow out to it keeping a fixed hover point.

## Capability View
If you were modeling from the very beginning, capability and to some extent activity view would be very useful for requirements traceability. The capability view shows the very highest level capability and then breaks that down through indentures of what other capabilities it would need and what activities support that. You can see it as being sort of the structure behind the activity diagrams with the various blocks and diagrams all being activities but rather than relating to each other through time and information flow they relate by what capability they support and what higher level activity they feed. Since the MRSD project is somewhat rushed, it is unlikely you'll use either view, but it is valuable to think about. For instance, each functional requirement should be a capability, what the system can do. From there, you can flow down activities it would have to perform to have those capabilities and then activity diagrams to show the flow of information. CV and AV diagrams are very useful in keeping the team focused on requirements and doing what needs to be done.

### Data and Information View
You can always dive straight into DIV-2 rather than DIV-1, but if you start early with modeling a DIV-1 is useful. We'll cover the second level here. Each line in the activity diagram had a name (or should have). That was a data flow, and these data flows naturally have information. In the DIV-2, you show what this information is.

![SIDD-colorless-DIV-2 DockMotionDetails](assets/SubsystemInterfaceModeling-7097d.png)

It may seem trivial at first, but these views show not just information, but their type, names, and what larger information boxes they flow into. This can make integration much easier as everyone knows what every subsystem needs from every other subsystem and if the naming conventions are kept to, there is no confusion in how to get that information.

The flow up of these arrows show that the lower boxes become part of larger information packages sent along data flows at higher levels. In the above case, the multiple arrow means that more than one of these dockMotionFrequencyDomain objects may be included in the DockMotionDetail object/package. Interface design is much easier when the designer knows what information will flow through. Ideally one defines the name, type, and flow of every variable that crosses a boundary, subsystem to subsystem or even internally between functions.

## Sequence Diagram
There is software that will even turn these diagrams into functional code. In industry, it's becoming a practice in some companies to complete these models and then build the system they show. While it doesn't mean you can go without testing, it does mean that with a full system model, you can be confident that you're satisfying requirements and minimizing the difficulty of integrating many subsystems into a system of systems. Most of these rely on sequence diagrams, which are the activity diagrams given a strict temporal progression. This system sends this data to this system which does this and outputs something else to another system or the user.

## Summary
Some brief thoughts on the other views
- All View
  - Useful in a very large framework to track the other views, mission, vision, and other very high abstraction concepts. Of little use until you can call yourself an enterprise rather than a business.
- OV - 1
- This is actually extremely useful in helping to identify stakeholders and how they affect business and development. Highly recommended for any development project with more than one working group contributing to the life cycle within a business.
- OV - 2 (the first kind)
  - If you've done an OV-1, you'll want this to clarify all the things sketched out one level above.
- OV - 2 (the second kind)
  - Covered.
- OV - 3
  - When you're big enough for an AV, you'll need this one. Also useful if you have multiple working groups and want extremely clear communication of who is doing what.
- OV - 4 (first)
  - Within working groups, invaluable to making sure that everyone knows their role, who to ask for help, and who knows how to do what. Doubles as a handy lookup chart when its time to do performance reviews.
- OV - 4 (second)
  - Same as above, but now you've turned it into an Organization chart. Large companies would find it helpful. Startups, probably not.
- OV - 5a
  - If you really need to show how everything is done (maybe you want to be ISO certified), then this is a great way to do it.
- OV - 5b
  - One level down, shows how you do all the things you related in 5a. Again, if you have a process and need to codify it for ISO or the like, this is where it happens.
- OV - 6a
  Only looks useful if you have an organization contracting with the government (or if you are the government).
- OV - 6b
  - High level schedule.
- OV - 6c
  - Digging deeper into what makes one item in the schedule move on to the next.
- CV - 1
  - Capability view so high that it's looking at organizational capability rather than product capability. Useful if your entire company is being represented and run through modeling.
- CV - 2 through 4
  - Digging deeper. It is also recommended using CVs for product capabilities as well as enterprise (and in fact, that's more useful at least at the start).
- CV - 5 through 7
  - Relating capabilities to other views, so that you can see how they support and relate to your activities, services, and organizational development.
- PV - 1 (first)
  - Generalizes how your organizations' projects will flow. Project View isn't useful until you're enterprise level and have many projects running in tandem and need a metric by which to compare them
- PV - 1 (second)
  - More of the same.
- PV - 2
  - Shows how each project in the enterprise is progressing.
- PV - 3
  - What internal groups are assigned to what projects.
- DIV - 1
  - Defining what information you work with. Useful if you have some types of information (like standardized lists or blocks) that get used frequently.
- DIV - 2
  - Lower level, showing exact fields and types like String, Int and other objects.
- DIV - 3
  - Structure of the data. Used if you're building something with a really serious database.
- StdV - 1
  - What standards apply to each system and subsystem
- StdV - 2
  - How you think the standards will change in the operational lifetime of the system.
- SV - 1
  - This is what we would have used if we hadn't simplified things a lot by just using internal capabilities and activities. The high level functionality of an individual system.
- SV - 2
  - Digging down, how each subsystem communicates with other subsystems.
- SV - 3
  - Shared resources and which subsystems need them.
- SV - 4a
  - What you need to do to make each part of the system work (ie. activities carried out to support the development of each subsystem).
- SV - 4b
  - Similar to a cyberphysical architecture, relating your functions to physical subsystems and how the information flows between them.
- SV - 5
  - Relates SV-4 to OV-5, so pretty much more of the same as OV-5 in that if you need a really solid definition of your process you can use this. The idea being that each activity necessary to implement a system should have an operational action associated with it. Only of use to a company that is not at all interested in being agile but instead needs to document everything they do.
- SV - 6
  - For proving that yes, you did do something and so it was someone else's fault when it went wrong.
- SV - 7 (first)
  - Skip SV 5 and 6, SV 7 is what you want for tracking things. It's really handy for test plans and making sure you've met your performance goals. Defines every measure of effectiveness, what kind of measure it is, and the threshold.
- SV - 7 (second)
  - The results from testing measurements codified in the other SV - 7.
- SV - 8
  - Shows a brief description of each version of a system developed. One thing this structure doesn't replace is configuration control and you're probably better off either linking this to your control somehow or just not doing it.
- SV - 9
  - Forecasts of technology and skills so you can get ready to implement capabilities you can't implement now. Good for keeping track of developing technology so that ideas for improvement don't get lost.
- SV - 10a
  - List of constraints on a system. Sort of all the requirements you're not really supposed to have because they implement a negative (system shall not do this).
- SV - 10b
  - A copy of OV-6b but specific to a system.
- SV - 10c
  - Same as above only for OV-6c. If you haven't got a lot of systems, don't bother with OV-6, just do these.

Some more high level information can be found [here](http://www.omgwiki.org/UPDMAlpha/lib/exe/fetch.php?media=doc:overview_of_updm_for_systems_engineers_dodaf_and_omg_graham_bleakley_21_march_2013.pdf). IBM is the publisher of Rational Rhapsody, one of the most popular pieces of modeling software which allows for these diagrams to be turned into code. The other big name in system modeling is No Magic, linked at the start of the page. There is also a lot more to SysML, including the modeling of constraints, personnel, equipment, and everything else that goes into the development and running of a system. A full system model is an enormously useful tool which incorporates all the best aspects of a WBS, schedule, design document, budget, equipment list, styleguide, and requirements spec. When combined with a tool like DOORS (a requirements management database), a system model can be the central repository and generator of all project planning data.
