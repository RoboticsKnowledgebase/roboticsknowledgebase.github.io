---
title: Linear Actuator Resources and Quick Reference
published: true
---
Linear movement is necessary for many types of robotics and automation, from simple 3D Printers with belts and threaded rod to complex high precision stages for Industrial Automation equipment, there always is a need to move something in a straight line.

This article will be a high level overview of linear actuator stage components and will cover basic types, terms, and resources for Electric linear actuation. We will cover Linear Rail/Guides, Belt Drives, Screw Based Systems, Rack and Pinions, Linear Motors (Synchronous, Induction, and a short mention of Piezoelectric), and indexing a linear stage. 

We will also have a section with common vendors of linear motion equipment and other useful articles for producing linear motion systems.

## Linear Rails and Guideways
When you want to produce linear motion, you first must constrain the movement to only one degree of freedom along the direction of you wish to go. This is usually accomplished using linear guides or linear bearings/bushings if you cannot otherwise constrain the actuator.

Linear bearing/bushing systems commonly consist of a bearing/bushing and a cylindrical rod of some sort with the bearing and rod having matched inner and outer diameters respectively within a certain tolerance. If it is a bearing system, there will be recirculating ball or cylindrical bearings rolling against the direction of motion of the bearing. If it is a bushing, it will usually be made of a material with good tribological (wear resistance) properties, such as oil infused bronze or tribologically optimized plastics. 

> A useful resource if you need a bushing that has good wear resistance, but you cannot find the right form factor, is Igus. They provide tribologically optimized Materials Extrusion printer filament for printing custom bushings (See Further Reading.)

The downside to Linear Rails of the bearing/bushing type is that they are usually unconstrained in rotation so you must constrain rotation in another way (eg. a leadscrew.)

Linear Guideways on the other hand, constrain motion in rotation also. These also usually use recirculating bearings to guide motion but the profile of the rod is non-circular. The different types of bearing configurations such as roller bearing, two point contact, or gothic arch affect the load rating in different directions. 

> One useful tip while designing linear guideway platforms with multiple parallel guideways (ex. a XY gantry system) is to machine extremely small alignment shelves for the linear guideways to be referenced against. This will ensure that as long as the machining is done parallel to the machines axis and in one operation that the shelves and thus the guideways will be as parallel as the machine that produced the shelves.

Linear guideways can provide extremely precise linear motion but are usually prices higher in comparison to circular bearings/bushings. Many relatively cheap chinese knockoffs of brands such as Hiwin are available on Ebay. Common problems with these models can include large amounts of slop in the carriage fit and grooves created in your rails over time as cheap rails may be softer than the hardened steel ball bearings. 

One thing to consider while designing a system using either rails or guides with bearings is the preload applied by the manufacturer. When a bearing is preloaded, there is an extremely small interference fit between the ball bearings and the gap between the bearing surfaces in the carriage and on the guide/rail. this ensures there is no slop between the rail and the carriage but increases both price and rolling resistance.

> Other solutions that exist include roller bearing carriages that ride on roller bearing wheels sometimes in v-slot extrusion and air bearings/hydrostatic bearings which ride on a small layer of air/oil respectively for a extremely smooth, low friction ride. There are also linear motion elements called ball splines which constrain linear motion, but can also be used to transmit rotational motion to the carriage.

## Belt Drive systems
A belt drive is, at its heart, a conveyor belt with the carriage attached to the conveyor. In the simplest setup, a Timing Belt is driven by rotary motor with the belt looping around an idler at the other end of travel. The load is then attached to fixed point on the belt. As long as the belt is straight going in the direction of travel, the load will move the diameter of the drive wheel without the teeth. The large advantage to belt drive in prototyping systems is a much lower cost for large systems compared to all other options in addition to relatively high speeds and accelerations. The disadvantage is relatively low load capacity for actuation force.

Important considerations for belt drive systems include belt material, tooth profile, and proper belt tensioning. 

The material of a belt determines how much stretch a belt will have which is directly related to the hysteresis in a belt system, where a belt acts like a spring until the drive motor has moved enough to overcome the stretch in the belt. This also affects how often the belt will have to be tensioned. Belts commonly come with kevlar, carbon fiber, or steel cores to prevent stretching.

It is extremely important to get matching tooth profiles for your drive wheel and your belt. The different profile types are designed for different types of load and have different pitches (teeth/unit length) and profile designs. If you get a belt with the same pitch but a different profile, they may not fit together correctly, causing the belt to slip or move a small amount more or less than desired while under the design loads. Common timing belt profiles include GT2 and MXL.

Correct tension is critical to belt drive operation. Remember to include some method to adjust belt tension if your design. As belts wear, they stretch and thus loose tension. If there is not enough tension, the belt can slip on the teeth producing "lost motion." Extremely over tensioned belts are harmful also, they will unnecessarily load your belt , drive wheel, and motor shaft, causing premature wear and failure.

> Other belt configurations for XY Gantries exist other than a simple cartisan configuration. Other systems include H-Bot and [CoreXY](http://corexy.com/) which trade mechanical advantage for longer belt runs and thus more stretch and hysteresis. See Further Reading for more.

## Screw Based Systems
Screw based systems act upon the principle of a simple machine screw. Some sort of nut and screw are used with one being constrained to not rotate while the other rotates to drive the linear motion. Conventionally it is the screw that rotated, but in systems where multiple systems need to more on the same axis and may interfere with each other, you can rotate the nut and hold the screw to fit multiple actuators on the same screw.

Any screw system has a couple of common parameters. These are pitch, lead, starts, and diameter. 

Pitch is how many threads per linear distance unit exist on a screw. 

In contrast, lead is how much distance a nut will travel with one revolution of the screw. 

The relationship of these two quantities is known as starts, and what it actually represents are how many actual thread "lines" exist on a screw shaft, this value is similar to the number of flutes on a machine tool. 

The Lead, which is the important value when calibrating a system, is the pitch times the number of starts.

The two types of screw based systems in linear motion are leadscrews and ballscrews. 

A leadscrew is equivalent to a bushing in a linear rail system. The threads rely on low friction interfaces to slide relative to each other to produce linear motion. In practice, this is achieved by polishing the surface of the lead screw or teflon coating the inside surfaces of the threads. When a leadscrew is running for a long time or at high speeds it will generate a lot of heat which can cause differential expansion in the materials and cause a leadscrew to fail prematurely. Generally, depending on the lead, a leadscrew will me relatively slower and have less actuating force than other common varieties of linear motion.

In contrast, a ballscrew would be the "bearing" to the leadscrew's "bushing". A ballscrew uses recirculating bearing balls just like the linear rails and guides. This greatly lowers friction in the system and thus increases maximum speeds and decreases heat generation. This negates most of the disadvantage to leadscrew based systems, but comes at both a monetary cost as well as an engineering cost. The monetary cost for ballscrews is much higher than leadscrews due to the precision of manufacturing required. 

There are two types of ballscrews available at different price points, rolled ballscrews and ground ballscrews. Rolled ballscrews are formed by taking a piece of round material and rolling it between two forming dies which form the ball bearing profile for the threads. Ground ballscrews are precision ground by CNC machines. In general, rolled ballscrews will have looser tolerance and more variation at a lower cost.

The engineering cost is that another major difference is that leadscrews are relatively non-backdrivable depending on the lead angle which is the angle of the thread helix. This is desirable in some applications such as heavy Z-Axis on CNC machines. If power is disengaged from a leadscrew based actuator, friction will keep a heavy load from falling, whereas with a much lower friction system in the ballscrew, the axis will fall straight down if there is not a electronic or mechanical brake installed which stops it from turning while motor power is off.

As a screw system, if not preloaded, ballscrew and leadscrew will both have backlash, causing lost motion. This can be combated for ballscrews in the same way as linear bearings, but leadscrews, unlike bushings, and also reduce backlash by using what is called an Anti-Backlash nut. This consists of two lead nuts connected by a spring that is slightly compressed when the lead nut is screwed onto the lead screw. One lead nut will be pressed against the top of the threads while the other will be pressed against the bottom, preventing backlash. This does increase friction and thus heat generation.

Ballscrews and leadscrews also act as gearing based on the lead. A lower lead will provide more linear force for the same torque. They also are extremely rigid systems and commonly show up in high force, high precision applications such as manufacturing. 

## Rack and Pinion Systems
A Rack and Pinion is essentially a gear with a circumference of your desired travel that has been unrolled.
As with traditional gears, rack and pinions can have straight cut or helical gear teeth and can deliver high forces for long travel ranges at a reasonable cost. Like a belt, a Rack and Pinion can maintain relatively high speeds, but unlike a belt, a rack and pinion can drive heavy loads without needing tensioning. 

The actuation force is higher for lower pinion diameters, but your achievable speeds will go down as the circumference decreases. Circumference will also determine the resolution with indexed motors with a higher resolution for smaller circumferences.

The major disadvantage to rack and pinion systems is that, as with gear systems, backlash will cause lost motion when switching travel directions. Unlike a leadscrew, removing backlash on a rack and pinion is not cheap. Solutions include having two identical pinions slightly next to each other slightly offset in angle. A torsion spring is then used to have one gear press against one side of the rack teeth while the other is pressed against the opposite side. There are also electronic solutions which do much the same thing. This can push the price of a ballscrew system lower than that of a rack and pinion very quickly depending on tolerances.

## Linear Motors
Linear Motors are brushless rotational motors unrolled  into long flat stages. Linear motors are made up of two sections, the forcer and the platen. The forcer contains the energized coils that create magnetic fields to move the "rotor" (note, nothing actually rotates, the rotor in this case is just the moving part, the stator is the stationary part) while the platen is either a flat aluminum plate for an induction motor or a piece with magnets end to end in a SNNSSN type configuration, where the repelling ends are forced together for what are called synchronous motors. 

DC "brushed" Linear Motors do exist, but they are not usually used for linear motion due to current requirements and high friction at common linear motor actuation speeds. They are most commonly used in railgun type applications. 

Induction motors use Lenz's Law to create opposing magnetic fields in the metal plate below them with coils of wire around a magnetic core. They are commonly 3 phase AC motors and are sometimes used in high performance applications. They also produce a levitating effect which makes them well suited to maglev type applications. 

Synchronous motors use permanent magnets and multiple phases instead of induction. This makes them more expensive for long travel than Induction motors. The magnetic field of the permanent magnets provides much increased force and accelerations. There are a couple of common constructions of synchronous motors including U-channel, linear shaft motors, and a special case, linear stepper motors. 

U-channel motors have a set of coils held between the permanent magnets on both sides of a piece of precision machined u-channel. Linear shaft motors have the magnets constrained inside a precision ground shaft like a linear bushing with the coils in the carriage going around the shaft. Linear stepper motors are simply 2 phase (90 degree phase offset) brushless linear motors and can be controlled with standard stepper motor drivers for high holding torque in open loop configurations.

The major advantage to offset the extremely large cost of linear motors is their ability to generate incredible accelerations and speeds compared to even belts and ballscrews. Maximum accelerations of more than 10Gs are possible with small loads. Another factor is that linear motors can be completely non-contact unlike all other linear motion systems with their linear guides if air or hydrostatic guides are used. This provides for extremely smooth motion for sensitive applications like metrology where the slightest imperfection in a guiderail would otherwise cause inaccuracy.

The last, uncommon type of linear motor is the piezoelectric motor. These motors use the piezoelectric effect to essentially walk along a shaft in extremely small increments. These motors can produce extremely precise and accurate movement with the highest of resolutions, but are as slow as a snail. If you need a piezoelectric motor, you'll probably know you need a piezoelectric motor. 

## Stage Indexing
In order to perform precise position control for linear actuators, some sort of index is needed so that a reference point can be established. This can come in several forms.

One extremely frugal way to perform this is with a hard stop, if you are using a stepping type motor in open loop mode for any linear actuator, if you drive the axis into a hard stop, you know your position. This was what was done with old floppy drives in order to get a zero position with no sensor. 

Many different types of sensor can be used to index an axis. These include physical switches, optical interrupters, capacitive and laser distance measurement sensors, and one very important class of sensors known as encoders. 

Encoders are used to determine how far an axis has moved, and potentially where the axis is on its travel. They operate via either optical or magnetic disks(rotary motion) or strips(linear motion) which, when run past a sensor will produce a specific series of digital outputs corresponding to movement. These can come in absolute varieties where the digital outputs will tell you the exact position on the encoder disk or strip you are reading, or incremental varieties where the number of pulses counted will correspond to how far the rotary or linear axis has moved.  A incremental encoder can also come with a index channel witch can provide a point of reference on where to start in your travel. 

Both varieties of encoders can be used to implement closed loop position, velocity, and torque/force control with any variety of motor be it linear or rotary.

For non stepper type synchronous linear motors, there is one more type of relevant sensor. A synchronous linear motor must have feedback in order to time its phases to move in the correct order/direction. This can be accomplished with an linear encoder strip or Hall Effect Sensors. The Hall Effect Sensors will read the Magnetic field intensity and the peaks for north and south can be used to index where the coils are in relation to the magnets. This can also be used for coarse position feedback for control algorithms. 

## Sourcing Parts

Here are some common and reputable vendors for linear motion equipment.

 - Linear Guides and Rails: Misumi, McMaster Carr, Micromo (miniature rails/guides), Hiwin, THK, Nippon Bearing
 - Belt Drive Systems: McMaster Carr, Misumi, SPD-SI, Micromo (miniature), Parker
 - Screw Based Systems: McMaster Carr, Misumi, THK, Hiwin, Nippon Bearing, Lin Engineering(Leadscrews with integrated steppers), Parker
 - Rack and Pinion: McMaster Carr, Parker
 - Linear Motors: Parker, Faulhaber, Nippon Pulse, H2W Technologies(Linear Steppers/Motors), Northern Magnetics(Defunct, widely available on ebay, compatible with some H2W Technologies parts)
 - Indexing: Omron, McMaster Carr, Digikey, US Digital(Reasonably Priced Linear Strip Encoders)
 - Linear Stages(Actuator+guide integrated): THK, Hiwin, Parker, Nippon Bearing, H2W Technologies


## Summary
This has been a very wide overview of common linear actuation technologies. It is important to choose the right actuator type for your needs and budget. 


## Further Reading
 - [Hiwin Linear Guide Video](https://www.youtube.com/watch?v=2I44OT7c_MY)
 - [Hiwin Ballscrew Video](https://www.youtube.com/watch?v=K3i-Ecb698g)
 - [Hiwin Linear Motor Video](https://www.youtube.com/watch?v=mvkcupVxMEI)
 - [Linear Shaft Motor Video](https://www.youtube.com/watch?v=Bxs2PFg0luw&t=329s)
 - [Igus TriboFilament](https://www.igus.com/3d-print-material/3d-print-material)
