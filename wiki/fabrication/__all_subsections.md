/wiki/fabrication/3d-printers/
---
date: 2017-08-21
title: 3D Printers
---
**This article is a technology overview of the fundamentals of 3D Printers, especially "Fused Filament Modeling" (FFM) & "Fused Deposition Modeling" (FDM). It explains the fundamentals of 3D Printer technology and applications. It should be useful for those considering working with, designing, or modifying a 3D printer. It is not intended to be a guide for using the lab's 3D printers.**

## The Basics
### What is 3D printing?
Fundamentally, 3D Printing is the concept of additively producing a physical part from a 3D part file. Typically, this done by breaking a 3D part into horizontal layers and then printing each layer as a planar part of constant thickness. However, there are other methods. For example, [Carbon3D](http://carbon3d.com/), uses an alternative approach known as Digital Light Synthesis which is a two step process of printing with a support resin then melting the support structure away.

![Layering of Material to Create a Shape used in 3D Printing](assets/3DPrinters-10477.png)

Recently 3D printing has become somewhat of a revolution for three reasons:

1) Fast, Easy, Prototyping
3D printing is often able to produce prototype parts faster and easier than traditional manufacturing methods. In theory, 3D printers can work similarly to a traditional paper in which a user 'presses print' and returns some time later to extract a completed part from the printer.

2) Parts which are otherwise unattainable
3D printing enables manufacture of parts which are impossible (or extremely difficult) by traditional methods. Complex internal geometries for example which are often impossible to make by traditional machining can easily be produced by 3D printing. This can be especially helpful for [parts which require internal fluid .](http://3dprinting.com/news/ge-aviation-gets-faa-certification-for-first-3d-printed-jet-engine-part)

3) No cost for complexity:
Unlike traditional machining, 3D printed parts often incur no additional costs when part complexity is increased. Ribs, gussets, fillets, chamfers, internal geometries, etc all typically add additional steps to a traditional machining process, but require no additional overhead for 3D printing. This can free part designers to [create more complex parts and spend less time on Design for Manufacture (DFM).](https://en.wikipedia.org/wiki/Design_for_manufacturability)


## Different types of 3D printing
3D printing comes in many flavors. Varying requirements for precision, materials, speed, cost, size, and application have produced a wide range of different 3D printing technologies. A brief summary of some of the most common variations is as follows. A more detailed explanation is available [here](http://3dprintingindustry.com/3d-printing-basics-free-beginners-guide/processes/).

### Fused Deposition Modeling/Fused Filament Modeling (FDM/FFM)
When makers, hobbyists, and small companies discuss 3D printing they are almost exclusively referring to FDM/FFM. Makergear, Makerbot, RepRaps, Lulzbot, DeltaWasps, FlashForge, Ulitmaker, Witbox, and Craftbot, are all common examples of FDM/FFM printers.

FDM works by melting a plastic filament, which is then extruded through a nozzle and deposited onto a build surface layer by layer to produce a three dimensional part.

The acronym 'FDM' is trademarked by [Stratasys](http://www.stratasys.com/), one of the largest player in the 3D printing industry. Other 3D printer companies using this type of technology generally refer to it by a similar acronym although the technology is generally identical.

### Selective Laser Sintering (SLS)
Within larger engineering companies, SLS machines are sometimes available. These use high powered lasers to fuse together extremely fine metal powders in an oxygen-reduced environment. The reduced oxygen environment is essential to prevent oxides from forming which can inhibit the bonding of powder.

SLS printing is extremely expensive both in terms of capital equipment (millions) and feedstock. Trained operators are required to safely operate the machines as the fine metallic powder produces a serious inhalation hazard. The technology can however produce extremely complex parts out of difficult to machine materials such as titanium and stainless steel. CMU has a few of these printers on campus, including one that students can send parts to in the [Mechanical Engineering Shop](https://www.cmu.edu/me/facilities/student-shop.html).

### Stereolithography (SLA)
Stereolithography is an older technology that has made significant headway and gained more commercial adoption within the last 10 years. The process is based on using a vat of liquid photopolymer material, which hardens when exposed to certain wavelengths of light. To form a part a build plate is positioned within the vat onto which a layer of the part is projected in the correct wavelength. After being exposed for several seconds, the build plate moves by a layer thickness and the next layer is projected onto the surface of the first. A good video of the process is available [here](http://manufacturing.materialise.com/stereolithography).

There are two primary variants of the stereolithography; top down and bottom up. In top down approaches the build plate is initially positioned just below the surface of the liquid in the vat. In bottom up approaches the build plate starts at the bottom of the vat and the image is projected through a plate of glass on the bottom.

Typically, SLA machines also are operated with secondary equipment which further harden and/or clean the product after it is removed from the vat. SLA feedstocks prices vary significantly across companies and materials but an extremely wide range of polymers with vastly different mechanical properties are available.

SLA is known for producing parts with high quality surface finishes at a low price, albeit higher than FDM/FFM cost. It is common to encounter SLA machines (or proprietary variants of it) within medium-to-large engineering companies, although low cost desktop versions have begun to gain traction (i.e. [FormLabs](http://formlabs.com/)).

SLA has also spawned some interesting variants recently. The most notable is Carbon3D which combines bottom up SLA printing with an oxygen permeable print window on the bottom of the vat. The oxygen permeability reduces the curing speed of the material and thus changes from a 'layer by layer' paradigm to a 'continuous' print where a video is projected onto the material while the build platform slowly and continuously moves upward.
An example can be seen [here](https://www.youtube.com/watch?v=UpH1zhUQY0c).

### Material Jetting
Material Jetting is a process which can be visualized similar to an inkjet printer. A print head moves across a build surface while hundreds of nozzles deposit micro-droplets of a liquid polymer material. These droplets then harden (usually via photo-curing) to form the geometry of the part. After traveling across the print surface, the nozzle (or the build platform) moves by one layer thickness and the process repeats.

The [Stratysis Objet Series](https://en.wikipedia.org/wiki/Objet_Geometries#Technology) of printers is a commonly encountered material jetting machine. Objet machines are accessible to students in CMU's [Material Science Shop run by Larry Hayhurst](https://www.cmu.edu/cheme/about-us/labs/machine-shop/index.html) and in the [School of Architecture dFab lab](http://cmu-dfab.org/)

Material jetting machines are known to produce parts with good surface finishes (tolerances of .005 inches are achievable) and are capable of printing multi-material parts. Material jetting machines typically use a support material to support internal surfaces and provide a consistent surface finish. Because of this a huge amount of material is often needed to print even simple parts, as they are typically enclosed in around 1/8" of support material.

## Industry Segmentation
[IBISWorld](http://clients1.ibisworld.com/?u=XTrdcEUxGT9ofTiep6uaiw==&p=Evn+48Pl8G+/5sICIXpGHA==) provides a overview of the 3D printer manufacturing industry. This is focused on large high volume companies such as Straysis and [3D Systems](http://www.3dsystems.com/) and does not include many of the effects of smaller volume 'maker' targeted companies. It shows a roughly equal breakdown between SLA, SLS, and FDM technologies.
![Industry Segmentation by 3D Printing Technologies](assets/3DPrinters-a3389.png)




## How does a FDM/FFM 3D printer work?
### Parts of an FDM/FFM 3D printer
#### Hot End
Broadly referred to as the 'nozzle' the hot end is the portion of the 3D printer which melts, shapes, and deposits print filament to form the part. Typically it is made of the following components:
##### Nozzle
The nozzle is typically made of brass and is a conical nozzle with a small diameter hole through it. For low-cost machines (i.e. anything 'maker') it is usually attached to the heat block with a M6x1 thread. It is designed to be heat-conductive to maintain melting temperatures at the tip of the nozzle where small geometry limits the amount of thermal energy that can be transferred to the material. The length of the small diameter hole is often minimized as viscous friction affects become significant and require a large extrusion pressure to push the highly viscous molten plastic through a small port.

##### Heat Block
The heat block is generally made from aluminum (sometimes brass, i.e. ultimaker) and is responsible from transferring heat from the heater (usually a cartridge heater) and thermistor. It typically connects to both the nozzle and barrel with a single M6x1 thread. Strangely, no one seems to use a tapered thread which often leads to molten plastic leaking out from the top and bottom of the heatblock. Teflon tape helps, but a worthwhile project would be to replace this with a NPT thread or similar.

##### Barrel
The barrel is a tube (usually stainless steel, often with a PTFE liner) which connects the heat block to the extruder (possibly through a Bowden tube). The main design challenge of the barrel is to reduce the amount of heat which can travel across it, thus confining the melt zone to the heatblock/nozzle assembly. Too much heat traveling up the barrel leads to softening/melting of the filament which leads to jamming, excessive extrusion, and inconsistent prints. Barrels are typically made from stainless steel due to its low thermal conductivity. A PTFE liner also is commonly used since it can withstand the high temperatures at the nozzle, provides good thermal insulation between the filament and the metallic barrel, and perhaps most importantly molten filament will not stick to it's slippery surface.

Numerous barrel designs are available, with different features to try to limit heat transfer. Heat sinking features (such as aluminum fins) are common, as are 'heat brakes' which is a section of narrow diameter to reduce the cross-sectional area through which heat can transfer. These different features can have a significant affect on the performance of a 3D printer and thus it is recommended some experimentation be performed if working on a 3D printer project and print quality is sub-optimal. Some examples are [RepRap](https://reprapchampion.com/products/reprap-3d-printer-extruder-brass-barrel-for-mg-plus-hot-end-3-0mm-or-1-75mm), [CycleMore](http://www.my3dprinterblog.com/cyclemore-5pcs-hotend-barrel-m630mm-teflon-nozzle-for-mk8-tube-makerbot-3d-printer-extruder-hot-end/), and a larger [RepRap](http://www.my3dprinterblog.com/plus-barrel-m626-nozzle-throat-for-reprap-3d-printer-extruder-hot-end-1-75mm/).

##### Extruder/Filament Drive
The extruder/filament drive component is the mechanical assembly used to push the feedstock filament through the system. It typically features a stepper motor, some sort of gear reduction, and a 'drive gear' which engages with the filament to push it through the system. Feeding the filament requires large forces and thus significant design effort is applied towards the drive gear (making it firmly engage with the filament, not tear it) and the filament support to prevent it from buckling. There are numerous different extruder designs, some of which are must better suited for certain filament types than others (i.e. soft vs. hard filaments).

Many of the extruder designs can be 3D printed, so once you have one that at least partially works, it is easy to try numerous other designs and to make modifications.

Some of the most commonly used extruder drives:
- [Wade's Geared Extruder](http://reprap.org/wiki/Wade's_Geared_Extruder) - One of the first reliable extruder designs.
- Frequently used on reprap projects, [Greg's Hinged Extruder](http://reprap.org/wiki/Greg's_Hinged_Extruder), is a variant on the Wade extruder, which is similar to the design used on many commercial printers such as MakerGear and MakerBot.
- [Ultimaker extuder](https://ultimaker.com/en/products/extrusion-upgrade-kit), one of the best extruders for common PLA/ABS filament. Has a large gap between the drive gear and output tube thus making it less suitable for soft filaments (i.e. [ninjaflex](http://www.ninjaflex3d.com/products/ninjaflex-filaments/), [semiflex](http://www.ninjaflex3d.com/products/semiflex/)).

#### X-Y axes
The x-y axis construction of 3D printers is one of the most variable aspects between different designs. Within the 'traditional approaches' (in which there is one motor per axis) different printer manufactures chose to move the nozzle, print bed, or both. Drive mechanisms vary between rack gear drives, belt drives, and filament drives.

A number of clever alternative designs also exist which couple the motion between different motors to move the print bed or nozzle. The [CoreXY](http://corexy.com/) design mixes motion from two motors for the X-Y axes through a single filament or belt drive. [Delta geometries](http://reprap.org/wiki/Delta_geometry) mix motions between 3 vertical axes to produce coupled x-y-z movements transferred to the nozzle via connecting rods with ball joints. Proponents of these technologies argue that they provide more symmetric construction, reduced moving mass (enabling higher print speeds), and larger or more practical build volumes. Opponents cite increased mechanical complexity, increased software complexity, and varying positional tolerances throughout the build volume.

#### Z axis
The z axis is responsible for moving the build platform (or occasionally the nozzle) between layers. As such, it has significantly different requirements than the X-Y axes which are continuously rapidly moving throughout the print. Z axes are typically geared lower since speed is less critical but they must support a much larger mass. Backlash is also less of a design consideration since the axis typically only moves in one direction throughout the print.

#### Endstops
Since almost all 3D printer designs utilize stepper motors for axis motions, inclusion of endstops is necessary for homing the axes at power up. The importance of good endstop design should not be overlooked since its accuracy and repeatability will directly affect the quality of the print. The ability to provide mechanical adjustment is often essential as printer geometry will be varied via the use of different nozzles, barrels, extruders, etc.

Wiring and firmware must also be considered, since it is often desirable to have the endstops 'always active' so that they will prevent any of the axes trying to drive beyond it's mechanical limits, which can be caused by incorrectly generated G-Codes. Use of interrupt driven code for endstop checking is best for this application, since response time is critical and the processors in 3D printers are typically heavily taxed performing geometry calculations.

#### Build Platform + Build Chamber
The platform on which the part to be built is worth consideration. These are typically heated to increase material adhesion, and the temperature may be varied throughout the print to maximize platform adhesion in the beginning of the print, and prevent thermal warping of the part in later stages. Surface material is important since it also greatly affects the adhesion. Glass is commonly used since it provides a low cost extremely flat surface, but it is often modified (with kapton tape, spray adhesives, etc) to improve adhesion. Professional-grade 3D printers (i.e. Stratysis) typically use various thermoplastics for their platform which can be rapidly swapped between builds. An interesting concept which has gained some traction is flexible build platforms. When placed on a flat surface they provide a flat surface for part construction, but then allow easy part removal by flexing the platform after the print is finished. [This video](https://www.youtube.com/watch?v=1T5BdRFlCd8) highlights the concept well.

The [RepRap Wiki](http://reprap.org/wiki/Heated_Bed) has a great and exhaustive description of platform types and design considerations

Many printers (i.e. makerBot) also use heated build chambers. The concept is to prevent part warpage by maintaining a consistent temperature throughout the entire build volume. Often these chambers are heated by the build platform, with the entire build volume enclosed by an insulating chamber.

#### Fans
Fans are used extensively in 3D printers. They are commonly used for cooling the print material as it is deposited, cooling the barrel/hot-end assembly to prevent heat from melting the filament prematurely, and for cooling the stepper motor drivers. For the part-cooling fans, numerous control strategies are used including not running the fan for the initial layers (to maximize platform adhesion) or only running the fan when bridging gaps within the print.


/wiki/fabrication/cube-pro/
---
date: 2017-08-21
title: Building Prototypes in CubePro
---
[Cube Pro](https://www.3dsystems.com/shop/cubepro) is a 3D printing alternative to the popular MakerBot Replicator Series. This is a step-by-step tutorial to using CubePro for developing prototypes and parts. CubePro is available to Carnegie Mellon University Students in Mechanical Engineering cluster.

## Step 1: Material Check
Before starting your print, check the type of material and how much it is available for printing.

## Step 2: Building Model Using Cube Pro Software
For printing parts in Cube Pro, you need to develop a build file using the Cube Pro software available on the cluster computers. You need to import a `.STL` file and align the model on the plate in the software. Make sure that your model dimensions do not exceed the maximum dimensions that Cube Pro can print. After importing and aligning the model, select the printer jet (Jet 1 or Jet 2 depending on the cartridge installation in the Cube Pro) and material in the printer option in Cube Pro software. After selecting this, select “Build”. This open a dialogue where you can select the “Layer Resolution” (70, 200, 300 micron), “Print Strength” (Hollow, Strong, Almost Solid and Solid) and “Print Pattern” (Cross, Diamonds and Honeycomb), determine whether you need “Support Material” for your print and select build. This will create a build file for your model and tell you how time it will take to print.

## Step 3: Placing the build plate
Place the build plate in the build chamber. Make sure you use the right sized plate in the Cube pro. Magnets at the bottom of the plate will ensure that the plate stays in plate during the printing process.

## Step 4: Plate Leveling
By using the touch screen, open the “Setup” option. Scroll right until you reach the “Level Plate” option. Select “Level Plate” option. Now you will see a square with 3 triangles (representing the leveling screws beneath the build plate) with the top left screw selected by default. Put a sheet of paper on the build plate so that it is between the nozzle and the build plate. Using the leveling screws, raise (CCW motion) or lower (CW motion) the plate so that you are just able to pull the paper out without damaging the paper. By doing so, you not be able to push the paper between the nozzle and the build plate. Repeat the process for all the leveling screws.

## Step 5: Print Your Model
After leveling the plate, go to the home screen. Plug your pen-drive which has your build file for the model you want to print in the slot provided. The build file can be created by using the Cube Pro software on the cluster computers. After you have plugged the pen-drive in, select “Print” on the home screen. You will see different build files on your pen-drive displayed on the screen. Scroll and select the right build file and select “Print” on the bottom of the screen. Cube Pro will now ask you to apply glue on the build plate.

## Step 6: Applying Glue
After the build plate comes to rest, apply glue on it covering the area that is going to be covered while printing. A good estimate can be obtained from the build file which shows the expected area covered by the print. Glue should be applied in a smooth fashion without pressing too much on the plate. Also, glue should be applied sufficiently as too little or too much glue will create problems during and after print. Too little glue will lead to the material not sticking to the build plate during print and result in bad print or print failure. Too much glue will result in difficulty in getting the printed part off the build plate and may result in damage to the part. After you have applied the glue, close the Cube Pro door and select the check mark at the bottom of the screen. The Cube Pro will start printing the model.

## Step 7: Part Removal
After the print has completed, wait for some time so that the heated plate can cool down. After the cool down, remove the build plate carefully from the Cube Pro and run the part and the plate under water (preferably hot). Water dissolves the glue and allows easier removal of part. You might need to use a chisel to remove you part (depends on the material, print pattern and the amount of glue you have used). After removing the part, clean the part and the plate using paper towels. Also remove and stray chips that might have been left during the part removal on the plate so that you and others have a smooth surface for the next print.


/wiki/fabrication/fabrication_considerations_for_3D_printing/
---
date: 2019-05-16
title: Fabrication Considerations for 3D printing
---

There is more to 3D printing than designing components and parts and replicating those 3D solid models into useful assemblies.  When beginning a design, one must take into account the 3D filament to use, the 3D printing equipment, the model’s final resolution and the limitations of each 3D CAD software to fabricate quality parts.  By keeping these key elements in mind, along with repetition, the 3D printing practitioner and designer will maximize his/her efforts, resulting in high-quality finished products.

Following are some of the main consideration one must keep a note of while designing and fabricating parts using 3D printing:

## 1. Wall Thickness

The wall thickness defines the number of times the extruder will lay filament around the perimeter of your model before switching to your infill factor. (Infill is defined separately as a percentage. Typically, 10% or 20% works well for infill.)

Your wall thickness should be defined as a multiple of your nozzle diameter so a .4mm nozzle with a .8mm wall thickness would result in the printer laying a perimeter two thicknesses wide for each layer. This is the case with the MakerBot Replicator 2.

It is most important to keep this in mind when you are drawing very thin walls. When parts of your model are thin, the perimeter walls are so close together that it doesn’t leave much room for infill in between them. Sometimes, it may not properly fuse the walls to each other, leaving a hollow gap between the walls on the inside of the model.


## 2. Part Orientation
![part_orientation](http://www.ap242.org/image/image_gallery?uuid=a7eb3359-8620-457d-9923-9d4cbec5ba5e&groupId=52520&t=1503324782232)

Part Orientation plays a vital role in determining the final finish of the 3D printed part. In the image above, 4 possible configurations are shown in which the object can be 3D printed. The part orientation while printing affects the following:

- Time of printing
- Surface Quality
- Mechanical properties (Tensile and Shearing)
- Supports and Overhanging structure

In the above case, it is best to print the part in the configuration shown in lower left i.e. laying along the surface having the maximum surface area. 

## 3. Overhangs and supports
[overhangs_and_supports](https://cdn-images-1.medium.com/max/1500/1*vRzutfX5qpPH9NHZh-fHBQ.png)

A 3D printing overhang is any part of a print that extends outward, beyond the previous layer, without any direct support.

Usually, overhangs up to 45 degrees can still be printed without loss of quality. That's because any layer in a 45-degree overhang is 50% supported by the layer beneath. In other words, each new layer has enough support to remain intact and to make printing possible.

However, anything past 45 degrees approaches the horizontal and becomes difficult to print. Such overhangs are prone to curling, sagging, delamination, or collapsing. Exceeding 45 degrees means every new layer has less of the previous layer to bond with. This translates to a poor quality print with droopy filament strands.

How to deal with Overhangs?
- Enable support structures.
![N|Solid](https://cdn-images-1.medium.com/max/1000/1*aFNYLE7vSIA6r-gRyZOGpg.png)
- Orient the object to minimize the need for support structures.
- Slicing i.e. divide the part into subcomponents and glue them together at the end.

## 4. Warping & Shrinkage
[warping_and_shrinkage](https://static.wixstatic.com/media/e96ca7_9881d5b065274700a5ea433aec66f55f~mv2.jpg/v1/fill/w_410,h_273,al_c,q_80,usm_0.66_1.00_0.01/e96ca7_9881d5b065274700a5ea433aec66f55f~mv2.jpg)

Warping occurs due to material shrinkage while 3D printing, which causes the corners of the print to lift and detach from the build plate. When plastics are printed, they firstly expand slightly but contract as they cool down. If material contracts too much, this causes the print to bend up from the build plate. Some materials shrink more than others (e.g. PC has a higher shrinkage than PLA), which means there’s a larger chance of warping when using it.

How to deal with warping?
- Use Raft: A raft or brim is basically underground for your product. You print a number of very light layers under your product. This ensures fewer shrinkage differences and tensions at the bottom of your product. If there is warping, then this mainly affects the raft instead of your product. It is therefore important that the raft is wider than your product so that it does not cause problems when the corners curl up.
- Temperature Control: Choose a printer where the cooling of the product is as gradual as possible. For this purpose, a closed and conditioned box is essential, among other things. This way you have control over the temperature in the cabinet to minimize differences in shrinkage.
- Ensure the build plate is leveled correctly.
- Apply an adhesive:  When using a heated build plate, it’s recommended that you apply an adhesive to the glass plate.  Please refer to the type of material and compliant adhesive.

## 5. Tolerance
Tolerances describe how much deviation from a particular value is expected or acceptable. While 3D printing the part may slightly deviate from the actual dimensions. A tighter tolerance indicates higher dimensional accuracy. 

The tolerance can be improved by:
- Use better filament
- Properly calibrate the printer
- Check the motion components
- Place the component at the center of the printing base

## 6. Material
The choice of material can be crucial for designing 3D printed materials. Here we discuss the two most widely used materials for 3D printing.

### PLA
PLA (Polylactic Acid) is one of the two most commonly used desktop 3D printing filaments. It is the "default" recommended material for many desktop 3D printers, and with good reason - PLA is useful in a broad range of printing applications, has the virtue of being both odorless and low-warp, and does not require a heated bed. 

Printing Filament Properties:
- PLA filament is a stiff but brittle 3D printing material.
- Well suited for prototyping
- Best for low-stress applications. Should not be the major load bearing of a mechanical structure.
- Best 3D printer material for beginners due to the ease of printing and minimal warp.

Technologies: FDM, SLA, SLS

PLA material properties for printing:
- Strength: High | Flexibility: Low | Durability: Medium
- Difficulty to use: Low/li>
- Print temperature: 180°C – 230°C
- Print bed temperature: 20°C – 60°C (but not needed)
- Shrinkage/warping: Minimal
- Soluble: No



### ABS
Acrylonitrile Butadiene Styrene or ABS for short is created from Acrylonitrile, Butadiene and Styrene polymers. It is a material commonly used in personal or household 3D printing, which is done using primarily FDM. If you are used to printing with PLA, you’ll probably find ABS a little trickier to print with.

ABS Filament: What are the Proper Settings?
It can be quite some task for a newcomer to find the right settings. No spool is alike, and the specifications may vary. Make sure you cover these bases:

- Close the printer: Before printing, make sure your 3D printer is either closed or that the ambient temperature of your environment isn’t too hot or too cold.
- Use a heated bed: Set the temperature to 110 degrees Celcius. Also, think about using Kapton (blue) on your glass build plate. If you still have problems with the first layer adhesion of your ABS filament, you should use a raft.
- Experiment with the temperature: Set the nozzle temperature to 230 degrees and work your way from there.
- If you see a lot of plastic strings between separate parts and the printer oozes material, lower the temperature by 5 degrees.
- If the filament doesn’t stick, the parts aren’t sturdy or the surfaces have a rough feel, increase the temperature by 5 degrees.


ABS 3D Printing Filament Properties:

- ABS filament is strong, ductile material with wear resistance and heat tolerance.
- Common 3D prints with ABS are Interlocking parts like gears, parts exposed to UV and heat like a car cup holder, or prototyping.
- A wide selection of methods for excellent post-processing.

Technologies: FDM, Binder Jetting, SLA, PolyJetting

ABS properties:
- Strength: High | Flexibility: Medium | Durability: High
- Difficulty to use: Medium
- Print temperature: 210°C – 250°C
- Print bed temperature: 80°C – 110°C
- Shrinkage/ warping: Considerable
- Soluble: In esters, ketones, and acetone.

## Further Reading:
1. Design considerations for 3D printing: https://www.3dhubs.com/knowledge-base/key-design-considerations-3d-printing

2. Calibrate your 3D printer: https://all3dp.com/2/how-to-calibrate-a-3d-printer-simply-explained

3. Calibrate extruder: https://all3dp.com/2/extruder-calibration-6-easy-steps-2

4. Dimensionality test: https://www.youmagine.com/designs/make-2016-3d-printer-shootout-models

5. How to fix warping: Tutorial bu Ultimaker: https://ultimaker.com/en/resources/19537-how-to-fix-warping

6. Accuracy of 3D printing materials: https://www.3dhubs.com/knowledge-base/dimensional-accuracy-3d-printed-parts

## References:
1. Printing using ABS material: https://all3dp.com/abs-3d-printer-filament-explained

2. 3D Printing material available: https://all3dp.com/1/3d-printer-filament-types-3d-printing-3d-filament

3. Printing using PLA material: https://all3dp.com/1/3d-printing-materials-guide-3d-printer-material/#pla

4. Testing Tolerance of 3D printer: https://all3dp.com/2/3d-printer-tolerance-test-and-improve-your-3d-printer-s-tolerances

5. Ultimaker manual of 3D printing materials: https://ultimaker.com/en/resources/manuals/materials

6. Overhanging structures in 3D printing over 45 degrees: https://all3dp.com/2/3d-printing-overhang-how-to-master-overhangs-exceeding-45

7. How to deal with material overhangs in 3D printing: https://medium.com/bravovictornovember/3d-print-overhangs-and-how-to-deal-with-them-9eed6a7bcb5d


/wiki/fabrication/machining-prototyping/
---
date: 2017-08-21
title: Machining and Prototyping
---
https://www.youtube.com/user/dgelbart/videos - 18 incredible videos on basic machining and prototyping of mechanical components

https://www.youtube.com/user/KEF791/featured - Keith Fenner has some detailed videos on machining gear systems


/wiki/fabrication/makerbot-replicator-2x/
---
date: 2017-08-21
title: Makerbot Replicator 2x Overview
---
[Makerbot Replicator 2x](https://store.makerbot.com/printers/replicator2x/) is a popular 3D printer used by hobbyists. This page is for information on the Makerbot Replicator 2x.

## Software
Software is available for [download here](http://www.makerbot.com/makerware/).

## Getting Started
#### From SolidWorks to 3D-Printed Beauty
This is a good starting points for anyone looking to convert files from their SolidWorks or another CAD files to get it printed on the Makerbot.
1. From your SolidWorks file or any other CAD file, save the file as a `.STL` file extension.
2. Open up the Makerbot software downloaded from the link above.
3. On the top right, click "Add File" and select your file with the .STL file extension.
4. when the file shows up on your screen in the Makerbot software, position your part in the right position and orientation desired. (Use the move objects and rotate objects button on the left-hand side. The "lay flat" feature in the Rotate and "Put on Platform" in the Move are quite useful).
5. If when moving and rotating the object, the move and rotate buttons show that no item is selected, click on the object once to select and continue.
6. Once positioning/rotating is done, select "Settings" on the top right.
7. Resolutions: How high of a resolution do you want. (Mainly in the z-direction, from 0.15mm to 0.3mm) The higher the resolution, the better it looks, but also the slower it gets printed.
8. Raft/Support: Select if you like to use the raft feature. Support is automatically generated and no control can be applied there. Raft is an optional feature where the entire block is built on top of a layer of the support. It usually adds complexity.
9. Go to "Temperature" in the bottom box, 2nd tabl. Change the temperature for the heat plate to 135C.
10. When done, click on "Export" in the top right corner.
11. Save the Export on you SD card that you'll use for the broth.
12. Take the SD card, go to the Makerbot, and press "Build from SD Card"
13. Once in a while, the machines would encounter other problems leading to a failed print. Stop the machine, examine the system and reprint from the SD Card.

## Tips
- Read the manual!
- Keep the door closed and the top on to keep uniform heating.
- Default settings can be applied from the Settings menu.
- You can use the Makerware software to change speed, resolution, and heating elements if your build needs special settings.

## Custom Print Profiles
### Space Jockey Custom Profile
This is 2013-14 Team B's Makerbot slicer profile designed for faster, stronger parts (faster infill, 3 shells), with the temperature settings already tweaked for the MRSD MakerBot. Download and unzip to User/My Things/Profiles, and then select "Space Jockey Hull Parts" in your print settings in Makerware.

[Download Space_Jockey_Profile.zip](assets/Space_Jockey_Profile.zip)

## Issues Log
1. **Issue:** Large objects would start peeling off the left side of the plate.
  - **Fix:** Re-leveling the plate helped
2. **Issue:** Large objects began curling on the sides, perhaps due to higher layers cooling and contracting, pulling up the sides.
  - **Fix:** Increasing plate temperature to ~ 120C
3. **Issue:** Makerbot kept doing random and strange things 3/4 of the way through the first layer (stop moving printhead and extrude or retract filament, send the printhead to strange location)
  - **Fix:** Use an SD card with less than 1GB of data on it. (Was using a 2GB card, as soon as I dropped it below 1 GB of data on the card, it started working fine again with the exact same file on it)
4. **Issue:** Makerbot prints do not stick to plate while printing and prints fail because each layer is offset by the moving prints as a result
  - **Fix:** Increase the heat plate temperature to 130C. (You should really just use this every time instead of the default). If the print looks like it's about to come off the heat plate, try to fix it in place with little bits of tape.
5. **Issue:** Makerbot is moving, but no plastic is coming out of the extruder.
  - **Fix:** The extrude is clogged. You must cancel and reprint everything. Cancel the build, wait for it to stop printing, and follows the instruction to load the filament. Steps are below:
    1. Press cancel to cancel print.
    2. Wait until the Makerbot stops.
    3. Go to "Change Filament"
    4. Press "Unload xxx extruder"
    5. When prompted by Makerbot, pull out the extruder from the top.
    6. Make sure the extruder channel is clear, press "Load xxx extruder"
    7. Push the clipped filament back in until plastic starts to come off the bottom of the extruder.


/wiki/fabrication/milling-process/
---
date: 2017-08-21
title: Milling Process
---
Milling is one of the most important machining processes. Knowledge of milling would be very helpful in prototyping. Almost all mechatronic design projects would require milling at one time or the other.

The following 2 part tutorial is very helpful in gaining insights into the usage of the tool.
1. [Part 1](https://www.youtube.com/watch?v=U99asuDT97I)
2. [Part 2](https://www.youtube.com/watch?v=RIbdYmmhPDI&feature=youtu.beTut)


/wiki/fabrication/rapid-prototyping/
---
date: 2017-08-21
title: Rapid Prototyping
---

One of the biggest challenges a designer faces is validating their design works. The more complex the system design becomes, the more difficult it becomes to accurately predict the behavior of the system. Development of full scale prototype takes time and could prove costly both in terms of time and money if mistakes happen. This is where rapid prototyping comes into play. Rapid prototyping allows the designer to start with a low precision model (developed using paper and pen) and move to increasingly higher precision prototypes as the design iterates. Each iteration provides data which helps a designer to understand the shortcomings of their design and improve it for the next iteration. The positives of doing rapid prototyping are that they help provide a visual of the future state of the system, eliminate errors and prevent time consuming and costly modifications in the final stages of the project. Some of the pros of rapid prototyping are:
- To decrease development time on final product
- To decrease costly mistakes
- To increase effective communication of the design
- To minimize engineering changes
- To eliminate redundant features early in the design


Some of the techniques that can be used for developing a rapid prototype are:
- CAD Design
- Cardboard/Wood
- 3D Printing
- Laser cutting

When developing prototype, the major thing to focus on is the purpose of the prototype. The prototype can be of two types:
- Vertical Prototype: in-depth functionality of a few key features
- Horizontal Prototype: Complete visual representation of the model with no underlying functionality

It is important to keep in mind what the purpose of the prototype is. A good rule is to focus on the 20% of the functionality that will be used 80% of the time. While it always useful to develop prototypes that pertain to different subsystems to check their validity and effectiveness, the complete model plays an important role too. It helps you to determine different properties of your design, such as the weight and dimensions of the system. You can determine whether you are overshooting your target and where the scope of improvement is.

The important thing to keep in mind that rapid prototyping does is that it helps you to test and fail early in your design process. Quickly developing systems and testing them allows you to understand the failure modes and work towards eliminating those. Faster you develop and test, more time you have to build a robust and fail proof system. Ra


/wiki/fabrication/series-A-pro/
---
date: 2017-12-16
title: Series A Pro
---
The Series A Pro 3-D printer is a reliable platform that offers a large build volume, flexible controls (including the option to add custom G-code), print monitoring, a large range of print materials. Compared to more beginner-friendly but restrictive options, like the MakerBot, the Series A allows more control and flexibility to the quality, speed, and strength of your print. It also allows the use of more print material options including PLA, PETT, PETG, Nylon, and NijaFlex. 

## Software
The Series A Pro uses a custom version of Cura slicer software,[Cura Type A](https://www.typeamachines.com/downloads#Software), to prepare .STL files for printing. Install, but do not setup Cura until you are ready to setup the Series A Pro on your machine. 


## Getting Started
The Series A Pro uses a web platform to manage and monitor your prints. Once Cura is installed and you have your .STL prepared, turn on the Series A Pro and wait for about 3 minutes. A WiFi network named "Series1-10610" should appear. Connect to this network. 

> You will not have internet access for the remainder of the setup to until you begin your print! Make sure everything you need to print is pre-downloaded.

In a bowser window, go to <http://series1-10610.local:5000>. This should load the Octoprint Series A web interface. 

Once you are connected, open Cura Type A. When prompted, type in our printer's serial number and API key:
- Serial number: 10610
- API Key: Click on the wrench icon in the upper right hand corner of the Series A webpage and go to the API tab. Copy and paste the API key. 
![Find your API Key here](assets/SeriesA_APIKey.jpg)

Click the "Configure" button to prepare your printer. Cura Type A should open. 

In order to print, configure and save your .STL file as .GCODE. In your web interface, upload your file and hit "Print" in the upper left corner of the page. You can monitor your print status by checking the webcam feed, tempurature, or GCODE Viewer tabs. 

## Preparing your print
The Series A Pro printer generally uses PLA (as opposed to ABS). PLA requires very different print settings, and trying to print with ABS settings may damage the machine. Generally: 
- Do not print above 220C. 210C is a good starting point.
- The bed tempurature should be around 50C

### Print Profiles
The standard profile is a good compromise between speed and print quality for PLA on the Series A Pro. 
[Download SeriesA_Standard_Profile.zip](assets/SeriesA_Standard_Profile.zip)
![Print Quality of Standard Profile](assets/SeriesA_PLA_SP.jpg)

## Tips
#### Changing Fillament
Unlike the MakerBot, there is no automating fillament changing funtion. On the Series A web interface, click the "Tempurature" tab and set the printer tempurature to 220C by typing in the indicated text box, clicking enter, and waiting a few seconds for the printer to update. You should see the light red line in the adjacent tempurature graph jump to 220C, and the dark red line should follow. Once the tempurature reaches 220, pull out the old fillament, insert the new fillament, and switch to the "Control" tab. Click the extrude button until the fillament begins to extrude. 

#### Leveling the Build Plate
In theory, the Series A self levels before every build. If you're having any issues with first layer adhesion, turn the printer off and move the print head to the corners. Adjust the leveling screw under each corner until a peice of paper just barely fits between the plate and the nozzel. Repeat this process twice for each corner. 

#### Cleaning the Build Plate
Occasionally, a print will become stuck to the build plate. First, make sure the print and the plate have cooled completely before you try to remove the print. Before you begin a print, put down a fresh layer of painter's tape. Your print should come off of painters tape more smoothely and will not damage the build plate in the even that it gets stuck. 

#### Poor print quality
Cura Type A allows you to adjust almost every aspect of your printer's settings. If you are not satisfied with your print quality, there is almost certainly a fix online. Common problems are:
- "Ringing" on the corners of your prints: Reduce print head acceleration
- Poor platform adhesion after a few layers: Print on a raft, or decrease the surface area that touches the build plate

## Summary
Though there is a steeper initial learning curve, the Series A Pro allows much more flexibility in the quality, speed, and stregth of your 3D prints. 

## See Also

## Further Reading 

## References

<https://typeamachines.zendesk.com/hc/en-us>


/wiki/fabrication/soldering/
---
date: 2017-08-21
title: Soldering
---
A complete and comprehensive 'How to' guide on soldering. A complete set of tutorials has been provided to guide students on effective soldering techniques, the caveats, and other relevant information.

This guide is useful for beginners, who are new to soldering or have minimal soldering experience. Though it is quite handy for electrical engineers too, who have been out of touch of soldering for a while. The content of this video series is owned by [Pace USA](http://www.paceusa.com/).

1. [Basic Soldering Lesson 1: Solder & Flux](https://youtu.be/vIT4ra6Mo0s?list=PL926EC0F1F93C1837).
  - Choosing the correct flux, solder and the soldering iron is crucial in achieving a good solder. This video teaches us about that.
2. [Basic Soldering Lesson 2: Soldering to PCB Terminals](https://youtu.be/Mrhg5A1a1mU?list=PL926EC0F1F93C1837).
  - Wire stripping, selecting the correct tools and soldering to PCB terminals is crucial to have a flawless circuit. This video teaches us about that.
3. [Basic Soldering Lesson 3: Cup Terminals](https://youtu.be/_GLeCt_u3U8?list=PL926EC0F1F93C1837)
  - Soldering to different types of terminals is useful when dealing with different instruments with various connectors. This video teaches us about that.
4. [Basic Soldering Lesson 4: Bifurcated Terminals](https://youtu.be/hvTiql-ED4A?list=PL926EC0F1F93C1837)
  - Soldering to different types of terminals is useful when dealing with different instruments with various connectors. This video teaches us about that.
5. [Basic Soldering Lesson 5: Hook and Pierced Terminals](https://youtu.be/sN3V8hMiUb4?list=PL926EC0F1F93C1837)
  - Soldering to different types of terminals is useful when dealing with different instruments with various connectors. This video teaches us about that.
6. [Basic Soldering Lesson 6: Component Soldering](https://youtu.be/AY5M-lGxvzo?list=PL926EC0F1F93C1837)
  - This video teaches us about proper component soldering, the prepping procedures and proper component mounts. This focuses on single sided board.
7. [Basic Soldering Lesson 7: Integrated Circuits: The DIP-Type Package](https://youtu.be/VgcPxdnjwt4?list=PL926EC0F1F93C1837)
  - This video teaches us about proper component soldering, the prepping procedures and proper component mounts. This focuses on double sided board.
8. [Basic Soldering Lesson 8: Integrated Circuits: The TO-5 Type Package](https://youtu.be/sTv3gK9tAKA?list=PL926EC0F1F93C1837)
  - This video teaches us about proper component soldering, the prepping procedures and proper component mounts. This focuses on double sided board.
9. [Basic Soldering Lesson 9: Integrated Circuits: The Flatpack & Other Planar-mounted Components](https://youtu.be/Nq5ngauITsw?list=PL926EC0F1F93C1837)
  - This video focuses on flat pack soldering on PCB.


/wiki/fabrication/turning-process/
---
date: 2017-08-21
title: Turning Process
---
Turning process is done on lathe machines and is ideal for creating components with rotational symmetry. Components such as shafts, sleeves, pulleys etc can be manufactured/modified on a lathe. This two part tutorial gives in a good insight into the usage of the lathe machines.

1. [Part 1](https://www.youtube.com/watch?v=H0AyVUfl8-k)
2. [Part 2](https://www.youtube.com/watch?v=Q7QUiCJJmew&feature=youtu.be)
