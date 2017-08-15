---
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
