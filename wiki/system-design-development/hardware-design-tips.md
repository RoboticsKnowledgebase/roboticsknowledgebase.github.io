---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2026-05-01 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Hardware Design Tips for Robotics
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

## Mechanical Design

### 1. To learn CAD, Do the ModelMania Problems.

After you have crossed the tutorials stage, try . **SOLIDWORKS ModelMania** has 26 years worth of problems, free, at different difficulty levels: [SOLIDWORKS ModelMania Archive](https://blogs.solidworks.com/products/solidworks/26-years-of-model-mania/)

A few personal tips:
- Try following the rule of not exiting the sketch till it is fully constrined. Even better, use parametric variables so that your model's most important dimensions remain editable and independent. 
- Try rebuilding the same part a different way after you finish it, it broadens design perspective. 
- Open source software will always be the best choice if you know how to use them. Learning  [FreeCAD](https://www.freecad.org/) would be a great bulletproofing for the future (not dependant on licenses for CAD software). It also has a free API/CAD kernel if you are interested in exploring generative CAD design using MCP servers and LLMs. 

---

### 2. FEA Simulation 

Get the free ANSYS student license through CMU: [CMU Software Access](https://www.cmu.edu/computing/software/access.html)

ANSYS is what most companies use. It covers structural, thermal, and fluid sim. It shows up on job descriptions constantly.

COMSOL is better for research work where you're coupling multiple physics together (e.g., heat + structure + electrical all at once). More flexible, more common in academic papers.

Examples of usage:
- Checking if a robot link will break under load: ANSYS
- Thermal analysis on a motor or PCB: ANSYS
- Piezoelectric actuator or anything multi-physics for a paper: COMSOL
- Airflow around a drone → ANSYS Fluent
---

## Electronics & Mechatronics Design

### 1. Drop Eagle. Learn KiCad.

Autodesk Eagle is being phased out and the free version has a tiny board size limit. KiCad is free, open source, has no restrictions, and is what you'll want if you ever start a company and don't want to pay software licensing fees forever.

KiCad has everything: hierarchical schematics, 3D board preview, SPICE simulation, custom symbol/footprint editors. It's also what a lot of startups and open hardware projects use now.

To get started:
- Do the [DigiKey KiCad 9.0 Tutorials Playlist](https://www.youtube.com/watch?v=0Q6gU7-QqUg&list=PLEBQazB0HUyQ5YJSdCBb79orXaR3Uk5vm&index=2) (<1h play time) to get a really quick jumpstart as it covers the full schematic to layout to gerber to BOM workflow.
- Practice making your own symbols and footprints. You will always have some weird component that's not in any library.
- Always check community footprints against the actual datasheet before trusting them.
- Check your Gerbers in [Tracespace](https://tracespace.io/view/) before submitting to a fab.

For prototyping boards: JLCPCB and PCBWay both work great with KiCad exports, cheap and fast.

---

### 2. PCB layout tips

A board that works in simulation or on paper but has zero test points or indicators is a nightmare to debug in real life. Here are some general PCB design tips:

- **Biforcate the placement of power and signal processing electronics on the board** — This ensures that none of the high EMI/EMC devices like inductors and switching regulators can cause noise on the low-power signal/GPIO traces.
- **Copper thickness** - DO NOT forget to mention the thicknes off copper traces (in oz usually) while placing your PCB order, as this can mean that your circuit can't even take 25% of its designed current load as the traces would be much thinner. 
- **Test points** on every power rail and important signal. Just a pad. Saves hours.
- **Status LEDs** on your power rails (3.3V, 5V, 12V etc.) with a current limiting resistor. You want to immediately know if a rail isn't coming up.
- **Current sense resistors** on high-current paths — small value (0.01–0.1Ω), measure voltage across it with your ADC or multimeter.
- **Label your connectors** on the silkscreen. Pin 1 marker, net name, polarity. You will forget this in the lab at 2am one day.

---

### 3. Simulate Your Power Circuit Before You Spin the Board

Most board failures are catchable before you even order the PCB.

Use **LTspice** (Analog Devices) as it has models for most common components. Use it to:
- Check your gate drive waveforms
- Catch oscillation in feedback loops (buck converters especially)
- Test your design at edge cases (min/max input voltage, max current)

---

### 4. Wiring and wire management

- **Use PG Cable Glands and conduits** - Use [PG glands](https://www.mcmaster.com/products/pg-glands/cord-grips-2~/) and [cable conduits/hose cariers](https://www.mcmaster.com/products/cable-conduit/) to provide strain relief on cables that go in/out of enclosures, and also provide water and dirt resistance to the electronics inside the enclosure.
- **Use [DIN rails and DIN-Rail Mount Terminal Blocks](https://www.mcmaster.com/products/din-rails/)** - They are the industrial alternative to breadboards and WAGO connectors. They provide a standardized wire connection interface.
**Use colour coded signal wires** - This is one of the most trivial-seeming tips but the one that gets overlooked the most, resulting in burnt ICs and shorted Jetson Nano's. General colour schemes are given below:
    - VCC|GND : RED|BLACK
    - SDA|SCL : YELLOW|WHITE
    - CAN-H|CAN-L : WHITE|BLUE/BLACK
    - UART TX|RX : GREEN|WHITE
    - PWM : ORANGE
- **Twist your differential pair wires** - For CAN and RS-485, physically twist the wire pairs together. They cancel out common-mode noise through EMI rejection. If your CAN bus is dropping frames or throwing errors, untwisted wires are a likely culprit before you go hunting for software bugs. BUT, NEVER TWIST I2C CABLES TOGETHER. They are not differetnial signals and cause interference/result in the bus not working at all.
- **Emegency Stop Wiring** -  Do not wire the emergency stop directly on the power wire as this may cause spraking between switch contacts due to the high back EMF from a decelrating motor. Use a relay/DC contactor with a flyback diode across the terminals of the input (control circuit) and an RC snubber circuit across the contacts of the relay will prevent voltage spikes when the load is disconnected.  