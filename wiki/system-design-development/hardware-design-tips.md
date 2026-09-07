---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2026-05-01 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Hardware Design Tips
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

This article covers practical hardware design tips for robotics engineers across 
mechanical and electronics/mechatronics design. These are drawn from hands-on 
experience building robotics systems and are meant to complement formal coursework. 
Topics include CAD workflows, FEA simulation tool selection, PCB design for 
debuggability, power circuit simulation, and wiring best practices including 
emergency-stop wiring, color coding, and cable management.

## Mechanical Design

### 1. Learn CAD with the ModelMania Problems

After completing introductory tutorials, work through **SOLIDWORKS ModelMania** problems. The archive has 26 years of free problems at different difficulty levels: [SOLIDWORKS ModelMania Archive](https://blogs.solidworks.com/products/solidworks/26-years-of-model-mania/)

A few personal tips:
- Try not to exit a sketch until it is fully constrained. Better yet, use parametric variables so the model's most important dimensions remain editable and independent.
- Try rebuilding the same part in a different way after you finish it; this broadens your design perspective.
- Open-source tools such as [FreeCAD](https://www.freecad.org/) can be a good option when avoiding CAD software license dependencies. It also has a free API and CAD kernel for exploring generative CAD design using MCP servers and LLMs.

---

### 2. FEA Simulation 

Finite element analysis (FEA) simulates how a design responds to forces, heat, and other physical effects before it is built. It helps newcomers identify potential structural, thermal, or fluid-related issues early in the design process.

Get the free ANSYS student license through CMU: [CMU Software Access](https://www.cmu.edu/computing/software/access.html)

ANSYS is widely used in industry. It covers structural, thermal, and fluid simulations.

COMSOL is useful for research work that couples multiple physics, such as heat, structure, and electrical effects. It is more flexible and common in academic papers.

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

- **Separate power and signal-processing electronics on the board** — This helps prevent high-EMI/EMC devices, such as inductors and switching regulators, from introducing noise into low-power signal and GPIO traces.
- **Copper thickness** — Specify copper trace thickness (usually in oz) when placing a PCB order. If the traces are too thin, the circuit may not carry its designed current load.
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

- **Use PG cable glands and conduits** — Use [PG glands](https://www.mcmaster.com/products/pg-glands/cord-grips-2~/) and [cable conduits/hose carriers](https://www.mcmaster.com/products/cable-conduit/) to provide strain relief for cables entering and leaving enclosures, while improving water and dirt resistance for the electronics inside.
- **Use [DIN rails and DIN-Rail Mount Terminal Blocks](https://www.mcmaster.com/products/din-rails/)** - They are the industrial alternative to breadboards and WAGO connectors. They provide a standardized wire connection interface.
- **Use color-coded signal wires** — This simple practice is often overlooked, but it can prevent burnt ICs and shorted Jetson Nanos. General color schemes are given below:
  - VCC|GND: RED|BLACK
  - SDA|SCL: YELLOW|WHITE
  - CAN-H|CAN-L: WHITE|BLUE/BLACK
  - UART TX|RX: GREEN|WHITE
  - PWM: ORANGE
- **Twist differential-pair wires** — For CAN and RS-485, physically twist wire pairs together to reject common-mode noise. If a CAN bus drops frames or reports errors, untwisted wires are a likely culprit before software issues.

  Never twist I2C wires together. I2C is not a differential signal, and twisting can introduce interference that prevents the bus from working.

- **Emergency-stop wiring** — Do not wire an emergency stop directly into a high-current power wire, because a decelerating motor's back EMF can cause sparking at the switch contacts. Use a relay or DC contactor, a flyback diode across the control-circuit terminals, and an RC snubber across the relay contacts to limit voltage spikes when disconnecting the load.

## Summary
These are just starting tips, design only comes from experience!

## See Also
- [Printed Circuit Board Design](/wiki/system-design-development/pcb-design/)
- [Mechanical Design](/wiki/system-design-development/mechanical-design/)
- [Cable Management](/wiki/system-design-development/cable-management/)

## Further Reading
- [KiCad Getting Started — DigiKey YouTube Playlist](https://www.youtube.com/watch?v=0Q6gU7-QqUg)
- [SOLIDWORKS ModelMania Archive](https://blogs.solidworks.com/products/solidworks/26-years-of-model-mania/)
- [LTspice Simulator — Analog Devices](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)
