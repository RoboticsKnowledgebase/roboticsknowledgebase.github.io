---
date: 2025-04-30
# Comprehensive Engineering Guide: Building a Custom Drone for the DARPA Triage Challenge
---
## Introduction and Requirements Engineering

The DARPA Triage Challenge requires UAVs capable of evaluating casualty vital signs in disaster scenarios while meeting specific technical constraints:

- Total weight under 5kg (including all components)
- Flight duration exceeding 15 minutes (20 minutes optimal)
- Maximum diameter of 1.5m with propellers
- Ability to carry approximately 1kg of mission-specific payload

This tutorial will guide you through two implementation approaches - starting with a modified commercial platform and progressing to a fully custom design.

## Part 1: Rapid Prototyping with DJI Matrice 100 + PX4

### 1.1 System Architecture

The initial implementation leverages the DJI Matrice 100 airframe with a PX4 flight controller replacement. This hybrid approach provides:

- Validated airframe and motors from DJI
- Open-source flight control software via PX4
- Rapid deployment for algorithm testing

### 1.2 Integration Procedure

#### 1.2.1 Components (please see "[full list of drone](https://umilesgroup.com/en/what-are-the-parts-of-a-drone-full-list/)")
- DJI Matrice 100 platform (DJI platform flexible)
- Pixhawk flight controller (versions flexible)
- GPS + Compass module
- [Power module](https://discuss.px4.io/t/power-module/9328)
- Telemetry radio (optional)
- Appropriate cables and connectors (pixhawk 4)

#### 1.2.2 Flight Controller Installation

1. **Mount the Pixhawk** securely to the center plate of the M100 using vibration-dampening foam to isolate it from motor vibrations.

2. **Connect power**
   - The 3DR Power Module connects between the battery and ESCs
   - Wire the 6-pin connector from the power module to the "POWER" port on the Pixhawk
   - This enables both power delivery and battery voltage/current monitoring

3. **Connect mandatory peripherals**:
   - GPS/Compass → "GPS" port (provides positioning and heading data)

4. **Connect optional peripherals** (as needed):
   - 3DR Telemetry Radio → "TELEM1" port (for wireless ground station connectivity)
   - I²C splitter → "I²C" port (for connecting multiple I²C peripherals)

#### 1.2.3 Power System Configuration

The power module is critical for proper operation:

1. **Physical installation**:
   - Connect the XT60 connector labeled "From battery" to your battery output
   - Connect the XT60 connector labeled "To ESC or PDB" to your power distribution board

2. **Wiring explanation**:
   - The red/black wires carry high-current power to your motors
   - The 6-pin connector carries:
     - Regulated 5.3V power for the flight controller
     - Analog voltage sensing signal
     - Current sensing signal
   - Pin 4 connects to an analog pin on the flight controller for monitoring

3. **Software configuration**:
   - In QGroundControl, navigate to Power Settings
   - Set the correct voltage divider and amperage per volt values (typically found on the power module documentation)
   - Calibrate battery voltage by measuring with a multimeter and adjusting in software

### 1.3 Flight Testing and Validation

Before proceeding to autonomous operation:

1. **Initial calibration**:
   - Compass calibration: Perform rotation in all axes
   - Accelerometer calibration: Place vehicle in all six orientations
   - Radio calibration: Set endpoints and center positions for all channels

2. **Manual flight testing**:
   - Begin with short hover tests in Position mode
   - Verify stability and control response
   - Test return-to-home functionality
   - Measure actual flight time with payload

3. **Software configuration**:
   - Set appropriate battery failsafe levels (must return to home base at 10% as required by Darpa)
   - Configure geofence boundaries if operating in restricted areas
   - Set maximum altitude and distance limits

## Part 2: Custom Drone Engineering from First Principles

For optimal performance and maximum control over the design, a fully custom drone allows precise optimization for the mission requirements.

### 2.1 Design Workflow

1. **Preliminary design** with performance estimation tools
2. **Component selection** based on performance requirements
3. **CAD modeling** for mechanical integration
4. **Electrical system design** with proper power distribution
5. **Assembly and integration**
6. **Testing and refinement**

### 2.2 Performance Modeling and Component Selection

#### 2.2.1 Performance Calculation

Using eCalc (www.ecalc.ch/xcoptercalc.php) allows simulation of flight performance before building:

1. **Input parameters**:
   - Aircraft type (Quadcopter)
   - Aircraft weight (target <4kg without payload)
   - Battery configuration (cell count, capacity)
   - Motor specifications (KV rating, power)
   - Propeller dimensions
   - ESC specifications

2. **Analyze results**:
   - Expected hover time
   - Mixed flight time (more realistic than hover time)
   - Maximum thrust
   - Motor temperature estimates
   - Power system efficiency

3. **Iterative optimization**:
   - Adjust component selection to achieve >20 minute flight time
   - Ensure thrust-to-weight ratio exceeds 2:1 for adequate control
   - Verify motor and ESC loading remains below 80% for thermal safety

#### 2.2.2 Critical Component Selection

For a 5kg total weight target:

1. **Frame**: 
   - Carbon fiber construction for maximum strength-to-weight ratio
   - Approximately 450-550mm motor-to-motor distance depending on propeller size

2. **Motors**:
   - 350-400KV brushless motors for 6S battery configuration
   - 120-150W continuous power rating per motor
   - T-Motor MN4014 or similar quality recommended

3. **ESCs**:
   - 30-40A continuous rating
   - BLHeli_32 or KISS firmware for precise control
   - Consider 4-in-1 ESC as shown in for cleaner wiring

4. **Propellers**:
   - 15-18" diameter for efficiency
   - Consider foldable propellers for transport
   - Carbon fiber for rigidity and weight savings

5. **Battery**:
   - 6S (22.2V) 2P configuration for efficiency (S refers to Serial, and P refers to Parallel)
   - 10,000-18,000mAh capacity (split into two packs for weight distribution)
   - Minimum 15C discharge rating

6. **Flight Controller**:
   - Pixhawk 4 or newer recommended
   - Must support all required peripherals
   - Position near center of gravity with vibration isolation

### 2.3 Structural Design and Fabrication

#### 2.3.1 Frame Configuration

A well-designed frame includes:

1. **Central plates**:
   - Upper and lower carbon fiber plates (3-4mm thickness)
   - Separated by spacers to create electronics bay
   - Vibration damping between layers for sensitive electronics

2. **Arms**:
   - Carbon fiber tube construction (16-20mm diameter, 1-2mm wall thickness)
   - Analyze inertia section properties to minimize vibration
   - Securely attached to central plates with aluminum brackets and locking hardware

3. **Landing gear**:
   - Lightweight but sturdy design
   - Consider shock-absorbing elements
   - 3D printed from high-strength materials like nylon or carbon-filled filaments

4. **Motor mounts**:
   - Aluminum for heat dissipation
   - Secure attachment to carbon tube arms
   - Designed for easy motor replacement

#### 2.3.2 CAD Design Process

1. **Create 3D model** of all components
2. **Verify clearances** between all moving parts
3. **Analyze weight distribution** for center of gravity
4. **Design cable routing** paths for clean installation
5. **Validate structural integrity** through FEA if available

### 2.4 Electrical System Integration

#### 2.4.1 Power Distribution

Following the wiring diagram:

1. **Power flow**:
   - Battery → Power module → Power distribution board → ESCs → Motors
   - Battery → Power module → 5V regulator → Flight controller and accessories
   
2. **Current capacity planning**:
   - Main power wires: 12AWG silicone-insulated
   - Signal wires: 22-26AWG depending on current requirements
   - All connections soldered and heat-shrunk or using high-quality connectors

3. **Redundancy considerations**:
   - Separate BECs (Battery Elimination Circuits) for critical systems
   - Isolation of sensitive electronics from power system noise

#### 2.4.2 Control System Wiring

Detailed wiring diagrams for control connections:

1. **Motor signal routing**:
   - Flight controller → ESC signal inputs
   - Maintain equal wire lengths for timing consistency

2. **Sensor integration**:
   - GPS placed highest on frame with clear sky view
   - Compass mounted away from power wires to avoid interference
   - External barometer if needed for altitude precision

3. **RC receiver connection**:
   - PPM Sum connection simplifies wiring
   - Receiver antenna positioned for optimal signal reception

4. **Telemetry system**:
   - Positioned for line-of-sight to ground station
   - Antenna orientation optimized for signal pattern

#### 2.4.3 Control Board Pinout Mapping

1. **Customized Board to PixHawk adaptation**:
   - Follow the pin mapping diagrams precisely
   - Motor numbering must match flight controller expectations
   - Signal directions (RX/TX) must be preserved

2. **Main connection groups**:
   - Motor control signals (M1-M4)
   - Sensor inputs (GPS, compass)
   - Power connections
   - Telemetry and control links

### 2.5 Software Configuration

#### 2.5.1 Flight Controller Setup

1. **Firmware installation**:
   - Flash PX4 firmware using QGroundControl
   - Select appropriate airframe configuration (Generic Quadcopter)

2. **Sensor calibration sequence**:
   - Compass (rotate in all axes)
   - Accelerometer (six position calibration)
   - Gyroscope (keep still during calibration)
   - Level horizon
   - RC transmitter

3. **Motor configuration**:
   - Verify motor rotation directions match diagram 
   - Test motor response through the Motors tab in QGroundControl

#### 2.5.2 Flight Parameters

1. **PID tuning**:
   - Start with conservative default values
   - Gradually increase P-gain until oscillation, then reduce by 25%
   - Adjust I-gain to eliminate drift
   - Add D-gain to dampen oscillations

2. **Safety parameters**:
   - Configure battery failsafe levels
   - Set return-to-home altitude
   - Configure geofence boundaries

### 2.6 Assembly and Integration Procedure

1. **Frame assembly**:
   - Construct central frame with all mounting hardware
   - Attach arms securely with locking compound on bolts
   - Install landing gear

2. **Power system installation**:
   - Mount power distribution board
   - Install ESCs with proper cooling consideration
   - Connect power module

3. **Control system integration**:
   - Mount flight controller with vibration isolation
   - Install GPS mast oriented toward front of vehicle
   - Connect all peripherals according to pinout diagrams

4. **Motor installation**:
   - Mount motors to arms
   - Verify motor rotation directions
   - Install propellers (proper orientation is critical)

5. **Final wiring and cleanup**:
   - Secure all cables with zip ties or cable organizers
   - Protect connections from vibration
   - Ensure no wires can contact moving parts

6. **Center of gravity balancing**:
   - Adjust component positions to center CG
   - Verify balance in all axes

### 2.7 Testing Protocol

#### 2.7.1 Pre-flight Checklist

1. **Visual inspection**:
   - All fasteners secure
   - No damaged components
   - Propellers correctly mounted
   - Wiring secure and protected

2. **Power system check**:
   - Battery voltage verification
   - Power connections secure
   - ESC initialization tones normal

3. **Control system check**:
   - Transmitter control response
   - Flight mode switching
   - Fail-safe activation test

#### 2.7.2 Incremental Flight Testing

1. **Maiden flight**:
   - Open area with no obstacles
   - Calm weather conditions
   - Brief hover test at low altitude
   - Verify stability and control response

2. **Performance validation**:
   - Test flight duration with payload
   - Measure power consumption
   - Verify thermal performance of motors and ESCs

3. **Mission-specific testing**:
   - Simulate actual mission profile
   - Test all automated functions
   - Verify payload operation during flight

## Part 3: Advanced Customization for DARPA Triage Challenge

### 3.1 Mission-Specific Modifications

1. **Payload integration**:
   - Custom mounting for sensors
   - Vibration isolation for cameras
   - Cable management for payload connections

2. **Computing hardware**:
   - Companion computer (Jetson Orin or similar)
   - Proper cooling for sustained operation
   - Power management for extended flight time

3. **Sensor package**:
   - Cameras positioned for maximum field of view
   - Thermal sensors for casualty detection
   - Obstacle avoidance sensors

### 3.2 Operational Considerations

1. **Field maintenance**:
   - Carry spare propellers, fasteners, and cables
   - Quick-swap battery system
   - Field diagnostic procedures

2. **Environmental adaptations**:
   - Dust/moisture protection for electronics
   - Operating temperature range considerations
   - Wind resistance capability

3. **Regulatory compliance**:
   - Remote ID transmitter
   - Appropriate lighting
   - Registration markings

## Conclusion

Building a custom drone for the DARPA Triage Challenge requires systematic engineering from requirements definition through testing and validation. By following this process, you can create a platform optimized for the specific mission requirements while maintaining the flexibility to adapt as needed.

The hybrid approach of starting with a commercial platform allows for rapid algorithm development while the fully custom design enables optimization for the mission's unique constraints. Both approaches require careful integration of mechanical, electrical, and software systems to create a reliable and effective UAV platform.


#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)


## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.