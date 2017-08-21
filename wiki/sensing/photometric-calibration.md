---
title: Photometric Calibration
---

## Background
Digital cameras of today have CMOS based sensors to convert the light incident (irradiance) on them into digital values. These sensors have a characteristic Inverse Camera Response Function (ICRF) which maps the irradiance to the pixel value generated (typically between 0-255). In the cameras we use, the ICRF curve is adjusted so that the color reproduced in the digital pixels resemble what our human eye can see. This is particularly useful for consumer products but when one is using cameras for scientific applications ranging from vision systems in autonomous cars to 3D reconstruction, it is imperative to have the true pixel value to be calibrated to the true irradiance values on the CMOS sensor.

## Problem
The goal is to obtain the absolute value of light intensity and calibrate the CMOS sensor output of the camera to match the said absolute value of light intensity. Highest accuracy and precision are desired.

There are two ways of approaching this problem:
1. **Method I:** Get the value of the intensity of light of the surface using Photometers/Lux meters/Radiometers.
2. **Method II:** Use a standardized light source with controllable wavelength and intensity.

A comparative overview of the two stated methods has been given below in **Table 1**, each advantage is given an *unweighted score of 1:*
**Table 1**

 |Method I |	Method II
---|---|---
Principle of Operation |	Uses a transducer to convert light intensity to a digital signal. |	Uses a transducer to convert digital signals into light waves.
Sensor/Transducer |	Silicon(doped) Photodiode |	Silicon(doped) LED / Tungsten Filament
Cost |	**$ - Cheap** | 	$$$ - Expensive
Luminous efficiency error | 	9% - High | 	**0.001% - Low**
Dependence on ambient light |	In-effective/false positives under fluorescent lighting | 	**Independent of ambient lighting**
Response time | 	5 s | 	**0.500 s**
Characteristics of oblique incidence/ Luminance Spatial uniformity | 	Incidence: 10° ±1.5% / 30° ±3% / 60° ±10% /80° ±30% | 	**Spatial Uniformity: >94% over 360° x 200° field of view**
Spectral range | Lux meter: 1; Photometer: 850 nm to 940 nm | **Visible, 850 nm to 940 nm**
Spectral mismatch |	1% | **>0.00001%**
Luminescence range |	**0.0 to 999 cd/m2** | 	0 to 700 cd/m2
Typical application| Lux meter: Ambient light; Photometer/Radiometer: Color of surfaces. | **Calibration of Lux meters, Photometers, Radiometers, Cameras & other optical equipment.**
Operational features | Comparatively less stable output; Needs regular calibration; Integration with desktop on select models. | **Precise control. Easy integration with desktop. Long life. Stable output**
Total score | 2/10 | 	**7/10**


## Result
Method II is the most desirable way to go about solving the problem at hand.

## Recommendations
1. [Gamma Scientific](http://www.gamma-sci.com/products/light_sources/)
2. [Labsphere](https://www.labsphere.com/labsphere-products-solutions/imager-sensor-calibration/)


## References
Use these for choosing the type of validation of photometric calibration:

1. https://www.labsphere.com/site/assets/files/2928/pb-13089-000_rev_00_waf.pdf
- http://ericfossum.com/Publications/Papers/1999%20Program%20Test%20Methodologies%20for%20Digital%20Camera%20on%20a%20Chip%20Image%20Sensors.pdf
- http://sensing.konicaminolta.us/2013/10/measuring-light-intensity-using-a-lux-meter/
- http://tmi.yokogawa.com/products/portable-and-bench-instruments/luxmeters/digital-lux-meters/
- http://ens.ewi.tudelft.nl/Education/courses/et4248/Papers/Niclass12.pdf
- http://photo.net/learn/dark_noise/
- http://ro.ecu.edu.au/cgi/viewcontent.cgi?article=2497&context=ecuworks
- http://personal.ph.surrey.ac.uk/~phs1pr/mphys-dissertations/2007/Wallis.pdf
- http://tmi.yokogawa.com/files/uploaded/BU510_EN_050.pdf
- http://repository.tudelft.nl/islandora/object/uuid:5ea21702-d6fb-484c-8fbf-15c5b8563ff1/datastream/OBJ/download

## Foot Notes
LED light source is preferred over tungsten filament based source as it has the below-stated superiority:
1. Life
2. Heat/IR – minimum
3. Multitude of Wavelengths generated
4. Precise selection of Wavelength & Intensity
