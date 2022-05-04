---
date: {}
title: Thermal Cameras
published: true
---
## Types of Thermal Cameras
Types of Thermal Cameras (classified on the basis of operation):

### 1. Cooled Cameras
- #### Working
  - They have an internal cryogenic cooler that only cools the sensor to temperatures as low as 77° Kelvin (-196°C or -321°F). This dramatically increases the sensitivity of the Cooled Thermal cameras allowing them to see day and night at longer ranges than uncooled cameras as their greater sensitivity.
  - Most cooled thermal cameras are InSb and have to be cooled continuously

- #### Use cases
  - When very good object detection is needed at very long range
  - When you have a lot of money
  - When weight and power constraints are forgiving

### 2. Uncooled Cameras
- #### Working
  - Uncooled cameras are based on VoX infrared sensor and are often uncooled 
  - Due to continuous operation and increase in temperature of the focal plane array, there is a drift in the electrical properties of the sensor elements.
  - This requires compensation/correction which can be done in two ways: one-point and two-point Non Uniformity Correction. This is covered in greater detail later in this wiki.

- #### Use cases
  - For most projects with reasonable budgets
  - For use with UAVs where weight and power budgets are tight

- #### Our experiences
  - IR crossover
    - Difficulty in segmenting out hot objects due to a phenomenon called IR crossover.
    - This is basically a situation outdoors during some times of the day where the sunlight scattered by the environment washes out the output of the camera.
  - NUC failure
    - Persistent ghosting effect seen in the camera output.
    - This was due to mechanical damage to the camera NUC apparatus (see NUC section for more information)
  - Intermittent NUC also intermittently drops the output frame rate which may not be acceptable depending on the use case.

### 3. Radiometric Cameras
- #### Working 
  - A radiometric thermal camera measures the temperature of a surface by interpreting the intensity of an infrared signal reaching the camera. .
  - Reports pixel-wise temperature values for entire image captured

- #### Use cases
  - When absolute temperature of objects is needed
  - To ease implementation of temperature based segmentation instead of using more complicated algorithms on uncooled camera images
  - Especially useful when there is a high contrast between temperatures of objects.
  - Not as useful to try and segment humans in an indoor environment where temperature difference can’t be used reliably for segmentation

- #### Our experiences
  - We found this type of camera very helpful in combatting the IR crossover phenomenon seen outdoors during daytime testing.
  - Also helped us side-step any issues due to NUC
  - We ended up using the Seek Thermal camera for this purpose (refer to resources section for relevant links to datasheet and SDK)

## Uncooled Thermal Camera NUC

### What is NUC?

**Non-uniformity correction (NUC)** is a procedure in uncooled thermal cameras to compensate for detector drift that occurs as the scene and environment change. Basically, the camera's own heat can interfere with its temperature readings. To improve accuracy, the camera measures the IR radiation from its own optics and then adjusts the image based on those readings. NUC adjusts gain and offset for each pixel, producing a higher quality, more accurate image.

There are two types of NUC calibration:
- **Two-point NUC**

  - A two point NUC is a means to capture the gain and drift of each of the pixel elements while looking at a simulated black-body. In this case, the entire image should have an output of zero intensity as it is looking at a black-body. Any deviation from this is stored as the offset for this pixel in a lookup table. This process is then performed over a range of operating temperatures and during operation, based on the ambient temperature, the offsets from the lookup tables are applied. This is usually a factory calibration routine performed by the vendor before sale.

  - This still does not suffice at times, with washed out images seen some times during operation. The one-point NUC is a means to help alleviate this. 


- **One-point NUC**

  - This procedure works by intermittently concealing the detector with a plane so that light does not fall on this. The plane is then assumed to be a black body and calibration is performed against this. This phenomenon is accompanied by a drop in frame rate as no images are captured during this time (In the case of the FLIR Boson, it also makes an audible “click” sound).
  - In case of the FLIR Boson, there is a pre-set temperature delta, which determines the when the NUC occurs. Every time the detector temperature changes by this amount, the NUC is initiated. NUC is also frequently referred to as FFC in FLIR documentation
  - The FLIR Boson app also allows control over the duration for which the NUC occurs, giving some control of the drop in frame rate.

## Debug tips and our experience with the FLIR Boson
- Quite a few of the cameras we picked up from inventory gave images with a persistent ghosting effect.
- This we later realized was due to mechanical damage due to impact/ the cameras being dropped previously.
- This caused the NUC shutter to prevent from engaging whenever the NUC routine as getting called, leading to the current scene in view being used as the template to be used as the sample black-body output to be compensated against.
- An update in the ghosting pattern coinciding with a click sound is a reliable symptom for this mode of failure.
- This failure can sometimes be intermittent based on how tight the fasteners on the detector stack are and the orientation of the camera.
- Some ways to lessen the extent of the problem is to use the FLIR Boson app and try adjusting the following settings:
  - Increase the temperature threshold for performing FFC.
    - This will just increase the temperature change interval between which the FFC is performed. In case you do not anticipate very great change in detector temperature (short term use, cool ambient temperatures etc) this might delay the FFC long enough to not occur during operation.
  - Disable FFC altogether
    - This will just never initiate the FFC process.
  - However, the **FLIR Boson performs one FFC on bootup regardless**. So, you will have to power on the camera with a lens cap or looking at a uniform low temperature featureless scene in either case.
- We also found the camera to be deficient in giving a very high contrast of fires/ hotspots in open areas in broad daylight due to the IR crossover phenomenon.

**NOTE:** The efficacy of all these methods must be determined experimentally based on your use cases and operating conditions. Please use the suggestions above only as a list of possible options to try out and not a prescriptive solution.

## Resources
- [Seek Thermal Camera Driver](https://github.com/howde-robotics/seek_driver)
- [Seek SDK](https://developer.thermal.com/support/home) (will require creating an account)
- [Seek Thermal Camera Datasheet](https://www.digikey.com/en/products/detail/seek-thermal/S304SP/10492240)
- [FLIR Boson App](https://www.flir.com/support/products/boson/#Downloads)

## Further Reading
- www.infinitioptics.com/glossary/cooled-thermal
- www.flir.com/discover/suas/uas-radiometric-temperature-measurements
- www.flir.com/discover/security/radiometric/the-benefits-and-challenges-of-radiometric-thermal-technology
- www.flir.com/discover/professional-tools/what-is-a-non-uniformity-correction-nuc