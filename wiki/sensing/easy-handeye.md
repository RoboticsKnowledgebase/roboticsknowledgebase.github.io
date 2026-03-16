---
date: 2024-05-01
title: Hand-Eye Calibration with easy_handeye
---

*[Easy Handeye](https://github.com/IFL-CAMP/easy_handeye)* from the TUM Computer Aided Medical Procedures Lab is an extremely useful tool for calibrating a camera with an arm. There are two types of hand-eye calibration:

- **eye-in-hand** - The camera is mounted on the arm end-effector and it is necessary to determine the static transform between the camera frame and the end-effector.
- **eye-on-base** - The camera is mounted in the workspace of the arm and it is necessary to determine the static transform between the camera frame and the base of the arm.

## Procedure
Print a marker from [ArUco Marker Generator](https://chev.me/arucogen/), be sure to select 'Original ArUco' for the dictionary. Measure the edge length of the marker in meters and record for later. For eye-in-hand calibration, the ArUco marker should be affixed to a static object. For eye-on-base calibration, the ArUco marker should be affixed to the end-effector.

Install aruco ros

```
sudo apt-get install ros-noetic-aruco-ros
```

Clone the easy_handeye repository into a new or existing ROS workspace and make the workspace.

```
cd catkin_ws/src
git clone git@github.com:IFL-CAMP/easy_handeye.git
cd ../ && catkin_make
```

Select one of the [example launch files](https://github.com/IFL-CAMP/easy_handeye/tree/master/docs/example_launch), move it to `easy_handeye/easy_handeye/launch`. This launch file launches the camera, robotic arm, aruco detector, and handeye calibration tool.

If you cannot find a launch file that matches your camera arm configuraiton, you will have to modify an existing one.

Within the launch file, replace the camera and arm launch commands with the launch commands corresponding to your system.

Within the launch file set the following parameters:
- `eye_on_hand` - True for eye-on-hand, False for eye-on-base
- `marker_size` - The size of your ArUco marker in meters
- `marker_id` - The ID of your ArUco marker
- `tracking_base_frame` - The RGB frame of your camera
- `tracking_marker_frame`- The frame of the ArUco marker detection
- `robot_base_frame`- The frame of the base of the robotic arm
- `robot_effector_frame` - The frame of the end effector of the robotic arm

Once all parameters are changed, run the launch file.

```
roslaunch easy_handeye <YOUR_LAUNCH_FILE>
```

Move the arm in manual mode to an angle where the camera can see the aruco marker and hit capture. Repeat this 10-20 times for different arm configurations then hit compute. Try to get diverse arm positions. This will give you the translation and rotation of the camera relative to the base or end effector depending on which calibration being performed.

One thing to note is that sometimes this calibration needs to be adjusted manually by hand. If possible, verify the calibration in different arm configurations and adjust the translation and rotation if necessary.

## References
- [Easy Handeye](https://github.com/IFL-CAMP/easy_handeye)