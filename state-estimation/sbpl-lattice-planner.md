---
title: SBPL Lattice Planner
---
This tutorial covers implementing the Search Based Planning Lab's Lattice Planner in ROS indigo

## What is the problem to install SBPL_lattice_planner?
When you try the official way to install sbpl_lattice_planner, you'll get some below error because navigation_experimental package which include sbpl_lattice_planner is not being maintained in the ROS Indigo version and has some compatibility problem with recent version.
```
git clone https://github.com/ros-planning/navigation_experimental.git

-- +++ processing catkin package: 'assisted_teleop'
-- ==> add_subdirectory(navigation_experimental/assisted_teleop)
-- Using these message generators: gencpp;genlisp;genpy
CMake Error at navigation_experimental/assisted_teleop/CMakeLists.txt:25 (find_package):
  By not providing "FindEigen.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "Eigen", but
  CMake did not find one.
```

## How to resolve the build error
So, we need other way to get around this error, there is other sbpl_lattice_planner maintained by Technische University Darmstadt.
```
git clone https://github.com/tu-darmstadt-ros-pkg/sbpl_lattice_planner.git
```
Catkin_make would succeed to build the sbpl_lattice_planner. If not, there is some package dependency problem with regards to the SBPL, you can install SBPL library by apt-get.
```
sudo apt-get install ros-indigo-sbpl
```

## Furthure Reading
1. Navigation Experimental Git: https://github.com/ros-planning/navigation_experimental
2. SBPL_lattice_planner in `tu-darmstadt-ros-pkg` Git: https://github.com/tu-darmstadt-ros-pkg/sbpl_lattice_planner
