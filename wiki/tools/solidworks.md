---
title: Getting started with Solidworks
---

[SolidWorks](https://www.solidworks.com/) is a 3D Computer Aided Design program commonly used for mechanical design. The following guide introduces a tutorial series from SolidWorks, in addition to offering some basic tips for setting up your environment.

## Tutorial Series
[This tutorial series](https://www.solidworks.com/sw/resources/solidworks-tutorials.htm) guides through the steps of making a simple part to an assembly and finally a drawing. Simulation and sustainability are advanced topics which are not needed for designing basic parts or mounts, not designed for heavy load bearing.

## General Guidelines:
A basic knowledge of Solidworks is needed to understand these guidelines.

1) To hide or show all dimensions, go to feature manager, Right click on "Annotations" and highlight "Hide all Feature Dimensions"

2) With multiple drawings up, to switch between them without minimizing and maximizing press "Ctrl Tab", this will cycle through the open drawings.

3) Smart Mating: To smart mate two parts, click on the smart mate icon, then DOUBLE CLICK on the part you want to mate, it will become transparent, then drag it to the part you want to mate to.

4) When sweeping a complex shape use the sweep path if it falls on what will be the outside of your part to cut away excess material before sweeping the profile.

5) Use symmetry about part origin when possible.

6) Use a rational order of operations, especially if other people may be using your parts. Reasons for transgressing this rule would be to accomplish a workaround that because of a bug or limitation cannot be done a better way.

7) Use of sketch relations rather than explicit dimensions (for example to center a hole on a block, use a midpoint relation to a diagonal) when the "design intent" allows.

8) Fully defined sketches behave better than under defined, and defined using relations which are relative to existing geometry are better than explicit dimensions, especially when changes are expected.

9) In order to rotate portions of a sketch in relationship to the origin and/or other portions of a sketch you can use either constraining relationships and a dimension or a circular repeat pattern.

10) Use faces for mating relationships in an assembly over edges and points when possible. Faces will give you a more stable position for part.

11) When mating objects with little or no flat edges use reference planes for mating the part to the assembly.

12) When applying profiles to edges to be swept, when positioning the profile get it as close to the point where you want to place it before adding relations to nail the position down.

13) When mating and the part mates to the wrong side or wrong direction, click the anti align to get it to mate the way you want it.

14) To have drawing views with suppressed parts you must specify a different configuration to show the part(s) suppressed and un-suppressed and then import the different named views into the drawing.

15) To find out what mates, dependencies, etc. that are associated with your features right click on the top level assembly and click on view dependencies in the menu. This will then organize all of your mates and other dependencies under which feature it is associated with.

16) When something in a drawing or part will not rebuild, even after making changes, press CTRL-Q at the same time to force SolidWorks to rebuild everything in the model or drawing.

17) Create and analyse the drawing of the assembly, with sectional views and all dimensions, before 3D printing or laser cutting anything.
