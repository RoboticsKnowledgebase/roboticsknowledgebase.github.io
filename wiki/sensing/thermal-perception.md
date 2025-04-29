---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2025-04-29 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Perception via Thermal Imaging
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

In this article, we discuss strategies to implement key steps in a robotic perception pipeline using thermal cameras.
Specifically, we discuss the conditions under which a thermal camera provides more utility than an RGB camera, followed
by implementation details to perform camera calibration, dene depth estimation and odometry using thermal cameras.

## Why Thermal Cameras?

Thermal cameras are useful in key situations where normal RGB cameras fail - notably, in perceptual degradation like
smoke and darkness.
Furthermore, unlike LiDAR and RADAR, thermal cameras do not emit any detectable radiation.
If your robot is expected to operate in darkness and smoke-filled areas, thermal cameras are a means for your robot to
perceive the environment in nearly the same way as visual cameras would in ideal conditions.