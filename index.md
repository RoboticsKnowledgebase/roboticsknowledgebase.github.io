---
# You don't need to edit this file, it's empty on purpose.
# Edit theme's home layout instead if you wanna make some changes
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
title: Welcome to the Robotics Knowledgebase
layout: splash
header:
  caption: Photo by Ricardo Gomez Angel on [Unsplash](https://unsplash.com)
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/ricardo-gomez-angel-162935.jpg
  cta_label: "Start Learning"
  cta_url: "http://roboticsknowledgebase.com/wiki/"
excerpt: "The Robotics Knowledgebase exists to advance knowledge in the robotics discipline."
---
{% include google-search.html %}


## Table of Contents
{% assign url_parts = page.url | split: '/' %}
{% assign url_parts_size = url_parts | size %}
{% assign rm = url_parts | last %}
{% assign base_url = page.url | replace: rm %}

<ul>
{% for node in site.pages %}
  {% if node.url contains base_url %}
    {% assign node_url_parts = node.url | split: '/' %}
    {% assign node_url_parts_size = node_url_parts | size %}
    {% assign filename = node_url_parts | last %}
    {% if url_parts_size == node_url_parts_size and filename != 'index.html' %}
      <li><a href='{{node.url}}'>{{node.title}}</a></li>
    {% endif %}
  {% endif %}
{% endfor %}
</ul>


Overview	of	Intelligent,	Electrical-Mechanical	Systems
- The	CMU	definition	of	robotics
Common	Platforms
- Hardware
  - Drones
  - Ground	Vehicles
  - Underwater
- Software
  - ROS
  - Orocos	RTT
  - Multi-threaded	C++
System	Design and Development
- System	Engineering
  - Requirements	writing
  - Test	Design
- Mechanical
  - Mechanical Design
  - Mechanical Computer Aided Design
  - Fabrication Techniques
- Electrical
  - Electrical Design
  - Electrical Computer Aided Design
  - Printed Circuit Boards
- Software	Engineering
  - Programming Embedded Systems
  - Data Structures
- System Integration
Subsystems
- Mechatronics
  - Sensors
  - Actuators
  - Controls
- Mobility
- Manipulation
- Machine	Learning
  - Classification
  - Regression
  - Feature	Generation
  - Training	and	Testing
- Computer	Vision
- State Estimation
  - Localization and	Mapping
- Robot	Planning
  - Motion
  - Path
Math
- Linear Algebra
- Probability
Project	Management
- Methods
- Tools
- Team	Communication
- Budgeting
- Ordering

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/smbryan/smbryan.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
