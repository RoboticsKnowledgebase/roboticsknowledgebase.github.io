---
title: Wiki Index
---
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
