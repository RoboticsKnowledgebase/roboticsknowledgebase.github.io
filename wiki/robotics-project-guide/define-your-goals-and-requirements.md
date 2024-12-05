---
# layout: single
title: Define your goals and requirements
mermaid: true
---

Embarking on a robotics project begins with a clear understanding of your objectives. Defining your goals and requirements is crucial as it sets the foundation for all subsequent decisions, from choosing the hardware to selecting the programming language. This section will guide you through the essential considerations to ensure your project is well-planned and aligned with your vision.

# Establish the Purpose of Your Robot
Start by clarifying the primary function of your robot:

- What problem is your robot intended to solve?
- Who are the end-users?
- In what environment will it operate?

Example: For our quadrupedal robot Tod, the goal is to assist elderly shoppers by carrying their products in a shopping center. The end-users are senior citizens who may have difficulty carrying heavy items, and the environment is a bustling retail space with various obstacles.

# Define Functional Requirements
Functional requirements detail what your robot must be able to do:

- Mobility: Should it walk, roll, or fly? Does it need to navigate stairs or uneven surfaces?
- Payload Capacity: How much weight does it need to carry?
- Navigation: Will it operate autonomously or require human control?
- Interaction: Does it need to communicate with users or other systems?

Example: Tod must:
- Walk steadily on smooth surfaces.
- Carry up to 20 kg of goods.
- Navigate autonomously through the shopping center.
- Follow designated paths and avoid obstacles.
- Respond to basic user commands.

# Set Performance Requirements
These are measurable criteria to evaluate your robot's effectiveness:

- Speed: How fast should it move?
- Battery Life: How long should it operate before recharging?
- Precision: What level of accuracy is needed in its movements?
- Reliability: What is the acceptable failure rate?

Example: Tod should:
- Move at a speed of up to 1 m/s to match walking speeds.
- Operate for at least 4 hours to cover a full shopping day.
- Navigate with an accuracy of ±10 cm to prevent collisions.
- Have a failure rate of less than 1% over its operational lifetime.

# Identify Constraints and Limitations
Understanding limitations helps in realistic planning:
- Budget: What is the maximum amount you can spend?
- Timeframe: What are your deadlines?
- Resources: What equipment and expertise are available?
- Regulations: Are there safety or compliance standards to meet?

Example: Constraints for Tod include:
 - A budget of $50,000.
 - Completion within 12 months.
 - Limited access to advanced fabrication tools.
 - Compliance with public safety regulations for robots.

# Prioritize Requirements
Not all requirements are equal—prioritize them:

- Must-Have: Essential for basic operation.
- Should-Have: Important but not critical.
- Nice-to-Have: Enhancements that can be deferred.

Example:
- Must-Have: Autonomous navigation, obstacle avoidance, 20 kg payload.
- Should-Have: User-friendly interface, voice command recognition.
- Nice-to-Have: Real-time mapping, advanced AI features.

6. Create a Requirements Document
Documenting ensures clarity and serves as a reference:
- Executive Summary: Outline the project's purpose.
- Detailed Requirements: List all functional and performance criteria.
- Constraints: Note all limitations.
- Assumptions: State any conditions assumed during planning.
- Acceptance Criteria: Define how success will be measured.

7. Visualize the Workflow
Understanding how your progress impact later work can be helpful. Here's a flowchart illustrating the process:




8. Analyze Impact on Subsequent Steps
Defining your goals early affects all later stages:

Choosing a Robot: Your payload and mobility needs will influence whether you select a ready-made platform like Unitree Go1 or design a custom robot (see Choose a Robot).
Peripheral Hardware: Requirements for navigation and interaction will determine the sensors and actuators needed (see Find Out What Peripheral Hardware You Need).
Programming Language: Performance needs might lead you to choose C++ for speed, while complex algorithms or machine learning applications might favor Python (see Choose Your Language).
Communication Method: Depending on real-time requirements and system complexity, you might opt for ROS or ROS2 (see Choose Your Communication Method).
Simulation: The need for testing in a virtual environment will guide your choice of simulator tools (see Choose Your Simulator).

9. Decision-Making Aids
Utilize decision trees or tables to weigh your options:

mermaid
Copy code
graph TD;
    A[Project Goals] --> B{Payload Requirement?};
    B -->|Less than 5kg| C[Consider Smaller Robots];
    B -->|5kg - 20kg| D[Quadrupedal Robots like Unitree];
    B -->|More than 20kg| E[Custom-Built Robot Needed];

    D --> F{Need Advanced Features?};
    F -->|Yes| G[Look into ANYmal];
    F -->|No| H[Unitree Go1 or Go2];

10. Examples of Technical Considerations
Speed vs. Control: If your robot needs to react quickly to obstacles, a language like C++ might be preferred for its execution speed (we'll delve deeper in Choose Your Language).
Machine Learning Capabilities: For advanced features like voice recognition, Python offers extensive libraries and easier integration (more on this in Choose Your Language).
Real-Time Communication: If your robot requires real-time data processing, ROS2 provides improved performance over ROS1 (explored further in Choose Your Communication Method).
11. Prepare for Iteration
Remember that initial requirements might evolve. Be prepared to revisit and adjust your goals as you progress.

12. Next Steps
With a solid foundation of goals and requirements, you're ready to proceed to the next phase:

Choose a Robot: Select a robot platform that aligns with your defined needs.
Find Out What Peripheral Hardware You Need: Identify sensors and actuators essential for your robot's functionality.
Choose Your Language: Decide on the programming language that best suits your project's requirements.
By thoroughly defining your goals and requirements, you set yourself up for a smoother development process, minimizing unexpected challenges and ensuring that each subsequent decision is informed and purposeful.