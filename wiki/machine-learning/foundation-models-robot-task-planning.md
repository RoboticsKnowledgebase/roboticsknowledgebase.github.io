---
date: 2026-04-30
title: Foundation Models (LLMs/VLMs) for Robot Task Planning
---
Foundation models, including large language models (LLMs) and vision-language models (VLMs), have rapidly become a transformative force in robotics. By leveraging the broad world knowledge, common-sense reasoning, and generalization capabilities acquired during internet-scale pre-training, these models enable robots to interpret open-ended natural language instructions, reason about scenes, and compose long-horizon task plans without task-specific reward engineering. This article surveys the key approaches for integrating foundation models into robotic systems, covering affordance-grounded planners like SayCan, code-generation frameworks such as Code as Policies, vision-language-action models like RT-2, and 3D value-map composition with VoxPoser. Practical implementation patterns, code examples with ROS 2, and best practices for safe deployment are also discussed.

## Why Foundation Models Matter for Robotics

Traditional robot task planning relies on manually specified symbolic planners (PDDL, behavior trees) or task-specific learned policies. These approaches struggle with open-vocabulary instructions, novel objects, and long-horizon reasoning. Foundation models address these limitations by providing:

- **Broad world knowledge**: Understanding of object properties, spatial relationships, and common-sense physics acquired from large-scale text and image data.
- **Language grounding**: The ability to interpret free-form natural language commands and map them to structured action sequences.
- **Zero-shot generalization**: Handling novel objects and task descriptions without retraining.
- **Compositional reasoning**: Breaking complex instructions into ordered subtask sequences through chain-of-thought prompting.

The general paradigm is to use a foundation model as a high-level semantic planner that proposes candidate actions or code, while low-level controllers (motion planners, PID controllers, learned visuomotor policies) handle execution. This separation of concerns keeps the system modular and allows each layer to be improved independently.

## Key Approaches

### SayCan: Grounding Language in Robot Affordances

SayCan, introduced by Ahn et al. (2022), addresses a fundamental problem: LLMs can suggest semantically reasonable actions, but they have no knowledge of what the robot can actually do in its current environment. SayCan combines the "say" (LLM knowledge of useful actions) with the "can" (learned affordance functions that score the feasibility of each action given the current state).

The scoring mechanism works as follows. Given a natural language instruction $i$ and a set of candidate skills $\{a_1, a_2, \ldots, a_n\}$, SayCan selects the next action by:

$$a^* = \arg\max_{a_k} \; p_{\text{LLM}}(a_k \mid i, h) \cdot p_{\text{afford}}(a_k \mid s)$$

where:
- $p_{\text{LLM}}(a_k \mid i, h)$ is the LLM's probability that skill $a_k$ is a useful next step given instruction $i$ and history $h$
- $p_{\text{afford}}(a_k \mid s)$ is the affordance score from a learned value function indicating how likely $a_k$ is to succeed in the current state $s$

The affordance functions are typically trained via reinforcement learning on real robot data. Each primitive skill (e.g., "pick up the sponge", "go to the counter") has an associated value function that provides a success probability. This product ensures that the selected action is both semantically relevant and physically executable.

#### Strengths and Limitations

SayCan works well for table-top manipulation and mobile manipulation with a fixed skill library. However, it requires pre-trained affordance functions for every skill, making it difficult to scale to new capabilities. The discrete skill set also limits the expressiveness of plans.

### Code as Policies and ProgPrompt

Rather than selecting from a fixed skill library, Code as Policies (CaP), proposed by Liang et al. (2023), prompts an LLM to directly generate executable Python code that calls robot perception and control APIs. The key insight is that code is a more expressive and composable representation for robot behavior than flat action sequences.

A typical prompt provides the LLM with:
1. Available API functions (e.g., `pick(obj)`, `place(obj, location)`, `get_obj_pos(name)`)
2. Perception query functions (e.g., `detect_objects()`, `get_color(obj)`)
3. A few in-context examples mapping instructions to code
4. The current instruction

```python
# Example: Code as Policies prompt structure
SYSTEM_PROMPT = """You are a robot policy generator. You have access to the
following functions:

# Perception
detect_objects() -> list[str]       # returns names of visible objects
get_obj_pos(name: str) -> tuple     # returns (x, y, z) position
get_obj_color(name: str) -> str     # returns color string

# Actions
pick(name: str) -> bool             # grasp the named object
place(name: str, pos: tuple) -> bool # place held object at position
move_to(pos: tuple) -> bool         # move end-effector to position
say(text: str) -> None              # speak text aloud

Write Python code to accomplish the user's instruction.
Use only the functions above. Add error handling."""

# Example in-context demonstration
EXAMPLE = """
# Instruction: "Put the red block on top of the blue block"
objects = detect_objects()
red_block = [o for o in objects if 'red' in get_obj_color(o) and 'block' in o][0]
blue_block = [o for o in objects if 'blue' in get_obj_color(o) and 'block' in o][0]
blue_pos = get_obj_pos(blue_block)
pick(red_block)
# Stack on top: offset z by block height
place(red_block, (blue_pos[0], blue_pos[1], blue_pos[2] + 0.05))
"""
```

ProgPrompt (Singh et al., 2023) extends this idea by generating Pythonic programs with explicit assertions and precondition checks, making the generated plans more robust to execution failures.

#### Strengths and Limitations

Code generation is highly flexible and compositional — loops, conditionals, and arithmetic come for free. However, it introduces risks: LLM-generated code can contain bugs, call undefined functions, or produce unsafe motions. Sandboxing and validation are essential.

### RT-2 and RT-X: Vision-Language-Action Models

RT-2 (Brohan et al., 2023) takes a fundamentally different approach by fine-tuning a VLM end-to-end to directly output robot actions. Rather than using a foundation model as a planner that calls external controllers, RT-2 treats action generation as a sequence modeling problem. Robot actions are tokenized as text strings (e.g., discretized into bins), and the model is trained on both internet-scale vision-language data and robot demonstration data.

The architecture builds on PaLI-X or PaLM-E as the backbone VLM. Given an image observation $o_t$ and a language instruction $l$, RT-2 outputs action tokens:

$$a_t = \text{VLM}(o_t, l)$$

where $a_t$ is a discretized action vector including end-effector displacement, rotation, and gripper state.

RT-X extends this concept across multiple robot embodiments. The Open X-Embodiment dataset aggregates demonstration data from over 20 different robot platforms. Cross-embodiment training produces policies that transfer better to new robots and tasks than single-embodiment training.

#### Strengths and Limitations

VLA models offer the tightest integration between perception, language understanding, and action — everything is in one forward pass. However, they require large-scale robot demonstration data for fine-tuning, inference latency can be high (hundreds of milliseconds per action), and the monolithic architecture is harder to debug than modular systems.

### VoxPoser: 3D Value Maps from VLMs

VoxPoser (Huang et al., 2023) uses LLMs and VLMs to compose 3D affordance and constraint maps in the robot's workspace, which are then used by a motion planner. Given an instruction like "pour water into the cup without spilling", VoxPoser:

1. Uses an LLM to decompose the instruction into spatial objectives and constraints
2. Queries a VLM (or open-vocabulary detector) to localize relevant objects in 3D
3. Composes a voxel value map where high values indicate goal regions and low values indicate obstacle or constraint regions
4. Feeds the value map to a motion planner (e.g., MPC or trajectory optimization) as a cost function

This approach is zero-shot — it does not require any robot demonstration data. The 3D value maps serve as an interface between the semantic understanding of foundation models and the geometric reasoning of classical planners.

### CLIP and Open-Vocabulary Detection

CLIP (Contrastive Language-Image Pretraining) and its derivatives (OWL-ViT, Grounding DINO, GLIP) enable robots to detect and localize objects described by arbitrary text queries, eliminating the need for fixed object categories.

In a robotics pipeline, open-vocabulary detection typically serves as the perception backbone:

```python
# Using Grounding DINO for open-vocabulary object detection
from groundingdino.util.inference import load_model, predict

# Load the pre-trained Grounding DINO model
model = load_model("groundingdino/config/GroundingDINO_SwinT.py",
                    "weights/groundingdino_swint_ogc.pth")

# Detect objects matching a text query in the camera image
def detect_objects(image, text_query, box_threshold=0.3, text_threshold=0.25):
    """Detect objects in image matching the natural language query.

    Args:
        image: RGB image from robot camera
        text_query: natural language description (e.g., "red mug. blue plate.")
        box_threshold: confidence threshold for bounding boxes
        text_threshold: confidence threshold for text matching

    Returns:
        boxes: detected bounding boxes in xyxy format
        phrases: matched text phrases for each detection
    """
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_query,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    return boxes, phrases
```

Open-vocabulary detectors are often combined with depth cameras to produce 3D object poses, which are then fed to grasp planners or the foundation model planner.

## Architecture Patterns

### Hierarchical Planning: Foundation Model + Low-Level Controller

The most common architecture separates high-level semantic planning from low-level control:

```
┌─────────────────────────────────────────────┐
│         Natural Language Instruction         │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│   Foundation Model (LLM / VLM)              │
│   - Task decomposition                      │
│   - Subtask sequencing                      │
│   - Code or plan generation                 │
└──────────────────┬──────────────────────────┘
                   │  Subtask / Code / Action
┌──────────────────▼──────────────────────────┐
│   Skill Library / Low-Level Controllers     │
│   - MoveIt for motion planning              │
│   - Diffusion policy for manipulation       │
│   - PID / MPC for locomotion                │
└──────────────────┬──────────────────────────┘
                   │  Joint commands
┌──────────────────▼──────────────────────────┐
│              Robot Hardware                  │
└─────────────────────────────────────────────┘
```

This pattern keeps the foundation model in the loop for high-level decisions while relying on well-tested controllers for safe physical execution. It also allows swapping foundation models without changing the control stack.

### Prompt Engineering for Robotics

Effective prompting for robot task planning differs from general LLM prompting:

1. **Define the action space explicitly**: List every available function with type signatures and brief descriptions. The LLM cannot infer capabilities it was not told about.
2. **Provide physical constraints**: Include workspace bounds, payload limits, and collision constraints in the system prompt.
3. **Use structured output formats**: Request JSON or Python code rather than free-form text to simplify parsing.
4. **Include failure recovery**: Prompt the model to include try/except blocks or precondition checks.
5. **Chain-of-thought for sequencing**: Ask the model to first list the steps in natural language, then generate code for each step.

### Vision-Language Grounding Pipeline

For tasks requiring visual understanding, a common pipeline is:

1. Capture RGB-D image from robot camera
2. Run open-vocabulary detector (Grounding DINO, OWL-ViT) to localize mentioned objects
3. Project detections to 3D using depth data and camera intrinsics
4. Pass object names, positions, and spatial relations to the LLM planner
5. LLM generates an action plan referencing objects by name
6. Execute plan using low-level controllers with 3D target positions

## Practical Implementation: LLM Task Planner with ROS 2

The following example demonstrates a minimal LLM-based task planner integrated with ROS 2. The node receives a natural language instruction, queries an LLM to generate a structured plan, and publishes action goals.

```python
#!/usr/bin/env python3
"""ROS 2 node that uses an LLM to generate pick-and-place task plans."""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from openai import OpenAI


class LLMTaskPlanner(Node):
    def __init__(self):
        super().__init__('llm_task_planner')

        # Subscribe to natural language commands
        self.cmd_sub = self.create_subscription(
            String, '/task_command', self.command_callback, 10)

        # Publisher for action goals (consumed by a MoveIt action server)
        self.goal_pub = self.create_publisher(
            PoseStamped, '/move_goal', 10)

        # Publisher for planner status feedback
        self.status_pub = self.create_publisher(
            String, '/planner_status', 10)

        # Initialize the LLM client
        self.llm_client = OpenAI()

        # Known object positions from perception (updated by detector node)
        self.object_positions = {}
        self.obj_sub = self.create_subscription(
            String, '/detected_objects', self.objects_callback, 10)

        self.get_logger().info('LLM Task Planner ready.')

    def objects_callback(self, msg):
        """Update known object positions from the perception pipeline."""
        self.object_positions = json.loads(msg.data)

    def command_callback(self, msg):
        """Handle incoming natural language task commands."""
        instruction = msg.data
        self.get_logger().info(f'Received instruction: {instruction}')

        # Build the prompt with current scene context
        plan = self.generate_plan(instruction)
        if plan is None:
            self.publish_status('Planning failed: LLM returned no valid plan.')
            return

        # Execute the generated plan step by step
        self.execute_plan(plan)

    def generate_plan(self, instruction: str) -> list:
        """Query the LLM to produce a structured task plan.

        Returns a list of action dicts, e.g.:
        [{"action": "pick", "object": "red_cup"},
         {"action": "place", "object": "red_cup", "target": [0.5, 0.2, 0.1]}]
        """
        # Format known objects into a scene description
        scene = json.dumps(self.object_positions, indent=2)

        system_prompt = f"""You are a robot task planner. The robot has a
single arm with a parallel-jaw gripper. It can perform these actions:

- pick(object_name): grasp the named object at its known position
- place(object_name, [x, y, z]): place the held object at the target
- move_to([x, y, z]): move gripper to a position without grasping

Current scene (object positions in meters):
{scene}

Workspace bounds: x=[0.0, 0.8], y=[-0.4, 0.4], z=[0.0, 0.5]

Output a JSON array of action steps. Each step has:
  "action": one of "pick", "place", "move_to"
  "object": object name (for pick/place)
  "target": [x, y, z] (for place/move_to)

Only use objects that appear in the current scene. Verify target
positions are within workspace bounds. Output ONLY valid JSON."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            plan_text = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if plan_text.startswith("```"):
                plan_text = plan_text.split("\n", 1)[1].rsplit("```", 1)[0]

            plan = json.loads(plan_text)
            self.get_logger().info(f'Generated plan with {len(plan)} steps.')
            return plan

        except Exception as e:
            self.get_logger().error(f'LLM planning failed: {e}')
            return None

    def execute_plan(self, plan: list):
        """Execute a structured plan by publishing goal poses."""
        for i, step in enumerate(plan):
            action = step.get("action")
            self.publish_status(f'Step {i+1}/{len(plan)}: {action}')

            if action == "pick":
                obj_name = step["object"]
                if obj_name not in self.object_positions:
                    self.publish_status(f'Object {obj_name} not found.')
                    return
                pos = self.object_positions[obj_name]
                self.publish_goal(pos)

            elif action in ("place", "move_to"):
                target = step.get("target")
                if target is None:
                    self.publish_status(f'No target for {action}.')
                    return
                self.publish_goal(target)

    def publish_goal(self, position: list):
        """Publish a PoseStamped goal for the motion planner."""
        goal = PoseStamped()
        goal.header.frame_id = "base_link"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(position[0])
        goal.pose.position.y = float(position[1])
        goal.pose.position.z = float(position[2])
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)

    def publish_status(self, text: str):
        """Publish a status message for monitoring."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = LLMTaskPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

To use this node:

1. Launch your robot's perception pipeline to populate the `/detected_objects` topic with a JSON dictionary of object names and `[x, y, z]` positions.
2. Start the planner: `ros2 run your_package llm_task_planner`
3. Send a command: `ros2 topic pub /task_command std_msgs/String "data: 'pick up the red mug and place it on the shelf'"`

The planner queries the LLM, generates a structured JSON plan, and publishes sequential goal poses for a downstream motion planner (e.g., MoveIt 2) to execute.

## Challenges

### Latency

Foundation model inference introduces significant latency. A single GPT-4o API call takes 500ms–3s depending on prompt length and output size. For real-time control at 10–100 Hz, this is prohibitive. The standard mitigation is to use the foundation model only for high-level planning (called once per task or subtask) and rely on fast low-level controllers for real-time execution.

On-device models (e.g., quantized LLaMA variants on NVIDIA Jetson) can reduce latency to 100–300ms per query but sacrifice capability. Recent work on speculative decoding and KV-cache optimization is closing this gap.

### Hallucination and Correctness

LLMs may generate plans that reference non-existent objects, call undefined APIs, violate physical constraints, or skip critical safety steps. Common mitigations include:

- **Grounding with perception**: Verify all referenced objects exist in the current scene before execution.
- **Affordance scoring**: Use learned value functions (as in SayCan) to filter infeasible actions.
- **Code validation**: Parse and statically analyze generated code before execution. Check that all function calls are in the allowed API set.
- **Plan verification**: Use a secondary model or rule-based checker to validate the plan against known constraints.

### Safety

Executing LLM-generated actions on physical hardware introduces safety risks. A malformed plan could cause collisions, drop objects, or damage the robot. Essential safeguards include:

- **Workspace bounds checking**: Reject any target position outside the robot's safe workspace.
- **Collision checking**: Run generated trajectories through a collision checker (e.g., MoveIt's planning scene) before execution.
- **Human-in-the-loop confirmation**: For high-stakes actions, display the plan and require human approval before execution.
- **Emergency stop integration**: Ensure the system respects hardware e-stop signals at all times, independent of the planner.

### Compute Requirements

VLA models like RT-2 require powerful GPUs (A100-class or higher) for real-time inference. Deploying these on mobile robots with limited compute budgets is challenging. Edge deployment strategies include model distillation, quantization (INT8/INT4), and offloading inference to a cloud server connected via low-latency networking.

### Sim-to-Real Gap for Learned Representations

Foundation models trained on internet data may have representation biases that do not align with the robot's sensory modality. For example, CLIP embeddings trained on web images may not generalize well to low-resolution depth images from a wrist-mounted camera. Domain adaptation, fine-tuning on robot-specific data, or using robot-specialized VLMs (e.g., SpatialVLM, RoboVLM) can mitigate this gap.

## Best Practices

### Prompt Design

- **Be explicit about the action space**: List every available function with types and descriptions. Omit nothing the model needs.
- **Include physical constraints**: Workspace bounds, payload limits, reachability constraints, and collision objects should be in the prompt.
- **Provide diverse examples**: Include 3–5 in-context demonstrations covering different task types (sorting, stacking, tool use).
- **Request structured output**: JSON or Python code is far easier to parse and validate than free-form text.
- **Use chain-of-thought**: Prompt the model to reason step by step before generating the final plan. This improves accuracy on complex, multi-step tasks.

### Fallback Strategies

- **Retry with rephrasing**: If the LLM returns an invalid plan, retry with a more constrained prompt that narrows the output format.
- **Graceful degradation**: If planning fails after retries, fall back to a pre-defined behavior tree or a safe home position. See the [Behavior Trees](/wiki/planning/behavior-tree/) article for implementing fallback behaviors.
- **Incremental execution**: Execute one subtask at a time, re-querying the LLM after each step with updated scene state. This allows mid-plan correction.

### Human-in-the-Loop Verification

For deployment outside controlled lab settings, consider a confirmation loop:

1. LLM generates a plan.
2. The plan is displayed to a human operator (on a screen or via speech).
3. The operator approves, modifies, or rejects the plan.
4. Only approved plans are executed.

This pattern is especially important during the early stages of deployment when trust in the system has not yet been established through extensive testing.

### Evaluation and Testing

- **Unit test the prompt**: Create a test suite of instruction-scene pairs with expected plan outputs. Run these against the LLM and check correctness.
- **Simulation testing**: Execute generated plans in simulation (e.g., Isaac Sim, Gazebo) before deploying on hardware.
- **Log everything**: Record all instructions, generated plans, execution traces, and outcomes for post-hoc analysis and prompt refinement.

## Summary

Foundation models have opened a new paradigm for robot task planning, enabling systems that understand open-ended natural language instructions and reason about novel objects and scenes. The key architectural insight is to use foundation models for high-level semantic reasoning while relying on established low-level controllers for safe physical execution. Approaches range from affordance-grounded planning (SayCan) to code generation (Code as Policies) to end-to-end vision-language-action models (RT-2). Each offers different tradeoffs between flexibility, data requirements, and integration complexity. Successful deployment requires careful attention to safety, latency, hallucination mitigation, and human oversight. As foundation models continue to improve in speed and reliability, their role in robotics will expand from research prototypes to production systems.

## See Also:
- [NLP for Robotics](/wiki/machine-learning/nlp-for-robotics/) - Background on transformer models and natural language processing for robotic systems
- [Introduction to Diffusion Models and Diffusion Policy](/wiki/machine-learning/intro-to-diffusion/) - Diffusion-based visuomotor policies used as low-level controllers in foundation model architectures
- [Imitation Learning With a Focus on Humanoids](/wiki/machine-learning/imitation-learning/) - Data collection and policy training for humanoid robots, including NVIDIA GR00T's System-1/System-2 architecture
- [Behavior Trees](/wiki/planning/behavior-tree/) - Structured fallback behaviors for when foundation model planning fails

## Further Reading
- [Google DeepMind RT-2 Blog Post](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) - Overview of the RT-2 vision-language-action model with demonstration videos
- [Open X-Embodiment Project](https://robotics-transformer-x.github.io/) - Cross-embodiment robot learning dataset and models spanning 20+ robot platforms
- [Code as Policies Project Page](https://code-as-policies.github.io/) - Interactive examples of LLM-generated robot policies with video demonstrations
- [VoxPoser Project Page](https://voxposer.github.io/) - Zero-shot 3D value map composition from vision-language models for manipulation
- [Grounding DINO GitHub Repository](https://github.com/IDEA-Research/GroundingDINO) - Open-vocabulary object detection model commonly used in robotic perception pipelines

## References
- M. Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," in *Proc. Conference on Robot Learning (CoRL)*, 2022.
- A. Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," in *Proc. Conference on Robot Learning (CoRL)*, 2023.
- S. Huang et al., "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models," in *Proc. Conference on Robot Learning (CoRL)*, 2023.
- J. Liang et al., "Code as Policies: Language Model Programs for Embodied Control," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2023.
- A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. International Conference on Machine Learning (ICML)*, 2021.
- I. Singh et al., "ProgPrompt: Generating Situated Robot Task Plans Using Large Language Models," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2023.
- Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2024.
