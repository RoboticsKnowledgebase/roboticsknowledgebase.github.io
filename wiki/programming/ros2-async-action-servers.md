---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2023-12-04 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: ROS2 Actions for Asynchronous Tasks
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

## Action Servers and Clients

Action servers are an intermediate option between ROS services and ROS publisher-subscriber models.

![](/assets/images/Action-SingleActionClient.gif)

- **Services**
  - Take input request and execute an action
  - Provide only one response to the client (possibly even before the task is completed)
- **Subscribers and Publishers**
  - Continuously look for incoming messages and take up more compute resources
  - Require two different unambiguous topics for the subscriber and publisher
- **Actions**
  - Like services they only start execution (in an execute_callback)
    - Can provide response on input acceptance (goal_handle.accept())
    - Can provide continuous feedback when the task is being executed (goal_handle.publish_feedback())
    - Can provide final success/failure completion response after the task has ended
  - All above communication is defined in a ```.action``` file

## Basic Skeleton of an Action Server and Client

The [ROS2 documentation](https://docs.ros.org/en/foxy/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html) has simple implementation shown below

### Action Server
```python
from action_tutorials_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci, # filename in action folder (Fibonacci.action)
            'fibonacci', # name which shows up when we do `ros2 action list`
            self.execute_callback # callback when runs when client requests server
            )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])
            self.get_logger().info('Feedback: {0}'.format(feedback_msg.partial_sequence))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result

def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    rclpy.spin(fibonacci_action_server)


if __name__ == '__main__':
    main()
```

### Action Client

```python
from action_tutorials_interfaces.action import Fibonacci

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))

```

## Handling Asynchronous Tasks

The above action servers perform simple tasks within a while loop and everything in the node
runs in the same thread. However, let's consider a specific example to understand 'asynchronosity'

### Example: Build an Action Server that commands a robot

Specifically:
- The action server takes inputs from a client
- the action server requests a downstream node (which is also an action server) to navigate
- When downstream node is performing the task, we need to transmit feedback to user
- Only after downstream node has completed it's action, we can return success/failure to user


The main pain points above are:
1. Waiting for downstream nodes to complete a task
2. Relaying feedback when waiting

We'll call this node ***MissionControl*** Node in line with the example. Additionally, the downstream
action server we are communicating with is called ***StateMachine***

#### Naive Approach

```python
class MissionControlActionServer(Node):

    def __init__(self):
        super().__init__('mission_control_action_server')
        self.get_logger().info("Starting State Machine Action Server")

        self.declare_parameter('robot_namespace_param', 'robot')
        robot_namespace = self.get_parameter('robot_namespace_param').get_parameter_value().string_value
        self.get_logger().info(f"robo2 namespace is {robot_namespace}")

        action_server_name = "MissionControl"
        self.get_logger().info(f"Mission Control Machine Action Server Name is {action_server_name}")
        robot_state_machine_name = robot_namespace + "/StateMachine"
        self.get_logger().info(f"robot StateMachine Server being used for client is {robot_state_machine_name}")

        # Construct the action server
        self._action_server = ActionServer(
            self,
            MissionControl,
            action_server_name,
            self.execute_callback,)

        # Construct the action client (node and name should be same as defined in action server)
        self._robot_state_machine_client = ActionClient(self, StateMachine, robot_state_machine_name)

    ############# Pose Subscribers ########################
        robot_map_pose_topic = robot_namespace + "/map_pose"

        self._robot_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            robot_map_pose_topic,  # Topic on which pose is being relayed
            self._robot_pose_callback,
            10  # Adjust the queue size as needed
        )

    def _robot_pose_callback(self, msg):
        self._robot_latest_pose = msg
    #! ######################################################

    def get_final_result(self, success_status):
        result = MissionControl.Result()
        result.success = success_status
        return result

    def publish_mission_control_feedback(self):
        # check if both robot1 and robot1 feedback has arrived:
        if self._robot_input_feedback_pose and self._robot_input_feedback_state:
            # publish feedback to high level action servers
            output_feedback_msg = MissionControl.Feedback()
            output_feedback_msg.pose_feedback = self._robot_input_feedback_pose
            output_feedback_msg.state_feedback = self._robot_input_feedback_state
            self._mission_control_goal_handle.publish_feedback(output_feedback_msg)

    ########## Send Goals to Individual Robots ################################################

    def robot_send_goal(self, robot_goal_package):
        self.get_logger().info('Calling Robot Action Server...')

        try:
            self._robot_state_machine_client.wait_for_server(timeout_sec=5)
        except:
            self.get_logger().error('Timeout: Action server not available, waited for 5 seconds')
            return

        self._send_goal_future = self._robot_state_machine_client.send_goal_async(
                                        robot_goal_package,
                                        feedback_callback=self.robot_client_feedback_callback)
        self._send_goal_future.add_done_callback(self.robot_client_goal_response_callback)

    ########### robot Functions ##########################################################

    def robot_client_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Robot 2 Goal rejected :(')
            return

        self.get_logger().info('Robot 2 Goal accepted :)')
        self._goal_accepted = True

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robot_client_get_result_callback)

    def robot_client_get_result_callback(self, future):
        result = future.result().result # success/failure result from lower level action server
        return result

    def robot_client_feedback_callback(self, input_feedback_msg):
        # get feedback from low level action server
        self._robot_input_feedback_pose = input_feedback_msg.feedback.pose_feedback
        self._robot_input_feedback_state = input_feedback_msg.feedback.state_feedback
        self.get_logger().info(f'Received feedback: robot_ pos x={input_feedback_pose_x},\
                                 robot_ pos y = {input_feedback_pose_y}')
        self.get_logger().info(f'Received feedback: robot_ state={self._robot_input_feedback_state}')

        self.publish_mission_control_feedback()


    ############### MAIN LOOP START ################################################
    def execute_callback(self, goal_handle):
        """
        Each Robot Task will be split into Undocking -> Navigation -> Docking
        """
        self._mission_control_goal_handle = goal_handle

        # INPUT FROM USER
        dock_ids = goal_handle.request.robot_specific_dock_ids
        self.get_logger().info(f'Input dock IDs are {dock_ids}')

        end_goal_robot.header.stamp = self.get_clock().now().to_msg()
        # some mapping from dock_id to map_pose
        dock_to_map_pose = {}
        end_goal_robot.pose = dock_to_map_pose[dock_ids]

        # check to make sure we have current robot pose (robot 2)
        assert self._robot_latest_pose is not None
        start_goal_robot = PoseStamped()
        start_goal_robot.header = self._robot_latest_pose.header
        start_goal_robot.pose = self._robot_latest_pose.pose.pose

        robot_goal_package = StateMachine.Goal()
        robot_goal_package.start_dock_id = dock_ids[0]
        robot_goal_package.end_dock_id = dock_ids[1]

        self.robot_send_goal(robot_goal_package)
        self.get_logger().info('Got result')

        # potentially add a time.sleep()

        goal_handle.succeed()
        return self.get_final_result(True)


def MissionControlServer(args=None):
    rclpy.init(args=args)
    print("ARGS IS", args)
    # start the Action Server
    mission_control_action_server = MissionControlActionServer()
    rclpy.spin(mission_control_action_server)

if __name__ == '__main__':
    MissionControlServer()
```

The above action server is going to have the following issues:
1. It will return success to user almost immediately as it will not wait for low level action server result to come in
2. If we weret to add a time.sleep() as commented in the code, we may get the low level action server result, but
   we the execute_callback thread will block the entire node (including any other subscriber like the map_pose subscriber)
3. It effectively becomes a service :/

#### Solution
To be able to block one function's execution while keeping other functions unblocked, we need to do the following:
- Make the execute callback thread separate from the ROS node
- Create events which allow us to block execution

We can make the callback a separate thread (or a thread which can be re-entered in our case) by
assiging only the callback to a 'callback group'. We have two options for such a callback group:
- **ReentrantCallbackGroup**
- **MutuallyExclusiveCallbackGroup**

Both achieve the functionality we require above. However, each ROS node is also generally equipped
to only run one thread. To allow it to run multiple threads, we use **'MultiThreadedExecutors'**

Finally, to block function execution, we use threading.Event(), which is not a ROS trick
but just a python feature.

Here's a new version of the same node which can now handle async tasks

```python
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Event

class MissionControlActionServer(Node):

    def __init__(self):
        super().__init__('mission_control_action_server')
        self.get_logger().info("Starting State Machine Action Server")

        robot_namespace = "/"

        action_server_name = "MissionControl"
        self.get_logger().info(f"Action Server Name is {action_server_name}")

        # define the low level action server for which this node is a client
        robot_state_machine_name = robot_namespace + "/StateMachine"

        # Define the callback group to make executor run on a separate thread
        self.callback_group = ReentrantCallbackGroup()
        # Construct the action server
        self._action_server = ActionServer(
            self,
            MissionControl,
            action_server_name,
            self.execute_callback,
            callback_group=self.callback_group
            )

        # Construct the action client (node and name should be same as defined in action server)
        self._robot_state_machine_client = ActionClient(self, StateMachine,
                                                        robot_state_machine_name)

        # Create an instance of the Event class and set it's internal blocking to active
        self._robot_action_complete = Event()
        self._robot_action_complete.clear() # event cleared/not_set = blocking

    ############# Pose Subscribers ########################
        robot_map_pose_topic = robot_namespace + "/map_pose"

        self._robot_pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            robot_map_pose_topic,  # Topic on which pose is being relayed
            self._robot_pose_callback,
            10  # Adjust the queue size as needed
        )

    def _robot_pose_callback(self, msg):
        self._robot_latest_pose = msg
    #! ######################################################

    def get_final_result(self, success_status):
        result = MissionControl.Result()
        result.success = success_status
        return result

    def publish_mission_control_feedback(self):
        # check if both robot1 and robot1 feedback has arrived:
        if self._robot_input_feedback_pose and self._robot_input_feedback_state:
            # publish feedback to high level action servers
            output_feedback_msg = MissionControl.Feedback()
            output_feedback_msg.pose_feedback = self._robot_input_feedback_pose
            output_feedback_msg.state_feedback = self._robot_input_feedback_state
            self._mission_control_goal_handle.publish_feedback(output_feedback_msg)

    ########## Send Goals to Individual Robots ################################################

    def robot_send_goal(self, robot_goal_package):
        self.get_logger().info('Calling Robot Action Server...')

        try:
            self._robot_state_machine_client.wait_for_server(timeout_sec=5)
        except:
            self.get_logger().error('Timeout: Action server not available, waited for 5 seconds')
            return

        self._send_goal_future = self._robot_state_machine_client.send_goal_async(
                                        robot_goal_package,
                                        feedback_callback=self.robot_client_feedback_callback)
        self._send_goal_future.add_done_callback(self.robot_client_goal_response_callback)

    ########### robot Functions ##########################################################

    def robot_client_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Robot 2 Goal rejected :(')
            return

        self.get_logger().info('Robot 2 Goal accepted :)')
        self._goal_accepted = True
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robot_client_get_result_callback)

    def robot_client_get_result_callback(self, future):
        result = future.result().result # success/failure result from lower level action server
        self.get_logger().info(f'Robot Result: {result}')
        if result is True:
            self.get_logger().info(f"Robot Goal Reached")
            self._robot_mission_success = True # used to give mission success respose upstream
            self._robot_action_complete.set() # sets the Event class to be complete
        else:
            self._robot_mission_success = False # used to give mission failure respose upstream
            self.get_logger().error(f"Robot Goal Not Reached!")
            self._robot_action_complete.set() # sets the Event class to be complete

    def robot_client_feedback_callback(self, input_feedback_msg):
        # get feedback from low level action server
        self._robot_input_feedback_pose = input_feedback_msg.feedback.pose_feedback
        self._robot_input_feedback_state = input_feedback_msg.feedback.state_feedback
        self.get_logger().info(f'Received feedback: robot_ pos x={input_feedback_pose_x},\
                                 robot_ pos y = {input_feedback_pose_y}')
        self.get_logger().info(f'Received feedback: robot_ state={self._robot_input_feedback_state}')

        self.publish_mission_control_feedback()


    ############### MAIN LOOP START ################################################
    def execute_callback(self, goal_handle):
        """
        Each Robot Task will be split into Undocking -> Navigation -> Docking
        """
        self._mission_control_goal_handle = goal_handle

        # INPUT FROM USER
        dock_ids = goal_handle.request.robot_specific_dock_ids
        self.get_logger().info(f'Input dock IDs are {dock_ids}')

        end_goal_robot.header.stamp = self.get_clock().now().to_msg()
        # some mapping from dock_id to map_pose
        dock_to_map_pose = {}
        end_goal_robot.pose = dock_to_map_pose[dock_ids]

        # check to make sure we have current robot pose (robot 2)
        assert self._robot_latest_pose is not None
        start_goal_robot = PoseStamped()
        start_goal_robot.header = self._robot_latest_pose.header
        start_goal_robot.pose = self._robot_latest_pose.pose.pose

        robot_goal_package = StateMachine.Goal()
        robot_goal_package.start_dock_id = dock_ids[0]
        robot_goal_package.end_dock_id = dock_ids[1]

        ######### Give Goals to robot and wait ###########
        self._robot_action_complete.clear()

        self.robot_send_goal(robot_goal_package)

        self._robot_action_complete.wait() # This is what blocks execution of this thread
        self.get_logger().info('Got result')

        goal_handle.succeed()

        if self._robot_mission_success:
            self.get_logger().info("Returning Success")
            return self.get_final_result(True)
        else:
            return self.get_final_result(False)


def MissionControlServer(args=None):
    rclpy.init(args=args)
    # Define the MultiThreadedExecutor() to allow multiple threads in above node
    executor = MultiThreadedExecutor()
    mission_control_action_server = MissionControlActionServer()
    rclpy.spin(mission_control_action_server, executor)

if __name__ == '__main__':
    MissionControlServer()
```