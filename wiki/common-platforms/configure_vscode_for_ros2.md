# Debugging and Building ROS 2 Packages in Visual Studio Code

This tutorial covers how to configure Visual Studio Code (VS Code) for building and debugging ROS 2 packages using C++. It assumes that you already have both ROS 2 (e.g., Humble) and VS Code installed and set up properly.

---

## 1. Background: Why Debugging Matters

Sometimes your C++ ROS 2 code compiles successfully, but fails during runtime—perhaps with a segmentation fault and little information to guide you. To address such issues effectively, it's essential to:

* Enable debug symbol generation during compilation.
* Use a debugger like GDB.
* Connect VS Code to GDB for a visual debugging experience.

---

## 2. Example ROS 2 Publisher Code

We'll use a basic publisher node in C++ as our example.

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node {
public:
  MinimalPublisher() : Node("minimal_publisher") {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        auto message = std_msgs::msg::String();
        message.data = "Hello, world!";
        publisher_->publish(message);
      }
    );
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

---

## 3. Compiling with Debug Symbols

To generate debugging symbols, compile your workspace using the following command:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

You can automate this with VS Code using the `tasks.json` setup below.

---

## 4. Setting Up `tasks.json`

Create a file named `.vscode/tasks.json` in the root of your workspace:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build",
      "type": "shell",
      "command": "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
    },
    {
      "label": "debug",
      "type": "shell",
      "command": "echo -e '\n\nRun the node using the following prefix: \n  ros2 run --prefix 'gdbserver localhost:3000' <package_name> <executable_name> \n\nAnd modify the executable path in .vscode/launch.json file \n' && source /opt/ros/humble/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo"
    },
    {
      "label": "test",
      "type": "shell",
      "command": "colcon test && colcon test-result"
    }
  ]
}
```

---

## 5. Setting Up `launch.json`

To attach VS Code's debugger to a ROS 2 node, configure `.vscode/launch.json` like this:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "C++ Debugger",
      "request": "launch",
      "type": "cppdbg",
      "miDebuggerServerAddress": "localhost:3000",
      "cwd": "/",
      "program": "/home/$USER/Workspaces/ros2_cpp_ws/install/udemy_ros2_pkg/lib/udemy_ros2_pkg/service_client"
    }
  ]
}
```

Replace the `program` path with the absolute path to your ROS 2 node executable.

---

## 6. Running the Debugger

### Step-by-step:

1. Build your workspace using:

   ```bash
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
   ```
2. Run your node under GDB server:

   ```bash
   ros2 run --prefix 'gdbserver localhost:3000' <package_name> <executable_name>
   ```
3. In VS Code, press `F5` or go to `Run > Start Debugging`.
4. VS Code will connect to the GDB server and break at the error location if there is a crash.

---

## 7. Using the VS Code ROS Extension

VS Code’s ROS extension adds convenience:

* `ROS: Show Status` – View running nodes and topics.
* `ROS: Create Terminal` – Opens a sourced terminal.
* `ROS: Run ROS Command` – Run nodes from a graphical menu.

Access these via the command palette (`F1`) and typing `ROS`.

---

## 8. Tips for Debugging

* Use the left panel during debugging to inspect variables and the call stack.
* Use `readlink` to confirm symlink targets (for `--symlink-install` builds):

  ```bash
  readlink install/your_pkg/lib/your_pkg/your_executable
  ```
* To fix symbol issues, clean your build:

  ```bash
  rm -rf build/ install/ log/
  ```

---

## 9. Summary

With the above setup:

* You can compile and debug C++ ROS 2 nodes directly inside VS Code.
* The `tasks.json` makes builds reproducible.
* The `launch.json` connects VS Code's debugger to your running ROS node.
* You can track variables, memory, and exceptions visually during runtime.

---

Now you're ready to debug efficiently in ROS 2 using Visual Studio Code!
