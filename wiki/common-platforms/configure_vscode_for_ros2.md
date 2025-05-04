
> This tutorial assumes you already have ROS 2 and Visual Studio Code installed and properly set up on your system.

In this tutorial, we'll walk through setting up a development environment in Visual Studio Code (VS Code) to debug and compile ROS 2 packages effectively. We'll start with configuring debugging tools to help identify runtime errors and then explore how to streamline your build process using VS Code's task configurations.

## Debugging in VS Code

Let's begin with debugging.

Open VS Code and load a ROS 2 package. For this example, we'll be using a simple publisher node written in C++. We'll start by intentionally commenting out the line that creates the publisher to simulate a bug. Even though the code appears fine to C++ linting tools, compiling it with `colcon build` will complete successfully.

![image](https://github.com/user-attachments/assets/6eadbdbd-1b8e-4a15-99f7-0a303aa4e57d)


However, when you try to run the node, you’ll get a segmentation fault. This error isn't informative, so we'll set up proper debugging using `gdb` and VS Code.

![image](https://github.com/user-attachments/assets/d09892fb-50d6-4d78-b8c9-ea76a9dfe1b4)


### Using GDB

First, build your ROS 2 workspace with debugging symbols:

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug
```

Then, run the node using `gdbserver`:

```bash
ros2 run --prefix 'gdbserver localhost:3000' <your_package> <your_node>
```

If `gdbserver` is not installed, you can install it with:

```bash
sudo apt update
sudo apt install gdbserver
```

So here we see processing our node listening on Port 3000, but now we need to configure our Vs code

to communicate with the debugger.

![image](https://github.com/user-attachments/assets/f9d6c269-eda2-4839-96e0-628bd8b95b70)


Now, configure VS Code to connect to the `gdbserver`. Create a `.vscode/launch.json` file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to gdbserver",
      "type": "cppdbg",
      "request": "launch",
      "miDebuggerServerAddress": "localhost:3000",
      "program": "/path/to/install/<your_package>/lib/<your_package>/<your_node>",
      "cwd": "${workspaceFolder}",
      "MIMode": "gdb",
      "externalConsole": false,
      "stopAtEntry": false
    }
  ]
}
```



Once saved, go to the Debug tab in VS Code and click **Start Debugging**. You'll see the line where the segmentation fault occurred highlighted. The debug pane shows variables, their values, and the call stack—helping identify issues such as an uninitialized publisher.

![image](https://github.com/user-attachments/assets/2b411708-6e57-4882-a9c7-480f60ea1c68)

![image](https://github.com/user-attachments/assets/77b4a867-58d8-4dbb-bfc1-1102aed37ec9)


## Simlink Install with colcon
By default, when you run `colcon build`, the build system **copies** the compiled binaries, libraries, and resources into the `install/` directory. While this works fine, it can be **inefficient and slow**—especially during iterative development—because every time you rebuild, files are duplicated into the install space.
To improve build efficiency, use the `--symlink-install` flag with colcon. This creates symbolic links instead of copying files into the `install/` directory:

```bash
colcon build --symlink-install
```

If you've already built the workspace, delete `build/`, `install/`, and `log/` directories first:

```bash
rm -rf build/ install/ log/
```

Then rerun the build command with the symlink flag.

## Automating Builds and Debugging with `tasks.json`

Now, it can be inconvenient to remember and re-type long commands like `colcon build --symlink-install`, especially if you often switch between workspaces. VS Code allows us to automate such tasks using a `tasks.json` file.

### Creating a Build Task

Open your workspace in VS Code. In the file explorer, navigate to the `.vscode/` directory. If it doesn’t exist, create it. Inside that directory, create a file named `tasks.json`.

Here’s an example configuration:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build",
      "type": "shell",
      "command": "source /opt/ros/humble/setup.bash && colcon build --symlink-install",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": []
    },
    {
      "label": "build (debug)",
      "type": "shell",
      "command": "source /opt/ros/humble/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug",
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "debug-helper",
      "type": "shell",
      "command": "echo "\nRun your node with:\nros2 run --prefix 'gdbserver localhost:3000' <your_package> <your_node>\n\nThen update launch.json to point to the right executable."",
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

> 🔧 **Tip:** If you want to source your full `.bashrc` instead of just ROS setup, modify the `command` line:
>
> ```bash
> source ~/.bashrc && colcon build --symlink-install
> ```

To run these tasks:
- Go to **Terminal → Run Build Task** or press **Ctrl+Shift+B**.
- Choose `build`, `build (debug)`, or `debug-helper`.

This automates builds and reminds you of necessary debug setup steps.

> ⚠️ **Important**: Make sure VS Code is opened in your **ROS 2 workspace root**. If you're in the wrong directory, `colcon build` may fail or build the wrong packages.

---

## Using the ROS 2 VS Code Extension

The [ROS Tools extension](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros) in VS Code offers convenient tools for interacting with ROS 2.

### Common ROS Commands in VS Code

Press **F1** and type `ROS` to view available commands:

- **ROS: Show Status** – Displays current ROS nodes, topics, and status.
- **ROS: Create ROS Terminal** – Opens a terminal with the correct ROS environment sourced.
- **ROS: Run a ROS Executable** – Lets you select a package and node to run from a dropdown.
- **ROS: Update C++ Properties** – Updates your `c_cpp_properties.json` file to enable IntelliSense in C++ ROS projects.
- **ROS: Install ROS Dependencies (rosdep)** – Installs missing dependencies in your workspace using `rosdep install`.

> 🧠 Even if you’re using ROS 2 (e.g., Humble), some commands still reference `rosrun` from ROS 1. These labels are legacy but still work for launching executables.

### Running a Node via Extension

1. Press **F1** → `ROS: Run a ROS Executable`.
2. Choose your package.
3. Select the executable (e.g., `minimal_publisher`).
4. If needed, enter arguments.
5. A terminal will open and run the node with the correct environment.

### Creating a ROS-Sourced Terminal

Select **ROS: Create ROS Terminal** from the F1 menu. This opens a new terminal tab with ROS environment variables sourced, allowing you to run commands like:

```bash
ros2 topic echo /your_topic
```

### Updating C++ IntelliSense

If your C++ symbols or headers aren’t resolving correctly, use:

```text
ROS: Update C++ Properties
```

This command re-generates `c_cpp_properties.json` with include paths specific to your workspace and ROS installation.

---

## Troubleshooting and Best Practices

- Always open VS Code in your **workspace root** directory.
- Use `--symlink-install` to reduce rebuild times.
- Use `launch.json` to attach VS Code’s debugger to ROS nodes.
- Keep reusable scripts or environment helpers in your workspace for consistency.

---
## See Also:

* [Debugging ROS 2](https://docs.ros.org/en/rolling/Tutorials/Debugging-ROS-2.html)
* [colcon documentation](https://colcon.readthedocs.io/en/released/index.html)

## Further Reading

* [VS Code launch.json reference](https://code.visualstudio.com/docs/editor/debugging)
* [ROS 2 Tools Extension on Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros)

