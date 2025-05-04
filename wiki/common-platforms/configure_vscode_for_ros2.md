
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

To improve build efficiency, use the `--symlink-install` flag with colcon. This creates symbolic links instead of copying files into the `install/` directory:

```bash
colcon build --symlink-install
```

If you've already built the workspace, delete `build/`, `install/`, and `log/` directories first:

```bash
rm -rf build/ install/ log/
```

Then rerun the build command with the symlink flag.

## Automating Tasks with tasks.json

You can simplify your build process in VS Code using `tasks.json` under `.vscode/`. Here's an example configuration:

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
      "label": "debug",
      "type": "shell",
      "command": "echo \"\nRun your node with:\nros2 run --prefix 'gdbserver localhost:3000' <your_package> <your_node>\n\nThen update launch.json to point to the right executable.\"",
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

You can now trigger builds via **Terminal > Run Build Task** or `Ctrl+Shift+B`.

## Useful VS Code ROS 2 Extension Commands

![image](https://github.com/user-attachments/assets/a6fcc84a-beca-4650-876f-2b4d9d7f4318)

Click on the gear icon and by default we could set the Ros2 distribution to Humble.

So this is cool because once you do this via code will automatically know how to implement certain functionality

based on the ROS distro you are using.

![image](https://github.com/user-attachments/assets/54980fdd-5871-414c-b97e-4c330adb1e4b)




VS Code’s ROS 2 extension provides useful shortcuts:

* `ROS: Show Status` – shows active nodes and topics.
* `ROS: Create ROS Terminal` – opens a terminal with the ROS environment sourced.
* `ROS: Run a ROS Executable` – select a package and executable to run.
* `ROS: Update C++ Properties` – updates `c_cpp_properties.json` for intellisense.

Access these with `F1` → search `ROS`.



## Summary

In this tutorial, we:

* Learned to build ROS 2 workspaces with debug symbols.
* Set up `gdbserver` with VS Code.
* Used symlink installs to streamline builds.
* Automated common tasks with `tasks.json`.
* Explored ROS-related commands in the VS Code extension.

## See Also:

* [Debugging ROS 2](https://docs.ros.org/en/rolling/Tutorials/Debugging-ROS-2.html)
* [colcon documentation](https://colcon.readthedocs.io/en/released/index.html)

## Further Reading

* [VS Code launch.json reference](https://code.visualstudio.com/docs/editor/debugging)
* [ROS 2 Tools Extension on Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros)

