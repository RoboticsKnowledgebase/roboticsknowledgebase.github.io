# Debugging and Compiling ROS 2 Packages in Visual Studio Code

This tutorial explains how to set up **Visual Studio Code** for building and debugging ROS 2 packages effectively.

---

## 🧱 1. Preparing for Debugging

Imagine this scenario: You comment out a line that creates a publisher in your ROS 2 C++ node. The code compiles fine, but running the node gives a **segmentation fault**, with no helpful error message.

To debug such issues, we'll configure **GDB debugging** inside VS Code.

---

## 🧰 2. Install GDB Server (If Needed)

Ensure `gdbserver` is installed:

```bash
sudo apt update
sudo apt install gdbserver
```

---

## ⚙️ 3. Running a Node with GDB Server

Use this command to run a ROS 2 node with GDB server:

```bash
ros2 run --prefix 'gdbserver localhost:3000' <package_name> <executable_name>
```

Replace `<package_name>` and `<executable_name>` with your specific values.

---

## 🐞 4. Configure the Debugger in VS Code

Create or edit the file `.vscode/launch.json`:

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

> 📝 Replace the `program` path with your actual executable path inside `install/`.

Now press `F5` to launch the debugger and catch segmentation faults right where they happen.

---

## 🔄 5. Use `--symlink-install` for Faster Builds

By default, `colcon build` copies files from `build/` to `install/`. You can make builds faster by creating symbolic links:

```bash
colcon build --symlink-install
```

To switch to this method, first clean your workspace:

```bash
rm -rf build/ install/ log/
colcon build --symlink-install
```

This is especially helpful when you're making small changes to files like `package.xml`, launch files, or interface definitions.

---

## 🔧 6. Automate Build and Debug Tasks with `tasks.json`

Add this to `.vscode/tasks.json`:

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

Then in VS Code:
- Press `Ctrl+Shift+B` to build.
- Open the command palette (F1) → "Run Task" → choose `debug` or `test`.
- Use the printed instructions to launch your node under GDB.

---

## 📌 Final Tips

- Open your ROS workspace at the root level (`~/ros2_ws`) in VS Code.
- The ROS VS Code extension adds tools like:
  - Sourced terminals
  - Node runners
  - ROS topic graph visualizers

---

Happy debugging! 🛠️🐢
