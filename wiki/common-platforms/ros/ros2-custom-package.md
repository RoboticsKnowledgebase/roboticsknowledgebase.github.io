---
date: 2025-05-03
title: Building ROS2 Custom Packages
---

When working on a larger ROS 2 project, you'll often need to add new packages to an existing workspace. This tutorial provides a step‑by‑step guide on how to create and integrate a new ROS 2 package (targeting **ROS 2 Humble**) into an existing workspace. We focus on using **colcon** as the build tool and cover both Python‑based packages (using **ament_python**) and C++ packages (using **ament_cmake**). You will learn how to set up the package manifest and build files, build the workspace with colcon, verify that the new package is recognized by the ROS 2 environment, and debug common issues that can arise. By the end of this tutorial, you'll be confident in adding new ROS 2 packages to your workspace and troubleshooting integration problems.

**Note:** The primary example uses a Python package, but we include notes on integrating C++ packages. We'll also discuss a real‑world debugging scenario (a rosbag2 composable node issue we encountered when developing our [ROS2 Humble Intra‑Process Communication Recorder](/wiki/tools/ros2-humble-ipc-recorder.md)) to illustrate how to tackle workspace integration challenges in practice.

## Table of Contents
- [Introduction](#introduction)
- [Background: ROS 2 Workspaces and Build Systems](#background-ros-2-workspaces-and-build-systems)
- [Step 1: Creating a New ROS 2 Package in an Existing Workspace](#step-1-creating-a-new-ros-2-package-in-an-existing-workspace)
  - [Package Manifest (package.xml)](#package-manifest-packagexml)
  - [Python Package Setup (ament_python)](#python-package-setup-ament_python)
  - [C++ Package Setup (ament_cmake)](#c-package-setup-ament_cmake)
- [Step 2: Building the Workspace with Colcon](#step-2-building-the-workspace-with-colcon)
- [Step 3: Sourcing the Workspace and Validating Integration](#step-3-sourcing-the-workspace-and-validating-integration)
- [Common Pitfalls and Debugging Tips](#common-pitfalls-and-debugging-tips)
  - [Case Study: Rosbag2 Composable Node Integration Issue](#case-study-rosbag2-composable-node-integration-issue)
- [Summary](#summary)
- [See Also](#see-also)
- [Further Reading](#further-reading)
- [References](#references)

## Introduction

Adding a new package to a ROS 2 workspace involves more than just writing code - you must properly configure build files and manifests so that the package can be built and recognized by the ROS 2 ecosystem. In this tutorial, we walk through creating a new ROS 2 package inside an existing workspace and ensuring it's correctly integrated. We will use **ROS 2 Humble** for examples. Our focus is on using **colcon** (the standard ROS 2 build tool) and the ament build system. We will create a simple example package (in Python) and discuss what each of the key files (`package.xml`, `setup.py`, and if applicable, `CMakeLists.txt`) should contain. Then we'll build the workspace, demonstrate how to verify the package's presence (so that other packages or tools can see it), and troubleshoot common mistakes.

Whether you are creating a new node, library, or other ROS 2 component, following these steps will help avoid issues like “package not found” errors or build system mix‑ups. We also highlight a real debugging scenario involving a rosbag2 composable node to show how to generalize problem‑solving strategies for package integration issues.

## Background: ROS 2 Workspaces and Build Systems

Before diving in, it's important to understand how ROS 2 organizes code and builds packages:

- **ROS 2 Workspace:** A workspace is a directory (often named something like `ros2_ws`) that contains a `src/` subdirectory. All ROS 2 packages live under `src/`. You can have multiple packages in one workspace and build them together. Colcon will create separate `build/`, `install/`, and `log/` directories alongside `src/` when you build the workspace. An existing workspace likely already has these; adding a new package means putting it in the `src` folder and rebuilding.
- **Colcon Build Tool:** ROS 2 uses **colcon** instead of ROS 1's catkin tools. Colcon builds all packages in a workspace (or selected packages) and installs their files into the `install/` directory. Colcon is an evolution of ROS build tools like `catkin_make` / `ament_tools`, and it supports both CMake (C/C++ code) and Python package build types.
- **Ament Build System:** ROS 2 packages use *ament* (either `ament_cmake` or `ament_python`) as the build system. **ament_cmake** is used for C++ (or mixed C++/Python) packages and relies on CMakeLists files and CMake macros, while **ament_python** is used for pure Python packages and relies on Python's `setup.py` for installation. These build types are specified in the package manifest. By contrast, ROS 1 used Catkin; mixing ROS 1 Catkin packages into a ROS 2 workspace is not straightforward and generally not recommended without special bridging tools.
- **Package Manifest (`package.xml`):** Every ROS 2 package has a `package.xml` file (format 2 is standard for ROS 2) which declares the package's name, version, dependencies, and build type. This file is critical for colcon to identify how to build the package and what other packages it needs. The manifest also ensures the package gets indexed so ROS 2 tools can find it.
- **Ament Index vs Catkin Index:** ROS 2 uses an *ament resource index* to locate packages and their resources. When a package installs, it registers itself by creating a marker file in `install/share/ament_index/resource_index/packages/<package_name>`. ROS 1 (Catkin) relied on a different mechanism (crawling directories or `ROS_PACKAGE_PATH`). If a ROS 2 package is not correctly registered in the ament index, tools like `ros2 run` or `ros2 pkg list` will not see it. Misconfiguring a ROS 2 package as a Catkin (ROS 1) package can cause it to install under a Catkin‑specific index (or not be indexed at all), leading to “package not found” issues. Always use the appropriate ROS 2 build type so that the package is indexed in ROS 2's ament resource index, not in a Catkin index.

With these concepts in mind, let's proceed to creating and building a new package.

## Step 1: Creating a New ROS 2 Package in an Existing Workspace

First, navigate to the `src` directory of your existing workspace (for example, `~/ros2_ws/src`). We will create a new package named `my_package`. ROS 2 provides a handy command-line tool to bootstrap a package:

```bash
# Navigate to the workspace's source directory
$ cd ~/ros2_ws/src

# Create a new package named my_package, using ament_python as build type and add a dependency on rclpy (for ROS 2 Python client library)
$ ros2 pkg create my_package --build-type ament_python --dependencies rclpy
````

The `ros2 pkg create` command will create a new folder `my_package/` in the `src` directory. Inside, you should see a structure like:

```plaintext
my_package/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── my_package
├── my_package/   # Python module directory
│   └── __init__.py
└── test/        # (optional) testing files
```

Colcon's package creation template populates these files with boilerplate. For a Python package, it includes an empty Python module (`my_package/__init__.py`), an empty resource marker file, and some sample test files. If you used the `--node-name` option, it might also create a sample Python node script (e.g. `my_node.py`) and some boilerplate code in it.

> **Note:** If you wanted a C++ package instead, you would specify `--build-type ament_cmake`. The structure would then include a `CMakeLists.txt`, `src/` folder (with a sample .cpp if a node name was given), and an `include/<package_name>/` directory for headers, instead of the Python-specific files. We'll cover C++ differences shortly.

Next, we need to examine and possibly edit the key files to ensure our package is properly configured.

### Package Manifest (`package.xml`)

The `package.xml` is the manifest that describes your package. Open `my_package/package.xml` in a text editor. You should see tags for `<name>`, `<version>`, `<description>`, maintainers, licenses, and dependencies. ROS 2 package manifests are format 2 XML. Key points to check or modify:

* **Name, Version, Description:** Make sure `<name>my_package</name>` is set (the template does this). Update `<description>` from the placeholder “TODO: ...” to a meaningful description of your package. Set the license tag to your chosen license (e.g. `<license>Apache-2.0</license>` or other OSI-approved license).
* **Maintainer:** Ensure there's at least one `<maintainer email="...">Your Name</maintainer>` tag. Fill it with your info if the template left a generic placeholder.
* **Build Type:** The template should include something like `<build_type>ament_python</build_type>` inside an `<export>` section for a Python package, or `ament_cmake` for a C++ package. This tells colcon which build tool to use for this package. Verify this is correct. A wrong build type (for example, mistakenly using `ament_cmake` for a purely Python package or vice versa) can cause build issues. For instance, using `cmake` or `catkin` as the build type for a ROS 2 package is a common misconfiguration - colcon either won't know how to build it or will treat it as a plain CMake project without proper registration, leading to the package not appearing in ROS 2's index.
* **Dependencies:** The `ros2 pkg create` command we ran already added `rclpy` as a dependency in the manifest (it should appear as `<exec_depend>rclpy</exec_depend>` and possibly in `<build_depend>` if needed). Add any other dependencies your package needs. For example, if your Python code will import messages from `geometry_msgs`, add:

  ```xml
  <exec_depend>geometry_msgs</exec_depend>
  ```

  Use `<build_depend>` for compile-time dependencies (mostly for C++ code or generating interfaces) and `<exec_depend>` for run dependencies (needed at runtime). You can also use `<depend>` which implies both build and exec dependency in format 2.
* **Dependency Versioning:** It's usually not necessary to specify versions for dependencies unless you have a specific requirement. The presence of the tag is enough for colcon/rosdep to ensure the dependency is present.
* **Export Section:** Ensure the `<export>` tag contains the build\_type. For ament\_python it will be:

  ```xml
  <export>
    <build_type>ament_python</build_type>
  </export>
  ```

  The template handles this. This section can also include other export information (for example, plugin definitions for pluginlib, but that's advanced usage).

After editing, your `package.xml` might look like this (minimal example for our Python package):

```xml
<?xml version="1.0"?>
<package format="2">
  <name>my_package</name>
  <version>0.0.0</version>
  <description>A simple example package for ROS 2 (Python).</description>
  <maintainer email="[email protected]">Your Name</maintainer>
  <license>Apache-2.0</license>

  <!-- Build tool dependencies: ament_python for Python packages -->
  <buildtool_depend>ament_python</buildtool_depend>

  <!-- Runtime and build dependencies -->
  <depend>rclpy</depend>
  <exec_depend>geometry_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Notice we used `<buildtool_depend>ament_python</buildtool_depend>` to indicate we need the ament\_python build tool. We also listed `rclpy` and `geometry_msgs` as dependencies (so that our package will be able to import `rclpy` and messages at runtime). Adjust these depends according to your package's needs.

For a C++ package, the manifest would be similar but use `ament_cmake` for build tool and build type, and you'd list any C++ library/package dependencies under `<depend>` (or specifically `<build_depend>` and `<exec_depend>` as appropriate).

**Common Manifest Pitfall:** Forgetting to list a dependency in `package.xml` can lead to build failures or (worse) successful build but runtime errors. Colcon relies on the manifest to order package builds. If package A depends on B but you didn't declare it, colcon might try to build A before B (leading to include or import errors), or `rosdep` won't know to install B. Always update `package.xml` with all dependencies (including things like message packages, libraries, or ROS 2 core packages your code uses).

### Python Package Setup (ament\_python)

Since our example package is Python-based, we need to ensure the Python packaging files are correctly set up. There are typically two important files for an `ament_python` package: `setup.py` and `setup.cfg`, plus the Python module code itself.

#### `setup.py`

Open `my_package/setup.py`. This is a standard Python `setuptools` setup script, adapted for ROS 2. It defines how to install your package and any executable scripts. The template created by `ros2 pkg create` likely contains something similar to:

```python
from setuptools import setup

package_name = 'my_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),               # Ensure package is indexed
        ('share/' + package_name, ['package.xml']),   # Install package.xml
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='[email protected]',
    maintainer='Your Name',
    maintainer_email='[email protected]',
    description='A simple example package for ROS2 (Python)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'executable_name = module.path:main_function',
        ],
    },
)
```

Let's break down the important parts of `setup.py`:

* **`packages=[package_name]`:** This tells setuptools to look for a Python package (directory with `__init__.py`) named `my_package` and install it. The template ensures your `my_package/` directory (with `__init__.py`) is included.
* **`data_files`:** This is crucial in ROS 2 packaging. Here we install two things:

  * A marker file for the ament index: The empty file in `resource/my_package` gets copied into `share/ament_index/resource_index/packages/` with the package name. This “marks” the package as a ROS 2 package in the ament resource index so that ROS 2 tools know it's present. (Colcon was historically adding this automatically for Python packages, but it's required to do it explicitly moving forward.)
  * The `package.xml`: Installing the manifest to `share/my_package/` is standard so that tools can inspect package metadata at runtime if needed.
* **`entry_points`:** This is where we define console scripts (executables). If you want to make a Python node directly runnable with `ros2 run`, you should add an entry here. For example, if you have a Python file `my_package/my_node.py` with a function `main()`, you can add:

  ```python
  entry_points={
      'console_scripts': [
          'my_node = my_package.my_node:main',
      ],
  }
  ```

  This will install a command-line script named `my_node` (in ROS 2 environments, it will be namespaced under the package, but `ros2 run` knows to look for it). Then running `ros2 run my_package my_node` will execute the `main()` function in `my_node.py`. In our example template, we haven't created a `my_node.py` yet, but you can create it under `my_package/` and use this mechanism.
* **Metadata fields** like `author`, `maintainer`, `description`, `license` - these should match or complement your package.xml. The template likely filled them with placeholders or the data you provided via command options. Fill them in accordingly (in our snippet above we put example values).

After configuring `setup.py`, if you create a Python node script, e.g., `my_package/my_node.py`, ensure it has a `main()` function (or whatever entry point you specified). For example, a simple node might be:

```python
# my_package/my_node.py
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info("Hello from my_node!")

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

This is just a basic example; the important part is that `main()` will be called when you do `ros2 run my_package my_node` (after building and sourcing). The console\_scripts entry point connects the dots.

#### `setup.cfg`

The `setup.cfg` file is often a one-liner needed for ROS 2 Python packages that have console scripts. It typically contains:

```ini
[build_scripts]
executable = /usr/bin/env python3
```

This ensures the installed scripts use the correct Python interpreter. The template likely included this file already. You usually don't need to modify it except to verify it exists in the package. Its presence allows ROS 2 to find the executables via `ros2 run`.

#### Resource Folder and Marker

As mentioned, the `resource/` directory contains a file named exactly as your package (`my_package` with no extension). It's an empty file. During installation (via `setup.py` data\_files), this gets copied to the ament index. Ensure this file exists and is named correctly. The template should have created it for you. If it's missing, create the folder `resource` and inside it an empty file `my_package`. This small detail is what allows ROS 2 tools to identify that your package is installed. If omitted, you might encounter warnings or issues with package recognition (e.g., a warning that the package doesn't explicitly install a marker, which will be required in the future).

**Recap for Python package:** After these checks, our Python package has:

* Proper `package.xml` with `ament_python` and dependencies.
* A `setup.py` that installs the marker and `package.xml`, and sets up any console script entry points.
* A `setup.cfg` for ensuring correct script shebang.
* Python module code (and possibly a script with a `main()` function).

This is sufficient for colcon to build and install the package.

### C++ Package Setup (ament\_cmake)

If your new package is C++ or you need to integrate C++ code (or libraries) in the workspace, the process is similar but involves a CMake build. Suppose we created `my_cpp_pkg` with `--build-type ament_cmake`. Key files would be `CMakeLists.txt` (instead of setup.py) and the `package.xml` would list `ament_cmake` as the build type.

For a minimal C++ package:

* **CMakeLists.txt:** This file defines how to build the C++ code. A simple template might contain:

  ```cmake
  cmake_minimum_required(VERSION 3.8)
  project(my_cpp_pkg)

  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)  # for example, if using rclcpp

  add_executable(my_node src/my_node.cpp)
  ament_target_dependencies(my_node rclcpp std_msgs)  # list your dependencies

  install(TARGETS my_node
    DESTINATION lib/${PROJECT_NAME})

  ament_package()
  ```

  This CMakeLists does a few important things:

  * Calls `find_package(ament_cmake REQUIRED)` and any other dependencies (like `rclcpp`, message packages, etc.). This makes sure the CMake knows about those packages' include paths and libraries.
  * Uses `add_executable` to compile a node from source (if you have a `src/my_node.cpp`).
  * Uses `ament_target_dependencies` to link the target against ROS 2 libraries (here `rclcpp` and `std_msgs` as an example).
  * Installs the compiled executable to the correct location (`lib/my_cpp_pkg`). This is crucial: if you don't install your targets or other artifacts, they won't be placed into the `install/` space and ROS 2 won't be able to use them.
  * Calls `ament_package()` at the end. This macro does a lot, including adding the package to the ament index (it generates that same marker file for C++ packages automatically) and exporting CMake config files for downstream packages. **Forgetting to call `ament_package()` is a common mistake** - without it, your package might build but won't be findable by other packages or fully registered.

* **Source code and headers:** Place your `.cpp` files under `src/` and headers (if any) under `include/my_cpp_pkg/`. The template from `ros2 pkg create` likely made an empty `src/` and `include/` directory. If you used `--node-name`, it may have created a simple `my_node.cpp` that prints “Hello World” to stdout. Ensure your `CMakeLists.txt` references the correct file names.

* **package.xml:** Similar to the Python example, but with `<buildtool_depend>ament_cmake</buildtool_depend>` and `<build_type>ament_cmake</build_type>` in the export section. Also list dependencies like `rclcpp` or `std_msgs` in the manifest (`<depend>rclcpp</depend>` etc.), matching what you find\_package in CMake.

After setting up, a minimal `package.xml` for C++ might be:

```xml
<package format="2">
  <name>my_cpp_pkg</name>
  <version>0.0.0</version>
  <description>Example ROS2 C++ package</description>
  <maintainer email="[email protected]">Your Name</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

With these in place, your C++ package will compile and integrate. One additional note: if your package is a library or plugin (not just an executable), you would use `add_library(...)` in CMake and also install the library, plus possibly export plugin information. For example, for a *composable node* (which is a plugin), you'd create a library and use `rclcpp_components_register_nodes` macro and install a plugin description XML. That's more advanced, but keep in mind that integration requires installing those resources too.

**Common C++ Pitfall:** Not installing targets or forgetting `ament_package()`. If you build a library or node and don't install it, it stays in the build directory and won't be available at runtime in the install space. Similarly, without `ament_package()`, other packages won't be able to `find_package(my_cpp_pkg)` because no package index entry or CMake config gets generated. Always follow the pattern above.

## Step 2: Building the Workspace with Colcon

After creating the package and configuring its files, the next step is to build the workspace. Since we are adding to an existing workspace, make sure any existing packages are also in a good state (their manifests are correct, etc.).

Before building, it's generally good to run `rosdep` to install any system dependencies for all packages in your workspace:

```bash
# From the workspace root (e.g., ~/ros2_ws)
$ rosdep install --from-paths src --ignore-src --rosdistro humble -y
```

This finds all `<depend>` and `<exec_depend>` that are system packages and attempts to install them (for example, if you depend on OpenCV or other system libraries). If your new package only depends on core ROS packages that are already installed, rosdep will likely say all requirements are already satisfied.

Now, build the workspace:

```bash
# Ensure you are at the root of the workspace (the directory that contains src/)
$ cd ~/ros2_ws

# Build the entire workspace
$ colcon build
```

Colcon will locate all packages in `src/` (including the new `my_package` and any others) and build them in the correct order. You should see output logging each package's build process. In our case, it should find `my_package` (ament\_python) and process its setup.py, and produce an install for it under `install/my_package`. If you also have `my_cpp_pkg`, colcon will compile it via CMake and place its artifacts under `install/my_cpp_pkg`.

If the workspace is large, you can choose to build only the new package to save time:

```bash
$ colcon build --packages-select my_package
```

This will build `my_package` and nothing else (except its dependencies if not built yet). In an existing workspace scenario, `--packages-select` is handy if you know only your new package and maybe a few others changed. Note that if your new package depends on another package in the workspace that hasn't been built yet, you should either build that dependency first or let colcon build it by not excluding it.

A successful build will end with a summary like:

```
Summary: 1 package finished [X.Y seconds]
```

for a single package build, or listing multiple if you built the whole workspace. If there are errors, read them carefully:

* **Compilation errors (for C++):** Check your code and CMakeLists. Common issues include missing includes or forgetting to link to a library (manifest vs CMake mismatch).
* **Installation or setup errors (for Python):** If colcon complains about missing files or not finding `setup.py`, ensure you ran colcon from the workspace root, not inside the package. Running `colcon build` from the `src/` directory can confuse it (it may not find packages properly). Always run from the root (where `build/` and `install/` will be created).
* **Dependency not found:** If you see an error like “Could not find package configuration file provided by X”, you likely have a missing dependency or forgot to source the underlay (see Pitfalls section). For example, if your package depends on something in ROS 2 that's not installed, or another package in the workspace that failed to build, colcon will stop. Installing the dependency via apt or adding it to the workspace and building it should resolve this.

After a successful build, colcon will have populated the `install/` directory with subdirectories for each package (unless you used `--merge-install`, which puts all files in a single prefix). For each package, key things in `install/` are:

* `install/my_package/lib/` - contains executables or libraries (for Python packages, console entry scripts go here).
* `install/my_package/share/` - contains the package.xml and potentially ament index resource file, and any launch files or other resources if you installed them.
* For Python, `install/my_package/lib/python3.X/site-packages/my_package` - the Python module code gets installed here so it's accessible via Python path.

## Step 3: Sourcing the Workspace and Validating Integration

Once built, you need to **source the workspace's setup script** to add the new package to your environment. Open a new terminal (or source in the current one) and execute:

```bash
$ source ~/ros2_ws/install/setup.bash
```

Make sure you have also sourced the ROS 2 installation (e.g., `/opt/ros/humble/setup.bash`) either earlier or in your `~/.bashrc`. If your ROS 2 environment wasn't sourced, do that first, then the workspace. Sourcing `install/setup.bash` will overlay the workspace on top of the ROS distribution. This sets up environment variables so that ROS 2 commands are aware of the new package and its contents. (In Windows, you'd call the `.bat` file; in zsh or other shells there are analogous scripts.)

Now, let's validate that the new package is recognized and accessible:

* **List the package:** Run `ros2 pkg list | grep my_package`. You should see `my_package` in the output. ROS 2's package listing uses the ament index to find packages. If it doesn't show up, something is wrong with the installation or indexing (we'll troubleshoot that soon).

* **Run executables:** If your package has a runtime executable (entry point or C++ node), try running it. For our Python example, if we added the console script for `my_node`, do:

  ```bash
  $ ros2 run my_package my_node
  ```

  This should start the node and you should see its log output (e.g., “Hello from my\_node!” from the example code). If ROS 2 complains “Package ‘my\_package' not found” or “Executable ‘my\_node' not found”, then the integration isn't complete:

  * “Package not found” indicates the environment is not aware of the package (likely you didn't source the setup.bash of the workspace, or the package failed to install properly).
  * “Executable not found” indicates the package is known, but the console script or binary isn't found. This could mean you didn't set up the entry point (for Python) or didn't install the target (for C++). Double-check `setup.py` entry\_points or CMake install directives.

* **Check environment variables:** For Python packages, `PYTHONPATH` (or the more modern approach of using the installed site-packages) is handled by the setup script. After sourcing, you can run `python3 -c "import my_package"` in the terminal to see if the module imports. It should succeed (doing nothing if the module is empty). If it fails, the package isn't on the Python path, implying the setup script wasn't sourced correctly or installation failed.

* **Ament index check:** You can manually verify that the marker file is present. Look for `install/share/ament_index/resource_index/packages/my_package`. If that file is missing, ROS 2 will not recognize the package even if everything else is there. If you find it missing, it means the installation step in setup.py didn't happen or was incorrect. (Earlier colcon versions implicitly created it for python, but that's changing.) The fix is to add the proper `data_files` entry as shown in setup.py above.

* **Using the package from another package:** If you have another package in the workspace that depends on `my_package`, try building that as a further test. Colcon should be able to find `my_package` via `find_package` (CMake) or as a dependency (package.xml). If it can't, then something is off with how `my_package` was registered (again, likely missing `ament_package()` or marker).

Everything up to this point ensures that the package is correctly built and integrated. Next, we'll cover common pitfalls and how to debug them when things don't go as expected.

## Common Pitfalls and Debugging Tips

Even when following the general procedure, it's easy to run into issues. Here are some common problems and how to address them:

* **Forgetting to Source the Workspace:** After building, if you forget to source the new `install/setup.bash`, ROS 2 commands won't know about your new package. This leads to errors like `Package 'my_package' not found` when you try to run or launch it. The fix is simple: source the workspace overlay. If you open new terminals frequently, consider adding `source ~/ros2_ws/install/setup.bash` to your `~/.bashrc` (after sourcing the base ROS setup) for convenience.
* **Running Colcon in the Wrong Directory:** Ensure you run `colcon build` from the root of the workspace (the directory that contains `src/`), *not* from within `src/` or inside a package. Running colcon from the wrong location can result in it not finding any packages or only building one and not installing things correctly. One user error was running `colcon build` inside the `src` folder, causing the package to not be found after build. Always `cd` to the workspace root before building.
* **Missing Dependencies:** If colcon fails with an error about missing package configurations, e.g.:

  ```
  CMake Error at CMakeLists.txt: find_package(some_dep REQUIRED) ... not found
  ```

  or a Python import error during runtime, you likely didn't install or declare a dependency. Use `rosdep` to install system dependencies. Check that every package or module your code uses is listed in `package.xml`. For C++ dependencies, also ensure you have the corresponding `find_package` in CMake. In one case, a user's C++ component had an undefined symbol at runtime - it turned out they hadn't linked against a needed library in CMake, which is solved by adding it to `ament_target_dependencies`.
* **Misconfigured Package Type (Catkin vs Ament):** As noted, ROS 2 does not use Catkin. If you accidentally use a Catkin-style package (e.g., copying a ROS1 package into ROS2 workspace without conversion), colcon might identify it as `type: ros.catkin`. By default, ROS 2 installations don't even include the Catkin build tool, so you'll get errors about `Findcatkin.cmake` not found. The solution is to migrate the package to ROS 2:

  * Update the `package.xml` to format 2 and change `<buildtool_depend>catkin</buildtool_depend>` to the appropriate ament dependency.
  * If it's pure CMake without ROS, you might set `<build_type>cmake` and manage CMakeLists manually, but then ensure you install a marker file to register it (similar to what ament\_cmake/ament\_python do) so ROS 2 sees it.
  * Generally, it's best to use `ament_cmake` or `ament_python`. Only use `cmake` build type for special cases, and be aware you must handle the indexing and installation.
  * If you truly need to build a ROS1 catkin package inside a ROS2 workspace, there are colcon extensions and workarounds, but that's advanced (see **catment** in ROS 2 design docs, which allows mixing, but requires additional setup).
* **Package Builds but Not Recognized (Missing Ament Index Resource):** If `ros2 pkg list` doesn't show your package even after sourcing, check for the presence of the marker file as mentioned. For Python packages, a common oversight is not including the `data_files` section in `setup.py` to install the resource file. Colcon currently might warn:

  ```
  WARNING: Package 'your_package' doesn't explicitly install a marker in the package index...
  ```

  This warns that in the future it won't auto-create it. The fix is to include the snippet in setup.py as we showed. For C++ packages, forgetting `ament_package()` can lead to a missing marker; ensure it's in CMakeLists.
* **Console Script Not Found:** If you can see your package with `ros2 pkg list` but `ros2 run my_package my_script` says the script is not found, then likely the `entry_points` in setup.py wasn't set or the setup.py didn't execute. Check that the console\_scripts entry matches the name you're using and that you rebuilt after adding it. For C++, if an executable is not found, ensure the target was installed in CMake.
* **Mixing Release and Debug or Multiple Overlays:** Sometimes, especially with C++ code or when mixing source builds of core packages with binaries, you can have library path issues. For example, if you built a custom version of a core library in your workspace, you must source that workspace *before* using it, or ROS 2 might still use the system-installed version, causing ABI mismatches. A typical scenario: you build a newer `rclcpp` or `rosbag2` from source in your overlay but forget to source it, then your nodes might load the wrong .so file. Always ensure you're using the intended version of each package by sourcing order or not mixing duplicate packages.
* **Cleaning and Rebuilding:** If you make changes to package.xml or setup.py and things still seem off, try cleaning the workspace. You can remove the `build/`, `install/`, and `log/` directories (or use `colcon build --clean` if available) and rebuild fresh. This ensures no stale artifacts are causing confusion.
* **Using `colcon list`:** A handy command is `colcon list` (in the workspace root). It will list all packages colcon sees and their build types. Use this to verify your new package is being detected and that its type is correct (it will show `ament_python` or `ament_cmake`). If your package doesn't appear in `colcon list`, then either the directory is not in the `src` folder, or the `package.xml` is malformed (colcon couldn't parse it). Fix the manifest or location as needed.

### Case Study: Rosbag2 Composable Node Integration Issue

Consider a real example: integrating ROS 2's rosbag2 (the recording and playback tool) in a workspace to use its components. Rosbag2 provides composable nodes (C++ components) for the recorder and player. Suppose you want to customize rosbag2 or use it alongside your packages, so you include rosbag2 source in your workspace or build a package that depends on it.

We encountered a problem where after building rosbag2 from source in our workspace, trying to load the rosbag2 recorder as a component in a launch file failed. The error was along the lines of *“Failed to load component… class not found”* when using `ros2 component load` or a composable launch. This was puzzling because the rosbag2 packages were built and present.

After investigation, a few issues were identified:

* The **plugin description file** (which lists the component plugin for rosbag2 recorder/player) was not being found. In rosbag2, these are usually installed in the `share/` directory of the package and registered in the ament index under `resources` for pluginlib. If the installation or index registration wasn't correct, the component loader can't discover the class.
* There was a **version mismatch**: the user had rosbag2 installed via binaries (apt) and also built from source. If the environment was not carefully managed, the running system might have been mixing the two - for example, trying to load the component from the wrong library (.so). The ROS 2 environment might have found the binary version's plugin description, but the source-built library or vice versa, causing an undefined symbol error or missing class error.
* It turned out that the user's new workspace was not fully overriding the system installation because they hadn't sourced the local workspace or they only built some parts of rosbag2 but not others (like maybe the `rosbag2_composition` package which provides the loading functionality).

**Resolution:** The solution in that case was to ensure:

* All relevant rosbag2 packages were built in the workspace (including any plugin or composition-related ones).
* The workspace `install/setup.bash` was properly sourced *and* that no conflicting sourcing of the system installation came afterward to override it.
* Verify that the plugin XML (for rosbag2 components, this might be installed as `share/rosbag2_composition/rosbag2_composition.xml` or similar) is present in the install space and that the `ament_index/resource_index` has an entry for it (pluginlib uses the ament index to find the plugin xml by package name).
* Use `ros2 pkg prefix rosbag2_composition` to see which path ROS 2 is using for that package - it should point to your workspace install, not `/opt/ros/humble`. If it doesn't, then your overlay didn't override the binary (perhaps you didn't build that package, or the overlay order is wrong).
* In the end, rebuilding with all components and carefully sourcing fixed the issue. The rosbag2 recorder could then be loaded as a component successfully.

**General lesson:** When integrating a complex package like rosbag2:

* Make sure to include all necessary sub-packages and plugins.
* Avoid mixing source and binary versions of the same package in the runtime environment; it's usually all or nothing to prevent conflicts.
* If a component fails to load with an undefined symbol, check that you've linked all dependent libraries and that the component was built against the correct version of those libraries (no ABI incompatibility). Undefined symbols often mean a missing dependency in the CMake or a version mismatch.
* Leverage ROS 2 introspection tools: `ros2 pkg prefix`, `ros2 pkg xml`, or just inspecting the `install` directory to ensure files are in place.

By analyzing the rosbag2 issue, we see that most integration problems boil down to either build configuration errors (not building or installing something correctly) or environment setup mistakes (not sourcing or mixing installs). Systematically checking each of these aspects will resolve the majority of “my new package doesn't work” scenarios.

## Summary

Building a new ROS 2 package in an existing workspace involves careful setup of the package's manifest and build files, using colcon to compile and install it, and then making sure the environment is updated to include it. In this guide, we created an example package `my_package` using Python (ament\_python) and highlighted how to configure `package.xml`, `setup.py`, and related files. We also touched on how a C++ (ament\_cmake) package setup differs, including the need for `CMakeLists.txt` and proper CMake macros.

Key takeaways:

* Always specify the correct `<build_type>` in package.xml (`ament_python` for Python, `ament_cmake` for C++). Incorrect types can lead to your package not being recognized by ROS 2.
* Ensure all dependencies are declared in package.xml and that for Python packages you explicitly install the ament index resource marker (so the package is discoverable by ROS 2 tools).
* Use `colcon build` from the workspace root and source the workspace's setup script after building. Without sourcing, your new package remains invisible to ROS 2.
* Verify integration by listing and running the package's nodes. If something is not found, retrace your steps: check installation paths, entry points, and environment sourcing.
* Common issues like missing dependencies, not installing executables, or mixing build systems can be debugged by inspecting build logs and the install directory. Leverage community Q\&A and official docs when stuck on specific errors - often the solution is a small fix in the manifest or CMake/setup file.
* The rosbag2 case study illustrated that even when everything builds, runtime issues can occur if the environment is inconsistent. Always ensure your workspace overlay is properly applied and avoid conflicting installations when possible.

By following this tutorial and the best practices outlined, you should be able to confidently add new packages to your ROS 2 Humble workspace and have them work seamlessly with existing packages.

## See Also:

- [ROS Introduction](/wiki/common-platforms/ros/ros-intro)
- [ROS 2 Humble IPC Recorder](/wiki/tools/ros2-humble-ipc-recorder)
- [Building an iOS App for ROS2 Integration – A Step-by-Step Guide](/wiki/common-platforms/ros2_ios_app_with_swift)

## Further Reading

- [ROS 2 Tutorial - Writing a Simple Publisher and Subscriber (Python)](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html)  
- [ROS 2 Tutorial - Writing a Simple Publisher and Subscriber (C++)](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html)  
- [ROS2 Create ROS Package with Colcon](https://www.youtube.com/watch?v=4zGUDisw4UI)
- [ROS2 Build Packages with Colcon](https://www.youtube.com/watch?v=KLvUMtYI_Ag)
- [ROS2 Publisher and Subscriber Package C++](https://www.youtube.com/watch?v=rGsyQHwWObA)
- [CMake Tutorial for Absolute Beginners - From GCC to CMake including Make and Ninja](https://www.youtube.com/watch?v=NGPo7mz1oa4)


## References

\[1] Open Robotics, “*Creating a ROS 2 Package (Humble)*,” *ROS 2 Tutorials*, 2022. Accessed: May 3 2025. \[Online]. Available: [https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)

\[2] Open Robotics, “*Using colcon to build packages*,” *ROS 2 Documentation (Foxy)*, May 12 2020. Accessed: May 3 2025. \[Online]. Available: [https://docs.ros.org/en/foxy/Tutorials/Colcon-Tutorial.html](https://docs.ros.org/en/foxy/Tutorials/Colcon-Tutorial.html)


