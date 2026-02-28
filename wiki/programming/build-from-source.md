---
# Jekyll 'Front Matter' goes here. Most are set by default, and should NOT be
# overwritten except in special circumstances. 
# You should set the date the article was last updated like this:
date: 2020-05-04 # YYYY-MM-DD
# This will be displayed at the bottom of the article
# You should set the article's title:
title: Building software from source
# The 'title' is automatically displayed at the top of the page
# and used in other parts of the site.
---

This tutorial describes how to configure and build libraries from source for your unique system.
This is often necessary when your software requirements are not met by the packages available in the repositories of
your distribution (the version of the package you want is not available from `apt` for your version of Ubuntu, for
example).
This tutorial will cover the process for customizing and building packages from source as you need them.
Along the way, common pitfalls and how to avoid them will be discussed.
You will also discover how to avoid polluting your system environment to avoid conflicts between the packages you build
and the ones installed by your package manager.

## Why build your packages from source?

There are several reasons you might want to build packages from source, a few of these are listed below:

- The package (+ version) you want is not available from your system package manager.
- You need a customized build of the package (building OpenCV with CUDA support for example).
- For optimized builds that fully utilize all the features your hardware has to offer.

## Process outline and key pieces

To build software from source code, these are the broad steps that must be followed:

- Obtain the source code: Usually `git clone` and `git checkout` and then apply patches if necessary.
- Get dependencies: Use your system package manager or build the packages that this package depends on so they are
  available to this package.
- Configure the build: Use `cmake` to set options for the build, including which features to compile with and the path
  where the package must be installed.
- Build and install the package: Use `cmake` to build and install the package.
- Linking against the built package: Set environment/cmake variables so that other packages can find the package
  that you just built.

This tutorial will use the popular computer-vision library OpenCV as a guiding example due to its popularity and
complexity, which will concretely demonstrate all steps.

## Step 0: Prerequisites

To build software, the system needs to have the following installed:

- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/) as a preferred build tool.
- A C/C++ compiler, linker, and standard libraries/headers (usually preinstalled).

This collection is called the "toolchain".
On Ubuntu all of this can be installed with:

```bash
sudo apt install build-essential
```

## Step 1: Getting the source code

### Which version do I require

If this package is being compiled to satisfy the dependency of other software that you need, you should check which
exact version that dependent package needs. This can be done by analyzing the CMakeLists.txt file of the dependent (
final) package, and finding a line resembling:

```cmake
find_package(OpenCV 4 ...)
```

This tells you that this package requires OpenCV 4 (Any 4.x.y version works).

Get source code for this version.

### Cloning source code

Get the source code on your computer into a well-organized folder (in a separate directory, that directory is alongside
directories that contain source-code for other packages).

Instructions should be present on the package website. Usually, this involves:

- `git clone` the package.
- `git checkout` to the release that is desired, e.g. `git checkout 4.11.0` to get that version of OpenCV, obtained from
  the GitHub releases page.

Here you would modify the source code (such as apply any quick fixes/patches as required).

**Also follow any prescribed instructions as the package demands**

To build OpenCV with the contrib feature set, we must also download and extract some more content. Follow instructions
listed on the website.

## Step 2: Get dependencies

This library will depend on other software. Get that software, usually from your package manager, by running a command
similar to (on ubuntu):

```bash
sudo apt install lib-<libname>-dev
```

For example, to get the "png" dependency to allow OpenCV to read png images, just run:

```bash
sudo apt install libpng-dev
```

You can proceed to the next step, and install these as CMake will complain that it did not find a dependency.

## Step 3: Configure the build

This is the crucial step. Here, you decide how the package needs to be compiled, this includes:

- Which features you need: In OpenCV, for example, which image formats (png, jpg, etc.) you want to support
  reading/writing to?
- Do you want OpenCV to support CUDA?
- Do you want to build support for Python?
- Disable/enable features (GUI support, for example).

### Setting the environment

Be in the environment under which you will run this package.

Concretely, this means: If you want OpenCV to be built in a particular Python virtual environment, in your terminal,
source this Python Environment.

Also set `CMAKE_PREFIX_PATH` (explained later) to allow CMake to find other packages.

### Set Options

Now, create a well-organized build directory, where build time intermediates will be stored (these are temporary files
emitted and
consumed by the compiler and linker on the way to making the final binaries and libraries).

Change into the build directory and configure the package as follows:

```bash
cmake -S <path_to_source_code> -DCMAKE_INSTALL_PREFIX=<path_to_install_directory> -DOPENCV_EXTRA_MODULES_PATH=../../src/OpenCV/opencv_contrib-4.x/modules -DWITH_TIFF=ON -DCMAKE_BUILD_TYPE=Release
```

Let's break down this command:

- `-S` is where the source code to be compiled it. This directory must contain the `CMakeLists.txt` file.
- `-D<option>=<value>` The options being configured. Most are package-specific (such as `-DOPENCV_EXTRA_MODULES_PATH`
  and `-DWITH_TIFF=ON`). Some package-agnostic options (the `-DCMAKE_<option>=<value>`) are elaborated upon below:
    - `-DCMAKE_INSTALL_PREFIX=<install_directory>`: This option tells CMake teh directory under which to place all the
      libraries, binaries, headers and auxiliary files. By default, CMake will try to install to your root system and
      this can cause conflicts with the package manager. **It is highly recommended to set this to a well-organized
      location.**
    - `-DCMAKE_BUILD_TYPE=Release`: This tells CMake to build the package in "Release" mode, i.e. enable optimizations
      and disable debug information. **Set this for optimal performance**.

Upon running the above command, CMake will check your system to see if everything is available.
If it complains about not finding something, wither install it via the method described in the section above, or build
it from source and include its install prefix in the `CMAKE_PREFIX_PATH` environment variable.

If this step is successful, proceed to building and installing the package.

## Step 4: Build and install the package

To build and install the package, simply run the following from the build directory:

```bash
cmake --build . --target install
```

This will start compilation and linking and install the library to the directory specified in the configuration step.

Compiling is compute intensive and large packages can take a long time (5-6 hours to compile PyTorch on an Orin).

Errors in this step can be dealt with by consulting GitHub issues and searching the internet. Common fixes include
installing a more modern compiler, changing versions of dependencies or changing package source code.

When this step completes, the install directory will contain:

```text
├── bin
├── include
├── lib
├── lib64
└── share
```

Note that multiple packages can install to the same install directory.

## Step 5: Link against the built package

You most likely built this package as a dependency to some other target package you need to build.

While building your target package, you need to inform CMake where to find the dependency you just built.

To achieve this, before configuring the target package, add the install directory of the package you just built to the
`CMAKE_PREFIX_PATH` environment variable, like so in bash:

```bash
export CMAKE_PREFIX_PATH="<install_directory>:$CMAKE_PREFIX_PATH"
```

Here, <install_directory> is the same as specified in the variable
`-DCMAKE_INSTALL_PREFIX` during configuration, which has the structure mentioned above below it.

After setting this environment variable, continue the configure process as usual, and CMake will pick up the dependency
from where you installed it.

While running the library, you might encounter errors about not finding a particular `.so` file. To fix these, simply
add the `<install_directory>/lib/` or the `<install_directory>/lib64/` directory to the `LD_LIBRARY_PATH` environment
variable to enable the Linux Dynamic Linker to find the dynamic libraries where you installed them.
An example of this process is

```bash
export LD_LIBRARY_PATH="<install_directory>/lib:$LD_LIBRARY_PATH"
```

## Summary

This tutorial went over the exact process to build `cmake` enabled projects from source.

The ability to do this reduces your reliance on the package manager and helps you get out of dependency issues.

Moreover, using the above framework you can install and use multiple versions of the same library on your system.

## See Also:

- [A short guide on using CMake](https://roboticsknowledgebase.com/wiki/programming/cmake/)