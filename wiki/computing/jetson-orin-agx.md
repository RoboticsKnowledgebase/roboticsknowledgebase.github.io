---
# Jekyll 'Front Matter'
# Most fields are set by default and should NOT be overwritten except in special circumstances.
# Set the date the article was last updated:
date: 2025-05-04  # YYYY-MM-DD
# This will be displayed at the bottom of the article
# Set the article's title:
title: Jetson Orin AGX
# The 'title' is automatically displayed at the top of the page and used elsewhere.
---

This page familiarizes you with the capabilities of the NVIDIA Jetson Orin AGX, highlights the best resources for support, outlines its strengths and weaknesses, and explains how to leverage this SoC effectively. It offers a high-level overview along with deep dives into essential components and practical examples.

## Introduction to the Orin AGX

The NVIDIA Orin AGX 64GB is NVIDIA’s edge-compute platform, engineered for machine learning workloads with an emphasis on power efficiency. It has configurable power modes and wide input-voltage tolerance which lets you balance performance and energy consumption. Most importantly, it has 64 GB of unified memory (shared by CPU and GPU), which simplifies data movement and maximizes throughput.

### Comparing CPU vs. Accelerated Hardware

- **CPU**: Ideal for sequential code, system services, control loops, and tasks that lack parallelism. It delivers predictable latency but lower overall throughput for data-parallel workloads.  
- **GPU/DLA/PVA/SPE**: These accelerators excel at parallel processing, vision, and neural-network inference. Offloading heavy kernels (e.g., convolution, stereo depth, sensor fusion) to specialized engines dramatically boosts performance and frees the CPU for control logic.

Use the CPU when you need deterministic execution or when coding small, control-oriented routines. Offload bulk data processing—such as deep learning inference and image pipelines—to the GPU (CUDA/Tensor Cores), DLA (INT8/FP16 inference), PVA (vision primitives), or SPE (real-time sensor tasks) to maximize efficiency.

## Hardware Peripherals

The Jetson Orin AGX SoC integrates:

1. **CPU** – 12-core Arm Cortex-A78AE  
2. **GPU** – NVIDIA Ampere architecture with 2048 CUDA cores and 64 Tensor Cores  
3. **DLA** – 2 dedicated Deep Learning Accelerators  
4. **PVA** – 1 Programmable Vision Accelerator  
5. **SPE** – Sensor Processing Engine (Cortex-R5 in the Always-On cluster)  
6. **Memory** – 64 GB LPDDR5 unified memory  
7. **Storage** – Onboard 64 GB (eMMC/NVMe) + M.2 PCIe NVMe expansion  
8. **I/O** – 40-pin GPIO header, USB, Ethernet, DisplayPort, and more  

### CPU – Central Processing Unit

**Use cases**: System management, UI rendering, control loops, and tasks with limited parallelism.

- 12× Arm Cortex-A78AE cores  (AE stands for Automotive Efficiency)
- Hardware virtualization support  

### GPU – Graphics Processing Unit

**Use cases**: Parallel compute, CUDA kernels, graphics, and deep-learning training/fine-tuning.

- Ampere GPU: 2048 CUDA cores, 64 Tensor Cores  
- Up to 5.3 TFLOPS (FP32), ~170 TOPS (INT8 sparse)  

### DLA – Deep Learning Accelerator

**Use cases**: High-efficiency INT8/FP16 neural-network inference (via TensorRT).

- 2× NVDLA v2.0 engines  
- ~105+ TOPS INT8 each  

### PVA – Programmable Vision Accelerator

**Use cases**: Vision primitives (resizing, filtering, feature detection, stereo matching).

- Dual-7-way VLIW engines + DMA + embedded R5 core  
- Accessible via NVIDIA VPI library  

### SPE – Sensor Processing Engine

**Use cases**: Real-time, low-power sensor handling, wake-on-sensor tasks, IMU preprocessing.

- Cortex-R5 @ 200 MHz, 256 KB SRAM  
- Part of the Always-On cluster for sub-20 μs response  

### Memory & Storage

- **64 GB LPDDR5** unified memory (204.8 GB/s)  
- **Expandable storage** via M.2 NVMe (up to 2 TB typical)  

### I/O Interfaces

#### 40-Pin Header

- Raspberry Pi–compatible breakout with GPIO, UART, I²C, SPI, I²S, PWM, CAN, DMIC  
- Pinmux and function mapping via the Jetson Orin PinMux spreadsheet  
- Pinout diagram: [JetsonHacks GPIO Pinout](https://jetsonhacks.com/nvidia-jetson-agx-orin-gpio-header-pinout/)  
- Detailed guide: [NVIDIA Orin Pin Function Guide (PDF)](https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx_orin/Jetson_AGX_Orin_Pin_Function_Names_Guide_DA-11060-001_v1.0.pdf)  

#### USB Ports

- 2× USB 3.2 Gen 2 Type-A (next to Ethernet)  
- 1× USB 3.2 Gen 2 Type-C (dual-role for power/flashing)  
- 1× USB 3.2 Gen 2 Type-A + 1× USB 2.0 Micro-B (debug/serial)  

#### Ethernet

- 1× RJ45 10GBASE-T  

#### Video Output

- 1× DisplayPort 1.4a  

## Software Environment

The platform runs **L4T (Linux for Tegra)**, an Ubuntu-based OS that includes **JetPack** (CUDA, cuDNN, TensorRT, multimedia APIs, and more). Jetpack is the jetson Software Development Kit.

### Initial Setup

1. **Host requirements**: Ubuntu 20.04/22.04 x86_64, NVIDIA SDK Manager or L4T BSP package, development tools (`dtc`, Python 3).  
2. **Recovery mode**: Connect USB-C (host), hold **RECOVER** button, then power on.  
3. **Flash L4T**:  
   ```bash
   sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1
   ```


## Kernel Customization: GPIO & Pinmux

Orin’s pins are multiplexed by default. To reassign pins or enable interfaces:

- **Jetson-IO tool** (`sudo jetson-io.py`): Interactive CLI/GUI to configure header pins and generate device-tree overlays.  
- **Device Tree Overlays**: Write or edit `.dts` fragments in `/boot/extlinux/extlinux.conf` to enable I²C, SPI, UART, or GPIO.  
- **Sysfs GPIO**: Export and control via `/sys/class/gpio`:
  ```bash
  echo 139 | sudo tee /sys/class/gpio/export
  echo out | sudo tee /sys/class/gpio/gpio139/direction
  echo 1 | sudo tee /sys/class/gpio/gpio139/value
  ```  

## Using the SPE for PPS (Pulse Per Second)

Pulse Per Second if a 1Hz pulse that can be sent to other systems to sync their internal clocks and to avoid drift. [NVIDIA PPS Documentation](https://docs.nvidia.com/networking/display/nvidia5ttechnologyusermanualv10/pulse+per+second+(pps))

For applications requiring precise timing, use the SPE to generate a 1 Hz PPS signal:

1. **Enable PPS pins** (GPIO pins 16 and 32).  
2. Install and run `pps-tools`, then `ppscheck` to verify the PPS source.  
3. Record the exact timestamp of each pulse in shared memory accessible to both CPU and SPE.  
4. Use this timestamp to accurately tag sensor data (e.g., camera frames).  

## Getting Started with the SPE

1. Download the GNU Arm toolchain: [Arm GNU Toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads/14-2-rel1)  
2. Build and flash your R5 firmware following NVIDIA’s guide: [Real-Time Compiling Guide](https://docs.nvidia.com/jetson/spe/rt-compiling.html#autotoc_md0)  

Offloading sensor handling to the SPE reduces CPU/GPU load, improving overall system responsiveness and power efficiency.


## Useful Links
| Link | Use |
|-|-|
| `https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/Reference_Documentation` | All Jetson Orin downloads |
| `(https://developer.download.nvidia.com/assets/embedded/secure/jetson/agx_orin/Orin-TRM_DP10508002_v1.2p.pdf?__token__=exp=1746415687~hmac=df824d5111d503a71811a2fd8a2ea1d8fea81f0fa90a3aed4ba650e36812abd9&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9)` | The Technical Reference manual |
|`(https://developer.nvidia.com/embedded/jetpack)`| Jetpack SDK download tutorial |
| `https://developer.nvidia.com/embedded/linux-tegra-r321` | L4T guide + installation tutorial |
|`https://jetsonhacks.com/`|Tons of useful tutorials and software packages|

---

*Last updat