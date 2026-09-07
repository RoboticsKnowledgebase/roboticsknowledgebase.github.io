var GRAPH_DATA = {
  categories: [
    { id: "cat-bootcamp", label: "Bootcamp", color: "#66BB6A", url: "/wiki/bootcamp/" },
    { id: "cat-project-guide", label: "Project Guide", color: "#FFA726", url: "/wiki/robotics-project-guide/master-guide/" },
    { id: "cat-system-design", label: "System Design", color: "#AB47BC", url: "/wiki/system-design-development/" },
    { id: "cat-project-mgmt", label: "Project Mgmt", color: "#BCAAA4", url: "/wiki/project-management/" },
    { id: "cat-platforms", label: "Platforms", color: "#26A69A", url: "/wiki/common-platforms/" },
    { id: "cat-sensing", label: "Sensing", color: "#42A5F5", url: "/wiki/sensing/" },
    { id: "cat-actuation", label: "Actuation", color: "#EF5350", url: "/wiki/actuation/" },
    { id: "cat-manipulators", label: "Manipulators", color: "#EC407A", url: "/wiki/manipulators/" },
    { id: "cat-ml", label: "Machine Learning", color: "#FFCA28", url: "/wiki/machine-learning/" },
    { id: "cat-rl", label: "Reinforcement Learning", color: "#F48FB1", url: "/wiki/reinforcement-learning/" },
    { id: "cat-state-est", label: "State Estimation", color: "#7E57C2", url: "/wiki/state-estimation/" },
    { id: "cat-programming", label: "Programming", color: "#00ACC1", url: "/wiki/programming/" },
    { id: "cat-networking", label: "Networking", color: "#5C6BC0", url: "/wiki/networking/" },
    { id: "cat-simulation", label: "Simulation", color: "#78909C", url: "/wiki/simulation/" },
    { id: "cat-interfacing", label: "Interfacing", color: "#9CCC65", url: "/wiki/interfacing/" },
    { id: "cat-computing", label: "Computing", color: "#FFB300", url: "/wiki/computing/" },
    { id: "cat-fabrication", label: "Fabrication", color: "#D4E157", url: "/wiki/fabrication/" },
    { id: "cat-math", label: "Math", color: "#E040FB", url: "/wiki/math/" },
    { id: "cat-tools", label: "Tools", color: "#8D6E63", url: "/wiki/tools/" },
    { id: "cat-datasets", label: "Datasets", color: "#80DEEA", url: "/wiki/datasets/" },
    { id: "cat-planning", label: "Planning", color: "#FF7043", url: "/wiki/planning/" }
  ],

  articles: [
    // ── Bootcamp ──
    { id: "art-bootcamp-guide", label: "Bootcamp Guide", category: "cat-bootcamp", url: "/wiki/bootcamp/bootcamp-guide/" },

    // ── Robotics Project Guide ──
    { id: "art-define-goals", label: "Define Goals & Requirements", category: "cat-project-guide", url: "/wiki/robotics-project-guide/define-your-goals-and-requirements/" },
    { id: "art-choose-robot", label: "Choose a Robot", category: "cat-project-guide", url: "/wiki/robotics-project-guide/choose-a-robot/" },
    { id: "art-make-robot", label: "Make a Robot", category: "cat-project-guide", url: "/wiki/robotics-project-guide/make-a-robot/" },
    { id: "art-choose-language", label: "Choose a Language", category: "cat-project-guide", url: "/wiki/robotics-project-guide/choose-a-language/" },
    { id: "art-robot-comm", label: "Robot Communication", category: "cat-project-guide", url: "/wiki/robotics-project-guide/choose-comm/" },
    { id: "art-choose-sim", label: "Choose a Simulator", category: "cat-project-guide", url: "/wiki/robotics-project-guide/choose-a-sim/" },
    { id: "art-test-debug", label: "Test & Debug", category: "cat-project-guide", url: "/wiki/robotics-project-guide/test-and-debug/" },
    { id: "art-demo-day", label: "Demo Day", category: "cat-project-guide", url: "/wiki/robotics-project-guide/demo-day/" },
    { id: "art-humanoid-robot", label: "Humanoid Robot Overview", category: "cat-project-guide", url: "/wiki/robotics-project-guide/humanoid-robot/" },

    // ── System Design & Development ──
    { id: "art-system-engineering", label: "System Engineering", category: "cat-system-design", url: "/wiki/system-design-development/system-engineering/" },
    { id: "art-fsm", label: "Finite State Machines", category: "cat-system-design", url: "/wiki/system-design-development/finite-state-machine/" },
    { id: "art-mech-design", label: "Mechanical Design", category: "cat-system-design", url: "/wiki/system-design-development/mechanical-design/" },
    { id: "art-pcb-design", label: "PCB Design", category: "cat-system-design", url: "/wiki/system-design-development/pcb-design/" },
    { id: "art-cable-mgmt", label: "Cable Management", category: "cat-system-design", url: "/wiki/system-design-development/cable-management/" },
    { id: "art-subsystem-interface", label: "Subsystem Interface Modeling", category: "cat-system-design", url: "/wiki/system-design-development/subsystem-interface-modeling/" },
    { id: "art-in-loop-testing", label: "In Loop Testing", category: "cat-system-design", url: "/wiki/system-design-development/In-Loop-Testing/" },

    // ── Project Management ──
    { id: "art-drone-permits", label: "Drone Flight Permissions", category: "cat-project-mgmt", url: "/wiki/project-management/drone-flight-permissions/" },
    { id: "art-product-dev", label: "Product Dev in Complex Systems", category: "cat-project-mgmt", url: "/wiki/project-management/product-development-complex-systems/" },
    { id: "art-risk-mgmt", label: "Risk Management", category: "cat-project-mgmt", url: "/wiki/project-management/risk-management/" },
    { id: "art-jira", label: "Jira", category: "cat-project-mgmt", url: "/wiki/project-management/jira/" },
    { id: "art-notion", label: "Notion for Project Management", category: "cat-project-mgmt", url: "/wiki/project-management/using-notion-for-project-management/" },

    // ── Common Platforms ──
    { id: "art-dji-breakdown", label: "DJI Drone Breakdown", category: "cat-platforms", url: "/wiki/common-platforms/dji-drone-breakdown-for-technical-projects/" },
    { id: "art-custom-drone-darpa", label: "Custom Drone DARPA", category: "cat-platforms", url: "/wiki/common-platforms/building-custom-drone-for-darpa-triage-challenge/" },
    { id: "art-dji-sdk", label: "DJI SDK", category: "cat-platforms", url: "/wiki/common-platforms/dji-sdk/" },
    { id: "art-unitree-g1", label: "Unitree G1", category: "cat-platforms", url: "/wiki/common-platforms/unitree-g1/" },
    { id: "art-unitree-go1", label: "Unitree Go1", category: "cat-platforms", url: "/wiki/common-platforms/unitree-go1/" },
    { id: "art-nvidia-orin", label: "Nvidia Orin Interface", category: "cat-platforms", url: "/wiki/common-platforms/interfacing-with-nvidia-orin/" },
    { id: "art-pixhawk", label: "Pixhawk", category: "cat-platforms", url: "/wiki/common-platforms/pixhawk/" },
    { id: "art-asctec-pelican", label: "Asctec Pelican UAV", category: "cat-platforms", url: "/wiki/common-platforms/asctec-uav-setup-guide/" },
    { id: "art-husky", label: "Husky Interface", category: "cat-platforms", url: "/wiki/common-platforms/husky_interfacing_and_communication/" },
    { id: "art-ur5e", label: "UR5e Robotic Arm", category: "cat-platforms", url: "/wiki/common-platforms/ur5e/" },
    { id: "art-ros-intro", label: "ROS Introduction", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-intro/" },
    { id: "art-ros-arduino", label: "ROS Arduino Interface", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-arduino-interface/" },
    { id: "art-ros-motion-server", label: "ROS Motion Server", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-motion-server-framework/" },
    { id: "art-ros-cost-maps", label: "ROS Cost Maps", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-cost-maps/" },
    { id: "art-ros-mapping", label: "ROS Mapping & Localization", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-mapping-localization/" },
    { id: "art-ros-navigation", label: "ROS Navigation", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-navigation/" },
    { id: "art-ros-global-planner", label: "ROS Global Planner", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-global-planner/" },
    { id: "art-ros-unsupported-os", label: "ROS on Unsupported OS", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros-unsupported-os" },
    { id: "art-rc-cars", label: "Making RC Cars Autonomous", category: "cat-platforms", url: "/wiki/common-platforms/rccars-the-complete-guide/" },
    { id: "art-khepera4", label: "Khepera 4", category: "cat-platforms", url: "/wiki/common-platforms/khepera4/" },
    { id: "art-ros2-husky-nav", label: "ROS2 Navigation Husky", category: "cat-platforms", url: "/wiki/common-platforms/ros2-navigation-for-clearpath-husky/" },
    { id: "art-hello-robot", label: "Hello Robot Stretch RE1", category: "cat-platforms", url: "/wiki/common-platforms/hello-robot" },
    { id: "art-ros2-ios", label: "iOS App for ROS2", category: "cat-platforms", url: "/wiki/common-platforms/ros2-ios-app-with-swift/" },
    { id: "art-vscode-ros2", label: "VS Code for ROS2", category: "cat-platforms", url: "/wiki/common-platforms/configure-vscode-for-ros2/" },
    { id: "art-ros2-custom-pkg", label: "ROS2 Custom Packages", category: "cat-platforms", url: "/wiki/common-platforms/ros/ros2-custom-package/" },

    // ── Sensing ──
    { id: "art-trajectory-extract", label: "Traffic Camera Tracking", category: "cat-sensing", url: "/wiki/sensing/trajectory_extraction/" },
    { id: "art-adafruit-gps", label: "Adafruit GPS", category: "cat-sensing", url: "/wiki/sensing/adafruit-gps/" },
    { id: "art-apriltags", label: "AprilTags", category: "cat-sensing", url: "/wiki/sensing/apriltags/" },
    { id: "art-stag", label: "STag Markers", category: "cat-sensing", url: "/wiki/sensing/stag/" },
    { id: "art-camera-calib", label: "Camera Calibration", category: "cat-sensing", url: "/wiki/sensing/camera-calibration/" },
    { id: "art-handeye-calib", label: "Hand-Eye Calibration", category: "cat-sensing", url: "/wiki/sensing/handeye-calibration/" },
    { id: "art-cv-considerations", label: "Computer Vision Considerations", category: "cat-sensing", url: "/wiki/sensing/computer-vision-considerations/" },
    { id: "art-delphi-radar", label: "Delphi ESR Radar", category: "cat-sensing", url: "/wiki/sensing/delphi-esr-radar/" },
    { id: "art-pcl", label: "Point Cloud Library", category: "cat-sensing", url: "/wiki/sensing/pcl/" },
    { id: "art-photometric-calib", label: "Photometric Calibration", category: "cat-sensing", url: "/wiki/sensing/photometric-calibration/" },
    { id: "art-speech-recognition", label: "Speech Recognition", category: "cat-sensing", url: "/wiki/sensing/speech-recognition/" },
    { id: "art-stereo-vision", label: "Stereo Vision OpenCV", category: "cat-sensing", url: "/wiki/sensing/opencv-stereo/" },
    { id: "art-camera-imu-calib", label: "Camera-IMU Calibration", category: "cat-sensing", url: "/wiki/sensing/camera-imu-calibration/" },
    { id: "art-fiducial-markers", label: "Fiducial Markers", category: "cat-sensing", url: "/wiki/sensing/fiducial-markers/" },
    { id: "art-rtk-gps", label: "RTK GPS", category: "cat-sensing", url: "/wiki/sensing/gps/" },
    { id: "art-realsense", label: "Intel RealSense", category: "cat-sensing", url: "/wiki/sensing/realsense/" },
    { id: "art-total-station", label: "Robotic Total Station", category: "cat-sensing", url: "/wiki/sensing/robotic-total-stations/" },
    { id: "art-thermal-cameras", label: "Thermal Cameras", category: "cat-sensing", url: "/wiki/sensing/thermal-cameras/" },
    { id: "art-azure-block", label: "Azure Block Detection", category: "cat-sensing", url: "/wiki/sensing/azure-block-detection/" },
    { id: "art-uwb-positioning", label: "UWB Positioning", category: "cat-sensing", url: "/wiki/sensing/ultrawideband-beacon-positioning/" },
    { id: "art-sensor-noise", label: "Reducing Sensor Noise", category: "cat-sensing", url: "/wiki/sensing/sensor-noise/" },
    { id: "art-apple-vision-sensing", label: "Apple Vision Pro (Sensing)", category: "cat-sensing", url: "/wiki/sensing/apple-vision-pro/" },
    { id: "art-hololens", label: "Microsoft HoloLens 2", category: "cat-sensing", url: "/wiki/sensing/hololens-101/" },
    { id: "art-thermal-perception", label: "Thermal Perception", category: "cat-sensing", url: "/wiki/sensing/thermal-perception/" },

    // ── Controls & Actuation ──
    { id: "art-motor-feedback", label: "Motor Controller Feedback", category: "cat-actuation", url: "/wiki/actuation/motor-controller-feedback/" },
    { id: "art-pid-arduino", label: "PID Control on Arduino", category: "cat-actuation", url: "/wiki/actuation/pid-control-arduino/" },
    { id: "art-linear-actuator", label: "Linear Actuator Types", category: "cat-actuation", url: "/wiki/actuation/linear-actuator-resources/" },
    { id: "art-uln2003a", label: "ULN2003A Motor Controller", category: "cat-actuation", url: "/wiki/actuation/uln2003a-motor-controller/" },
    { id: "art-vedder-esc", label: "Vedder ESC", category: "cat-actuation", url: "/wiki/actuation/vedder-electronic-speed-controller/" },
    { id: "art-pure-pursuit", label: "Pure Pursuit Controller", category: "cat-actuation", url: "/wiki/actuation/Pure-Pursuit-Controller-for-Skid-Steering-Robot/" },
    { id: "art-moveit-hebi", label: "MoveIt & HEBI Integration", category: "cat-actuation", url: "/wiki/actuation/moveit-and-HEBI-integration/" },
    { id: "art-mpc", label: "Model Predictive Control", category: "cat-actuation", url: "/wiki/actuation/model-predictive-control/" },
    { id: "art-task-priority", label: "Task Prioritization Control", category: "cat-actuation", url: "/wiki/actuation/task-prioritization-control/" },
    { id: "art-drive-by-wire", label: "Drive-by-Wire Conversion", category: "cat-actuation", url: "/wiki/actuation/drive-by-wire/" },

    // ── Manipulators ──
    { id: "art-kuka-moveit2", label: "KUKA LBR with MoveIt2", category: "cat-manipulators", url: "/wiki/manipulators/integrating-kuka-lbr-manipulators-with-moveit2/" },

    // ── Machine Learning ──
    { id: "art-darknet", label: "Training Darknet", category: "cat-ml", url: "/wiki/machine-learning/train-darknet-on-custom-dataset/" },
    { id: "art-custom-semantic", label: "Custom Semantic Data", category: "cat-ml", url: "/wiki/machine-learning/custom-semantic-data/" },
    { id: "art-python-rl-libs", label: "Python RL Libraries", category: "cat-ml", url: "/wiki/machine-learning/python-libraries-for-reinforcement-learning/" },
    { id: "art-intro-rl", label: "Intro to RL", category: "cat-ml", url: "/wiki/machine-learning/intro-to-rl/" },
    { id: "art-diffusion", label: "Diffusion Models & Policy", category: "cat-ml", url: "/wiki/machine-learning/intro-to-diffusion/" },
    { id: "art-grpo-diffusion", label: "GRPO for Diffusion", category: "cat-ml", url: "/wiki/machine-learning/grpo-diffusion-policies/" },
    { id: "art-yolo-ros", label: "YOLO + ROS + CUDA", category: "cat-ml", url: "/wiki/sensing/ros-yolo-gpu/" },
    { id: "art-yolov5-tensorrt", label: "YOLOv5 on Jetson", category: "cat-ml", url: "/wiki/machine-learning/yolov5-tensorrt/" },
    { id: "art-mediapipe", label: "MediaPipe", category: "cat-ml", url: "/wiki/machine-learning/mediapipe-live-ml-anywhere/" },
    { id: "art-nlp-robotics", label: "NLP for Robotics", category: "cat-ml", url: "/wiki/machine-learning/nlp-for-robotics/" },
    { id: "art-distributed-pytorch", label: "Distributed PyTorch", category: "cat-ml", url: "/wiki/machine-learning/distributed-training-with-pytorch/" },
    { id: "art-imitation-learning", label: "Imitation Learning", category: "cat-ml", url: "/wiki/machine-learning/imitation-learning/" },

    // ── Reinforcement Learning ──
    { id: "art-rl-key-concepts", label: "RL Key Concepts", category: "cat-rl", url: "/wiki/reinforcement-learning/key-concepts-in-rl/" },
    { id: "art-rl-algorithms", label: "RL Algorithms", category: "cat-rl", url: "/wiki/reinforcement-learning/reinforcement-learning-algorithms/" },
    { id: "art-policy-gradient", label: "Policy Gradient Methods", category: "cat-rl", url: "/wiki/reinforcement-learning/intro-to-policy-gradient-methods/" },
    { id: "art-value-based-rl", label: "Value-Based RL", category: "cat-rl", url: "/wiki/reinforcement-learning/value-based-reinforcement-learning/" },

    // ── State Estimation ──
    { id: "art-amcl", label: "Adaptive Monte Carlo Localization", category: "cat-state-est", url: "/wiki/state-estimation/adaptive-monte-carlo-localization/" },
    { id: "art-sensor-fusion", label: "Sensor Fusion & Tracking", category: "cat-state-est", url: "/wiki/state-estimation/radar-camera-sensor-fusion/" },
    { id: "art-sbpl-lattice", label: "SBPL Lattice Planner", category: "cat-state-est", url: "/wiki/state-estimation/sbpl-lattice-planner/" },
    { id: "art-orb-slam2", label: "ORB-SLAM2 Setup", category: "cat-state-est", url: "/wiki/state-estimation/orb-slam2-setup/" },
    { id: "art-visual-servoing", label: "Visual Servoing", category: "cat-state-est", url: "/wiki/state-estimation/visual-servoing/" },
    { id: "art-cartographer", label: "Cartographer SLAM", category: "cat-state-est", url: "/wiki/state-estimation/Cartographer-ROS-Integration/" },
    { id: "art-gps-lacking", label: "GPS-Lacking State Estimation", category: "cat-state-est", url: "/wiki/state-estimation/gps-lacking-state-estimation-sensors/" },
    { id: "art-g2o-pose", label: "g2o Pose Graph Optimization", category: "cat-state-est", url: "/wiki/state-estimation/g2o-pose-graphs/" },

    // ── Programming ──
    { id: "art-boost", label: "Boost Library", category: "cat-programming", url: "/wiki/programming/boost-library/" },
    { id: "art-boost-maps", label: "Boost Maps & Vectors", category: "cat-programming", url: "/wiki/programming/boost-maps-vectors/" },
    { id: "art-build-source", label: "Building from Source", category: "cat-programming", url: "/wiki/programming/build-from-source/" },
    { id: "art-cmake", label: "CMake", category: "cat-programming", url: "/wiki/programming/cmake/" },
    { id: "art-eigen", label: "Eigen Library", category: "cat-programming", url: "/wiki/programming/eigen-library/" },
    { id: "art-git", label: "Git", category: "cat-programming", url: "/wiki/programming/git/" },
    { id: "art-multithreading", label: "Multithreaded Programming", category: "cat-programming", url: "/wiki/programming/multithreaded-programming/" },
    { id: "art-interviews", label: "Programming Interviews", category: "cat-programming", url: "/wiki/programming/google-programming-interviews/" },
    { id: "art-tutorials", label: "Tutorials & Resources", category: "cat-programming", url: "/wiki/programming/tutorials-resources/" },
    { id: "art-python-construct", label: "Python Construct", category: "cat-programming", url: "/wiki/programming/python-construct/" },
    { id: "art-yasmin-sm", label: "Yasmin State Machine", category: "cat-programming", url: "/wiki/programming/yasmin-ros2-state-machine/" },
    { id: "art-ros2-actions", label: "ROS2 Action Servers", category: "cat-programming", url: "/wiki/programming/ros2-async-action-servers/" },

    // ── Networking ──
    { id: "art-bluetooth", label: "Bluetooth Sockets", category: "cat-networking", url: "/wiki/networking/bluetooth-sockets/" },
    { id: "art-xbee", label: "Xbee Pro DigiMesh", category: "cat-networking", url: "/wiki/networking/xbee-pro-digimesh-900/" },
    { id: "art-rocon", label: "ROCON Multi-Master", category: "cat-networking", url: "/wiki/networking/rocon-multi-master/" },
    { id: "art-ros-distributed", label: "ROS over Multiple Machines", category: "cat-networking", url: "/wiki/networking/ros-distributed/" },
    { id: "art-wifi-hotspot", label: "WiFi Hotspot at Boot", category: "cat-networking", url: "/wiki/networking/wifi-hotspot/" },
    { id: "art-gstreamer-jetson", label: "GStreamer on Jetson", category: "cat-networking", url: "/wiki/networking/gstreamer-jetson-realtime-video/" },

    // ── Simulation ──
    { id: "art-custom-simulator", label: "Custom Simulator", category: "cat-simulation", url: "/wiki/simulation/Building-a-Light-Weight-Custom-Simulator/" },
    { id: "art-ros-arch-design", label: "ROS Architecture Design", category: "cat-simulation", url: "/wiki/simulation/Design-considerations-for-ROS-architectures" },
    { id: "art-carla", label: "CARLA Vehicles", category: "cat-simulation", url: "/wiki/simulation/Spawning-and-Controlling-Vehicles-in-CARLA" },
    { id: "art-autoware-ndt", label: "Autoware NDT Matching", category: "cat-simulation", url: "/wiki/simulation/NDT-Matching-with-Autoware/" },
    { id: "art-gazebo-graspable", label: "Gazebo Graspable Objects", category: "cat-simulation", url: "/wiki/simulation/gazebo-classic-simulation-of-graspable-and-breakable-objects" },
    { id: "art-isaac-sim", label: "NVIDIA Isaac Sim", category: "cat-simulation", url: "/wiki/simulation/simulation-isaacsim-setup/" },

    // ── Interfacing ──
    { id: "art-apple-vp-interface", label: "Apple Vision Pro Interface", category: "cat-interfacing", url: "/wiki/interfacing/apple-vision-pro-for-robotics-applications/" },
    { id: "art-myo", label: "Myo Gesture Control", category: "cat-interfacing", url: "/wiki/interfacing/myo/" },
    { id: "art-blink-led", label: "Blink(1) LED", category: "cat-interfacing", url: "/wiki/interfacing/blink-1-led/" },
    { id: "art-micro-ros", label: "micro-ROS", category: "cat-interfacing", url: "/wiki/interfacing/microros-for-ros2-on-microcontrollers/" },
    { id: "art-ros-bridge", label: "ROS1 - ROS2 Bridge", category: "cat-interfacing", url: "/wiki/interfacing/ros1_ros2_bridge/" },
    { id: "art-buffer-issues", label: "Buffer Issue Debugging", category: "cat-interfacing", url: "/wiki/interfacing/buffer-issues/" },

    // ── Computing ──
    { id: "art-aws", label: "AWS Quickstart", category: "cat-computing", url: "/wiki/computing/aws-quickstart/" },
    { id: "art-arduino", label: "Arduino", category: "cat-computing", url: "/wiki/computing/arduino/" },
    { id: "art-arduino-components", label: "Basic Arduino Components", category: "cat-computing", url: "/wiki/computing/basic-arduino-components/" },
    { id: "art-sbcs", label: "Single Board Computers", category: "cat-computing", url: "/wiki/computing/single-board-computers/" },
    { id: "art-cots-embedded", label: "COTS Embedded Systems", category: "cat-computing", url: "/wiki/computing/comparing-cots-embedded-systems/" },
    { id: "art-jetson-orin", label: "Jetson Orin AGX", category: "cat-computing", url: "/wiki/computing/jetson-orin-agx/" },
    { id: "art-ubuntu-kernel", label: "Upgrading Ubuntu Kernels", category: "cat-computing", url: "/wiki/computing/upgrading-ubuntu-kernel/" },
    { id: "art-ubuntu-chromebook", label: "Ubuntu on Chromebook", category: "cat-computing", url: "/wiki/computing/ubuntu-chromebook/" },
    { id: "art-gpu-setup", label: "GPU Setup for CV", category: "cat-computing", url: "/wiki/computing/setup-gpus-for-computer-vision/" },
    { id: "art-ubuntu-dual-boot", label: "Ubuntu Dual Boot", category: "cat-computing", url: "/wiki/computing/troubleshooting-ubuntu-dual-boot/" },

    // ── Fabrication ──
    { id: "art-3d-print-considerations", label: "3D Printing Considerations", category: "cat-fabrication", url: "/wiki/fabrication/fabrication_considerations_for_3D_printing/" },
    { id: "art-machining", label: "Machining & Prototyping", category: "cat-fabrication", url: "/wiki/fabrication/machining-prototyping/" },
    { id: "art-rapid-proto", label: "Rapid Prototyping", category: "cat-fabrication", url: "/wiki/fabrication/rapid-prototyping/" },
    { id: "art-turning", label: "Turning Process", category: "cat-fabrication", url: "/wiki/fabrication/turning-process/" },
    { id: "art-milling", label: "Milling Process", category: "cat-fabrication", url: "/wiki/fabrication/milling-process/" },
    { id: "art-soldering", label: "Soldering", category: "cat-fabrication", url: "/wiki/fabrication/soldering/" },
    { id: "art-3d-printers", label: "3D Printers Overview", category: "cat-fabrication", url: "/wiki/fabrication/3d-printers/" },
    { id: "art-makerbot", label: "MakerBot Replicator 2x", category: "cat-fabrication", url: "/wiki/fabrication/makerbot-replicator-2x/" },
    { id: "art-cubepro", label: "CubePro", category: "cat-fabrication", url: "/wiki/fabrication/cube-pro/" },
    { id: "art-series-a-pro", label: "Series A Pro Printer", category: "cat-fabrication", url: "/wiki/fabrication/series-A-pro/" },
    { id: "art-ultimaker", label: "Ultimaker Series", category: "cat-fabrication", url: "/wiki/fabrication/ultimaker-series/" },
    { id: "art-onshape", label: "Onshape Tutorial", category: "cat-fabrication", url: "/wiki/fabrication/onshape-tutorial/" },

    // ── Math ──
    { id: "art-gaussian", label: "Gaussian Processes & GMM", category: "cat-math", url: "/wiki/math/gaussian-process-gaussian-mixture-model/" },
    { id: "art-registration", label: "Registration Techniques", category: "cat-math", url: "/wiki/math/registration-techniques/" },

    // ── Tools ──
    { id: "art-docker", label: "Docker", category: "cat-tools", url: "/wiki/tools/docker/" },
    { id: "art-docker-security", label: "Docker Security", category: "cat-tools", url: "/wiki/tools/docker-security" },
    { id: "art-docker-pytorch", label: "Docker for PyTorch", category: "cat-tools", url: "/wiki/tools/docker-for-pytorch/" },
    { id: "art-vim", label: "Vim", category: "cat-tools", url: "/wiki/tools/vim/" },
    { id: "art-altium", label: "Altium CircuitMaker", category: "cat-tools", url: "/wiki/tools/altium-circuitmaker/" },
    { id: "art-solidworks", label: "SolidWorks", category: "cat-tools", url: "/wiki/tools/solidworks/" },
    { id: "art-tmux", label: "TMUX", category: "cat-tools", url: "/wiki/tools/tmux/" },
    { id: "art-udev", label: "udev Rules", category: "cat-tools", url: "/wiki/tools/udev-rules/" },
    { id: "art-ros-gui", label: "ROS GUI", category: "cat-tools", url: "/wiki/tools/ros-gui/" },
    { id: "art-rosbags-matlab", label: "Rosbags in Matlab", category: "cat-tools", url: "/wiki/tools/rosbags-matlab/" },
    { id: "art-viz-sim", label: "Visualization & Simulation", category: "cat-tools", url: "/wiki/tools/visualization-simulation/" },
    { id: "art-clion", label: "CLion IDE", category: "cat-tools", url: "/wiki/tools/clion/" },
    { id: "art-stream-rviz", label: "Stream Rviz Images", category: "cat-tools", url: "/wiki/tools/stream-rviz/" },
    { id: "art-roslibjs", label: "ROS JavaScript Library", category: "cat-tools", url: "/wiki/tools/roslibjs/" },
    { id: "art-mapviz", label: "MapViz ROS2", category: "cat-tools", url: "/wiki/planning/mapviz/" },
    { id: "art-gazebo-sim", label: "Gazebo Simulation", category: "cat-tools", url: "/wiki/tools/gazebo-simulation/" },
    { id: "art-code-editors", label: "VS Code & Vim Intro", category: "cat-tools", url: "/wiki/tools/code-editors-Introduction-to-vs-code-and-vim/" },
    { id: "art-qtcreator", label: "QtCreator with ROS", category: "cat-tools", url: "/wiki/tools/Qtcreator-ros/" },
    { id: "art-xarm-ros", label: "xarm_ros Guide", category: "cat-tools", url: "/wiki/tools/xarm-ros-guide/" },
    { id: "art-ipc-recorder", label: "ROS2 IPC Recorder", category: "cat-tools", url: "/wiki/tools/ros2-humble-ipc-recorder/" },
    { id: "art-docker-robotics", label: "Docker for Robotics", category: "cat-tools", url: "/wiki/tools/docker-for-robotics/" },

    // ── Datasets ──
    { id: "art-traffic-datasets", label: "Traffic Modelling Datasets", category: "cat-datasets", url: "/wiki/datasets/traffic-modelling-datasets/" },
    { id: "art-open-datasets", label: "Open-Source Datasets", category: "cat-datasets", url: "/wiki/datasets/open-source-datasets/" },

    // ── Planning ──
    { id: "art-planning-overview", label: "Planning Overview", category: "cat-planning", url: "/wiki/planning/planning-overview/" },
    { id: "art-astar", label: "A* Planner", category: "cat-planning", url: "/wiki/planning/astar_planning_implementation_guide/" },
    { id: "art-dynamic-astar", label: "Extensions to A*", category: "cat-planning", url: "/wiki/planning/non-a-star-planning/" },
    { id: "art-behavior-tree", label: "Behavior Trees", category: "cat-planning", url: "/wiki/planning/behavior-tree/" },
    { id: "art-specialized-planners", label: "PILZ & CHOMP Planners", category: "cat-planning", url: "/wiki/planning/pilz-chomp/" },
    { id: "art-coverage-planner", label: "Coverage Planner", category: "cat-planning", url: "/wiki/planning/coverage-planning-implementation-guide/" },
    { id: "art-resolved-rates", label: "Resolved Rates", category: "cat-planning", url: "/wiki/planning/resolved-rates/" },
    { id: "art-multi-robot", label: "Multi-Robot Planning", category: "cat-planning", url: "/wiki/planning/multi-robot-planning/" },
    { id: "art-rrt-prm", label: "RRT & PRM Variants", category: "cat-planning", url: "/wiki/planning/rrt-prm-planning/" },
    { id: "art-frenet", label: "Frenet Frame Planning", category: "cat-planning", url: "/wiki/planning/frenet-frame-planning/" },
    { id: "art-move-base-flex", label: "Move Base Flex", category: "cat-planning", url: "/wiki/planning/move-base-flex/" },
    { id: "art-chomp", label: "CHOMP Path Planning", category: "cat-planning", url: "/wiki/planning/chomp-planning/" },
    { id: "art-glop", label: "Google GLOP Optimizer", category: "cat-planning", url: "/wiki/planning/glop/" },
    { id: "art-advanced-moveit", label: "Advanced MoveIt", category: "cat-planning", url: "/wiki/planning/advanced-moveit-manipulator-planning/" }
  ],

  // Cross-category semantic connections
  edges: [
    // ── Sensing <-> State Estimation ──
    { source: "art-camera-calib", target: "art-visual-servoing", label: "camera for visual control" },
    { source: "art-camera-imu-calib", target: "art-cartographer", label: "sensor calibration for SLAM" },
    { source: "art-realsense", target: "art-orb-slam2", label: "depth camera for SLAM" },
    { source: "art-pcl", target: "art-cartographer", label: "point clouds for SLAM" },
    { source: "art-adafruit-gps", target: "art-gps-lacking", label: "GPS sensing" },
    { source: "art-rtk-gps", target: "art-gps-lacking", label: "GPS technology" },
    { source: "art-fiducial-markers", target: "art-visual-servoing", label: "visual tracking" },
    { source: "art-delphi-radar", target: "art-sensor-fusion", label: "radar sensor fusion" },
    { source: "art-stereo-vision", target: "art-orb-slam2", label: "stereo for SLAM" },
    { source: "art-uwb-positioning", target: "art-gps-lacking", label: "indoor positioning" },

    // ── Sensing internal connections ──
    { source: "art-apriltags", target: "art-fiducial-markers", label: "fiducial detection" },
    { source: "art-stag", target: "art-fiducial-markers", label: "fiducial markers" },
    { source: "art-camera-calib", target: "art-handeye-calib", label: "calibration pipeline" },
    { source: "art-camera-calib", target: "art-camera-imu-calib", label: "calibration" },
    { source: "art-camera-calib", target: "art-photometric-calib", label: "camera calibration" },
    { source: "art-thermal-cameras", target: "art-thermal-perception", label: "thermal imaging pipeline" },
    { source: "art-thermal-cameras", target: "art-sensor-noise", label: "sensor noise" },
    { source: "art-apple-vision-sensing", target: "art-hololens", label: "AR/VR sensing" },

    // ── Sensing <-> Machine Learning ──
    { source: "art-yolo-ros", target: "art-cv-considerations", label: "vision detection" },
    { source: "art-trajectory-extract", target: "art-cv-considerations", label: "computer vision" },
    { source: "art-mediapipe", target: "art-cv-considerations", label: "ML vision" },
    { source: "art-yolo-ros", target: "art-realsense", label: "camera + detection" },

    // ── Machine Learning <-> Reinforcement Learning ──
    { source: "art-intro-rl", target: "art-rl-key-concepts", label: "RL foundations" },
    { source: "art-python-rl-libs", target: "art-rl-algorithms", label: "RL tools" },
    { source: "art-intro-rl", target: "art-policy-gradient", label: "RL methods" },
    { source: "art-intro-rl", target: "art-value-based-rl", label: "RL methods" },

    // ── Machine Learning internal ──
    { source: "art-diffusion", target: "art-grpo-diffusion", label: "diffusion policy" },
    { source: "art-darknet", target: "art-yolo-ros", label: "YOLO training" },
    { source: "art-darknet", target: "art-custom-semantic", label: "custom datasets" },
    { source: "art-imitation-learning", target: "art-diffusion", label: "learning from demonstration" },

    // ── Planning <-> Actuation ──
    { source: "art-advanced-moveit", target: "art-moveit-hebi", label: "MoveIt integration" },
    { source: "art-chomp", target: "art-moveit-hebi", label: "motion planning" },
    { source: "art-specialized-planners", target: "art-moveit-hebi", label: "CHOMP/PILZ planning" },
    { source: "art-planning-overview", target: "art-mpc", label: "control + planning" },
    { source: "art-move-base-flex", target: "art-pure-pursuit", label: "mobile robot control" },
    { source: "art-resolved-rates", target: "art-task-priority", label: "manipulator control" },

    // ── Planning <-> State Estimation ──
    { source: "art-planning-overview", target: "art-amcl", label: "navigation stack" },
    { source: "art-move-base-flex", target: "art-amcl", label: "ROS navigation" },
    { source: "art-sbpl-lattice", target: "art-planning-overview", label: "lattice planning" },
    { source: "art-astar", target: "art-dynamic-astar", label: "A* variants" },

    // ── Planning <-> Manipulators ──
    { source: "art-advanced-moveit", target: "art-kuka-moveit2", label: "MoveIt2 manipulator" },
    { source: "art-resolved-rates", target: "art-kuka-moveit2", label: "manipulator control" },
    { source: "art-chomp", target: "art-kuka-moveit2", label: "path planning" },

    // ── Common Platforms <-> State Estimation ──
    { source: "art-ros-navigation", target: "art-amcl", label: "ROS nav stack" },
    { source: "art-ros-mapping", target: "art-cartographer", label: "SLAM mapping" },
    { source: "art-ros2-husky-nav", target: "art-amcl", label: "robot navigation" },
    { source: "art-ros-cost-maps", target: "art-sbpl-lattice", label: "costmap planning" },

    // ── Common Platforms <-> Planning ──
    { source: "art-ros-global-planner", target: "art-planning-overview", label: "ROS planning" },
    { source: "art-ros-navigation", target: "art-move-base-flex", label: "navigation framework" },

    // ── Common Platforms <-> Actuation ──
    { source: "art-ur5e", target: "art-moveit-hebi", label: "robot arm control" },
    { source: "art-unitree-go1", target: "art-mpc", label: "locomotion control" },
    { source: "art-unitree-g1", target: "art-humanoid-robot", label: "humanoid platform" },
    { source: "art-rc-cars", target: "art-drive-by-wire", label: "vehicle control" },

    // ── Computing <-> Machine Learning ──
    { source: "art-gpu-setup", target: "art-distributed-pytorch", label: "GPU ML training" },
    { source: "art-jetson-orin", target: "art-yolov5-tensorrt", label: "Jetson deployment" },
    { source: "art-gpu-setup", target: "art-yolo-ros", label: "GPU for detection" },

    // ── Computing <-> Actuation ──
    { source: "art-arduino", target: "art-pid-arduino", label: "Arduino control" },
    { source: "art-arduino-components", target: "art-pid-arduino", label: "Arduino hardware" },

    // ── Computing <-> Interfacing ──
    { source: "art-arduino", target: "art-micro-ros", label: "microcontrollers" },

    // ── Computing <-> Networking ──
    { source: "art-jetson-orin", target: "art-gstreamer-jetson", label: "Jetson video" },
    { source: "art-sbcs", target: "art-wifi-hotspot", label: "SBC networking" },

    // ── Computing <-> Common Platforms ──
    { source: "art-nvidia-orin", target: "art-jetson-orin", label: "Nvidia hardware" },

    // ── Simulation <-> Common Platforms ──
    { source: "art-isaac-sim", target: "art-ros-intro", label: "ROS simulation" },
    { source: "art-ros-arch-design", target: "art-ros-intro", label: "ROS architecture" },
    { source: "art-gazebo-graspable", target: "art-ros-intro", label: "Gazebo + ROS" },

    // ── Simulation <-> Planning ──
    { source: "art-carla", target: "art-frenet", label: "autonomous driving" },
    { source: "art-autoware-ndt", target: "art-carla", label: "AV simulation" },

    // ── Simulation <-> Tools ──
    { source: "art-isaac-sim", target: "art-gazebo-sim", label: "simulation tools" },
    { source: "art-gazebo-graspable", target: "art-gazebo-sim", label: "Gazebo" },

    // ── Tools <-> Common Platforms ──
    { source: "art-gazebo-sim", target: "art-ros-intro", label: "ROS tools" },
    { source: "art-ros-gui", target: "art-ros-intro", label: "ROS visualization" },
    { source: "art-stream-rviz", target: "art-ros-intro", label: "ROS Rviz" },
    { source: "art-roslibjs", target: "art-ros-intro", label: "ROS web tools" },
    { source: "art-docker-robotics", target: "art-ros-intro", label: "Docker + ROS" },
    { source: "art-qtcreator", target: "art-ros-intro", label: "ROS IDE" },
    { source: "art-xarm-ros", target: "art-ur5e", label: "robot arm tools" },

    // ── Tools <-> System Design ──
    { source: "art-altium", target: "art-pcb-design", label: "PCB tools" },
    { source: "art-solidworks", target: "art-mech-design", label: "CAD tools" },

    // ── Tools <-> Fabrication ──
    { source: "art-solidworks", target: "art-machining", label: "CAD → manufacturing" },
    { source: "art-onshape", target: "art-solidworks", label: "CAD tools" },

    // ── Tools <-> Computing ──
    { source: "art-docker-pytorch", target: "art-gpu-setup", label: "ML infrastructure" },
    { source: "art-docker", target: "art-aws", label: "cloud containers" },

    // ── Tools internal ──
    { source: "art-docker", target: "art-docker-security", label: "Docker security" },
    { source: "art-docker", target: "art-docker-pytorch", label: "Docker ML" },
    { source: "art-docker", target: "art-docker-robotics", label: "Docker robotics" },
    { source: "art-vim", target: "art-code-editors", label: "code editors" },

    // ── Networking <-> Common Platforms ──
    { source: "art-ros-distributed", target: "art-ros-intro", label: "ROS networking" },
    { source: "art-rocon", target: "art-ros-intro", label: "multi-robot ROS" },

    // ── System Design <-> Programming ──
    { source: "art-fsm", target: "art-yasmin-sm", label: "state machines" },

    // ── Interfacing <-> Common Platforms ──
    { source: "art-ros-bridge", target: "art-ros-intro", label: "ROS bridge" },
    { source: "art-micro-ros", target: "art-ros2-custom-pkg", label: "ROS2 microcontrollers" },

    // ── Interfacing <-> Sensing ──
    { source: "art-apple-vp-interface", target: "art-apple-vision-sensing", label: "Apple Vision Pro" },
    { source: "art-apple-vp-interface", target: "art-hololens", label: "AR/VR interfaces" },

    // ── Machine Learning <-> Robotics Project Guide ──
    { source: "art-imitation-learning", target: "art-humanoid-robot", label: "humanoid learning" },
    { source: "art-diffusion", target: "art-humanoid-robot", label: "diffusion for robots" },

    // ── Math <-> State Estimation ──
    { source: "art-gaussian", target: "art-amcl", label: "probabilistic methods" },
    { source: "art-registration", target: "art-orb-slam2", label: "point registration" },
    { source: "art-registration", target: "art-pcl", label: "point cloud registration" },

    // ── Math <-> Machine Learning ──
    { source: "art-gaussian", target: "art-intro-rl", label: "probabilistic models" },

    // ── Datasets <-> Sensing ──
    { source: "art-traffic-datasets", target: "art-trajectory-extract", label: "traffic data" },

    // ── Datasets <-> Machine Learning ──
    { source: "art-open-datasets", target: "art-darknet", label: "training data" },
    { source: "art-open-datasets", target: "art-custom-semantic", label: "semantic data" },

    // ── Fabrication internal ──
    { source: "art-3d-print-considerations", target: "art-3d-printers", label: "3D printing" },
    { source: "art-machining", target: "art-rapid-proto", label: "prototyping" },
    { source: "art-turning", target: "art-milling", label: "machining processes" },
    { source: "art-machining", target: "art-turning", label: "machining" },
    { source: "art-machining", target: "art-milling", label: "machining" },
    { source: "art-3d-printers", target: "art-makerbot", label: "3D printer model" },
    { source: "art-3d-printers", target: "art-cubepro", label: "3D printer model" },
    { source: "art-3d-printers", target: "art-ultimaker", label: "3D printer model" },
    { source: "art-soldering", target: "art-pcb-design", label: "PCB assembly" },

    // ── Fabrication <-> System Design ──
    { source: "art-3d-print-considerations", target: "art-mech-design", label: "design for manufacturing" },

    // ── RL internal ──
    { source: "art-rl-algorithms", target: "art-policy-gradient", label: "RL algorithm type" },
    { source: "art-rl-algorithms", target: "art-value-based-rl", label: "RL algorithm type" },
    { source: "art-rl-key-concepts", target: "art-rl-algorithms", label: "RL foundations" },

    // ── Project Guide internal ──
    { source: "art-choose-robot", target: "art-make-robot", label: "robot selection" },
    { source: "art-choose-sim", target: "art-test-debug", label: "testing workflow" },
    { source: "art-robot-comm", target: "art-choose-language", label: "implementation" },

    // ── Project Guide <-> Common Platforms ──
    { source: "art-choose-sim", target: "art-gazebo-sim", label: "simulator choice" },
    { source: "art-choose-robot", target: "art-ros-intro", label: "ROS for robots" },

    // ── Project Management <-> Common Platforms ──
    { source: "art-drone-permits", target: "art-dji-breakdown", label: "drone regulations" },
    { source: "art-drone-permits", target: "art-pixhawk", label: "UAV operations" },

    // ── State Estimation internal ──
    { source: "art-cartographer", target: "art-orb-slam2", label: "SLAM methods" },
    { source: "art-g2o-pose", target: "art-orb-slam2", label: "pose optimization" },
    { source: "art-sensor-fusion", target: "art-amcl", label: "localization" },

    // ── Common Platforms <-> Networking ──
    { source: "art-ros-arduino", target: "art-arduino", label: "Arduino + ROS" },

    // ── Simulation <-> State Estimation ──
    { source: "art-autoware-ndt", target: "art-cartographer", label: "localization methods" },

    // ── Computing <-> Common Platforms (via ROS2 tools) ──
    { source: "art-vscode-ros2", target: "art-clion", label: "ROS IDEs" }
  ]
};
