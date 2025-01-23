<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">DonkeyCar Ros</h1>
</p>


<!-- ABOUT THE PROJECT -->
## About The Project
### Introduction
This is a new software stack for donkey car based on ROS1. This project is currently in its early stages. I'm working on it to add more features in the future.
The software stack can only work on jetson nano.

### Project Layout
```bash
src/controller   # The controller nodes that send the control commends based on user operations or neural network
src/actuator     # The actuator nodes that subscirbe the control commends and control the steer and motor
src/recorder     # Used to collect data if you want to train an autopilot
src/donkeycar    # Store launch files, model files, as well as collected data
```

<!-- GETTING STARTED -->
## Getting Started

### Dependencies

- ros-melodic-desktop-full
- ros-melodic-cv-bridge
- ros-melodic-joy
- ros-melodic-gscam
- opencv
- cuda
- tensorrt

### Download && Build
1. Install ros-melodic
   ```bash
   sudo apt install ros-melodic-desktop-full
   source /opt/ros/melodic/setup.bash
   ```
2. Clone the workspace
   ```bash
   git clone https://github.com/Trustworthy-Engineered-Autonomy-Lab/donkeycar_ros.git
   ```
4. Install dependent ros packages
   ```bash
   cd donkeycar_ros
   rosdep install --from-paths src
   ```
5. Build and export the workspace
   ```bash
   catkin_make
   source devel/setup.bash
   ```
   
<!-- USAGE EXAMPLES -->
## Usage
### Drive
```bash
roslaunch donkeycar drive.launch
```
Then you can drive the car by joystick
### Collect
```bash
roslaunch donkeycar collect.launch
```
Then you can press the **Start** button on joystick to start recording and press the **Back** button to stop. the images will be saved under the data folder.
### Autopilot

_Will be ready soon_

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- Authors -->
## Authors

Zhongzheng R. Zhang - zhangrenzhongzheng@outlook.com
