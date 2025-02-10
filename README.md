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
src/inferencer   # Inferencer plugins for the nn_controller_node
```
### Dependencies
- ros-melodic-desktop-full
- ros-melodic-cv-bridge
- ros-melodic-joy
- ros-melodic-gscam
- opencv
- tensorrt (optional)
- tensorflow (optional)

<!-- GETTING STARTED -->
## Getting Started On Jetson Nano

### Install ros-melodic

see [Ubuntu install of ROS Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu) for details.

### Download && Build
1. Clone the workspace
   ```bash
   git clone https://github.com/Trustworthy-Engineered-Autonomy-Lab/donkeycar_ros.git
   ```
2. Install dependent ros packages
   ```bash
   cd donkeycar_ros
   rosdep install --from-paths src
   ```
3. Build and export the workspace
   ```bash
   catkin_make
   source devel/setup.bash
   ```
   
<!-- USAGE EXAMPLES -->
## Usage
### Drive the car
```bash
roslaunch donkeycar drive.launch
```
Then you can drive the car by joystick
### Collect data
```bash
roslaunch donkeycar collect.launch
```
Then you can press the **Start** button on joystick to start recording and press the **Back** button to stop. the images will be saved under the data folder.
### Use autopilot
1. Create a `models` folder under the [donkey package folder](./src/donkeycar/).
2. Download the latest model from [Coming soon]() to the models folder just created.
3. run the command
```bash
roslaunch donkeycar autopilot.launch
```
Then AI model will control the steer, you can control the throttle by the joystick.
## Contribute
### Write a inferencer plugin
check out [Inferencer plugins](./src/inferencer/README.md) for details.
### Submit an issue.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- Authors -->
## Authors

Zhongzheng R. Zhang - zhangrenzhongzheng@outlook.com
