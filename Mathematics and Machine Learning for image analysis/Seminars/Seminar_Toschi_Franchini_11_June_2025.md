# Seminar_Toschi_Franchini_11_June_2025.pdf

## 第2页

About me
2
Alessandro Toschi
●
PhD student in Computer and Data Science @ UNIMORE (started in 2022)
○
My research topic: “Vehicle motion planning and control in complex environments”
●
UNIMORE Racing team member since 2021
Academic background:
●
BS in Automation Engineering @ UNIBO & Tongji University
●
MSc in Advanced Automotive Electronic Engineering @ MUNER


---

## 第3页

The Lab
3
✓
Complex workload-intensive tasks
○
Perception, planning, ML/DNN
✓
Latency-critical control tasks
○
Cyber-physical interaction
○
Tight actuation loops
meets Automotive/Racing
Embedded boards
https://hipert.unimore.it/


---

## 第4页

Roboracer (F1tenth)
4
Roboracer


---

## 第5页

5
Indy Autonomous Challenge
 by Energy Systems Network (USA) <<
Since 2021 <<
Indy Light Dallara AV-24 <<
1M$ first prize <<
1st track: Indianapolis Motor Speedway <<
Abu Dhabi Autonomous Racing League
>> by ASPIRE (UAE)
>> Since 2024
>> Super Formula Dallara EAV-24
>> 2.25M$ overall price
>> 1st track: Yas Marina Circuit
Autonomous Racing


---

## 第6页

6
3x 3D Lidars
6x Cameras GS
3x GNSS w RTK
2x 2D radar
phonic wheels
3x 3D Lidars
7x Cameras RS
1x GNSS w RTK
4x 4D radar
1x Optical 
sensor
phonic wheels
Lidar
Camera
GNSS
Radar
Sensors


---

## 第7页

Las Vegas Motor 
Speedway (LVMS) - CES
07.01.2022 & 07.01.2023 & 
11.01.2024 & 09.01.2025
Indianapolis Motor 
Speedway (IMS)
23.10.2021 & 06.09.2024
Autodromo di Monza
16-18.06.2023
Texas Motor Speedway 
(TMS)
11.11.2022
YAS marina circuit
25-27.04.2024
Fastest lap + 2nd place
2nd place in 2-vehicle race
3rd place
1st place 4-vehicle race 
exhibition
3rd place
3rd place
2nd place at time trials
3rd place in the race
Racing Events


---

## 第8页

High Speed - More than 280 km/h
8


---

## 第9页

Road Course
9


---

## 第10页

Head to Head Racing
10


---

## 第11页

Autonomous Driving Context
11
Autonomous Driving Software
Perceive
Plan
Act
Environment


---

## 第12页

Software Stack Architecture
12
Localization & 
State 
Estimation
Object 
Detection & 
Tracking
Trajectory 
Planning
Motion Control
Sensors 
Interface
Actuators 
Interface
Mission 
Planning
Mapping
Racing Line 
Optimization
Supervisor
Motion 
Forecasting
Multi-Sensors 
Calibration
System 
Dynamics 
Identiﬁcation
Oﬄine
Online
Telemetry
Simulation
Base Station
Race Control


---

## 第13页

State 
Estimation
13


---

## 第14页

Localization & State Estimation
Centimeter-level accuracy position sources:
●
GPS-RTK.
●
LiDAR-Localization.
Pose Estimation (EKF):
●
Outlier Pre-Filtering.
●
Multirate corrections and delay 
compensation.
●
Kineto-Dynamic Bicycle Model.
GPS-
RTK
LiDAR-
Loc.
Optical 
Sensor
IMU
Wheels 
Speed
Multirate EKF
Raw Poses
Velocities & Accelerations
Vehicle Pose
14


---

## 第15页

Side Slip Angle Estimation
Side Slip Angle Estimation:
●
Kalman Filter with Pacejka tire model
●
Lateral velocity corrections from LiDAR 
Odometry.
●
WIP: use also Radar odometry
Approaches:
1.
Started with Unscented Kalman Filter
2.
Now using an Extended Kalman Filter
3.
WIP: Moving Horizon Estimation
Lateral velocity and tires model parameter estimation.
15


---

## 第16页

Lidar Localization
16


---

## 第17页

Visual-Inertial Odometry
17


---

## 第18页

18
Perception


---

## 第19页

Object Detection - Cameras
19
●
Object detection with YOLOv4
●
Optimized inference on tkDNN2
●
~240 FPS on GPU (6 streams)
●
Frequency used: 20Hz
●
Range: ~ 200 m


---

## 第20页

Object Detection - Lidars
20
●
Ground removal
●
Object detection with PointPillars
●
Optimized inference on tkDNNv2
●
~50 FPS on GPU
●
Frequency used: 20 Hz
●
Range: ~130 m


---

## 第21页

Object Detection - Radars 
21
●
Filtering and clustering
●
~20 FPS on CPU
●
Frequency used: 20 Hz
●
Range: ~150 m


---

## 第22页

Detections Fusion
22
●
Based on precise multi-sensors calibration
●
Uses consensus from multiple sensor to correctly detect an opponent
●
Robustness to failures of multiple sensors


---

## 第23页

Object Detection & Tracking
Sensor Fusion:
●
LiDAR: PointPillars1.
●
Camera: YOLOv42.
●
RADAR: Point cloud processing and 
clustering.
Object Tracking: 
●
Motion prediction and matching.
[1] A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom, “PointPillars: Fast 
Encoders for Object Detection From Point Clouds,” in 2019 IEEE/CVF Conference on Computer 
Vision and Pattern Recognition (CVPR), Long Beach, CA, USA: IEEE, Jun. 2019, pp. 
12689–12697. doi: 10.1109/CVPR.2019.01298.
[2] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal Speed and Accuracy of 
Object Detection,” Apr. 23, 2020, arXiv: arXiv:2004.10934. doi: 10.48550/arXiv.2004.10934.
23
Lidar
Camera
GNSS
Radar
Sensors set for the IAC AV-24.


---

## 第24页

24
Perception
Sensor Fusion:
●
LiDAR: 
PointPillars.
●
Camera: YOLOv4.
●
RADAR: Point 
cloud processing 
and clustering.
Object Tracking: 
●
Motion prediction 
and matching.


---

## 第25页

Object Tracking & Forecasting
25
●
Tracker: used to track the 
opponent position and velocity
●
Forecasting: predict the opponent 
action for the next 6 seconds
●
Motion Forecasting methods:
○
Kalman Filter
○
Frenet Planner (currently used)
○
Neural Network


---

## 第26页

Frenét Graph-Based planner
26
Trajectory
selection
Trajectories 
collision check
Cost assignment 
to each trajectory
Generation of a 
set of trajectories
Opponent
Safety 
margin
Time to collision


---

## 第27页

27
Vehicle 
Dynamics


---

## 第28页

Vehicle Dynamics - Simulations
28
●
High-Fidelity Multibody Modeling and 
Simulation (Dymola)
●
Simulation Environment for Autonomous 
Software testing (C++, Python, Simulink)
●
Automated Batch Simulations with 
Extensive Boundary and Initial Conditions.


---

## 第29页

Vehicle Dynamics - Grip Estimation
29
●
Real-Time Tire-Road Grip Estimation
●
Flexible and Modular Tire Formulations
●
Online Parameter Adaptation for 
Autonomous Controller
●
Non-linear regression (with Ceres solver) 
using estimated vehicle slips


---

## 第30页

30
Moving Horizon 
Optimization


---

## 第31页

Moving Horizon Optimization
31
Moving Horizon Estimation
Model Predictive Control


---

## 第32页

MPC Problem Formulation
32
friction ellipse 
constraint
physical inputs and 
rate of change 
constraints
track 
constraints
yaw rate 
regularizer
input rate 
regularizer
rear slip angle 
regularizer
path following 
weights
velocity tracking 
weight


---

## 第33页

Planner and Controller - Model Predictive Control
33
Single-track model
Curvilinear 
coordinates
Hpipm solver
Past inputs: steering angle, 
throttle, brake
Actual state: position, 
heading, speeds, yaw rate
Future inputs:
steering angle, throttle, 
brake
Future errors
Predicted 
future states
Local trajectory as 
reference
Constraints: 
speed, throttle, brake, 
steering angle...
Cost function: 
Velocity and path tracking 
error, progression, inputs rate 
of change...
Vehicle
●
Dynamics discretized and linearized in 
time
●
Time horizon of 
○
2.6 seconds - Controller
○
6 seconds - Planner
●
Discretization and sampling at 40 
ms
●
Execution at
○
100Hz - Controller
○
20 Hz - Planner
●
Quadratic problem solved using 
HPIPM (High-Performance Interior 
Point Method) solver and using 
CppADCodeGen for automatic 
differentiation.


---

## 第34页

34
Motion Planning
and Decision 
Making


---

## 第35页

Global Planning - Lap-time Optimization
35
●
Optimal Global Trajectory for Minimum 
Lap-Time: Optimization in Julia using a 
single/double track model 
●
Strategy-Aware Lap-Time Minimization
●
Driving Style Comparison
●
Lap-Time Sensitivity to Vehicle 
Parameters


---

## 第36页

Global Planning - Lap-time Optimization
36
●
Optimal Global Trajectory for Minimum 
Lap-Time: Optimization in Julia using a 
single/double track model 
●
Strategy-Aware Lap-Time Minimization
●
Driving Style Comparison
●
Lap-Time Sensitivity to Vehicle 
Parameters


---

## 第37页

Optimization-Based Trajectory Planning
●
Model Predictive Control (MPC) 
framework.
●
Planner serves as a Reference Governor 
for the motion controller.
●
Constraints as a comprehensive approach 
to enforce safety and vehicle handling.
GG-V diagrams incorporating track banking (data from 
the IAC at IMS, 2024).
37


---

## 第38页

Drivable tunnels and trajectory planning in three different 
maneuvers.
Multi-Planner Scheme for Overtakes
Drivable Tunnel Generator:
●
Convex constraint subsets of the drivable 
space.
●
Rule-based (e.g., right-of-way).
Multi-Planner Scheme:
●
Multiple planner instances for different 
maneuvers (e.g., overtake left/right, 
follow).
Decision Making:
●
Selects the “best” maneuver among the 
available options.
38
Left
Right
Planner
Instance
Drivable 
Tunnels
Gen.
Maneuver 
Selector


---

## 第39页

Local Planning - Overtakes
39
●
Path tracking and Planning with Model 
Predictive Control (MPC).
●
Hierarchical approach
●
Tube-based Overtake Planning
○
Comparing ego prediction with opponents 
forecastings
○
Decision-making with heuristics
○
Rule-compliant
FMU-based Simulation at Yas Marina Circuit


---

## 第40页

Local Planning - Overtakes
40
Experimental testing at Las Vegas Motor Speedway
submitted paper


---

## 第41页

41
Motion Control


---

## 第42页

Control - Enhanced Bicycle Model
42
Formulation developed for 
considering:
●
Banking
●
Tires asymmetry
●
Internal Combustion Engine
●
Differential effects
●
Combined Slip effects


---

## 第43页

Control - Enhanced Bicycle Model
43


---

## 第44页

Motion and Vehicle Stability Control
●
MPC-Curv
●
Longitudinal Controller: Feedforward + PID
●
Predictive Gear Controller
●
Lower level controller
○
Traction Control
○
Active Braking System
○
Electronic Stability Control
●
Emergency Controller:
○
Uses RADAR-based wall detection to pull 
over safely in case of localization failures.
44
MPC-Curv
Longitudinal
Controller
Gear 
Controller
Vehicle
ESP + TC + ABS
Gear
Steering
Target Acceleration
Throttle & Brakes
Throttle & Brakes


---

## 第45页

Motion Planning and Control - full scheme
45


---

## 第46页

WHAT COULD POSSIBLY 
GO WRONG? 


---

## 第47页

Many things can happen
●
Edge cases not properly considered
●
Bugs in the code
●
Sensors faults and bugs
●
Car’s electronics and/or mechanics failures
●
…


---

## 第48页

Contacts and Events
Our Last Event
●
Modena 09/06/25, Motor 
valley Fest
●
Marzaglia track record 
(see onboard beside)
Next Event
●
MIMO Monza 27-29/06/25
unimoreracing
Unimore Racing
alessandro.toschi@unimore.it
unimoreracing@unimore.it


---

## 第49页

Research - References
49
●
A. Raji et al., "Motion Planning and Control for Multi Vehicle Autonomous Racing at High Speeds," 
2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)
●
A. Raji et al., "A Tricycle Model to Accurately Control an Autonomous Racecar with Locked 
Differential," 2023 IEEE 11th International Conference on Systems and Control (ICSC)
●
A. Raji et al. “er.autopilot 1.1: A Software Stack for Autonomous Racing on Oval and Road Course 
Tracks”, 2024, Field Robotics, 4, 99–137.
●
A. Toschi et al., "Guess the Drift with LOP-UKF: LiDAR Odometry and Pacejka Model for Real-Time 
Racecar Sideslip Estimation," 2024 IEEE Intelligent Vehicles Symposium (IV)
●
F. Prignoli et al., "RADAR-Based Safe Pull-Over of Autonomous Racing Cars in Localization Failure 
Scenarios" 2024 European Control Conference (ECC)
●
A. Raji, “Model Predictive Planning and Control for Autonomous Racing, from HPC to Embedded 
Platforms”, PhD Thesis, Unipr, 2024
●
A. Remonda et al. “A Simulation Benchmark for Autonomous Racing with Large-Scale Human 
Data”, NeurIPS, 2024


---

