# PiCar-X — Autonomous Patrol Final Project

## Line Tracking with Emergency Stop

### 1. Introduction

   This project implements an autonomous line-tracking robot with an emergency stop mechanism using the Picar-X platform. The system combines real-time sensor fusion, Python-based control, and safety protocols to enable safe and autonomous navigation.

### 2. Goal

   To design a functional robotic application that demonstrates sensor fusion, autonomous navigation, and emergency-stop safety protocols using the PiCar-X platform.
   
### 3. Methodology

####   Hardware

   Sensors:
   
   3-Channel Grayscale Module (Line detection)
   
   Ultrasonic Sensor (Time of flight distance measurement)

   Actuators:
   
   DC Gear Motors (Propulsion)

   Servo Motor (Ackermann Steering)

####   Software

   Control Loop: Python-based generic decision loop running at ~50Hz.

   Python 3 for programming the robot Modular code structure with file: patrol.py 

### 4. System Behavior

####   Sensor & Actuator Integration

   The robot fuses data from the grayscale module and ultrasonic sensor to adjust motor power and steering angle.

####   Autonomous Navigation

   Uses line-contrast information to follow a predefined path without human intervention.

####   Real-Time Perception

   A 50 Hz control loop continuously reads sensor inputs and updates steering based on lateral error.

####   Safety Protocol — Emergency Stop

   If the ultrasonic sensor detects an object at < 15 cm, the robot immediately stops.

### 5. How to Run

   sudo python3 patrol.py
   Note: sudo may be required for GPIO access on Raspberry Pi.
   
### 6. Code

   Python script is located in the code/ folder.
   Main execution file: patrol.py


