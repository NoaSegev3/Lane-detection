# About
A Python-based system for real-time lane and vehicle detection, supporting multiple scenarios such as daytime, nighttime, and curved roads. Includes features for lane change detection and vehicle proximity alerts.

# Lane and Vehicle Detection System
This project was developed as part of the **Introduction to Image Processing and Analysis** course. It implements a robust lane and vehicle detection system for various driving conditions, including daytime, nighttime, curved roads, and vehicle proximity detection.

## Features
- **Daytime Lane Detection**: Uses Canny edge detection and Hough transform to identify and highlight lane markings.
- **Nighttime Lane Detection**: Enhances visibility using histogram equalization and adaptive thresholding for nighttime driving conditions.
- **Curved Lane Detection**: Detects curved lanes by fitting polynomials and ensuring lane consistency.
- **Vehicle Detection**: Identifies nearby vehicles using Haar cascades and proximity constraints.
- **Lane Change Detection**: Alerts for potential lane changes based on lane width and center shifts.

## Project Structure
- `main.py`: Entry point of the project, which allows selecting different detection modes (`DAY_TIME`, `NIGHT_TIME`, `DETECT_CURVES`, `VEHICLE_DETECTION`) through command-line arguments.
- `curve_lane_detector.py`: Logic for detecting and rendering curved lanes.
- `daytime_lane_detector.py`: Handles daytime lane detection with edge and line detection algorithms.
- `nighttime_lane_detector.py`: Focuses on lane detection under low-light conditions.
- `lane_change_detector.py`: Monitors and identifies lane changes based on width and center variations.
- `lane_utils.py`: Shared utilities for lane detection, such as region of interest masking and lane stabilization.
- `vehicle_detector.py`: Detects vehicles and excludes lanes from detection using bounding box merging and stabilization techniques.
- `haarcascade_car.xml`: Pre-trained Haar cascade model for vehicle detection.

## Prerequisites
To run this project, ensure you have the following software installed on your system:
- Python 3.8 or later
- OpenCV 4.5 or later
- NumPy

## Getting Started
Follow these steps to set up and run the project locally:
1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Place the input videos in the videos/ directory.
4. Run the project by specifying the desired detection mode. For example, to run the project in daytime detection mode: python main.py DAY_TIME.

   
