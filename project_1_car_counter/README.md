# Car Counter Using YOLO and Python

This project is a car counter that detects and counts the number of cars passing through a predefined line in a video using YOLO for object detection and Python.

## Project Overview

The main objective of this project is to develop a system that can automatically **detect** and **count** vehicles passing through a specific line in a video feed. This can be particularly useful for traffic management, monitoring, and data collection.

## Technologies Used

- **Python**: The primary programming language used for developing the project.
- **OpenCV**: Used for video processing and handling image operations.
- **YOLO (You Only Look Once)**: A state-of-the-art, real-time object detection system. The **YOLOv8** model is used in this project.
- **cvzone**: An OpenCV wrapper to make computer vision tasks easier.
- **SORT (Simple Online and Realtime Tracking)**: An algorithm used for tracking objects detected by YOLO.

## How It Works

1. **Video Capture**: The system captures frames from the input video file (`cars.mp4`).
2. **Masking**: A mask image (`mask.png`) is applied to focus on the region of interest in the video.
3. **Object Detection**: YOLOv8 is used to detect objects in the masked region of each frame.
4. **Tracking**: The SORT algorithm tracks the detected objects across frames.
5. **Counting**: The system counts vehicles that cross a predefined line.

## Dependencies

Make sure you have the following dependencies installed:
- OpenCV
- YOLO (via the `ultralytics` package)
- cvzone
- SORT


