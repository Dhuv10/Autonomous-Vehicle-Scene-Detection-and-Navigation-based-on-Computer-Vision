# 🚗 Autonomous Vehicle Scene Detection and Navigation

This project simulates autonomous vehicle behavior by combining real-time object detection, traffic sign recognition, object tracking, and lane detection using modern computer vision techniques.

---

## 📌 Project Overview

The goal is to develop a computer vision-based system that can:
- Detect and track surrounding objects using **YOLOv8**
- Recognize traffic signs using a **CNN trained on the German Traffic Sign Dataset**
- Track moving objects using **Kalman Filter**, **DeepSORT**, and **Tracktor**
- Detect road lanes using **edge detection** and **Hough transform**

---

## 🧰 Technologies Used

- **Language**: Python
- **Libraries**: OpenCV, NumPy, TensorFlow/Keras, Ultralytics YOLOv8, matplotlib
- **Object Detection**: YOLOv8 (COCO dataset)
- **Traffic Sign Classification**: CNN (German Traffic Sign Dataset)
- **Object Tracking**: Kalman Filter, DeepSORT, Tracktor
- **Lane Detection**: Classical image processing (Canny + Hough Transform)
- **IDE**: Jupyter Notebook, VS Code

---

## 🧠 Features

- Real-time object detection with bounding boxes and confidence scores
- Robust traffic sign classification with preprocessed CNN pipeline
- Smooth object tracking using multiple tracking algorithms
- Dynamic lane detection and path estimation
- Modular and testable code architecture

---

## 📁 Project Structure

```plaintext
autonomous-vehicle-scene-detection/
├── yolov8/                  # YOLO object detection scripts
├── traffic_sign_cnn/        # CNN training and inference scripts
├── tracking/                # Kalman, DeepSORT, Tracktor scripts
├── lane_detection/          # Edge + Hough lane detection
├── results/                 # Output screenshots, demo videos
├── notebooks/               # Jupyter analysis notebooks
├── requirements.txt         # Required Python libraries
└── README.md
