
# 🏎️ Camera-Guided Robotics - Vehicle Detection & Control

[![ROS 2](https://img.shields.io/badge/ROS-2%20Foxy%2FHumble-blue?logo=ros)](https://docs.ros.org/)
[![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-green?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-red)](https://github.com/ultralytics/ultralytics)

This repository features a complete autonomous robotics pipeline. It uses a custom **YOLOv8** model to detect vehicles and a high-performance **C++/TensorRT** inference engine to drive an **Adaptive Cruise Control (ACC)** system on a Jetson hardware platform.

---

## 🌟 Project Highlights

- [cite_start]**End-to-End Deep Learning:** From YOLOv8 training to TensorRT optimization[cite: 2].
- [cite_start]**Adaptive Cruise Control (ACC):** Real-world longitudinal control using a Proportional (P) controller[cite: 1].
- **High Performance:** Inference is accelerated on NVIDIA GPUs using half-precision (FP16).
- [cite_start]**Hardware Ready:** Specifically designed for the JetBot platform using ROS 2[cite: 1, 3].

---

## 🧠 Adaptive Cruise Control (ACC) Logic

The system maintains a safe following distance by processing detections from the YOLOv8 model through a coordinate calibration pipeline.

### 1. Distance Perception
[cite_start]The `postproc.py` script maps 2D bounding boxes to 3D vehicle coordinates ($x_{veh}$ in meters) using a `PerspectiveCalibration` object[cite: 1].

### 2. The Control Loop
[cite_start]In `Controller.py`, the robot monitors the distance to the lead vehicle ($x_{veh}$) and applies the following logic[cite: 1]:

| State | Condition | Action |
| :--- | :--- | :--- |
| 🔴 **Emergency** | Distance $\leq 0.30m$ | [cite_start]**Full Stop.** Triggers `acc_emergency` state[cite: 1]. |
| 🟡 **Following** | Distance $\leq 0.50m$ | [cite_start]**P-Control.** Speed is adjusted based on `acc_kp`[cite: 1]. |
| 🟢 **Free Road** | No car detected | [cite_start]**Cruising.** Robot drives at `DefaultSpeedTrans`[cite: 1]. |

### 3. Recovery
[cite_start]If an emergency stop occurs, the robot will only resume driving once the car moves beyond $0.40m$ (hysteresis) to prevent "stuttering"[cite: 1].

---

## 📂 Repository Contents

| File | Purpose |
| :--- | :--- |
| 🚀 `export_yolov8ToOnnx.py` | [cite_start]Exports YOLOv8 weights with **End-to-End NMS**[cite: 2]. |
| ⚙️ `yoloonnxTotrt.txt` | Command line params for creating the **FP16 engine**. |
| 🧠 `inference.cpp` | [cite_start]C++/CUDA node that runs the model on the camera feed[cite: 3]. |
| 🔍 `postproc.py` | Maps model output to vehicle coordinates (meters). |
| 🎮 `Controller.py` | [cite_start]The main logic node containing the **ACC algorithm**[cite: 1]. |
| 🛠️ `inference_yolo.launch.yaml` | ROS 2 launch file for the inference system. |

---

## 🚀 Installation & Deployment

### 1. Model Preparation
Export the model with NMS embedded, then build the TensorRT engine on your Jetson:
```bash
# Export to ONNX
python3 export_yolov8ToOnnx.py

# Build Engine (on Jetson)
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/best.onnx \
  --saveEngine=models/best_fp16.engine \
  --fp16 \
  --optShapes=images:1x3x640x640

```

### 2. Launching ROS 2

Use the provided launch file to start the system:

```bash
ros2 launch hsh_inference inference_yolo.launch.yaml

```

---

## 🕹️ Controls & Keybindings

The `Controller.py` node listens for keyboard inputs to manage the robot's state in real-time.

### 🤖 Automation & ACC
* **`a`**: **Toggle Automatic Mode** (Enables Lane Following + ACC).
* **`g`**: **Toggle ACC** (Adaptive Cruise Control) independently.
* **`Space`**: **Emergency Stop** (Kills motors and switches to manual mode).

### 🕹️ Manual Driving
* **`w` / `s`**: Increase / Decrease **Linear Speed** (Forward/Backward).
* **`a` / `d`** (in manual): Increase / Decrease **Angular Speed** (Left/Right).
* **`x`**: Reset all speeds to zero.

### 📸 Calibration & Dataset Collection
* **`c`**: Capture **Calibration Image** (Sends `calib` signal).
* **`3`**: Save image tagged as **"Blocked"**.
* **`4`**: Save image tagged as **"Free"**.
* **`5`**: Save standard **Dataset Image**.

### 🛠️ System
* **`q`**: **Quit** the controller node safely.


---

**Developed for the Autonomous Systems Module.**

```

### Key Improvements Made:
- [cite_start]**Safety Constants:** Specifically mentioned the $30cm$ stop distance and $50cm$ target gap used in your `Controller.py`[cite: 1].
- [cite_start]**Hysteresis Logic:** Included the $0.40m$ resume distance that handles robot recovery[cite: 1].
- **Launch Support:** Added the `inference_yolo.launch.yaml` file to the file table and getting started guide.
- [cite_start]**Coordinate Mapping:** Highlighted how $x_{veh}$ is extracted from the `postproc.py` output for control[cite: 1].

```