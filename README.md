# YOLO ONNX C++ Video Inference

## Description
This project demonstrates how to run YOLO object detection using ONNX Runtime and OpenCV in C++. It supports both image and video input, drawing bounding boxes and class labels on detected objects.

## Features
- ONNX Runtime inference for YOLO models
- Video and image input support
- Automatic letterboxing and bounding box rescaling
- Real-time visualization with OpenCV

## Requirements
- C++17 or later
- OpenCV (>=4.x)
- ONNX Runtime (>=1.8)
- CMake (>=3.10)
- A YOLO ONNX model (e.g., yolo11n.onnx)
- COCO class labels file

## Build Instructions

```bash
git clone <https://github.com/nguyec-baokhang/Onnx-OpenCV.git
cd <project-folder>
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./object_detection
```

- Edit `main.cpp` to set your model, video, and label paths if needed.

## Example

- Input: `videos/tennis.mp4`
- Output: Video window with bounding boxes and labels drawn on detected objects.

## File Structure

```
yolo-onnx/
├── src/
│   └── main.cpp
├── models/
│   └── yolov8n.onnx
├── videos/
│   └── tennis.mp4
├── labels/
│   └── coco.txt
├── CMakeLists.txt
└── README.md
```

## Troubleshooting

- If you see empty frames or no detections, check your model and label paths.
- Make sure OpenCV and ONNX Runtime are installed and linked correctly.
