# Multi-Model Multi-Platform Object Detection Comparison

## 📖 Project Overview
This project is a **multi-model, multi-platform object detection comparison system** built with Streamlit.  
It supports detection using **YOLOv3-Tiny, YOLOv5, YOLOv8, and MobileNetSSD** models, and allows performance comparison across CPU, GPU, and external platforms (FPGA, Jetson, Raspberry Pi, etc.).

## ✨ Features
- 🔍 **Multiple Models**: YOLOv3-Tiny, YOLOv5, YOLOv8, MobileNetSSD  
- ⚡ **Multi-Platform**: Local CPU, Local GPU, and external CSV data import  
- 📊 **Performance Metrics**: Inference time, FPS, CPU & Memory usage, Detections count  
- 📈 **Visualization**: Automatic chart plotting for performance comparison  
- 📥 **CSV Import**: Supports importing detection results from other platforms
## 📂 Directory Structure
mmp-object-detection/
│
├── app.py # Streamlit main app
├── config.py # Configuration file
├── yolo_wrapper.py # Model inference wrapper
├── processor.py # Detection processing
├── log_utils.py # Logging utilities
├── models/ # Model files
│ ├── v3tiny/ # YOLOv3-Tiny model files
│ ├── v5/ # YOLOv5 model
│ ├── v8/ # YOLOv8 model
│ └── mobilenetssd/ # MobileNetSSD (deploy.prototxt & mobilenet_iter_73000.caffemodel)
├── live_input/ # Uploaded or fetched images
├── results/ # Detection results
└── logs/ # Detection logs (CSV)




## 🚀 Installation & Usage

### 1️⃣ Environment Setup
Make sure you have Python **3.10** installed. Then install dependencies:
```bash
pip install -r requirements.txt
2️⃣ Prepare Model Files
Place YOLO models in corresponding folders (models/v3tiny/, models/v5/, models/v8/).

Place MobileNetSSD files in:

bash

models/mobilenetssd/deploy.prototxt
models/mobilenetssd/mobilenet_iter_73000.caffemodel
3️⃣ Run the Application
streamlit run app.py
The system will open in your browser at http://localhost:8501
