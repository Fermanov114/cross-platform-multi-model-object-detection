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
```
mmp-object-detection/
│
├── app.py                 # Streamlit main app
├── config.py              # Configuration file
├── yolo_wrapper.py        # Model inference wrapper
├── processor.py           # Detection processing
├── log_utils.py           # Logging utilities
├── models/                # Model files
│   ├── v3tiny/            # YOLOv3-Tiny model files
│   ├── v5/                # YOLOv5 model
│   ├── v8/                # YOLOv8 model
│   └── mobilenetssd/      # MobileNetSSD (deploy.prototxt & mobilenet_iter_73000.caffemodel)
├── live_input/            # Uploaded or fetched images
├── results/               # Detection results
└── logs/                  # Detection logs (CSV)
```

## 🚀 Installation & Usage

### 1️⃣ Environment Setup
Make sure you have Python **3.10** installed. Then install dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Model Files
- Place your models in corresponding folders (`models/v3tiny/`, `models/v5/`, `models/v8/`,`models/mobilenetssd`).


### 3️⃣ Run the Application
```bash
streamlit run app.py
```
The system will open in your browser at `http://localhost:8501`.

### 4️⃣ Import External CSV for Comparison
- In the app sidebar, use **"Upload CSV"** to upload detection results from other platforms.
- The system will generate comparison charts automatically.

---

# 多模型多平台目标检测对比系统

## 📖 项目简介
本项目是一个基于 Streamlit 构建的 **多模型、多平台目标检测对比系统**。  
支持使用 **YOLOv3-Tiny、YOLOv5、YOLOv8、MobileNetSSD** 模型进行检测，并可对比 CPU、GPU 以及外部平台（FPGA、Jetson、树莓派等）的性能表现。

## ✨ 功能特点
- 🔍 **多种检测模型**：YOLOv3-Tiny、YOLOv5、YOLOv8、MobileNetSSD  
- ⚡ **多平台支持**：本地 CPU、本地 GPU、外部 CSV 数据导入  
- 📊 **性能指标**：推理时间、FPS、CPU & 内存占用、检测数量  
- 📈 **可视化对比**：自动生成性能对比图表  
- 📥 **CSV 导入**：支持从其他平台导入检测结果

## 📂 目录结构
```
mmp-object-detection/
│
├── app.py                 # Streamlit 主程序
├── config.py              # 配置文件
├── yolo_wrapper.py        # 模型推理封装
├── processor.py           # 检测结果处理
├── log_utils.py           # 日志工具
├── models/                # 模型文件
│   ├── v3tiny/            # YOLOv3-Tiny 模型文件
│   ├── v5/                # YOLOv5 模型文件
│   ├── v8/                # YOLOv8 模型文件
│   └── mobilenetssd/      # MobileNetSSD（deploy.prototxt & mobilenet_iter_73000.caffemodel）
├── live_input/            # 上传或获取的图片
├── results/               # 检测结果
└── logs/                  # 检测日志（CSV）
```

## 🚀 安装与使用

### 1️⃣ 环境准备
确保已安装 Python **3.10**，然后安装依赖：
```bash
pip install -r requirements.txt
```

### 2️⃣ 准备模型文件
- 将模型放到对应文件夹（`models/v3tiny/`, `models/v5/`, `models/v8/`,`models/mobilenetss`）。


### 3️⃣ 运行程序
```bash
streamlit run app.py
```
系统将在浏览器打开 `http://localhost:8501`。

### 4️⃣ 导入外部 CSV 对比
- 在侧边栏使用 **"上传 CSV"** 按钮上传其他平台的检测结果。
- 系统会自动生成对比图表。
