import cv2
import torch
import numpy as np
from ultralytics import YOLO as YOLOv8
from pathlib import Path

BASE_DIR = Path(__file__).parent

# === YOLOv3-Tiny paths ===
YOLOV3_CFG = BASE_DIR / "models" / "v3tiny" / "yolov3-tiny.cfg"
YOLOV3_WEIGHTS = BASE_DIR / "models" / "v3tiny" / "yolov3-tiny.weights"
YOLOV3_NAMES = BASE_DIR / "models" / "v3tiny" / "coco.names"

# === YOLOv5 path ===
YOLOV5_PT = BASE_DIR / "models" / "v5" / "yolov5s.pt"

# === YOLOv8 path ===
YOLOV8_PT = BASE_DIR / "models" / "v8" / "yolov8n.pt"

# === MobileNet SSD (Caffe) paths ===
MOBILENETSSD_DIR = BASE_DIR / "models" / "mobilenetssd"
MOBILENET_SSD_PROTO = MOBILENETSSD_DIR / "deploy.prototxt"
MOBILENET_SSD_WEIGHTS = MOBILENETSSD_DIR / "mobilenet_iter_73000.caffemodel"
MOBILENET_SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Cache for loaded models
_loaded_models = {}


def load_model(model_type: str, device="cpu"):
    key = f"{model_type}_{device}"

    if key in _loaded_models:
        return _loaded_models[key]

    if model_type == "YOLOv3-Tiny":
        net = cv2.dnn.readNetFromDarknet(str(YOLOV3_CFG), str(YOLOV3_WEIGHTS))
        if device == "cpu":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == "cuda":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(YOLOV3_NAMES, "r") as f:
            names = [line.strip() for line in f.readlines()]
        _loaded_models[key] = ("YOLOv3-Tiny", net, names)

    elif model_type == "YOLOv5":
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(YOLOV5_PT), device=device)
        _loaded_models[key] = ("YOLOv5", model)

    elif model_type == "YOLOv8":
        model = YOLOv8(str(YOLOV8_PT))
        model.to(device)
        _loaded_models[key] = ("YOLOv8", model)

    elif model_type == "MobileNetSSD":
        net = cv2.dnn.readNetFromCaffe(str(MOBILENET_SSD_PROTO), str(MOBILENET_SSD_WEIGHTS))
        if device == "cpu":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == "cuda":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        _loaded_models[key] = ("MobileNetSSD", net)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return _loaded_models[key]


def infer(image, model_type: str, device="cpu"):
    loaded = load_model(model_type, device)

    if loaded[0] == "YOLOv3-Tiny":
        _, net, names = loaded
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(net.getUnconnectedOutLayersNames())
        results = []
        for output in layer_outputs:
            for det in output:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > 0.5:
                    box = det[0:4] * np.array([width, height, width, height])
                    cx, cy, w, h = box.astype("int")
                    x = int(cx - w / 2); y = int(cy - h / 2)
                    results.append({
                        "class": names[class_id],
                        "confidence": confidence,
                        "bbox": (x, y, x + int(w), y + int(h))
                    })
        return results

    elif loaded[0] == "YOLOv5":
        _, model = loaded
        results = model(image)
        df = results.pandas().xyxy[0]
        return [
            {
                "class": row["name"],
                "confidence": float(row["confidence"]),
                "bbox": (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
            } for _, row in df.iterrows()
        ]

    elif loaded[0] == "YOLOv8":
        _, model = loaded
        results = model(image)
        dets = results[0].boxes.data.cpu().numpy()
        names = results[0].names
        return [
            {
                "class": names[int(cls)],
                "confidence": float(conf),
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            } for x1, y1, x2, y2, conf, cls in dets
        ]

    elif loaded[0] == "MobileNetSSD":
        _, net = loaded
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     scalefactor=0.007843, size=(300, 300), mean=127.5)
        net.setInput(blob)
        detections = net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.5:
                continue
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
            cls_name = MOBILENET_SSD_CLASSES[idx] if 0 <= idx < len(MOBILENET_SSD_CLASSES) else str(idx)
            results.append({
                "class": cls_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })
        return results

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
