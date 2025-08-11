# processor.py (only the minimal parts changed to include image_name & image_sha in logs)
import time
import cv2
import torch
import psutil
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import RESULTS_DIR, LOG_FILE
from log_utils import append_log_entry
from yolo_wrapper import infer

def _banner(image, text_lines, color=(30, 144, 255)) -> None:
    h, w = image.shape[:2]
    pad = 8
    line_h = 22
    banner_h = pad * 2 + line_h * len(text_lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    y = pad + 16
    for line in text_lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h

def _draw_detections(image, detections, box_color=(60, 220, 60)) -> None:
    for det in detections or []:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(image, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

def _annotate(image, detections, header_lines, device: str) -> None:
    color = (60, 220, 60) if device == "cpu" else (30, 144, 255)
    _draw_detections(image, detections, box_color=color)
    _banner(image, header_lines, color=color)

def _run_single(img_bgr, model_name: str, device: str, platform_label: str,
                base_stem: str, image_name: str, image_sha: str) -> Tuple[Path, Dict]:
    start = time.time()
    dets = infer(img_bgr, model_name, device=device)  # device is 'cpu' or 'cuda'
    inf_ms = (time.time() - start) * 1000.0
    fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0

    # compute new fields
    det_count = len(dets or [])
    if det_count > 0:
        avg_conf = sum(float(d.get("confidence", 0.0)) for d in dets) / det_count
    else:
        avg_conf = 0.0

    proc = psutil.Process()
    cpu_pct = psutil.cpu_percent(interval=None)
    mem_pct = psutil.virtual_memory().percent
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    header = [
        f"Model: {model_name}",
        f"Device: {platform_label}",
        f"Inference: {inf_ms:.2f} ms | FPS: {fps:.2f} | Det: {det_count} | AvgConf: {avg_conf:.2f}"
    ]
    vis = img_bgr.copy()
    _annotate(vis, dets, header, device=device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{base_stem}_{model_name}_{platform_label}_{int(time.time())}.jpg"
    cv2.imwrite(str(out_path), vis)

    # log row with image name/sha + detections + avg_confidence (handled compatibly by log_utils)
    append_log_entry(
        LOG_FILE,
        platform_label, model_name,
        inf_ms, fps, cpu_pct, mem_pct, rss_mb,
        detections=det_count, avg_confidence=avg_conf,
        image_name=image_name, image_sha=image_sha
    )

    metrics = {
        "model": model_name,
        "platform": platform_label,
        "inference_ms": round(inf_ms, 2),
        "fps": round(fps, 2),
        "cpu_percent": round(cpu_pct, 1),
        "mem_percent": round(mem_pct, 1),
        "rss_mb": round(rss_mb, 1),
        "detections": det_count,
        "avg_confidence": round(avg_conf, 4),
        "detections_list": dets or []
    }
    return out_path, metrics

def process_image_dual(image_path: str, model_name: str) -> Tuple[Path, Optional[Path], Dict, Optional[Dict]]:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    base_stem = image_path.stem
    image_name = image_path.name
    # compute a stable SHA for matching (same image across platforms)
    ok, buf = cv2.imencode(".png", img)
    image_sha = ""
    if ok:
        image_sha = hashlib.sha1(buf.tobytes()).hexdigest()

    cpu_path, cpu_metrics = _run_single(
        img_bgr=img,
        model_name=model_name,
        device="cpu",
        platform_label="LocalCPU",
        base_stem=base_stem,
        image_name=image_name,
        image_sha=image_sha
    )

    gpu_path, gpu_metrics = None, None
    if torch.cuda.is_available():
        try:
            gpu_path, gpu_metrics = _run_single(
                img_bgr=img,
                model_name=model_name,
                device="cuda",
                platform_label="LocalGPU",
                base_stem=base_stem,
                image_name=image_name,
                image_sha=image_sha
            )
        except Exception:
            gpu_path, gpu_metrics = None, None

    return cpu_path, gpu_path, cpu_metrics, gpu_metrics
