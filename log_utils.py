# log_utils.py
import csv
from pathlib import Path
from datetime import datetime

CSV_HEADER = [
    "timestamp", "image_name", "image_sha",
    "platform", "model",
    "inference_ms", "fps",
    "cpu_percent", "mem_percent", "rss_mb",
    "detections"
]

def _ensure_header(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

def append_log_entry(
    csv_path: Path,
    platform: str,
    model: str,
    inference_ms: float,
    fps: float,
    cpu_percent: float,
    mem_percent: float,
    rss_mb: float,
    image_name: str = "",
    image_sha: str = "",
    detections: int = 0,
):
    """
    Append a row to the log CSV. Backward-compatible: old callers without image info still work.
    """
    _ensure_header(csv_path)
    row = [
        datetime.now().isoformat(timespec="seconds"),
        image_name, image_sha,
        platform, model,
        round(float(inference_ms), 4), round(float(fps), 4),
        round(float(cpu_percent), 2), round(float(mem_percent), 2), round(float(rss_mb), 2),
        int(detections),
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
