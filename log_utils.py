# log_utils.py
import csv
from pathlib import Path

# Legacy headers (kept for backward-compatible appends)
OLD7_HEADER = ["Platform", "Model", "Inference(ms)", "FPS", "CPU(%)", "Mem(%)", "RSS(MB)"]
NEW9_CAMEL  = OLD7_HEADER + ["Detections", "AvgConfidence"]
FULL11_CAMEL = ["ImageName", "ImageSHA"] + NEW9_CAMEL

# New recommended header (snake_case)
FULL11_SNAKE = [
    "image_name", "image_sha",
    "platform", "model",
    "inference_ms", "fps",
    "cpu_percent", "mem_percent", "rss_mb",
    "detections", "avg_confidence",
]

def _read_header(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if not first:
                return []
            return [c.strip() for c in first.split(",")]
    except FileNotFoundError:
        return []
    except Exception:
        return []

def append_log_entry(
    log_file,
    platform,
    model,
    inf_time,
    fps,
    cpu_usage,
    mem_usage,
    rss_mem,
    detections=None,
    avg_confidence=None,
    image_name=None,
    image_sha=None,
):
    """Append a row to the detection CSV.
    Rules (no breaking changes):
    - If file doesn't exist -> create with FULL11_SNAKE (clean long-term header).
    - If file exists and header matches one of legacy headers -> keep writing that format.
    - If file exists and header already FULL11_SNAKE -> write that format.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()
    header = _read_header(log_path) if file_exists else []

    if not file_exists or not header:
        header_to_use = FULL11_SNAKE
        write_header = True
    elif header == FULL11_SNAKE:
        header_to_use = FULL11_SNAKE
        write_header = False
    elif header == FULL11_CAMEL:
        header_to_use = FULL11_CAMEL
        write_header = False
    elif header == NEW9_CAMEL:
        header_to_use = NEW9_CAMEL
        write_header = False
    elif header == OLD7_HEADER:
        header_to_use = OLD7_HEADER
        write_header = False
    else:
        # Unknown header: be conservative, keep existing column count
        header_to_use = header
        write_header = False

    # normalize input numbers
    inf_time = float(inf_time)
    fps = float(fps)
    cpu_usage = float(cpu_usage)
    mem_usage = float(mem_usage)
    rss_mem = float(rss_mem)
    det = 0 if detections is None else int(detections)
    avgc = 0.0 if avg_confidence is None else float(avg_confidence)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header_to_use)

        if header_to_use == FULL11_SNAKE:
            w.writerow([
                image_name or "",
                image_sha or "",
                platform,
                model,
                f"{inf_time:.4f}",
                f"{fps:.4f}",
                f"{cpu_usage:.1f}",
                f"{mem_usage:.1f}",
                f"{rss_mem:.1f}",
                det,
                f"{avgc:.6f}",
            ])
        elif header_to_use == FULL11_CAMEL:
            w.writerow([
                image_name or "",
                image_sha or "",
                platform,
                model,
                f"{inf_time:.4f}",
                f"{fps:.4f}",
                f"{cpu_usage:.1f}",
                f"{mem_usage:.1f}",
                f"{rss_mem:.1f}",
                det,
                f"{avgc:.6f}",
            ])
        elif header_to_use == NEW9_CAMEL:
            w.writerow([
                platform,
                model,
                f"{inf_time:.4f}",
                f"{fps:.4f}",
                f"{cpu_usage:.1f}",
                f"{mem_usage:.1f}",
                f"{rss_mem:.1f}",
                det,
                f"{avgc:.6f}",
            ])
        elif header_to_use == OLD7_HEADER:
            w.writerow([
                platform,
                model,
                f"{inf_time:.4f}",
                f"{fps:.4f}",
                f"{cpu_usage:.1f}",
                f"{mem_usage:.1f}",
                f"{rss_mem:.1f}",
            ])
        else:
            # Unknown header length: try to fit common subset (platform..rss)
            row = [platform, model, f"{inf_time:.4f}", f"{fps:.4f}",
                   f"{cpu_usage:.1f}", f"{mem_usage:.1f}", f"{rss_mem:.1f}"]
            w.writerow(row[:len(header_to_use)])
