# config.py
from pathlib import Path

# ====== Base directories ======
BASE_DIR = Path(__file__).resolve().parent

# ====== Data folders ======
LIVE_INPUT_DIR = BASE_DIR / "live_input"   # Original / uploaded / downloaded images
RESULTS_DIR = BASE_DIR / "results"         # Processed & annotated images
LOGS_DIR = BASE_DIR / "logs"               # Performance logs
LOG_FILE = LOGS_DIR / "detections.csv"     # CSV log file

# ====== Model info ======
MODELS_DIR = BASE_DIR / "models"
MODELS = [
    "YOLOv3-Tiny",
    "YOLOv5",
    "YOLOv8",
    "MobileNetSSD"
]

# ====== Server URLs ======
SERVER_LATEST_URL = "http://206.189.125.120/get_latest.php?res=640x480"
SERVER_FOLDER_URL = "http://206.189.125.120/view.php?res=640x480"

# ====== Ensure directories exist ======
LIVE_INPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
