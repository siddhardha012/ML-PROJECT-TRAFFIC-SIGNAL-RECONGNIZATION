import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "traffic_signal_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

RESULTS_DIR = BASE_DIR / "results"
ACCURACY_PLOT_PATH = RESULTS_DIR / "accuracy.png"
LOSS_PLOT_PATH = RESULTS_DIR / "loss.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
DATABASE_PATH = BASE_DIR / "traffic_signal.db"

# Ensure all critical runtime directories exist
for directory in [DATASET_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, RESULTS_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image & Preprocessing Specifications
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Machine Learning Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# Prediction Parameters
CONFIDENCE_THRESHOLD = 0.50  # Below this threshold, prompt low confidence warning
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Default target traffic signal classes (can be dynamically extended by dataset folders)
DEFAULT_CLASSES = [
    "Stop",
    "No_Entry",
    "Traffic_Light",
    "Speed_Limit_30",
    "Speed_Limit_50",
    "Turn_Left",
    "Turn_Right",
]

# Pretty Display Name Mapping for UI presentation
CLASS_DISPLAY_NAMES = {
    "Stop": "Stop",
    "No_Entry": "No Entry",
    "Traffic_Light": "Traffic Light",
    "Speed_Limit_30": "Speed Limit 30 km/h",
    "Speed_Limit_50": "Speed Limit 50 km/h",
    "Turn_Left": "Turn Left Ahead",
    "Turn_Right": "Turn Right Ahead",
}

# Traffic Sign Color & Icon Theme Mapping for UI
CLASS_BADGE_THEMES = {
    "Stop": {"color": "#ef4444", "bg": "#fef2f2", "icon": "octagonal-sign"},
    "No_Entry": {"color": "#dc2626", "bg": "#fef2f2", "icon": "do-not-enter"},
    "Traffic_Light": {"color": "#f59e0b", "bg": "#fffbeb", "icon": "traffic-light"},
    "Speed_Limit_30": {"color": "#ea580c", "bg": "#fff7ed", "icon": "speedometer"},
    "Speed_Limit_50": {"color": "#ea580c", "bg": "#fff7ed", "icon": "speedometer"},
    "Turn_Left": {"color": "#2563eb", "bg": "#eff6ff", "icon": "arrow-turn-left"},
    "Turn_Right": {"color": "#2563eb", "bg": "#eff6ff", "icon": "arrow-turn-right"},
}
