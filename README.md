# Traffic Signal Recognition Using Machine Learning

An end-to-end Deep Learning and Computer Vision application that recognizes and classifies traffic signals from uploaded photos or live webcam streams. Built with Python, TensorFlow/Keras, OpenCV, and Flask, designed specifically for academic evaluation, final-year engineering projects, and research exploration.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Dataset Setup](#dataset-setup)
6. [Installation & Setup](#installation--setup)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [CLI Image Prediction](#cli-image-prediction)
10. [Running the Flask Web Application](#running-the-flask-web-application)
11. [Adding New Traffic Signal Classes](#adding-new-traffic-signal-classes)
12. [Project Documentation Guide](#project-documentation-guide)

---

## Project Overview

Traffic Signal and Sign Recognition (TSR) is an essential capability for Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles. This project implements a Deep Convolutional Neural Network (CNN) trained to classify road signals into their respective functional categories.

The system features:
- **Computer Vision Pipeline**: Validates image integrity, standardizes color spaces to RGB, scales inputs to $64 \times 64$, and normalizes pixel intensities to $[0, 1]$.
- **Direction-Safe Augmentation**: Applies random rotation ($\pm 8^\circ$), translation, and zoom while strictly avoiding horizontal flipping to preserve directional semantics (e.g. differentiating between *Turn Left* and *Turn Right*).
- **Interactive Flask UI**: A responsive, road-themed web interface offering drag-and-drop uploads, live browser webcam capture, confidence gauges, top-3 ranked predictions, low-confidence warning indicators, and a persistent SQLite prediction history log.

---

## Features

- **Multi-Class Detection**: Identifies 7 standard road signal categories out of the box:
  1. `Stop`
  2. `No_Entry`
  3. `Traffic_Light`
  4. `Speed_Limit_30`
  5. `Speed_Limit_50`
  6. `Turn_Left`
  7. `Turn_Right`
- **Dynamic Extensibility**: Automatically discovers classes from directory names; adding new classes requires zero code changes.
- **Modern Responsive Web UI**: Road/traffic themed dark mode with progress meters, color-coded status badges, and real-time previews.
- **Live Webcam Integration**: Capture frames straight from browser webcams and classify instantly via AJAX.
- **Top 3 Confidence Ranking**: Displays the primary class along with the next two most probable alternatives.
- **Safety Threshold Checks**: Flags ambiguous or low-confidence predictions ($< 50\%$) with a warning prompt to upload a clearer image.
- **Persistent SQLite Audit Log**: Automatically saves every prediction with thumbnail filename, predicted label, confidence percentage, and timestamp.
- **Presentation-Ready Plots**: Generates high-resolution training/validation accuracy, loss, and confusion matrix charts in the `results/` folder.

---

## Technologies

### Backend & Machine Learning
- **Python 3.10+**: Core programming language.
- **TensorFlow / Keras**: Deep Convolutional Neural Network modeling and training.
- **OpenCV & Pillow**: High-performance image loading, validation, and transformations.
- **NumPy & Pandas**: Array operations and statistical evaluations.
- **Scikit-learn**: Classification reports, macro/weighted metrics, and confusion matrix calculation.
- **Matplotlib**: Generation of academic evaluation graphs and confusion heatmaps.
- **SQLite3**: Local lightweight database for logging prediction transactions.

### Frontend
- **Flask**: Python WSGI web microframework.
- **HTML5 & CSS3**: Modern glassmorphic dark UI with traffic-themed accents.
- **JavaScript (Vanilla ES6)**: HTML5 File API drag-and-drop and WebRTC MediaDevices camera streaming.

---

## Project Structure

```text
Traffic_Signal_Recognition/
│
├── app.py                      # Flask web application server & routes
├── train_model.py              # CNN model training script with callbacks
├── predict.py                  # CLI inference utility for single images
├── evaluate_model.py           # Model testing & metrics report generator
├── generate_sample_data.py     # Synthetic dataset generator for rapid testing
├── requirements.txt            # Python dependencies
├── config.py                   # Centralized configuration & hyperparameters
├── README.md                   # Quickstart guide and technical documentation
├── DOCUMENTATION.md            # Comprehensive academic final-year project report
├── traffic_signal.db           # SQLite database for prediction history
│
├── dataset/
│   ├── train/                  # Training set organized in class subfolders
│   │   ├── Stop/
│   │   ├── No_Entry/
│   │   ├── Traffic_Light/
│   │   ├── Speed_Limit_30/
│   │   ├── Speed_Limit_50/
│   │   ├── Turn_Left/
│   │   └── Turn_Right/
│   │
│   └── test/                   # Test set organized in class subfolders
│       ├── Stop/
│       ├── No_Entry/
│       ├── Traffic_Light/
│       ├── Speed_Limit_30/
│       ├── Speed_Limit_50/
│       ├── Turn_Left/
│       └── Turn_Right/
│
├── models/
│   ├── traffic_signal_model.keras  # Saved Keras CNN model weights
│   └── class_names.json            # Ordered JSON list of class identifiers
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom responsive styles & animations
│   ├── js/
│   │   └── script.js           # Drag-and-drop & webcam capture controller
│   └── uploads/                # Directory for user-uploaded image files
│
├── templates/
│   ├── base.html               # Shared master layout with navbar & footer
│   ├── index.html              # Main dashboard: file upload & webcam modal
│   ├── result.html             # Detailed classification breakdown & gauges
│   ├── about.html              # Project architecture & theoretical background
│   └── history.html            # SQLite historical inference logs table
│
├── utils/
│   ├── __init__.py             # Python package marker
│   ├── preprocessing.py        # Image validation, RGB resize, normalization
│   ├── prediction.py           # Model wrapper, inference, ranking, badges
│   └── database.py             # SQLite helper functions
│
├── results/
│   ├── accuracy.png            # Training vs validation accuracy curve
│   ├── loss.png                # Training vs validation loss curve
│   └── confusion_matrix.png    # Annotated confusion matrix heatmap
│
└── tests/
    └── test_pipeline.py        # Unit test suite verifying components
```

---

## Dataset Setup

### Option A: Instant Synthetic Dataset (Zero Download Setup)
For instant evaluation and testing of the end-to-end pipeline without downloading hundreds of megabytes, run the built-in generator:
```bash
python generate_sample_data.py
```
This renders visually distinct signs for all 7 classes with varied backgrounds, daylight conditions, scaling, and rotations into `dataset/train/` and `dataset/test/`.

### Option B: Using Real-World Datasets (e.g. GTSRB)
For real-world camera captures, the **German Traffic Sign Recognition Benchmark (GTSRB)** is standard:
1. Download the GTSRB dataset from Kaggle or the official benchmark portal:
   - [Kaggle GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. Map GTSRB class folders to your desired class names:
   - Class `14` $\rightarrow$ `Stop`
   - Class `17` $\rightarrow$ `No_Entry`
   - Class `1`  $\rightarrow$ `Speed_Limit_30`
   - Class `2`  $\rightarrow$ `Speed_Limit_50`
   - Class `34` $\rightarrow$ `Turn_Left`
   - Class `33` $\rightarrow$ `Turn_Right`
3. Place images into their respective subfolders under `dataset/train/<Class_Name>/` and `dataset/test/<Class_Name>/`.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit).
- Git (optional).

### 2. Create and Activate Virtual Environment
Open PowerShell or Command Prompt in the project root:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1

# Or activate on Windows (Command Prompt)
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Model Training

To train the Deep Convolutional Neural Network on the dataset:
```bash
python train_model.py
```

### What `train_model.py` does:
1. Scans `dataset/train/` and detects all class subfolders automatically.
2. Validates every image file and skips corrupt or empty files gracefully.
3. Resizes images to $64 \times 64$ and normalizes pixels to $[0.0, 1.0]$.
4. Performs stratified 80/20 train/validation split.
5. Injects directional-safe data augmentation (rotation, translation, zoom).
6. Compiles the CNN using `Adam` optimizer and `categorical_crossentropy` loss.
7. Executes training with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
8. Saves the best weights to `models/traffic_signal_model.keras`.
9. Exports the class index mapping to `models/class_names.json`.
10. Saves evaluation graphs to `results/accuracy.png` and `results/loss.png`.

---

## Model Evaluation

To evaluate model accuracy, precision, recall, and generate a confusion matrix against the held-out test dataset:
```bash
python evaluate_model.py
```

### Output:
- Overall test accuracy percentage.
- Macro and weighted Precision, Recall, and F1-Scores.
- Full Scikit-learn classification report per class.
- Saves high-resolution confusion matrix heatmap to `results/confusion_matrix.png`.

---

## CLI Image Prediction

To quickly classify any single image from the command line:
```bash
python predict.py dataset/test/Stop/stop_test_001.jpg
```

Example Output:
```text
=======================================================
      TRAFFIC SIGNAL RECOGNITION RESULT
=======================================================
Image File:        stop_test_001.jpg
Predicted Signal:  Stop (Stop)
Confidence:        99.42%
-------------------------------------------------------
Top 3 Predictions:
  1. Stop                   99.42%  [████████████████████████░]
  2. No Entry                0.35%  [░░░░░░░░░░░░░░░░░░░░░░░░░]
  3. Speed Limit 30 km/h     0.11%  [░░░░░░░░░░░░░░░░░░░░░░░░░]
=======================================================
```

---

## Running the Flask Web Application

Launch the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Web Application Workflow:
1. **Home (`/`)**: Drag and drop an image file or click **Use Live Webcam** to capture a snapshot.
2. **Preview**: Instantly inspect the chosen file in the browser before submitting.
3. **Predict**: Click **Predict Traffic Signal** to submit the file.
4. **Results (`/result`)**: View the detected signal badge, confidence meter, top 3 probability breakdown, and low-confidence warning indicators.
5. **History (`/history`)**: Review past classifications logged in SQLite, with an option to clear history.
6. **About (`/about`)**: View technical project specifications, CNN diagram, and metrics.

---

## Adding New Traffic Signal Classes

The architecture is fully dynamic and automatically adjusts to any number of classes:
1. Create a new folder under `dataset/train/` with your class name (e.g. `dataset/train/Yield/` or `dataset/train/Pedestrian/`).
2. Add corresponding training images to that folder.
3. Create the matching folder under `dataset/test/` (e.g. `dataset/test/Yield/`) with test images.
4. (Optional) In `config.py`, add human-friendly names or badge styling in `CLASS_DISPLAY_NAMES` and `CLASS_BADGE_THEMES`.
5. Retrain the model:
   ```bash
   python train_model.py
   ```
The CNN's final Softmax layer and JSON mappings will automatically scale to include the new classes!

---

## Running Unit Tests

Run the test suite to verify preprocessing, error handling, database operations, and web routes:
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## Project Documentation Guide

For the academic report containing the Abstract, Problem Statement, Methodology, System Architecture, Mathematical Formulas, Advantages, Limitations, and Future Scope, see the [DOCUMENTATION.md](file:///C:/Users/siddh/.gemini/antigravity/scratch/Traffic_Signal_Recognition/DOCUMENTATION.md) file.
