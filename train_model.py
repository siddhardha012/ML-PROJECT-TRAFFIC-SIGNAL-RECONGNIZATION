"""
Traffic Signal Recognition - Training Pipeline
Trains a Deep Convolutional Neural Network (CNN) on traffic signal classes.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)

import config
from utils.preprocessing import validate_image, format_class_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_available_classes(train_dir: Path) -> List[str]:
    """
    Scan train directory for class subdirectories.
    Returns alphabetically sorted list of class names.
    """
    if not train_dir.exists():
        return []
    classes = [d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    classes.sort()
    return classes


def load_dataset_from_directory(
    data_dir: Path,
    target_size: Tuple[int, int] = (config.IMG_WIDTH, config.IMG_HEIGHT)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load, validate, resize, and normalize images from class subfolders.
    Handles corrupted/invalid images gracefully without halting.
    """
    classes = get_available_classes(data_dir)
    if not classes:
        raise FileNotFoundError(
            f"No class folders found in directory: {data_dir}\n"
            f"Expected folders like: {config.DEFAULT_CLASSES}"
        )

    images: List[np.ndarray] = []
    labels: List[int] = []
    class_counts: Dict[str, int] = {c: 0 for c in classes}
    corrupted_count = 0

    print("=" * 60, flush=True)
    print(f"Scanning dataset in: {data_dir}", flush=True)
    print("=" * 60, flush=True)

    for class_idx, class_name in enumerate(classes):
        class_folder = data_dir / class_name
        valid_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower().lstrip(".") in config.ALLOWED_EXTENSIONS
        ]

        for file_path in valid_files:
            is_valid, err, pil_img = validate_image(file_path)
            if not is_valid or pil_img is None:
                logger.warning(f"Skipping corrupted image: {file_path.name} ({err})")
                corrupted_count += 1
                continue

            try:
                # Resize to target size (64x64) and normalize to [0, 1]
                resized = pil_img.resize(target_size)
                img_array = np.array(resized, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(class_idx)
                class_counts[class_name] += 1
            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")
                corrupted_count += 1

    print("Dataset Loading Summary:", flush=True)
    for c_name, count in class_counts.items():
        print(f"  - {format_class_name(c_name)} ({c_name}): {count} images", flush=True)
    print(f"Total valid images loaded: {len(images)} (Corrupted/Skipped: {corrupted_count})", flush=True)

    if len(images) == 0:
        raise ValueError(
            f"No valid images were found in {data_dir}.\n"
            f"Please populate the folders or run: python generate_sample_data.py"
        )

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, classes


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Builds the CNN architecture specified in the project requirements:
    Input -> Augmentation (Direction-safe)
          -> Conv2D -> ReLU -> BatchNorm -> MaxPooling
          -> Conv2D -> ReLU -> BatchNorm -> MaxPooling
          -> Conv2D -> ReLU -> BatchNorm -> MaxPooling
          -> Flatten -> Dense -> Dropout -> Dense Softmax
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Direction-safe data augmentation layers (active only during training)
        # Horizontal flip is strictly avoided to preserve directional signs (Turn Left vs Turn Right)
        layers.RandomRotation(0.06, fill_mode="nearest", name="aug_rotation"),
        layers.RandomTranslation(height_factor=0.06, width_factor=0.06, fill_mode="nearest", name="aug_translation"),
        layers.RandomZoom(height_factor=(-0.06, 0.06), fill_mode="nearest", name="aug_zoom"),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1"),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2"),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool3"),

        # Flatten and Dense Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="output_softmax")
    ], name="TrafficSignalCNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_metrics(history, results_dir: Path):
    """Generate presentation-quality graphs for training & validation accuracy and loss."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    epochs = range(1, len(history.history["accuracy"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history.history["accuracy"], "b-o", linewidth=2, label="Training Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(epochs, history.history["val_accuracy"], "g-s", linewidth=2, label="Validation Accuracy")
    plt.title("Traffic Signal Model - Training vs Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([0, 1.05])
    plt.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy.png")
    plt.close()
    print(f"Accuracy graph saved to: {results_dir / 'accuracy.png'}", flush=True)

    # 2. Loss Plot
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history.history["loss"], "r-o", linewidth=2, label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], "m-s", linewidth=2, label="Validation Loss")
    plt.title("Traffic Signal Model - Training vs Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Categorical Cross-Entropy)", fontsize=12)
    plt.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(results_dir / "loss.png")
    plt.close()
    print(f"Loss graph saved to: {results_dir / 'loss.png'}", flush=True)


def train(epochs: int = 15, batch_size: int = config.BATCH_SIZE):
    """Full end-to-end training routine."""
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    print("Initializing Traffic Signal Recognition training pipeline...", flush=True)

    # Step 1: Load training images
    X, y, classes = load_dataset_from_directory(config.TRAIN_DIR)
    num_classes = len(classes)

    # Step 2: Convert labels to categorical one-hot encoding
    y_cat = to_categorical(y, num_classes=num_classes)

    # Step 3: Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print(f"Training split: {len(X_train)} samples | Validation split: {len(X_val)} samples", flush=True)

    # Step 4: Build CNN architecture
    model = build_cnn_model(config.INPUT_SHAPE, num_classes)
    model.summary()

    # Step 5: Configure Callbacks
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Step 6: Train the CNN directly on numpy tensors
    print(f"Beginning training for up to {epochs} epochs...", flush=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Step 7: Save class names mapping
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)
    print(f"Saved class names to: {config.CLASS_NAMES_PATH}", flush=True)

    # Step 8: Save final model if not already saved by checkpoint
    if not config.MODEL_PATH.exists():
        model.save(str(config.MODEL_PATH))
    print(f"Model successfully saved to: {config.MODEL_PATH}", flush=True)

    # Step 9: Generate Plots
    plot_training_metrics(history, config.RESULTS_DIR)

    # Step 10: Print final accuracies
    final_train_acc = history.history["accuracy"][-1] * 100
    final_val_acc = history.history.get("val_accuracy", [0])[-1] * 100
    print("\n" + "=" * 60, flush=True)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY", flush=True)
    print(f"Final Training Accuracy:   {final_train_acc:.2f}%", flush=True)
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%", flush=True)
    print(f"Trained Model saved to:    {config.MODEL_PATH}", flush=True)
    print(f"Classes file saved to:     {config.CLASS_NAMES_PATH}", flush=True)
    print("=" * 60 + "\n", flush=True)

    return history


if __name__ == "__main__":
    train()
