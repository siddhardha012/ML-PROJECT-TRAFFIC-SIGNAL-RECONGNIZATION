"""
Traffic Signal Recognition - Evaluation Pipeline
Calculates classification metrics (Accuracy, Precision, Recall, F1)
and generates a presentation-ready confusion matrix plot.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config
from utils.preprocessing import validate_image, format_class_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_test_dataset(test_dir: Path, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test images and ground truth labels corresponding to class_names."""
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    images = []
    labels = []
    found_classes = set()

    for idx, class_name in enumerate(class_names):
        folder = test_dir / class_name
        if not folder.exists() or not folder.is_dir():
            continue

        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower().lstrip(".") in config.ALLOWED_EXTENSIONS
        ]

        for file_path in files:
            is_valid, _, pil_img = validate_image(file_path)
            if not is_valid or pil_img is None:
                continue

            try:
                resized = pil_img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
                arr = np.array(resized, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(idx)
                found_classes.add(class_name)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")

    if not images:
        raise ValueError(
            f"No valid test images found in {test_dir}.\n"
            f"Please ensure images exist in subfolders: {[c for c in class_names]}"
        )

    logger.info(f"Loaded {len(images)} test samples across {len(found_classes)} classes.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), class_names


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Draw a clean, presentation-ready confusion matrix with annotations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_labels = [format_class_name(c) for c in class_names]

    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Traffic Signal Classification - Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, display_labels, rotation=40, ha="right", fontsize=10)
    plt.yticks(tick_marks, display_labels, fontsize=10)

    # Annotate cells with values
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=11,
                fontweight="bold"
            )

    plt.ylabel("True Class Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def evaluate():
    """Main evaluation function."""
    import tensorflow as tf

    logger.info("Starting model evaluation...")

    # Verify model and classes exist
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_PATH}.\n"
            "Please train the model first by running: python train_model.py"
        )
    if not config.CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names list not found at {config.CLASS_NAMES_PATH}."
        )

    # Load classes
    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load test dataset
    X_test, y_test, class_names = load_test_dataset(config.TEST_DIR, class_names)

    # Load trained model
    logger.info(f"Loading trained model from {config.MODEL_PATH}...")
    model = tf.keras.models.load_model(str(config.MODEL_PATH))

    # Inference on test set
    y_pred_probs = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    display_names = [format_class_name(c) for c in class_names]

    # Generate classification report string
    report = classification_report(
        y_test,
        y_pred,
        target_names=display_names,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    # Save Confusion Matrix Plot
    plot_confusion_matrix(cm, class_names, config.CONFUSION_MATRIX_PATH)

    # Print Report to console
    print("\n" + "=" * 65)
    print("TRAFFIC SIGNAL RECOGNITION - EVALUATION REPORT")
    print("=" * 65)
    print(f"Overall Test Accuracy:     {acc * 100:.2f}%")
    print(f"Macro Precision:           {prec_macro * 100:.2f}%")
    print(f"Macro Recall:              {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:            {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:         {f1_weighted * 100:.2f}%")
    print("-" * 65)
    print("Detailed Classification Breakdown by Class:")
    print(report)
    print("=" * 65)
    print(f"Confusion Matrix saved to: {config.CONFUSION_MATRIX_PATH}")
    print("=" * 65 + "\n")

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "classification_report": report
    }


if __name__ == "__main__":
    evaluate()
