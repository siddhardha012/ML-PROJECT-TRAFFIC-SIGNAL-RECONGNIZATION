"""
Traffic Signal Recognition - CLI Prediction Script
Loads a test image, runs CNN inference, and displays predicted class, confidence,
and top 3 predictions with threshold warnings.
"""

import sys
import argparse
from pathlib import Path
import config
from utils.prediction import TrafficSignalModelWrapper

def run_prediction(image_path_str: str):
    """Executes prediction on given image path and prints structured result."""
    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"[ERROR] Specified image path does not exist: {image_path_str}")
        sys.exit(1)

    wrapper = TrafficSignalModelWrapper.get_instance()

    if not wrapper.is_model_available():
        print("\n" + "!" * 65)
        print("[ERROR] Trained model or class definitions could not be found.")
        print(f"Model path:      {config.MODEL_PATH}")
        print(f"Class list path: {config.CLASS_NAMES_PATH}")
        print("Please train the model first by running:")
        print("    python train_model.py")
        print("!" * 65 + "\n")
        sys.exit(1)

    try:
        result = wrapper.predict(image_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to execute prediction: {e}")
        sys.exit(1)

    # Output formatting
    print("\n" + "=" * 55)
    print("      TRAFFIC SIGNAL RECOGNITION RESULT")
    print("=" * 55)
    print(f"Image File:        {image_path.name}")
    print(f"Predicted Signal:  {result['display_name']} ({result['predicted_class']})")
    print(f"Confidence:        {result['confidence']:.2f}%")

    if result["is_low_confidence"]:
        print("-" * 55)
        print(f"WARNING: {result['warning_message']}")

    print("-" * 55)
    print("Top 3 Predictions:")
    for rank, item in enumerate(result["top_predictions"], start=1):
        bar_len = int(item["confidence"] / 4)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {rank}. {item['display_name']:<22} {item['confidence']:>6.2f}%  [{bar}]")

    print("=" * 55 + "\n")
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Predict traffic signal type from an input image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        help="Path to the traffic sign image (JPG/PNG)"
    )

    args = parser.parse_args()

    if not args.image_path:
        print("\nUsage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/test/Stop/sample_1.jpg\n")
        sys.exit(1)

    run_prediction(args.image_path)

if __name__ == "__main__":
    main()
