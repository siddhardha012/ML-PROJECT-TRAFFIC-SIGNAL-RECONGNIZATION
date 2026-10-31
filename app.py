"""
Traffic Signal Recognition - Flask Web Application
Provides interactive web interface for uploading images, live webcam capture,
prediction results with confidence visualization, history logs, and project overview.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename

import config
from utils.preprocessing import allowed_file, validate_image, decode_base64_image
from utils.prediction import TrafficSignalModelWrapper, get_prediction_for_image
from utils.database import init_db, add_prediction, get_history, clear_history, get_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "traffic-signal-recognition-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)

# Ensure required runtime folders and database exist
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
init_db()


@app.context_processor
def inject_global_data():
    """Inject common metadata into all Jinja templates."""
    wrapper = TrafficSignalModelWrapper.get_instance()
    return {
        "model_ready": wrapper.is_model_available(),
        "year": datetime.now().year,
        "classes": config.DEFAULT_CLASSES
    }


@app.route("/")
def index():
    """Home page: Image upload, webcam trigger, and project feature highlights."""
    wrapper = TrafficSignalModelWrapper.get_instance()
    is_ready = wrapper.is_model_available()
    stats = get_stats()
    return render_template("index.html", is_ready=is_ready, stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle standard file upload form submission."""
    if "image" not in request.files:
        flash("No image file provided in upload request.", "warning")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file was selected for upload.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Invalid file extension. Supported formats are: {', '.join(config.ALLOWED_EXTENSIONS).upper()}", "warning")
        return redirect(url_for("index"))

    file_bytes = file.read()
    is_valid, err, pil_img = validate_image(file_bytes)
    if not is_valid or pil_img is None:
        flash(f"Uploaded file is corrupted or not a valid image: {err}", "danger")
        return redirect(url_for("index"))

    wrapper = TrafficSignalModelWrapper.get_instance()
    if not wrapper.is_model_available():
        flash("The machine learning model is not trained yet. Please run `python train_model.py` first.", "danger")
        return redirect(url_for("index"))

    # Generate a safe, collision-resistant filename
    original_stem = Path(secure_filename(file.filename)).stem or "upload"
    unique_filename = f"{original_stem}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = config.UPLOADS_DIR / unique_filename

    try:
        # Save validated image to static/uploads/
        pil_img.save(save_path, "JPEG", quality=95)

        # Run CNN prediction
        result = wrapper.predict(save_path)

        # Record prediction in SQLite database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_id = add_prediction(
            image_name=unique_filename,
            predicted_signal=result["display_name"],
            confidence=result["confidence"],
            timestamp=timestamp
        )

        return render_template(
            "result.html",
            result=result,
            image_url=url_for("static", filename=f"uploads/{unique_filename}"),
            timestamp=timestamp,
            record_id=record_id
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    """Handle webcam snapshot base64 submission via AJAX."""
    wrapper = TrafficSignalModelWrapper.get_instance()
    if not wrapper.is_model_available():
        return jsonify({
            "success": False,
            "error": "The ML model is not trained yet. Run `python train_model.py` first."
        }), 400

    data = request.get_json(silent=True) or {}
    image_data = data.get("image_data")

    if not image_data:
        return jsonify({"success": False, "error": "No image data received from webcam."}), 400

    try:
        raw_bytes = decode_base64_image(image_data)
        is_valid, err, pil_img = validate_image(raw_bytes)
        if not is_valid or pil_img is None:
            return jsonify({"success": False, "error": f"Invalid webcam snapshot: {err}"}), 400

        unique_filename = f"webcam_{uuid.uuid4().hex[:10]}.jpg"
        save_path = config.UPLOADS_DIR / unique_filename
        pil_img.save(save_path, "JPEG", quality=95)

        result = wrapper.predict(save_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        add_prediction(
            image_name=unique_filename,
            predicted_signal=result["display_name"],
            confidence=result["confidence"],
            timestamp=timestamp
        )

        return jsonify({
            "success": True,
            "result": result,
            "image_url": url_for("static", filename=f"uploads/{unique_filename}"),
            "timestamp": timestamp
        })

    except Exception as e:
        logger.error(f"Webcam prediction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/history")
def history():
    """Display past predictions in SQLite table."""
    records = get_history(limit=100)
    stats = get_stats()
    return render_template("history.html", records=records, stats=stats)


@app.route("/history/clear", methods=["POST"])
def clear_prediction_history():
    """Clear all records from prediction history table."""
    deleted_count = clear_history()
    flash(f"Successfully cleared {deleted_count} prediction record(s) from history.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """Academic project documentation, CNN architecture details, and tech stack."""
    results_exist = (
        config.ACCURACY_PLOT_PATH.exists() and
        config.LOSS_PLOT_PATH.exists() and
        config.CONFUSION_MATRIX_PATH.exists()
    )
    return render_template("about.html", results_exist=results_exist)


@app.route("/results/<path:filename>")
def get_result_file(filename):
    """Serve generated evaluation metric plots."""
    return send_from_directory(str(config.RESULTS_DIR), filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Programmatic REST API endpoint for integration."""
    wrapper = TrafficSignalModelWrapper.get_instance()
    if not wrapper.is_model_available():
        return jsonify({"status": "error", "message": "Model not trained."}), 503

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded in 'image' field."}), 400

    file = request.files["image"]
    file_bytes = file.read()
    is_valid, err, pil_img = validate_image(file_bytes)
    if not is_valid:
        return jsonify({"status": "error", "message": err}), 400

    unique_filename = f"api_{uuid.uuid4().hex[:8]}.jpg"
    save_path = config.UPLOADS_DIR / unique_filename
    pil_img.save(save_path, "JPEG")

    result = wrapper.predict(save_path)
    return jsonify({
        "status": "success",
        "predicted_class": result["predicted_class"],
        "display_name": result["display_name"],
        "confidence": result["confidence"],
        "is_low_confidence": result["is_low_confidence"],
        "top_predictions": result["top_predictions"]
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file is too large! Maximum allowed size is 16MB.", "danger")
    return redirect(url_for("index")), 413


@app.errorhandler(404)
def page_not_found(error):
    return render_template("index.html", error_message="The requested page could not be found."), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("index.html", error_message="An internal server error occurred."), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING TRAFFIC SIGNAL RECOGNITION WEB SERVER")
    print("URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
