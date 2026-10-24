"""
Flask Web Application for Skin Disease Image Classification
Project: Skin Disease Image Classification Using Machine Learning
Educational & Research Demonstration Tool (Not a medical diagnostic system)
"""

import os
import json
import uuid
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from database.db import log_prediction, get_history, clear_history

app = Flask(__name__)
app.secret_key = "skin-disease-classification-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "skin_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
PLOTS_FOLDER = os.path.join(STATIC_DIR, "plots")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Global model and metadata variables
model = None
classes = []
idx_to_class = {}
metrics_data = {}


def load_ml_resources():
    """Loads the trained Keras model, class mapping, and evaluation metrics."""
    global model, classes, idx_to_class, metrics_data

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                classes = c_data.get("classes", [])
                idx_to_class = {int(k): v for k, v in c_data.get("idx_to_class", {}).items()}
            print(f"Loaded {len(classes)} classes from {CLASS_NAMES_PATH}")
        except Exception as e:
            print(f"Error reading class names: {e}")

    # Load evaluation metrics
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            print("Loaded evaluation metrics successfully.")
        except Exception as e:
            print(f"Error reading evaluation metrics: {e}")

    # Load Keras CNN model
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow import keras
            model = keras.models.load_model(MODEL_PATH)
            print(f"Trained model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Notice: Model file not found at {MODEL_PATH}. Run 'python train_model.py' to generate it.")


# Initialize resources on startup
load_ml_resources()


def allowed_file(filename):
    """Validates filename extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocesses an input image for model prediction:
    1. Opens with Pillow.
    2. Converts strictly to 3-channel RGB.
    3. Resizes with Bilinear interpolation to target_size (128x128).
    4. Normalizes pixel values from [0, 255] to [0.0, 1.0].
    5. Adds batch dimension (1, 128, 128, 3).
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(resized, dtype=np.float32) / 255.0
        batch_array = np.expand_dims(img_array, axis=0)
        return batch_array, img.size


def get_dataset_stats():
    """Gathers real dataset statistics for presentation."""
    stats = {
        "classes": {},
        "total_images": 0,
        "sample_images": []
    }
    if os.path.exists(DATASET_DIR):
        for c in os.listdir(DATASET_DIR):
            c_path = os.path.join(DATASET_DIR, c)
            if os.path.isdir(c_path) and not c.startswith("."):
                files = [f for f in os.listdir(c_path) if allowed_file(f)]
                stats["classes"][c] = len(files)
                stats["total_images"] += len(files)
                # Pick 1 sample image per class for easy demonstration
                if files:
                    rel_sample = f"dataset/{c}/{files[0]}"
                    stats["sample_images"].append({
                        "class_name": c,
                        "filename": files[0],
                        "rel_path": rel_sample
                    })
    return stats


@app.route("/dataset/<path:filename>")
def serve_dataset_file(filename):
    """Serves benchmark dataset images for web presentation."""
    return send_from_directory(DATASET_DIR, filename)


@app.route("/")
def index():
    """Main application landing page with full workflow sections."""
    # Ensure model is checked if just trained
    global model
    if model is None and os.path.exists(MODEL_PATH):
        load_ml_resources()

    dataset_stats = get_dataset_stats()
    
    # Class descriptions for educational context
    class_descriptions = {
        "Basal_Cell_Carcinoma": "Most common form of non-melanocytic skin cancer, originating in basal cells. Highly treatable when detected early.",
        "Benign_Keratosis": "Non-cancerous skin growth including seborrheic keratoses and solar lentigines. Common in aging skin.",
        "Melanocytic_Nevus": "Benign melanocytic proliferation, commonly known as a mole. Uniform pigmentation and regular borders.",
        "Melanoma": "Malignant skin cancer originating from melanocytes. Demonstrates asymmetry, irregular borders, and color variations."
    }

    recent_history = get_history(limit=10)

    return render_template(
        "index.html",
        model_ready=(model is not None),
        classes=classes,
        class_descriptions=class_descriptions,
        dataset_stats=dataset_stats,
        metrics=metrics_data,
        history=recent_history
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload, preprocessing, model inference, and output formatting.
    Accepts both standard form submissions and AJAX fetch requests.
    """
    global model, classes, idx_to_class
    if model is None:
        if os.path.exists(MODEL_PATH):
            load_ml_resources()
        else:
            msg = "Model file not found. Please run 'python train_model.py' to train and save the model."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("index"))

    # Check if a file was uploaded
    if "image" not in request.files:
        msg = "No file part in the upload request."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        msg = "No file selected for upload."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        msg = "Invalid file type. Please upload a valid JPG, JPEG, or PNG skin photograph."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
        return redirect(url_for("index"))

    try:
        # Save uploaded image with unique filename to prevent collisions
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"upload_{uuid.uuid4().hex[:10]}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        # Preprocess image
        input_tensor, orig_dim = preprocess_image(save_path, target_size=(128, 128))

        # Model Prediction
        preds = model.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = idx_to_class.get(pred_idx, classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}")
        model_score = float(preds[pred_idx]) * 100.0

        # Detailed breakdown for each class
        # Log to SQLite history database
        try:
            log_prediction(unique_name, pred_class, round(model_score, 1))
        except Exception as e:
            print(f"Warning: Failed to log prediction: {e}")

        breakdown = []
        for i, prob in enumerate(preds):
            cls_name = idx_to_class.get(i, classes[i] if i < len(classes) else f"Class_{i}")
            breakdown.append({
                "class_name": cls_name,
                "display_name": cls_name.replace("_", " "),
                "probability": round(float(prob) * 100.0, 2),
                "is_top": (i == pred_idx)
            })
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        # Log to SQLite history database
        try:
            log_prediction(unique_name, pred_class, round(model_score, 1))
        except Exception as e:
            print(f"Warning: Failed to log prediction: {e}")

        result_data = {
            "success": True,
            "filename": unique_name,
            "image_url": url_for("static", filename=f"uploads/{unique_name}"),
            "orig_dimensions": f"{orig_dim[0]} × {orig_dim[1]} pixels",
            "preprocessed_dimensions": "128 × 128 × 3 (RGB, [0, 1])",
            "predicted_class": pred_class,
            "predicted_display": pred_class.replace("_", " "),
            "model_score": f"{model_score:.2f}%",
            "raw_score": round(model_score, 2),
            "breakdown": breakdown,
            "disclaimer": "This prediction is generated by a student educational CNN model trained on a localized subset of the ISIC HAM10000 dermatoscopy research dataset. It is NOT a clinical diagnosis and cannot replace professional dermatological examination."
        }

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify(result_data)

        # Standard form fallback
        dataset_stats = get_dataset_stats()
        return render_template(
            "index.html",
            model_ready=True,
            classes=classes,
            dataset_stats=dataset_stats,
            metrics=metrics_data,
            result=result_data
        )

    except Exception as e:
        msg = f"Error processing image: {str(e)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": msg}), 500
        flash(msg, "error")
        return redirect(url_for("index"))


@app.route("/sample-predict/<path:sample_path>")
def sample_predict(sample_path):
    """Allows one-click demo testing using an authentic image from the dataset folder."""
    global model, classes, idx_to_class
    if model is None:
        if os.path.exists(MODEL_PATH):
            load_ml_resources()
        else:
            return jsonify({"success": False, "error": "Model not loaded"}), 400

    full_path = os.path.join(BASE_DIR, sample_path)
    if not os.path.exists(full_path):
        return jsonify({"success": False, "error": "Sample file not found"}), 404

    try:
        # Copy to uploads for web viewing
        ext = os.path.splitext(full_path)[1].lstrip(".")
        unique_name = f"sample_{uuid.uuid4().hex[:8]}.{ext}"
        dest_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        
        with Image.open(full_path) as img:
            img.save(dest_path)
            orig_dim = img.size

        input_tensor, _ = preprocess_image(dest_path, target_size=(128, 128))
        preds = model.predict(input_tensor)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = idx_to_class.get(pred_idx, classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}")
        model_score = float(preds[pred_idx]) * 100.0

        # Log to SQLite history database
        try:
            log_prediction(unique_name, pred_class, round(model_score, 1))
        except Exception as e:
            print(f"Warning: Failed to log prediction: {e}")

        breakdown = []
        for i, prob in enumerate(preds):
            cls_name = idx_to_class.get(i, classes[i] if i < len(classes) else f"Class_{i}")
            breakdown.append({
                "class_name": cls_name,
                "display_name": cls_name.replace("_", " "),
                "probability": round(float(prob) * 100.0, 2),
                "is_top": (i == pred_idx)
            })
        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "success": True,
            "filename": unique_name,
            "image_url": url_for("static", filename=f"uploads/{unique_name}"),
            "orig_dimensions": f"{orig_dim[0]} × {orig_dim[1]} pixels",
            "preprocessed_dimensions": "128 × 128 × 3 (RGB, [0, 1])",
            "predicted_class": pred_class,
            "predicted_display": pred_class.replace("_", " "),
            "model_score": f"{model_score:.2f}%",
            "raw_score": round(model_score, 2),
            "breakdown": breakdown,
            "disclaimer": "This prediction is generated by a student educational CNN model trained on a localized subset of the ISIC HAM10000 dermatoscopy research dataset. It is NOT a clinical diagnosis and cannot replace professional dermatological examination."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent prediction history as JSON."""
    return jsonify({"success": True, "history": get_history(limit=25)})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears SQLite prediction history."""
    clear_history()
    return jsonify({"success": True})


if __name__ == "__main__":
    print("\nStarting Skin Disease Classification Web Server...")
    print("Open http://127.0.0.1:5000 in your web browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
