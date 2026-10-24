"""
PDF Report Generator for Skin Disease Image Classification
Generates an exact 11-page academic mini-project report matching the provided sample format.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
PDF_OUTPUT_PATH = os.path.join(BASE_DIR, "Skin_Disease_Image_Classification_Report.pdf")


def build_code_box(code_text):
    """Wraps code text inside a bordered table with light grey background."""
    escaped_lines = []
    for line in code_text.strip().split("\n"):
        line_esc = (line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace(" ", "&nbsp;"))
        escaped_lines.append(line_esc)
        
    formatted_html = "<br/>".join(escaped_lines)
    style = ParagraphStyle(
        "CodeStyle",
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e1e1e")
    )
    p = Paragraph(formatted_html, style)
    t = Table([[p]], colWidths=[504])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def build_report():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica-Oblique",
        fontSize=10.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "BodyJustify",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.2,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=14,
        spaceAfter=2
    )

    caption_style = ParagraphStyle(
        "FigCaption",
        fontName="Times-Italic",
        fontSize=9,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # PAGE 1: Problem Statement, Objectives, Software/Hardware Required
    # =========================================================================
    story.append(Paragraph("Skin Disease Detection from Dermoscopic Images using a Convolutional Neural Network (CNN)", title_style))
    story.append(Paragraph("(Skin Disease Image Classification using Deep Learning)", subtitle_style))

    story.append(Paragraph("Problem Statement:", heading_style))
    story.append(Paragraph(
        "Skin diseases and cutaneous neoplasms represent one of the most common categories of human illness globally, "
        "ranging from benign keratoses and melanocytic nevi to aggressive malignancies such as melanoma and basal cell carcinoma. "
        "Melanoma alone accounts for the majority of skin cancer-related fatalities due to its propensity for early metastasis, yet clinical "
        "differentiation between benign and malignant lesions requires extensive dermatological training and dermoscopic examination. "
        "In many community and rural healthcare environments, access to experienced dermatologists is scarce, resulting in delayed diagnoses "
        "or avoidable invasive biopsies.",
        body_style
    ))
    story.append(Paragraph(
        "There is, therefore, a pressing need for an automated, fast, reproducible, and accessible computer vision system that can analyze "
        "a dermoscopic skin image, preprocess the visual data without patient leakage, extract deep convolutional representations, and classify "
        "it into verified clinical categories with a quantifiable model confidence score. This educational project implements a web-based "
        "Skin Disease Image Classification system that performs automated image preprocessing, deep convolutional neural network feature learning "
        "(via transfer learning with MobileNetV2), held-out test evaluation using real clinical metrics (Accuracy, Precision, Recall, F1-Score, "
        "Confusion Matrix), and persistent SQLite prediction history logging.",
        body_style
    ))

    story.append(Paragraph("Objectives:", heading_style))
    objectives = [
        "1. To collect and preprocess a balanced, labelled benchmark dataset of dermoscopic skin lesion images from the ISIC Archive HAM10000 collection (Basal Cell Carcinoma, Benign Keratosis, Melanocytic Nevus, Melanoma).",
        "2. To standardize, resize (128×128 pixels), and normalize 3-channel RGB image tensors to continuous [0, 1] range.",
        "3. To prevent patient data leakage by partitioning the dataset using lesion/patient identifiers (GroupShuffleSplit).",
        "4. To design, train, and save a Convolutional Neural Network (CNN) / MobileNetV2 transfer learning model with training-time data augmentation.",
        "5. To build a Flask-based backend and modern light-themed web front-end enabling users to upload skin photographs and view predicted classes with softmax confidence scores.",
        "6. To evaluate the model on an unseen held-out test set and report genuine Accuracy, Precision, Recall, F1-score, and Confusion Matrix.",
        "7. To persist every prediction (filename, predicted class, confidence, timestamp) in a local SQLite database for auditing and record-keeping."
    ]
    for obj in objectives:
        story.append(Paragraph(obj, bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Software / Hardware Required:", heading_style))
    reqs_p1 = [
        "• Operating System: Windows / Linux / macOS",
        "• IDE: Visual Studio Code",
        "• Language: Python 3.10+ (Tested on Python 3.13)",
        "• Backend Framework: Flask 3.1 (RESTful API & Static Server)",
    ]
    for r in reqs_p1:
        story.append(Paragraph(r, bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Tech stack continued & Fig 1: Project Directory Structure
    # =========================================================================
    reqs_p2 = [
        "• Deep Learning: TensorFlow 2.21 / Keras 3.13 (MobileNetV2 / CNN)",
        "• Machine Learning & Metrics: scikit-learn (GroupShuffleSplit, classification_report, confusion_matrix)",
        "• Image Processing: Pillow (PIL), NumPy",
        "• Database: SQLite3",
        "• Frontend: HTML5, CSS3 (Light-Themed Modern UI), Vanilla JavaScript (AJAX & File API)",
        "• Dataset: ISIC Archive HAM10000 Scientific Collection (CC BY-NC 4.0 License)",
        "• Version Control: Git"
    ]
    for r in reqs_p2:
        story.append(Paragraph(r, bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Project Directory Structure:", heading_style))
    story.append(Paragraph(
        "The project (\"SkinDiseaseClassification\") is organized as a modular Flask application with dedicated dataset, "
        "trained models, database, and web frontend directories, as shown below.",
        body_style
    ))
    story.append(Spacer(1, 4))

    fig1_path = os.path.join(SCREENSHOTS_DIR, "fig1_project_structure.png")
    if os.path.exists(fig1_path):
        story.append(RLImage(fig1_path, width=440, height=395))
    story.append(Paragraph("Fig 1: Project directory structure of SkinDiseaseClassification in Visual Studio Code", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Procedure & Code Module 1 (Image Preprocessing)
    # =========================================================================
    story.append(Paragraph("Procedure:", heading_style))
    procedures = [
        "1. Collect a verified dermoscopic skin image dataset from the ISIC Archive HAM10000 collection and organize it into class subfolders (dataset/Basal_Cell_Carcinoma, dataset/Benign_Keratosis, dataset/Melanocytic_Nevus, dataset/Melanoma).",
        "2. Parse patient metadata (dataset_metadata.csv) and group images by lesion_id using GroupShuffleSplit (70% train, 15% validation, 15% test) to prevent data leakage.",
        "3. Launch the Flask application (app.py) and open the web front-end in the browser (http://127.0.0.1:5000).",
        "4. Load / upload a skin photograph through the interactive interface (Step 1).",
        "5. Preprocess the image — verify file integrity, convert to 3-channel RGB, resize to 128×128 pixels, and normalize pixel intensities to [0, 1] float tensor (Step 2).",
        "6. Extract high-level visual representations using deep convolutional neural network layers and inverted bottleneck feature maps (Step 3).",
        "7. Train the CNN model with training-time data augmentation (flips, rotations, zooms) using the Adam optimizer with Categorical Crossentropy loss and learning rate plateau scheduling, saving the model to models/skin_disease_model.keras (Step 4).",
        "8. Feed the normalized tensor to the trained model to obtain the predicted dataset class and softmax confidence score (Step 5).",
        "9. Evaluate the model on the held-out test set (60 images) and compute genuine Accuracy, Precision, Recall, F1-score, and Confusion Matrix (Step 6).",
        "10. Log every prediction (filename, predicted class, confidence, timestamp) into the SQLite database (database/skin_disease.db) and display the prediction history (Step 7)."
    ]
    for p in procedures:
        story.append(Paragraph(p, bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Implementation Code:", heading_style))
    story.append(Paragraph(
        "Representative implementation of the key modules used in the project, based on the pipeline shown in the application.",
        body_style
    ))
    story.append(Paragraph("1. Image Preprocessing (app.py & train_model.py)", heading_style))

    code1 = """import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    # 1. Open with Pillow and strictly enforce 3-channel RGB
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        # 2. Resize with Bilinear interpolation to standardized dimensions
        resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        # 3. Normalize pixel intensities from [0, 255] to float [0.0, 1.0]
        img_array = np.array(resized, dtype=np.float32) / 255.0
        # 4. Expand dimensions for single-sample batch inference (1, 128, 128, 3)
        batch_array = np.expand_dims(img_array, axis=0)
        return batch_array, img.size"""
    story.append(build_code_box(code1))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Code Module 2 (Architecture) & Code Module 3 (Model Training)
    # =========================================================================
    story.append(Paragraph("2. Deep Feature Extraction & CNN Architecture (train_model.py)", heading_style))
    code2 = """import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def build_model(arch="mobilenet", num_classes=4):
    inputs = layers.Input(shape=(128, 128, 3))
    # Training-time data augmentation (active only during fit)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.12)(x)
    x = layers.RandomZoom(0.1)(x)

    # Scale [0, 1] to [-1, 1] using native serializable Keras layer
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    
    # Pre-trained MobileNetV2 backbone (transfer learning)
    base_model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze pre-trained weights for stable fine-tuning
    x = base_model(x, training=False)
    
    # Dense classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model"""
    story.append(build_code_box(code2))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Model Training & Group Validation (train_model.py)", heading_style))
    code3 = """from sklearn.model_selection import GroupShuffleSplit
import numpy as np

# Split by lesion_id to guarantee zero patient/lesion leakage
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss_test.split(df, groups=df["lesion_id"].values))
train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Callback to dynamically reduce learning rate on loss plateau
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

model = build_model(arch="mobilenet", num_classes=4)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=18, batch_size=16,
    callbacks=[lr_callback]
)
model.save("models/skin_disease_model.keras")"""
    story.append(build_code_box(code3))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: Code Module 4 (Prediction Route)
    # =========================================================================
    story.append(Paragraph("4. Prediction Route (app.py)", heading_style))
    code4 = """from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np, uuid, os
from database.db import log_prediction

app = Flask(__name__)
model = keras.models.load_model("models/skin_disease_model.keras")
CLASSES = ["Basal_Cell_Carcinoma", "Benign_Keratosis", "Melanocytic_Nevus", "Melanoma"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["image"]
    
    # Save uploaded image with unique collision-free filename
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"upload_{uuid.uuid4().hex[:10]}.{ext}"
    dest_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(dest_path)

    # Preprocess image tensor to (1, 128, 128, 3)
    input_tensor, orig_dim = preprocess_image(dest_path, target_size=(128, 128))

    # Run deep model inference
    preds = model.predict(input_tensor)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASSES[pred_idx]
    confidence = float(preds[pred_idx]) * 100.0

    # Build softmax probability breakdown across all 4 disease classes
    breakdown = [
        {"class_name": c, "probability": round(float(preds[i]) * 100, 2), "is_top": (i == pred_idx)}
        for i, c in enumerate(CLASSES)
    ]
    breakdown.sort(key=lambda x: x["probability"], reverse=True)

    # Persist inference record into local SQLite diagnostic database
    log_prediction(unique_name, pred_class, round(confidence, 1))

    return jsonify({
        "success": True,
        "filename": unique_name,
        "predicted_class": pred_class,
        "predicted_display": pred_class.replace("_", " "),
        "model_score": f"{confidence:.2f}%",
        "breakdown": breakdown,
        "orig_dimensions": f"{orig_dim[0]} × {orig_dim[1]} pixels"
    })"""
    story.append(build_code_box(code4))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: Code Module 5 (Database) & Fig 2 (Step 1 - Load Input Image)
    # =========================================================================
    story.append(Paragraph("5. Saving Prediction History (database/db.py)", heading_style))
    code5 = """import sqlite3, os
from datetime import datetime

DB_PATH = "database/skin_disease.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def log_prediction(filename, prediction, confidence):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''INSERT INTO history (filename, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)''',
        (filename, prediction, confidence, datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p")))
    conn.commit()
    conn.close()"""
    story.append(build_code_box(code5))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Output (Step-by-Step Screenshots):", heading_style))

    fig2_path = os.path.join(SCREENSHOTS_DIR, "fig2_load_input.png")
    if os.path.exists(fig2_path):
        story.append(RLImage(fig2_path, width=480, height=225))
    story.append(Paragraph("Fig 2: Step 1 — Load Input Image: interface to upload a skin image or load a benchmark sample", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: Fig 3 (Preview) & Fig 4 (Preprocessing Pipeline)
    # =========================================================================
    fig3_path = os.path.join(SCREENSHOTS_DIR, "fig3_image_preview.png")
    if os.path.exists(fig3_path):
        story.append(RLImage(fig3_path, width=480, height=290))
    story.append(Paragraph("Fig 3: Step 1 — Input image preview after a dermoscopic skin image is uploaded", caption_style))
    story.append(Spacer(1, 6))

    fig4_path = os.path.join(SCREENSHOTS_DIR, "fig4_preprocessing.png")
    if os.path.exists(fig4_path):
        story.append(RLImage(fig4_path, width=480, height=290))
    story.append(Paragraph("Fig 4: Step 2 — Image preprocessing: original image vs. resized & normalized (128×128) tensor", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: Fig 5 (Dataset Overview) & Fig 6 (CNN Architecture)
    # =========================================================================
    fig5_path = os.path.join(SCREENSHOTS_DIR, "fig5_dataset_overview.png")
    if os.path.exists(fig5_path):
        story.append(RLImage(fig5_path, width=480, height=290))
    story.append(Paragraph("Fig 5: Step 3 — Dataset overview: class distribution and sample counts across the 4 skin disease classes", caption_style))
    story.append(Spacer(1, 6))

    fig6_path = os.path.join(SCREENSHOTS_DIR, "fig6_cnn_architecture.png")
    if os.path.exists(fig6_path):
        story.append(RLImage(fig6_path, width=480, height=270))
    story.append(Paragraph("Fig 6: Step 3 — Deep learning feature extraction: Convolutional Neural Network architecture layers", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: Fig 7 (Training Curves) & Fig 8 (Training Log)
    # =========================================================================
    fig7_path = os.path.join(SCREENSHOTS_DIR, "fig7_training_curves.png")
    if os.path.exists(fig7_path):
        story.append(RLImage(fig7_path, width=480, height=290))
    story.append(Paragraph("Fig 7: Step 4 — CNN training completed: training & validation accuracy and loss curves", caption_style))
    story.append(Spacer(1, 6))

    fig8_path = os.path.join(SCREENSHOTS_DIR, "fig8_training_log.png")
    if os.path.exists(fig8_path):
        story.append(RLImage(fig8_path, width=480, height=280))
    story.append(Paragraph("Fig 8: Step 4 — Epoch-by-epoch training log (Adam optimizer, learning rate schedule, 18 epochs)", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: Fig 9 (Prediction Result 1) & Fig 10 (Prediction Result 2)
    # =========================================================================
    fig9_path = os.path.join(SCREENSHOTS_DIR, "fig9_pred_sample1.png")
    if os.path.exists(fig9_path):
        story.append(RLImage(fig9_path, width=480, height=290))
    story.append(Paragraph("Fig 9: Step 5 — Classification result: Basal Cell Carcinoma, 97.1% confidence and probability breakdown", caption_style))
    story.append(Spacer(1, 6))

    fig10_path = os.path.join(SCREENSHOTS_DIR, "fig10_pred_sample2.png")
    if os.path.exists(fig10_path):
        story.append(RLImage(fig10_path, width=480, height=290))
    story.append(Paragraph("Fig 10: Step 5 — Classification result: Benign Keratosis, confidence score and softmax distribution", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 11: Fig 11 (Metrics), Fig 12 (SQLite History) & Result Paragraph
    # =========================================================================
    fig11_path = os.path.join(SCREENSHOTS_DIR, "fig11_evaluation_metrics.png")
    if os.path.exists(fig11_path):
        story.append(RLImage(fig11_path, width=480, height=210))
    story.append(Paragraph("Fig 11: Step 6 — Performance evaluation metrics (Accuracy 51.67%, Macro Precision 54.17%, Recall 54.72%, F1-Score 51.83%) and confusion matrix", caption_style))
    story.append(Spacer(1, 4))

    fig12_path = os.path.join(SCREENSHOTS_DIR, "fig12_history_log.png")
    if os.path.exists(fig12_path):
        story.append(RLImage(fig12_path, width=480, height=190))
    story.append(Paragraph("Fig 12: Step 7 — Prediction history log persisted in the SQLite diagnostic database", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Result:", heading_style))
    story.append(Paragraph(
        "A web-based Skin Disease Image Classification system was successfully designed and implemented using a Flask backend, "
        "a deep Convolutional Neural Network (CNN / MobileNetV2 transfer learning) with TensorFlow/Keras, and an SQLite database. "
        "Dermoscopic skin images are preprocessed into standardized 128×128 RGB normalized tensors, deep convolutional visual features "
        "are learned without patient data leakage, and the trained model classifies the image into one of 4 diagnostic categories "
        "(Basal Cell Carcinoma, Benign Keratosis, Melanocytic Nevus, Melanoma) with an associated confidence score. "
        "On the strictly held-out test set (60 images), the model achieved a genuine Accuracy of 51.67%, Macro Precision of 54.17%, "
        "Macro Recall of 54.72%, and an F1-Score of 51.83%, and every prediction is logged to a local SQLite diagnostic history database.",
        body_style
    ))

    # Build the document
    doc.build(story)
    print(f"Report successfully generated at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
