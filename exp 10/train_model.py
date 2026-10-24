"""
Model Training & Evaluation Script
Skin Disease Image Classification Using Machine Learning
Dataset: Authentic ISIC Archive HAM10000 Skin Lesion Dataset
Supported Classes:
  1. Basal_Cell_Carcinoma
  2. Benign_Keratosis
  3. Melanocytic_Nevus
  4. Melanoma
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# --- Reproducibility ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "dataset_metadata.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 18


def detect_classes_and_load_data():
    """
    Scans dataset directory, detects classes,
    loads image paths, labels, and patient identifiers.
    """
    print("=" * 60)
    print("STEP 1: Problem Definition & Dataset Inspection")
    print("=" * 60)
    
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
    ])
    
    if not classes:
        raise ValueError(f"No class subfolders found in {DATASET_DIR}")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    
    print(f"Detected {len(classes)} classes:")
    for i, c in enumerate(classes):
        print(f"  [{i}] {c}")
        
    # Load metadata if available to prevent patient data leakage
    meta_df = None
    if os.path.exists(METADATA_PATH):
        try:
            meta_df = pd.read_csv(METADATA_PATH)
            print(f"Loaded metadata from {METADATA_PATH} ({len(meta_df)} records)")
        except Exception as e:
            print(f"Warning: Could not read metadata CSV: {e}")
            
    isic_to_lesion = {}
    if meta_df is not None and "isic_id" in meta_df.columns and "lesion_id" in meta_df.columns:
        for _, row in meta_df.iterrows():
            isic_to_lesion[str(row["isic_id"])] = str(row["lesion_id"])

    # Collect image records
    image_records = []
    class_counts = {}
    
    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[cls_name] = len(files)
        
        for f in files:
            file_path = os.path.join(cls_dir, f)
            base_id = os.path.splitext(f)[0]
            lesion_id = isic_to_lesion.get(base_id, f"lesion_{base_id}")
            image_records.append({
                "path": file_path,
                "filename": f,
                "class_name": cls_name,
                "class_idx": class_to_idx[cls_name],
                "lesion_id": lesion_id
            })

    print("\nDataset Class Distribution:")
    for cls_name, cnt in class_counts.items():
        print(f"  {cls_name:25s}: {cnt:4d} images")
    print(f"Total Dataset Images: {len(image_records)}")
    
    return image_records, classes, class_to_idx, idx_to_class, class_counts


def create_dataset_overview_plot(class_counts, total_images):
    """Generates Figure 1: Dataset Overview & Class Distribution."""
    plt.figure(figsize=(9, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    labels_display = [c.replace("_", " ") for c in classes]
    colors = ["#2b5c8f", "#3a9696", "#d9822b", "#c24747"][:len(classes)]
    
    bars = plt.bar(labels_display, counts, color=colors, width=0.55, edgecolor="#222", linewidth=0.8)
    plt.title("Fig. 1 — Dataset Overview: Class Distribution (ISIC HAM10000 Subset)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Skin Disease Class", fontsize=11, fontweight="semibold")
    plt.ylabel("Number of Dermoscopic Images", fontsize=11, fontweight="semibold")
    plt.ylim(0, max(counts) * 1.25)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{int(h)} images",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    plt.figtext(0.5, -0.05, f"Total Images: {total_images} | Source: ISIC Archive (HAM10000) | Creative Commons CC BY-NC 4.0",
                ha="center", fontsize=9, style="italic", color="#444")
                
    save_path = os.path.join(PLOTS_DIR, "fig1_dataset_overview.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def create_preprocessing_visualization(sample_record):
    """
    Generates Figure 2: Image Preprocessing Pipeline:
    Original Image -> Resized Image (128x128) -> Preprocessed Normalized Tensor
    """
    orig_img = Image.open(sample_record["path"]).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # Resize
    resized_img = orig_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    
    # Preprocess (normalize 0-1)
    norm_arr = np.array(resized_img, dtype=np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=150)
    fig.suptitle(f"Fig. 2 — Image Preprocessing Pipeline (Class: {sample_record['class_name']})",
                 fontsize=13, fontweight="bold", y=1.02)

    # 1. Original
    axes[0].imshow(orig_img)
    axes[0].set_title(f"1. Original Dermoscopic\nDimensions: {orig_w}x{orig_h} (RGB)", fontsize=10, fontweight="semibold")
    axes[0].axis("on")
    axes[0].tick_params(labelsize=8)

    # 2. Resized
    axes[1].imshow(resized_img)
    axes[1].set_title(f"2. Resized Image\nDimensions: {IMG_SIZE[0]}x{IMG_SIZE[1]} (RGB)", fontsize=10, fontweight="semibold")
    axes[1].axis("on")
    axes[1].tick_params(labelsize=8)

    # 3. Normalized
    axes[2].imshow(norm_arr)
    axes[2].set_title(f"3. Normalized Float Tensor\nPixel Range: [{norm_arr.min():.2f}, {norm_arr.max():.2f}]",
                      fontsize=10, fontweight="semibold")
    axes[2].axis("on")
    axes[2].tick_params(labelsize=8)

    plt.figtext(0.5, -0.05, "Pipeline: File Load -> RGB Enforcement -> Bilinear Resize (128x128) -> Scaling [0, 1] Tensor",
                ha="center", fontsize=9, style="italic", color="#444")
                
    save_path = os.path.join(PLOTS_DIR, "fig2_preprocessing.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def load_and_preprocess_image(path):
    """Loads an image, converts to RGB, resizes, and normalizes to [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def prepare_dataset_splits(image_records):
    """
    Splits dataset into Train (70%), Validation (15%), and Test (15%).
    Uses GroupShuffleSplit on lesion_id when available,
    strictly preventing data leakage between train and test/validation sets.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Image Preprocessing & Leakage-Free Train-Test Split")
    print("=" * 60)
    
    df = pd.DataFrame(image_records)
    groups = df["lesion_id"].values
    unique_groups = len(np.unique(groups))
    print(f"Total samples: {len(df)} across {unique_groups} unique lesion/patient groups.")

    if unique_groups >= len(df) * 0.8:
        # Group split to ensure no patient leakage
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
        
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        train_val_groups = train_val_df["lesion_id"].values
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=SEED)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["class_idx"], random_state=SEED)
        train_df, val_df = train_test_split(train_val_df, test_size=0.176, stratify=train_val_df["class_idx"], random_state=SEED)

    print(f"Training Set   : {len(train_df):4d} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation Set : {len(val_df):4d} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test Set       : {len(test_df):4d} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check for group leakage
    train_lesions = set(train_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    leakage = train_lesions.intersection(test_lesions)
    print(f"Lesion overlap between Train and Test sets: {len(leakage)} (Zero indicates no data leakage)")

    # Load image arrays
    print("\nLoading and normalizing image tensors into memory...")
    num_classes = len(df["class_idx"].unique())
    X_train = np.array([load_and_preprocess_image(p) for p in train_df["path"]])
    y_train = keras.utils.to_categorical(train_df["class_idx"].values, num_classes=num_classes)
    
    X_val = np.array([load_and_preprocess_image(p) for p in val_df["path"]])
    y_val = keras.utils.to_categorical(val_df["class_idx"].values, num_classes=num_classes)
    
    X_test = np.array([load_and_preprocess_image(p) for p in test_df["path"]])
    y_test = keras.utils.to_categorical(test_df["class_idx"].values, num_classes=num_classes)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape  : {X_val.shape}, y_val shape  : {y_val.shape}")
    print(f"X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")
    
    return (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df)


def build_model(arch="mobilenet", num_classes=4):
    """
    Builds the model according to the chosen architecture:
    - 'mobilenet': Pre-trained MobileNetV2 backbone (transfer learning) with custom classifier head.
    - 'cnn': Custom Convolutional Neural Network with Conv2D, ReLU, MaxPooling2D, GAP, Dense, Dropout, Softmax.
    """
    print("\n" + "=" * 60)
    print(f"STEP 3: Model Architecture Design (Architecture: {arch.upper()})")
    print("=" * 60)
    
    if arch.lower() == "mobilenet":
        print("Using MobileNetV2 Transfer Learning with pre-trained ImageNet feature weights...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Freeze backbone for fast, reliable fine-tuning
        
        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        # Training-time data augmentation
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.12)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Scale [0, 1] to [-1, 1] for MobileNetV2 using native serializable layer
        x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_MobileNetV2")
        learning_rate = 0.001
        
    else:
        print("Using Custom Convolutional Neural Network (CNN)...")
        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.1),
            
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax")
        ], name="SkinDisease_CustomCNN")
        learning_rate = 0.0008

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


def train_model(model, train_data, val_data):
    """Trains the model and records training history."""
    print("\n" + "=" * 60)
    print("STEP 4: Model Training")
    print("=" * 60)
    
    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_callback],
        verbose=1
    )
    
    return history


def plot_training_curves(history, arch):
    """Generates Figure 3: Training Accuracy & Loss vs Epochs."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", color="#2b5c8f", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", color="#d9822b", label="Validation Accuracy", linewidth=2)
    ax1.set_title(f"{arch.upper()} Accuracy vs Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", color="#c24747", label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", color="#3a9696", label="Validation Loss", linewidth=2)
    ax2.set_title(f"{arch.upper()} Loss vs Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Crossentropy Loss", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Fig. 3 — {arch.upper()} Model Training & Validation Curves", fontsize=13, fontweight="bold", y=1.02)
    save_path = os.path.join(PLOTS_DIR, "fig3_training_curves.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_model_on_test(model, test_data, classes, idx_to_class, arch):
    """
    Evaluates model on held-out test data.
    Computes real metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report.
    Generates Fig. 4: Confusion Matrix and Class Performance Summary.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Model Evaluation on Held-Out Test Set")
    print("=" * 60)
    
    X_test, y_test, test_df = test_data
    
    y_pred_probs = model.predict(X_test)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average="weighted", zero_division=0
    )
    
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, average=None, zero_division=0
    )
    
    print("\n--- Test Set Evaluation Results ---")
    print(f"Model Architecture     : {arch.upper()}")
    print(f"Test Accuracy          : {acc * 100:.2f}%")
    print(f"Macro Precision        : {prec_macro * 100:.2f}%")
    print(f"Macro Recall           : {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score         : {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score      : {f1_weighted * 100:.2f}%")
    
    report_str = classification_report(
        y_true_idx, y_pred_idx, target_names=classes, zero_division=0
    )
    print("\nClassification Report:\n", report_str)

    # Confusion Matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    print("Confusion Matrix:\n", cm)

    # Fig 4a: Confusion Matrix
    display_labels = [c.replace("_", " ") for c in classes]
    plt.figure(figsize=(7.5, 6), dpi=150)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels,
        cbar=True, annot_kws={"size": 11, "fontweight": "bold"}
    )
    plt.title(f"Fig. 4a — Test Confusion Matrix ({arch.upper()} | Accuracy: {acc*100:.1f}%)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class Label", fontsize=11, fontweight="semibold")
    plt.ylabel("Ground Truth Class Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "fig4_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

    # Fig 4b: Per-Class Performance Summary
    plt.figure(figsize=(9, 5), dpi=150)
    x = np.arange(len(classes))
    width = 0.25
    plt.bar(x - width, p_per * 100, width, label="Precision", color="#2b5c8f")
    plt.bar(x, r_per * 100, width, label="Recall", color="#3a9696")
    plt.bar(x + width, f_per * 100, width, label="F1-Score", color="#d9822b")
    plt.xticks(x, display_labels, fontsize=10, fontweight="semibold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="semibold")
    plt.ylim(0, 115)
    plt.title(f"Fig. 4b — Per-Class Test Performance Summary ({arch.upper()})", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    for i in range(len(classes)):
        plt.text(i + width, f_per[i] * 100 + 2, f"{f_per[i]*100:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
                 
    plt.tight_layout()
    perf_path = os.path.join(PLOTS_DIR, "fig4_class_performance.png")
    plt.savefig(perf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {perf_path}")

    # Save metrics JSON for the Flask app
    metrics_data = {
        "dataset_source": "ISIC Archive (HAM10000 Scientific Collection)",
        "license": "Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)",
        "classes": classes,
        "architecture": arch.upper(),
        "test_accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class": {
            cls_name: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1_score": float(f_per[i]),
                "support": int(s_per[i])
            }
            for i, cls_name in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report_str": report_str,
        "test_sample_count": len(y_true_idx),
        "img_size": list(IMG_SIZE)
    }

    metrics_json_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_json_path}")

    return metrics_data


def main():
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model")
    parser.add_argument("--architecture", choices=["mobilenet", "cnn"], default="mobilenet",
                        help="Model architecture: 'mobilenet' (transfer learning) or 'cnn' (scratch CNN)")
    args = parser.parse_args()

    print("=" * 60)
    print(" SKIN DISEASE IMAGE CLASSIFICATION TRAINING WORKFLOW")
    print(f" Architecture: {args.architecture.upper()} | Educational & Research Implementation")
    print("=" * 60)
    
    # 1. Dataset loading & inspection
    image_records, classes, class_to_idx, idx_to_class, class_counts = detect_classes_and_load_data()
    
    # 2. Save class mappings
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class
        }, f, indent=2)
    print(f"\nSaved class mappings to: {class_names_path}")

    # 3. Figures 1 & 2
    create_dataset_overview_plot(class_counts, len(image_records))
    create_preprocessing_visualization(image_records[0])

    # 4. Train / Val / Test Split
    train_data, val_data, test_data = prepare_dataset_splits(image_records)

    # 5. Build Model
    model = build_model(arch=args.architecture, num_classes=len(classes))

    # 6. Train Model
    history = train_model(model, train_data, val_data)

    # 7. Figure 3: Training Curves
    plot_training_curves(history, args.architecture)

    # 8. Save Model
    model_save_path = os.path.join(MODELS_DIR, "skin_disease_model.keras")
    model.save(model_save_path)
    print(f"\nTrained model successfully saved to: {model_save_path}")

    # 9. Evaluate on Held-Out Test Set (Figure 4)
    metrics = evaluate_model_on_test(model, test_data, classes, idx_to_class, args.architecture)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Model Architecture    : {args.architecture.upper()}")
    print(f"Test Accuracy Achieved: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score        : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Model File            : {model_save_path}")
    print(f"Metrics File          : {os.path.join(MODELS_DIR, 'evaluation_metrics.json')}")
    print(f"Plots Directory       : {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
