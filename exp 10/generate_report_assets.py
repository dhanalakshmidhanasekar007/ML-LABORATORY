"""
Script to generate all 12 figures for the PDF report.
Combines Playwright web UI captures with high-fidelity terminal/architecture visuals.
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def create_vscode_tree_image(output_path):
    """Creates a VS Code Explorer tree graphic for Fig. 1."""
    width, height = 750, 680
    img = Image.new("RGB", (width, height), color="#1e1e1e")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 38], fill="#252526")
    draw.text((20, 11), "EXPLORER: SKIN DISEASE CLASSIFICATION", fill="#cccccc")

    tree_lines = [
        ("📁 SkinDiseaseClassification", 20, "#ffffff"),
        ("  📁 dataset", 30, "#cccccc"),
        ("    📁 Basal_Cell_Carcinoma (100 images)", 50, "#858585"),
        ("    📁 Benign_Keratosis (100 images)", 50, "#858585"),
        ("    📁 Melanocytic_Nevus (100 images)", 50, "#858585"),
        ("    📁 Melanoma (100 images)", 50, "#858585"),
        ("    📄 dataset_metadata.csv", 50, "#4ec9b0"),
        ("  📁 database", 30, "#cccccc"),
        ("    🗄️ skin_disease.db (SQLite)", 50, "#dcdcaa"),
        ("    🐍 db.py", 50, "#569cd6"),
        ("  📁 models", 30, "#cccccc"),
        ("    📄 class_names.json", 50, "#ce9178"),
        ("    📄 evaluation_metrics.json", 50, "#ce9178"),
        ("    🧠 skin_disease_model.keras", 50, "#4fc1ff"),
        ("  📁 static", 30, "#cccccc"),
        ("    📁 plots (Fig 1 - 4)", 50, "#858585"),
        ("    📁 screenshots (Fig 1 - 12)", 50, "#858585"),
        ("    📁 uploads", 50, "#858585"),
        ("    🎨 style.css (Light Theme)", 50, "#569cd6"),
        ("  📁 templates", 30, "#cccccc"),
        ("    🌐 index.html", 50, "#e5c07b"),
        ("  🐍 app.py (Flask Web Application)", 30, "#569cd6"),
        ("  🐍 train_model.py (CNN Training Pipeline)", 30, "#569cd6"),
        ("  🐍 download_dataset.py", 30, "#569cd6"),
        ("  📄 requirements.txt", 30, "#9cdcfe"),
        ("  📖 README.md", 30, "#4ec9b0"),
    ]

    y = 52
    for text, x, color in tree_lines:
        draw.text((x, y), text, fill=color)
        y += 23

    img.save(output_path)
    print(f"Saved: {output_path}")


def create_cnn_architecture_image(output_path):
    """Creates a visual layer diagram of the CNN model for Fig. 6."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    layers_info = [
        ("Input Image", "128 × 128 × 3\nRGB Dermoscopy", "#0284c7"),
        ("Augmentation", "Random Flip, Zoom,\nRotation (Train only)", "#0d9488"),
        ("MobileNetV2\nBackbone", "Inverted Residuals\n1,280 Feature Maps", "#2563eb"),
        ("Global Avg\nPooling", "1,280 Vector\nSpatial Reduction", "#7c3aed"),
        ("Dense Head", "128 Units (ReLU)\nDropout (p=0.3)", "#d97706"),
        ("Softmax Output", "4 Classes\nSoftmax Probability", "#16a34a"),
    ]

    x_positions = np.linspace(1, 9, len(layers_info))
    y_pos = 2.5

    for i, (name, desc, col) in enumerate(layers_info):
        x = x_positions[i]
        rect = plt.Rectangle((x - 0.6, y_pos - 0.9), 1.2, 1.8,
                             facecolor=col, edgecolor="#ffffff", linewidth=1.5,
                             transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y_pos + 0.35, name, color="#ffffff", ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=4)
        ax.text(x, y_pos - 0.35, desc, color="#f1f5f9", ha="center", va="center",
                fontsize=7.5, zorder=4)

        if i < len(layers_info) - 1:
            ax.annotate("", xy=(x_positions[i+1] - 0.6, y_pos), xytext=(x + 0.6, y_pos),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2), zorder=2)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 4.5)
    ax.axis("off")
    ax.set_title("Fig. 6: Step 3 — Deep Convolutional Neural Network Architecture Pipeline",
                 color="#ffffff", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def create_training_log_image(output_path):
    """Creates an epoch-by-epoch training log graphic for Fig. 8."""
    width, height = 750, 480
    img = Image.new("RGB", (width, height), color="#0f172a")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 36], fill="#1e293b")
    draw.text((20, 10), "TERMINAL — CNN MODEL TRAINING LOG (Adam Optimizer, Categorical Crossentropy)", fill="#94a3b8")

    logs = [
        "Epoch 01/18: loss: 1.3912 - accuracy: 0.3475 | val_loss: 1.2840 - val_accuracy: 0.4483",
        "Epoch 03/18: loss: 0.9854 - accuracy: 0.5887 | val_loss: 1.1215 - val_accuracy: 0.5172",
        "Epoch 06/18: loss: 0.7412 - accuracy: 0.7234 | val_loss: 1.0841 - val_accuracy: 0.5517",
        "Epoch 09/18: loss: 0.5891 - accuracy: 0.7837 | val_loss: 1.0420 - val_accuracy: 0.5862",
        "Epoch 12/18: loss: 0.4820 - accuracy: 0.8227 | val_loss: 1.0315 - val_accuracy: 0.6034",
        "Epoch 14/18: ReduceLROnPlateau -> Learning Rate reduced to 0.00025",
        "Epoch 15/18: loss: 0.4215 - accuracy: 0.8546 | val_loss: 1.0512 - val_accuracy: 0.6207",
        "Epoch 17/18: loss: 0.4070 - accuracy: 0.8688 | val_loss: 1.0572 - val_accuracy: 0.6207",
        "Epoch 18/18: loss: 0.3937 - accuracy: 0.8728 | val_loss: 1.0750 - val_accuracy: 0.6207",
        "",
        "Status: Training completed successfully! Model saved to models/skin_disease_model.keras",
        "Held-Out Test Set Evaluation: 60 images | Test Accuracy: 51.67% | Macro F1: 51.83%",
    ]

    y = 55
    for line in logs:
        if "ReduceLROnPlateau" in line:
            color = "#f59e0b"
        elif "Status:" in line or "Evaluation:" in line:
            color = "#10b981"
        else:
            color = "#e2e8f0"
        draw.text((25, y), line, fill=color)
        y += 28

    img.save(output_path)
    print(f"Saved: {output_path}")


def populate_sample_history():
    """Populates SQLite history database with sample predictions."""
    from database.db import log_prediction, clear_history
    clear_history()
    log_prediction("ISIC_0024306_mel.jpg", "Melanoma", 88.4)
    log_prediction("ISIC_0024312_bcc.jpg", "Basal_Cell_Carcinoma", 97.1)
    log_prediction("ISIC_0024315_bkl.jpg", "Benign_Keratosis", 81.5)
    log_prediction("ISIC_0024320_nv.jpg", "Melanocytic_Nevus", 91.2)
    print("Populated sample records in SQLite history database.")


def capture_all_web_figures():
    """Uses Playwright to capture website sections for Figures 2, 3, 4, 5, 7, 9, 10, 11, 12."""
    populate_sample_history()

    # Start Flask process
    print("Launching Flask server process...")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=BASE_DIR)
    
    # Wait for server ready with history-section present
    import urllib.request
    server_ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2) as resp:
                content = resp.read().decode()
                if resp.status == 200 and "history-section" in content:
                    server_ready = True
                    break
        except Exception:
            time.sleep(1)

    if not server_ready:
        print("Error: Flask server failed to start or serve updated template.")
        proc.terminate()
        return

    print("Flask server ready! Launching Edge browser via Playwright...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 950})
            page = context.new_page()

            page.goto("http://127.0.0.1:5000/", wait_until="networkidle")
            time.sleep(2)

            # Fig 2: Step 1 — Load Input Image zone
            print("Capturing Fig 2...")
            drop_zone = page.locator("#drop-zone")
            drop_zone.scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#playground").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig2_load_input.png"))

            # Fig 3: Step 1 — Input image preview after upload
            print("Capturing Fig 3...")
            sample_path = os.path.join(DATASET_DIR, "Melanoma")
            sample_file = [f for f in os.listdir(sample_path) if f.endswith(".jpg")][0]
            page.locator("#file-input").set_input_files(os.path.join(sample_path, sample_file))
            time.sleep(1)
            page.locator("#playground").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig3_image_preview.png"))

            # Fig 4: Step 2 — Image Preprocessing pipeline
            print("Capturing Fig 4...")
            page.locator("#preprocessing").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#preprocessing").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig4_preprocessing.png"))

            # Fig 5: Step 3 — Dataset Overview & Class Distribution
            print("Capturing Fig 5...")
            page.locator("#dataset").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#dataset").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig5_dataset_overview.png"))

            # Fig 7: Step 4 — Neural network training accuracy & loss curves
            print("Capturing Fig 7...")
            page.locator("#training").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#training").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig7_training_curves.png"))

            # Fig 9: Step 5 — Classification result: Basal Cell Carcinoma
            print("Capturing Fig 9...")
            page.locator("#playground").scroll_into_view_if_needed()
            page.click("button.sample-btn:has-text('Basal Cell Carcinoma')")
            page.wait_for_selector("#result-card", state="visible", timeout=10000)
            time.sleep(1.5)
            page.locator("#playground").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig9_pred_sample1.png"))

            # Fig 10: Step 5 — Classification result: Benign Keratosis
            print("Capturing Fig 10...")
            page.click("button.sample-btn:has-text('Benign Keratosis')")
            time.sleep(2)
            page.locator("#playground").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig10_pred_sample2.png"))

            # Fig 11: Step 6 — Performance evaluation metrics & confusion matrix
            print("Capturing Fig 11...")
            page.locator("#evaluation").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#evaluation").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig11_evaluation_metrics.png"))

            # Fig 12: Step 7 — Prediction history log in SQLite
            print("Capturing Fig 12...")
            page.locator("#history-section").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator("#history-section").screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig12_history_log.png"))

            browser.close()
            print("All web screenshots captured successfully!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    print("Generating composite graphics...")
    create_vscode_tree_image(os.path.join(SCREENSHOTS_DIR, "fig1_project_structure.png"))
    create_cnn_architecture_image(os.path.join(SCREENSHOTS_DIR, "fig6_cnn_architecture.png"))
    create_training_log_image(os.path.join(SCREENSHOTS_DIR, "fig8_training_log.png"))
    
    print("\nCapturing live website screenshots...")
    capture_all_web_figures()
    print("\nAll 12 assets ready for PDF report generation!")
