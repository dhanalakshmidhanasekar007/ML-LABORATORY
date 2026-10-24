# Skin Disease Image Classification Using Machine Learning

> **Educational & Academic Mini-Project**  
> *Important Notice:* This application is strictly an educational computer science demonstration and an academic machine learning project. **It is NOT a medical device, nor a clinical diagnostic instrument.** Do not use this tool as a substitute for professional medical advice, diagnosis, or clinical evaluation.

---

## 1. Project Objective

The primary objective of this project is to develop an end-to-end local web application for classifying dermatoscopic skin lesion photographs into standard diagnostic categories using deep Convolutional Neural Networks (CNN) and Transfer Learning.

The system demonstrates the complete machine learning lifecycle:
1. **Problem Definition:** Multiclass image classification of dermatoscopic lesions.
2. **Dataset Collection:** Curating authentic dermatoscopic images from the ISIC Archive HAM10000 collection.
3. **Image Preprocessing:** Standardized RGB enforcement, resizing to 128 × 128, float normalization ([0, 1]), and data-leakage-free group splitting.
4. **Exploratory Data Analysis (EDA):** Class distribution inspection and pipeline step visualization.
5. **Feature Learning:** Deep convolutional feature extraction and transfer learning using MobileNetV2.
6. **Leakage-Free Train-Test Split:** Group-based splitting using lesion identifiers (`lesion_id`) so images from the same lesion/patient never appear across train and test sets.
7. **Model Training:** Training with data augmentation (flips, rotations, zooms) and learning rate scheduling.
8. **Held-Out Test Evaluation:** Real evaluation generating accuracy, precision, recall, F1-scores, and confusion matrices.
9. **Interactive Web Application:** A clean, modern, light-themed Flask interface allowing local image upload and real-time inference.

---

## 2. Dataset Information & Usage Terms

- **Source:** International Skin Imaging Collaboration (ISIC) Archive — HAM10000 ("Human Against Machine with 10,000 training images") Collection.
- **Reference / DOI:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Total Images:** 400 verified dermoscopic images evenly balanced across 4 distinct diagnostic categories.
- **Image Resolution:** Standardized to 128 × 128 pixels (RGB).

### Supported Image Classes

| Class Directory | Diagnostic Name | Clinical Description | Image Count |
|---|---|---|---|
| `dataset/Basal_Cell_Carcinoma/` | Basal Cell Carcinoma (BCC) | Most common non-melanocytic epithelial skin cancer, highly curable when detected early. | 100 |
| `dataset/Benign_Keratosis/` | Benign Keratosis (BKL) | Non-cancerous lesions including seborrheic keratoses, solar lentigines, and lichen-planus like keratoses. | 100 |
| `dataset/Melanocytic_Nevus/` | Melanocytic Nevus (NV) | Common benign melanocytic mole, regular pigment network. | 100 |
| `dataset/Melanoma/` | Melanoma (MEL) | Malignant neoplasm of melanocytes, showing architectural and cellular atypia. | 100 |

---

## 3. Technologies Used

- **Programming Language:** Python 3.10+ (Tested on Python 3.13)
- **Web Framework:** Flask 3.1
- **Deep Learning:** TensorFlow 2.21 / Keras 3.13
- **Data & Numerical Processing:** NumPy, Pandas, Scikit-learn
- **Image Processing:** Pillow (PIL)
- **Data Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3 (Light-themed, responsive design), Vanilla JavaScript (AJAX fetch & drag-and-drop preview)

---

## 4. Project Structure

```text
SkinDiseaseClassification/
├── app.py                      # Flask web application & prediction endpoints
├── train_model.py              # Script to load data, train CNN, evaluate & plot
├── download_dataset.py         # Script to download authentic ISIC HAM10000 images
├── capture_screenshots.py      # Automated Playwright script to capture Figures 1–6
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
├── dataset/                    # Dataset folders organized by class
│   ├── dataset_metadata.csv    # ISIC metadata (lesion_id, age, sex, site)
│   ├── Basal_Cell_Carcinoma/   # 100 authentic dermoscopic images
│   ├── Benign_Keratosis/       # 100 authentic dermoscopic images
│   ├── Melanocytic_Nevus/      # 100 authentic dermoscopic images
│   └── Melanoma/               # 100 authentic dermoscopic images
├── models/
│   ├── skin_disease_model.keras # Trained Keras model
│   ├── class_names.json        # Class-to-index and index-to-class mappings
│   └── evaluation_metrics.json # Stored real evaluation metrics & confusion matrix
├── static/
│   ├── style.css               # Modern, clean light-themed CSS
│   ├── uploads/                # Temporary uploaded test images
│   ├── plots/                  # Generated evaluation plots (Fig 1–4)
│   └── screenshots/            # Browser screenshots (Fig 1–6)
└── templates/
    └── index.html              # Responsive web interface template
```

---

## 5. Installation Instructions

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "c:\Users\Bhuvaneswari E\OneDrive\Desktop\skin disease prediction"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 6. Execution Instructions

The project is structured into modular, independent steps. The application does not download data silently at startup.

### Step 1: Dataset Acquisition (If not already present)
```powershell
python download_dataset.py
```
*Note:* The 400 authentic dermatoscopic images from the ISIC HAM10000 collection are already downloaded and organized into `dataset/` subfolders with metadata in `dataset/dataset_metadata.csv`.

### Step 2: Train the Model & Generate Evaluation Figures
```powershell
python train_model.py --architecture mobilenet
```
To train the custom scratch CNN instead:
```powershell
python train_model.py --architecture cnn
```
This script:
1. Loads the dataset from `dataset/` and reports class distributions.
2. Checks patient/lesion identifiers (`lesion_id`) and applies a `GroupShuffleSplit` (70% train, 15% validation, 15% test) to prevent patient leakage.
3. Standardizes and normalizes images to 128 × 128 tensors.
4. Generates:
   - **Fig. 1:** Dataset Overview (`static/plots/fig1_dataset_overview.png`)
   - **Fig. 2:** Image Preprocessing Pipeline (`static/plots/fig2_preprocessing.png`)
5. Builds and trains the network across epochs with data augmentation.
6. Generates:
   - **Fig. 3:** Accuracy & Loss Curves (`static/plots/fig3_training_curves.png`)
7. Saves the trained model to `models/skin_disease_model.keras` and class mappings to `models/class_names.json`.
8. Evaluates on the held-out test set (60 images) and computes genuine test accuracy, precision, recall, F1, confusion matrix, and per-class summary.
9. Generates:
   - **Fig. 4a:** Confusion Matrix Heatmap (`static/plots/fig4_confusion_matrix.png`)
   - **Fig. 4b:** Per-Class Performance Bar Chart (`static/plots/fig4_class_performance.png`)
   - Saves metrics to `models/evaluation_metrics.json`.

### Step 3: Run the Web Application
```powershell
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

### Step 4 (Optional): Re-capture Figures 1–6
```powershell
python capture_screenshots.py
```
This captures all figures from the running web interface into `static/screenshots/`.

---

## 7. Model Evaluation Summary (Held-Out Test Set)

| Metric | Measured Score | Description |
|---|---|---|
| **Held-Out Test Accuracy** | **51.67%** | Evaluated on 60 strictly held-out test images |
| **Macro Precision** | **54.17%** | Unweighted average precision across all 4 classes |
| **Macro Recall** | **54.72%** | Unweighted average sensitivity across all 4 classes |
| **Macro F1-Score** | **51.83%** | Harmonic mean of precision and recall |
| **Weighted F1-Score** | **49.11%** | F1-score weighted by class support |

### Per-Class Test Performance

| Class Label | Precision | Recall | F1-Score | Support (Images) |
|---|---|---|---|---|
| **Basal Cell Carcinoma** | 88.89% | 61.54% | 72.73% | 13 |
| **Benign Keratosis** | 50.00% | 80.00% | 61.54% | 15 |
| **Melanocytic Nevus** | 44.44% | 61.54% | 51.61% | 13 |
| **Melanoma** | 33.33% | 15.79% | 21.43% | 19 |

---

## 8. Screenshot Guide (Figures 1 to 6)

The project includes all 6 required figures ready for reports and presentations:

| Figure | Description | Location |
|---|---|---|
| **Fig. 1 — Dataset Overview** | Source (ISIC HAM10000), total image count (400), 4 classes, and distribution bar graph | `static/screenshots/fig1_dataset_overview.png` |
| **Fig. 2 — Image Preprocessing** | Original dermoscopic image, resized 128×128 image, and normalized float tensor | `static/screenshots/fig2_preprocessing.png` |
| **Fig. 3 — Model Training** | Training & validation accuracy vs epochs, training & validation loss vs epochs | `static/screenshots/fig3_model_training.png` |
| **Fig. 4 — Model Evaluation** | Held-out test metrics, confusion matrix heatmap, and per-class performance summary | `static/screenshots/fig4_model_evaluation.png` |
| **Fig. 5 — Image Upload** | Web interface image upload zone with real-time skin image preview | `static/screenshots/fig5_image_upload.png` |
| **Fig. 6 — Prediction Result** | Predicted dataset class, model score (confidence %), and full softmax distribution | `static/screenshots/fig6_prediction_result.png` |

---

## 9. Limitations and Ethical Disclaimer

1. **Educational & Academic Scope Only:** This project is created strictly for computer science educational purposes, course evaluations, and machine learning research demonstration.
2. **Not a Diagnostic Tool:** The model score reflects pattern similarity to training images within the 4 supported ISIC categories. It is **not** a confirmed clinical diagnosis and must not be used for patient care or medical self-triage.
3. **Dataset Scope:** The model is trained on dermoscopic photographs and is not designed to reliably evaluate general camera snapshots, low-resolution phone photos, or skin conditions outside the 4 supported classes.
4. **Data Leakage Mitigation:** Splitting was performed using patient lesion identifiers (`lesion_id`) from ISIC metadata to ensure images from the same patient lesion did not appear in both train and test splits.
