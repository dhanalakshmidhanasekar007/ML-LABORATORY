# Skin Disease Image Classification

Educational machine learning mini project for classifying dermoscopic skin lesion images with a trained model and a local Flask app. This project is for academic demonstration only; it is not a medical device and must not be used for diagnosis or treatment decisions.

## Repository contents

This upload includes the application and training source, trained model, model labels and evaluation metrics, database helper, HTML template, CSS, and dependency list. Dataset images, uploaded sample images, metadata containing lesion and demographic fields, generated image folders, and the runtime database are not included.

To train the model from scratch, obtain the dataset through the provided acquisition script and follow the dataset provider's terms. Do not commit private or patient-identifying information.

## Setup

Use Python 3.10 or newer. From the cloned `exp 10` directory, install dependencies and start the app:

```powershell
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/ in a browser.

## Train the model

After obtaining and placing the dataset in the format expected by `train_model.py`, run:

```powershell
python download_dataset.py
python train_model.py --architecture mobilenet
```

The training script saves model artifacts and evaluation outputs under `models/` and generated visualizations under `static/`.

## Disclaimer

Predictions are pattern estimates for a limited set of classes. They are not confirmed clinical diagnoses. Do not use this software for patient care, medical self-triage, or as a substitute for a qualified healthcare professional.
