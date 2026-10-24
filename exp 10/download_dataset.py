"""
Download script for authentic skin disease dermoscopic images from the ISIC Archive (HAM10000 collection).
Dataset: International Skin Imaging Collaboration (ISIC) HAM10000 Dataset.
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Supported Classes:
  1. Melanoma
  2. Melanocytic_Nevus
  3. Basal_Cell_Carcinoma
  4. Benign_Keratosis
"""

import os
import csv
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

CLASSES_CONFIG = {
    "Melanoma": {
        "folder": "Melanoma",
        "query": 'diagnosis_3:"Melanoma, NOS"',
        "count": 100,
    },
    "Melanocytic_Nevus": {
        "folder": "Melanocytic_Nevus",
        "query": 'diagnosis_3:"Nevus"',
        "count": 100,
    },
    "Basal_Cell_Carcinoma": {
        "folder": "Basal_Cell_Carcinoma",
        "query": 'diagnosis_3:"Basal cell carcinoma"',
        "count": 100,
    },
    "Benign_Keratosis": {
        "folder": "Benign_Keratosis",
        "query": 'diagnosis_3:"Pigmented benign keratosis"',
        "count": 100,
    },
}

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "dataset_metadata.csv")


def fetch_image_records(class_name, query, target_count):
    records = []
    limit = min(50, target_count)
    next_url = f"https://api.isic-archive.com/api/v2/images/search/?collections=212&query={urllib.parse.quote(query)}&limit={limit}"
    
    print(f"Fetching metadata for class: {class_name}...")
    while next_url and len(records) < target_count:
        req = urllib.request.Request(next_url, headers={"User-Agent": "SkinDiseaseClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        results = data.get("results", [])
        if not results:
            break
            
        for item in results:
            isic_id = item["isic_id"]
            thumb_url = item.get("files", {}).get("thumbnail_256", {}).get("url")
            full_url = item.get("files", {}).get("full", {}).get("url")
            img_url = thumb_url or full_url
            
            clin = item.get("metadata", {}).get("clinical", {})
            lesion_id = clin.get("lesion_id", f"unknown_{isic_id}")
            sex = clin.get("sex", "unknown")
            age = clin.get("age_approx", "unknown")
            site = clin.get("anatom_site_1", "unknown")
            
            records.append({
                "isic_id": isic_id,
                "class_name": class_name,
                "lesion_id": lesion_id,
                "sex": sex,
                "age_approx": age,
                "anatom_site": site,
                "image_url": img_url,
                "license": item.get("copyright_license", "CC-BY-NC 4.0")
            })
            if len(records) >= target_count:
                break
                
        next_url = data.get("next")
            
    print(f"  Found {len(records)} records for {class_name}")
    return records


def download_single_image(record, target_dir):
    filename = f"{record['isic_id']}.jpg"
    dest_path = os.path.join(target_dir, filename)
    record["local_path"] = os.path.relpath(dest_path, DATASET_DIR)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True, record
        
    try:
        req = urllib.request.Request(record["image_url"], headers={"User-Agent": "SkinDiseaseClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            img_data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(img_data)
        return True, record
    except Exception as e:
        print(f"Error downloading {record['isic_id']}: {e}")
        return False, record


def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    all_records = []
    
    for class_name, cfg in CLASSES_CONFIG.items():
        folder_path = os.path.join(DATASET_DIR, cfg["folder"])
        os.makedirs(folder_path, exist_ok=True)
        records = fetch_image_records(class_name, cfg["query"], cfg["count"])
        
        print(f"Downloading images for {class_name}...")
        successful = 0
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(download_single_image, rec, folder_path) for rec in records]
            for fut in as_completed(futures):
                ok, rec = fut.result()
                if ok:
                    successful += 1
                    all_records.append(rec)
                    
        print(f"  Downloaded {successful}/{len(records)} images for {class_name}.")

    # Save dataset metadata CSV
    if all_records:
        fieldnames = ["isic_id", "class_name", "lesion_id", "sex", "age_approx", "anatom_site", "local_path", "license"]
        with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)
        print(f"\nMetadata saved to {METADATA_PATH}")

    print("\nDataset download complete!")
    print(f"Dataset root: {DATASET_DIR}")
    total = 0
    for class_name, cfg in CLASSES_CONFIG.items():
        fdir = os.path.join(DATASET_DIR, cfg["folder"])
        count = len([f for f in os.listdir(fdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total += count
        print(f" - {class_name}: {count} images in {fdir}")
    print(f"Total dataset images: {total}")


if __name__ == "__main__":
    download_dataset()
