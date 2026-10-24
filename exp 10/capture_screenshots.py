"""
Automated Screenshot Capture Script
Captures Figures 1 to 6 from the running Flask application using Playwright & Microsoft Edge.
"""

import os
import time
import threading
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def start_flask():
    from app import app
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


def main():
    print("Starting Flask web server in background thread...")
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Wait for server startup
    import urllib.request
    server_ready = False
    for _ in range(25):
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=2) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            time.sleep(1)
            
    if not server_ready:
        print("Error: Flask server failed to start within timeout.")
        return

    print("Flask server ready! Launching Edge browser via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        print("Navigating to http://127.0.0.1:5000/ ...")
        page.goto("http://127.0.0.1:5000/", wait_until="networkidle")
        time.sleep(1.5)

        # 1. Fig 1 — Dataset Overview
        print("Capturing Fig. 1 — Dataset Overview...")
        dataset_sec = page.locator("#dataset")
        dataset_sec.scroll_into_view_if_needed()
        time.sleep(0.5)
        dataset_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig1_dataset_overview.png"))

        # 2. Fig 2 — Image Preprocessing
        print("Capturing Fig. 2 — Image Preprocessing Pipeline...")
        prep_sec = page.locator("#preprocessing")
        prep_sec.scroll_into_view_if_needed()
        time.sleep(0.5)
        prep_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig2_preprocessing.png"))

        # 3. Fig 3 — Model Training
        print("Capturing Fig. 3 — CNN Model Training Curves...")
        train_sec = page.locator("#training")
        train_sec.scroll_into_view_if_needed()
        time.sleep(0.5)
        train_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig3_model_training.png"))

        # 4. Fig 4 — Model Evaluation
        print("Capturing Fig. 4 — Model Evaluation & Confusion Matrix...")
        eval_sec = page.locator("#evaluation")
        eval_sec.scroll_into_view_if_needed()
        time.sleep(0.5)
        eval_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig4_model_evaluation.png"))

        # 5. Fig 5 — Image Upload Preview
        print("Capturing Fig. 5 — Image Upload & Preview...")
        playground_sec = page.locator("#playground")
        playground_sec.scroll_into_view_if_needed()
        
        # Pick a sample image to upload
        sample_folder = os.path.join(DATASET_DIR, "Melanoma")
        sample_files = [f for f in os.listdir(sample_folder) if f.endswith(".jpg")]
        sample_img_path = os.path.join(sample_folder, sample_files[0])
        
        # Set file input
        file_input = page.locator("#file-input")
        file_input.set_input_files(sample_img_path)
        time.sleep(1)
        playground_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig5_image_upload.png"))

        # 6. Fig 6 — Prediction Result
        print("Capturing Fig. 6 — Prediction Result & Model Score...")
        predict_btn = page.locator("#predict-btn")
        predict_btn.click()
        
        # Wait for result-card to be visible
        page.wait_for_selector("#result-card", state="visible", timeout=15000)
        time.sleep(1.5)
        playground_sec.screenshot(path=os.path.join(SCREENSHOTS_DIR, "fig6_prediction_result.png"))

        browser.close()
        print("\nAll 6 figures captured and saved successfully in static/screenshots/!")


if __name__ == "__main__":
    main()
