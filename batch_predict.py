"""
Nightly batch job:
  • scans incoming_images/ for *.jpg / *.png
  • sends each picture to /predict
  • writes top-3 results to CSV
  • moves processed images to processed/
"""

import csv
import datetime as dt
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import requests

# ── Config ────────────────────────────────────────────────────────────
API_URL       = "http://localhost:5000/predict"
INCOMING_DIR  = Path("incoming_images")
PROCESSED_DIR = Path("processed")

# ── Logging ───────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
file_base = os.path.splitext(os.path.basename(__file__))[0]          # "batch_predict"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{file_base}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────
def top3(probs: List[float]) -> List[Tuple[int, float]]:
    indexed = list(enumerate(probs))
    return sorted(indexed, key=lambda t: t[1], reverse=True)[:3]

def predict_file(path: Path) -> List[float]:
    with path.open("rb") as f:
        resp = requests.post(API_URL, files={"file": f}, timeout=10)
    resp.raise_for_status()
    return resp.json()["probabilities"]

# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    INCOMING_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    image_files = sorted(INCOMING_DIR.glob("*.[jp][pn]g"))  # .jpg / .png
    if not image_files:
        logger.info("No new images found -- exiting.")
        return

    csv_name = f"batch_predictions_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "rank", "class_id", "probability"])

        for img_path in image_files:
            try:
                probs = predict_file(img_path)
                for rank, (cls, prob) in enumerate(top3(probs), start=1):
                    writer.writerow([img_path.name, rank, cls, f"{prob:.4f}"])

                cls_ids = [cls for cls, _ in top3(probs)]
                logger.info("%s -> %d,%d,%d (top-3)", img_path.name, *cls_ids)

                shutil.move(str(img_path), PROCESSED_DIR / img_path.name)

            except Exception as exc:
                logger.error("Error processing %s: %s", img_path.name, exc)

    logger.info("Batch finished -- results in %s", csv_name)

if __name__ == "__main__":
    main()


#------------------------



#import os
#import base64
#import requests
#import csv
#import shutil
#from datetime import datetime


#import logging, os
#os.makedirs("logs", exist_ok=True)

#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#    handlers=[
#        logging.FileHandler("logs/api.log" if __name__ == "api" else "logs/batch.log"),
#        logging.StreamHandler()
#    ]
#)
#logger = logging.getLogger(__name__)


# Settings
#INPUT_FOLDER = "incoming_images"
#OUTPUT_FOLDER = "processed"
#OUTPUT_CSV = f"batch_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
#API_URL = "http://127.0.0.1:5000/predict"
#MODEL_VERSION = "fashion_model_v1"

# Gather PNGs in folder
#image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".png")]
#results = []

#for fname in image_files:
#    path = os.path.join(INPUT_FOLDER, fname)
#    with open(path, "rb") as f:
#        image_data = base64.b64encode(f.read()).decode("utf-8")

#    try:
#        response = requests.post(API_URL, json={"image": image_data})
#        prediction = response.json()#
#
#        # Sort top 3 predictions
#        top3 = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[:3]

        # Add to results
#        results.append([
#            datetime.now().isoformat(),
#            fname,
#            MODEL_VERSION,
#            top3[0][0], round(top3[0][1]*100, 2),
#            top3[1][0], round(top3[1][1]*100, 2),
#            top3[2][0], round(top3[2][1]*100, 2)
#        ])

#        print(f"{fname}: {top3[0][0]} ({top3[0][1]*100:.2f}%)")

#    except Exception as e:
#        print(f"❌ Error processing {fname}: {e}")

# Save CSV
#with open(OUTPUT_CSV, "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow([
#        "timestamp", "filename", "model_version",
#        "top1_label", "top1_score",
#        "top2_label", "top2_score",
#        "top3_label", "top3_score"
#    ])
#    writer.writerows(results)

#print(f"\n✅ Batch results saved to {OUTPUT_CSV}")

# Move files to /processed
#for fname in image_files:
#    src = os.path.join(INPUT_FOLDER, fname)
#    dst = os.path.join(OUTPUT_FOLDER, fname)
#    shutil.move(src, dst)

#print("✅ All images moved to /processed")
