import os
import base64
import requests
import csv
import shutil
from datetime import datetime

# Settings
INPUT_FOLDER = "incoming_images"
OUTPUT_FOLDER = "processed"
OUTPUT_CSV = f"batch_predictions_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
API_URL = "http://127.0.0.1:5000/predict"
MODEL_VERSION = "fashion_model_v1"

# Gather PNGs in folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".png")]
results = []

for fname in image_files:
    path = os.path.join(INPUT_FOLDER, fname)
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(API_URL, json={"image": image_data})
        prediction = response.json()

        # Sort top 3 predictions
        top3 = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[:3]

        # Add to results
        results.append([
            datetime.now().isoformat(),
            fname,
            MODEL_VERSION,
            top3[0][0], round(top3[0][1]*100, 2),
            top3[1][0], round(top3[1][1]*100, 2),
            top3[2][0], round(top3[2][1]*100, 2)
        ])

        print(f"{fname}: {top3[0][0]} ({top3[0][1]*100:.2f}%)")

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# Save CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "filename", "model_version",
        "top1_label", "top1_score",
        "top2_label", "top2_score",
        "top3_label", "top3_score"
    ])
    writer.writerows(results)

print(f"\n✅ Batch results saved to {OUTPUT_CSV}")

# Move files to /processed
for fname in image_files:
    src = os.path.join(INPUT_FOLDER, fname)
    dst = os.path.join(OUTPUT_FOLDER, fname)
    shutil.move(src, dst)

print("✅ All images moved to /processed")
